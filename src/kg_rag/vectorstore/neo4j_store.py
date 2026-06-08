"""
Neo4j ingestion for Jira tickets.

Install:
    pip install neo4j

Usage:
    from neo4j_store import Neo4jTicketStore
    store = Neo4jTicketStore(uri="bolt://localhost:7687", user="neo4j", password="secret")
    store.create_schema()
    store.ingest(tickets)          # list[JiraTicket]
    results = store.query_latest_by_product("CloudBackup Enterprise", tag="timeout")
    store.close()
"""

import sys
import logging
from typing import Optional
from datetime import timedelta
from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging
from neo4j import GraphDatabase
from kg_rag.models import JiraTicket
from kg_rag.config import settings

logger = logging.getLogger(__name__)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

tqdm_logging.set_level(logging.INFO)
tqdm_logging.set_log_rate(timedelta(seconds=5))


class Neo4jTicketStore:
    def __init__(
            self,
            uri: str = settings.NEO4J_URI,
            user: str = settings.NEO4J_USER,
            password: str = settings.NEO4J_PASSWORD,
            database: str = settings.NEO4J_DATABASE,
            batch_size: int = 500,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.batch_size = batch_size
        logger.info("Neo4j connected to %s", uri)

    def close(self):
        self.driver.close()

    # ------------------------------------------------------------------
    # Schema: constraints + indexes
    # ------------------------------------------------------------------

    def create_schema(self):
        """
        Create uniqueness constraints (which auto-create indexes) and
        additional indexes for common query patterns.
        Run once before ingestion.
        """
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Ticket)       REQUIRE t.ticket_id       IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Customer)      REQUIRE c.customer_id     IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Organization)  REQUIRE o.organization_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product)       REQUIRE p.product_key     IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Agent)         REQUIRE a.agent_id        IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (k:KBArticle)     REQUIRE k.article_id      IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (tg:Tag)          REQUIRE tg.name           IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:ResolutionTemplate) REQUIRE r.template_id IS UNIQUE",
        ]
        indexes = [
            # Time-based queries — the key insight: index, not a time-tree node
            "CREATE INDEX ticket_created_at   IF NOT EXISTS FOR (t:Ticket) ON (t.created_at)",
            "CREATE INDEX ticket_resolved_at  IF NOT EXISTS FOR (t:Ticket) ON (t.resolved_at)",
            # Common filter fields
            "CREATE INDEX ticket_priority     IF NOT EXISTS FOR (t:Ticket) ON (t.priority)",
            "CREATE INDEX ticket_severity     IF NOT EXISTS FOR (t:Ticket) ON (t.severity)",
            "CREATE INDEX ticket_region       IF NOT EXISTS FOR (t:Ticket) ON (t.region)",
            "CREATE INDEX ticket_environment  IF NOT EXISTS FOR (t:Ticket) ON (t.environment)",
            "CREATE INDEX ticket_sentiment    IF NOT EXISTS FOR (t:Ticket) ON (t.customer_sentiment)",
            "CREATE INDEX ticket_resolution   IF NOT EXISTS FOR (t:Ticket) ON (t.resolution_code)",
            # Product lookups
            "CREATE INDEX product_name        IF NOT EXISTS FOR (p:Product) ON (p.name)",
        ]
        with self.driver.session(database=self.database) as session:
            for cypher in constraints + indexes:
                session.run(cypher)
        logger.info("Schema (constraints + indexes) created.")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, tickets: list[JiraTicket]) -> None:
        """Ingest all tickets into Neo4j using MERGE for idempotency."""
        total = len(tickets)
        batches = list(range(0, total, self.batch_size))

        logger.info("Ingesting %d tickets into Neo4j (%d batches)...", total, len(batches))

        with tqdm(batches,
                  desc="  Neo4j ingestion",
                  unit="batch",
                  dynamic_ncols=True,
                  leave=True,
                  ) as pbar:
            for start in pbar:
                batch = tickets[start: start + self.batch_size]
                with self.driver.session(database=self.database) as session:
                    session.execute_write(self._ingest_batch, batch)
                pbar.set_postfix({"ingested": min(start + self.batch_size, total)})

        logger.info("✅ Neo4j ingestion complete — %d tickets", total)

    @staticmethod
    def _ingest_batch(tx, batch: list[JiraTicket]):
        for t in batch:
            # --- Core nodes ---
            tx.run(
                """
                MERGE (c:Customer {customer_id: $customer_id})
                  ON CREATE SET c.tier = $customer_tier,
                                c.account_age_days = $account_age_days,
                                c.account_monthly_value = $account_monthly_value

                MERGE (o:Organization {organization_id: $organization_id})

                MERGE (c)-[:BELONGS_TO]->(o)
                """,
                customer_id=t.customer_id,
                customer_tier=t.customer_tier,
                account_age_days=t.account_age_days,
                account_monthly_value=t.account_monthly_value,
                organization_id=t.organization_id,
            )

            tx.run(
                """
                MERGE (p:Product {product_key: $product_key})
                  ON CREATE SET p.name    = $product,
                                p.version = $version,
                                p.module  = $module,
                                p.version_age_days = $version_age_days
                """,
                product_key=f"{t.product}::{t.product_version}::{t.product_module}",
                product=t.product,
                version=t.product_version,
                module=t.product_module,
                version_age_days=t.product_version_age_days,
            )

            tx.run(
                """
                MERGE (a:Agent {agent_id: $agent_id})
                  ON CREATE SET a.experience_months = $experience_months,
                                a.specialization    = $specialization
                """,
                agent_id=t.agent_id,
                experience_months=t.agent_experience_months,
                specialization=t.agent_specialization,
            )

            # --- Ticket node ---
            tx.run(
                """
                MERGE (tk:Ticket {ticket_id: $ticket_id})
                SET tk.created_at               = datetime($created_at),
                    tk.updated_at               = datetime($updated_at),
                    tk.resolved_at              = CASE WHEN $resolved_at IS NOT NULL
                                                      THEN datetime($resolved_at) ELSE null END,
                    tk.subject                  = $subject,
                    tk.description              = $description,
                    tk.error_logs               = $error_logs,
                    tk.category                 = $category,
                    tk.subcategory              = $subcategory,
                    tk.priority                 = $priority,
                    tk.severity                 = $severity,
                    tk.channel                  = $channel,
                    tk.customer_sentiment       = $customer_sentiment,
                    tk.resolution_code          = $resolution_code,
                    tk.resolution               = $resolution,
                    tk.resolution_time_hours    = $resolution_time_hours,
                    tk.resolution_attempts      = $resolution_attempts,
                    tk.escalated                = $escalated,
                    tk.escalation_reason        = $escalation_reason,
                    tk.transferred_count        = $transferred_count,
                    tk.satisfaction_score       = $satisfaction_score,
                    tk.business_impact          = $business_impact,
                    tk.affected_users           = $affected_users,
                    tk.environment              = $environment,
                    tk.region                   = $region,
                    tk.language                 = $language,
                    tk.known_issue              = $known_issue,
                    tk.bug_report_filed         = $bug_report_filed,
                    tk.previous_tickets         = $previous_tickets,
                    tk.similar_issues_30d       = $similar_issues_last_30_days,
                    tk.response_count           = $response_count,
                    tk.attachments_count        = $attachments_count,
                    tk.weekend_ticket           = $weekend_ticket,
                    tk.after_hours              = $after_hours,
                    tk.auto_suggestion_accepted = $auto_suggestion_accepted
                """,
                ticket_id=t.ticket_id,
                description=t.description,
                error_logs=t.error_logs,
                created_at=t.created_at,
                updated_at=t.updated_at,
                resolved_at=t.resolved_at,
                subject=t.subject,
                category=t.category,
                subcategory=t.subcategory,
                priority=t.priority,
                severity=t.severity,
                channel=t.channel,
                customer_sentiment=t.customer_sentiment,
                resolution_code=t.resolution_code,
                resolution=t.resolution,
                resolution_time_hours=t.resolution_time_hours,
                resolution_attempts=t.resolution_attempts,
                escalated=t.escalated,
                escalation_reason=t.escalation_reason,
                transferred_count=t.transferred_count,
                satisfaction_score=t.satisfaction_score,
                business_impact=t.business_impact,
                affected_users=t.affected_users,
                environment=t.environment,
                region=t.region,
                language=t.language,
                known_issue=t.known_issue,
                bug_report_filed=t.bug_report_filed,
                previous_tickets=t.previous_tickets,
                similar_issues_last_30_days=t.similar_issues_last_30_days,
                response_count=t.response_count,
                attachments_count=t.attachments_count,
                weekend_ticket=t.weekend_ticket,
                after_hours=t.after_hours,
                auto_suggestion_accepted=t.auto_suggestion_accepted,
            )

            # --- Ticket relationships ---
            tx.run(
                """
                MATCH (c:Customer {customer_id: $customer_id})
                MATCH (tk:Ticket  {ticket_id:   $ticket_id})
                MERGE (c)-[:SUBMITTED]->(tk)
                """,
                customer_id=t.customer_id, ticket_id=t.ticket_id,
            )
            tx.run(
                """
                MATCH (tk:Ticket  {ticket_id:  $ticket_id})
                MATCH (p:Product  {product_key: $product_key})
                MERGE (tk)-[:ABOUT]->(p)
                """,
                ticket_id=t.ticket_id,
                product_key=f"{t.product}::{t.product_version}::{t.product_module}",
            )
            tx.run(
                """
                MATCH (a:Agent  {agent_id:  $agent_id})
                MATCH (tk:Ticket {ticket_id: $ticket_id})
                MERGE (a)-[:RESOLVED]->(tk)
                """,
                agent_id=t.agent_id, ticket_id=t.ticket_id,
            )

            # --- Tags ---
            for tag in t.tags:
                tx.run(
                    """
                    MERGE (tg:Tag {name: $tag})
                    WITH tg
                    MATCH (tk:Ticket {ticket_id: $ticket_id})
                    MERGE (tk)-[:TAGGED_WITH]->(tg)
                    """,
                    tag=tag, ticket_id=t.ticket_id,
                )

            # --- KB articles ---
            for article_id in t.kb_articles_viewed:
                helpful = article_id in t.kb_articles_helpful
                tx.run(
                    """
                    MERGE (kb:KBArticle {article_id: $article_id})
                    WITH kb
                    MATCH (tk:Ticket {ticket_id: $ticket_id})
                    MERGE (tk)-[r:VIEWED_KB]->(kb)
                    SET r.helpful = $helpful
                    """,
                    article_id=article_id,
                    ticket_id=t.ticket_id,
                    helpful=helpful,
                )

            # --- Related tickets ---
            for related_id in t.related_tickets:
                tx.run(
                    """
                    MERGE (r:Ticket {ticket_id: $related_id})
                    WITH r
                    MATCH (tk:Ticket {ticket_id: $ticket_id})
                    MERGE (tk)-[:RELATED_TO]->(r)
                    """,
                    related_id=related_id, ticket_id=t.ticket_id,
                )

            # --- Resolution template ---
            if t.resolution_template_used:
                tx.run(
                    """
                    MERGE (rt:ResolutionTemplate {template_id: $template_id})
                    WITH rt
                    MATCH (tk:Ticket {ticket_id: $ticket_id})
                    MERGE (tk)-[:USED_TEMPLATE]->(rt)
                    """,
                    template_id=t.resolution_template_used,
                    ticket_id=t.ticket_id,
                )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def query_customer_history(self, customer_id: str) -> list[dict]:
        """Full ticket history for a customer, newest first."""
        cypher = """
        MATCH (c:Customer {customer_id: $customer_id})-[:SUBMITTED]->(t:Ticket)-[:ABOUT]->(p:Product)
        RETURN t.ticket_id          AS ticket_id,
               t.subject            AS subject,
               t.created_at         AS created_at,
               t.priority           AS priority,
               t.satisfaction_score AS satisfaction_score,
               t.resolution_code    AS resolution_code,
               p.name               AS product
        ORDER BY t.created_at DESC
        """
        with self.driver.session(database=self.database) as session:
            results = []
            for r in session.run(cypher, customer_id=customer_id):
                row = dict(r)
                if row.get("created_at"):
                    row["created_at"] = row["created_at"].isoformat()
                results.append(row)
            return results

    def query_ticket_detail(self, ticket_id: str) -> dict:
        """Fetch full detail for a single ticket by ID."""
        cypher = """
        MATCH (t:Ticket {ticket_id: $ticket_id})
        OPTIONAL MATCH (c:Customer)-[:SUBMITTED]->(t)
        OPTIONAL MATCH (a:Agent)-[:RESOLVED]->(t)
        OPTIONAL MATCH (t)-[:ABOUT]->(p:Product)
        RETURN t.ticket_id          AS ticket_id,
               t.subject            AS subject,
               t.description        AS description,
               t.resolution         AS resolution,
               t.resolution_code    AS resolution_code,
               t.error_logs         AS error_logs,
               t.priority           AS priority,
               t.severity           AS severity,
               t.category           AS category,
               t.escalated          AS escalated,
               t.region             AS region,
               t.environment        AS environment,
               t.satisfaction_score AS satisfaction_score,
               t.created_at         AS created_at,
               p.name               AS product,
               p.module             AS product_module,
               c.customer_id        AS customer_id,
               c.tier               AS customer_tier,
               a.agent_id           AS agent_id,
               a.specialization     AS agent_specialization
        """
        with self.driver.session(database=self.database) as session:
            record = session.run(cypher, ticket_id=ticket_id).single()
            if not record:
                return {}
            row = dict(record)
            if row.get("created_at"):
                row["created_at"] = row["created_at"].isoformat()
            return row

    def query_agent_workload(self, top_n: int = 10) -> list[dict]:
        """Agents ranked by number of resolved tickets."""
        cypher = """
        MATCH (a:Agent)-[:RESOLVED]->(tk:Ticket)
        RETURN a.agent_id          AS agent_id,
               a.specialization    AS specialization,
               COUNT(tk)           AS tickets_resolved,
               AVG(tk.resolution_time_hours) AS avg_resolution_hours,
               AVG(tk.satisfaction_score)    AS avg_satisfaction
        ORDER BY tickets_resolved DESC
        LIMIT $top_n
        """
        with self.driver.session(database=self.database) as session:
            return [dict(r) for r in session.run(cypher, top_n=top_n)]

    def query_recurring_issues(self, min_count: int = 5) -> list[dict]:
        """Tags that frequently co-occur with escalated or critical tickets."""
        cypher = """
        MATCH (tk:Ticket)-[:TAGGED_WITH]->(tg:Tag)
        WHERE tk.escalated = true OR tk.priority = 'critical'
        RETURN tg.name          AS tag,
               COUNT(tk)        AS ticket_count,
               AVG(tk.affected_users) AS avg_affected_users,
               COLLECT(DISTINCT tk.product)[0..5] AS sample_products
        ORDER BY ticket_count DESC
        LIMIT $min_count
        """
        with self.driver.session(database=self.database) as session:
            return [dict(r) for r in session.run(cypher, min_count=min_count)]

    def query_kb_effectiveness(self) -> list[dict]:
        """Which KB articles are most viewed vs actually helpful."""
        cypher = """
        MATCH (tk:Ticket)-[r:VIEWED_KB]->(kb:KBArticle)
        RETURN kb.article_id                      AS article_id,
               COUNT(r)                           AS times_viewed,
               SUM(CASE WHEN r.helpful THEN 1 ELSE 0 END) AS times_helpful,
               ROUND(100.0 * SUM(CASE WHEN r.helpful THEN 1 ELSE 0 END) / COUNT(r), 1)
                                                  AS helpfulness_pct
        ORDER BY times_viewed DESC
        LIMIT 20
        """
        with self.driver.session(database=self.database) as session:
            return [dict(r) for r in session.run(cypher)]

    def query_from_cypher(self, cypher: str) -> list[dict]:
        """Execute an LLM-generated Cypher query with a read-only transaction."""
        with self.driver.session(database=self.database) as session:
            # Read-only transaction prevents any accidental writes
            result = session.execute_read(lambda tx: list(tx.run(cypher)))
            return [dict(r) for r in result]

    def query_related_tickets(self, ticket_id: str, limit: int = 10) -> list[dict]:
        cypher = """
        MATCH (t:Ticket {ticket_id: $ticket_id})-[:RELATED_TO]->(r:Ticket)
        OPTIONAL MATCH (r)-[:ABOUT]->(p:Product)
        RETURN r.ticket_id       AS ticket_id,
               r.subject         AS subject,
               r.priority        AS priority,
               r.resolution      AS resolution,
               r.resolution_code AS resolution_code,
               r.created_at      AS created_at,
               p.name            AS product
        ORDER BY r.created_at DESC
        LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            results = []
            for r in session.run(cypher, ticket_id=ticket_id, limit=limit):
                row = dict(r)
                if row.get("created_at"):
                    row["created_at"] = row["created_at"].isoformat()
                results.append(row)
            return results

    def query_from_filters(self, filters: dict, limit: int = 20) -> list[dict]:
        """Build and run a Cypher query from LLM-extracted filter dict."""
        conditions = []
        params: dict = {"limit": limit}

        if filters.get("product_name"):
            conditions.append("p.name = $product_name")
            params["product_name"] = filters["product_name"]

        if filters.get("priority"):
            priority = filters["priority"]
            if isinstance(priority, list):
                conditions.append("t.priority IN $priority")
                params["priority"] = priority
            else:
                conditions.append("t.priority = $priority")
                params["priority"] = priority

        if filters.get("severity"):
            severity = filters["severity"]
            if isinstance(severity, list):
                conditions.append("t.severity IN $severity")
                params["severity"] = severity
            else:
                conditions.append("t.severity = $severity")
                params["severity"] = severity

        if filters.get("region"):
            region = filters["region"]
            if isinstance(region, list):
                conditions.append("t.region IN $region")
                params["region"] = region
            else:
                conditions.append("t.region = $region")
                params["region"] = region

        if filters.get("environment"):
            conditions.append("t.environment = $environment")
            params["environment"] = filters["environment"]

        if filters.get("escalated") is not None:
            conditions.append("t.escalated = $escalated")
            params["escalated"] = filters["escalated"]

        if filters.get("date_from"):
            conditions.append("t.created_at >= datetime($date_from)")
            params["date_from"] = filters["date_from"]

        if filters.get("date_to"):
            conditions.append("t.created_at <= datetime($date_to)")
            params["date_to"] = filters["date_to"]

        if filters.get("customer_tier"):
            conditions.append("c.tier = $customer_tier")
            params["customer_tier"] = filters["customer_tier"]

        if filters.get("customer_id"):
            conditions.append("c.customer_id = $customer_id")
            params["customer_id"] = filters["customer_id"]

        if filters.get("category"):
            conditions.append("t.category = $category")
            params["category"] = filters["category"]

        if filters.get("resolution_code"):
            conditions.append("t.resolution_code = $resolution_code")
            params["resolution_code"] = filters["resolution_code"]

        if filters.get("tag"):
            conditions.append("(t)-[:TAGGED_WITH]->(:Tag {name: $tag})")
            params["tag"] = filters["tag"]

        if filters.get("satisfaction_score_max") is not None:
            conditions.append("t.satisfaction_score <= $satisfaction_score_max")
            params["satisfaction_score_max"] = filters["satisfaction_score_max"]

        if filters.get("text_search"):
            conditions.append(
                "("
                "toLower(t.subject)     CONTAINS $text_search OR "
                "toLower(t.description) CONTAINS $text_search OR "
                "toLower(t.error_logs)  CONTAINS $text_search"
                ")"
            )
            params["text_search"] = filters["text_search"].lower()

        if filters.get("product_module"):
            module = filters["product_module"]
            if isinstance(module, list):
                conditions.append("p.module IN $product_module")
                params["product_module"] = module
            else:
                conditions.append("p.module = $product_module")
                params["product_module"] = module

        # build main WHERE clause
        where = ("WHERE " + "\n  AND ".join(conditions)) if conditions else ""

        # agent conditions — use MATCH when filtering by agent, OPTIONAL MATCH otherwise
        agent_conditions = []
        if filters.get("agent_specialization"):
            agent_conditions.append("a.specialization = $agent_specialization")
            params["agent_specialization"] = filters["agent_specialization"]

        if filters.get("agent_id"):
            agent_conditions.append("a.agent_id = $agent_id")
            params["agent_id"] = filters["agent_id"]

        # hard MATCH when agent filter present — excludes tickets with no agent
        # OPTIONAL MATCH otherwise — keeps tickets even without agent
        if agent_conditions:
            agent_match = "MATCH (a:Agent)-[:RESOLVED]->(t)"
            agent_where = "WHERE " + "\n  AND ".join(agent_conditions)
        else:
            agent_match = "OPTIONAL MATCH (a:Agent)-[:RESOLVED]->(t)"
            agent_where = ""

        cypher = f"""
        MATCH (c:Customer)-[:SUBMITTED]->(t:Ticket)-[:ABOUT]->(p:Product)
        {where}
        {agent_match}
        {agent_where}
        RETURN t.ticket_id          AS ticket_id,
               t.subject            AS subject,
               t.priority           AS priority,
               t.severity           AS severity,
               t.category           AS category,
               t.created_at         AS created_at,
               t.escalated          AS escalated,
               t.region             AS region,
               t.environment        AS environment,
               t.description        AS description,
               t.error_logs         AS error_logs,
               t.resolution         AS resolution,
               t.resolution_code    AS resolution_code,
               t.satisfaction_score AS satisfaction_score,
               p.name               AS product,
               p.module             AS product_module,
               c.customer_id        AS customer_id,
               c.tier               AS customer_tier,
               a.agent_id           AS agent_id,
               a.specialization     AS agent_specialization
        ORDER BY t.created_at DESC
        LIMIT $limit
        """

        with self.driver.session(database=self.database) as session:
            return [dict(r) for r in session.run(cypher, params)]
