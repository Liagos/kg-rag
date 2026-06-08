import re
import json
import anthropic
from openai import OpenAI

from kg_rag.config import settings

openai_client    = OpenAI(api_key=settings.OPEN_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

NEO4J_SCHEMA = """
Nodes:
  - Ticket: ticket_id, subject, priority, severity, created_at, resolved_at,
            resolution_time_hours, escalated, customer_sentiment, business_impact,
            affected_users, region, environment, language, known_issue
  - Customer: customer_id, tier, account_age_days, account_monthly_value
  - Organization: organization_id
  - Product: name, version, module, version_age_days
  - Agent: agent_id, specialization, experience_months
  - Tag: name
  - KBArticle: article_id

Relationships:
  (Customer)-[:SUBMITTED]->(Ticket)
  (Customer)-[:BELONGS_TO]->(Organization)
  (Ticket)-[:ABOUT]->(Product)
  (Agent)-[:RESOLVED]->(Ticket)
  (Ticket)-[:TAGGED_WITH]->(Tag)
  (Ticket)-[:VIEWED_KB]->(KBArticle)
  (Ticket)-[:RELATED_TO]->(Ticket)
"""


# ------------------------------------------------------------------
# Unified LLM call
# ------------------------------------------------------------------

def _llm_call(prompt: str, model: str, system: str | None = None) -> str:
    """Unified LLM call supporting OpenAI and Anthropic models."""
    if model.startswith("claude"):
        messages = [{"role": "user", "content": prompt}]
        kwargs   = dict(
            model=model,
            max_tokens=1000,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        response = anthropic_client.messages.create(**kwargs)
        return response.content[0].text
    else:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = openai_client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
        )
        return response.choices[0].message.content


# ------------------------------------------------------------------
# Filter extraction
# ------------------------------------------------------------------

def extract_neo4j_query_llm(question: str, model: str = settings.llm) -> str:
    """Use the LLM to generate a Cypher query from a natural language question."""
    prompt = f"""
You are a Neo4j expert. Given a natural language question, generate a Cypher query.

SCHEMA:
{NEO4J_SCHEMA}

RULES:
- Always RETURN ticket_id, subject, priority, created_at at minimum
- Always add LIMIT 20 unless the question asks for aggregations
- Use datetime() for date comparisons: created_at > datetime('2024-01-01')
- For text search use toLower() + CONTAINS: toLower(t.subject) CONTAINS 'timeout'
- Never use DETACH DELETE or any write operations
- Return ONLY the Cypher query, no explanation, no markdown

QUESTION: {question}
"""
    return _llm_call(prompt, model=model).strip()


def extract_neo4j_filters_llm(question: str, model: str = settings.llm) -> dict:
    """Extract Neo4j filters from a natural language question."""
    prompt = """
Extract search filters from this question as JSON.

Available filters:
  product_name          (string)
  product_module        (e.g. "encryption_layer" | "backup_service" | "monitoring" | "compression_engine" | "event_handler" | "error_handler")
  resolution_code       (e.g. "PATCH_APPLIED" | "CONFIG_CHANGE" | "DUPLICATE" | "WONT_FIX" | "USER_EDUCATION" | "FEATURE_ADDED")
  text_search           (string) — error codes, keywords to search in logs/description/subject
  tag                   (string)
  category              ("Technical Issue" | "Feature Request" | "Account Management" | "Data Issue" | "Security")
  priority              ("critical" | "high" | "medium" | "low") — single value or list e.g. ["critical", "high"]
  severity              ("P0" | "P1" | "P2" | "P3" | "P4") — single value or list e.g. ["P1", "P2"]
  region                ("APAC" | "EU" | "NA" | "LATAM" | "MEA") — single value or list e.g. ["APAC", "EU"]
  environment           ("production" | "staging" | "development")
  escalated             (true | false)
  date_from             (ISO date string e.g. "2024-11-01") — use for date range start
  date_to               (ISO date string e.g. "2024-11-30") — use for date range end
  customer_tier         ("starter" | "growth" | "premium" | "enterprise")
  customer_id           (e.g. "CUST-02387")
  agent_id              (e.g. "AGENT-027")
  agent_specialization  (e.g. "performance" | "database" | "enterprise" | "general" | "security")
  satisfaction_score_max (integer — use when question mentions low satisfaction, not satisfied, unhappy e.g. 2)

Note: NEVER use created_at directly. Always use date_from and date_to for date ranges.

RESOLUTION CODE RULES — read carefully:
- ONLY extract resolution_code if the question explicitly names one of the exact codes above
- NEVER infer or guess a resolution_code from words like "fixed", "resolved", "patched", "configured"
- "how was it resolved" → NO resolution_code
- "tickets resolved as DUPLICATE" → resolution_code: "DUPLICATE"
- "tickets with CONFIG_CHANGE resolution" → resolution_code: "CONFIG_CHANGE"
- "what configuration change was made" → NO resolution_code
- "tickets resolved by applying a patch" → NO resolution_code (do not guess PATCH_APPLIED)
- "show me PATCH_APPLIED tickets" → resolution_code: "PATCH_APPLIED"

Examples:

"tickets with ERROR_TIMEOUT_429 in logs"
  → {"text_search": "ERROR_TIMEOUT_429"}

"latest medium CloudBackup Enterprise ticket with ERROR_TIMEOUT_429"
  → {"product_name": "CloudBackup Enterprise", "priority": "medium", "text_search": "ERROR_TIMEOUT_429"}

"high or critical CloudBackup Enterprise tickets in encryption_layer"
  → {"product_name": "CloudBackup Enterprise", "priority": ["critical", "high"], "product_module": "encryption_layer"}

"escalated production tickets in APAC from enterprise customers"
  → {"escalated": true, "environment": "production", "region": "APAC", "customer_tier": "enterprise"}

"P1 or P2 tickets in APAC or EU"
  → {"severity": ["P1", "P2"], "region": ["APAC", "EU"]}

"P1 security tickets resolved in EU"
  → {"severity": "P1", "region": "EU"}

"high or critical tickets in encryption_layer"
  → {"priority": ["critical", "high"], "product_module": "encryption_layer"}

"show me all tickets for customer CUST-02387"
  → {"customer_id": "CUST-02387"}

"tickets handled by performance specialists"
  → {"agent_specialization": "performance"}

"security tickets in November 2024"
  → {"category": "Security", "date_from": "2024-11-01", "date_to": "2024-11-30"}

"latest security ticket in November 2024 for CloudBackup Enterprise in APAC"
  → {"product_name": "CloudBackup Enterprise", "category": "Security", "region": "APAC", "date_from": "2024-11-01", "date_to": "2024-11-30"}

"tickets resolved by AGENT-027 where customer was not satisfied"
  → {"agent_id": "AGENT-027", "satisfaction_score_max": 2}

"what configuration change was made to resolve the issue in CloudBackup Enterprise"
  → {"product_name": "CloudBackup Enterprise"}

"show me DUPLICATE tickets for StreamProcessor in EU"
  → {"product_name": "StreamProcessor", "resolution_code": "DUPLICATE", "region": "EU"}

"tickets resolved by applying a patch in APAC"
  → {"region": "APAC"}

Return ONLY valid JSON. Use null for filters not mentioned.
Do NOT wrap in markdown code blocks.

QUESTION: """ + question

    try:
        raw = _llm_call(prompt, model=model)
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        if not raw:
            return {}

        parsed = json.loads(raw)

        if "text_search" in parsed and parsed["text_search"]:
            parsed["text_search"] = parsed["text_search"].lower()

        return {k: v for k, v in parsed.items() if v is not None}

    except json.JSONDecodeError as e:
        print(f"[extract_neo4j_filters_llm] JSON parse error: {e}\nRaw: {raw!r}")
        return {}
    except Exception as e:
        print(f"[extract_neo4j_filters_llm] Unexpected error: {e}")
        return {}


def extract_filters_llm(question: str, model: str = settings.llm) -> dict:
    system_prompt = """
    You extract search filters from Jira ticket questions.

    Return ONLY valid JSON. No markdown, no explanation.

    Supported fields:
    - ticket_id      (e.g. "TK-2024-013193") — extract if a specific ticket ID is mentioned
    - product        (e.g. "CloudBackup Enterprise", "StreamProcessor", "DataSync Pro")
    - priority       ("critical" | "high" | "medium" | "low") — single value or list e.g. ["critical", "high"]
    - severity       ("P0" | "P1" | "P2" | "P3" | "P4") — single value or list e.g. ["P1", "P2"]
    - category       ("Technical Issue" | "Feature Request" | "Account Management" | "Data Issue" | "Security")
    - region         ("EU" | "LATAM" | "APAC" | "NA" | "MEA") — single value or list e.g. ["APAC", "EU"]
    - environment    ("production" | "staging" | "development" | "sandbox")
    - escalated      (true | false)
    - resolution_code ("PATCH_APPLIED" | "CONFIG_CHANGE" | "DUPLICATE" | "WONT_FIX" | "USER_EDUCATION" | "FEATURE_ADDED")
    - customer_tier  ("free" | "starter" | "growth" | "premium" | "enterprise")
    - agent_id       (e.g. "AGENT-027")
    - language       (e.g. "en", "de", "fr")
    - known_issue    (true | false)
    - created_at     (Unix epoch seconds, use {"$gte": ..., "$lt": ...} for ranges)
                     Common conversions:
                     2024-01-01  → 1704067200
                     2024-12-01  → 1733011200
                     2024-12-31  → 1735603200
                     2025-01-01  → 1735689600

                     Full year 2024:  {"$gte": 1704067200, "$lt": 1735689600}
                     Full year 2023:  {"$gte": 1672531200, "$lt": 1704067200}
                     December 2024:   {"$gte": 1733011200, "$lt": 1735689600}
                     Q1 2024:         {"$gte": 1704067200, "$lt": 1711929600}

    RESOLUTION CODE RULES — read carefully:
    - ONLY extract resolution_code if the question explicitly names one of the exact codes above
    - NEVER infer or guess a resolution_code from words like "fixed", "resolved", "patched", "configured"
    - "how was it fixed" → NO resolution_code
    - "tickets resolved as DUPLICATE" → resolution_code: "DUPLICATE"
    - "tickets with CONFIG_CHANGE resolution" → resolution_code: "CONFIG_CHANGE"
    - "what configuration change was made" → NO resolution_code (this is a question about content, not a filter)
    - "tickets resolved by applying a patch" → NO resolution_code (do not guess PATCH_APPLIED)
    - "show me PATCH_APPLIED tickets" → resolution_code: "PATCH_APPLIED"

    Examples:

    Question:
    "what is mentioned in ticket TK-2024-013193"
    Response:
    {"ticket_id": "TK-2024-013193"}

    Question:
    "Show CloudBackup Enterprise tickets from December 2024"
    Response:
    {"product": "CloudBackup Enterprise", "created_at": {"$gte": 1733011200, "$lt": 1735689600}}

    Question:
    "what was the resolution for TK-2024-026126 in CloudBackup Enterprise"
    Response:
    {"ticket_id": "TK-2024-026126", "product": "CloudBackup Enterprise"}

    Question:
    "what configuration change was made to resolve the issue in ticket TK-2024-086613"
    Response:
    {"ticket_id": "TK-2024-086613"}

    Question:
    "critical StreamProcessor incidents"
    Response:
    {"product": "StreamProcessor", "priority": "critical"}

    Question:
    "high or critical CloudBackup Enterprise tickets in APAC or EU"
    Response:
    {"product": "CloudBackup Enterprise", "priority": ["critical", "high"], "region": ["APAC", "EU"]}

    Question:
    "P1 or P2 escalated tickets in APAC"
    Response:
    {"severity": ["P1", "P2"], "escalated": true, "region": "APAC"}

    Question:
    "escalated production tickets in APAC from enterprise customers"
    Response:
    {"escalated": true, "environment": "production", "region": "APAC", "customer_tier": "enterprise"}

    Question:
    "P1 security tickets resolved as duplicate in EU"
    Response:
    {"severity": "P1", "category": "Security", "resolution_code": "DUPLICATE", "region": "EU"}

    Question:
    "show me tickets resolved by applying a patch for StreamProcessor"
    Response:
    {"product": "StreamProcessor"}

    Question:
    "show me PATCH_APPLIED tickets for CloudBackup Enterprise"
    Response:
    {"product": "CloudBackup Enterprise", "resolution_code": "PATCH_APPLIED"}

    If no filters exist, return {}.
    """

    try:
        raw = _llm_call(question, model=model, system=system_prompt)
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        if not raw:
            return {}

        return json.loads(raw)

    except Exception:
        return {}