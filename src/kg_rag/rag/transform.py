from typing import Dict
from datetime import datetime


def _clean(v):
    return v if v not in [None, ""] else "N/A"


# =========================================================
# UTIL
# =========================================================

def to_epoch(ts) -> int | None:
    """Convert ISO string or datetime to Unix epoch integer.
    Returns None if ts is None or empty — lets callers decide the fallback.
    """
    if not ts:
        return None
    if isinstance(ts, str):
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return int(dt.timestamp())
    if isinstance(ts, datetime):
        return int(ts.timestamp())
    return int(ts)  # assume already numeric


# =========================================================
# TEXT FOR EMBEDDING
# =========================================================

def build_text(ticket: dict) -> str:
    ticket_id       = ticket.get("ticket_id", "")
    created_at      = ticket.get("created_at", "")      or ""
    product         = ticket.get("product", "")          or ""
    module          = ticket.get("product_module", "")   or ""
    category        = ticket.get("category", "")         or ""
    subcategory     = ticket.get("subcategory", "")      or ""
    region          = ticket.get("region", "")           or ""
    environment     = ticket.get("environment", "")      or ""
    language        = ticket.get("language", "")         or ""
    priority        = ticket.get("priority", "")         or ""
    severity        = ticket.get("severity", "")         or ""
    escalated       = ticket.get("escalated", "")
    escalation_reason = ticket.get("escalation_reason", "") or ""
    customer_tier   = ticket.get("customer_tier", "")    or ""
    agent_id        = ticket.get("agent_id", "")         or ""
    sentiment       = ticket.get("customer_sentiment","") or ""
    business_impact = ticket.get("business_impact", "")  or ""
    resolution_code = ticket.get("resolution_code", "")  or ""
    satisfaction    = ticket.get("satisfaction_score", "")
    affected_users  = ticket.get("affected_users", "")
    subject         = ticket.get("subject", "")          or ""
    description     = ticket.get("description", "")      or ""
    resolution      = ticket.get("resolution", "")       or ""
    error_logs      = ticket.get("error_logs", "")       or ""
    feedback        = ticket.get("feedback_text", "")    or ""

    tags     = ticket.get("tags") or []
    tags_str = ", ".join(str(t) for t in tags if t is not None)

    parts = [
        f"Ticket ID: {ticket_id}",
        f"Created: {created_at}",
        f"Product: {product}  Module: {module}",
        f"Category: {category}  Subcategory: {subcategory}",
        f"Region: {region}  Environment: {environment}  Language: {language}",
        f"Priority: {priority}  Severity: {severity}  Escalated: {escalated}",
        f"Escalation Reason: {escalation_reason}",
        f"Customer Tier: {customer_tier}  Sentiment: {sentiment}  Business Impact: {business_impact}",
        f"Satisfaction Score: {satisfaction}  Affected Users: {affected_users}",
        f"Agent: {agent_id}  Resolution Code: {resolution_code}",
        f"Subject: {subject}",
        f"Description:\n{description}",
    ]

    if resolution:
        parts.append(f"Resolution:\n{resolution}")

    if error_logs:
        parts.append(f"Error Logs:\n{error_logs}")

    if feedback:
        parts.append(f"Customer Feedback: {feedback}")

    if tags_str:
        parts.append(f"Tags: {tags_str}")

    return "\n\n".join(parts)


# =========================================================
# METADATA FOR FILTERING
# =========================================================

def build_metadata(ticket: dict) -> dict:
    meta = {
        "ticket_id":          ticket.get("ticket_id"),
        "created_at":         to_epoch(ticket.get("created_at")),
        "customer_id":        ticket.get("customer_id"),
        "organization_id":    ticket.get("organization_id"),
        "product":            ticket.get("product"),
        "product_version":    ticket.get("product_version"),
        "priority":           ticket.get("priority"),
        "severity":           ticket.get("severity"),
        "category":           ticket.get("category"),
        "region":             ticket.get("region"),
        "environment":        ticket.get("environment"),
        "escalated":          ticket.get("escalated"),
        "resolution_code":    ticket.get("resolution_code"),
        "customer_tier":      ticket.get("customer_tier"),
        "agent_id":           ticket.get("agent_id"),
        "language":           ticket.get("language"),
        "known_issue":        ticket.get("known_issue"),
    }

    related = ticket.get("related_tickets")
    if isinstance(related, list) and len(related) > 0:
        meta["related_tickets"] = related

    return {k: v for k, v in meta.items() if v is not None}


# -------------------------
# 1. ISSUE DOCUMENT
# -------------------------
def ticket_to_issue_document(t: dict) -> str:
    return f"""
Ticket ID: {_clean(t.get('ticket_id'))}
Product: {_clean(t.get('product'))} (v{_clean(t.get('product_version'))})
Module: {_clean(t.get('product_module'))}

Category: {_clean(t.get('category'))}
Priority: {_clean(t.get('priority'))}
Severity: {_clean(t.get('severity'))}

Business Impact: {_clean(t.get('business_impact'))}
Customer Tier: {_clean(t.get('customer_tier'))}
Region: {_clean(t.get('region'))}

Subject:
{_clean(t.get('subject'))}

Description:
{_clean(t.get('description'))}
""".strip()


# -------------------------
# 2. RESOLUTION DOCUMENT
# -------------------------
def ticket_to_resolution_document(t: dict) -> str:
    return f"""
Ticket ID: {_clean(t.get('ticket_id'))}
Product: {_clean(t.get('product'))}
Module: {_clean(t.get('product_module'))}

Resolution:
{_clean(t.get('resolution'))}
""".strip()


# -------------------------
# 3. METADATA DOCUMENT
# -------------------------
def ticket_to_metadata_document(t: dict) -> str:
    return f"""
Ticket ID: {_clean(t.get('ticket_id'))}
Product: {_clean(t.get('product'))}
Module: {_clean(t.get('product_module'))}
Category: {_clean(t.get('category'))}
Priority: {_clean(t.get('priority'))}
Severity: {_clean(t.get('severity'))}
Region: {_clean(t.get('region'))}
""".strip()


# -------------------------
# METADATA (FOR FILTERING ONLY)
# -------------------------
def ticket_to_metadata(t: dict) -> Dict:
    return {
        "ticket_id": t.get("ticket_id"),
        "product": t.get("product"),
        "product_version": t.get("product_version"),
        "product_module": t.get("product_module"),

        "category": t.get("category"),
        "subcategory": t.get("subcategory"),

        "priority": t.get("priority"),
        "severity": t.get("severity"),

        "business_impact": t.get("business_impact"),
        "customer_tier": t.get("customer_tier"),
        "region": t.get("region"),

        "escalated": t.get("escalated"),
        "satisfaction_score": t.get("satisfaction_score"),
        "resolution_code": t.get("resolution_code"),

        "language": t.get("language"),
    }