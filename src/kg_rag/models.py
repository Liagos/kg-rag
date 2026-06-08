"""
Shared data models for Jira ticket ingestion.
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class JiraTicket:
    ticket_id: str
    created_at: str
    updated_at: str
    customer_id: str
    customer_tier: str
    organization_id: str
    product: str
    product_version: str
    product_module: str
    category: str
    subcategory: str
    priority: str
    severity: str
    channel: str
    subject: str
    description: str
    customer_sentiment: str
    previous_tickets: int
    resolution: str
    resolution_code: str
    resolved_at: Optional[str]
    resolution_time_hours: float
    resolution_attempts: int
    agent_id: str
    agent_experience_months: int
    agent_specialization: str
    agent_actions: list[str]
    escalated: bool
    escalation_reason: str
    transferred_count: int
    satisfaction_score: Optional[int]
    feedback_text: str
    resolution_helpful: Optional[bool]
    tags: list[str]
    related_tickets: list[str]
    kb_articles_viewed: list[str]
    kb_articles_helpful: list[str]
    environment: str
    account_age_days: int
    account_monthly_value: float
    similar_issues_last_30_days: int
    product_version_age_days: int
    known_issue: bool
    bug_report_filed: bool
    resolution_template_used: str
    auto_suggested_solutions: list[str]
    auto_suggestion_accepted: bool
    ticket_text_length: int
    response_count: int
    attachments_count: int
    contains_error_code: bool
    contains_stack_trace: bool
    business_impact: str
    affected_users: int
    weekend_ticket: bool
    after_hours: bool
    language: str
    region: str
    error_logs: str = ""
    stack_trace: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "JiraTicket":
        """Create a JiraTicket from a raw dictionary, safely handling missing fields."""
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in fields}
        return cls(**filtered)

    def rag_text(self) -> str:
        """Build the text blob used for embedding in ChromaDB."""
        parts = [
            f"Ticket: {self.ticket_id}",
            f"Subject: {self.subject}",
            f"Description: {self.description}",
            f"Product: {self.product} {self.product_version} ({self.product_module})",
            f"Category: {self.category} / {self.subcategory}",
            f"Priority: {self.priority} | Severity: {self.severity}",
            f"Resolution: {self.resolution}",
            f"Tags: {', '.join(self.tags)}",
            f"Sentiment: {self.customer_sentiment}",
            f"Region: {self.region} | Environment: {self.environment}",
        ]
        return "\n".join(parts)

    def rag_metadata(self) -> dict:
        """Flat metadata dict for ChromaDB (scalar values only)."""
        return {
            "ticket_id": self.ticket_id,
            "created_at": self.created_at,
            "product": self.product,
            "product_version": self.product_version,
            "product_module": self.product_module,
            "category": self.category,
            "priority": self.priority,
            "severity": self.severity,
            "customer_id": self.customer_id,
            "organization_id": self.organization_id,
            "agent_id": self.agent_id,
            "resolution_code": self.resolution_code,
            "escalated": self.escalated,
            "satisfaction_score": self.satisfaction_score or 0,
            "business_impact": self.business_impact,
            "affected_users": self.affected_users,
            "region": self.region,
            "environment": self.environment,
            "language": self.language,
            "resolution_time_hours": self.resolution_time_hours,
            "known_issue": self.known_issue,
        }