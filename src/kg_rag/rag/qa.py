import re
import anthropic
from typing import Literal
from openai import OpenAI

from kg_rag.config import settings
from kg_rag.retrievers.hybrid import HybridRetriever
from kg_rag.retrievers.reranker import Reranker
from kg_rag.query_understanding.filters import extract_filters_llm, extract_neo4j_filters_llm
from kg_rag.vectorstore.neo4j_store import Neo4jTicketStore

openai_client    = OpenAI(api_key=settings.OPEN_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)  # ← fixed
retriever        = HybridRetriever()
reranker         = Reranker()
neo4j            = Neo4jTicketStore()


# ------------------------------------------------------------------
# Unified LLM call
# ------------------------------------------------------------------

def _llm_call(prompt: str, model: str) -> str:
    """Unified LLM call supporting OpenAI and Anthropic models."""
    if model.startswith("claude"):
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    else:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


# ------------------------------------------------------------------
# Graph enrichment
# ------------------------------------------------------------------

def _enrich_from_docs(top_docs: list[dict]) -> str:
    """Hybrid mode: enrich ChromaDB results with graph context."""
    lines = []
    for item in top_docs:
        ticket_id   = item["doc"].metadata.get("ticket_id")
        customer_id = item["doc"].metadata.get("customer_id")

        if ticket_id:
            related = neo4j.query_related_tickets(ticket_id, limit=3)
            if related:
                ids = [r["ticket_id"] for r in related]
                lines.append(f"[{ticket_id}] related tickets: {ids}")

        if customer_id:
            history = neo4j.query_customer_history(customer_id)
            if history:
                scores = [h["satisfaction_score"] for h in history if h.get("satisfaction_score")]
                avg    = round(sum(scores) / len(scores), 1) if scores else "n/a"
                lines.append(
                    f"[{ticket_id}] customer {customer_id}: "
                    f"{len(history)} past tickets, avg satisfaction {avg}"
                )

    return "\n".join(lines) if lines else ""


def highlight_match(text: str, search_term: str, context_chars: int = 100) -> str:
    if not text or not search_term:
        return ""
    match = re.search(re.escape(search_term), text, re.IGNORECASE)
    if not match:
        return ""
    start   = max(0, match.start() - context_chars)
    end     = min(len(text), match.end() + context_chars)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def _enrich_from_filters(neo4j_filters: dict) -> str:
    """Neo4j-only mode: query graph directly from LLM-extracted filters."""
    results = neo4j.query_from_filters(neo4j_filters, limit=10)
    if not results:
        return ""

    text_search = neo4j_filters.get("text_search", "")

    lines = []
    for r in results:
        line = (
            f"[{r['ticket_id']}] {r['subject']} | "
            f"priority={r['priority']} | "
            f"severity={r.get('severity', 'unknown')} | "
            f"category={r.get('category', 'unknown')} | "
            f"escalated={r.get('escalated', 'unknown')} | "
            f"region={r.get('region', 'unknown')} | "
            f"environment={r.get('environment', 'unknown')} | "
            f"product={r['product']} | "
            f"module={r.get('product_module', 'unknown')} | "
            f"customer_id={r.get('customer_id', 'unknown')} | "
            f"customer_tier={r.get('customer_tier', 'unknown')} | "
            f"agent_id={r.get('agent_id', 'unknown')} | "
            f"agent_specialization={r.get('agent_specialization', 'unknown')} | "
            f"resolution_code={r.get('resolution_code', 'unknown')} | "
            f"satisfaction_score={r.get('satisfaction_score', 'unknown')} | "
            f"created={r['created_at']} | "
            f"error_logs={r.get('error_logs', '')}"
        )

        if text_search:
            snippet = highlight_match(r.get("description", ""), text_search)
            if snippet:
                line += f" | description_match={snippet}"

        lines.append(line)

    return "\n".join(lines)


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------

def _build_prompt(
    question: str,
    history_text: str,
    context: str,
    graph_context: str,
) -> str:                          # ← returns str not tuple

    if context and graph_context:
        tickets_section = f"## TICKETS\n{context}\n\n## GRAPH CONTEXT\n{graph_context}"
    elif context:
        tickets_section = f"## TICKETS\n{context}"
    elif graph_context:
        tickets_section = f"## TICKETS\n{graph_context}"
    else:
        tickets_section = ""

    return f"""
You are a Jira support assistant helping support engineers and managers analyse ticket data.

Answer ONLY using the provided tickets.
If the answer is not explicitly supported by the tickets, say you cannot find it.

---

## CONVERSATION HISTORY
{history_text}

---

## CURRENT QUESTION
{question}

---

{tickets_section}

---

## TICKET FIELD REFERENCE
Each ticket may include the following fields:
- ticket_id            : unique identifier e.g. TK-2024-019045
- subject              : short title of the issue
- priority             : critical | high | medium | low
- severity             : P0 | P1 | P2 | P3 | P4
- escalated            : whether the ticket was escalated (True/False)
- region               : APAC | EU | NA | LATAM | MEA
- environment          : production | staging | development
- product              : product name e.g. CloudBackup Enterprise
- module               : product module e.g. encryption_layer
- category             : Technical Issue | Feature Request | Account Management | Data Issue | Security
- customer_id          : customer ID e.g. CUST-02387
- customer_tier        : starter | growth | premium | enterprise
- agent_id             : agent ID who resolved the ticket e.g. AGENT-044
- agent_specialization : agent specialization e.g. performance | database | enterprise | general
- created              : ticket creation timestamp — use this to determine recency
- resolution_code      : how the ticket was resolved e.g. PATCH_APPLIED | CONFIG_CHANGE | DUPLICATE
- resolution           : full resolution text — available on follow-up by ticket_id
- error_logs           : raw error log content
- description_match    : snippet from description matching the search term
- satisfaction_score   : customer satisfaction score 1-5 (1=very unsatisfied, 5=very satisfied)

---

## RULES
- Use conversation history ONLY as context, not as source of truth.
- Do NOT assume anything not present in the tickets.
- If there is a contradiction, trust the tickets over history.
- When asked for the "latest" ticket, use the created timestamp to determine recency.
- When asked about an agent, use the agent_id and agent_specialization fields.
- When asked about a resolution, use resolution_code and error_logs as primary sources.
- When asked for the full resolution text, say it is available — the user can ask "what was the full resolution for TK-XXXX".
- If multiple tickets match, list them all unless the question asks for a single result.
- For follow-up questions about a specific ticket, reference it by ticket_id.
- When asked about a date range, use the created timestamp to filter results.
- customer and tier refer to customer_id and customer_tier respectively.
"""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_ticket_id(question: str) -> str | None:
    match = re.search(r"TK-\d{4}-\d{6}", question)
    return match.group(0) if match else None


def _empty_meta(mode: str) -> dict:
    """Return empty meta dict for early returns."""
    return {"chroma_hits": 0, "neo4j_hits": 0, "mode": mode}


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def ask(
    question: str,
    chat_history: list[dict] | None = None,
    filters: dict | None = None,
    mode: Literal["chroma", "neo4j", "hybrid"] = "hybrid",
    model: str = settings.llm,
) -> tuple[str, dict]:

    context       = ""
    graph_context = ""
    top_docs      = []

    # ---------------------------
    # 1. CHAT HISTORY
    # ---------------------------
    if chat_history is None:
        chat_history = []

    history_text = "\n".join(
        f"Q: {h.get('question')}\nA: {h.get('answer')}"
        for h in chat_history
    )

    # ---------------------------
    # 2. CHROMA PATH
    # ---------------------------
    if mode in ("chroma", "hybrid"):

        auto_filters = extract_filters_llm(question, model=model)
        if filters:
            auto_filters.update(filters)

        candidates = retriever.retrieve(question, k=50, filters=auto_filters)

        if not candidates:
            if mode == "chroma":
                return "No relevant tickets found.", _empty_meta(mode)
        else:
            top_docs = reranker.rerank(
                query=question,
                documents=candidates,
                top_k=10,
            )
            if not top_docs and mode == "chroma":
                return "No relevant tickets found after reranking.", _empty_meta(mode)

            context = "\n\n".join(
                f"[score={item['score']:.3f}]\n{item['doc'].content}"
                for item in top_docs
            )

    # ---------------------------
    # 3. NEO4J PATH
    # ---------------------------
    if mode in ("neo4j", "hybrid"):
        ticket_id = _extract_ticket_id(question)
        if ticket_id:
            detail  = neo4j.query_ticket_detail(ticket_id)
            related = neo4j.query_related_tickets(ticket_id, limit=5)

            graph_context = ""
            if detail:
                graph_context = (
                    f"[{detail['ticket_id']}] {detail['subject']} | "
                    f"priority={detail.get('priority', 'unknown')} | "
                    f"category={detail.get('category', 'unknown')} | "
                    f"region={detail.get('region', 'unknown')} | "
                    f"product={detail.get('product', 'unknown')} | "
                    f"agent_id={detail.get('agent_id', 'unknown')} | "
                    f"resolution_code={detail.get('resolution_code', 'unknown')} | "
                    f"satisfaction_score={detail.get('satisfaction_score', 'unknown')} | "
                    f"created={detail.get('created_at', '')}\n"
                    f"description: {detail.get('description', '')}\n"
                    f"resolution: {detail.get('resolution', '')}\n"
                    f"error_logs: {detail.get('error_logs', '')}"
                )

            if related:
                related_lines = "\n".join(
                    f"  - [{r['ticket_id']}] {r.get('subject', 'stub ticket — not fully ingested')} | "
                    f"priority={r.get('priority', 'unknown')} | "
                    f"product={r.get('product', 'unknown')} | "
                    f"resolution_code={r.get('resolution_code', 'unknown')} | "
                    f"resolution={r.get('resolution', 'unknown')} | "
                    f"created={r.get('created_at', 'unknown')}"
                    for r in related
                )
                graph_context += f"\n\nRelated tickets:\n{related_lines}"

        else:
            neo4j_filters = extract_neo4j_filters_llm(question, model=model)
            graph_context = _enrich_from_filters(neo4j_filters)

        if not graph_context and mode == "neo4j":
            return "No relevant tickets found in graph database.", _empty_meta(mode)

    # ---------------------------
    # 4. GUARD: nothing found
    # ---------------------------
    if not context and not graph_context:
        return "No relevant tickets found.", _empty_meta(mode)

    # ---------------------------
    # 5. PROMPT + LLM CALL
    # ---------------------------
    prompt = _build_prompt(question, history_text, context, graph_context)

    meta = {
        "chroma_hits": len(top_docs),
        "neo4j_hits":  1 if graph_context else 0,
        "mode":        mode,
    }

    return _llm_call(prompt, model=model), meta