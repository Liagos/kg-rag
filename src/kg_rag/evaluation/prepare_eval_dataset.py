"""
Evaluation dataset generator for ChromaDB vs Neo4j comparison.

Generates eval samples across 6 question types:
  - factual
  - troubleshooting
  - procedural
  - structured
  - filter
  - graph

Supports both OpenAI and Anthropic models.

Usage:
    python prepare_eval_dataset.py \
        --source data/raw/cloud_backup_stream_processor_tickets.json \
        --output data/evaluation/eval_dataset_v2.jsonl \
        --model gpt-4.1

    python prepare_eval_dataset.py \
        --source data/raw/cloud_backup_stream_processor_tickets.json \
        --output data/evaluation/eval_dataset_v2.jsonl \
        --model claude-sonnet-4-6
"""

import json
import random
import argparse
import logging

from pathlib import Path
from typing import Optional

import anthropic
from openai import OpenAI
from datetime import datetime


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPEN_API_KEY = None
ANTHROPIC_API_KEY = None

try:
    from kg_rag.config import settings

    OPEN_API_KEY = settings.OPEN_API_KEY
    ANTHROPIC_API_KEY = settings.ANTHROPIC_API_KEY

except Exception:
    import os

    OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

openai_client = OpenAI(
    api_key=OPEN_API_KEY
)

anthropic_client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY
)


# ---------------------------------------------------------------------------
# Unified LLM caller
# ---------------------------------------------------------------------------

def call_llm(
    model: str,
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """
    Unified interface for OpenAI + Anthropic models.
    """

    # ---------------- OPENAI ----------------

    if model.startswith("gpt"):
        response = openai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.choices[0].message.content.strip()

    # ---------------- CLAUDE ----------------

    if model.startswith("claude"):
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        text_parts = []

        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        return "\n".join(text_parts).strip()

    raise ValueError(f"Unsupported model: {model}")


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def sample_for_factual(
    data: list[dict],
    n: int,
) -> list[dict]:
    pool = [
        t for t in data
        if t.get("error_logs")
        and t.get("resolution")
        and t.get("description")
    ]

    return random.sample(
        pool,
        min(n, len(pool)),
    )


def sample_for_troubleshooting(
    data: list[dict],
    n: int,
) -> list[dict]:

    pool = [
        t for t in data
        if t.get("resolution")
        and len(t.get("resolution", "")) > 80
        and t.get("description")
    ]

    return random.sample(
        pool,
        min(n, len(pool)),
    )


def sample_for_procedural(
    data: list[dict],
    n: int,
) -> list[dict]:

    pool = [
        t for t in data
        if t.get("category") == "Technical Issue"
        and t.get("resolution")
        and t.get("error_logs")
    ]

    return random.sample(
        pool,
        min(n, len(pool)),
    )


def sample_for_structured(
    data: list[dict],
    n: int,
) -> list[dict]:

    pool = [
        t for t in data
        if t.get("product")
        and t.get("priority")
        and t.get("region")
        and t.get("created_at")
    ]

    by_priority = {}

    for t in pool:
        p = t["priority"]
        by_priority.setdefault(p, []).append(t)

    sampled = []

    per_bucket = max(1, n // 4)

    for priority_tickets in by_priority.values():
        sampled.extend(
            random.sample(
                priority_tickets,
                min(
                    per_bucket,
                    len(priority_tickets),
                ),
            )
        )

    return sampled[:n]


def sample_for_filter(
    data: list[dict],
    n: int,
) -> list[dict]:

    pool = [
        t for t in data
        if t.get("escalated")
        and t.get("customer_tier")
        and t.get("environment")
        and t.get("region")
        and t.get("agent_id")
    ]

    return random.sample(
        pool,
        min(n, len(pool)),
    )


def sample_for_graph(
    data: list[dict],
    n: int,
) -> list[dict]:

    pool = [
        t for t in data
        if t.get("related_tickets")
        and len(t["related_tickets"]) >= 2
        and t.get("resolution")
    ]

    return random.sample(
        pool,
        min(n, len(pool)),
    )


# ---------------------------------------------------------------------------
# Ground truth doc builder
# ---------------------------------------------------------------------------

def build_ground_truth_doc(
    t: dict,
) -> str:

    tags = ", ".join(t.get("tags") or [])

    return (
        f"Ticket ID: {t['ticket_id']}\n"
        f"Product: {t.get('product')} "
        f"(v{t.get('product_version')})\n"
        f"Module: {t.get('product_module')}\n"
        f"Category: {t.get('category')}  "
        f"Priority: {t.get('priority')}  "
        f"Severity: {t.get('severity')}\n"
        f"Region: {t.get('region')}  "
        f"Environment: {t.get('environment')}\n"
        f"Customer Tier: {t.get('customer_tier')}  "
        f"Escalated: {t.get('escalated')}\n"
        f"Agent: {t.get('agent_id')}  "
        f"Specialization: {t.get('agent_specialization')}\n"
        f"Satisfaction Score: "
        f"{t.get('satisfaction_score')}\n"
        f"Subject: {t.get('subject')}\n"
        f"Description:\n"
        f"{t.get('description')}\n"
        f"Resolution:\n"
        f"{t.get('resolution')}\n"
        f"Error Logs:\n"
        f"{t.get('error_logs')}\n"
        f"Tags: {tags}"
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FACTUAL_PROMPT = """
You are creating evaluation questions for a RAG system over support tickets.

Generate ONE factual question that includes the ticket ID.

The question should ask about:
- error code
- root cause
- product version
- module
- agent
- resolution code

Avoid vague questions.

Ticket:
{doc}

Return ONLY valid JSON:

{{
  "question": "...",
  "answer": "...",
  "difficulty": "easy" | "medium" | "hard"
}}
"""


TROUBLESHOOTING_PROMPT = """
You are creating evaluation questions for a RAG system over support tickets.

Generate ONE troubleshooting question and answer.

The question should ask:
- how the issue was diagnosed
- how it was resolved
- what fixed the issue

Ticket:
{doc}

Return ONLY valid JSON:

{{
  "question": "...",
  "answer": "...",
  "difficulty": "easy" | "medium" | "hard"
}}
"""


PROCEDURAL_PROMPT = """
You are creating evaluation questions for a RAG system over support tickets.

Generate ONE procedural question and answer.

The question should ask:
- what steps should be followed
- how to resolve a similar issue

Ticket:
{doc}

Return ONLY valid JSON:

{{
  "question": "...",
  "answer": "...",
  "difficulty": "medium" | "hard"
}}
"""


STRUCTURED_PROMPT = """
You are creating evaluation questions for a RAG system over support tickets.

Generate ONE structured filter question and answer.

Question examples:
- latest critical tickets for X product
- tickets in Y region
- high severity issues in production

Ticket:
{doc}

Return ONLY valid JSON:

{{
  "question": "...",
  "answer": "...",
  "difficulty": "easy" | "medium"
}}
"""


FILTER_PROMPT = """
You are creating evaluation questions for a RAG system over support tickets.

Generate ONE multi-filter question and answer.

Combine 3+ filters:
- environment
- escalated
- customer tier
- priority
- region
- agent

Ticket:
{doc}

Return ONLY valid JSON:

{{
  "question": "...",
  "answer": "...",
  "difficulty": "medium" | "hard"
}}
"""


GRAPH_PROMPT = """
You are creating evaluation questions for a RAG system over support tickets.

Generate ONE graph traversal question and answer.

The ticket has related tickets:
{related}

The answer should mention:
- related ticket IDs
- relationships
- resolutions

Ticket:
{doc}

Return ONLY valid JSON:

{{
  "question": "...",
  "answer": "...",
  "difficulty": "medium" | "hard"
}}
"""


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def generate_sample(
    model: str,
    ticket: dict,
    question_type: str,
    eval_idx: int,
) -> dict | None:

    doc = build_ground_truth_doc(ticket)

    prompts = {
        "factual": FACTUAL_PROMPT.format(
            doc=doc,
        ),

        "troubleshooting": TROUBLESHOOTING_PROMPT.format(
            doc=doc,
        ),

        "procedural": PROCEDURAL_PROMPT.format(
            doc=doc,
        ),

        "structured": STRUCTURED_PROMPT.format(
            doc=doc,
        ),

        "filter": FILTER_PROMPT.format(
            doc=doc,
        ),

        "graph": GRAPH_PROMPT.format(
            doc=doc,
            related=ticket.get(
                "related_tickets",
                [],
            ),
        ),
    }

    prompt = prompts[question_type]

    try:
        raw = call_llm(
            model=model,
            prompt=prompt,
            temperature=0.3,
        )

        raw = (
            raw.replace("```json", "")
            .replace("```", "")
            .strip()
        )

        generated = json.loads(raw)

    except Exception as e:
        logger.warning(
            "Failed to generate %s sample for %s: %s",
            question_type,
            ticket["ticket_id"],
            e,
        )

        return None

    # backend mode mapping

    mode_map = {
        "factual": "chroma",
        "troubleshooting": "chroma",
        "procedural": "chroma",
        "structured": "neo4j",
        "filter": "neo4j",
        "graph": "neo4j",
    }

    # metadata filters

    metadata_filters = {}

    if question_type in (
        "factual",
        "troubleshooting",
        "procedural",
    ):
        metadata_filters = {
            k: v
            for k, v in {
                "product": ticket.get("product"),
                "priority": ticket.get("priority"),
                "region": ticket.get("region"),
                "category": ticket.get("category"),
            }.items()
            if v
        }

    return {
        "ticket_id": ticket["ticket_id"],

        "eval_id": (
            f"{ticket['ticket_id']}"
            f"_eval_{question_type}_{eval_idx}"
        ),

        "question": generated["question"],

        "answer": generated["answer"],

        "difficulty": generated.get(
            "difficulty",
            "medium",
        ),

        "question_type": question_type,

        "mode": mode_map[question_type],

        "generation_model": model,

        "relevant_ticket_id": ticket["ticket_id"],

        "metadata_filters": metadata_filters,

        "ground_truth_document": doc,

        "ground_truth_metadata": {
            "ticket_id": ticket["ticket_id"],
            "product": ticket.get("product"),
            "product_version": ticket.get(
                "product_version"
            ),
            "product_module": ticket.get(
                "product_module"
            ),
            "category": ticket.get("category"),
            "priority": ticket.get("priority"),
            "severity": ticket.get("severity"),
            "region": ticket.get("region"),
            "environment": ticket.get(
                "environment"
            ),
            "escalated": ticket.get("escalated"),
            "customer_tier": ticket.get(
                "customer_tier"
            ),
            "agent_id": ticket.get("agent_id"),
            "satisfaction_score": ticket.get(
                "satisfaction_score"
            ),
            "resolution_code": ticket.get(
                "resolution_code"
            ),
            "related_tickets": ticket.get(
                "related_tickets",
                [],
            ),
        },
    }


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_dataset(
    source_path: str,
    output_path: Optional[str] = None,
    model: str = "gpt-4.1",
    n_factual: int = 20,
    n_troubleshooting: int = 20,
    n_procedural: int = 15,
    n_structured: int = 20,
    n_filter: int = 15,
    n_graph: int = 10,
    seed: int = 42,
):
    # build dynamic output path if not provided
    if output_path is None:
        source_stem = Path(source_path).stem
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/evaluation/eval_{source_stem}_{timestamp}.jsonl"
        logger.info("Output path not specified — using: %s", output_path)

    random.seed(seed)

    logger.info(
        "Loading tickets from %s",
        source_path,
    )

    with open(
        source_path,
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    logger.info(
        "Loaded %d tickets",
        len(data),
    )

    logger.info(
        "Using generation model: %s",
        model,
    )

    plan = [
        (
            "factual",
            sample_for_factual,
            n_factual,
        ),

        (
            "troubleshooting",
            sample_for_troubleshooting,
            n_troubleshooting,
        ),

        (
            "procedural",
            sample_for_procedural,
            n_procedural,
        ),

        (
            "structured",
            sample_for_structured,
            n_structured,
        ),

        (
            "filter",
            sample_for_filter,
            n_filter,
        ),

        (
            "graph",
            sample_for_graph,
            n_graph,
        ),
    ]

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_written = 0

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as out:

        for (
            question_type,
            sampler,
            n,
        ) in plan:

            logger.info(
                "Generating %d %s samples...",
                n,
                question_type,
            )

            tickets = sampler(data, n)

            for idx, ticket in enumerate(tickets):

                sample = generate_sample(
                    model=model,
                    ticket=ticket,
                    question_type=question_type,
                    eval_idx=idx,
                )

                if sample:
                    out.write(
                        json.dumps(
                            sample,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    total_written += 1

            logger.info(
                "✅ %s done — %d samples written so far",
                question_type,
                total_written,
            )

    logger.info(
        "🎉 Dataset complete — %d samples → %s",
        total_written,
        output_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Generate eval dataset "
            "for ChromaDB vs Neo4j"
        )
    )

    parser.add_argument(
        "--source",
        default=(
            "data/raw/"
            "cloud_backup_stream_processor_tickets.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output path — defaults to data/evaluation/eval_{source}_{timestamp}.jsonl",
    )

    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help=(
            "Generation model "
            "(gpt-* or claude-*)"
        ),
    )

    parser.add_argument(
        "--n-factual",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--n-troubleshooting",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--n-procedural",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--n-structured",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--n-filter",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--n-graph",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    generate_dataset(
        source_path=args.source,
        output_path=args.output,
        model=args.model,
        n_factual=args.n_factual,
        n_troubleshooting=args.n_troubleshooting,
        n_procedural=args.n_procedural,
        n_structured=args.n_structured,
        n_filter=args.n_filter,
        n_graph=args.n_graph,
        seed=args.seed,
    )