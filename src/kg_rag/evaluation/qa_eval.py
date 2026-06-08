"""
QA Evaluation — ChromaDB vs Neo4j comparison.

Evaluates answer quality using an LLM judge that scores semantic correctness
rather than exact match. A different ticket can score highly if it contains
the same information as the ground truth.

Supports both OpenAI and Anthropic models.

Usage:
    python -m kg_rag.evaluation.qa_eval \
        --dataset data/evaluation/eval_dataset_v2.jsonl \
        --modes chroma neo4j \
        --models gpt-4.1 claude-sonnet-4-6 \
        --judge-model gpt-4.1 \
        --output data/evaluation/results/qa_eval_results.jsonl
"""

import sys
import json
import logging
import argparse
import re
import time
from datetime import timedelta


from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging

import anthropic
from openai import OpenAI

from kg_rag.config import settings
from kg_rag.rag.qa import ask

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
tqdm_logging.set_level(logging.INFO)
tqdm_logging.set_log_rate(timedelta(seconds=5))

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

openai_client = OpenAI(api_key=settings.OPEN_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)  # ← fixed field name


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

@dataclass
class QAEvalSample:
    eval_id: str
    question: str
    answer: str
    difficulty: str
    question_type: str
    mode: str
    relevant_ticket_id: str
    metadata_filters: dict
    ground_truth_document: str
    ground_truth_metadata: dict


@dataclass
class QAEvalResult:
    eval_id: str
    question: str
    mode: str  # full key e.g. "neo4j:gpt-4.1"
    model: str
    difficulty: str
    question_type: str
    expected: str
    actual: str
    correct: bool
    score: float
    reasoning: str


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_qa_eval_dataset(path: str) -> list[QAEvalSample]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(QAEvalSample(
                eval_id=d["eval_id"],
                question=d["question"],
                answer=d["answer"],
                difficulty=d["difficulty"],
                question_type=d["question_type"],
                mode=d.get("mode", "hybrid"),
                relevant_ticket_id=d["relevant_ticket_id"],
                metadata_filters=d.get("metadata_filters", {}),
                ground_truth_document=d.get("ground_truth_document", ""),
                ground_truth_metadata=d.get("ground_truth_metadata", {}),
            ))
    logger.info("Loaded %d eval samples", len(samples))
    return samples


def filter_samples(samples: list[QAEvalSample]) -> list[QAEvalSample]:
    """Keep only samples where question references a ticket ID,
    or question type doesn't need a specific ticket."""
    filtered = []
    for s in samples:
        if re.search(r"TK-\d{4}-\d+", s.question):
            filtered.append(s)
        elif s.question_type in ("structured", "filter", "graph"):
            filtered.append(s)
    logger.info("Filtered %d → %d samples", len(samples), len(filtered))
    return filtered


# ---------------------------------------------------------------------------
# Unified LLM caller with retry
# ---------------------------------------------------------------------------

def call_llm(
        model: str,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 1024,
) -> str:
    """Unified interface for OpenAI and Anthropic models with exponential backoff retry."""
    for attempt in range(3):
        try:
            if model.startswith("claude"):
                response = anthropic_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "\n".join(
                    block.text for block in response.content
                    if hasattr(block, "text")
                ).strip()
            else:
                # default to OpenAI for gpt-*, o1, o3, and any future models
                response = openai_client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt == 2:
                raise
            wait = 2 ** attempt  # 1s, 2s
            logger.warning(
                "LLM call failed (attempt %d/3), retrying in %ds: %s",
                attempt + 1, wait, e,
            )
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """
You are evaluating a Jira support ticket assistant.

Your job is to score whether the ACTUAL answer correctly addresses the QUESTION,
using the GROUND TRUTH as reference.

IMPORTANT:
The actual answer may reference a DIFFERENT ticket than the ground truth
but still be correct if it contains the same information:
- same error code
- same resolution approach
- same product issue
- same root cause

Score based on information quality, NOT ticket ID matching.

---

QUESTION:
{question}

GROUND TRUTH ANSWER:
{expected}

GROUND TRUTH DOCUMENT:
{ground_truth}

ACTUAL ANSWER:
{actual}

---

Score the actual answer from 0.0 to 1.0:

1.0 = fully correct — information matches, complete answer
0.7 = mostly correct — key information present, minor details missing
0.5 = partially correct — some relevant information but incomplete or vague
0.3 = mostly wrong — little relevant information
0.0 = wrong, missing, contradicts ground truth, or says "cannot find"

Return ONLY valid JSON, no markdown:
{{"score": 0.8, "reasoning": "one sentence explanation"}}
"""


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

def llm_judge(
        question: str,
        expected: str,
        actual: str,
        ground_truth_document: str,
        judge_model: str,
) -> tuple[float, str]:
    """Score an answer using an LLM judge. Returns (score, reasoning)."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected=expected,
        actual=actual,
        ground_truth=ground_truth_document[:1000],
    )
    try:
        raw = call_llm(model=judge_model, prompt=prompt, temperature=0)
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return float(parsed["score"]), parsed["reasoning"]
    except Exception as e:
        logger.warning("Judge failed for question '%s': %s", question[:60], e)
        return 0.0, f"Judge error: {e}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_qa(
        samples: list[QAEvalSample],
        modes: list[str],
        models: list[str],
        judge_model: str,
) -> dict[str, list[QAEvalResult]]:
    """Run ask() for each mode+model combination and score with LLM judge."""

    results: dict[str, list[QAEvalResult]] = {
        f"{mode}:{model}": []
        for mode in modes
        for model in models
    }

    for sample in tqdm(samples,
                       desc="Evaluating QA",
                       unit="sample",
                       dynamic_ncols=True,
                       leave=True,
                       ):
        for mode in modes:
            for model in models:
                key = f"{mode}:{model}"
                logger.debug("  [%s] %s", key, sample.question[:60])

                try:
                    answer, _ = ask(  # ← unpack, discard meta
                        sample.question,
                        mode=mode,
                        model=model,
                    )
                    actual = answer
                except Exception as e:
                    logger.warning("ask() failed [%s] %s: %s", key, sample.eval_id, e)
                    actual = f"ERROR: {e}"

                score, reasoning = llm_judge(
                    question=sample.question,
                    expected=sample.answer,
                    actual=actual,
                    ground_truth_document=sample.ground_truth_document,
                    judge_model=judge_model,
                )

                results[key].append(QAEvalResult(
                    eval_id=sample.eval_id,
                    question=sample.question,
                    mode=key,  # ← full key e.g. "neo4j:gpt-4.1"
                    model=model,
                    difficulty=sample.difficulty,
                    question_type=sample.question_type,
                    expected=sample.answer,
                    actual=actual,
                    correct=score >= 0.7,
                    score=score,
                    reasoning=reasoning,
                ))

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_qa_metrics(results: list[QAEvalResult]) -> dict:
    if not results:
        return {}
    avg_score = sum(r.score for r in results) / len(results)
    accuracy = sum(1 for r in results if r.correct) / len(results)
    return {
        "count": len(results),
        "accuracy": round(accuracy, 4),
        "avg_score": round(avg_score, 4),
        "correct": sum(1 for r in results if r.correct),
    }


def print_qa_metrics(title: str, metrics: dict):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        print(f"  {k:<15}: {v}")


def print_comparison(results: dict[str, list[QAEvalResult]]):
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'mode:model':<40} {'accuracy':<12} {'avg_score':<12} {'correct/total'}")
    print(f"  {'-' * 80}")
    for key, mode_results in results.items():
        m = compute_qa_metrics(mode_results)
        print(
            f"  {key:<40} "
            f"{m['accuracy']:<12} "
            f"{m['avg_score']:<12} "
            f"{m['correct']}/{m['count']}"
        )


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def run_full_report(results: dict[str, list[QAEvalResult]]):
    for key, mode_results in results.items():

        # overall
        print_qa_metrics(
            f"QA METRICS — {key.upper()}",
            compute_qa_metrics(mode_results),
        )

        # by question type
        by_type: dict[str, list] = defaultdict(list)
        for r in mode_results:
            by_type[r.question_type].append(r)
        for qtype, subset in sorted(by_type.items()):
            print_qa_metrics(
                f"  {key.upper()} — type: {qtype}",
                compute_qa_metrics(subset),
            )

        # by difficulty
        by_diff: dict[str, list] = defaultdict(list)
        for r in mode_results:
            by_diff[r.difficulty].append(r)
        for diff, subset in sorted(by_diff.items()):
            print_qa_metrics(
                f"  {key.upper()} — difficulty: {diff}",
                compute_qa_metrics(subset),
            )

    print_comparison(results)

    # worst 5 per mode:model
    for key, mode_results in results.items():
        worst = sorted(mode_results, key=lambda r: r.score)[:5]
        print(f"\n{'=' * 60}")
        print(f"WORST 5 — {key.upper()}")
        print(f"{'=' * 60}")
        for r in worst:
            print(f"  [{r.score:.2f}] {r.eval_id}")
            print(f"         Q: {r.question[:80]}")
            print(f"         Reason: {r.reasoning}")
            print()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: dict[str, list[QAEvalResult]], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, mode_results in results.items():
            for r in mode_results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    logger.info("Results saved → %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="QA evaluation — ChromaDB vs Neo4j")

    parser.add_argument(
        "--dataset",
        default="data/evaluation/eval_dataset_v2.jsonl",
        help="Path to eval dataset jsonl",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["chroma", "neo4j"],
        choices=["chroma", "neo4j", "hybrid"],
        help="Modes to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4.1"],
        help="e.g. gpt-4.1 claude-sonnet-4-6",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1",
        help="Model used as evaluator/judge",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation/results/qa_eval_results.jsonl",
        help="Path to save results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples for quick testing",
    )

    args = parser.parse_args()

    samples = load_qa_eval_dataset(args.dataset)
    samples = filter_samples(samples)

    if args.limit:
        samples = samples[:args.limit]
        logger.info("Limited to %d samples", len(samples))

    results = evaluate_qa(
        samples=samples,
        modes=args.modes,
        models=args.models,
        judge_model=args.judge_model,
    )

    run_full_report(results)
    save_results(results, args.output)
