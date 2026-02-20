"""
Unified eval runner for topic blocker guardrails.

Runs every eval JSONL against every blocker implementation and prints
a confusion matrix for each combination.

Structure:
  evals/engine.jsonl           — tests the matching engine (synthetic policy)
  evals/block_investment.jsonl — tests "Block investment questions" topic
  results/                     — eval results saved here (JSON)

Run all evals:
  pytest tests/test_litellm/proxy/guardrails/guardrail_hooks/content_filter/topic_blocker/test_eval.py -v -s

Run a specific eval:
  pytest ... -k "engine"
  pytest ... -k "investment_keyword"
  pytest ... -k "investment_embedding_minilm"
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import List

import pytest

sys.path.insert(0, os.path.abspath("../../"))

from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.topic_blocker.keyword_blocker import (
    DeniedTopic,
    TopicBlocker,
)

EVAL_DIR = os.path.join(os.path.dirname(__file__), "evals")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── Helpers ───────────────────────────────────────────────────────


def _load_jsonl(filename: str) -> List[dict]:
    """Load eval cases from a JSONL file. One JSON object per line."""
    cases = []
    path = os.path.join(EVAL_DIR, filename)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(
                {
                    "sentence": obj["sentence"],
                    "expected": obj["expected"],
                    "test": obj["test"],
                }
            )
    return cases


def _run(checker, text: str) -> str:
    """Run a blocker's check method, return 'BLOCK' or 'ALLOW'."""
    try:
        checker.check(text)
        return "ALLOW"
    except HTTPException as e:
        if e.status_code == 403:
            return "BLOCK"
        raise


def _confusion_matrix(checker, cases: List[dict], label: str):
    """Run all cases, print confusion matrix, save results JSON."""
    tp = fp = tn = fn = 0
    wrong = []
    rows = []

    for case in cases:
        expected = case["expected"]
        actual = _run(checker, case["sentence"])
        correct = expected == actual

        rows.append(
            {
                "sentence": case["sentence"],
                "expected": expected,
                "actual": actual,
                "correct": correct,
                "test": case["test"],
            }
        )

        if expected == "BLOCK" and actual == "BLOCK":
            tp += 1
        elif expected == "ALLOW" and actual == "ALLOW":
            tn += 1
        elif expected == "BLOCK" and actual == "ALLOW":
            fn += 1
            wrong.append(f"  FN: {case['sentence']!r:60s} — {case['test']}")
        elif expected == "ALLOW" and actual == "BLOCK":
            fp += 1
            wrong.append(f"  FP: {case['sentence']!r:60s} — {case['test']}")

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    accuracy = (tp + tn) / total if total > 0 else 0

    # Print
    print("\n")
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)
    print(f"  Total cases:  {total}")
    print(f"  Correct:      {tp + tn}")
    print(f"  Wrong:        {fp + fn}")
    print()
    print(f"  TP (correctly blocked):  {tp}")
    print(f"  TN (correctly allowed):  {tn}")
    print(f"  FP (wrongly blocked):    {fp}")
    print(f"  FN (wrongly allowed):    {fn}")
    print()
    print(f"  Precision:  {precision:.1%}")
    print(f"  Recall:     {recall:.1%}")
    print(f"  F1:         {f1:.1%}")
    print(f"  Accuracy:   {accuracy:.1%}")
    print()
    if wrong:
        print("WRONG ANSWERS:")
        for line in wrong:
            print(line)
    else:
        print("ALL CASES CORRECT")
    print("=" * 70)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_label = label.lower().replace(" ", "_").replace("—", "-")
    result = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "wrong": wrong,
        "rows": rows,
    }
    result_path = os.path.join(RESULTS_DIR, f"{safe_label}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Blocker factories ─────────────────────────────────────────────


def _keyword_blocker_engine():
    """Keyword blocker configured for the synthetic engine eval."""
    return TopicBlocker(
        denied_topics=[
            DeniedTopic(
                topic_name="test_engine",
                identifier_words=["alpha", "bravo"],
                block_words=["red", "blue"],
                always_block_phrases=["block this phrase", "also block this"],
                exception_phrases=["alpha safe context", "bravo safe context"],
            ),
        ]
    )


def _keyword_blocker_investment():
    """Keyword blocker configured for 'Block investment questions'."""
    return TopicBlocker(
        denied_topics=[
            DeniedTopic(
                topic_name="investment_questions",
                identifier_words=[
                    "stock", "stocks", "equity", "equities", "shares", "ticker",
                    "nasdaq", "dow jones", "s&p 500", "nyse",
                    "bond", "bonds", "treasury",
                    "mutual fund", "etf", "index fund", "hedge fund",
                    "crypto", "cryptocurrency", "bitcoin", "ethereum", "blockchain",
                    "portfolio", "brokerage", "trading", "forex",
                    "options trading", "futures trading", "commodities",
                    "dividend", "capital gains", "ipo", "reit",
                    "401k", "ira", "roth", "pension", "annuity",
                    "financial advisor", "wealth management", "robo-advisor",
                ],
                block_words=[
                    "buy", "sell", "invest", "price", "value", "worth",
                    "return", "returns", "profit", "loss", "gain",
                    "performance", "recommend", "advice", "should i",
                    "best", "top", "how to", "how do", "strategy",
                    "forecast", "prediction", "outlook", "analysis",
                    "compare", "risk", "grow", "allocate", "diversify",
                    "yield", "ratio", "this year", "right now",
                    "good time", "safe", "start", "open", "work",
                ],
                always_block_phrases=[
                    "should i invest", "investment advice", "financial advice",
                    "how to invest", "how to trade", "stock tips", "trading tips",
                    "best stocks to buy", "best crypto to buy",
                    "best etf", "best mutual fund", "best index fund",
                    "market prediction", "stock market forecast",
                    "retirement planning", "grow my wealth",
                    "is bitcoin a good investment", "is gold a safe investment",
                    "is real estate a good investment", "emerging markets",
                    "pe ratio",
                ],
                exception_phrases=[
                    "in stock", "stock up", "stock room", "stock inventory",
                    "invest time", "invest effort", "invest energy",
                    "invested in learning", "invested in a good",
                    "return policy", "return this item", "return the item",
                    "share the document", "share with me", "share your",
                    "options menu", "options are available",
                    "bond with", "bonding",
                    "gold standard", "golden rule",
                    "gain access", "gained access",
                    "loss of data", "loss prevention",
                    "trading card", "not interested in investing",
                    "portfolio of work", "token-based",
                    "yield sign", "returns on my serve",
                    "futures schedule",
                ],
            ),
        ]
    )


EMBEDDING_TOPICS = [
    "investment advice",
    "stock trading",
    "buying stocks",
    "selling stocks",
    "stock price",
    "stock market forecast",
    "cryptocurrency investment",
    "bitcoin trading",
    "mutual fund recommendation",
    "ETF recommendation",
    "index fund advice",
    "bond investment",
    "portfolio allocation",
    "retirement planning",
    "401k advice",
    "Roth IRA",
    "dividend investing",
    "hedge fund",
    "forex trading",
    "options trading",
    "financial advisor",
    "wealth management",
    "capital gains",
    "brokerage account",
    "real estate investment",
    "gold investment",
    "emerging markets investing",
    "passive income from investments",
    "securities trading",
    "fixed income instruments",
    "capital markets",
    "derivatives trading",
    "day trading",
    "building wealth",
    "nest egg retirement savings",
]


def _embedding_blocker_minilm():
    """Embedding blocker: all-MiniLM-L6-v2 (80MB, ~2-5ms)."""
    from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.topic_blocker.embedding_blocker import (
        EmbeddingTopicBlocker,
    )

    return EmbeddingTopicBlocker(
        blocked_topics=EMBEDDING_TOPICS,
        threshold=0.5,
        model_name="all-MiniLM-L6-v2",
    )


# ── Engine eval — keyword blocker ─────────────────────────────────


class TestEngineKeyword:
    """Engine eval with the keyword blocker."""

    @pytest.fixture(scope="class")
    def blocker(self):
        return _keyword_blocker_engine()

    @pytest.fixture(scope="class")
    def cases(self):
        return _load_jsonl("engine.jsonl")

    def test_confusion_matrix(self, blocker, cases):
        _confusion_matrix(blocker, cases, "Engine — Keyword Blocker")


# ── Investment eval — keyword blocker ─────────────────────────────


class TestInvestmentKeyword:
    """Investment eval with the keyword blocker."""

    @pytest.fixture(scope="class")
    def blocker(self):
        return _keyword_blocker_investment()

    @pytest.fixture(scope="class")
    def cases(self):
        return _load_jsonl("block_investment.jsonl")

    def test_confusion_matrix(self, blocker, cases):
        _confusion_matrix(blocker, cases, "Block Investment — Keyword Blocker")


# ── Investment eval — embedding MiniLM ────────────────────────────


class TestInvestmentEmbeddingMiniLM:
    """Investment eval with all-MiniLM-L6-v2 (80MB)."""

    @pytest.fixture(scope="class")
    def blocker(self):
        return _embedding_blocker_minilm()

    @pytest.fixture(scope="class")
    def cases(self):
        return _load_jsonl("block_investment.jsonl")

    def test_confusion_matrix(self, blocker, cases):
        _confusion_matrix(blocker, cases, "Block Investment — Embedding MiniLM (80MB)")


