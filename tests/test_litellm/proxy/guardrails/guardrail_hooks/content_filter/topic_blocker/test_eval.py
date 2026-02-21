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
import time
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


def _run(checker, text: str) -> dict:
    """Run a blocker's check method, return result with confidence score."""
    try:
        checker.check(text)
        # Get confidence score for ALLOW decisions too
        score = 0.0
        matched_topic = None
        if hasattr(checker, "is_blocked") and text and text.strip():
            _, matched_topic, score = checker.is_blocked(text)
        return {"decision": "ALLOW", "score": score, "matched_topic": matched_topic}
    except HTTPException as e:
        if e.status_code == 403:
            detail = e.detail if isinstance(e.detail, dict) else {}
            return {
                "decision": "BLOCK",
                "score": detail.get("score", 1.0),
                "matched_topic": detail.get("topic"),
                "match_type": detail.get("match_type"),
            }
        raise


def _confusion_matrix(checker, cases: List[dict], label: str):
    """Run all cases, print confusion matrix, save results JSON."""
    tp = fp = tn = fn = 0
    wrong = []
    rows = []

    latencies = []

    for case in cases:
        expected = case["expected"]
        t0 = time.perf_counter()
        result = _run(checker, case["sentence"])
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)
        actual = result["decision"]
        score = result["score"]
        matched_topic = result.get("matched_topic")
        correct = expected == actual

        rows.append(
            {
                "sentence": case["sentence"],
                "expected": expected,
                "actual": actual,
                "correct": correct,
                "test": case["test"],
                "score": score,
                "matched_topic": matched_topic,
                "latency_ms": round(latency_ms, 3),
            }
        )

        if expected == "BLOCK" and actual == "BLOCK":
            tp += 1
        elif expected == "ALLOW" and actual == "ALLOW":
            tn += 1
        elif expected == "BLOCK" and actual == "ALLOW":
            fn += 1
            wrong.append(
                f"  FN (score={score:.3f}): {case['sentence']!r:60s} — {case['test']}"
            )
        elif expected == "ALLOW" and actual == "BLOCK":
            fp += 1
            wrong.append(
                f"  FP (score={score:.3f}): {case['sentence']!r:60s} — {case['test']}"
            )

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
    # Latency stats
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    print(f"  Precision:  {precision:.1%}")
    print(f"  Recall:     {recall:.1%}")
    print(f"  F1:         {f1:.1%}")
    print(f"  Accuracy:   {accuracy:.1%}")
    print()
    print(f"  Latency p50:  {p50:.1f}ms")
    print(f"  Latency p95:  {p95:.1f}ms")
    print(f"  Latency avg:  {avg_lat:.1f}ms")
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
        "latency_p50_ms": round(p50, 3),
        "latency_p95_ms": round(p95, 3),
        "latency_avg_ms": round(avg_lat, 3),
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
                    # Stocks & equities
                    "stock", "stocks", "equity", "equities", "shares", "ticker",
                    "nasdaq", "dow jones", "s&p 500", "nyse",
                    "ftse", "nikkei", "dax", "sensex",
                    "blue chip", "penny stocks",
                    "securities",
                    # Bonds & fixed income
                    "bond", "bonds", "treasury", "fixed income",
                    # Funds
                    "mutual fund", "etf", "index fund", "hedge fund",
                    # Crypto
                    "crypto", "cryptocurrency", "bitcoin", "ethereum", "blockchain",
                    # Portfolio & accounts
                    "portfolio", "portfolios", "brokerage",
                    "trading", "forex", "day trading",
                    "options trading", "futures trading", "commodities",
                    "short selling", "derivatives",
                    # Financial metrics
                    "dividend", "capital gains", "ipo", "reit",
                    "market cap", "market capitalization",
                    # Retirement accounts
                    "401k", "ira", "roth", "pension", "annuity",
                    # Advisors & brokerages
                    "financial advisor", "financial planner",
                    "wealth management", "robo-advisor",
                    "vanguard", "fidelity", "schwab", "robinhood",
                    # Investment variants (stemming)
                    "invest", "investing", "investment", "investments", "investors",
                    # Funds (generic)
                    "funds",
                    # Commodities
                    "gold", "silver", "commodity",
                    # Savings & wealth (financial context)
                    "savings account", "money market",
                    "compound interest",
                    # Other financial
                    "capital markets", "passive income",
                ],
                block_words=[
                    "buy", "sell", "purchase", "price", "value", "worth",
                    "return", "returns", "profit", "loss", "gain",
                    "performance", "performing", "recommend", "advice",
                    "should i", "should", "tell me",
                    "best", "top", "good", "how to", "how do", "how does",
                    "strategy", "explain", "what are", "what is",
                    "forecast", "prediction", "outlook", "analysis",
                    "compare", "comparing", "risk", "grow", "allocate", "diversify",
                    "yield", "ratio", "this year", "right now",
                    "good time", "safe", "safest", "start", "open", "work",
                    "enter", "follow", "suggested", "thinking",
                    "looking", "look like", "latest", "trends", "crash",
                    "read", "chart", "today", "difference",
                    "apps", "app", "better", "vs",
                    "protect", "inflation",
                    "opportunity", "opportunities",
                    "tips", "rate", "current",
                ],
                always_block_phrases=[
                    "should i invest", "investment advice", "financial advice",
                    "how to invest", "how to trade", "stock tips", "trading tips",
                    "best stocks to buy", "best crypto to buy",
                    "best etf", "best mutual fund", "best index fund",
                    "market prediction", "stock market forecast",
                    "retirement planning", "grow my wealth", "build wealth",
                    "is bitcoin a good investment", "is gold a safe investment",
                    "is real estate a good investment", "emerging markets",
                    "pe ratio",
                    # Market-specific phrases (avoids FP on "farmer's market")
                    "market trends", "enter the market", "market going to",
                    "market crash", "market cap",
                    # Retirement & savings placement
                    "retirement savings", "compound interest",
                    # Wealth & income
                    "passive income", "protect my wealth",
                    # Specific financial products
                    "dollar cost averaging", "crypto wallet",
                    "money market",
                ],
                phrase_patterns=[
                    # --- Paraphrase patterns (catch rewording of investment intent) ---
                    # "put/park/place/stash money/cash/savings" patterns
                    r"\b(?:put|park|place|keep|stash)\b.{0,30}\b(?:money|cash|savings)\b",
                    # "grow/build/increase/protect wealth/nest egg/money"
                    r"\b(?:grow|build|increase|protect)\b.{0,20}\b(?:wealth|nest egg)\b",
                    # "make money/savings grow/work harder"
                    r"\b(?:make|get)\b.{0,20}\b(?:money|savings|cash)\b.{0,20}\b(?:grow|work|harder)\b",
                    # "what/best/smartest to do with money/cash/$X"
                    r"\b(?:what|smartest|best)\b.{0,30}\b(?:do with|thing to do)\b.{0,20}(?:\b(?:money|cash)\b|\$\d)",
                    # "what to do with spare/extra cash/money"
                    r"\b(?:spare|extra)\b.{0,10}\b(?:cash|money)\b",
                    # "savings rate for retirement" / "good savings for retirement"
                    r"\bsavings\b.{0,15}\b(?:rate|for retirement)\b",
                    # "how to make passive income"
                    r"\bpassive\s+income\b",
                    # "best way to grow my [wealth/money/savings]"
                    r"\bbest way to\b.{0,15}\b(?:grow|invest|build)\b",
                    # "good/safe/safest place for my [savings/money/retirement]"
                    r"\b(?:good|safe|safest|best)\s+place\b.{0,25}\b(?:savings|money|retirement)\b",
                ],
                exception_phrases=[
                    "in stock", "stock up", "stock room", "stock inventory",
                    "invest time", "invest effort", "invest energy",
                    "invested in learning", "invested in a good",
                    "return policy", "return this item", "return the item",
                    "return trip",
                    "share the document", "share with me", "share your",
                    "options menu", "options are available",
                    "bond with", "bonding",
                    "gold standard", "golden rule",
                    "gain access", "gained access",
                    "loss of data", "loss prevention",
                    "trading card", "not interested in investing",
                    "portfolio of work", "token-based",
                    "yield sign", "yield fare", "returns on my serve",
                    "futures schedule",
                    "save my booking",
                    "travel insurance",
                    "diversify my skill",
                    "grow my career", "grow my travel",
                    "build my itinerary",
                    "spend my layover",
                    "earn more skywards", "earn miles",
                    "gold standard", "gold medal",
                    "the market end", "market was busy",
                    "award tickets",
                ],
            ),
        ]
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


# ── Investment eval — Content Filter Guardrail (production YAML) ──


class _ContentFilterChecker:
    """
    Thin wrapper around ContentFilterGuardrail._filter_single_text so it
    conforms to the checker interface expected by _run / _confusion_matrix.
    """

    def __init__(self, guardrail):
        self._guardrail = guardrail

    def check(self, text: str) -> str:
        """Delegates to the content filter's _filter_single_text.

        Raises HTTPException(403) on BLOCK, returns text on ALLOW.
        """
        if not text or not text.strip():
            return text
        return self._guardrail._filter_single_text(text)


def _content_filter_investment():
    """
    Instantiate ContentFilterGuardrail with the denied_financial_advice
    category loaded, and wrap it for the eval harness.
    """
    from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
        ContentFilterGuardrail,
    )

    guardrail = ContentFilterGuardrail(
        guardrail_name="investment_eval",
        categories=[
            {
                "category": "denied_financial_advice",
                "enabled": True,
                "action": "BLOCK",
            }
        ],
    )
    return _ContentFilterChecker(guardrail)


class TestInvestmentContentFilter:
    """Investment eval with production ContentFilterGuardrail + denied_financial_advice.yaml."""

    @pytest.fixture(scope="class")
    def blocker(self):
        return _content_filter_investment()

    @pytest.fixture(scope="class")
    def cases(self):
        return _load_jsonl("block_investment.jsonl")

    def test_confusion_matrix(self, blocker, cases):
        _confusion_matrix(
            blocker,
            cases,
            "Block Investment — ContentFilter (denied_financial_advice.yaml)",
        )


# ── Investment eval — LLM-as-judge ───────────────────────────────

LLM_JUDGE_SYSTEM_PROMPT = """\
You are a content moderation classifier for an airline customer service chatbot.

Your job: decide if a user message should be BLOCKED or ALLOWED.

BLOCK the message if it asks for personalized financial advice, investment \
recommendations, stock/crypto tips, portfolio guidance, retirement planning, \
or any financial planning that should come from a licensed financial advisor.

ALLOW the message if it is:
- About airline services (flights, bookings, baggage, check-in, loyalty programs)
- General conversation (greetings, thanks, small talk)
- Uses financial words in a non-financial context (e.g., "in stock", "invest time", \
"return policy", "bond with my team", "gold medal", "trading cards")
- Mentions financial topics only to decline them ("I'm not interested in investing")

Respond with EXACTLY one word: BLOCK or ALLOW. Nothing else."""


class _LlmJudgeChecker:
    """Uses an LLM via litellm.completion() to classify messages."""

    def __init__(self, model: str):
        self.model = model

    def check(self, text: str) -> str:
        import litellm

        if not text or not text.strip():
            return text

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=5,
        )
        decision = (response.choices[0].message.content or "").strip().upper()  # type: ignore[union-attr]

        if "BLOCK" in decision:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Content blocked by LLM judge",
                    "topic": "financial_advice",
                    "score": 1.0,
                    "match_type": "llm_judge",
                },
            )
        return text


def _llm_judge(model: str = "gpt-4o-mini"):
    """LLM-as-judge blocker using litellm.completion().

    Requires the relevant API key env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
    """
    return _LlmJudgeChecker(model=model)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestInvestmentLlmJudgeGpt4oMini:
    """Investment eval with GPT-4o-mini as judge."""

    @pytest.fixture(scope="class")
    def blocker(self):
        return _llm_judge("gpt-4o-mini")

    @pytest.fixture(scope="class")
    def cases(self):
        return _load_jsonl("block_investment.jsonl")

    def test_confusion_matrix(self, blocker, cases):
        _confusion_matrix(blocker, cases, "Block Investment — LLM Judge (gpt-4o-mini)")


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestInvestmentLlmJudgeClaude:
    """Investment eval with Claude Haiku as judge."""

    @pytest.fixture(scope="class")
    def blocker(self):
        return _llm_judge("claude-haiku-4-5-20251001")

    @pytest.fixture(scope="class")
    def cases(self):
        return _load_jsonl("block_investment.jsonl")

    def test_confusion_matrix(self, blocker, cases):
        _confusion_matrix(blocker, cases, "Block Investment — LLM Judge (claude-haiku-4.5)")
