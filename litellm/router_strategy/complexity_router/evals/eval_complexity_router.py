"""
Evaluation suite for the ComplexityRouter.

Tests the router's ability to correctly classify prompts into complexity tiers.
Run with: python -m litellm.router_strategy.complexity_router.evals.eval_complexity_router
"""
import os

# Add parent to path for imports
import sys

# ruff: noqa: T201
from dataclasses import dataclass
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.config import ComplexityTier


@dataclass
class EvalCase:
    """A single evaluation case."""
    prompt: str
    expected_tier: ComplexityTier
    description: str
    system_prompt: Optional[str] = None
    # Allow some flexibility - if actual tier is in acceptable_tiers, still passes
    acceptable_tiers: Optional[List[ComplexityTier]] = None


# ─── Evaluation Dataset ───

EVAL_CASES: List[EvalCase] = [
    # === SIMPLE tier cases ===
    EvalCase(
        prompt="Hello!",
        expected_tier=ComplexityTier.SIMPLE,
        description="Basic greeting",
    ),
    EvalCase(
        prompt="What is Python?",
        expected_tier=ComplexityTier.SIMPLE,
        description="Simple definition question",
    ),
    EvalCase(
        prompt="Who is Elon Musk?",
        expected_tier=ComplexityTier.SIMPLE,
        description="Simple factual question",
    ),
    EvalCase(
        prompt="What's the capital of France?",
        expected_tier=ComplexityTier.SIMPLE,
        description="Simple geography question",
    ),
    EvalCase(
        prompt="Thanks for your help!",
        expected_tier=ComplexityTier.SIMPLE,
        description="Simple thank you",
    ),
    EvalCase(
        prompt="Define machine learning",
        expected_tier=ComplexityTier.SIMPLE,
        description="Definition request",
    ),
    EvalCase(
        prompt="When was the iPhone released?",
        expected_tier=ComplexityTier.SIMPLE,
        description="Simple date question",
    ),
    EvalCase(
        prompt="How many planets are in our solar system?",
        expected_tier=ComplexityTier.SIMPLE,
        description="Simple count question",
    ),
    EvalCase(
        prompt="Yes",
        expected_tier=ComplexityTier.SIMPLE,
        description="Single word response",
    ),
    EvalCase(
        prompt="What time is it in Tokyo?",
        expected_tier=ComplexityTier.SIMPLE,
        description="Simple time zone question",
    ),
    
    # === MEDIUM tier cases ===
    EvalCase(
        prompt="Explain how REST APIs work and when to use them",
        expected_tier=ComplexityTier.MEDIUM,
        description="Technical explanation",
        acceptable_tiers=[ComplexityTier.SIMPLE, ComplexityTier.MEDIUM],
    ),
    EvalCase(
        prompt="Write a short poem about the ocean",
        expected_tier=ComplexityTier.MEDIUM,
        description="Creative writing - short",
        acceptable_tiers=[ComplexityTier.SIMPLE, ComplexityTier.MEDIUM],
    ),
    EvalCase(
        prompt="Summarize the main differences between SQL and NoSQL databases",
        expected_tier=ComplexityTier.MEDIUM,
        description="Technical comparison",
        acceptable_tiers=[ComplexityTier.MEDIUM, ComplexityTier.COMPLEX],
    ),
    EvalCase(
        prompt="What are the benefits of using TypeScript over JavaScript?",
        expected_tier=ComplexityTier.MEDIUM,
        description="Technical comparison question",
        acceptable_tiers=[ComplexityTier.SIMPLE, ComplexityTier.MEDIUM],
    ),
    EvalCase(
        prompt="Help me debug this error: TypeError: Cannot read property 'map' of undefined",
        expected_tier=ComplexityTier.MEDIUM,
        description="Debugging help",
        acceptable_tiers=[ComplexityTier.MEDIUM, ComplexityTier.COMPLEX],
    ),
    
    # === COMPLEX tier cases ===
    EvalCase(
        prompt="Design a distributed microservice architecture for a high-throughput "
               "real-time data processing pipeline with Kubernetes orchestration, "
               "implementing proper authentication and encryption protocols",
        expected_tier=ComplexityTier.COMPLEX,
        description="Complex architecture design",
        acceptable_tiers=[ComplexityTier.COMPLEX, ComplexityTier.REASONING],
    ),
    EvalCase(
        prompt="Write a Python function that implements a binary search tree with "
               "insert, delete, and search operations. Include proper error handling "
               "and optimize for memory efficiency.",
        expected_tier=ComplexityTier.COMPLEX,
        description="Complex coding task",
        acceptable_tiers=[ComplexityTier.MEDIUM, ComplexityTier.COMPLEX],
    ),
    EvalCase(
        prompt="Explain the differences between TCP and UDP protocols, including "
               "use cases for each, performance implications, and how they handle "
               "packet loss in distributed systems",
        expected_tier=ComplexityTier.COMPLEX,
        description="Deep technical explanation",
        acceptable_tiers=[ComplexityTier.MEDIUM, ComplexityTier.COMPLEX],
    ),
    EvalCase(
        prompt="Create a comprehensive database schema for an e-commerce platform "
               "that handles users, products, orders, payments, shipping, reviews, "
               "and inventory management with proper indexing strategies",
        expected_tier=ComplexityTier.COMPLEX,
        description="Complex database design",
        acceptable_tiers=[ComplexityTier.MEDIUM, ComplexityTier.COMPLEX, ComplexityTier.REASONING],
    ),
    EvalCase(
        prompt="Implement a rate limiter using the token bucket algorithm in Python "
               "that supports multiple rate limit tiers and can be used across "
               "distributed systems with Redis as the backend",
        expected_tier=ComplexityTier.COMPLEX,
        description="Complex distributed systems coding",
        acceptable_tiers=[ComplexityTier.MEDIUM, ComplexityTier.COMPLEX, ComplexityTier.REASONING],
    ),
    
    # === REASONING tier cases ===
    EvalCase(
        prompt="Think step by step about how to solve this: A farmer has 17 sheep. "
               "All but 9 die. How many are left? Explain your reasoning.",
        expected_tier=ComplexityTier.REASONING,
        description="Explicit reasoning request",
    ),
    EvalCase(
        prompt="Let's think through this carefully. Analyze the pros and cons of "
               "microservices vs monolithic architecture for a startup with 5 engineers. "
               "Consider scalability, development speed, and operational complexity.",
        expected_tier=ComplexityTier.REASONING,
        description="Multiple reasoning markers + analysis",
    ),
    EvalCase(
        prompt="Reason through this problem: If I have a function that's O(n^2) and "
               "I need to process 1 million items, what are my options to optimize it? "
               "Walk me through each approach step by step.",
        expected_tier=ComplexityTier.REASONING,
        description="Algorithm reasoning",
    ),
    EvalCase(
        prompt="I need you to think carefully and analyze this code for potential "
               "security vulnerabilities. Consider injection attacks, authentication "
               "bypasses, and data exposure risks. Show your reasoning process.",
        expected_tier=ComplexityTier.REASONING,
        description="Security analysis with reasoning",
        acceptable_tiers=[ComplexityTier.COMPLEX, ComplexityTier.REASONING],
    ),
    EvalCase(
        prompt="Step by step, explain your reasoning as you evaluate whether we should "
               "use PostgreSQL or MongoDB for our new project. Consider our requirements: "
               "complex queries, high write volume, and eventual consistency is acceptable.",
        expected_tier=ComplexityTier.REASONING,
        description="Database decision with explicit reasoning",
    ),
    
    # === Edge cases / regression tests ===
    EvalCase(
        prompt="What is the capital of France?",
        expected_tier=ComplexityTier.SIMPLE,
        description="Regression: 'capital' should not trigger 'api' keyword",
    ),
    EvalCase(
        prompt="I tried to book a flight but the entry form wasn't working",
        expected_tier=ComplexityTier.SIMPLE,
        description="Regression: 'tried' and 'entry' should not trigger code keywords",
        acceptable_tiers=[ComplexityTier.SIMPLE, ComplexityTier.MEDIUM],
    ),
    EvalCase(
        prompt="The poetry of digital art is fascinating",
        expected_tier=ComplexityTier.SIMPLE,
        description="Regression: 'poetry' should not trigger 'try' keyword",
        acceptable_tiers=[ComplexityTier.SIMPLE, ComplexityTier.MEDIUM],
    ),
    EvalCase(
        prompt="Can you recommend a good book about country music history?",
        expected_tier=ComplexityTier.SIMPLE,
        description="Regression: 'country' should not trigger 'try' keyword",
        acceptable_tiers=[ComplexityTier.SIMPLE, ComplexityTier.MEDIUM],
    ),
]


def run_eval() -> Tuple[int, int, List[dict]]:
    """
    Run the evaluation suite.
    
    Returns:
        Tuple of (passed, total, failures)
    """
    # Create router with default config
    mock_router = MagicMock()
    router = ComplexityRouter(
        model_name="eval-router",
        litellm_router_instance=mock_router,
    )
    
    passed = 0
    total = len(EVAL_CASES)
    failures = []
    
    print("=" * 70)  # noqa: T201
    print("COMPLEXITY ROUTER EVALUATION")  # noqa: T201
    print("=" * 70)  # noqa: T201
    print()  # noqa: T201
    
    for i, case in enumerate(EVAL_CASES, 1):
        tier, score, signals = router.classify(case.prompt, case.system_prompt)
        
        # Check if pass
        is_exact_match = tier == case.expected_tier
        is_acceptable = (
            case.acceptable_tiers is not None and 
            tier in case.acceptable_tiers
        )
        is_pass = is_exact_match or is_acceptable
        
        if is_pass:
            passed += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            failures.append({
                "case": i,
                "description": case.description,
                "prompt": case.prompt[:80] + "..." if len(case.prompt) > 80 else case.prompt,
                "expected": case.expected_tier.value,
                "actual": tier.value,
                "score": round(score, 3),
                "signals": signals,
                "acceptable": [t.value for t in case.acceptable_tiers] if case.acceptable_tiers else None,
            })
        
        # Print result
        print(f"[{i:2d}] {status} | {case.description}")  # noqa: T201
        print(f"     Expected: {case.expected_tier.value:10s} | Got: {tier.value:10s} | Score: {score:+.3f}")  # noqa: T201
        if signals:
            print(f"     Signals: {', '.join(signals)}")  # noqa: T201
        if not is_pass:
            print(f"     Prompt: {case.prompt[:60]}...")  # noqa: T201
        print()  # noqa: T201
    
    # Summary
    print("=" * 70)  # noqa: T201
    print(f"RESULTS: {passed}/{total} passed ({100*passed/total:.1f}%)")  # noqa: T201
    print("=" * 70)  # noqa: T201
    
    if failures:
        print("\nFAILURES:")  # noqa: T201
        print("-" * 70)  # noqa: T201
        for f in failures:
            print(f"Case {f['case']}: {f['description']}")  # noqa: T201
            print(f"  Expected: {f['expected']}, Got: {f['actual']} (score: {f['score']})")  # noqa: T201
            print(f"  Signals: {f['signals']}")  # noqa: T201
            if f['acceptable']:
                print(f"  Acceptable: {f['acceptable']}")  # noqa: T201
            print()  # noqa: T201
    
    return passed, total, failures


def main():
    """Main entry point."""
    passed, total, failures = run_eval()
    
    # Exit with error code if too many failures
    pass_rate = passed / total
    if pass_rate < 0.80:
        print(f"\n❌ EVAL FAILED: Pass rate {pass_rate:.1%} is below 80% threshold")  # noqa: T201
        sys.exit(1)
    elif pass_rate < 0.90:
        print(f"\n⚠️  EVAL WARNING: Pass rate {pass_rate:.1%} is below 90%")  # noqa: T201
        sys.exit(0)
    else:
        print(f"\n✅ EVAL PASSED: Pass rate {pass_rate:.1%}")  # noqa: T201
        sys.exit(0)


if __name__ == "__main__":
    main()
