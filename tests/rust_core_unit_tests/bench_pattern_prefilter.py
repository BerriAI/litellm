"""
Benchmark: Rust RegexSet pre-filter vs serial Python regex loop, for the
common "clean text, nothing matches" case in ContentFilterGuardrail.

Run with:
    python tests/rust_core_unit_tests/bench_pattern_prefilter.py

Requires the Rust extension built in release mode:
    cd rust && maturin develop --uv --release

ContentFilterGuardrail's regex-pattern loop (content_filter.py,
_filter_single_text) checks every compiled pattern against every message on
every guardrail-enabled request, unconditionally. It never short-circuits,
because it must collect every match, not just the first one. Most production
traffic contains no PII at all, so on the overwhelmingly common path every
pattern is checked and every one fails to match.

  python path   loop over each eligible pattern, regex.search(text) per pattern

  rust path     RegexSet.is_match(text): one combined automaton run that
                 tests all eligible patterns simultaneously
"""

import json
import os
import re
import timeit

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.pattern_prefilter import (
    AlwaysMatchPrefilter,
    build_rust_pattern_prefilter,
)

_PATTERNS_JSON = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "litellm",
    "proxy",
    "guardrails",
    "guardrail_hooks",
    "litellm_content_filter",
    "patterns.json",
)


def _load_simple_pattern_sources() -> list[str]:
    """Mirrors ContentFilterGuardrail's partition: patterns with no
    contextual keyword-proximity config are eligible for the fast path."""
    with open(_PATTERNS_JSON) as f:
        data = json.load(f)
    known_keys = {
        "name",
        "display_name",
        "pattern",
        "category",
        "action",
        "description",
    }
    sources = []
    for entry in data["patterns"]:
        extra = {k: v for k, v in entry.items() if k not in known_keys}
        if extra.get("keyword_pattern") or extra.get("allow_word_numbers"):
            continue
        sources.append(entry["pattern"])
    return sources


SIMPLE_PATTERN_SOURCES = _load_simple_pattern_sources()
COMPILED_PYTHON_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in SIMPLE_PATTERN_SOURCES
]
PREFILTER, REJECTED = build_rust_pattern_prefilter(SIMPLE_PATTERN_SOURCES)
if isinstance(PREFILTER, AlwaysMatchPrefilter):
    raise SystemExit(
        "litellm_core not found. Build it first:\n"
        "  cd rust && maturin develop --uv --release"
    )
ELIGIBLE_PYTHON_PATTERNS = [
    p for i, p in enumerate(COMPILED_PYTHON_PATTERNS) if i not in REJECTED
]


def _python_loop(text: str) -> bool:
    return any(p.search(text) for p in ELIGIBLE_PYTHON_PATTERNS)


def _rust_prefilter(text: str) -> bool:
    return PREFILTER.any_match(text)


CLEAN_TEXTS = {
    "short  (~10 words)": "Can you help me write a short poem about the ocean at sunset?",
    "medium (~80 words)": (
        "I've been working on a quarterly report for our engineering team and "
        "I'm trying to summarize the key infrastructure improvements we made "
        "this quarter. We moved several services onto a faster build "
        "pipeline, improved our test coverage across the backend, and reduced "
        "average request latency by optimizing a few slow database queries. "
        "Could you help me turn these notes into a few clear paragraphs "
        "suitable for a leadership update, keeping the tone professional "
        "but not overly formal?"
    ),
    "long   (~400 words)": (
        "Can you help me draft a detailed design document for a new internal "
        "tool? "
        + " ".join(
            [
                "The tool should let engineers browse service dependencies, "
                "review recent releases, and check on-call rotations without "
                "needing to jump between five different dashboards."
            ]
            * 12
        )
    ),
}

WARMUP = 200
REPEATS = 5


def _measure_us(fn, arg) -> float:
    fn(arg)
    times = timeit.repeat(lambda: fn(arg), number=WARMUP, repeat=REPEATS)
    return min(times) / WARMUP * 1_000_000


def main() -> None:
    print()
    print(
        f"{len(SIMPLE_PATTERN_SOURCES)} simple patterns total, "
        f"{len(REJECTED)} rejected by Rust (contextual logic stays in Python), "
        f"{len(ELIGIBLE_PYTHON_PATTERNS)} eligible for the fast path."
    )
    print("All text below is clean (no PII): the dominant case in production traffic,")
    print(
        "and the only one where ContentFilterGuardrail's loop runs to completion today."
    )
    print()

    col = max(len(label) for label in CLEAN_TEXTS) + 2
    print(
        f"{'Scenario':<{col}}  {'python: serial loop':>20}  "
        f"{'rust: RegexSet':>20}  {'speedup':>8}"
    )
    print("-" * (col + 56))

    for label, text in CLEAN_TEXTS.items():
        assert _python_loop(text) is False
        assert _rust_prefilter(text) is False
        py_us = _measure_us(_python_loop, text)
        rs_us = _measure_us(_rust_prefilter, text)
        speedup = py_us / rs_us
        print(
            f"{label:<{col}}"
            f"  {py_us:>19.1f}µs"
            f"  {rs_us:>19.1f}µs"
            f"  {speedup:>7.1f}x"
        )

    print()
    print("rust: one combined NFA pass tests all eligible patterns simultaneously.")
    print("python: each pattern runs its own re.search call over the same text.")
    print()


if __name__ == "__main__":
    main()
