"""
SWE-bench Compression Evaluation
==================================
Measures litellm.compress() impact on SWE-bench Lite problems.

Each instance includes ~27k tokens of BM25-retrieved repo context — large
enough to meaningfully stress compression without requiring Docker or GitHub
API calls.

Usage:
    python tests/eval_swe_bench.py --model gpt-4o --problems 10
    python tests/eval_swe_bench.py --model claude-sonnet-4-20250514 --problems 25
    python tests/eval_swe_bench.py --model gpt-4o-mini --problems 50 --compression-trigger 8000

Requires:
    pip install datasets

Proxy eval metrics (no Docker / test runner required):
  - has_diff:          model produced a valid unified diff
  - file_overlap:      fraction of gold-patch files present in generated patch
  - exact_file_match:  generated patch touches exactly the same files as gold patch

Full SWE-bench pass rate (FAIL_TO_PASS) requires the official evaluation
harness with Docker — not in scope here. The proxy metrics are a lightweight
signal for whether compression degrades patch quality.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import litellm  # noqa: E402
from litellm.compression import compress as litellm_compress  # noqa: E402

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "You are an expert software engineer resolving GitHub issues. "
    "You will be given an issue description and relevant source files. "
    "Produce a minimal unified diff patch that fixes the issue. "
    "Output ONLY the patch starting with `diff --git`, no explanation."
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_via_datasets(n: int, split: str) -> list[dict]:
    """Load via the HuggingFace `datasets` library (preferred if available)."""
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Lite_bm25_27K", split=split)
    problems = []
    for i, item in enumerate(ds):
        if n > 0 and i >= n:
            break
        problems.append(dict(item))
    return problems


def _load_via_api(n: int, split: str) -> list[dict]:
    """Fallback: fetch rows directly from the HuggingFace dataset API (no deps)."""
    import json
    import urllib.request

    url = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=princeton-nlp/SWE-bench_Lite_bm25_27K"
        f"&config=default&split={split}&offset=0&length={n}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "litellm-eval"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return [row["row"] for row in data["rows"]]


def load_problems(n: int = 10, split: str = "test") -> list[dict]:
    """Load n problems from princeton-nlp/SWE-bench_Lite_bm25_27K."""
    print("Loading SWE-bench_Lite_bm25_27K ...", flush=True)

    # Try the HuggingFace API first — it's pure HTTP with no native deps,
    # so it never triggers pyarrow/numpy binary incompatibilities that can
    # poison the process.  Fall back to the `datasets` library only if the
    # API call fails.
    try:
        problems = _load_via_api(n, split)
    except Exception:
        try:
            problems = _load_via_datasets(n, split)
        except Exception as e:
            print(f"ERROR: Could not load dataset ({type(e).__name__}: {e})")
            sys.exit(1)

    print(f"Loaded {len(problems)} problems.\n")
    return problems


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def build_messages(instance: dict) -> list[dict]:
    """
    Build the message list for a SWE-bench instance.

    Structure:
      - system: instruction to produce a patch
      - user: problem statement + hints (the issue)
      - user: retrieved repo context (~27k tokens, the thing we compress)
      - user: final instruction
    """
    issue = instance["problem_statement"]
    hints = instance.get("hints_text", "").strip()
    context = instance["text"]  # BM25-retrieved file contents

    issue_content = f"## GitHub Issue\n\n{issue}"
    if hints:
        issue_content += f"\n\n## Hints\n\n{hints}"

    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": issue_content},
        {
            "role": "user",
            "content": f"## Relevant source files\n\n{context}",
        },
        {
            "role": "user",
            "content": (
                "Based on the issue and source files above, produce a minimal "
                "unified diff patch. Output only the patch."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------


def parse_patch_files(patch: str) -> set[str]:
    """Extract modified file paths from a unified diff."""
    return set(re.findall(r"^diff --git a/(.*) b/", patch, re.MULTILINE))


def extract_patch(text: str) -> str:
    """Pull the diff out of an LLM response."""
    # Prefer fenced code block
    m = re.search(r"```(?:diff|patch)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fall back to first `diff --git` line
    idx = text.find("diff --git")
    if idx != -1:
        return text[idx:].strip()
    return text.strip()


def is_valid_diff(patch: str) -> bool:
    return bool(
        re.search(r"^@@.*@@", patch, re.MULTILINE) and "---" in patch and "+++" in patch
    )


# ---------------------------------------------------------------------------
# Proxy evaluation
# ---------------------------------------------------------------------------


def proxy_eval(generated_text: str, instance: dict) -> dict:
    """
    Evaluate a generated patch without running the test suite.

    Returns:
        has_diff:          bool — model produced a valid unified diff
        file_overlap:      float — fraction of gold files present in patch
        exact_file_match:  bool — generated patch touches exactly the right files
        gold_files:        list[str]
        generated_files:   list[str]
    """
    generated_patch = extract_patch(generated_text)
    gold_files = parse_patch_files(instance["patch"])
    generated_files = parse_patch_files(generated_patch)

    has_diff = is_valid_diff(generated_patch)

    file_overlap = (
        len(gold_files & generated_files) / len(gold_files) if gold_files else 0.0
    )
    exact_file_match = (gold_files == generated_files) and bool(gold_files)

    return {
        "has_diff": has_diff,
        "file_overlap": round(file_overlap, 3),
        "exact_file_match": exact_file_match,
        "gold_files": sorted(gold_files),
        "generated_files": sorted(generated_files),
    }


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SWERunResult:
    instance_id: str
    mode: str  # "baseline" or "compressed"
    has_diff: bool
    file_overlap: float
    exact_file_match: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    compression_ratio: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Single instance evaluation
# ---------------------------------------------------------------------------


def eval_instance(
    instance: dict,
    model: str,
    use_compression: bool,
    compression_trigger: int,
    embedding_model: Optional[str] = None,
) -> SWERunResult:
    mode = "compressed" if use_compression else "baseline"
    messages = build_messages(instance)
    compression_ratio = 0.0

    if use_compression:
        result = litellm_compress(
            messages=messages,
            model=model,
            compression_trigger=compression_trigger,
            embedding_model=embedding_model,
        )
        messages = result["messages"]
        compression_ratio = result["compression_ratio"]

    try:
        t0 = time.time()
        resp = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=4096,
        )
        latency_ms = (time.time() - t0) * 1000

        generated_text = resp.choices[0].message.content or ""
        usage = resp.usage
        ev = proxy_eval(generated_text, instance)

        return SWERunResult(
            instance_id=instance["instance_id"],
            mode=mode,
            has_diff=ev["has_diff"],
            file_overlap=ev["file_overlap"],
            exact_file_match=ev["exact_file_match"],
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
            compression_ratio=compression_ratio,
        )
    except Exception as e:
        return SWERunResult(
            instance_id=instance["instance_id"],
            mode=mode,
            has_diff=False,
            file_overlap=0.0,
            exact_file_match=False,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=0.0,
            compression_ratio=0.0,
            error=str(e)[:500],
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(results: list[SWERunResult]) -> dict:
    if not results:
        return {}
    valid = [r for r in results if not r.error]
    errors = len(results) - len(valid)
    return {
        "total": len(results),
        "errors": errors,
        "has_diff_rate": round(
            sum(r.has_diff for r in results) / len(results) * 100, 1
        ),
        "avg_file_overlap": round(statistics.mean(r.file_overlap for r in results), 3),
        "exact_file_match_rate": round(
            sum(r.exact_file_match for r in results) / len(results) * 100, 1
        ),
        "avg_prompt_tokens": round(statistics.mean(r.prompt_tokens for r in results)),
        "avg_total_tokens": round(statistics.mean(r.total_tokens for r in results)),
        "avg_latency_ms": round(statistics.mean(r.latency_ms for r in results), 1),
        "avg_compression_ratio": round(
            statistics.mean(r.compression_ratio for r in results), 4
        ),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    model: str,
    num_problems: int = 10,
    compression_trigger: int = 10_000,
    embedding_model: Optional[str] = None,
) -> dict:
    """
    Run baseline vs compressed evaluation on SWE-bench Lite problems.

    Parameters:
        model:               LLM model name (litellm format).
        num_problems:        How many SWE-bench Lite problems to run.
        compression_trigger: Token count above which compression activates.
                             The bm25_27K dataset has ~27k tokens of context
                             per problem, so a trigger of 10k–20k is sensible.
        embedding_model:     Optional embedding model for semantic scoring.
    """
    problems = load_problems(n=num_problems)

    print(f"{'=' * 60}")
    print("SWE-bench Compression Eval")
    print(f"{'=' * 60}")
    print(f"Model:               {model}")
    print(f"Problems:            {len(problems)}")
    print(f"Compression trigger: {compression_trigger} tokens")
    print(f"Embedding model:     {embedding_model or 'None (BM25 only)'}")
    print(f"{'=' * 60}\n")

    baseline_results: list[SWERunResult] = []
    compressed_results: list[SWERunResult] = []

    for i, instance in enumerate(problems):
        iid = instance["instance_id"]

        print(f"[{i+1}/{len(problems)}] {iid}")

        print(f"  baseline   ...", end=" ", flush=True)
        r_base = eval_instance(
            instance,
            model,
            use_compression=False,
            compression_trigger=compression_trigger,
        )
        baseline_results.append(r_base)
        if r_base.error:
            print(f"ERROR: {r_base.error[:80]}")
        else:
            print(
                f"{'✓' if r_base.has_diff else '✗'} diff  "
                f"file_overlap={r_base.file_overlap:.2f}  "
                f"{r_base.prompt_tokens} tok"
            )

        print(f"  compressed ...", end=" ", flush=True)
        r_comp = eval_instance(
            instance,
            model,
            use_compression=True,
            compression_trigger=compression_trigger,
            embedding_model=embedding_model,
        )
        compressed_results.append(r_comp)
        if r_comp.error:
            print(f"ERROR: {r_comp.error[:80]}")
        else:
            print(
                f"{'✓' if r_comp.has_diff else '✗'} diff  "
                f"file_overlap={r_comp.file_overlap:.2f}  "
                f"{r_comp.prompt_tokens} tok  "
                f"(ratio: {r_comp.compression_ratio:.2%})"
            )

    base_agg = aggregate(baseline_results)
    comp_agg = aggregate(compressed_results)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  Baseline:")
    print(f"    Has-diff rate:       {base_agg['has_diff_rate']}%")
    print(f"    Avg file overlap:    {base_agg['avg_file_overlap']:.3f}")
    print(f"    Exact file match:    {base_agg['exact_file_match_rate']}%")
    print(f"    Avg prompt tokens:   {base_agg['avg_prompt_tokens']}")
    print(f"    Avg latency:         {base_agg['avg_latency_ms']}ms")

    print(f"\n  Compressed:")
    print(f"    Has-diff rate:       {comp_agg['has_diff_rate']}%")
    print(f"    Avg file overlap:    {comp_agg['avg_file_overlap']:.3f}")
    print(f"    Exact file match:    {comp_agg['exact_file_match_rate']}%")
    print(f"    Avg prompt tokens:   {comp_agg['avg_prompt_tokens']}")
    print(f"    Avg latency:         {comp_agg['avg_latency_ms']}ms")
    print(f"    Avg compression:     {comp_agg['avg_compression_ratio']:.2%}")

    token_savings = base_agg["avg_prompt_tokens"] - comp_agg["avg_prompt_tokens"]
    token_pct = (
        round(token_savings / base_agg["avg_prompt_tokens"] * 100, 1)
        if base_agg["avg_prompt_tokens"]
        else 0
    )
    print(f"\n  Delta (compressed vs baseline):")
    print(f"    Token savings:       {token_savings} ({token_pct}%)")
    print(
        f"    Latency delta:       {base_agg['avg_latency_ms'] - comp_agg['avg_latency_ms']:+.1f}ms"
    )
    print(
        f"    Has-diff delta:      {comp_agg['has_diff_rate'] - base_agg['has_diff_rate']:+.1f}%"
    )
    print(
        f"    File overlap delta:  {comp_agg['avg_file_overlap'] - base_agg['avg_file_overlap']:+.3f}"
    )
    print(
        f"    Exact match delta:   {comp_agg['exact_file_match_rate'] - base_agg['exact_file_match_rate']:+.1f}%"
    )

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    report_path = f"eval_swe_bench_report_{ts}.json"
    report = {
        "model": model,
        "timestamp": ts,
        "num_problems": len(problems),
        "compression_trigger": compression_trigger,
        "embedding_model": embedding_model,
        "baseline": base_agg,
        "compressed": comp_agg,
        "baseline_results": [asdict(r) for r in baseline_results],
        "compressed_results": [asdict(r) for r in compressed_results],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWE-bench Compression Evaluation")
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="Model name (litellm format)"
    )
    parser.add_argument(
        "--problems",
        type=int,
        default=10,
        help="Number of SWE-bench Lite problems to run (default: 10)",
    )
    parser.add_argument(
        "--compression-trigger",
        type=int,
        default=10_000,
        help="Token threshold to activate compression (default: 10000). "
        "The bm25_27K dataset has ~27k tokens of context per problem.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model for semantic scoring (e.g. text-embedding-3-small)",
    )
    args = parser.parse_args()

    run_benchmark(
        model=args.model,
        num_problems=args.problems,
        compression_trigger=args.compression_trigger,
        embedding_model=args.embedding_model,
    )
