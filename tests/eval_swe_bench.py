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
    "Your response must contain ONLY the patch in unified diff format. "
    "Start with `diff --git a/path b/path`, then `---`, `+++`, and "
    "`@@` hunks. Do NOT include any explanation, commentary, or markdown "
    "fences — just the raw diff text."
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
    """Fallback: fetch rows directly from the HuggingFace dataset API (no deps).

    The API returns at most 100 rows per request, so we paginate.
    """
    import json
    import urllib.request

    # 0 means "all" — SWE-bench Lite has 300 test instances
    target = n if n > 0 else 300
    page_size = 100
    all_rows: list[dict] = []

    for offset in range(0, target, page_size):
        length = min(page_size, target - offset)
        url = (
            "https://datasets-server.huggingface.co/rows"
            "?dataset=princeton-nlp/SWE-bench_Lite_bm25_27K"
            f"&config=default&split={split}&offset={offset}&length={length}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "litellm-eval"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        rows = [row["row"] for row in data["rows"]]
        all_rows.extend(rows)
        if len(rows) < length:
            break  # no more data

    return all_rows


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
    """Extract modified file paths from a unified diff.

    Tries `diff --git a/path b/path` first, then falls back to
    `--- a/path` lines for diffs that omit the git header.
    """
    files = set(re.findall(r"^diff --git a/(.*?) b/", patch, re.MULTILINE))
    if not files:
        # Fallback: extract from --- a/path lines
        files = set(re.findall(r"^--- a/(.+)", patch, re.MULTILINE))
    return files


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


def _parse_hunk_line_ranges(patch: str) -> dict[str, list[tuple[int, int]]]:
    """Parse a unified diff into {filepath: [(start, end), ...]} for modified line ranges."""
    current_file = None
    ranges: dict[str, list[tuple[int, int]]] = {}
    for line in patch.split("\n"):
        m = re.match(r"^diff --git a/(.*?) b/", line)
        if m:
            current_file = m.group(1)
            if current_file not in ranges:
                ranges[current_file] = []
            continue
        if not current_file:
            m2 = re.match(r"^--- a/(.+)", line)
            if m2:
                current_file = m2.group(1)
                if current_file not in ranges:
                    ranges[current_file] = []
                continue
        m3 = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if m3 and current_file:
            start = int(m3.group(1))
            length = int(m3.group(2) or "1")
            ranges[current_file].append((start, start + length))
    return ranges


def _extract_changed_lines(patch: str) -> set[str]:
    """Extract the actual added/removed lines (stripped) from a diff."""
    lines = set()
    for line in patch.split("\n"):
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            stripped = line[1:].strip()
            if stripped:
                lines.add(stripped)
    return lines


def _line_range_overlap(
    ranges_a: dict[str, list[tuple[int, int]]],
    ranges_b: dict[str, list[tuple[int, int]]],
    tolerance: int = 10,
) -> float:
    """Compute fraction of gold hunk line ranges that overlap with generated ranges.

    Uses a tolerance window: a generated hunk counts as overlapping a gold hunk
    if their line ranges are within ``tolerance`` lines of each other.  This
    accounts for LLM-generated patches having slightly different line numbers
    than the gold patch (due to context window differences, reformatting, etc.)
    while still targeting the same logical code region.
    """
    shared_files = set(ranges_a.keys()) & set(ranges_b.keys())
    if not shared_files:
        return 0.0

    total_gold_hunks = 0
    overlapping_hunks = 0

    for f in shared_files:
        for g_start, g_end in ranges_a[f]:
            total_gold_hunks += 1
            for c_start, c_end in ranges_b[f]:
                # Ranges overlap (with tolerance) if they're within tolerance
                # lines of each other
                if (c_start - tolerance) <= g_end and (c_end + tolerance) >= g_start:
                    overlapping_hunks += 1
                    break  # count each gold hunk at most once

    if total_gold_hunks == 0:
        return 0.0
    return min(overlapping_hunks / total_gold_hunks, 1.0)


def proxy_eval(generated_text: str, instance: dict) -> dict:
    """
    Evaluate a generated patch without running the test suite.

    Returns:
        has_diff:            bool  — model produced a valid unified diff
        file_overlap:        float — fraction of gold files present in patch
        exact_file_match:    bool  — generated patch touches exactly the right files
        hunk_overlap:        float — fraction of gold line ranges covered by generated hunks
        content_similarity:  float — Jaccard similarity of changed lines (added/removed)
    """
    generated_patch = extract_patch(generated_text)
    gold_patch = instance["patch"]
    gold_files = parse_patch_files(gold_patch)
    generated_files = parse_patch_files(generated_patch)

    has_diff = is_valid_diff(generated_patch)

    file_overlap = (
        len(gold_files & generated_files) / len(gold_files) if gold_files else 0.0
    )
    exact_file_match = (gold_files == generated_files) and bool(gold_files)

    # Hunk-level: do they modify the same line ranges?
    gold_ranges = _parse_hunk_line_ranges(gold_patch)
    gen_ranges = _parse_hunk_line_ranges(generated_patch)
    hunk_overlap = _line_range_overlap(gold_ranges, gen_ranges)

    # Content-level: Jaccard similarity of the actual changed lines
    gold_lines = _extract_changed_lines(gold_patch)
    gen_lines = _extract_changed_lines(generated_patch)
    if gold_lines or gen_lines:
        content_similarity = len(gold_lines & gen_lines) / len(gold_lines | gen_lines)
    else:
        content_similarity = 0.0

    return {
        "has_diff": has_diff,
        "file_overlap": round(file_overlap, 3),
        "exact_file_match": exact_file_match,
        "hunk_overlap": round(hunk_overlap, 3),
        "content_similarity": round(content_similarity, 3),
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
    hunk_overlap: float
    content_similarity: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float = 0.0
    compression_ratio: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Single instance evaluation
# ---------------------------------------------------------------------------


def _run_with_retrieval_loop(
    model: str,
    messages: list[dict],
    tools: list[dict],
    cache: dict[str, str],
    max_retrievals: int = 5,
) -> tuple[str, object, float, float]:
    """
    Call the model, and if it invokes litellm_content_retrieve, fulfill
    the tool call from the cache and re-call until the model produces a
    final text response (or we hit max_retrievals).

    Returns (generated_text, final_usage, total_latency_ms, total_cost).
    """
    total_latency = 0.0
    total_cost = 0.0
    total_usage = None
    kwargs: dict = {
        "model": model,
        "messages": list(messages),
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    if tools:
        kwargs["tools"] = tools

    for _ in range(max_retrievals + 1):
        t0 = time.time()
        resp = litellm.completion(**kwargs)
        total_latency += (time.time() - t0) * 1000
        total_cost += resp._hidden_params.get("response_cost", 0) or 0
        total_usage = resp.usage

        choice = resp.choices[0]

        # If the model produced tool calls, fulfill them and loop
        tool_calls = getattr(choice.message, "tool_calls", None)
        if tool_calls:
            # Append the assistant message with tool calls
            kwargs["messages"].append(choice.message.model_dump())

            for tc in tool_calls:
                if tc.function.name == "litellm_content_retrieve":
                    import json as _json

                    args = _json.loads(tc.function.arguments)
                    key = args.get("key", "")
                    content = cache.get(key, f"[key {key!r} not found in cache]")
                    kwargs["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": content,
                        }
                    )
                else:
                    kwargs["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "[unknown tool]",
                        }
                    )
            continue

        # No tool calls — model produced a final text response
        return choice.message.content or "", total_usage, total_latency, total_cost

    # Exhausted retries — return whatever we have
    return resp.choices[0].message.content or "", total_usage, total_latency, total_cost


def eval_instance(
    instance: dict,
    model: str,
    use_compression: bool,
    compression_trigger: int,
    compression_target: Optional[int] = None,
    embedding_model: Optional[str] = None,
) -> SWERunResult:
    mode = "compressed" if use_compression else "baseline"
    messages = build_messages(instance)
    compression_ratio = 0.0
    tools: list[dict] = []
    cache: dict[str, str] = {}

    if use_compression:
        compress_kwargs: dict = {
            "messages": messages,
            "model": model,
            "input_type": "openai_chat_completions",
            "compression_trigger": compression_trigger,
            "embedding_model": embedding_model,
        }
        if compression_target is not None:
            compress_kwargs["compression_target"] = compression_target
        result = litellm_compress(**compress_kwargs)
        messages = result["messages"]
        tools = result["tools"]
        cache = result["cache"]
        compression_ratio = result["compression_ratio"]

    try:
        generated_text, usage, latency_ms, cost = _run_with_retrieval_loop(
            model=model,
            messages=messages,
            tools=tools,
            cache=cache,
        )
        ev = proxy_eval(generated_text, instance)

        return SWERunResult(
            instance_id=instance["instance_id"],
            mode=mode,
            has_diff=ev["has_diff"],
            file_overlap=ev["file_overlap"],
            exact_file_match=ev["exact_file_match"],
            hunk_overlap=ev["hunk_overlap"],
            content_similarity=ev["content_similarity"],
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            compression_ratio=compression_ratio,
        )
    except Exception as e:
        return SWERunResult(
            instance_id=instance["instance_id"],
            mode=mode,
            has_diff=False,
            file_overlap=0.0,
            exact_file_match=False,
            hunk_overlap=0.0,
            content_similarity=0.0,
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
        "avg_hunk_overlap": round(statistics.mean(r.hunk_overlap for r in results), 3),
        "avg_content_similarity": round(
            statistics.mean(r.content_similarity for r in results), 3
        ),
        "avg_prompt_tokens": round(statistics.mean(r.prompt_tokens for r in results)),
        "avg_total_tokens": round(statistics.mean(r.total_tokens for r in results)),
        "avg_latency_ms": round(statistics.mean(r.latency_ms for r in results), 1),
        "avg_compression_ratio": round(
            statistics.mean(r.compression_ratio for r in results), 4
        ),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 6),
        "avg_cost_usd": round(statistics.mean(r.cost_usd for r in results), 6),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    model: str,
    num_problems: int = 10,
    compression_trigger: int = 10_000,
    compression_target: Optional[int] = None,
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
    effective_target = (
        compression_target
        if compression_target is not None
        else compression_trigger * 7 // 10
    )
    print(f"Compression trigger: {compression_trigger} tokens")
    print(f"Compression target:  {effective_target} tokens")
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
            compression_target=compression_target,
        )
        baseline_results.append(r_base)
        if r_base.error:
            print(f"ERROR: {r_base.error[:80]}")
        else:
            print(
                f"{'✓' if r_base.has_diff else '✗'} diff  "
                f"file_overlap={r_base.file_overlap:.2f}  "
                f"{r_base.prompt_tokens} tok  "
                f"${r_base.cost_usd:.4f}"
            )

        print(f"  compressed ...", end=" ", flush=True)
        r_comp = eval_instance(
            instance,
            model,
            use_compression=True,
            compression_trigger=compression_trigger,
            compression_target=compression_target,
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
                f"${r_comp.cost_usd:.4f}  "
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
    print(f"    Avg hunk overlap:    {base_agg['avg_hunk_overlap']:.3f}")
    print(f"    Avg content sim:     {base_agg['avg_content_similarity']:.3f}")
    print(f"    Avg prompt tokens:   {base_agg['avg_prompt_tokens']}")
    print(f"    Avg latency:         {base_agg['avg_latency_ms']}ms")
    print(f"    Total cost:          ${base_agg['total_cost_usd']:.4f}")
    print(f"    Avg cost/problem:    ${base_agg['avg_cost_usd']:.6f}")

    print(f"\n  Compressed:")
    print(f"    Has-diff rate:       {comp_agg['has_diff_rate']}%")
    print(f"    Avg file overlap:    {comp_agg['avg_file_overlap']:.3f}")
    print(f"    Exact file match:    {comp_agg['exact_file_match_rate']}%")
    print(f"    Avg hunk overlap:    {comp_agg['avg_hunk_overlap']:.3f}")
    print(f"    Avg content sim:     {comp_agg['avg_content_similarity']:.3f}")
    print(f"    Avg prompt tokens:   {comp_agg['avg_prompt_tokens']}")
    print(f"    Avg latency:         {comp_agg['avg_latency_ms']}ms")
    print(f"    Total cost:          ${comp_agg['total_cost_usd']:.4f}")
    print(f"    Avg cost/problem:    ${comp_agg['avg_cost_usd']:.6f}")
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
    print(
        f"    Hunk overlap delta:  {comp_agg['avg_hunk_overlap'] - base_agg['avg_hunk_overlap']:+.3f}"
    )
    print(
        f"    Content sim delta:   {comp_agg['avg_content_similarity'] - base_agg['avg_content_similarity']:+.3f}"
    )
    cost_savings = base_agg["total_cost_usd"] - comp_agg["total_cost_usd"]
    cost_pct = (
        round(cost_savings / base_agg["total_cost_usd"] * 100, 1)
        if base_agg["total_cost_usd"]
        else 0
    )
    print(f"    Cost savings:        ${cost_savings:.4f} ({cost_pct}%)")

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
        "--compression-target",
        type=int,
        default=None,
        help="Target token count after compression (default: 70%% of trigger). "
        "Higher values preserve more context at the cost of less compression.",
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
        compression_target=args.compression_target,
        embedding_model=args.embedding_model,
    )
