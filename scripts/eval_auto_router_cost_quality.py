"""
Auto-router cost/quality bakeoff.

Reuses HumanEval-style PROBLEMS from eval_compression.py. For each problem,
runs several routing arms, stores raw completions + unit-test pass/fail +
GPT-5.5 judge scores, then prints a cost / quality / latency table.

Arms (default):
  1. premium baseline (intended: claude-opus-4-8; falls back if unavailable)
  2. complexity router (today)
  3. adaptive router (today)
  4. hybrid complexity + adaptive soft floors

Usage:
  set -a && source litellm/.env && set +a
  python scripts/eval_auto_router_cost_quality.py --problems 12 --out-dir eval_results/auto_router_bakeoff
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Allow `import litellm` + sibling script imports when run from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import litellm
from litellm import Router, completion_cost
from litellm.types.router import AdaptiveRouterPreferences, AdaptiveRouterWeights, RequestType

from scripts.eval_compression import PROBLEMS, SYSTEM_MSG, extract_code, run_tests

JUDGE_MODEL = "openai/gpt-5.5"
JUDGE_SYSTEM = (
    "You are grading a Python coding assistant. Score only whether the submitted "
    "code correctly implements the requested function. Ignore style and verbosity. "
    'Reply with JSON only: {"score": <1-5 integer>, "reason": "<one sentence>"}. '
    "5 = fully correct, 3 = partial, 1 = wrong or empty."
)


@dataclass
class ArmResult:
    arm: str
    problem_id: str
    passed: bool
    generated_code: str
    raw_response: str
    chosen_model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    classified_tier: Optional[str] = None
    error: str = ""
    judge_score: Optional[int] = None
    judge_reason: str = ""
    judge_raw: str = ""


@dataclass
class ArmSummary:
    arm: str
    n: int
    pass_rate: float
    avg_judge_score: float
    total_cost_usd: float
    avg_cost_usd: float
    avg_latency_ms: float
    latency_overhead_ms_vs_baseline: float
    chosen_models: dict[str, int] = field(default_factory=dict)


def _load_dotenv_files() -> None:
    for rel in ("litellm/.env", "litellm/proxy/.env"):
        path = _REPO_ROOT / rel
        if not path.exists():
            # Also try parent monorepo checkout paths used in this worktree setup.
            alt = Path("/Users/krrishdholakia/Documents/litellm") / rel
            path = alt if alt.exists() else path
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _normalize_tiers(tiers: dict[str, Any]) -> dict[str, list[str]]:
    return {k: (v if isinstance(v, list) else [v]) for k, v in tiers.items()}


def _model_list_for_bakeoff(
    *,
    cheap: str,
    mid: str,
    premium: str,
    include_complexity: bool,
    include_adaptive: bool,
    include_hybrid: bool,
) -> list[dict[str, Any]]:
    """Shared underlying deployments + optional router control planes."""
    underlying = [
        {
            "model_name": "cheap",
            "litellm_params": {
                "model": cheap,
                "input_cost_per_token": 0.00000015,
            },
            "model_info": {
                "adaptive_router_preferences": {
                    "quality_tier": 1,
                    "strengths": [],
                }
            },
        },
        {
            "model_name": "mid",
            "litellm_params": {
                "model": mid,
                "input_cost_per_token": 0.0000025,
            },
            "model_info": {
                "adaptive_router_preferences": {
                    "quality_tier": 2,
                    "strengths": [
                        RequestType.CODE_GENERATION,
                        RequestType.CODE_UNDERSTANDING,
                        RequestType.WRITING,
                    ],
                }
            },
        },
        {
            "model_name": "premium",
            "litellm_params": {
                "model": premium,
                "input_cost_per_token": 0.000005,
            },
            "model_info": {
                "adaptive_router_preferences": {
                    "quality_tier": 3,
                    "strengths": [
                        RequestType.CODE_GENERATION,
                        RequestType.ANALYTICAL_REASONING,
                        RequestType.TECHNICAL_DESIGN,
                    ],
                }
            },
        },
    ]

    tiers = {
        "SIMPLE": ["cheap"],
        "MEDIUM": ["mid", "cheap"],
        "COMPLEX": ["mid", "premium"],
        "REASONING": ["premium", "mid"],
    }

    routers: list[dict[str, Any]] = []
    if include_complexity:
        routers.append(
            {
                "model_name": "complexity-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_default_model": "mid",
                    "complexity_router_config": {
                        "adaptive": False,
                        "tiers": {k: v[0] for k, v in tiers.items()},
                        "default_model": "mid",
                    },
                },
            }
        )
    if include_adaptive:
        routers.append(
            {
                "model_name": "adaptive-router",
                "litellm_params": {
                    "model": "auto_router/adaptive_router",
                    "adaptive_router_default_model": "cheap",
                    "adaptive_router_config": {
                        "available_models": ["cheap", "mid", "premium"],
                        "weights": {"quality": 0.7, "cost": 0.3},
                    },
                },
            }
        )
    if include_hybrid:
        routers.append(
            {
                "model_name": "hybrid-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_default_model": "mid",
                    "complexity_router_config": {
                        "adaptive": True,
                        "adaptive_weights": {"quality": 0.7, "cost": 0.3},
                        "tiers": tiers,
                        "default_model": "mid",
                    },
                },
            }
        )
    return underlying + routers


def _build_messages(problem: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {
            "role": "user",
            "content": ("Complete the following Python function. Return ONLY the code.\n\n" + problem["prompt"]),
        },
    ]


def _safe_cost(response: Any) -> float:
    try:
        return float(completion_cost(completion_response=response) or 0.0)
    except Exception:
        return 0.0


def _chosen_from_response(response: Any, fallback: str) -> str:
    hidden = getattr(response, "_hidden_params", None) or {}
    if isinstance(hidden, dict):
        additional = hidden.get("additional_headers") or {}
        if isinstance(additional, dict):
            header = additional.get("x-litellm-adaptive-router-model")
            if header:
                return str(header)
        model_id = hidden.get("model_id") or hidden.get("litellm_model")
        if model_id:
            return str(model_id)
    model = getattr(response, "model", None)
    return str(model or fallback)


def _code_with_prompt_imports(problem: dict[str, Any], code: str) -> str:
    """HumanEval stubs often need typing imports that models omit from completions."""
    import_lines = [
        line for line in problem["prompt"].splitlines() if line.startswith("from ") or line.startswith("import ")
    ]
    if not import_lines:
        return code
    prelude = "\n".join(import_lines) + "\n\n"
    if any(line in code for line in import_lines):
        return code
    return prelude + code


def _logical_model_name(router: Optional[Router], chosen: str) -> str:
    if router is None or not chosen:
        return chosen
    idx = getattr(router, "model_id_to_deployment_index_map", {}).get(chosen)
    if idx is None:
        return chosen
    model_list = getattr(router, "model_list", None) or []
    if idx >= len(model_list):
        return chosen
    deployment = model_list[idx]
    if isinstance(deployment, dict):
        return str(deployment.get("model_name") or chosen)
    return str(getattr(deployment, "model_name", None) or chosen)
    hidden = getattr(response, "_hidden_params", None) or {}
    if isinstance(hidden, dict):
        for key in ("model", "custom_llm_provider"):
            pass
        additional = hidden.get("additional_headers") or {}
        if isinstance(additional, dict):
            header = additional.get("x-litellm-adaptive-router-model")
            if header:
                return str(header)
        model_id = hidden.get("model_id") or hidden.get("litellm_model")
        if model_id:
            return str(model_id)
    model = getattr(response, "model", None)
    return str(model or fallback)


def call_arm(
    *,
    arm: str,
    router: Optional[Router],
    model_name: str,
    problem: dict[str, Any],
    premium_direct: Optional[str] = None,
) -> ArmResult:
    return asyncio.run(
        acall_arm(
            arm=arm,
            router=router,
            model_name=model_name,
            problem=problem,
            premium_direct=premium_direct,
        )
    )


async def acall_arm(
    *,
    arm: str,
    router: Optional[Router],
    model_name: str,
    problem: dict[str, Any],
    premium_direct: Optional[str] = None,
) -> ArmResult:
    messages = _build_messages(problem)
    t0 = time.time()
    try:
        if arm == "premium_baseline":
            assert premium_direct is not None
            response = await litellm.acompletion(
                model=premium_direct,
                messages=messages,
                max_tokens=2048,
            )
            chosen = premium_direct
            classified = None
        else:
            assert router is not None
            # Must use async path: sync router.completion skips pre-routing hooks.
            response = await router.acompletion(
                model=model_name,
                messages=messages,
                max_tokens=2048,
            )
            chosen = _chosen_from_response(response, fallback=model_name)
            classified = None
            if model_name in getattr(router, "complexity_routers", {}):
                cr = router.complexity_routers[model_name]
                user_text = messages[-1]["content"]
                tier, _, _ = cr.classify(user_text)
                classified = tier.value
                if chosen in {"complexity-router", "hybrid-router", model_name}:
                    chosen = getattr(response, "model", None) or chosen

        latency_ms = (time.time() - t0) * 1000
        raw = response.choices[0].message.content or ""
        code = _code_with_prompt_imports(problem, extract_code(raw))
        passed, error = run_tests(code, problem["tests"])
        usage = response.usage
        if arm != "premium_baseline":
            chosen = _logical_model_name(router, str(chosen))
        return ArmResult(
            arm=arm,
            problem_id=problem["id"],
            passed=passed,
            generated_code=code,
            raw_response=raw,
            chosen_model=str(chosen),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            cost_usd=_safe_cost(response),
            latency_ms=latency_ms,
            classified_tier=classified,
            error=error,
        )
    except Exception as e:
        return ArmResult(
            arm=arm,
            problem_id=problem["id"],
            passed=False,
            generated_code="",
            raw_response="",
            chosen_model="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            latency_ms=(time.time() - t0) * 1000,
            error=str(e)[:800],
        )


def judge_result(problem: dict[str, Any], result: ArmResult) -> ArmResult:
    if result.error and not result.raw_response:
        result.judge_score = 1
        result.judge_reason = f"call failed: {result.error[:120]}"
        return result
    payload = {
        "role": "user",
        "content": (
            f"Function prompt:\n{problem['prompt']}\n\n"
            f"Submitted code:\n{result.generated_code or result.raw_response}\n\n"
            f"Unit tests (for your reference; do not execute):\n{problem['tests']}"
        ),
    }
    try:
        response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                payload,
            ],
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        result.judge_raw = raw
        parsed = json.loads(raw)
        result.judge_score = int(parsed.get("score", 1))
        result.judge_reason = str(parsed.get("reason", ""))[:300]
        # Attribute judge cost onto a side field? Keep separate; bakeoff table is arm cost only.
        result.cost_usd = float(result.cost_usd)  # unchanged
    except Exception as e:
        result.judge_score = None
        result.judge_reason = f"judge failed: {str(e)[:200]}"
        result.judge_raw = ""
    return result


def summarize(arm: str, results: list[ArmResult], baseline_avg_latency: float) -> ArmSummary:
    n = len(results)
    passed = sum(1 for r in results if r.passed)
    judge_scores = [r.judge_score for r in results if isinstance(r.judge_score, int)]
    costs = [r.cost_usd for r in results]
    latencies = [r.latency_ms for r in results]
    chosen: dict[str, int] = {}
    for r in results:
        key = r.chosen_model or "unknown"
        chosen[key] = chosen.get(key, 0) + 1
    avg_lat = statistics.mean(latencies) if latencies else 0.0
    return ArmSummary(
        arm=arm,
        n=n,
        pass_rate=round(100.0 * passed / n, 1) if n else 0.0,
        avg_judge_score=round(statistics.mean(judge_scores), 2) if judge_scores else 0.0,
        total_cost_usd=round(sum(costs), 6),
        avg_cost_usd=round(statistics.mean(costs), 6) if costs else 0.0,
        avg_latency_ms=round(avg_lat, 1),
        latency_overhead_ms_vs_baseline=round(avg_lat - baseline_avg_latency, 1),
        chosen_models=chosen,
    )


def _probe_premium_model(preferred: str, fallbacks: list[str]) -> str:
    for model in [preferred, *fallbacks]:
        try:
            litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Say ok"}],
                max_tokens=8,
            )
            return model
        except Exception as e:
            print(f"premium probe failed for {model}: {str(e)[:140]}")
    raise RuntimeError("No premium baseline model available")


def main() -> None:
    asyncio.run(amain())


async def amain() -> None:
    parser = argparse.ArgumentParser(description="Auto-router cost/quality bakeoff")
    parser.add_argument("--problems", type=int, default=0, help="0 = all")
    parser.add_argument("--out-dir", type=str, default="eval_results/auto_router_bakeoff")
    parser.add_argument("--premium-model", type=str, default="anthropic/claude-opus-4-8")
    parser.add_argument("--cheap-model", type=str, default="openai/gpt-4o-mini")
    parser.add_argument("--mid-model", type=str, default="openai/gpt-4o")
    parser.add_argument(
        "--premium-fallback",
        type=str,
        default="openai/gpt-5.5",
        help="Used when --premium-model is unavailable (e.g. Anthropic out of credits)",
    )
    parser.add_argument("--skip-hybrid", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    _load_dotenv_files()
    litellm.drop_params = True

    problems = PROBLEMS[: args.problems] if args.problems > 0 else PROBLEMS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    premium = await asyncio.to_thread(_probe_premium_model, args.premium_model, [args.premium_fallback, args.mid_model])
    if premium != args.premium_model:
        print(f"NOTE: premium baseline falling back to {premium} (requested {args.premium_model} unavailable)")

    include_hybrid = not args.skip_hybrid
    model_list = _model_list_for_bakeoff(
        cheap=args.cheap_model,
        mid=args.mid_model,
        premium=premium,
        include_complexity=True,
        include_adaptive=True,
        include_hybrid=include_hybrid,
    )
    router = Router(model_list=model_list)

    arms = [
        ("premium_baseline", None, premium),
        ("complexity", "complexity-router", None),
        ("adaptive", "adaptive-router", None),
    ]
    if include_hybrid:
        arms.append(("hybrid", "hybrid-router", None))

    all_results: dict[str, list[ArmResult]] = {name: [] for name, _, _ in arms}

    for problem in problems:
        print(f"\n=== {problem['id']} ===")
        for arm_name, router_model, direct in arms:
            print(f"  {arm_name} ...", end=" ", flush=True)
            result = await acall_arm(
                arm=arm_name,
                router=None if arm_name == "premium_baseline" else router,
                model_name=router_model or premium,
                problem=problem,
                premium_direct=direct,
            )
            if not args.skip_judge:
                result = await asyncio.to_thread(judge_result, problem, result)
            all_results[arm_name].append(result)
            status = "PASS" if result.passed else f"FAIL:{result.error[:40]}"
            judge = result.judge_score if result.judge_score is not None else "-"
            print(
                f"{status} judge={judge} cost=${result.cost_usd:.6f} "
                f"lat={result.latency_ms:.0f}ms model={result.chosen_model}"
            )
            (out_dir / f"{ts}_{arm_name}_{problem['id']}.json").write_text(json.dumps(asdict(result), indent=2))

    baseline_avg_latency = (
        statistics.mean(r.latency_ms for r in all_results["premium_baseline"])
        if all_results["premium_baseline"]
        else 0.0
    )
    summaries = [summarize(arm, all_results[arm], baseline_avg_latency) for arm, _, _ in arms]

    report = {
        "timestamp": ts,
        "premium_model": premium,
        "requested_premium_model": args.premium_model,
        "cheap_model": args.cheap_model,
        "mid_model": args.mid_model,
        "judge_model": JUDGE_MODEL,
        "num_problems": len(problems),
        "summaries": [asdict(s) for s in summaries],
        "results": {arm: [asdict(r) for r in rows] for arm, rows in all_results.items()},
    }
    report_path = out_dir / f"{ts}_summary.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 88)
    print(f"{'arm':18} {'pass%':>7} {'judge':>6} {'cost$':>10} {'avg_lat':>10} {'overhead':>10}")
    print("-" * 88)
    for s in summaries:
        print(
            f"{s.arm:18} {s.pass_rate:7.1f} {s.avg_judge_score:6.2f} "
            f"{s.total_cost_usd:10.6f} {s.avg_latency_ms:10.1f} "
            f"{s.latency_overhead_ms_vs_baseline:10.1f}"
        )
    print("=" * 88)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
