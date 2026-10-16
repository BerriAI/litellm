"""
Hybrid Router v1.0 Evaluation.

Thompson Sampling with session-based judge feedback. Runs prompts in sessions
of BATCH_SIZE. After each session, judges all responses, then feeds composite
(efficiency + quality) rewards into the Thompson Sampling bandit. This lets
the bandit learn per-tier model preferences that balance speed and correctness.

Assumes:
  - Qwen3.5-9B served at http://localhost:8001/v1
  - Qwen3.5-4B served at http://localhost:8002/v1
  - CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-9B --port 8001 --served-model-name Qwen3.5-9b --gpu-memory-utilization 0.90
  - CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3.5-4B --port 8002 --served-model-name Qwen3.5-4b --gpu-memory-utilization 0.90
Usage:
    python scripts/test_custom_router_euro_v1.py

Environment variables:
    VLLM_9B_BASE          - 9B model endpoint (default: http://localhost:8001/v1)
    VLLM_4B_BASE          - 4B model endpoint (default: http://localhost:8002/v1)
    HYBRID_EVAL_CONCURRENCY - Concurrent inference requests (default: 16)
    JUDGE_CONCURRENCY     - Concurrent judge requests (default: 32)
    JUDGE_TIMEOUT         - Judge request timeout in seconds (default: 15)
    JUDGE_SAMPLE_SIZE     - Number of prompts to judge (default: 100)
    BATCH_SIZE            - Prompts per session (default: 200)
    QUALITY_WEIGHT        - Quality vs efficiency weight, 0-1 (default: 0.5)
    SKIP_JUDGE            - Set to "1" to skip accuracy evaluation
    QUICK_MODE            - Set to "1" for 5 sessions of 20 prompts (100 total)
    RUN_TIER_STATIC       - Set to "1" to run tier-static baseline comparison
    RUN_ROUND_ROBIN       - Set to "1" to run round-robin baseline after the main eval
    CPU_INFERENCE         - Set to "1" when vLLM serve is hosted on CPU; runs batched inference mode
"""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import litellm
from litellm import Router
from litellm.router_strategy.hybrid_router import HybridRouter, HybridRouterConfig

litellm.set_verbose = False

VLLM_9B_BASE = os.getenv("VLLM_9B_BASE", "http://localhost:8001/v1")
VLLM_4B_BASE = os.getenv("VLLM_4B_BASE", "http://localhost:8002/v1")

MODEL_9B = "Qwen3.5-9b"
MODEL_4B = "Qwen3.5-4b"

JUDGE_MODEL = "openai/google.gemma-3-12b-it"
JUDGE_CONCURRENCY = int(os.getenv("JUDGE_CONCURRENCY", "32"))
JUDGE_TIMEOUT = float(os.getenv("JUDGE_TIMEOUT", "15"))
JUDGE_SAMPLE_SIZE = int(os.getenv("JUDGE_SAMPLE_SIZE", "100"))

CONCURRENCY = int(os.getenv("HYBRID_EVAL_CONCURRENCY", "16"))
QUALITY_WEIGHT = float(os.getenv("QUALITY_WEIGHT", "0.5"))

_quick = os.getenv("QUICK_MODE", "0") == "1"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20" if _quick else "200"))
MAX_PROMPTS = 100 if _quick else None
RUN_TIER_STATIC = os.getenv("RUN_TIER_STATIC", "0") == "1"
RUN_ROUND_ROBIN = os.getenv("RUN_ROUND_ROBIN", "0") == "1"
CPU_INFERENCE = os.getenv("CPU_INFERENCE", "0") == "1"

CALIBRATION_PROMPT = "What is 2 + 2?"
CALIBRATION_MAX_TOKENS = 64


async def calibrate_target_tpt(router: Router, smallest_model: str) -> float:
    t0 = time.perf_counter()
    resp = await router.acompletion(
        model=smallest_model,
        messages=[{"role": "user", "content": CALIBRATION_PROMPT}],
        max_tokens=CALIBRATION_MAX_TOKENS,
    )
    latency = time.perf_counter() - t0
    tokens = resp.usage.completion_tokens if resp.usage else CALIBRATION_MAX_TOKENS
    tpt = latency / max(tokens, 1)
    print(f"  Calibration ({smallest_model}): latency={latency:.2f}s, tokens={tokens}, tpt={tpt:.3f}s/tok")
    return tpt


async def main():
    from scripts.efficiency_eval_prompts import PROMPT_METADATA

    router = Router(
        model_list=[
            {
                "model_name": "qwen-local-9b",
                "litellm_params": {
                    "model": f"openai/{MODEL_9B}",
                    "api_base": VLLM_9B_BASE,
                    "api_key": "not-needed",
                },
                "model_info": {
                    "id": "Qwen-9b",
                    "cache_creation_input_token_cost": 0.0,
                    "cache_read_input_token_cost": 0.0,
                },
            },
            {
                "model_name": "qwen-local-4b",
                "litellm_params": {
                    "model": f"openai/{MODEL_4B}",
                    "api_base": VLLM_4B_BASE,
                    "api_key": "not-needed",
                },
                "model_info": {
                    "id": "Qwen-4b",
                    "cache_creation_input_token_cost": 0.0,
                    "cache_read_input_token_cost": 0.0,
                },
            },
        ],
    )

    print("Calibrating target_tpt...")
    target_tpt = await calibrate_target_tpt(router, "qwen-local-4b")
    print()

    hybrid_config = HybridRouterConfig(
        tier_candidates={
            "SIMPLE": ("qwen-local-4b", "qwen-local-9b"),
            "MEDIUM": ("qwen-local-4b", "qwen-local-9b"),
            "COMPLEX": ("qwen-local-9b",),
            "REASONING": ("qwen-local-9b",),
        },
        bandit="thompson",
        target_tpt=target_tpt,
        tier_priors={
            "SIMPLE": {"qwen-local-4b": (5.0, 1.0), "qwen-local-9b": (1.0, 2.0)},
            "MEDIUM": {"qwen-local-4b": (1.0, 1.0), "qwen-local-9b": (2.0, 1.0)},
        },
    )
    hybrid = HybridRouter(hybrid_config)

    prompts = list(PROMPT_METADATA)
    if MAX_PROMPTS is not None:
        prompts = prompts[:MAX_PROMPTS]
    num_sessions = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE

    skip_judge = os.getenv("SKIP_JUDGE", "0") == "1"
    req_semaphore = asyncio.Semaphore(CONCURRENCY)
    judge_semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)

    bar_width = 40
    max_tokens_by_tier = {1: 64, 2: 128, 3: 256, 4: 512, 5: 512}

    print(f"=== Hybrid Router v1.0 ===")
    print(f"  Prompts: {len(prompts)}, Batch size: {BATCH_SIZE}, Sessions: {num_sessions}")
    print(f"  Quality weight: {QUALITY_WEIGHT}")
    print(f"  Judge: {JUDGE_MODEL}, Skip judge: {skip_judge}")
    print(f"  Concurrency: inference={CONCURRENCY}, judge={JUDGE_CONCURRENCY}")
    mode_label = "tier-static baseline" if RUN_TIER_STATIC else "bandit"
    if RUN_ROUND_ROBIN:
        mode_label += " + round-robin baseline"
    print(f"  Mode: {mode_label}")
    print()

    all_results: list[dict] = []
    session_summaries: list[dict] = []

    total_steps = len(prompts) * 2
    completed_steps = 0

    def update_progress(session_idx: int, phase: str) -> None:
        nonlocal completed_steps
        completed_steps += 1
        pct = completed_steps / total_steps
        filled = int(bar_width * pct)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"\r  [{bar}] {pct*100:.1f}% | Session {session_idx+1}/{num_sessions} ({phase})", end="", flush=True)

    t_total_start = time.perf_counter()

    if RUN_TIER_STATIC:
        print("BASELINE: Tier-Static (SIMPLE->4B, MEDIUM/COMPLEX/REASONING->9B)\n")

        for session_idx in range(num_sessions):
            batch_start = session_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(prompts))
            batch = prompts[batch_start:batch_end]

            decisions = []
            for meta in batch:
                tier, _ = hybrid.route(meta["prompt"])
                tier_key = tier.value if hasattr(tier, "value") else tier
                model = "qwen-local-4b" if tier_key == "SIMPLE" else "qwen-local-9b"
                decisions.append({"tier": tier, "model": model})

            async def run_one(meta, model, _session_idx=session_idx):
                async with req_semaphore:
                    t0 = time.perf_counter()
                    try:
                        resp = await router.acompletion(
                            model=model,
                            messages=[{"role": "user", "content": meta["prompt"]}],
                            max_tokens=max_tokens_by_tier[meta["tier"]],
                        )
                        latency = time.perf_counter() - t0
                        tokens = resp.usage.completion_tokens if resp.usage else 0
                        text = resp.choices[0].message.content if resp.choices else ""
                        update_progress(_session_idx, "inference")
                        return {"latency": latency, "tokens": tokens, "text": text or "", "success": True}
                    except Exception:
                        update_progress(_session_idx, "inference")
                        return {"latency": time.perf_counter() - t0, "tokens": 0, "text": "", "success": False}

            t0 = time.perf_counter()
            if CPU_INFERENCE:
                indexed_batch = list(enumerate(zip(batch, decisions)))
                responses = [None] * len(indexed_batch)
                models_in_batch = sorted(set(dec["model"] for dec in decisions))
                for model in models_in_batch:
                    model_items = [(i, meta, dec) for i, (meta, dec) in indexed_batch if dec["model"] == model]
                    model_responses = await asyncio.gather(*[
                        run_one(meta, dec["model"]) for _, meta, dec in model_items
                    ])
                    for (i, _, _), resp in zip(model_items, model_responses):
                        responses[i] = resp
            else:
                responses = await asyncio.gather(*[
                    run_one(meta, dec["model"]) for meta, dec in zip(batch, decisions)
                ])
            session_wall = time.perf_counter() - t0

            if skip_judge:
                scores = [1] * len(batch)
                completed_steps += len(batch)
                update_progress(session_idx, "judge-skip")
            else:
                async def judge_one(prompt, text, _session_idx=session_idx):
                    if not text:
                        update_progress(_session_idx, "judging")
                        return 0
                    async with judge_semaphore:
                        judge_prompt = (
                            "You are an accuracy judge. Given a question and an answer, "
                            "determine if the answer is correct.\n\n"
                            "Respond with ONLY a single digit: 1 if the answer is correct, "
                            "0 if it is incorrect or incomplete.\n\n"
                            f"Question: {prompt}\n\nAnswer: {text}\n\nVerdict (1 or 0):"
                        )
                        try:
                            resp = await asyncio.wait_for(
                                litellm.acompletion(
                                    model=JUDGE_MODEL,
                                    messages=[{"role": "user", "content": judge_prompt}],
                                    max_tokens=32,
                                    temperature=0.0,
                                ),
                                timeout=JUDGE_TIMEOUT,
                            )
                            verdict = resp.choices[0].message.content.strip()
                            update_progress(_session_idx, "judging")
                            return 1 if verdict.startswith("1") else 0
                        except Exception:
                            update_progress(_session_idx, "judging")
                            return -1

                scores = await asyncio.gather(*[
                    judge_one(m["prompt"], r["text"]) for m, r in zip(batch, responses)
                ])

            model_counts: dict[str, int] = {}
            for meta, dec, resp, score in zip(batch, decisions, responses, scores):
                model = dec["model"]
                model_counts[model] = model_counts.get(model, 0) + 1
                all_results.append({
                    "session": session_idx,
                    "model": model,
                    "tier": dec["tier"].value if hasattr(dec["tier"], "value") else dec["tier"],
                    "category": meta["category"],
                    "difficulty": meta["tier"],
                    "latency": resp["latency"],
                    "tokens": resp["tokens"],
                    "success": resp["success"],
                    "judge_score": score,
                    "reward": 0.0,
                })

            successful = [r for r in responses if r["success"]]
            avg_lat = sum(r["latency"] for r in successful) / len(successful) if successful else 0
            judged = [s for s in scores if s >= 0]
            accuracy = sum(1 for s in judged if s == 1) / len(judged) if judged else 0

            print(f"\n  Session {session_idx+1}/{num_sessions}: "
                  f"models={model_counts}, "
                  f"acc={accuracy:.3f}, "
                  f"lat={avg_lat:.3f}s, "
                  f"wall={session_wall:.1f}s")

    else:
        for session_idx in range(num_sessions):
            batch_start = session_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(prompts))
            batch = prompts[batch_start:batch_end]

            decisions = []
            for meta in batch:
                tier, model = hybrid.route(meta["prompt"])
                if QUALITY_WEIGHT >= 1.0:
                    model = "qwen-local-9b"
                elif QUALITY_WEIGHT <= 0.0:
                    model = "qwen-local-4b"
                decisions.append({"tier": tier, "model": model})

            async def run_one(meta, model, _session_idx=session_idx):
                async with req_semaphore:
                    t0 = time.perf_counter()
                    try:
                        resp = await router.acompletion(
                            model=model,
                            messages=[{"role": "user", "content": meta["prompt"]}],
                            max_tokens=max_tokens_by_tier[meta["tier"]],
                        )
                        latency = time.perf_counter() - t0
                        tokens = resp.usage.completion_tokens if resp.usage else 0
                        text = resp.choices[0].message.content if resp.choices else ""
                        update_progress(_session_idx, "inference")
                        return {"latency": latency, "tokens": tokens, "text": text or "", "success": True}
                    except Exception:
                        update_progress(_session_idx, "inference")
                        return {"latency": time.perf_counter() - t0, "tokens": 0, "text": "", "success": False}

            t0 = time.perf_counter()
            if CPU_INFERENCE:
                indexed_batch = list(enumerate(zip(batch, decisions)))
                responses = [None] * len(indexed_batch)
                models_in_batch = sorted(set(dec["model"] for dec in decisions))
                for model in models_in_batch:
                    model_items = [(i, meta, dec) for i, (meta, dec) in indexed_batch if dec["model"] == model]
                    model_responses = await asyncio.gather(*[
                        run_one(meta, dec["model"]) for _, meta, dec in model_items
                    ])
                    for (i, _, _), resp in zip(model_items, model_responses):
                        responses[i] = resp
            else:
                responses = await asyncio.gather(*[
                    run_one(meta, dec["model"]) for meta, dec in zip(batch, decisions)
                ])
            session_wall = time.perf_counter() - t0

            if skip_judge:
                scores = [1] * len(batch)
                completed_steps += len(batch)
                update_progress(session_idx, "judge-skip")
            else:
                async def judge_one(prompt, text, _session_idx=session_idx):
                    if not text:
                        update_progress(_session_idx, "judging")
                        return 0
                    async with judge_semaphore:
                        judge_prompt = (
                            "You are an accuracy judge. Given a question and an answer, "
                            "determine if the answer is correct.\n\n"
                            "Respond with ONLY a single digit: 1 if the answer is correct, "
                            "0 if it is incorrect or incomplete.\n\n"
                            f"Question: {prompt}\n\nAnswer: {text}\n\nVerdict (1 or 0):"
                        )
                        try:
                            resp = await asyncio.wait_for(
                                litellm.acompletion(
                                    model=JUDGE_MODEL,
                                    messages=[{"role": "user", "content": judge_prompt}],
                                    max_tokens=32,
                                    temperature=0.0,
                                ),
                                timeout=JUDGE_TIMEOUT,
                            )
                            verdict = resp.choices[0].message.content.strip()
                            update_progress(_session_idx, "judging")
                            return 1 if verdict.startswith("1") else 0
                        except Exception:
                            update_progress(_session_idx, "judging")
                            return -1

                scores = await asyncio.gather(*[
                    judge_one(m["prompt"], r["text"]) for m, r in zip(batch, responses)
                ])

            session_rewards: list[float] = []
            model_counts: dict[str, int] = {}
            for meta, dec, resp, score in zip(batch, decisions, responses, scores):
                model = dec["model"]
                tier = dec["tier"]
                model_counts[model] = model_counts.get(model, 0) + 1

                if resp["success"] and score >= 0:
                    reward = hybrid.record_quality(
                        tier, model, resp["latency"], resp["tokens"],
                        quality_score=float(score),
                        quality_weight=QUALITY_WEIGHT,
                    )
                elif resp["success"] and score == -1:
                    reward = hybrid.record_success(tier, model, resp["latency"], resp["tokens"])
                else:
                    hybrid.record_failure(tier, model)
                    reward = 0.0

                session_rewards.append(reward)
                all_results.append({
                    "session": session_idx,
                    "model": model,
                    "tier": tier.value if hasattr(tier, "value") else tier,
                    "category": meta["category"],
                    "difficulty": meta["tier"],
                    "latency": resp["latency"],
                    "tokens": resp["tokens"],
                    "success": resp["success"],
                    "judge_score": score,
                    "reward": reward,
                })

            successful = [r for r in responses if r["success"]]
            avg_lat = sum(r["latency"] for r in successful) / len(successful) if successful else 0
            judged = [s for s in scores if s >= 0]
            accuracy = sum(1 for s in judged if s == 1) / len(judged) if judged else 0
            avg_reward = sum(session_rewards) / len(session_rewards) if session_rewards else 0

            summary = {
                "session": session_idx,
                "n": len(batch),
                "model_dist": model_counts,
                "avg_reward": avg_reward,
                "avg_latency": avg_lat,
                "accuracy": accuracy,
                "wall_time": session_wall,
            }
            session_summaries.append(summary)

            print(f"\n  Session {session_idx+1}/{num_sessions}: "
                  f"models={model_counts}, "
                  f"acc={accuracy:.3f}, "
                  f"lat={avg_lat:.3f}s, "
                  f"reward={avg_reward:.3f}, "
                  f"wall={session_wall:.1f}s")

    total_wall = time.perf_counter() - t_total_start
    print(f"\r  [{'#' * bar_width}] 100.0% | Done{' ' * 30}")
    print()

    print(f"\n{'='*70}")
    print("FINAL RESULTS -- Hybrid Router v1.0")
    print(f"{'='*70}")

    total_success = [r for r in all_results if r["success"]]
    total_judged = [r for r in all_results if r["judge_score"] >= 0]
    total_correct = [r for r in total_judged if r["judge_score"] == 1]

    print(f"\n  Total requests: {len(all_results)}")
    print(f"  Success rate: {len(total_success)}/{len(all_results)}")
    print(f"  Total wall time: {total_wall:.1f}s")
    if total_judged:
        print(f"  Overall accuracy: {len(total_correct)}/{len(total_judged)} "
              f"({len(total_correct)/len(total_judged)*100:.1f}%)")
    overall_avg_lat = 0.0
    overall_tps = 0.0
    overall_total_tok = 0
    if total_success:
        overall_avg_lat = sum(r["latency"] for r in total_success) / len(total_success)
        overall_total_tok = sum(r["tokens"] for r in total_success)
        total_lat_sum = sum(r["latency"] for r in total_success)
        overall_tps = overall_total_tok / total_lat_sum if total_lat_sum > 0 else 0
        print(f"  Avg latency: {overall_avg_lat:.3f}s")
        print(f"  Total tokens: {overall_total_tok}")
        print(f"  Throughput: {overall_tps:.1f} tok/s")

    print(f"\n--- Per-Model Stats ---")
    models = sorted(set(r["model"] for r in all_results))
    for m in models:
        m_results = [r for r in all_results if r["model"] == m]
        m_success = [r for r in m_results if r["success"]]
        m_judged = [r for r in m_results if r["judge_score"] >= 0]
        m_correct = [r for r in m_judged if r["judge_score"] == 1]
        m_lat = sum(r["latency"] for r in m_success) / len(m_success) if m_success else 0
        m_lat_sum = sum(r["latency"] for r in m_success)
        m_tps = sum(r["tokens"] for r in m_success) / m_lat_sum if m_lat_sum > 0 else 0
        m_acc = len(m_correct) / len(m_judged) if m_judged else 0
        print(f"  {m}: n={len(m_results)}, acc={m_acc:.3f}, avg_lat={m_lat:.3f}s, tps={m_tps:.1f}")

    print(f"\n--- Per-Tier Stats ---")
    print(f"  {'Tier':<10} {'N':>5} {'Acc':>6} {'AvgLat':>8} {'TPS':>7} {'Model Dist'}")
    for tier in ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]:
        tier_results = [r for r in all_results if r["tier"] == tier]
        if not tier_results:
            continue
        t_success = [r for r in tier_results if r["success"]]
        t_judged = [r for r in tier_results if r["judge_score"] >= 0]
        t_correct = [r for r in t_judged if r["judge_score"] == 1]
        t_acc = len(t_correct) / len(t_judged) if t_judged else 0
        t_lat = sum(r["latency"] for r in t_success) / len(t_success) if t_success else 0
        t_lat_sum = sum(r["latency"] for r in t_success)
        t_tps = sum(r["tokens"] for r in t_success) / t_lat_sum if t_lat_sum > 0 else 0
        model_dist: dict[str, int] = {}
        for r in tier_results:
            model_dist[r["model"]] = model_dist.get(r["model"], 0) + 1
        print(f"  {tier:<10} {len(tier_results):>5} {t_acc:>5.3f} {t_lat:>8.3f} {t_tps:>6.1f} {model_dist}")

    print(f"\n--- Per-Category Per-Difficulty ---")
    print(f"  {'Cat/Diff':<12} {'N':>4} {'Acc':>6} {'AvgLat':>8} {'Model Dist'}")
    for cat in ("math", "code"):
        for pt in range(1, 6):
            cat_results = [r for r in all_results if r["category"] == cat and r["difficulty"] == pt]
            if not cat_results:
                continue
            c_success = [r for r in cat_results if r["success"]]
            c_judged = [r for r in cat_results if r["judge_score"] >= 0]
            c_correct = [r for r in c_judged if r["judge_score"] == 1]
            c_acc = len(c_correct) / len(c_judged) if c_judged else 0
            c_lat = sum(r["latency"] for r in c_success) / len(c_success) if c_success else 0
            model_dist = {}
            for r in cat_results:
                model_dist[r["model"]] = model_dist.get(r["model"], 0) + 1
            print(f"  {cat}/t{pt:<8} {len(cat_results):>4} {c_acc:>5.3f} {c_lat:>8.3f} {model_dist}")

    if not RUN_TIER_STATIC:
        print(f"\n--- Session Convergence ---")
        print(f"  {'Session':<8} {'4B%':>6} {'9B%':>6} {'Acc':>6} {'Lat':>7} {'Reward':>7}")
        for s in session_summaries:
            n = s["n"]
            pct_4b = s["model_dist"].get("qwen-local-4b", 0) / n * 100
            pct_9b = s["model_dist"].get("qwen-local-9b", 0) / n * 100
            print(f"  {s['session']+1:<8} {pct_4b:>5.1f}% {pct_9b:>5.1f}% "
                  f"{s['accuracy']:>5.3f} {s['avg_latency']:>6.3f} {s['avg_reward']:>6.3f}")

        print(f"\n--- Final Bandit State (Thompson posteriors) ---")
        state = hybrid.state()
        for tier, bandit_state in state["tier_bandits"].items():
            print(f"  {tier}:")
            print(f"    counts: {bandit_state['counts']}")
            means_str = {k: f"{v:.4f}" for k, v in bandit_state["means"].items()}
            print(f"    means:  {means_str}")
            if "alpha" in bandit_state:
                alpha_str = {k: f"{v:.2f}" for k, v in bandit_state["alpha"].items()}
                beta_str = {k: f"{v:.2f}" for k, v in bandit_state["beta"].items()}
                print(f"    alpha:  {alpha_str}")
                print(f"    beta:   {beta_str}")

    rr_results: list[dict] = []
    if RUN_ROUND_ROBIN:
        print(f"\n{'='*70}")
        print("BASELINE -- Round-Robin (no intelligence)")
        print(f"{'='*70}")

        rr_models = ["qwen-local-4b", "qwen-local-9b"]
        t_rr_start = time.perf_counter()

        for session_idx in range(num_sessions):
            batch_start = session_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(prompts))
            batch = prompts[batch_start:batch_end]

            async def rr_run_one(idx, meta, _session_idx=session_idx):
                model = rr_models[idx % len(rr_models)]
                async with req_semaphore:
                    t0 = time.perf_counter()
                    try:
                        resp = await router.acompletion(
                            model=model,
                            messages=[{"role": "user", "content": meta["prompt"]}],
                            max_tokens=max_tokens_by_tier[meta["tier"]],
                        )
                        latency = time.perf_counter() - t0
                        tokens = resp.usage.completion_tokens if resp.usage else 0
                        text = resp.choices[0].message.content if resp.choices else ""
                        return {"model": model, "latency": latency, "tokens": tokens, "text": text or "", "success": True}
                    except Exception:
                        return {"model": model, "latency": time.perf_counter() - t0, "tokens": 0, "text": "", "success": False}

            global_offset = batch_start
            if CPU_INFERENCE:
                indexed_batch = list(enumerate(batch))
                responses = [None] * len(indexed_batch)
                model_assignments = [rr_models[(global_offset + j) % len(rr_models)] for j in range(len(batch))]
                models_in_batch = sorted(set(model_assignments))
                for model in models_in_batch:
                    model_items = [(j, meta) for j, meta in indexed_batch if model_assignments[j] == model]
                    model_responses = await asyncio.gather(*[
                        rr_run_one(global_offset + j, meta) for j, meta in model_items
                    ])
                    for (j, _), resp in zip(model_items, model_responses):
                        responses[j] = resp
            else:
                responses = await asyncio.gather(*[
                    rr_run_one(global_offset + j, meta) for j, meta in enumerate(batch)
                ])

            if skip_judge:
                scores = [1] * len(batch)
            else:
                async def rr_judge_one(prompt, text):
                    if not text:
                        return 0
                    async with judge_semaphore:
                        judge_prompt = (
                            "You are an accuracy judge. Given a question and an answer, "
                            "determine if the answer is correct.\n\n"
                            "Respond with ONLY a single digit: 1 if the answer is correct, "
                            "0 if it is incorrect or incomplete.\n\n"
                            f"Question: {prompt}\n\nAnswer: {text}\n\nVerdict (1 or 0):"
                        )
                        try:
                            resp = await asyncio.wait_for(
                                litellm.acompletion(
                                    model=JUDGE_MODEL,
                                    messages=[{"role": "user", "content": judge_prompt}],
                                    max_tokens=32,
                                    temperature=0.0,
                                ),
                                timeout=JUDGE_TIMEOUT,
                            )
                            verdict = resp.choices[0].message.content.strip()
                            return 1 if verdict.startswith("1") else 0
                        except Exception:
                            return -1

                scores = await asyncio.gather(*[
                    rr_judge_one(m["prompt"], r["text"]) for m, r in zip(batch, responses)
                ])

            for meta, resp, score in zip(batch, responses, scores):
                rr_results.append({
                    "session": session_idx,
                    "model": resp["model"],
                    "category": meta["category"],
                    "difficulty": meta["tier"],
                    "latency": resp["latency"],
                    "tokens": resp["tokens"],
                    "success": resp["success"],
                    "judge_score": score,
                })

            successful = [r for r in responses if r["success"]]
            avg_lat = sum(r["latency"] for r in successful) / len(successful) if successful else 0
            judged = [s for s in scores if s >= 0]
            accuracy = sum(1 for s in judged if s == 1) / len(judged) if judged else 0
            model_counts = {}
            for r in responses:
                model_counts[r["model"]] = model_counts.get(r["model"], 0) + 1
            print(f"  Session {session_idx+1}/{num_sessions}: "
                  f"models={model_counts}, acc={accuracy:.3f}, lat={avg_lat:.3f}s")

        rr_wall = time.perf_counter() - t_rr_start
        rr_success = [r for r in rr_results if r["success"]]
        rr_judged = [r for r in rr_results if r["judge_score"] >= 0]
        rr_correct = [r for r in rr_judged if r["judge_score"] == 1]
        rr_avg_lat = sum(r["latency"] for r in rr_success) / len(rr_success) if rr_success else 0
        rr_total_tok = sum(r["tokens"] for r in rr_success)
        rr_lat_sum = sum(r["latency"] for r in rr_success)
        rr_tps = rr_total_tok / rr_lat_sum if rr_lat_sum > 0 else 0
        rr_acc = len(rr_correct) / len(rr_judged) if rr_judged else 0

        print(f"\n  Round-Robin Summary:")
        print(f"    Requests: {len(rr_results)}, Success: {len(rr_success)}")
        print(f"    Accuracy: {rr_acc:.3f} ({len(rr_correct)}/{len(rr_judged)})")
        print(f"    Avg latency: {rr_avg_lat:.3f}s")
        print(f"    Throughput: {rr_tps:.1f} tok/s")
        print(f"    Wall time: {rr_wall:.1f}s")

        if total_success:
            print(f"\n  --- Comparison: Hybrid Router vs Round-Robin ---")
            print(f"    {'Metric':<16} {'Hybrid':>10} {'RoundRobin':>12} {'Delta':>10}")
            hybrid_acc = len(total_correct) / len(total_judged) if total_judged else 0
            print(f"    {'Accuracy':<16} {hybrid_acc:>9.3f} {rr_acc:>11.3f} {hybrid_acc - rr_acc:>+9.3f}")
            print(f"    {'Avg Latency':<16} {overall_avg_lat:>9.3f}s {rr_avg_lat:>10.3f}s {overall_avg_lat - rr_avg_lat:>+9.3f}s")
            print(f"    {'Throughput':<16} {overall_tps:>8.1f} t/s {rr_tps:>9.1f} t/s {overall_tps - rr_tps:>+8.1f} t/s")

    mode = "tier-static" if RUN_TIER_STATIC else ("round-robin" if RUN_ROUND_ROBIN and not all_results else "bandit")
    output_path = "hybrid_v1_eval_results.json"
    output = {
        "config": {
            "mode": mode,
            "quality_weight": QUALITY_WEIGHT,
            "batch_size": BATCH_SIZE,
            "concurrency": CONCURRENCY,
            "tier_candidates": {k: list(v) for k, v in hybrid_config.tier_candidates.items()},
            "target_tpt": hybrid_config.target_tpt,
            "judge_model": JUDGE_MODEL,
        },
        "summary": {
            "total_requests": len(all_results),
            "total_success": len(total_success),
            "total_judged": len(total_judged),
            "total_correct": len(total_correct),
            "overall_accuracy": len(total_correct) / len(total_judged) if total_judged else 0,
            "avg_latency": overall_avg_lat,
            "throughput_tps": overall_tps,
            "total_wall_time": total_wall,
        },
        "session_summaries": session_summaries,
        "all_results": all_results,
    }
    if not RUN_TIER_STATIC:
        output["bandit_state"] = hybrid.state()
    if rr_results:
        output["round_robin_baseline"] = {
            "results": rr_results,
            "summary": {
                "total_requests": len(rr_results),
                "total_success": len(rr_success),
                "accuracy": rr_acc,
                "avg_latency": rr_avg_lat,
                "throughput_tps": rr_tps,
                "wall_time": rr_wall,
            },
        }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
