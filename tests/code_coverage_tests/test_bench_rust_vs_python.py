import json
import random
from collections import Counter
from pathlib import Path
from typing import Final

import pytest

from scripts.bench_rust_vs_python.flame import Profile, bucket_ms, function_ms, parse_folded, tree_rows
from scripts.bench_rust_vs_python.gen_requests import WARMUP_SEQ_START, Step, build_plan, step_windows
from scripts.bench_rust_vs_python.report import (
    RequestMetrics,
    RunMetrics,
    delta,
    request_metrics,
    run_metrics,
    segments,
    sustained,
    verdict,
)
from scripts.bench_rust_vs_python.report import Step as WindowStep

T0: Final = 1_000_000.0
CPU_S_PER_WALL_S: Final = 0.5


def _write_run(run_dir: Path, *, rust: bool) -> None:
    run_dir.mkdir()
    sends: Final = {WARMUP_SEQ_START + 1: 0.0, WARMUP_SEQ_START + 2: 5.0, **{seq: 9.0 + seq for seq in range(1, 11)}}
    plan: Final = run_dir / "plan.json"
    _ = plan.write_text(json.dumps([{"seq": seq, "send_at_s": at} for seq, at in sends.items()]))
    _ = plan.with_suffix(".steps.json").write_text(json.dumps(step_windows(10.0, (Step(1.0, 10.0),))))
    records: Final = [
        {
            "seq": seq,
            "marker": f"perfreq-{seq:06d}-n",
            "ok": seq != 10,
            "stream": seq % 2 == 0,
            "started_at": T0 + at + 0.001,
            "total_s": 2.0,
            "first_token_s": 0.9,
            "litellm_overhead_ms": 7.0,
            "litellm_rust": rust,
        }
        for seq, at in sends.items()
    ]
    _ = (run_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    mock: Final = [
        {
            "marker": f"perfreq-{seq:06d}-n",
            "delay_s": 0.8,
            "generation_s": 1.0,
            "stream": False,
            "output_tokens": 80,
            "user_agent": "python-httpx/0.28.1" if seq in (1, 2, WARMUP_SEQ_START + 1) else "litellm-rust/1.0",
        }
        for seq in sends
    ]
    _ = (run_dir / "mock.json").write_text(json.dumps({"requests": mock}))
    samples: Final = [
        {
            "at": T0 + second,
            "proxy_cpu_s": CPU_S_PER_WALL_S * second,
            "proxy_rss_bytes": 2**30,
            "loadgen_cpu_s": 0.1 * second,
        }
        for second in range(0, 25)
    ]
    _ = (run_dir / "samples.json").write_text(json.dumps(samples))
    meta: Final = {"variant": "A1", "plan": str(plan), "prometheus_requests_delta": 11.0, "ok": 11, "s3_objects": 11}
    _ = (run_dir / "meta.json").write_text(json.dumps(meta))


def test_run_metrics_drops_warmup_and_charges_window_cpu_to_successful_requests(tmp_path: Path) -> None:
    _write_run(tmp_path / "run", rust=True)
    run: Final = run_metrics(tmp_path / "run")

    assert run.measured == 10
    assert run.errors == 1
    assert run.window_s == pytest.approx(9.0)
    assert run.cpu_ms_per_request == pytest.approx(1000 * CPU_S_PER_WALL_S * 9.0 / 9)
    assert run.loadgen_cpu_share == pytest.approx(0.1)
    assert run.overheads_s == pytest.approx((0.2,) * 9)
    assert len(run.ttft_overheads_s) == 4
    assert run.ttft_overheads_s == pytest.approx((0.1,) * 4)
    assert run.rust_share == 1.0
    assert run.prometheus_matches and run.s3_matches


def _request(send_at_s: float, *, overhead_s: float, ok: bool = True) -> RequestMetrics:
    return RequestMetrics(
        send_at_s=send_at_s,
        ok=ok,
        stream=False,
        rust=False,
        overhead_s=overhead_s,
        ttft_overhead_s=None,
        header_overhead_ms=None,
    )


def test_sustained_rps_stops_at_the_first_step_that_breaks_a_limit() -> None:
    steps: Final = (WindowStep(5.0, 0.0, 10.0), WindowStep(10.0, 10.0, 20.0), WindowStep(15.0, 20.0, 30.0))
    healthy: Final = tuple(_request(at + 0.5, overhead_s=0.05) for at in range(0, 30))
    slow_middle: Final = tuple(_request(at + 0.5, overhead_s=0.9 if 10 <= at < 20 else 0.05) for at in range(0, 30))
    failing_first: Final = tuple(_request(at + 0.5, overhead_s=0.05, ok=at >= 1) for at in range(0, 30))

    assert sustained(healthy, steps) == 15.0
    assert sustained(slow_middle, steps) == 5.0
    assert sustained(failing_first, steps) is None


def _run(variant: str, cpu_ms: float, overhead_s: float) -> RunMetrics:
    return RunMetrics(
        variant=variant,
        run=variant,
        measured=100,
        errors=0,
        rust_share=0.0,
        window_s=60.0,
        cpu_ms_per_request=cpu_ms,
        loadgen_cpu_share=0.1,
        peak_rss_mb=500.0,
        rss_growth_mb=0.0,
        prometheus_matches=True,
        s3_matches=True,
        overheads_s=(overhead_s,) * 50,
        ttft_overheads_s=(),
        header_overheads_ms=(),
        sustained_rps=None,
    )


def test_delta_reports_a_clear_change_and_no_change_for_equal_variants() -> None:
    jitter: Final = random.Random(3)
    old: Final = tuple(_run("old", 40 + jitter.uniform(-0.5, 0.5), 0.05) for _ in range(5))
    faster: Final = tuple(_run("new", 30 + jitter.uniform(-0.5, 0.5), 0.05) for _ in range(5))
    same: Final = tuple(_run("new", 40 + jitter.uniform(-0.5, 0.5), 0.05) for _ in range(5))

    improvement: Final = delta("cpu_ms_per_request", old, faster, seed=1)
    unchanged: Final = delta("cpu_ms_per_request", old, same, seed=1)

    assert improvement.change_pct == pytest.approx(-25, abs=2)
    assert improvement.ci_low_pct < improvement.change_pct < improvement.ci_high_pct < 0
    assert verdict(improvement, floor=3.0) == "lower"
    assert verdict(improvement, floor=30.0) == "no change"
    assert unchanged.ci_low_pct < 0 < unchanged.ci_high_pct
    assert verdict(unchanged, floor=None) == "no change"
    assert verdict(delta("cpu_ms_per_request", old[:1], faster[:1], seed=1), floor=None) == "single run, no CI"


def test_plan_is_deterministic_open_loop_with_warmup_first() -> None:
    plan: Final = build_plan(
        path="messages", seed=5, warmup=Step(10.0, 30.0), measured=(Step(10.0, 300.0),), master_key="k"
    )
    again: Final = build_plan(
        path="messages", seed=5, warmup=Step(10.0, 30.0), measured=(Step(10.0, 300.0),), master_key="k"
    )
    warmup: Final = tuple(item for item in plan if item.seq >= WARMUP_SEQ_START)
    measured: Final = tuple(item for item in plan if item.seq < WARMUP_SEQ_START)

    assert plan == again
    assert plan != build_plan(
        path="messages", seed=6, warmup=Step(10.0, 30.0), measured=(Step(10.0, 300.0),), master_key="k"
    )
    assert max(item.send_at_s or 0 for item in warmup) < 30.0 <= min(item.send_at_s or 0 for item in measured)
    assert 2700 < len(measured) < 3300
    assert 0.8 < sum(item.stream for item in measured) / len(measured) < 0.9
    assert all(45_000 <= item.shape.input_tokens <= 55_000 for item in plan)
    assert {item.endpoint for item in plan} == {"messages"}


def test_ramp_step_windows_follow_the_warmup_back_to_back() -> None:
    windows: Final = step_windows(60.0, (Step(5.0, 30.0), Step(10.0, 30.0)))

    assert windows == [{"rps": 5.0, "start_s": 60.0, "end_s": 90.0}, {"rps": 10.0, "start_s": 90.0, "end_s": 120.0}]


def test_flame_keeps_worker_request_handling_and_drops_only_waiting_threads() -> None:
    worker: Final = (
        'process 1:"python proxy_cli.py";process 2:"python -c from multiprocessing.spawn import spawn_main; '
        'spawn_main(tracker_fd=12, pipe_handle=13) --multiprocessing-fork";'
        "target (/x/site-packages/uvicorn/supervisors/multiprocess.py:9);"
        "user_api_key_auth (/home/user/bench/wt/main/litellm/proxy/auth/user_api_key_auth.py:88)"
    )
    stacks, kinds = parse_folded(
        [
            f"{worker} 30",
            'process 1:"python proxy_cli.py";pong (multiprocessing.py:1);_recv (/usr/lib/python3.12/multiprocessing/connection.py:4);read (libc.so.6) 50',
            'process 2:"w";_worker (concurrent/futures/thread.py:3);PyThread_acquire_lock_timed (python3.12);0x7f42 (libc.so.6) 900',
            'process 2:"w";encode (tiktoken/core.py:5);PyEval_RestoreThread (python3.12);pthread_cond_timedwait (libc.so.6) 7',
            'process 3:"w";poll (x.py:1);0x7f16b38a5098 (libc.so.6) 1',
            'process 4:"w";poll (x.py:1);0x7f421daa5098 (libc.so.6) 2',
        ]
    )

    assert kinds == Counter({"idle": 950, "cpu": 33, "gil_wait": 7})
    assert (
        stacks[
            (
                "target (uvicorn/supervisors/multiprocess.py)",
                "user_api_key_auth (litellm/proxy/auth/user_api_key_auth.py)",
            )
        ]
        == 30
    )
    assert stacks[("poll (x.py)", "[native] (libc.so.6)")] == 3


def test_flame_scales_each_profile_to_its_unprofiled_cpu_and_charges_the_leaf_most_bucket() -> None:
    profile: Final = Profile(
        cpu_stacks={
            ("main (proxy_server.py)", "loads (json/decoder.py)"): 30,
            ("main (proxy_server.py)", "route (litellm/router.py)"): 10,
        },
        kinds={"cpu": 40},
        scale_ms=0.5,
    )
    own, inclusive = function_ms(profile)

    assert bucket_ms(profile) == Counter({"body parse / pydantic / json": 15.0, "router selection": 5.0})
    assert inclusive["main (proxy_server.py)"] == 20.0
    assert own["loads (json/decoder.py)"] == 15.0
    assert {row["name"]: row["ms_per_request"] for row in tree_rows("A1", profile)} == {
        "main (proxy_server.py)": 20.0,
        "loads (json/decoder.py)": 15.0,
        "route (litellm/router.py)": 5.0,
    }


def test_run_log_keeps_the_upstream_user_agent_the_mock_recorded() -> None:
    from scripts.bench_rust_vs_python.run_bench import MockLog

    served: Final = (
        '{"requests":[{"marker":"m","delay_s":0.8,"stream":true,"output_tokens":9,"user_agent":"litellm/1.0"}]}'
    )

    assert MockLog.model_validate_json(served).requests[0].user_agent == "litellm/1.0"


def test_segments_split_the_added_time_at_the_moments_the_provider_saw() -> None:
    record: Final = {
        "marker": "perfreq-000001-n",
        "ok": True,
        "stream": True,
        "started_at": 100.0,
        "total_s": 3.0,
        "first_token_s": 1.0,
    }
    upstream: Final = {"received_at": 100.05, "delay_s": 0.8, "generation_s": 2.0}
    split: Final = segments(record, upstream)
    overhead: Final = request_metrics(record, 0.0, {"perfreq-000001-n": upstream}).overhead_s

    assert split is not None and overhead is not None
    assert split.pre_upstream_s == pytest.approx(0.05)
    assert split.stream_start_s == pytest.approx(0.15)
    assert split.tail_s == pytest.approx(0.15)
    assert split.pre_upstream_s + split.tail_s == pytest.approx(overhead)
    assert segments({**record, "stream": False}, upstream).stream_start_s is None
    assert segments(record, {"delay_s": 0.8, "generation_s": 2.0}) is None


def test_sequential_window_spans_the_measured_requests_as_they_ran() -> None:
    from scripts.bench_rust_vs_python.report import measured_window

    records: Final = [
        {"seq": WARMUP_SEQ_START + 1, "started_at": 100.0, "total_s": 0.1},
        {"seq": 1, "started_at": 100.2, "total_s": 0.1},
        {"seq": 2, "started_at": 100.4, "total_s": 0.3},
    ]

    assert measured_window(records, {WARMUP_SEQ_START + 1: 0.0, 1: 0.0, 2: 0.0}) == pytest.approx((100.2, 100.7))
    assert measured_window(records, {WARMUP_SEQ_START + 1: 0.0, 1: 0.2, 2: 0.4}) == pytest.approx((100.2, 100.4))


def test_rounds_order_runs_every_variant_once_per_round() -> None:
    from scripts.bench_rust_vs_python.run_bench import run_order

    order: Final = run_order(("A0", "A1", "HP"), reps=3, seed=4, order="rounds")

    assert [sorted(name for name, rep in order if rep == r) for r in (1, 2, 3)] == [["A0", "A1", "HP"]] * 3
    assert [rep for _, rep in order] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert sorted(run_order(("A0", "A1"), reps=2, seed=4, order="shuffle")) == [
        ("A0", 1),
        ("A0", 2),
        ("A1", 1),
        ("A1", 2),
    ]
