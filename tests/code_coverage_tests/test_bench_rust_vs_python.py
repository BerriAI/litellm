import json
import random
from collections import Counter
from pathlib import Path
from typing import Final

import pytest

from scripts.bench_rust_vs_python.flame import Profile, bucket_of, parse_folded
from scripts.bench_rust_vs_python.gen_requests import WARMUP_SEQ_START, Step, build_plan, step_windows
from scripts.bench_rust_vs_python.report import (
    RequestMetrics,
    RunMetrics,
    delta,
    run_metrics,
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
        {"marker": f"perfreq-{seq:06d}-n", "delay_s": 0.8, "generation_s": 1.0, "stream": False, "output_tokens": 80}
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


def test_flame_buckets_charge_the_frame_closest_to_the_leaf() -> None:
    logging_then_json: Final = [
        "main (proxy_server.py:1)",
        "async_log (litellm/integrations/s3_v2.py:9)",
        "loads (json/decoder.py:3)",
    ]
    folded: Final = parse_folded(
        [
            ";".join(logging_then_json) + " 30",
            "main (proxy_server.py:1);route (litellm/router_strategy/simple_shuffle.py:4) 10",
            "main (proxy_server.py:1);spin (unknown.py:1) 5",
        ]
    )

    assert bucket_of(logging_then_json) == "body parse / pydantic"
    assert folded == Counter({"body parse / pydantic": 30, "router selection": 10, "other": 5})
    assert Profile(samples=folded, rate_hz=100.0, requests=10).ms_per_request("router selection") == pytest.approx(10.0)
