"""Frozen request plan for one benchmark path, replayed byte for byte by every variant and repetition.

Arrivals are open loop: Poisson at a fixed rate (`--rps`), or stepped through a ramp (`--ramp`). Warmup requests
come first and carry markers from `WARMUP_SEQ_START` up, so analysis drops them by sequence number.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import accumulate, count, takewhile
from pathlib import Path
from typing import Final, Literal

PERF_DIR: Final = Path(__file__).resolve().parents[2] / "tests" / "e2e" / "perf"
sys.path.insert(0, str(PERF_DIR))

from pydantic import TypeAdapter  # noqa: E402  # import after the harness path is on sys.path
from setup.wire import WireRequest, WireShape  # noqa: E402  # import after the harness path is on sys.path

BenchPath = Literal["messages", "chat"]
WARMUP_SEQ_START: Final = 900_000
INPUT_TOKENS: Final = (45_000, 55_000)
MAX_TOKENS: Final = (1024, 2048, 4096)
STREAM_SHARE: Final = 0.85
MASTER_KEY_ENV: Final = "LITELLM_MASTER_KEY"
MODELS: Final = {"messages": "claude-sonnet-5-5", "chat": "gpt-6.1-sol"}
ENDPOINTS: Final = {"messages": "messages", "chat": "chat_completions"}
URL_PATHS: Final = {"messages": "/v1/messages", "chat": "/v1/chat/completions"}
_PLAN: Final = TypeAdapter(list[WireRequest])


@dataclass(frozen=True, slots=True)
class Step:
    rps: float
    seconds: float


def poisson_offsets(rng: random.Random, steps: tuple[Step, ...]) -> tuple[float, ...]:
    starts: Final = (0.0, *accumulate(step.seconds for step in steps))
    return tuple(offset for step, start in zip(steps, starts, strict=False) for offset in _arrivals(rng, step, start))


def _arrivals(rng: random.Random, step: Step, start: float) -> Iterator[float]:
    gaps: Final = accumulate(rng.expovariate(step.rps) for _ in count())
    return (start + gap for gap in takewhile(lambda gap: gap < step.seconds, gaps))


def wire_request(
    rng: random.Random,
    *,
    path: BenchPath,
    seq: int,
    send_at_s: float,
    master_key: str,
    input_tokens: tuple[int, int] = INPUT_TOKENS,
) -> WireRequest:
    max_tokens: Final = rng.choice(MAX_TOKENS)
    return WireRequest(
        seq=seq,
        marker=f"perfreq-{seq:06d}-n",
        caller="key:master",
        endpoint=ENDPOINTS[path],
        path=URL_PATHS[path],
        model=MODELS[path],
        shape=WireShape(
            name=f"coding-agent-{input_tokens[1] // 1000}k",
            input_tokens=rng.randint(*input_tokens),
            tools=True,
            output_tokens=max_tokens,
        ),
        authorization=f"Bearer {master_key}",
        stream=rng.random() < STREAM_SHARE,
        blocked=False,
        send_at_s=send_at_s,
    )


def build_plan(
    *,
    path: BenchPath,
    seed: int,
    warmup: Step,
    measured: tuple[Step, ...],
    master_key: str,
    input_tokens: tuple[int, int] = INPUT_TOKENS,
) -> tuple[WireRequest, ...]:
    rng: Final = random.Random(seed)
    warmup_offsets: Final = poisson_offsets(rng, (warmup,))
    measured_offsets: Final = tuple(warmup.seconds + offset for offset in poisson_offsets(rng, measured))
    warm: Final = tuple(
        wire_request(
            rng,
            path=path,
            seq=WARMUP_SEQ_START + index,
            send_at_s=offset,
            master_key=master_key,
            input_tokens=input_tokens,
        )
        for index, offset in enumerate(warmup_offsets, start=1)
    )
    timed: Final = tuple(
        wire_request(rng, path=path, seq=index, send_at_s=offset, master_key=master_key, input_tokens=input_tokens)
        for index, offset in enumerate(measured_offsets, start=1)
    )
    return (*warm, *timed)


def sequential_plan(
    *, path: BenchPath, seed: int, warmup: int, measured: int, master_key: str, input_tokens: tuple[int, int]
) -> tuple[WireRequest, ...]:
    """Requests sent one after another by a single client: every send time is 0, so each goes out as soon as the
    previous one finishes, and CPU per request is measured without contention between concurrent requests."""
    rng: Final = random.Random(seed)
    seqs: Final = (*range(WARMUP_SEQ_START + 1, WARMUP_SEQ_START + warmup + 1), *range(1, measured + 1))
    return tuple(
        wire_request(rng, path=path, seq=seq, send_at_s=0.0, master_key=master_key, input_tokens=input_tokens)
        for seq in seqs
    )


def step_windows(warmup_seconds: float, measured: tuple[Step, ...]) -> list[dict[str, float]]:
    starts: Final = tuple(accumulate((step.seconds for step in measured), initial=warmup_seconds))
    return [
        {"rps": step.rps, "start_s": start, "end_s": start + step.seconds}
        for step, start in zip(measured, starts, strict=False)
    ]


def parse_range(spec: str) -> tuple[int, int]:
    low, high = (int(part) for part in spec.split(","))
    return (low, high)


def parse_ramp(spec: str) -> tuple[Step, ...]:
    first, last, step, seconds = (float(part) for part in spec.split(","))
    count: Final = int(round((last - first) / step)) + 1
    return tuple(Step(rps=first + index * step, seconds=seconds) for index in range(count))


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--path", choices=("messages", "chat"), required=True)
    _ = parser.add_argument("--seed", type=int, required=True)
    _ = parser.add_argument("--rps", type=float, help="fixed open-loop rate for the measured window")
    _ = parser.add_argument("--seconds", type=float, default=300.0, help="measured window at --rps")
    _ = parser.add_argument("--ramp", help="first,last,step,step_seconds for a stepped ramp instead of --rps")
    _ = parser.add_argument("--warmup-rps", type=float)
    _ = parser.add_argument("--warmup-seconds", type=float, default=60.0)
    _ = parser.add_argument("--master-key", default="sk-bench-master")
    _ = parser.add_argument("--input-tokens", default="45000,55000", help="low,high input tokens per request")
    _ = parser.add_argument("--sequential", type=int, help="measured requests sent one at a time, instead of --rps")
    _ = parser.add_argument("--warmup-requests", type=int, default=300, help="discarded requests before --sequential")
    _ = parser.add_argument("--out", type=Path, required=True)
    args: Final = parser.parse_args()
    out_path: Final[Path] = args.out  # pyright: ignore[reportAny]
    sequential: Final[int | None] = args.sequential  # pyright: ignore[reportAny]
    if sequential is not None:
        plan_seq: Final = sequential_plan(
            path=args.path,  # pyright: ignore[reportAny]
            seed=int(args.seed),  # pyright: ignore[reportAny]
            warmup=int(args.warmup_requests),  # pyright: ignore[reportAny]
            measured=sequential,
            master_key=str(args.master_key),  # pyright: ignore[reportAny]
            input_tokens=parse_range(str(args.input_tokens)),  # pyright: ignore[reportAny]
        )
        _ = out_path.write_bytes(_PLAN.dump_json(list(plan_seq)))
        _ = out_path.with_suffix(".steps.json").write_text("[]")
        _ = sys.stdout.write(f"{out_path}: {len(plan_seq)} requests sent one at a time\n")
        return
    ramp_spec: Final[str | None] = args.ramp  # pyright: ignore[reportAny]
    rps: Final[float | None] = args.rps  # pyright: ignore[reportAny]
    if (ramp_spec is None) == (rps is None):
        parser.error("pass exactly one of --rps and --ramp")
    measured: Final = parse_ramp(ramp_spec) if ramp_spec is not None else (Step(rps or 0.0, float(args.seconds)),)  # pyright: ignore[reportAny]
    warmup_rps: Final = float(args.warmup_rps or measured[0].rps)  # pyright: ignore[reportAny]
    plan: Final = build_plan(
        path=args.path,  # pyright: ignore[reportAny]
        seed=int(args.seed),  # pyright: ignore[reportAny]
        warmup=Step(warmup_rps, float(args.warmup_seconds)),  # pyright: ignore[reportAny]
        measured=measured,
        master_key=str(args.master_key),  # pyright: ignore[reportAny]
        input_tokens=parse_range(str(args.input_tokens)),  # pyright: ignore[reportAny]
    )
    out: Final[Path] = args.out  # pyright: ignore[reportAny]
    _ = out.write_bytes(_PLAN.dump_json(list(plan)))
    _ = out.with_suffix(".steps.json").write_text(json.dumps(step_windows(float(args.warmup_seconds), measured)))  # pyright: ignore[reportAny]
    _ = sys.stdout.write(f"{out}: {len(plan)} requests, last send at {plan[-1].send_at_s:.1f}s" + "\n")


if __name__ == "__main__":
    main()
