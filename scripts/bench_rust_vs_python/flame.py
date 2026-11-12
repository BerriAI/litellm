"""py-spy folded stacks from two profiled run dirs (old, new) to a per-subsystem CPU table, a pruned flame tree and a
per-function comparison, each in CPU ms per request.

py-spy with --native counts blocked threads as running, and attaching it slows the proxy, so samples are first
classified (cpu, idle, gil_wait) and each profile is then scaled to the CPU per request the same run used before the
profiler attached. Widths and shares come from the profile, absolute milliseconds from the unprofiled measurement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

SampleKind = Literal["cpu", "idle", "gil_wait"]

BUCKETS: Final = (
    ("Python<->Rust crossing", re.compile(r"pythonize|json_from_py|json_to_py|marshal\.rs|trampoline\.rs|pyo3")),
    ("Rust code", re.compile(r"\.rs\)|_native::|litellm_[a-z_]+::|tokio|serde|reqwest|hyper")),
    ("router selection", re.compile(r"router_strategy|router_core|router_rust|router\.py|router_api")),
    ("proxy auth", re.compile(r"user_api_key_auth|proxy/auth/")),
    ("stream reassembly for logging", re.compile(r"stream_assembly|streaming_chunk_builder|passthrough_logging")),
    ("logging, s3_v2, prometheus", re.compile(r"litellm_logging|integrations/s3|integrations/prometheus|botocore")),
    ("body parse / pydantic / json", re.compile(r"pydantic|orjson|json|http_parsing_utils")),
    ("event loop / async I/O", re.compile(r"loop\.c|uv__|uvloop|asyncio/|h11|starlette|fastapi")),
)
OTHER: Final = "other"
MIN_SHARE: Final = 0.004
MAX_DEPTH: Final = 60

_PROCESS: Final = re.compile(r'process \d+:"[^"]*";?')
_LINE: Final = re.compile(r":\d+\)$")
_ADDRESS: Final = re.compile(r"^0x[0-9a-f]+ \(")
_PATH_PREFIXES: Final = (
    re.compile(r"\(/[^()]*?/site-packages/"),
    re.compile(r"\(/[^()]*?/wt/[^/()]+/"),
    re.compile(r"\(/usr/lib/python3\.\d+/"),
)
_WAITING: Final = re.compile(
    r"^(pthread_cond_timedwait|pthread_cond_wait|PyThread_acquire_lock_timed|epoll_pwait|epoll_wait|sem_clockwait|"
    + r"sem_timedwait|nanosleep|clock_nanosleep|select|poll|ppoll) \((libc|libpthread|python3)"
)
_SUPERVISOR_PIPE: Final = re.compile(r"_recv \(multiprocessing/connection\.py\)")
_GIL_WAIT: Final = re.compile(r"PyEval_RestoreThread|take_gil")


def clean(frame: str) -> str:
    """One frame with its line number dropped, unresolved native addresses folded per library and absolute paths
    trimmed, so the same function lines up across processes and checkouts."""
    without_line: Final = _ADDRESS.sub("[native] (", _LINE.sub(")", frame))
    return _strip_paths(without_line, _PATH_PREFIXES)


def _strip_paths(frame: str, prefixes: Sequence[re.Pattern[str]]) -> str:
    return frame if not prefixes else _strip_paths(prefixes[0].sub("(", frame), prefixes[1:])


def frames_of(stack: str) -> tuple[str, ...]:
    return tuple(clean(frame) for frame in _PROCESS.sub("", stack).split(";") if frame)


def classify(frames: Sequence[str]) -> SampleKind:
    """Threads parked in a wait or in uvicorn's supervisor pipe read are idle, and a thread reacquiring the GIL is
    waiting, not working. Every worker's stack starts in uvicorn's supervisor module, so only the pipe read counts."""
    if any(_GIL_WAIT.search(frame) for frame in frames):
        return "gil_wait"
    if any(_WAITING.match(frame) or _SUPERVISOR_PIPE.search(frame) for frame in frames):
        return "idle"
    return "cpu"


@dataclass(frozen=True, slots=True)
class Profile:
    cpu_stacks: Mapping[tuple[str, ...], int]
    kinds: Mapping[SampleKind, int]
    scale_ms: float
    """Milliseconds per request one CPU sample stands for once the profile is scaled to unprofiled CPU."""

    @property
    def cpu_samples(self) -> int:
        return sum(self.cpu_stacks.values())


def parse_folded(lines: Iterable[str]) -> tuple[Counter[tuple[str, ...]], Counter[SampleKind]]:
    parsed: Final = tuple(_parse_line(line) for line in lines if line.strip())
    stacks: Final = Counter[tuple[str, ...]]()
    kinds: Final = Counter[SampleKind]()
    for frames, kind, count in parsed:
        kinds[kind] += count
        if kind == "cpu":
            stacks[frames[:MAX_DEPTH]] += count
    return stacks, kinds


def _parse_line(line: str) -> tuple[tuple[str, ...], SampleKind, int]:
    stack, count = line.rstrip().rsplit(" ", 1)
    frames: Final = frames_of(stack)
    return frames, classify(frames), int(count)


def unprofiled_cpu_ms(run_dir: Path) -> float:
    """CPU per request between warmup start and the profiler attaching, from the run's own samples and records."""
    meta: Final = json.loads((run_dir / "meta.json").read_text())
    plan: Final = json.loads(Path(meta["plan"]).read_text())
    warmup_s: Final = min(float(item["send_at_s"]) for item in plan if int(item["seq"]) < 900_000)
    launched: Final = float(meta["launched_at"])
    attached: Final = float(meta.get("profile_started_at") or launched + warmup_s + 5)
    start: Final = launched + 15
    end: Final = attached - 2
    samples: Final = json.loads((run_dir / "samples.json").read_text())
    window: Final = [s for s in samples if start <= float(s["at"]) <= end]
    started: Final = sum(
        1
        for line in (run_dir / "records.jsonl").read_text().splitlines()
        if line and start <= float(json.loads(line)["started_at"]) <= end
    )
    cpu_s: Final = float(window[-1]["proxy_cpu_s"]) - float(window[0]["proxy_cpu_s"])
    return 1000 * cpu_s / max(1, started)


def load_profile(run_dir: Path) -> Profile:
    with (run_dir / "profile.folded").open() as folded:
        stacks, kinds = parse_folded(folded)
    return Profile(cpu_stacks=stacks, kinds=kinds, scale_ms=unprofiled_cpu_ms(run_dir) / max(1, sum(stacks.values())))


def bucket_of(frames: Sequence[str]) -> str:
    return next(
        (name for frame in reversed(frames) for name, pattern in BUCKETS if pattern.search(frame)),
        OTHER,
    )  # comprehension-ok: leaf-first search over frames then buckets


def bucket_ms(profile: Profile) -> Counter[str]:
    totals: Final = Counter[str]()
    for frames, count in profile.cpu_stacks.items():
        totals[bucket_of(frames)] += count * profile.scale_ms
    return totals


def tree_rows(variant: str, profile: Profile) -> list[dict[str, object]]:
    """Flame tree nodes (id, parent, name, inclusive ms per request), dropping any under MIN_SHARE of the profile."""
    inclusive: Final = Counter[tuple[str, ...]]()
    for frames, count in profile.cpu_stacks.items():
        for depth in range(1, len(frames) + 1):
            inclusive[frames[:depth]] += count
    total: Final = max(1, profile.cpu_samples)
    kept: Final = sorted((path for path, n in inclusive.items() if n >= total * MIN_SHARE), key=lambda p: (len(p), p))
    ids: Final = {path: f"{variant}{index}" for index, path in enumerate(kept)}
    return [
        {
            "variant": variant,
            "id": ids[path],
            "parent": ids.get(path[:-1], ""),
            "name": path[-1],
            "depth": len(path) - 1,
            "ms_per_request": round(inclusive[path] * profile.scale_ms, 3),
            "share": round(inclusive[path] / total, 5),
        }
        for path in kept
        if len(path) == 1 or path[:-1] in ids
    ]


def function_ms(profile: Profile) -> tuple[Counter[str], Counter[str]]:
    """Per function: own CPU (as the running frame) and inclusive CPU (anywhere in the stack, once per sample)."""
    own: Final = Counter[str]()
    inclusive: Final = Counter[str]()
    for frames, count in profile.cpu_stacks.items():
        own[frames[-1]] += count * profile.scale_ms
        for name in set(frames):
            inclusive[name] += count * profile.scale_ms
    return own, inclusive


def function_deltas(old: Profile, new: Profile, top: int = 40) -> list[dict[str, object]]:
    old_own, old_inclusive = function_ms(old)
    new_own, new_inclusive = function_ms(new)
    names: Final = set(old_own) | set(new_own)
    rows: Final = [
        {
            "function": name,
            "python_self_ms": round(old_own[name], 3),
            "rust_self_ms": round(new_own[name], 3),
            "delta_self_ms": round(new_own[name] - old_own[name], 3),
            "python_inclusive_ms": round(old_inclusive[name], 3),
            "rust_inclusive_ms": round(new_inclusive[name], 3),
            "delta_inclusive_ms": round(new_inclusive[name] - old_inclusive[name], 3),
        }
        for name in names
    ]
    return sorted(rows, key=lambda row: -abs(float(str(row["delta_self_ms"]))))[:top]


def table(old: Profile, new: Profile) -> str:
    old_ms: Final = bucket_ms(old)
    new_ms: Final = bucket_ms(new)
    names: Final = tuple(name for name, _ in BUCKETS) + (OTHER,)
    rows: Final = "\n".join(
        f"| {name} | {old_ms[name]:.2f} | {new_ms[name]:.2f} | {new_ms[name] - old_ms[name]:+.2f} |"
        for name in sorted(names, key=lambda n: -abs(new_ms[n] - old_ms[n]))
    )
    return (
        "| bucket | old ms/req | new ms/req | delta |\n|---|---|---|---|\n"
        + f"{rows}\n| total | {sum(old_ms.values()):.2f} | {sum(new_ms.values()):.2f} | "
        + f"{sum(new_ms.values()) - sum(old_ms.values()):+.2f} |"
    )


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("old", type=Path, help="profiled run dir with profile.folded, meta.json, samples.json")
    _ = parser.add_argument("new", type=Path)
    _ = parser.add_argument("--out", type=Path, required=True)
    args: Final = parser.parse_args()
    old_dir: Final[Path] = args.old  # pyright: ignore[reportAny]
    new_dir: Final[Path] = args.new  # pyright: ignore[reportAny]
    out: Final[Path] = args.out  # pyright: ignore[reportAny]
    out.mkdir(parents=True, exist_ok=True)
    old: Final = load_profile(old_dir)
    new: Final = load_profile(new_dir)
    markdown: Final = table(old, new)
    _ = (out / "subsystems.md").write_text(markdown + "\n")
    names: Final = tuple(str(json.loads((d / "meta.json").read_text())["variant"]) for d in (old_dir, new_dir))
    _ = (out / "flame_tree.json").write_text(json.dumps(tree_rows(names[0], old) + tree_rows(names[1], new)))
    _ = (out / "function_deltas.json").write_text(json.dumps(function_deltas(old, new)))
    _ = sys.stdout.write(markdown + "\n")


if __name__ == "__main__":
    main()
