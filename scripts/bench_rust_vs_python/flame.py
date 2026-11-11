"""py-spy folded stacks to a per-subsystem CPU table (ms per request, old vs new) and, when inferno is on PATH,
per-variant and differential flame graph SVGs.

Each sample is charged to the bucket of the frame closest to the leaf that matches any bucket, so a JSON parse
inside a logging callback counts as logging only when no more specific bucket claims it first."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

BUCKETS: Final = (
    ("Python<->Rust bridge", re.compile(r"rust_bridge|_native|pyo3|python_bridge|pythonize")),
    ("router selection", re.compile(r"router_strategy|router_core|router_rust|litellm_router|router\.py|router_api")),
    ("proxy auth", re.compile(r"user_api_key_auth|proxy/auth/|_check_master_key")),
    ("streaming / SSE", re.compile(r"streaming_handler|async_sse|sse_|StreamingResponse|CustomStreamWrapper|framer")),
    ("cost calc", re.compile(r"cost_calculator|cost_per_token|completion_cost|litellm_cost")),
    (
        "logging, s3_v2, prometheus",
        re.compile(
            r"litellm_logging|integrations/s3|integrations/prometheus|prometheus_client|botocore|boto3|custom_batch_logger|spend_tracking"
        ),
    ),
    (
        "provider transform",
        re.compile(r"llms/anthropic|llms/openai|transformation|inference_messages|inference_chat|llms::"),
    ),
    (
        "body parse / pydantic",
        re.compile(r"pydantic|orjson|json/decoder|json/encoder|_read_request_body|serde_json|model_validate"),
    ),
    ("HTTP client I/O", re.compile(r"httpx|httpcore|aiohttp|h11|ssl\.py|hyper|reqwest|http_handler")),
    ("asyncio", re.compile(r"asyncio/|uvloop|base_events|selector_events")),
)
OTHER: Final = "other"


@dataclass(frozen=True, slots=True)
class Profile:
    samples: Counter[str]
    rate_hz: float
    requests: int

    def ms_per_request(self, bucket: str) -> float:
        return 1000 * self.samples[bucket] / self.rate_hz / max(1, self.requests)


def bucket_of(frames: Sequence[str]) -> str:
    return next(
        (name for frame in reversed(frames) for name, pattern in BUCKETS if pattern.search(frame)),
        OTHER,
    )  # comprehension-ok: leaf-first search over frames then buckets


def parse_folded(lines: Iterable[str]) -> Counter[str]:
    parsed: Final = (line.rstrip().rsplit(" ", 1) for line in lines if line.strip())
    return _sum_by_bucket((bucket_of(stack.split(";")), int(count)) for stack, count in parsed)


def _sum_by_bucket(pairs: Iterable[tuple[str, int]]) -> Counter[str]:
    totals: Final = Counter[str]()
    for bucket, count in pairs:
        totals[bucket] += count
    return totals


def load_profile(run_dir: Path) -> Profile:
    meta: Final = json.loads((run_dir / "meta.json").read_text())
    with (run_dir / "profile.folded").open() as folded:
        return Profile(
            samples=parse_folded(folded), rate_hz=float(meta["profile_rate_hz"]), requests=int(meta["profile_requests"])
        )


def flamegraphs(old: Path, new: Path, out: Path) -> tuple[Path, ...]:
    tool: Final = shutil.which("inferno-flamegraph")
    differ: Final = shutil.which("inferno-diff-folded")
    if tool is None or differ is None:
        return ()
    outputs: Final = (out / "old.svg", out / "new.svg", out / "diff.svg")
    for source, target in ((old, outputs[0]), (new, outputs[1])):
        with source.open("rb") as stdin, target.open("wb") as stdout:
            _ = subprocess.run([tool], stdin=stdin, stdout=stdout, check=True)
    diff: Final = subprocess.run([differ, "--normalize", str(old), str(new)], capture_output=True, check=True).stdout
    with outputs[2].open("wb") as stdout:
        _ = subprocess.run([tool], input=diff, stdout=stdout, check=True)
    return outputs


def table(old: Profile, new: Profile) -> str:
    names: Final = tuple(name for name, _ in BUCKETS) + (OTHER,)
    deltas: Final = {name: new.ms_per_request(name) - old.ms_per_request(name) for name in names}
    total: Final = sum(deltas.values())
    rows: Final = "\n".join(
        f"| {name} | {old.ms_per_request(name):.2f} | {new.ms_per_request(name):.2f} | {deltas[name]:+.2f} | "
        f"{100 * deltas[name] / total if total else 0:+.0f}% |"
        for name in sorted(names, key=lambda n: -abs(deltas[n]))
    )
    return (
        "| bucket | old ms/req | new ms/req | delta | share of change |\n|---|---|---|---|---|\n"
        f"{rows}\n| total | {sum(old.ms_per_request(n) for n in names):.2f} | "
        f"{sum(new.ms_per_request(n) for n in names):.2f} | {total:+.2f} | |"
    )


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("old", type=Path, help="run dir with profile.folded")
    _ = parser.add_argument("new", type=Path)
    _ = parser.add_argument("--out", type=Path, required=True)
    args: Final = parser.parse_args()
    old_dir: Final[Path] = args.old  # pyright: ignore[reportAny]
    new_dir: Final[Path] = args.new  # pyright: ignore[reportAny]
    out: Final[Path] = args.out  # pyright: ignore[reportAny]
    out.mkdir(parents=True, exist_ok=True)
    markdown: Final = table(load_profile(old_dir), load_profile(new_dir))
    _ = (out / "subsystems.md").write_text(markdown + "\n")
    svgs: Final = flamegraphs(old_dir / "profile.folded", new_dir / "profile.folded", out)
    _ = sys.stdout.write(markdown + "\n")
    _ = sys.stdout.write(("\n".join(str(svg) for svg in svgs) or "inferno not on PATH, skipped SVGs") + "\n")


if __name__ == "__main__":
    main()
