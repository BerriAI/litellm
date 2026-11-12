"""Runs one benchmark variant against a frozen plan, or a randomized matrix of variants and repetitions.

Each run gets a fresh proxy from the variant's worktree, a fresh mock provider, a fresh S3 server and a fresh
Prometheus multiprocess dir. The harness side (mocks, Locust runner, corpus) always comes from this checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

import boto3
import yaml
from pydantic import BaseModel, ConfigDict

HERE: Final = Path(__file__).resolve().parent
REPO: Final = HERE.parents[1]
PERF_DIR: Final = REPO / "tests" / "e2e" / "perf"
MOCKS_DIR: Final = PERF_DIR / "setup" / "mocks"
LOCUSTFILE: Final = PERF_DIR / "setup" / "locustfile.py"
CONFIG: Final = HERE / "configs" / "bench.yaml"
MASTER_KEY: Final = "sk-bench-master"
PROXY_PORT: Final = 4000
MOCK_PORT: Final = 8090
S3_PORT: Final = 9000
BUCKET: Final = "bench-logs"
SAMPLE_INTERVAL_S: Final = 1.0
PROFILE_RATE_HZ: Final = 200
CLK_TCK: Final = os.sysconf("SC_CLK_TCK")
PAGE_SIZE: Final = os.sysconf("SC_PAGE_SIZE")
MOCK_KIND: Final = {"messages": "anthropic", "chat": "openai"}
STRIPPED_ENV_PREFIXES: Final = ("LITELLM_", "PROMETHEUS_", "AWS_")

BenchPath = Literal["messages", "chat"]


class Variant(BaseModel):
    model_config = ConfigDict(frozen=True)

    worktree: str
    path: BenchPath
    env: dict[str, str]
    expect_rust: bool = False


class Variants(BaseModel):
    model_config = ConfigDict(frozen=True)

    commits: dict[str, str]
    variants: dict[str, Variant]


class MockEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    marker: str | None
    delay_s: float
    generation_s: float = 0.0
    stream: bool
    output_tokens: int
    user_agent: str | None = None


class MockLog(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests: list[MockEntry]


@dataclass(frozen=True, slots=True)
class Cores:
    proxy: str
    loadgen: str
    services: str
    workers: int


@dataclass(frozen=True, slots=True)
class Sample:
    at: float
    proxy_cpu_s: float
    proxy_rss_bytes: int
    proxy_processes: int
    loadgen_cpu_s: float


@dataclass(frozen=True, slots=True)
class Latency:
    seconds: float
    jitter_seconds: float
    token_interval_seconds: float


def load_variants() -> Variants:
    return Variants.model_validate(yaml.safe_load((HERE / "variants.yaml").read_text()))


def _http(method: str, url: str, body: Mapping[str, object] | None = None, timeout: float = 10.0) -> bytes:
    data: Final = None if body is None else json.dumps(body).encode()
    request: Final = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def busy_ports(ports: Sequence[int]) -> tuple[int, ...]:
    def listening(port: int) -> bool:
        with socket.socket() as probe:
            return probe.connect_ex(("127.0.0.1", port)) == 0

    return tuple(port for port in ports if listening(port))


def wait_until_up(url: str, process: subprocess.Popen[bytes], timeout_s: float) -> None:
    deadline: Final = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{process.args!r} exited with {process.returncode} before {url} came up")
        try:
            _ = _http("GET", url, timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{url} not up after {timeout_s}s")


def _stat_fields(pid: int) -> list[str] | None:
    try:
        raw: Final = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return raw[raw.rindex(")") + 2 :].split()


def descendants(root: int) -> tuple[int, ...]:
    parents: Final = {
        pid: int(fields[1])
        for pid in (int(entry) for entry in os.listdir("/proc") if entry.isdigit())
        if (fields := _stat_fields(pid)) is not None
    }
    found: Final = [root]  # mutable-ok: breadth-first walk of the process tree
    for pid in found:
        found.extend(child for child, parent in parents.items() if parent == pid)
    return tuple(found)


def cpu_seconds(pids: Sequence[int]) -> float:
    fields: Final = tuple(f for f in (_stat_fields(pid) for pid in pids) if f is not None)
    return sum(int(f[11]) + int(f[12]) for f in fields) / CLK_TCK


def rss_bytes(pids: Sequence[int]) -> int:
    fields: Final = tuple(f for f in (_stat_fields(pid) for pid in pids) if f is not None)
    return sum(int(f[21]) for f in fields) * PAGE_SIZE


class Sampler(threading.Thread):
    """CPU and RSS of the whole proxy process tree and the load generator, once a second."""

    def __init__(self, proxy_pid: int, loadgen_pid: int) -> None:
        super().__init__(daemon=True)
        self.proxy_pid: Final = proxy_pid
        self.loadgen_pid: Final = loadgen_pid
        self.samples: Final[list[Sample]] = []  # mutable-ok: appended by the sampling thread
        self.stopped: Final = threading.Event()
        self.exited_cpu_s = 0.0

    def run(self) -> None:
        while not self.stopped.wait(SAMPLE_INTERVAL_S):
            pids = descendants(self.proxy_pid)
            self.samples.append(
                Sample(
                    at=time.time(),
                    proxy_cpu_s=cpu_seconds(pids),
                    proxy_rss_bytes=rss_bytes(pids),
                    proxy_processes=len(pids),
                    loadgen_cpu_s=cpu_seconds(descendants(self.loadgen_pid)),
                )
            )


def prometheus_requests(base_url: str) -> float:
    text: Final = _http("GET", f"{base_url}/metrics", timeout=30).decode()
    return sum(
        float(line.rsplit(" ", 1)[1]) for line in text.splitlines() if line.startswith("litellm_requests_metric_total")
    )


def s3_object_count(endpoint: str) -> int:
    client: Final = boto3.client(
        "s3", endpoint_url=endpoint, aws_access_key_id="bench", aws_secret_access_key="bench", region_name="us-east-1"
    )
    pages: Final = client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET)
    return sum(page.get("KeyCount", 0) for page in pages)


def wait_for_s3(endpoint: str, expected: int, timeout_s: float) -> int:
    deadline: Final = time.monotonic() + timeout_s
    while True:
        count = s3_object_count(endpoint)
        if count >= expected or time.monotonic() >= deadline:
            return count
        time.sleep(2)


def proxy_env(variant: Variant, run_dir: Path) -> dict[str, str]:
    inherited: Final = {k: v for k, v in os.environ.items() if not k.startswith(STRIPPED_ENV_PREFIXES)}
    multiproc: Final = run_dir / "prometheus"
    multiproc.mkdir()
    return {
        **inherited,
        "LITELLM_MASTER_KEY": MASTER_KEY,
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_LOG": "ERROR",
        "PROMETHEUS_MULTIPROC_DIR": str(multiproc),
        "BENCH_S3_URL": f"http://127.0.0.1:{S3_PORT}",
        "BENCH_MOCK_ANTHROPIC_URL": f"http://127.0.0.1:{MOCK_PORT}",
        "BENCH_MOCK_OPENAI_URL": f"http://127.0.0.1:{MOCK_PORT}/v1",
        **variant.env,
    }


def harness_python() -> str:
    return str(Path(os.environ.get("BENCH_HARNESS_PYTHON", shutil.which("python3") or "python3")))


def start(
    command: Sequence[str], *, cores: str, cwd: Path, env: Mapping[str, str], log: Path
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        ["taskset", "-c", cores, *command],
        cwd=cwd,
        env=dict(env),
        stdout=log.open("wb"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        _ = process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        _ = process.wait(timeout=10)


@dataclass(frozen=True, slots=True)
class RunSpec:
    variant_name: str
    variant: Variant
    sha: str
    worktree: Path
    plan: Path
    run_dir: Path
    cores: Cores
    latency: Latency
    users: int
    profile_seconds: float | None


def run_once(spec: RunSpec) -> dict[str, object]:
    busy: Final = busy_ports((PROXY_PORT, MOCK_PORT, S3_PORT))
    if busy:
        raise RuntimeError(f"ports {busy} are already in use, another benchmark run is probably still going")
    spec.run_dir.mkdir(parents=True)
    harness_env: Final = {k: v for k, v in os.environ.items() if not k.startswith(STRIPPED_ENV_PREFIXES)}
    python: Final = harness_python()
    s3: Final = start(
        [str(Path(python).with_name("moto_server")), "-p", str(S3_PORT)],
        cores=spec.cores.services,
        cwd=spec.run_dir,
        env=harness_env,
        log=spec.run_dir / "s3.log",
    )
    mock: Final = start(
        [python, "server.py", MOCK_KIND[spec.variant.path], "--host", "127.0.0.1", "--port", str(MOCK_PORT)],
        cores=spec.cores.services,
        cwd=MOCKS_DIR,
        env=harness_env,
        log=spec.run_dir / "mock.log",
    )
    proxy: subprocess.Popen[bytes] | None = None
    try:
        s3_url: Final = f"http://127.0.0.1:{S3_PORT}"
        mock_url: Final = f"http://127.0.0.1:{MOCK_PORT}"
        wait_until_up(f"{s3_url}/moto-api/", s3, 60)
        boto3.client(
            "s3", endpoint_url=s3_url, aws_access_key_id="bench", aws_secret_access_key="bench", region_name="us-east-1"
        ).create_bucket(Bucket=BUCKET)
        wait_until_up(f"{mock_url}/health", mock, 60)
        _ = _http("POST", f"{mock_url}/mock/latency", asdict(spec.latency))
        proxy = start(
            [
                str(spec.worktree / ".venv" / "bin" / "python"),
                "litellm/proxy/proxy_cli.py",
                "--config",
                str(CONFIG),
                "--port",
                str(PROXY_PORT),
                "--num_workers",
                str(spec.cores.workers),
            ],
            cores=spec.cores.proxy,
            cwd=spec.worktree,
            env=proxy_env(spec.variant, spec.run_dir),
            log=spec.run_dir / "proxy.log",
        )
        proxy_url: Final = f"http://127.0.0.1:{PROXY_PORT}"
        wait_until_up(f"{proxy_url}/health/liveliness", proxy, 300)
        time.sleep(5)
        requests_before: Final = prometheus_requests(proxy_url)
        records_path: Final = spec.run_dir / "records.jsonl"
        records_path.touch()
        plan_items: Final = json.loads(spec.plan.read_bytes())
        peak_rps: Final = max(1.0, len(plan_items) / max(1.0, float(plan_items[-1]["send_at_s"])))
        loadgen: Final = start(
            [
                python,
                "-P",
                "-m",
                "locust",
                "--headless",
                "--only-summary",
                "--locustfile",
                str(LOCUSTFILE),
                "--host",
                proxy_url,
                "--users",
                str(spec.users),
                "--spawn-rate",
                str(spec.users),
                "--run-time",
                f"{int(plan_items[-1]['send_at_s'] + 600)}s",
                "--exit-code-on-error",
                "0",
            ],
            cores=spec.cores.loadgen,
            cwd=spec.run_dir,
            env={
                **harness_env,
                "PYTHONPATH": str(PERF_DIR),
                "PERF_PLAN_FILE": str(spec.plan),
                "PERF_RECORDS_FILE": str(records_path),
                "PERF_RPS": f"{peak_rps:.3f}",
                "PERF_USERS": str(spec.users),
                "PERF_REQUEST_TIMEOUT": "180",
            },
            log=spec.run_dir / "locust.log",
        )
        launched_at: Final = time.time()
        sampler: Final = Sampler(proxy.pid, loadgen.pid)
        sampler.start()
        profiler: subprocess.Popen[bytes] | None = None
        profile_started_at: float | None = None
        if spec.profile_seconds is not None:
            warmup_s: Final = min(float(item["send_at_s"]) for item in plan_items if item["seq"] < 900_000)
            time.sleep(max(0.0, warmup_s + 5 - (time.time() - launched_at)))
            profile_started_at = time.time()
            profiler = start(
                [
                    str(Path(python).with_name("py-spy")),
                    "record",
                    "--native",
                    "--subprocesses",
                    "--rate",
                    str(PROFILE_RATE_HZ),
                    "--format",
                    "raw",
                    "-o",
                    str(spec.run_dir / "profile.folded"),
                    "--pid",
                    str(proxy.pid),
                    "--duration",
                    str(int(spec.profile_seconds)),
                ],
                cores=spec.cores.services,
                cwd=spec.run_dir,
                env=harness_env,
                log=spec.run_dir / "py-spy.log",
            )
        _ = loadgen.wait()
        if profiler is not None:
            _ = profiler.wait(timeout=spec.profile_seconds + 120 if spec.profile_seconds else 120)
        sampler.stopped.set()
        sampler.join()
        records: Final = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        ok_count: Final = sum(1 for record in records if record["ok"])
        s3_objects: Final = wait_for_s3(s3_url, ok_count, timeout_s=120)
        time.sleep(3)
        requests_after: Final = prometheus_requests(proxy_url)
        mock_log: Final = MockLog.model_validate_json(_http("GET", f"{mock_url}/mock/requests", timeout=60))
        meta: Final = {
            "variant": spec.variant_name,
            "sha": spec.sha,
            "path": spec.variant.path,
            "env": spec.variant.env,
            "expect_rust": spec.variant.expect_rust,
            "plan": str(spec.plan),
            "cores": asdict(spec.cores),
            "latency": asdict(spec.latency),
            "users": spec.users,
            "launched_at": launched_at,
            "records": len(records),
            "ok": ok_count,
            "prometheus_requests_delta": requests_after - requests_before,
            "s3_objects": s3_objects,
            "locust_exit": loadgen.returncode,
            "profiled": profiler is not None,
            "profile_rate_hz": PROFILE_RATE_HZ,
            "profile_requests": sum(
                1
                for record in records
                if record["ok"]
                and profile_started_at is not None
                and spec.profile_seconds is not None
                and profile_started_at <= record["started_at"] <= profile_started_at + spec.profile_seconds
            ),
        }
        _ = (spec.run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        _ = (spec.run_dir / "samples.json").write_text(json.dumps([asdict(sample) for sample in sampler.samples]))
        _ = (spec.run_dir / "mock.json").write_text(mock_log.model_dump_json())
        return meta
    finally:
        if proxy is not None:
            stop(proxy)
        stop(mock)
        stop(s3)


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--worktrees", type=Path, required=True)
    _ = parser.add_argument("--plans", type=Path, required=True, help="dir with <path>.json plans")
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument("--variants", required=True, help="comma separated names from variants.yaml")
    _ = parser.add_argument("--reps", type=int, default=5)
    _ = parser.add_argument("--seed", type=int, default=0, help="shuffles the run order")
    _ = parser.add_argument("--plan-name", default="{path}", help="plan file stem, {path} is substituted")
    _ = parser.add_argument("--users", type=int, default=400)
    _ = parser.add_argument("--profile-seconds", type=float)
    _ = parser.add_argument("--proxy-cores", default="0,1")
    _ = parser.add_argument("--loadgen-cores", default="2")
    _ = parser.add_argument("--service-cores", default="3")
    _ = parser.add_argument("--workers", type=int, default=2)
    _ = parser.add_argument("--first-byte-s", type=float, default=0.8)
    _ = parser.add_argument("--first-byte-jitter-s", type=float, default=0.2)
    _ = parser.add_argument("--token-interval-s", type=float, default=0.0125)
    args: Final = parser.parse_args()
    catalog: Final = load_variants()
    names: Final = tuple(str(args.variants).split(","))  # pyright: ignore[reportAny]
    runs: Final = [(name, rep) for rep in range(1, int(args.reps) + 1) for name in names]  # pyright: ignore[reportAny]  # mutable-ok: shuffled in place below
    random.Random(int(args.seed)).shuffle(runs)  # pyright: ignore[reportAny]
    cores: Final = Cores(
        proxy=str(args.proxy_cores),
        loadgen=str(args.loadgen_cores),  # pyright: ignore[reportAny]
        services=str(args.service_cores),
        workers=int(args.workers),  # pyright: ignore[reportAny]
    )
    latency: Final = Latency(
        seconds=float(args.first_byte_s),
        jitter_seconds=float(args.first_byte_jitter_s),  # pyright: ignore[reportAny]
        token_interval_seconds=float(args.token_interval_s),  # pyright: ignore[reportAny]
    )
    out: Final[Path] = args.out  # pyright: ignore[reportAny]
    for index, (name, rep) in enumerate(runs, start=1):
        variant = catalog.variants[name]
        worktree = Path(args.worktrees) / variant.worktree  # pyright: ignore[reportAny]
        plan = Path(args.plans) / f"{str(args.plan_name).format(path=variant.path)}.json"  # pyright: ignore[reportAny]
        run_dir = out / f"{name}-r{rep}"
        _ = sys.stdout.write(f"[{index}/{len(runs)}] {name} rep {rep} -> {run_dir}" + "\n")
        try:
            meta = run_once(
                RunSpec(
                    variant_name=name,
                    variant=variant,
                    sha=catalog.commits[variant.worktree],
                    worktree=worktree,
                    plan=plan,
                    run_dir=run_dir,
                    cores=cores,
                    latency=latency,
                    users=int(args.users),  # pyright: ignore[reportAny]
                    profile_seconds=args.profile_seconds,  # pyright: ignore[reportAny]
                )
            )
        except Exception as error:  # noqa: BLE001  # one failed run is recorded and the matrix moves on
            run_dir.mkdir(parents=True, exist_ok=True)
            _ = (run_dir / "error.txt").write_text(f"{type(error).__name__}: {error}\n")
            _ = sys.stdout.write(f"FAILED {name} rep {rep}: {error}\n")
            continue
        _ = sys.stdout.write(
            json.dumps({k: meta[k] for k in ("ok", "records", "prometheus_requests_delta", "s3_objects")}) + "\n"
        )


if __name__ == "__main__":
    main()
