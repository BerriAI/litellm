import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Final, Literal

import httpx
import psutil
from bifrost_bench import Counters, FrozenModel, Settings, run_bifrost
from matrix import PROFILE_NAMES
from messages_probe import verify_sdk
from pydantic import Field

ROOT: Final = Path(__file__).resolve().parent
BINARY: Final = ROOT / "target/release/gateway-split-bench"
LOGGER: Final = logging.getLogger(__name__)


class Metrics(Counters):
    upload_bytes: int = 0


class Options(FrozenModel):
    bifrost: Path = ROOT / "runtime/tools/suite"
    suite_root: Path = ROOT / "runtime/tools/bifrost-benchmarking"
    output: Path = ROOT / "results/smoke/trials.jsonl"
    profile: str = "litellm-current"
    sizes: tuple[int, ...] = (262144, 1048576, 4194304)
    concurrencies: tuple[int, ...] = (1, 1000, 2000)
    responses: tuple[bool, ...] = (False, True)
    seconds: int = Field(default=60, ge=1)
    repetitions: int = Field(default=3, ge=1)
    warmup: int = Field(default=5, ge=1)
    seed: int = 20261019
    machine_class: str = "local"


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    port: int
    delegate: bool
    process_names: tuple[str, ...]


VARIANTS: Final = (
    Variant("direct_axum", 19401, False, ("axum",)),
    Variant("direct_fastapi", 19402, False, ("fastapi",)),
    Variant("axum_python", 19403, True, ("axum_dispatch", "fastapi")),
    Variant("fastapi_rust", 19404, False, ("python_dispatch", "axum")),
)


@contextmanager
def launch(name: str, args: Sequence[str], env: Mapping[str, str]) -> Generator[psutil.Process]:
    with (ROOT / "runtime" / f"{name}.log").open("w") as log:
        process: Final = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
        try:
            yield psutil.Process(process.pid)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def probe_ready(port: int) -> bool:
    try:
        with httpx.Client(trust_env=False, timeout=1) as client:
            return client.get(f"http://127.0.0.1:{port}/metrics").is_success
    except httpx.HTTPError:
        return False


def ready(port: int) -> None:
    for _ in range(100):
        if probe_ready(port):
            return
        time.sleep(0.1)
    sys.exit(f"Listener {port} failed, inspect runtime logs")


def require_free_ports() -> None:
    for port in range(19400, 19405):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener.bind(("127.0.0.1", port))
            except OSError:
                sys.exit(f"Port {port} is occupied, stop the conflicting listener before running")


def metrics(client: httpx.Client) -> Metrics:
    return Metrics.model_validate_json(client.get("http://127.0.0.1:19400/metrics").content)


def verify(variant: Variant, client: httpx.Client, python_body: str) -> Mapping[str, str | int | float | bool]:
    verify_sdk(variant.port, variant.delegate)
    headers: Final = {"x-bench-id": "probe", "content-type": "application/json"} | (
        {"x-bench-python": "1"} if variant.delegate else {}
    )
    payload: Final = {"model": "fixture-model", "max_tokens": 32, "messages": [{"role": "user", "content": "hello"}]}
    data: Final = json.dumps(payload).encode()
    url: Final = f"http://127.0.0.1:{variant.port}/v1/messages?beta=true"
    response: Final = client.post(url, content=data, headers=headers)
    assert response.status_code == 200
    assert response.headers["x-body-sha256"] == hashlib.sha256(data).hexdigest()
    assert response.headers["x-bench-id"] == "probe"
    before_upload: Final = metrics(client)

    def upload() -> Iterator[bytes]:
        yield data[:1]
        time.sleep(0.1)
        observed: Final = metrics(client).upload_bytes - before_upload.upload_bytes
        assert (observed > 0) == (variant.name == "direct_axum" or python_body == "stream")
        yield data[1:]

    upload_response: Final = client.post(
        url, content=upload(), headers=headers | {"x-bench-upload-probe": "1", "content-length": str(len(data))}
    )
    assert upload_response.status_code == 200
    assert upload_response.headers["x-body-sha256"] == hashlib.sha256(data).hexdigest()
    before_errors: Final = metrics(client)
    for status in (429, 500):
        verify_error(client, url, data, headers, status)
    after_errors: Final = metrics(client)
    assert after_errors.requests - before_errors.requests == 2
    before_stream: Final = metrics(client)
    start: Final = time.perf_counter()
    with client.stream(
        "POST",
        url,
        json=payload | {"stream": True},
        headers=headers | {"x-bench-events": "10000", "x-bench-delay-ms": "10"},
    ) as stream:
        assert stream.status_code == 200
        first: Final = next(line for line in stream.iter_lines() if line.startswith("data:"))
        assert "message_start" in first
    time.sleep(0.25)
    after_stream: Final = metrics(client)
    assert after_stream.active == before_stream.active
    assert after_stream.cancelled - before_stream.cancelled == 1
    return {
        "variant": variant.name,
        "body_digest": True,
        "anthropic_sdk_messages": True,
        "slow_upload_verified": True,
        "headers": True,
        "status_passthrough": True,
        "error_backend_attempts": after_errors.requests - before_errors.requests,
        "cancelled_upstream": True,
        "cancel_observation_ms": (time.perf_counter() - start) * 1000,
    }


def verify_error(client: httpx.Client, url: str, data: bytes, headers: Mapping[str, str], status: int) -> None:
    failed: Final = client.post(url, content=data, headers=dict(headers) | {"x-bench-status": str(status)})
    assert failed.status_code == status
    assert failed.text == "injected failure"


class HostMemory(FrozenModel):
    total: int


class CLI(Options):
    response: Literal["both", "buffered", "sse"]


def run(options: Options) -> None:
    (ROOT / "runtime").mkdir(exist_ok=True)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with (Path(tempfile.gettempdir()) / "litellm-gateway-benchmark.lock").open("a+") as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_free_ports()
        env: Final = dict(os.environ) | {
            "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT.parents[2]))),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LITELLM_HTTP2": "False",
            "DISABLE_AIOHTTP_TRANSPORT": "False",
            "NO_PROXY": "127.0.0.1,localhost",
            "BENCH_STREAM_REQUESTS": "0",
        }

        def start(name: str, args: Sequence[str], dispatch: bool = False) -> psutil.Process:
            return stack.enter_context(launch(name, args, env | {"BENCH_DISPATCH": "1" if dispatch else "0"}))

        def python_server(name: str, port: int, dispatch: bool = False) -> psutil.Process:
            return start(
                name,
                (
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "python_app:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--loop",
                    "uvloop",
                    "--http",
                    "auto",
                    "--workers",
                    "1",
                    "--no-access-log",
                    "--timeout-keep-alive",
                    "60",
                ),
                dispatch,
            )

        processes: Final = {
            "backend": start("backend", (str(BINARY), "serve", "--port", "19400")),
            "axum": start("axum", (str(BINARY), "serve", "--role", "stream", "--port", "19401")),
            "fastapi": python_server("fastapi", 19402),
            "axum_dispatch": start("axum_dispatch", (str(BINARY), "serve", "--role", "stream", "--port", "19403")),
            "python_dispatch": python_server("python_dispatch", 19404, True),
        }
        for port in range(19400, 19405):
            ready(port)
        with httpx.Client(trust_env=False, timeout=5) as client:
            probes: Final = tuple(verify(variant, client, "buffered") for variant in VARIANTS)
            options.output.with_suffix(".probes.json").write_text(json.dumps(probes, indent=2) + "\n")
        metadata: Final = {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpus": psutil.cpu_count(),
            "memory_bytes": HostMemory.model_validate(psutil.virtual_memory(), from_attributes=True).total,
            "machine_class": options.machine_class,
            "python": sys.version,
            "git_sha": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "fastapi",
                    "starlette",
                    "uvicorn",
                    "uvloop",
                    "httpx",
                    "aiohttp",
                    "psutil",
                    "pydantic",
                )
            },
            "rust": subprocess.check_output(("rustc", "--version"), text=True).strip(),
            "parameters": options.model_dump(mode="json"),
            "resolved_http_parser": "httptools" if importlib.util.find_spec("httptools") else "h11",
            "dependency_lock_sha256": hashlib.sha256((ROOT.parents[2] / "uv.lock").read_bytes()).hexdigest(),
            "harness_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in tuple(ROOT.glob("*.py"))
                + tuple((ROOT / "src").glob("*.rs"))
                + (ROOT / "Cargo.lock", ROOT / "Cargo.toml")
            },
            "bifrost_binary_sha256": hashlib.sha256(options.bifrost.read_bytes()).hexdigest(),
            "bifrost_patch_sha256": hashlib.sha256((ROOT / "bifrost.patch").read_bytes()).hexdigest(),
        }
        options.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        run_bifrost(
            processes,
            VARIANTS,
            Settings(
                binary=options.bifrost.resolve(),
                suite_root=options.suite_root.resolve(),
                output=options.output,
                seconds=options.seconds,
                repetitions=options.repetitions,
                warmup_seconds=options.warmup,
                python_body="buffered",
                rust_client="hyper-stream",
                python_client="litellm-aiohttp",
                http_parser="httptools" if importlib.util.find_spec("httptools") else "h11",
                event_loop="uvloop",
                profile=options.profile,
                seed=options.seed,
                sizes=options.sizes,
                concurrencies=options.concurrencies,
                responses=options.responses,
            ),
        )


def stop(signum: int, frame: FrameType | None) -> None:
    sys.exit(128 + signum)


def main() -> None:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--bifrost", type=Path, default=ROOT / "runtime/tools/suite")
    parser.add_argument("--suite-root", type=Path, default=ROOT / "runtime/tools/bifrost-benchmarking")
    parser.add_argument("--output", type=Path, default=ROOT / "results/smoke/trials.jsonl")
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="litellm-current")
    parser.add_argument(
        "--sizes", type=int, nargs="+", choices=(262144, 1048576, 4194304), default=(262144, 1048576, 4194304)
    )
    parser.add_argument("--concurrencies", type=int, nargs="+", choices=(1, 1000, 2000), default=(1, 1000, 2000))
    parser.add_argument("--response", choices=("both", "buffered", "sse"), default="both")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--machine-class", default="local")
    raw: Final = CLI.model_validate(vars(parser.parse_args()))
    response: Final = raw.response
    options: Final = Options.model_validate(
        raw.model_dump(exclude={"response"})
        | {"responses": (False, True) if response == "both" else (response == "sse",)}
    )
    signal.signal(signal.SIGTERM, stop)
    run(options)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
