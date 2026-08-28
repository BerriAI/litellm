from __future__ import annotations

import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import SimpleQueue
from socketserver import BaseServer
from typing import Final, Protocol, TextIO, cast

import psutil
import pyperf  # pyright: ignore[reportMissingTypeStubs]  # Upstream does not publish typing stubs
from pydantic import BaseModel, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, object])


class MemoryInfo(BaseModel):
    rss: int
    uss: int | None = None


def memory_info(pid: int) -> MemoryInfo:
    process: Final = psutil.Process(pid)
    try:
        return MemoryInfo.model_validate(process.memory_full_info(), from_attributes=True)
    except (psutil.AccessDenied, AttributeError):
        return MemoryInfo.model_validate(process.memory_info(), from_attributes=True)


class BenchmarkValues(Protocol):
    def get_values(self) -> Sequence[float]: ...


PROBE: Final = r"""
import sys
import time

def guard(event, args):
    if event in ("socket.connect", "socket.sendto"):
        address = args[-1]
        allowed = isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1")
    elif event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
        allowed = args[0] in ("127.0.0.1", "::1", "localhost")
    else:
        return
    if not allowed:
        import os
        sys.stderr.write("BENCH_EGRESS_BLOCKED\n")
        sys.stderr.flush()
        os._exit(1)

sys.addaudithook(guard)
mode = sys.argv[1]
if mode == "diagnostic":
    import json
    baseline = frozenset(sys.modules)

def stage(name):
    if mode == "diagnostic":
        print("BENCH:" + json.dumps({"stage": name, "modules": sorted(set(sys.modules) - baseline)}), flush=True)
        if sys.stdin.readline() != "continue\n":
            sys.exit("Missing diagnostic acknowledgement")

started = time.perf_counter_ns()
import litellm
imported = time.perf_counter_ns()

if mode == "import_exit":
    sys.exit(0)

stage("after_import")
litellm.telemetry = False
arguments = dict(
    model="openai/benchmark-model",
    messages=[{"role": "user", "content": "Hi"}],
    api_base=sys.argv[2],
    api_key="benchmark-dummy-key",
    num_retries=0,
    timeout=10,
    max_tokens=1,
)
configured = time.perf_counter_ns()
stage("after_configuration")
first_started = time.perf_counter_ns()
first = litellm.completion(**arguments)
first_finished = time.perf_counter_ns()
if first.choices[0].message.content != "ok":
    sys.exit("Unexpected first response")
stage("after_first_response")
second_started = time.perf_counter_ns()
second = litellm.completion(**arguments)
second_finished = time.perf_counter_ns()
if second.choices[0].message.content != "ok":
    sys.exit("Unexpected second response")
stage("after_second_response")
if mode != "diagnostic":
    import json
    print("BENCH:" + json.dumps(dict(
        import_ns=imported-started,
        configuration_ns=configured-imported,
        first_request_ns=first_finished-first_started,
        second_request_ns=second_finished-second_started,
        import_to_first_response_ns=first_finished-started,
        imported_at_ns=imported,
        first_response_at_ns=first_finished,
    )))
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        sys.exit(message)


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def command(
    arguments: Sequence[str], directory: Path, environment: Mapping[str, str], log: TextIO, timeout: float = 900
) -> str:
    log.write(f"\n$ {arguments!r}\n")
    log.flush()
    with subprocess.Popen(
        arguments,
        cwd=directory,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process(process)
            process.communicate()
            sys.exit(f"Command timed out after {timeout}s; see {log.name}")
        log.write(output)
        log.flush()
        require(process.returncode == 0, f"Command failed ({process.returncode}); see {log.name}")
        return output


def runtime_environment(home: Path) -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "HOME": str(home),
        "TMPDIR": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_LOG": "ERROR",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "AWS_EC2_METADATA_DISABLED": "true",
    }


class ProviderHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def __init__(
        self,
        request: socket.socket,
        client_address: tuple[str, int],
        server: BaseServer,
        *,
        requests: SimpleQueue[bool],
    ) -> None:
        self.requests = requests
        super().__init__(request, client_address, server)

    def log_message(self, format: str, *args: object) -> None:
        return

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        self.requests.put(code == 200)

    def do_POST(self) -> None:
        body: Final = JSON_OBJECT.validate_json(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        valid: Final = (
            self.path == "/v1/chat/completions"
            and self.headers.get("Authorization") == "Bearer benchmark-dummy-key"
            and body.get("model") == "benchmark-model"
            and body.get("messages") == [{"role": "user", "content": "Hi"}]
            and not body.get("stream", False)
        )
        if not valid:
            self.send_error(400, "Unexpected benchmark request")
            return
        payload: Final = json.dumps(
            {
                "id": "chatcmpl-benchmark",
                "object": "chat.completion",
                "created": 0,
                "model": "benchmark-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def provider() -> Generator[tuple[str, SimpleQueue[bool]]]:
    requests: Final[SimpleQueue[bool]] = SimpleQueue()
    with ThreadingHTTPServer(("127.0.0.1", 0), partial(ProviderHandler, requests=requests)) as server:
        thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", requests
        finally:
            server.shutdown()
            thread.join()


def summary(values: Sequence[float], unit: str) -> dict[str, object]:
    require(bool(values), "Cannot summarize an empty sample")
    median: Final = statistics.median(values)
    return {
        "unit": unit,
        "samples": tuple(values),
        "count": len(values),
        "median": median,
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
        "mad": statistics.median(abs(value - median) for value in values),
        "min": min(values),
        "max": max(values),
    }


def probe(
    python: Path,
    base_url: str,
    directory: Path,
    environment: Mapping[str, str],
    log: TextIO,
    requests: SimpleQueue[bool],
    *,
    diagnostic: bool = False,
    timeout: float = 120,
) -> tuple[dict[str, int], tuple[dict[str, object], ...]]:
    require(requests.empty(), "Unexpected provider request between probes")
    mode: Final = "diagnostic" if diagnostic else "timing"
    launched: Final = time.perf_counter_ns()
    with subprocess.Popen(
        (str(python), "-I", "-B", "-c", PROBE, mode, base_url),
        cwd=directory,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        start_new_session=True,
    ) as process:
        timer: Final = threading.Timer(timeout, stop_process, args=(process,))
        timer.start()
        try:
            if process.stdout is None or process.stdin is None:
                sys.exit("Probe pipes unavailable")
            incoming: Final = process.stdout
            outgoing: Final = process.stdin

            def record(line: str) -> dict[str, object]:
                payload: Final = JSON_OBJECT.validate_json(line.removeprefix("BENCH:"))
                if diagnostic:
                    memory: Final = memory_info(process.pid)
                    return acknowledge({**payload, "rss_bytes": memory.rss, "uss_bytes": memory.uss})
                return payload

            def acknowledge(payload: dict[str, object]) -> dict[str, object]:
                outgoing.write("continue\n")
                outgoing.flush()
                return payload

            lines: Final = cast(Iterator[str], iter(incoming.readline, ""))
            records: Final = tuple(record(line) for line in lines if line.startswith("BENCH:"))
            process.wait()
            require(process.returncode == 0, f"Probe failed or timed out; see {log.name}")
            require(requests.qsize() == 2, f"Expected two provider requests, received {requests.qsize()}")
            require(all(requests.get() for _ in range(2)), "Provider rejected a request")
            if diagnostic:
                require(len(records) == 4, "Incomplete diagnostic probe")
                return {}, records
            require(len(records) == 1, "Missing timing record")
            timings: Final = {key: int(value) for key, value in records[0].items() if isinstance(value, int)}
            return {
                **{key: value for key, value in timings.items() if not key.endswith("_at_ns")},
                "launch_to_import_ns": timings["imported_at_ns"] - launched,
                "launch_to_first_response_ns": timings["first_response_at_ns"] - launched,
            }, ()
        finally:
            timer.cancel()
            stop_process(process)
            process.wait()


def startup(
    python: Path,
    output: Path,
    directory: Path,
    environment: Mapping[str, str],
    log: TextIO,
    samples: int,
    timeout: float,
) -> dict[str, object]:
    for name, code, arguments in (
        ("python_startup_exit", "pass", ()),
        ("import_process_exit", f"exec({PROBE!r})", ("import_exit",)),
    ):
        command(
            (
                sys.executable,
                "-m",
                "pyperf",
                "command",
                "--copy-env",
                "--name",
                name,
                "--processes",
                str(samples),
                "--values",
                "1",
                "--loops",
                "1",
                "--warmups",
                "1",
                "--timeout",
                str(int(timeout)),
                "-o",
                str(output / f"{name}.json"),
                "--",
                str(python),
                "-I",
                "-B",
                "-c",
                code,
                *arguments,
            ),
            directory,
            environment,
            log,
            timeout * (samples + 1),
        )
    return {
        name: summary(load_pyperf(output / f"{name}.json").get_values(), "seconds")
        for name in ("python_startup_exit", "import_process_exit")
    }


def load_pyperf(path: Path) -> BenchmarkValues:
    return cast(BenchmarkValues, pyperf.Benchmark.load(str(path)))  # pyright: ignore[reportUnknownMemberType]  # Untyped upstream API
