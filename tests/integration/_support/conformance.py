import hashlib
import os
import queue
import signal
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import httpx
from pydantic import BaseModel, Field, JsonValue, TypeAdapter

if TYPE_CHECKING:
    from integration._support.mcp import McpPeer


class ConformanceCheck(BaseModel):
    id: str
    status: Literal["SUCCESS", "FAILURE", "WARNING", "INFO"]
    errorMessage: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


def verify_archive(archive: Path, expected_sha256: str) -> None:
    actual: Final = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert actual == expected_sha256, f"SHA-256 mismatch for {archive.name}: {actual}"


def read_checks(directory: Path, scenario: str) -> tuple[ConformanceCheck, ...]:
    reports: Final = tuple(directory.rglob("checks.json"))
    assert len(reports) == 1, f"conformance {scenario}: expected one fresh report, found {len(reports)}"
    checks: Final = TypeAdapter(tuple[ConformanceCheck, ...]).validate_json(reports[0].read_bytes())
    assert len({check.id for check in checks}) == len(checks), f"conformance {scenario}: duplicate checks"
    assert tuple(check.status for check in checks if check.id == scenario) == ("SUCCESS",), (
        f"conformance {scenario}: required scenario did not pass: {checks}"
    )
    assert all(check.status != "FAILURE" for check in checks), f"conformance {scenario}: failed checks: {checks}"
    if scenario == "tools-call-simple-text":
        result: Final = next(check for check in checks if check.id == scenario).details.get("result")
        # The official scenario accepts any nonempty text, including tool errors.
        assert isinstance(result, dict) and result.get("isError", False) is False, f"reference payload: {result}"
        assert result.get("content") == [{"type": "text", "text": "This is a simple text response for testing."}], (
            f"reference payload: {result}"
        )
    return checks


def run_scenario(root: Path, url: str, scenario: str, directory: Path) -> tuple[ConformanceCheck, ...]:
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "runner.log").open("w") as log:
        result: Final = subprocess.run(
            [
                "node",
                "--import",
                "tsx",
                "src/index.ts",
                "server",
                "--url",
                url,
                "--scenario",
                scenario,
                "--spec-version",
                "2025-11-25",
                "--output-dir",
                str(directory.resolve()),
                "--timeout",
                "30000",
            ],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=45,
            check=False,
        )
    assert result.returncode == 0, (directory / "runner.log").read_text()
    return read_checks(directory, scenario)


@contextmanager
def reference_server(root: Path, directory: Path, port: int) -> Iterator["McpPeer"]:
    from integration._support.client import eventually
    from integration._support.mcp import McpPeer
    from integration._support.process import signal_group, stop_root_process

    directory.mkdir(parents=True, exist_ok=True)
    url: Final = f"http://127.0.0.1:{port}"
    with (directory / "reference.log").open("w") as log:
        process: Final = subprocess.Popen(
            ["node", "--import", "tsx", "everything-server.ts"],
            cwd=root / "legacy-reference/examples/servers/typescript",
            env={"PATH": os.environ["PATH"], "PORT": str(port)},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            with httpx.Client(timeout=1, trust_env=False) as client:

                def ready() -> bool:
                    assert process.poll() is None, (directory / "reference.log").read_text()
                    try:
                        return client.get(url + "/mcp").status_code == 400
                    except httpx.TransportError:
                        return False

                eventually(ready, bool, seconds=15)
            yield McpPeer(url + "/mcp", queue.Queue())
        finally:
            stopped: Final = stop_root_process(process)
            if not stopped:
                signal_group(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            assert stopped, "Official reference required forced cleanup"


@contextmanager
def authenticated_endpoint(target: str, key: str) -> Iterator[str]:
    from integration._support.asgi import asgi_server
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import StreamingResponse
    from starlette.routing import Mount
    from starlette.types import Receive, Scope, Send

    async def forward(scope: Scope, receive: Receive, send: Send) -> None:
        request: Final = Request(scope, receive)
        headers: Final = tuple(
            (name, value) for name, value in request.headers.raw if name.lower() != b"authorization"
        ) + ((b"authorization", f"Bearer {key}".encode()),)
        body: Final = await request.body()
        async with httpx.AsyncClient(timeout=35, trust_env=False) as client:
            async with client.stream(request.method, target, headers=headers, content=body) as response:
                streamed: Final = StreamingResponse(response.aiter_raw(), status_code=response.status_code)
                streamed.raw_headers = [
                    (name, value)
                    for name, value in response.headers.raw
                    if name.lower() not in (b"transfer-encoding", b"connection")
                ]
                await streamed(scope, receive, send)

    with asgi_server(Starlette(routes=[Mount("/mcp", app=forward)])) as url:
        yield url + "/mcp/"
