import hashlib
import json
import os
import queue
import signal
import subprocess
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import httpx
import psutil
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

if TYPE_CHECKING:
    from litellm.proxy._experimental.mcp_server.capabilities import RevisionSupport

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
    required: Final = {
        "server-session-lifecycle": (
            "server-session-initialized-accepted",
            "server-session-delete-accepted",
            "server-session-terminated-returns-404",
        ),
        "server-sse-multiple-streams": (
            "server-accepts-multiple-post-streams",
            "server-sse-streams-functional",
        ),
        "server-initialize": ("server-initialize", "server-session-id-visible-ascii", "wire-schema-valid"),
        "tools-list": ("tools-list", "tools-name-format", "wire-schema-valid"),
        "tools-call-image": ("tools-call-image", "wire-schema-valid"),
    }.get(scenario, (scenario, "wire-schema-valid") if scenario in OFFICIAL_SCENARIOS else (scenario,))
    for identity in required:
        assert tuple(check.status for check in checks if check.id == identity) == ("SUCCESS",), (
            f"conformance {scenario}: required check {identity} did not pass: {checks}"
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
    from integration._support.process import group_members, signal_group, stop_root_process

    directory.mkdir(parents=True, exist_ok=True)
    url: Final = f"http://127.0.0.1:{port}"
    with (directory / "reference.log").open("w") as log:
        process: Final = subprocess.Popen(
            ["node", "--import", "tsx", "everything-server.ts"],
            cwd=root / "examples/servers/typescript",
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
            residual: Final = group_members(process.pid)
            if residual:
                signal_group(process.pid, signal.SIGTERM)
                psutil.wait_procs(residual, timeout=5)
            remaining: Final = group_members(process.pid)
            if remaining:
                signal_group(process.pid, signal.SIGKILL)
                psutil.wait_procs(remaining, timeout=3)
            process.wait(timeout=3)
            assert not group_members(process.pid), "Official reference child survived cleanup"
            assert stopped and not remaining, "Official reference required forced cleanup"


@contextmanager
def authenticated_endpoint(
    target: str,
    key: str | None,
    alias: str | None,
    negotiations: queue.Queue[tuple[str, str]] | None = None,
) -> Iterator[str]:
    from integration._support.asgi import asgi_server
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import StreamingResponse
    from starlette.routing import Mount
    from starlette.types import Receive, Scope, Send

    async def forward(scope: Scope, receive: Receive, send: Send) -> None:
        request: Final = Request(scope, receive)
        original: Final = await request.body()
        body: Final = prefixed_request(original, alias) if alias is not None else original
        headers: Final = tuple(
            (name, value)
            for name, value in request.headers.raw
            if (key is None or name.lower() != b"authorization")
            and (name.lower() != b"content-length" or body == original)
        ) + (((b"authorization", f"Bearer {key}".encode()),) if key is not None else ())
        async with httpx.AsyncClient(timeout=35, trust_env=False) as client:
            async with client.stream(request.method, target, headers=headers, content=body) as response:

                async def observed_chunks() -> AsyncIterator[bytes]:
                    capture: Final = (
                        negotiations is not None and response.status_code == 200 and is_initialization(body)
                    )
                    chunks: Final[list[bytes]] = []
                    async for chunk in response.aiter_raw():
                        if capture:
                            chunks.append(chunk)
                        yield chunk
                    if capture and negotiations is not None:
                        negotiations.put(read_negotiation(body, b"".join(chunks), response.headers["content-type"]))

                streamed: Final = StreamingResponse(observed_chunks(), status_code=response.status_code)
                streamed.raw_headers = [
                    (name, value)
                    for name, value in response.headers.raw
                    if name.lower() not in (b"transfer-encoding", b"connection")
                ]
                await streamed(scope, receive, send)

    with asgi_server(Starlette(routes=[Mount("/mcp", app=forward)])) as url:
        yield url + "/mcp/"


def prefixed_request(body: bytes, alias: str) -> bytes:
    try:
        request: Final = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(request, dict) or request.get("method") not in ("tools/call", "prompts/get"):
        return body
    params: Final = request.get("params")
    if not isinstance(params, dict) or not isinstance(params.get("name"), str):
        return body
    return json.dumps({**request, "params": {**params, "name": f"{alias}-{params['name']}"}}).encode()


def require_negotiations(expected: str, observed: tuple[tuple[str, str], ...]) -> None:
    assert observed and all(pair == (expected, expected) for pair in observed), (
        f"conformance negotiation: expected {expected} on the wire, observed {observed}"
    )


class ExecutionReport(BaseModel):
    collected: tuple[str, ...]
    passed: tuple[str, ...]
    skipped: tuple[str, ...]
    complete: bool


def require_passes(directory: Path, required: tuple[str, ...]) -> None:
    report: Final = ExecutionReport.model_validate_json((directory / "execution.json").read_bytes())
    assert required and report.complete, "conformance execution was incomplete"
    missing: Final = set(required) - set(report.collected)
    assert not missing, f"conformance cases were not collected: {sorted(missing)}"
    unsuccessful: Final = set(required) - set(report.passed)
    assert not unsuccessful, f"conformance cases did not pass: {sorted(unsuccessful)}"
    skipped: Final = set(required).intersection(report.skipped)
    assert not skipped, f"conformance cases were skipped: {sorted(skipped)}"


def translation_cases() -> tuple[tuple[str, str, str, str], ...]:
    from litellm.proxy._experimental.mcp_server.capabilities import REVISION_SUPPORT, TRANSLATION_PAIRS
    from litellm.types.mcp import MCP_LEGACY_VERSIONS, MCPTransport

    completed: Final = {revision for revision, support in REVISION_SUPPORT.items() if support.completed}
    assert completed == set(MCP_LEGACY_VERSIONS), "New revisions need conformance cases before activation"
    assert TRANSLATION_PAIRS == frozenset(product(completed, repeat=2)), "Declared translation coverage changed"
    return tuple(
        (downstream, upstream, peer, ingress)
        for (downstream, upstream), peer, ingress in product(
            sorted(TRANSLATION_PAIRS), ("http", "sse", "stdio"), ("http", "sse")
        )
        if MCPTransport(peer) in REVISION_SUPPORT[upstream].transports
        and MCPTransport(ingress) in REVISION_SUPPORT[downstream].transports
    )


# Applicable legacy server scenarios for the operations advertised by capabilities.py.
# The simple-text runner/reference mismatch has an explicit SDK gap case instead.
OFFICIAL_SCENARIOS: Final = (
    "server-initialize",
    "server-sse-multiple-streams",
    "ping",
    "tools-list",
    "tools-call-image",
    "tools-call-audio",
    "tools-call-embedded-resource",
    "tools-call-mixed-content",
    "tools-call-error",
    "tools-call-with-progress",
    "resources-list",
    "resources-read-text",
    "resources-read-binary",
    "resources-templates-read",
    "prompts-list",
    "prompts-get-simple",
    "prompts-get-with-args",
    "prompts-get-embedded-resource",
    "prompts-get-with-image",
)


def official_cases() -> tuple[tuple[str, str], ...]:
    from litellm.types.mcp import MCP_LEGACY_VERSIONS

    return tuple(product(OFFICIAL_SCENARIOS, MCP_LEGACY_VERSIONS))


class CapabilityContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revisions: tuple[str, ...] = Field(min_length=1)
    operations: tuple[str, ...]
    extensions: tuple[str, ...]
    tests: tuple[str, ...] = Field(min_length=1)


class ConformanceBaseline(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revisions: tuple[str, ...] = Field(min_length=1)
    ingress_transports: tuple[str, ...] = Field(min_length=1)
    upstream_transports: tuple[str, ...] = Field(min_length=1)
    official_scenarios: tuple[str, ...] = Field(min_length=1)
    capabilities: tuple[CapabilityContract, ...] = Field(min_length=1)
    required_tests: tuple[str, ...] = Field(min_length=1)


def require_capability_coverage(
    baseline: ConformanceBaseline,
    required: tuple[str, ...],
    support: Mapping[str, "RevisionSupport"],
) -> None:
    families: Final = {node.split("[", 1)[0] for node in required}
    for contract in baseline.capabilities:
        assert set(contract.revisions) <= set(baseline.revisions), "Conformance capability maps an untested revision"
        assert set(contract.tests) <= families, "Conformance capability maps a test outside the required baseline"
    for revision, advertised in support.items():
        if not advertised.completed:
            continue
        contracts: Final = tuple(contract for contract in baseline.capabilities if revision in contract.revisions)
        assert contracts, f"Conformance capability coverage missing for revision {revision}"
        operations: Final = {operation for contract in contracts for operation in contract.operations}
        extensions: Final = {extension for contract in contracts for extension in contract.extensions}
        assert advertised.operations <= operations, (
            f"Conformance capability coverage missing for {revision}: {sorted(advertised.operations - operations)}"
        )
        assert advertised.extensions <= extensions, (
            f"Conformance capability coverage missing for {revision}: {sorted(advertised.extensions - extensions)}"
        )


def required_conformance_nodes() -> tuple[str, ...]:
    from litellm.proxy._experimental.mcp_server.capabilities import REVISION_SUPPORT

    baseline: Final = ConformanceBaseline.model_validate_json(
        (Path(__file__).resolve().parents[1] / "mcp/conformance_baseline.json").read_bytes()
    )
    official: Final = tuple(
        "tests/integration/mcp/test_mcp_official_conformance.py::test_official_scenario_through_gateway["
        + "-".join(case)
        + "]"
        for case in product(baseline.official_scenarios, baseline.revisions)
    )
    matrix: Final = tuple(
        "tests/integration/mcp/test_mcp_transports.py::test_pinned_revision_pairs_list_and_call_through_gateway["
        + "-".join(case)
        + "]"
        for case in product(
            baseline.revisions, baseline.revisions, baseline.upstream_transports, baseline.ingress_transports
        )
    )
    required: Final = official + matrix + baseline.required_tests
    assert len(set(required)) == len(required), "Conformance baseline contains duplicate required cases"
    require_capability_coverage(baseline, required, REVISION_SUPPORT)
    return required


def read_negotiation(request: bytes, response: bytes, content_type: str) -> tuple[str, str]:
    from httpx_sse import EventSource
    from mcp.types import InitializeRequestParams, InitializeResult

    body: Final = json.loads(request)
    received: Final = httpx.Response(200, content=response, headers={"content-type": content_type})
    messages: Final = (
        tuple(event.json() for event in EventSource(received).iter_sse() if event.data)
        if content_type.startswith("text/event-stream")
        else (received.json(),)
    )
    replies: Final = tuple(
        message["result"] for message in messages if message.get("id") == body["id"] and "result" in message
    )
    assert len(replies) == 1, "conformance negotiation: missing or duplicated initialize response"
    return InitializeRequestParams.model_validate(body["params"]).protocol_version, InitializeResult.model_validate(
        replies[0]
    ).protocol_version


def is_initialization(body: bytes) -> bool:
    try:
        parsed: Final = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return False
    return isinstance(parsed, dict) and parsed.get("method") == "initialize"
