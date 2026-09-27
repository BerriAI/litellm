import hashlib
import json
import os
import queue
from pathlib import Path
from typing import Final
from unittest.mock import Mock, patch

import pytest
from tests.integration._support.conformance import (
    is_initialization,
    read_negotiation,
    prefixed_request,
    read_checks,
    require_negotiations,
    require_passes,
    verify_archive,
)
from pydantic import ValidationError


@pytest.mark.asyncio
async def test_negotiation_records_preserve_the_peer_call_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from integration._support.mcp import math_service
    from integration.mcp.test_mcp_user_env_vars import UPSTREAM_CALLS
    from mcp.server.context import ServerRequestContext
    from mcp.types import InitializeResult

    records: Final[list[dict[str, object]]] = []
    service: Final = math_service(record=records.append)
    context: Final = ServerRequestContext(
        session=Mock(),
        lifespan_context=None,
        protocol_version="2025-03-26",
        method="initialize",
        params={"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "client", "version": "1"}},
    )

    async def initialize(request: ServerRequestContext[object, object]) -> InitializeResult:
        return InitializeResult.model_validate(
            {
                "protocolVersion": request.protocol_version,
                "capabilities": {},
                "serverInfo": {"name": "peer", "version": "1"},
            }
        )

    await service.middleware[-1](context, initialize)
    calls: Final = UPSTREAM_CALLS.validate_python(tuple(records))
    assert len(calls) == 1 and calls[0].headers == {} and calls[0].body == {}
    assert records[0]["negotiation"] == {"requested": "2025-03-26", "returned": "2025-03-26"}


@pytest.mark.parametrize("file_backed", (False, True))
def test_peer_drain_preserves_received_requests(
    tmp_path: Path, file_backed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from integration._support.mcp import McpPeer

    request: Final = {"body": {"method": "tools/call"}, "headers": {"authorization": "synthetic"}}
    negotiation: Final = {"body": {}, "headers": {}, "negotiation": {"requested": "2025-03-26", "returned": "2025-03-26"}}
    records: Final = (request, negotiation)
    observed: Final[queue.Queue[dict[str, object]]] = queue.Queue()
    path: Final = tmp_path / "peer.jsonl" if file_backed else None
    if path is not None:
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
    else:
        for record in records:
            observed.put(record)
    peer: Final = McpPeer("http://peer", observed, record=path)
    assert peer.drain() == (request,)
    assert peer.drain() == ()
    if path is not None:
        with path.open("a") as sink:
            sink.write("".join(json.dumps(record) + "\n" for record in records))
    else:
        for record in records:
            observed.put(record)
    assert peer.drain(include_negotiation=True) == records
    assert peer.drain(include_negotiation=True) == ()


@pytest.mark.parametrize("explicit", (False, True))
def test_owned_proxy_isolates_automatic_coverage_unless_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from integration._support.client import Gateway
    from integration._support.process import owned_proxy_process

    monkeypatch.setenv("COVERAGE_PROCESS_CONFIG", "parent-config")
    monkeypatch.setenv("COVERAGE_PROCESS_START", "parent.toml")
    overrides: Final = {"COVERAGE_PROCESS_CONFIG": "child-config"} if explicit else {}
    gateway: Final = Gateway(Mock(), "owner-key", "http://upstream")
    with (
        patch("integration._support.process.subprocess.Popen") as spawn,
        patch("integration._support.process.httpx.Client") as client,
        patch("integration._support.process.group_members", return_value=()),
    ):
        spawn.return_value.poll.return_value = None
        client.return_value.__enter__.return_value.get.return_value.status_code = 200
        with owned_proxy_process(gateway, tmp_path, overrides):
            child: Final = spawn.call_args.kwargs["env"]
            assert child["COVERAGE_PROCESS_CONFIG"] == ("child-config" if explicit else "")
            assert child["COVERAGE_PROCESS_START"] == ""
            assert child["LITELLM_MASTER_KEY"] == gateway.key
    assert os.environ["COVERAGE_PROCESS_CONFIG"] == "parent-config"


@pytest.mark.parametrize("selected", ("test_mcp_transports.py", "test_mcp_credentials.py", ""))
@pytest.mark.parametrize("missing_required", (False, True))
def test_runner_requires_conformance_for_selected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selected: str, missing_required: bool
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from integration import run
    from integration._support.conformance import required_conformance_nodes

    files: Final = tuple(
        str(path.relative_to(Path.cwd()))
        for path in sorted((Path.cwd() / "tests/integration/mcp").glob("test_*.py"))
        if not selected or path.name == selected
    )
    required: Final = tuple(node for node in required_conformance_nodes() if node.split("::")[0] in files)
    collected: Final = required + tuple(f"{path}::other" for path in files)
    omitted: Final = required[:1] if missing_required else ()
    (tmp_path / "execution.json").write_text(
        json.dumps(
            {
                "collected": collected,
                "passed": tuple(n for n in collected if n not in omitted),
                "skipped": omitted,
                "complete": True,
            }
        )
    )
    monkeypatch.setattr("sys.argv", ["run.py", "mcp", "--results", str(tmp_path), *(files if selected else ())])
    with patch("integration.run.subprocess.call", return_value=0):
        if omitted:
            with pytest.raises(AssertionError, match="conformance"):
                run.main()
        else:
            assert run.main() == 0


def test_changed_archive_is_rejected(tmp_path: Path) -> None:
    archive: Final = tmp_path / "reference.tar.gz"
    archive.write_bytes(b"changed reference")
    with pytest.raises(AssertionError, match="SHA-256"):
        verify_archive(archive, hashlib.sha256(b"approved reference").hexdigest())


def test_matching_archive_is_accepted(tmp_path: Path) -> None:
    archive: Final = tmp_path / "reference.tar.gz"
    archive.write_bytes(b"approved reference")
    assert verify_archive(archive, "5a870d3ee9520e3a912c735d007d7b40f6d3cbf8b61d909ebb8361f423d4ba1e") is None


@pytest.mark.parametrize(
    "checks",
    (
        [],
        [{"id": "tools-list", "status": "FAILURE"}],
        [{"id": "tools-list", "status": "WARNING"}],
        [{"id": "wire-schema", "status": "SUCCESS"}],
        [{"id": "tools-list", "status": "SUCCESS"}, {"id": "wire-schema", "status": "FAILURE"}],
        [{"id": "tools-list", "status": "SUCCESS"}, {"id": "tools-list", "status": "SUCCESS"}],
    ),
)
def test_incomplete_or_failed_scenario_cannot_pass(tmp_path: Path, checks: list[dict[str, str]]) -> None:
    (tmp_path / "checks.json").write_text(json.dumps(checks))
    with pytest.raises(AssertionError, match="conformance"):
        read_checks(tmp_path, "tools-list")


def test_skipped_scenario_without_a_report_cannot_pass(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="conformance"):
        read_checks(tmp_path, "tools-list")


def test_multiple_reports_cannot_supply_a_stale_pass(tmp_path: Path) -> None:
    for name in ("previous", "current"):
        directory: Final = tmp_path / name
        directory.mkdir()
        (directory / "checks.json").write_text('[{"id":"tools-list","status":"SUCCESS"}]')
    with pytest.raises(AssertionError, match="conformance"):
        read_checks(tmp_path, "tools-list")


def test_passing_scenario_keeps_nonbinding_diagnostics(tmp_path: Path) -> None:
    (tmp_path / "checks.json").write_text(
        json.dumps(
            [
                {"id": identity, "status": "INFO" if identity == "advisory" else "SUCCESS"}
                for identity in ("tools-list", "tools-name-format", "wire-schema-valid", "advisory")
            ]
        )
    )
    checks: Final = read_checks(tmp_path, "tools-list")
    assert checks[-1].id == "advisory" and checks[-1].status == "INFO"


def test_unrecognized_result_status_cannot_pass(tmp_path: Path) -> None:
    (tmp_path / "checks.json").write_text('[{"id":"tools-list","status":"SKIPPED"}]')
    with pytest.raises(ValidationError, match="status"):
        read_checks(tmp_path, "tools-list")


@pytest.mark.parametrize(
    "result",
    (
        {"content": [{"type": "text", "text": "Tool not found"}], "isError": True},
        {"content": [{"type": "text", "text": "wrong upstream payload"}]},
    ),
)
def test_official_text_success_cannot_hide_an_error_or_wrong_payload(tmp_path: Path, result: dict[str, object]) -> None:
    (tmp_path / "checks.json").write_text(
        json.dumps([{"id": "tools-call-simple-text", "status": "SUCCESS", "details": {"result": result}}])
    )
    with pytest.raises(AssertionError, match="reference payload"):
        read_checks(tmp_path, "tools-call-simple-text")


def test_official_text_success_requires_the_reference_payload(tmp_path: Path) -> None:
    (tmp_path / "checks.json").write_text(
        json.dumps(
            [
                {
                    "id": "tools-call-simple-text",
                    "status": "SUCCESS",
                    "details": {
                        "result": {
                            "content": [{"type": "text", "text": "This is a simple text response for testing."}],
                            "isError": False,
                        }
                    },
                }
            ]
        )
    )
    assert read_checks(tmp_path, "tools-call-simple-text")[0].status == "SUCCESS"


@pytest.mark.parametrize("method", ("tools/call", "prompts/get"))
def test_only_fixture_name_is_translated_to_advertised_name(method: str) -> None:
    request: Final = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": method,
        "params": {"name": "test_tool", "_meta": {"progressToken": "keep"}},
    }
    changed: Final = json.loads(prefixed_request(json.dumps(request).encode(), "official"))
    assert changed == {**request, "params": {"name": "official-test_tool", "_meta": {"progressToken": "keep"}}}
    assert "arguments" not in changed["params"]


@pytest.mark.parametrize(
    "body",
    (
        b"malformed JSON",
        b"[]",
        b'{"method":"initialize","params":{"protocolVersion":"2025-03-26"}}',
        b'{"method":"tools/call","params":{"name":17}}',
    ),
)
def test_name_integration_preserves_other_requests(body: bytes) -> None:
    assert prefixed_request(body, "official") == body


@pytest.mark.parametrize("observed", ((), (("2025-11-25", "2025-11-25"),), (("2025-03-26", "2025-11-25"),)))
def test_wrong_or_missing_negotiation_cannot_pass(observed: tuple[tuple[str, str], ...]) -> None:
    with pytest.raises(AssertionError, match="negotiation"):
        require_negotiations("2025-03-26", observed)


def test_actual_requested_and_returned_revision_are_required() -> None:
    assert require_negotiations("2025-03-26", (("2025-03-26", "2025-03-26"),)) is None


@pytest.mark.parametrize(
    "collected,passed,skipped,complete",
    (
        ([], [], [], True),
        (["one"], ["one"], [], True),
        (["one", "two"], ["one"], ["two"], True),
        (["one", "two"], ["one", "two"], [], False),
    ),
)
def test_missing_skipped_or_incomplete_gate_cannot_pass(
    tmp_path: Path, collected: list[str], passed: list[str], skipped: list[str], complete: bool
) -> None:
    (tmp_path / "execution.json").write_text(
        json.dumps({"collected": collected, "passed": passed, "skipped": skipped, "complete": complete})
    )
    with pytest.raises(AssertionError, match="conformance"):
        require_passes(tmp_path, ("one", "two"))


def test_complete_gate_requires_every_declared_case(tmp_path: Path) -> None:
    (tmp_path / "execution.json").write_text(
        json.dumps(
            {
                "collected": ["one", "two", "unrelated"],
                "passed": ["one", "two"],
                "skipped": ["unrelated"],
                "complete": True,
            }
        )
    )
    assert require_passes(tmp_path, ("one", "two")) is None


@pytest.mark.parametrize("streamed", (False, True))
def test_negotiation_evidence_reads_the_actual_reply(streamed: bool) -> None:
    request: Final = json.dumps(
        {
            "id": 1,
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        }
    ).encode()
    result: Final = json.dumps(
        {
            "id": 1,
            "result": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "serverInfo": {"name": "test", "version": "1"},
            },
        }
    ).encode()
    response: Final = b"data:\n\n: heartbeat\n\nevent: message\ndata: " + result + b"\n\n" if streamed else result
    assert read_negotiation(request, response, "text/event-stream" if streamed else "application/json") == (
        "2025-03-26",
        "2025-11-25",
    )


def test_unrelated_rpc_response_cannot_supply_negotiation() -> None:
    with pytest.raises(AssertionError, match="initialize response"):
        read_negotiation(b'{"id":1}', b'{"id":2,"result":{}}', "application/json")


@pytest.mark.parametrize(
    "body,expected",
    ((b"", False), (b"[]", False), (b'{"method":"tools/list"}', False), (b'{"method":"initialize"}', True)),
)
def test_only_initialize_is_captured(body: bytes, expected: bool) -> None:
    assert is_initialization(body) is expected


@pytest.mark.parametrize(
    "scenario,identities",
    (
        (
            "server-session-lifecycle",
            (
                "server-session-initialized-accepted",
                "server-session-delete-accepted",
                "server-session-terminated-returns-404",
            ),
        ),
        (
            "server-sse-multiple-streams",
            ("server-accepts-multiple-post-streams", "server-sse-streams-functional"),
        ),
    ),
)
def test_complete_transport_checks_pass_and_missing_checks_fail(
    tmp_path: Path, scenario: str, identities: tuple[str, ...]
) -> None:
    report: Final = tmp_path / "checks.json"
    report.write_text(json.dumps([{"id": identity, "status": "SUCCESS"} for identity in identities]))
    assert len(read_checks(tmp_path, scenario)) == len(identities)
    for missing in identities:
        report.write_text(
            json.dumps([{"id": identity, "status": "SUCCESS"} for identity in identities if identity != missing])
        )
        with pytest.raises(AssertionError, match="required check"):
            read_checks(tmp_path, scenario)


@pytest.mark.parametrize(
    "scenario,identities",
    (
        ("server-initialize", ("server-initialize", "server-session-id-visible-ascii", "wire-schema-valid")),
        ("tools-list", ("tools-list", "tools-name-format", "wire-schema-valid")),
        ("tools-call-image", ("tools-call-image", "wire-schema-valid")),
    ),
)
def test_missing_secondary_checks_cannot_report_complete_conformance(
    tmp_path: Path, scenario: str, identities: tuple[str, ...]
) -> None:
    report: Final = tmp_path / "checks.json"
    for missing in identities:
        report.write_text(
            json.dumps([{"id": identity, "status": "SUCCESS"} for identity in identities if identity != missing])
        )
        with pytest.raises(AssertionError, match="conformance"):
            read_checks(tmp_path, scenario)
    report.write_text(json.dumps([{"id": identity, "status": "SUCCESS"} for identity in identities]))
    assert len(read_checks(tmp_path, scenario)) == len(identities)
