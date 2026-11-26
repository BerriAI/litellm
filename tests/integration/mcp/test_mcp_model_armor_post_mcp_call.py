"""Model Armor in `mode: [post_call, post_mcp_call]` scanning MCP tool results through a real proxy.

The scripted MCP server and the Model Armor and Google token endpoints are the only doubles.
"""

import json
import os
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, Scenario, gateway_from_environment
from integration._support.mcp import (
    EntryPoint,
    McpCaller,
    McpPeer,
    Outcome,
    ScriptedTool,
    register_mcp,
    scripted_peer,
)
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.shares_worker_state

TOKEN_PATH: Final = "/token"
DENIED_TOKEN_PATH: Final = "/denied-token"
STRICT: Final = "model-armor-mcp-strict"
FAIL_OPEN: Final = "model-armor-mcp-fail-open"
FAIL_OPEN_SLOW: Final = "model-armor-mcp-fail-open-slow"
FAIL_OPEN_NO_TOKEN: Final = "model-armor-mcp-fail-open-no-token"
SLOW_SECONDS: Final = 3
BLOCK_MARK: Final = "SYNTHETIC-ARMOR-INSPECT-MATCH"
SKIP_MARK: Final = "SYNTHETIC-ARMOR-FILTER-SKIPPED"
NO_REWRITE_MARK: Final = "SYNTHETIC-ARMOR-DEIDENTIFY-WITHOUT-REWRITE"
OUTAGE_MARK: Final = "SYNTHETIC-ARMOR-OUTAGE"
SLOW_MARK: Final = "SYNTHETIC-ARMOR-SLOW"
SLOW_RESCAN_MARK: Final = "SYNTHETIC-ARMOR-SLOW-RESCAN"
SENSITIVE: Final = "synthetic-ssn-078-05-1120"
REDACTED: Final = "[SYNTHETIC_REDACTED]"
ENTRIES: Final[tuple[EntryPoint, ...]] = ("mcp", "rest")


def _sanitize_target(template: str) -> str:
    return f"/v1/projects/synthetic-project/locations/us/templates/{template}:sanitizeModelResponse"


def _sdp(result: dict[str, object], skipped: bool = False) -> Reply:
    filters: Final[dict[str, object]] = {"sdp": {"sdpFilterResult": result}}
    rai: Final = {"rai": {"raiFilterResult": {"executionState": "EXECUTION_SKIPPED"}}} if skipped else {}
    return Reply(body=json.dumps({"sanitizationResult": {"filterResults": {**filters, **rai}}}).encode())


def _model_armor(request: Request) -> Reply:
    if request.target.startswith(DENIED_TOKEN_PATH):
        return Reply(status=400, body=json.dumps({"error": "invalid_grant"}).encode())
    if request.target.startswith(TOKEN_PATH):
        return Reply(
            body=json.dumps(
                {"access_token": "synthetic-google-token", "expires_in": 3600, "token_type": "Bearer"}
            ).encode()
        )
    text: Final[str] = next(iter(json.loads(request.body).values()))["text"]
    if not text.strip():
        return Reply(status=400, body=json.dumps({"error": {"code": 400, "message": "text is empty"}}).encode())
    skipped: Final = SKIP_MARK in text
    if OUTAGE_MARK in text:
        return Reply(status=500, body=json.dumps({"error": "synthetic Model Armor outage"}).encode())
    if SLOW_RESCAN_MARK in text and "\n" in text:
        return _sdp(
            {"deidentifyResult": {"matchState": "MATCH_FOUND", "data": {"text": text.replace("\n", " ")}}},
            skipped,
        )
    if SLOW_MARK in text:
        time.sleep(SLOW_SECONDS)
    if BLOCK_MARK in text:
        return _sdp({"inspectResult": {"matchState": "MATCH_FOUND"}}, skipped)
    if NO_REWRITE_MARK in text:
        return _sdp({"deidentifyResult": {"matchState": "MATCH_FOUND"}}, skipped)
    if SENSITIVE in text:
        return _sdp(
            {"deidentifyResult": {"matchState": "MATCH_FOUND", "data": {"text": text.replace(SENSITIVE, REDACTED)}}},
            skipped,
        )
    return _sdp({"inspectResult": {"matchState": "NO_MATCH_FOUND"}}, skipped)


def _guardrail(
    name: str,
    armor_url: str,
    private_key: str,
    *,
    fail_on_error: bool,
    timeout: float | None = None,
    token_path: str = TOKEN_PATH,
) -> dict[str, object]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "model_armor",
            "mode": ["post_call", "post_mcp_call"],
            "default_on": False,
            "mask_request_content": True,
            "mask_response_content": True,
            "skip_unscannable_attachments": True,
            "fail_on_error": fail_on_error,
            "project_id": "synthetic-project",
            "location": "us",
            "template_id": name,
            "api_endpoint": armor_url,
            **({"timeout": timeout} if timeout is not None else {}),
            "credentials": json.dumps(
                {
                    "type": "service_account",
                    "client_email": "synthetic@synthetic-project.iam.gserviceaccount.com",
                    "private_key": private_key,
                    "token_uri": armor_url + token_path,
                }
            ),
        },
    }


def _config(armor_url: str, root: Path) -> Path:
    private_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
    )
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        _guardrail(STRICT, armor_url, private_key, fail_on_error=True),
        _guardrail(FAIL_OPEN, armor_url, private_key, fail_on_error=False),
        _guardrail(FAIL_OPEN_SLOW, armor_url, private_key, fail_on_error=False, timeout=1),
        _guardrail(FAIL_OPEN_NO_TOKEN, armor_url, private_key, fail_on_error=False, token_path=DENIED_TOKEN_PATH),
    ]
    path: Final = root / "model-armor-post-mcp-call.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _note_tool() -> ScriptedTool:
    return ScriptedTool(
        "note",
        lambda params: {
            "content": [{"type": "text", "text": params["arguments"]["note"]}],
            "structuredContent": {"note": params["arguments"]["note"]},
            "isError": False,
        },
    )


def _blocks_tool() -> ScriptedTool:
    return ScriptedTool(
        "blocks",
        lambda params: {
            "content": [{"type": "text", "text": text} for text in params["arguments"]["texts"]],
            "isError": False,
        },
    )


def _lookup_tool() -> ScriptedTool:
    return ScriptedTool(
        "lookup",
        lambda params: {
            "content": [{"type": "text", "text": f"{params['arguments']['record']} {params['arguments']['marker']}"}],
            "isError": False,
        },
    )


def _model_calling_lookup(alias: str, record: str, marker: str, seen: list[str]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target.endswith("/models"):
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.target.endswith("/responses"), request.target
        seen.append(request.body.decode())
        answered: Final = any(
            isinstance(item, dict) and item.get("type") == "function_call_output"
            for item in json.loads(request.body).get("input", ())
        )
        output: Final[dict[str, object]] = (
            {
                "type": "message",
                "id": "msg_done",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            }
            if answered
            else {
                "type": "function_call",
                "id": "fc_lookup",
                "call_id": "call_lookup",
                "name": f"{alias}-lookup",
                "arguments": json.dumps({"record": record, "marker": marker}),
                "status": "completed",
            }
        )
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_lookup",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-4o-mini",
                    "output": [output],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    return respond


def _tool_result(outcome: Outcome) -> dict[str, object]:
    data: Final = tuple(line[5:].strip() for line in outcome.raw.splitlines() if line.startswith("data:"))
    body: Final = json.loads(data[-1] if data else outcome.raw)
    return body.get("result", body)


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    armor: Wire
    peer: McpPeer
    scenario: Scenario
    alias: str
    server_id: str

    def call_note(self, entry: EntryPoint, note: str, guardrails: tuple[str, ...]) -> Outcome:
        key: Final = self.scenario.key(
            object_permission={"mcp_servers": [self.server_id]},
            **({"metadata": {"guardrails": list(guardrails)}} if guardrails else {}),
        )
        return McpCaller(self.proxy, key, entry, self.alias).call(
            f"{self.alias}-note", {"note": note}, server_id=self.server_id
        )

    def call_blocks(self, entry: EntryPoint, texts: tuple[str, ...], guardrails: tuple[str, ...]) -> Outcome:
        key: Final = self.scenario.key(
            object_permission={"mcp_servers": [self.server_id]}, metadata={"guardrails": list(guardrails)}
        )
        return McpCaller(self.proxy, key, entry, self.alias).call(
            f"{self.alias}-blocks", {"texts": list(texts)}, server_id=self.server_id
        )

    def respond_with_lookup(
        self, record: str, marker: str, guardrails: tuple[str, ...], requested: tuple[str, ...] = ()
    ) -> tuple[str, str]:
        seen: Final[list[str]] = []
        with wire_server(_model_calling_lookup(self.alias, record, marker, seen)) as model_wire:
            model: Final = self.scenario.model(model="openai/responses/gpt-4o-mini", api_base=model_wire.url + "/v1")
            key: Final = self.scenario.key(
                object_permission={"mcp_servers": [self.server_id]},
                **({"metadata": {"guardrails": list(guardrails)}} if guardrails else {}),
            )
            response: Final = self.proxy.client.post(
                "/v1/responses",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "input": "look up the record",
                    **({"guardrails": list(requested)} if requested else {}),
                    "tools": [
                        {
                            "type": "mcp",
                            "server_url": "litellm_proxy",
                            "server_label": "litellm",
                            "require_approval": "never",
                        }
                    ],
                },
                timeout=90,
            )
        assert len(seen) == 2, f"the model must be asked again with the tool result: {response.text} {seen}"
        outputs: Final = [
            item["output"] for item in json.loads(seen[1])["input"] if item.get("type") == "function_call_output"
        ]
        assert len(outputs) == 1, seen[1]
        return response.text, str(outputs[0])

    def scans(self, marker: str) -> list[tuple[str, str]]:
        return [
            (request.target, next(iter(json.loads(request.body).values()))["text"])
            for request in self.armor.drain()
            if marker.encode() in request.body
        ]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    assert os.environ.get("LITELLM_LICENSE"), (
        "key-level guardrails are enterprise-only: these tests need LITELLM_LICENSE (CI forwards it); "
        "without it the tool call answers 403 and the failure reads as a regression instead of a missing license"
    )
    root: Final = tmp_path_factory.mktemp("model-armor-post-mcp-call")
    alias: Final = "armor" + uuid.uuid4().hex[:8]
    with (
        gateway_from_environment() as gateway,
        wire_server(_model_armor) as armor,
        scripted_peer(_note_tool(), _blocks_tool(), _lookup_tool()) as peer,
        owned_proxy_process(gateway, root, {}, config=_config(armor.url, root)) as owned,
        owned.gateway.scenario() as scenario,
    ):
        identity: Final = register_mcp(scenario, peer, alias)
        yield Rig(owned.gateway, armor, peer, scenario, alias, identity)


def test_model_armor_configured_for_post_mcp_call_is_loaded_and_listed(rig: Rig) -> None:
    listed: Final = rig.proxy.request("GET", "/v2/guardrails/list")
    assert listed.status_code == 200, listed.text
    modes: Final = {
        entry["guardrail_name"]: entry["litellm_params"]["mode"]
        for entry in listed.json()["guardrails"]
        if entry["guardrail_name"] in (STRICT, FAIL_OPEN)
    }
    assert modes == {STRICT: ["post_call", "post_mcp_call"], FAIL_OPEN: ["post_call", "post_mcp_call"]}, listed.text


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_tool_result_model_armor_flags_is_blocked_before_the_mcp_client_sees_it(rig: Rig, entry: EntryPoint) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex
    note: Final = f"{BLOCK_MARK} {marker}"

    outcome: Final = rig.call_note(entry, note, (STRICT,))

    assert outcome.error is not None and "Response blocked by Model Armor" in outcome.raw, outcome.raw
    assert marker not in outcome.raw, f"{entry}: the flagged tool result reached the client: {outcome.raw}"
    assert rig.scans(marker) == [(_sanitize_target(STRICT), f"{note}\nnote")]


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_deidentify_match_returns_the_sanitized_text_and_structured_content(rig: Rig, entry: EntryPoint) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex
    note: Final = f"account {SENSITIVE} {marker}"
    masked: Final = f"account {REDACTED} {marker}"

    outcome: Final = rig.call_note(entry, note, (STRICT,))

    assert outcome.error is None, outcome.raw
    result: Final = _tool_result(outcome)
    assert ([(block["type"], block["text"]) for block in result["content"]], result["structuredContent"]) == (
        [("text", masked)],
        {"note": masked},
    ), outcome.raw
    assert SENSITIVE not in outcome.raw, outcome.raw
    assert rig.scans(marker) == [(_sanitize_target(STRICT), f"{note}\nnote")]


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_model_armor_outage_blocks_the_tool_result_when_fail_on_error_is_true(rig: Rig, entry: EntryPoint) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex
    note: Final = f"{OUTAGE_MARK} {marker}"

    outcome: Final = rig.call_note(entry, note, (STRICT,))

    assert outcome.error is not None and marker not in outcome.raw, outcome.raw
    assert [target for target, _ in rig.scans(marker)] == [_sanitize_target(STRICT)]


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_model_armor_outage_passes_the_raw_tool_result_when_fail_on_error_is_false(
    rig: Rig, entry: EntryPoint
) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex
    note: Final = f"{OUTAGE_MARK} {marker}"

    outcome: Final = rig.call_note(entry, note, (FAIL_OPEN,))

    assert (outcome.error, outcome.text) == (None, note), outcome.raw
    assert [target for target, _ in rig.scans(marker)] == [_sanitize_target(FAIL_OPEN)]


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_default_off_guardrail_is_not_called_for_a_key_that_does_not_name_it(rig: Rig, entry: EntryPoint) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex
    note: Final = f"{BLOCK_MARK} {marker}"

    outcome: Final = rig.call_note(entry, note, ())

    assert (outcome.error, outcome.text) == (None, note), outcome.raw
    assert rig.scans(marker) == []


def test_a_flagged_tool_result_is_withheld_from_the_model_and_client_on_the_responses_api(rig: Rig) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex
    flagged: Final = f"{BLOCK_MARK} {marker}"

    client_body, tool_output = rig.respond_with_lookup(BLOCK_MARK, marker, (STRICT,))

    assert "Response blocked by Model Armor" in tool_output and flagged not in tool_output, tool_output
    assert flagged not in client_body, f"the flagged tool result reached the client: {client_body}"
    assert rig.scans(marker) == [(_sanitize_target(STRICT), flagged)]


def test_a_deidentify_match_is_masked_for_the_model_and_client_on_the_responses_api(rig: Rig) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex
    record: Final = f"account {SENSITIVE}"

    client_body, tool_output = rig.respond_with_lookup(record, marker, (STRICT,))

    assert tool_output == f"account {REDACTED} {marker}"
    assert f"{record} {marker}" not in client_body, f"the unmasked tool result reached the client: {client_body}"
    assert {target for target, _ in rig.scans(marker)} == {_sanitize_target(STRICT)}


def test_a_guardrail_named_on_the_request_masks_the_tool_result_on_the_responses_api(rig: Rig) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex

    _, tool_output = rig.respond_with_lookup(f"account {SENSITIVE}", marker, (), requested=(STRICT,))

    assert tool_output == f"account {REDACTED} {marker}"


def test_a_default_off_guardrail_is_not_called_on_the_responses_api_for_a_key_that_does_not_name_it(rig: Rig) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex

    _, tool_output = rig.respond_with_lookup(BLOCK_MARK, marker, ())

    assert tool_output == f"{BLOCK_MARK} {marker}"
    assert rig.scans(marker) == []


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_blank_block_is_left_out_of_the_scan_and_the_other_block_is_masked(rig: Rig, entry: EntryPoint) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex
    sensitive: Final = f"account {SENSITIVE} {marker}"

    outcome: Final = rig.call_blocks(entry, (" ", sensitive), (STRICT,))

    assert outcome.error is None, outcome.raw
    assert [block["text"] for block in _tool_result(outcome)["content"]] == [" ", f"account {REDACTED} {marker}"]
    assert rig.scans(marker) == [(_sanitize_target(STRICT), sensitive)]


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_tool_result_model_armor_only_partly_scanned_is_withheld_when_fail_on_error_is_true(
    rig: Rig, entry: EntryPoint
) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex

    outcome: Final = rig.call_blocks(entry, (f"{SKIP_MARK} {marker}",), (STRICT,))

    assert outcome.error is not None and marker not in outcome.raw, outcome.raw


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_match_found_while_a_filter_was_skipped_is_masked_when_failing_open(rig: Rig, entry: EntryPoint) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex

    outcome: Final = rig.call_blocks(entry, (f"{SKIP_MARK} {SENSITIVE} {marker}",), (FAIL_OPEN,))

    assert outcome.error is None, outcome.raw
    assert [block["text"] for block in _tool_result(outcome)["content"]] == [f"{SKIP_MARK} {REDACTED} {marker}"]


def test_the_apply_guardrail_endpoint_masks_a_response_and_fails_closed_on_an_outage(rig: Rig) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex

    masked: Final = rig.proxy.request(
        "POST",
        "/guardrails/apply_guardrail",
        {"guardrail_name": STRICT, "text": f"account {SENSITIVE} {marker}", "input_type": "response"},
    )
    closed: Final = rig.proxy.request(
        "POST",
        "/guardrails/apply_guardrail",
        {"guardrail_name": STRICT, "text": f"{OUTAGE_MARK} {marker}", "input_type": "response"},
    )
    opened: Final = rig.proxy.request(
        "POST",
        "/guardrails/apply_guardrail",
        {"guardrail_name": FAIL_OPEN, "text": f"{OUTAGE_MARK} {marker}", "input_type": "response"},
    )

    assert (masked.status_code, masked.json()["response_text"]) == (200, f"account {REDACTED} {marker}"), masked.text
    assert closed.status_code == 500 and marker not in closed.text, closed.text
    assert (opened.status_code, opened.json()["response_text"]) == (200, f"{OUTAGE_MARK} {marker}"), opened.text


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_deidentify_match_without_a_rewrite_is_withheld_when_fail_on_error_is_true(
    rig: Rig, entry: EntryPoint
) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex

    outcome: Final = rig.call_blocks(entry, (f"{NO_REWRITE_MARK} {SENSITIVE} {marker}",), (STRICT,))

    assert outcome.error is not None and marker not in outcome.raw and SENSITIVE not in outcome.raw, outcome.raw


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_deidentify_match_without_a_rewrite_passes_the_raw_text_when_failing_open(
    rig: Rig, entry: EntryPoint
) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex
    text: Final = f"{NO_REWRITE_MARK} {SENSITIVE} {marker}"

    outcome: Final = rig.call_blocks(entry, (text,), (FAIL_OPEN,))

    assert outcome.error is None, outcome.raw
    assert [block["text"] for block in _tool_result(outcome)["content"]] == [text]


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_model_armor_timeout_passes_the_raw_tool_result_when_fail_on_error_is_false(
    rig: Rig, entry: EntryPoint
) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex
    note: Final = f"{SLOW_MARK} {marker}"

    outcome: Final = rig.call_note(entry, note, (FAIL_OPEN_SLOW,))

    assert (outcome.error, outcome.text) == (None, note), outcome.raw


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_rescan_timeout_passes_the_raw_tool_result_when_fail_on_error_is_false(rig: Rig, entry: EntryPoint) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex
    texts: Final = (f"{SLOW_RESCAN_MARK} {marker}", "second block")

    outcome: Final = rig.call_blocks(entry, texts, (FAIL_OPEN_SLOW,))

    assert outcome.error is None, outcome.raw
    assert [block["text"] for block in _tool_result(outcome)["content"]] == list(texts)
    assert len(rig.scans(marker)) == 2


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_google_token_failure_passes_the_raw_tool_result_when_fail_on_error_is_false(
    rig: Rig, entry: EntryPoint
) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex
    note: Final = f"token refused {marker}"

    outcome: Final = rig.call_note(entry, note, (FAIL_OPEN_NO_TOKEN,))

    assert (outcome.error, outcome.text) == (None, note), outcome.raw
    assert rig.scans(marker) == []
