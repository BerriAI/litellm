"""Model Armor in `mode: [post_call, post_mcp_call]` scanning MCP tool results through a real proxy.

The scripted MCP server and the Model Armor and Google token endpoints are the only doubles.
"""

import json
import os
import uuid
from collections.abc import Iterator
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
STRICT: Final = "model-armor-mcp-strict"
FAIL_OPEN: Final = "model-armor-mcp-fail-open"
BLOCK_MARK: Final = "SYNTHETIC-ARMOR-INSPECT-MATCH"
OUTAGE_MARK: Final = "SYNTHETIC-ARMOR-OUTAGE"
SENSITIVE: Final = "synthetic-ssn-078-05-1120"
REDACTED: Final = "[SYNTHETIC_REDACTED]"
ENTRIES: Final[tuple[EntryPoint, ...]] = ("mcp", "rest")


def _sanitize_target(template: str) -> str:
    return f"/v1/projects/synthetic-project/locations/us/templates/{template}:sanitizeModelResponse"


def _sdp(result: dict[str, object]) -> Reply:
    return Reply(
        body=json.dumps({"sanitizationResult": {"filterResults": {"sdp": {"sdpFilterResult": result}}}}).encode()
    )


def _model_armor(request: Request) -> Reply:
    if request.target.startswith(TOKEN_PATH):
        return Reply(
            body=json.dumps(
                {"access_token": "synthetic-google-token", "expires_in": 3600, "token_type": "Bearer"}
            ).encode()
        )
    text: Final[str] = next(iter(json.loads(request.body).values()))["text"]
    if OUTAGE_MARK in text:
        return Reply(status=500, body=json.dumps({"error": "synthetic Model Armor outage"}).encode())
    if BLOCK_MARK in text:
        return _sdp({"inspectResult": {"matchState": "MATCH_FOUND"}})
    if SENSITIVE in text:
        return _sdp(
            {"deidentifyResult": {"matchState": "MATCH_FOUND", "data": {"text": text.replace(SENSITIVE, REDACTED)}}}
        )
    return _sdp({"inspectResult": {"matchState": "NO_MATCH_FOUND"}})


def _guardrail(name: str, armor_url: str, private_key: str, *, fail_on_error: bool) -> dict[str, object]:
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
            "credentials": json.dumps(
                {
                    "type": "service_account",
                    "client_email": "synthetic@synthetic-project.iam.gserviceaccount.com",
                    "private_key": private_key,
                    "token_uri": armor_url + TOKEN_PATH,
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
        scripted_peer(_note_tool()) as peer,
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
    assert rig.scans(marker) == [(_sanitize_target(STRICT), f"{note}\n{note}\nnote")]


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
    assert rig.scans(marker) == [
        (_sanitize_target(STRICT), f"{note}\n{note}\nnote"),
        (_sanitize_target(STRICT), note),
        (_sanitize_target(STRICT), note),
    ]


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
