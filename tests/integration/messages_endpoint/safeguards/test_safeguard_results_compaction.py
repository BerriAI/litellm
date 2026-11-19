"""Server-side compaction next to auto mode: the summary call and the classifier call both run as the caller, a caller
the summary model rejects sees the edit's error while the request still completes, and neither call follows a Claude Code
session the router bound or trips a team trace id gate the parent request passed with its body trace id."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

import pytest
from integration._support.client import list_value, object_value
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import (
    CHAT_MODEL,
    CHAT_TARGET,
    CLASSIFIER_MODEL,
    CLASSIFIER_TARGET,
    UNSUPPORTED,
    Rig,
    auto_mode_body,
    chat_backend,
    chat_completion_reply,
    classifier_flagging_everything,
    classifier_prompt,
    deployment,
    flagged,
    json_body,
    key_hash,
    messages_body,
    nonce,
    request_json,
    safeguards_rig,
    spend_rows_for,
    tool_id,
    tool_use_ids,
)
from integration._support.wire import Reply, Request
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

COMPACT_EDIT: Final[list[JsonValue]] = [
    {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 50_000}}
]
FILLER_TURNS: Final = 24
FILLER: Final = "lorem ipsum dolor sit amet consectetur " * 400
SESSION_ROUTER: Final = "safeguards-session-router"
ROUTED_MODEL: Final = "safeguards-routed"
ROUTED_TARGET: Final = "/routed/inference/v1/chat/completions"
SUMMARY: Final = "The user asked to clean the build directory."


def summarizing_or_classifying(request: Request) -> Reply:
    if "Tool calls to classify:" in classifier_prompt(request):
        return classifier_flagging_everything(request)
    return chat_completion_reply(f"<summary>{SUMMARY}</summary>")


def session_router(wire_url: str) -> Sequence[Mapping[str, JsonValue]]:
    tiers: Final = dict.fromkeys(("SIMPLE", "MEDIUM", "COMPLEX", "REASONING"), ROUTED_MODEL)
    return (
        deployment(ROUTED_MODEL, "fireworks_ai/accounts/fireworks/models/kimi-k2p5", f"{wire_url}/routed/inference/v1"),
        {
            "model_name": SESSION_ROUTER,
            "litellm_params": {
                "model": "auto_router/complexity_router",
                "complexity_router_default_model": ROUTED_MODEL,
                "complexity_router_config": {"tiers": tiers},
            },
        },
    )


@pytest.fixture(scope="module")
def compacting(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    settings: Final = {"context_management_summary_model": CLASSIFIER_MODEL}
    with safeguards_rig(
        tmp_path_factory.mktemp("compaction"), general_settings=settings, extra_models=session_router
    ) as rig:
        yield rig


def long_request(request_nonce: str) -> dict[str, JsonValue]:
    turns: Final[list[JsonValue]] = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index} {FILLER}"}
        for index in range(FILLER_TURNS)
    ]
    return {
        **auto_mode_body(CHAT_MODEL, request_nonce),
        "messages": [*turns, {"role": "user", "content": f"clean the build directory {request_nonce}"}],
        "context_management": {"edits": COMPACT_EDIT},
    }


def applied_edit(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    edits: Final = list_value(object_value(body["context_management"])["applied_edits"])
    assert len(edits) == 1, body
    return object_value(edits[0])


def test_summary_and_classifier_both_run_as_the_caller(compacting: Rig) -> None:
    with compacting.gateway.scenario() as scenario:
        key: Final = scenario.key()
        request_nonce: Final = nonce()
        compacting.register(
            request_nonce, chat_backend(tool_id(request_nonce), stream=False), summarizing_or_classifying
        )
        body: Final = json_body(compacting.post(long_request(request_nonce), key=key))
        edit: Final = applied_edit(body)
        assert edit["type"] == "compact_20260112" and "error" not in edit, edit
        assert int(str(edit["summary_input_tokens"])) > 0 and int(str(edit["summary_output_tokens"])) > 0, edit
        assert body["safeguard_results"] == flagged(tool_use_ids(body)), body
        received: Final = compacting.received(request_nonce)
        classifier_side: Final = compacting.classifier_calls(received)
        assert len(classifier_side) == 2, [request.target for request in received]
        backend_calls: Final = compacting.backend_calls(received)
        assert len(backend_calls) == 1, [request.target for request in received]
        forwarded: Final = json.dumps(request_json(backend_calls[0])["messages"])
        assert SUMMARY in forwarded, forwarded[:2000]
        assert len(forwarded) < len(json.dumps(long_request(request_nonce)["messages"])) / 2, len(forwarded)
        rows: Final = spend_rows_for(key, 3)
        assert len(rows) == 3, rows
        assert {str(row["api_key"]) for row in rows} == {key_hash(key)}, rows


def test_caller_without_the_summary_model_sees_the_edit_error(compacting: Rig) -> None:
    with compacting.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[CHAT_MODEL])
        request_nonce: Final = nonce()
        compacting.register(
            request_nonce, chat_backend(tool_id(request_nonce), stream=False), summarizing_or_classifying
        )
        body: Final = json_body(compacting.post(long_request(request_nonce), key=key))
        assert applied_edit(body)["error"] == "summary_model_access_denied", body
        assert body["safeguard_results"] == list(UNSUPPORTED), body
        received: Final = compacting.received(request_nonce)
        assert compacting.classifier_calls(received) == (), [request.target for request in received]
        assert len(compacting.backend_calls(received)) == 1


def assert_summarized_and_classified_on_their_models(rig: Rig, request_nonce: str, body: dict[str, JsonValue]) -> str:
    received: Final = rig.received(request_nonce)
    targets: Final = [request.target for request in received]
    edit: Final = applied_edit(body)
    assert edit["type"] == "compact_20260112" and "error" not in edit, (edit, targets)
    assert body["safeguard_results"] == flagged(tool_use_ids(body)), (body, targets)
    assert targets.count(CLASSIFIER_TARGET) == 2, targets
    backend_calls: Final = rig.backend_calls(received)
    assert len(backend_calls) == 1, targets
    assert SUMMARY in json.dumps(request_json(backend_calls[0])["messages"]), targets
    return backend_calls[0].target


def test_session_the_router_bound_keeps_the_summary_and_classifier_calls_on_their_models(compacting: Rig) -> None:
    with compacting.gateway.scenario() as scenario:
        key: Final = scenario.key()
        session: Final = {"x-claude-code-session-id": f"session-{uuid.uuid4().hex}"}
        binding_nonce: Final = nonce()
        compacting.register(binding_nonce, chat_backend(tool_id(binding_nonce), stream=False, finish_reason="stop"))
        binding: Final = json_body(
            compacting.post(messages_body(SESSION_ROUTER, binding_nonce), key=key, beta=None, headers=session)
        )
        assert binding["stop_reason"] == "end_turn", binding
        assert [request.target for request in compacting.received(binding_nonce)] == [ROUTED_TARGET]
        request_nonce: Final = nonce()
        compacting.register(
            request_nonce, chat_backend(tool_id(request_nonce), stream=False), summarizing_or_classifying
        )
        subagent: Final = {**session, "x-claude-code-agent-id": f"agent-{uuid.uuid4().hex[:12]}"}
        body: Final = json_body(compacting.post(long_request(request_nonce), key=key, headers=subagent))
        assert assert_summarized_and_classified_on_their_models(compacting, request_nonce, body) == ROUTED_TARGET


def test_body_trace_id_that_passed_the_team_gate_carries_the_summary_and_classifier_calls(compacting: Rig) -> None:
    with compacting.gateway.scenario() as scenario:
        team_id: Final = scenario.team(metadata={"require_trace_id": True})
        key: Final = scenario.key(team_id=team_id)
        refused: Final = compacting.post(auto_mode_body(CHAT_MODEL, nonce()), key=key)
        assert refused.status_code == 400 and "trace" in refused.text, refused.text
        request_nonce: Final = nonce()
        compacting.register(
            request_nonce, chat_backend(tool_id(request_nonce), stream=False), summarizing_or_classifying
        )
        traced: Final = {**long_request(request_nonce), "litellm_trace_id": f"trace-{uuid.uuid4().hex}"}
        body: Final = json_body(compacting.post(traced, key=key))
        assert assert_summarized_and_classified_on_their_models(compacting, request_nonce, body) == CHAT_TARGET
