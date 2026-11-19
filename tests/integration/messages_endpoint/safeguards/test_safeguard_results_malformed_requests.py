"""Malformed ``safeguards`` never crash the proxy or reach the classifier, and the field means nothing outside
/v1/messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import (
    BASH_TOOL,
    CHAT_MODEL,
    CLASSIFIER_CONTEXT,
    DANGEROUS_TOOL_USE_BETA,
    RESPONSES_MODEL,
    SAFEGUARDS,
    Rig,
    chat_backend,
    flagged,
    json_body,
    messages_body,
    nonce,
    one_classifier_call,
    request_json,
    responses_backend,
    tool_id,
    tool_use_ids,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(graceful_stop_seconds() + 240)

IGNORED: Final[Mapping[str, JsonValue]] = {
    "string": "dangerous_tool_use",
    "integer": 5,
    "empty_list": [],
    "bare_names": ["dangerous_tool_use"],
    "other_type": [{"type": "other"}],
    "five_kilobytes": "x" * 5_000,
}
CLASSIFIED: Final[Mapping[str, JsonValue]] = {
    "duplicated": [*SAFEGUARDS, *SAFEGUARDS],
    "context_string": [{"type": "dangerous_tool_use", "classifier_context": "not an object"}],
    "git_state_integer": [{"type": "dangerous_tool_use", "classifier_context": {**CLASSIFIER_CONTEXT, "git_state": 5}}],
    "no_context": [{"type": "dangerous_tool_use"}],
}


@pytest.mark.parametrize("variant", tuple(IGNORED))
def test_unrecognized_safeguards_are_dropped_before_the_backend(rig: Rig, variant: str) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
    body: Final = json_body(rig.post(messages_body(CHAT_MODEL, request_nonce, safeguards=IGNORED[variant])))
    assert len(tool_use_ids(body)) == 1, body
    assert "safeguard_results" not in body, body
    received: Final = rig.received(request_nonce)
    assert rig.classifier_calls(received) == (), [request.target for request in received]
    assert len(rig.backend_calls(received)) == 1, [request.target for request in received]
    assert "safeguards" not in request_json(rig.backend_calls(received)[0]), received


@pytest.mark.parametrize("variant", tuple(CLASSIFIED))
def test_odd_but_recognizable_safeguards_still_classify_once(rig: Rig, variant: str) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
    body: Final = json_body(rig.post(messages_body(CHAT_MODEL, request_nonce, safeguards=CLASSIFIED[variant])))
    ids: Final = tool_use_ids(body)
    assert body["safeguard_results"] == flagged(ids), body
    classifier_call, backend_calls = one_classifier_call(rig, request_nonce)
    assert len(backend_calls) == 1, backend_calls
    assert "safeguards" not in request_json(backend_calls[0]), backend_calls[0].body
    assert classifier_call.target.endswith("/chat/completions")


def test_chat_completions_ignores_safeguards(rig: Rig) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
    function_tool: Final[Mapping[str, JsonValue]] = {
        "type": "function",
        "function": {"name": "Bash", "parameters": BASH_TOOL["input_schema"]},
    }
    request: Final[Mapping[str, JsonValue]] = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": f"clean the build directory {request_nonce}"}],
        "tools": [dict(function_tool)],
    }
    plain: Final = rig.post(request, path="/v1/chat/completions")
    with_field: Final = rig.post({**request, "safeguards": list(SAFEGUARDS)}, path="/v1/chat/completions")
    assert with_field.status_code == plain.status_code == 200, (plain.text, with_field.text)
    assert "safeguard_results" not in with_field.json(), with_field.text
    assert rig.classifier_calls(rig.received(request_nonce)) == ()


def test_responses_api_ignores_safeguards(rig: Rig) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, responses_backend(tool_id(request_nonce), stream=False))
    request: Final[Mapping[str, JsonValue]] = {
        "model": RESPONSES_MODEL,
        "input": f"clean the build directory {request_nonce}",
        "tools": [{"type": "function", "name": "Bash", "parameters": BASH_TOOL["input_schema"]}],
    }
    plain: Final = rig.post(request, path="/v1/responses")
    with_field: Final = rig.post({**request, "safeguards": list(SAFEGUARDS)}, path="/v1/responses")
    assert with_field.status_code == plain.status_code == 200, (plain.text, with_field.text)
    assert "safeguard_results" not in with_field.json(), with_field.text
    assert rig.classifier_calls(rig.received(request_nonce)) == ()


def test_beta_header_alone_changes_nothing(rig: Rig) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
    body: Final = json_body(rig.post(messages_body(CHAT_MODEL, request_nonce), beta=DANGEROUS_TOOL_USE_BETA))
    assert "safeguard_results" not in body, body
    assert rig.classifier_calls(rig.received(request_nonce)) == ()
