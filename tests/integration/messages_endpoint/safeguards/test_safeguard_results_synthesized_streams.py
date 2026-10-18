"""A /v1/messages stream LiteLLM builds from a complete reply (web search interception, a fallback's answer, a guardrail
rewrite) carries the ``stop_details`` and ``safeguard_results`` a real Anthropic stream puts on its ``message_delta``."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Final
from urllib.parse import urlsplit

import pytest
from integration._support.client import eventually, object_value
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import (
    BASH_TOOL,
    CLAUDE_MODEL,
    CLAUDE_TARGET,
    TAVILY_TARGET,
    Responder,
    Rig,
    auto_mode_body,
    available,
    deployment,
    native_message,
    native_stream_frames,
    nonce,
    request_json,
    safeguards_rig,
    sse_events,
    stream_results,
    stream_tool_use_ids,
    text_block,
    tool_id,
    tool_use_block,
    unavailable,
)
from integration._support.wire import Reply, Request
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

CLAUDE: Final = "anthropic/claude-sonnet-4-6"
PRIMARY_MODEL: Final = "synthesized-primary"
FALLBACK_MODEL: Final = "synthesized-fallback"
REFUSED_MODEL: Final = "synthesized-refused"
MOCK_MODEL: Final = "synthesized-mock"
PIPELINE_MODEL: Final = "synthesized-pipeline"
AZURE_REFUSED_MODEL: Final = "synthesized-azure-refused"
CACHED_FALLBACK_MODEL: Final = "synthesized-cached-fallback"
PRIMARY_TARGET: Final = "/primary/v1/messages"
FALLBACK_TARGET: Final = "/fallback/v1/messages"
REFUSED_TARGET: Final = "/refused/v1/messages"
PIPELINE_TARGET: Final = "/pipeline/v1/messages"
AZURE_REFUSED_TARGET: Final = "/azure/anthropic/v1/messages"
CACHED_FALLBACK_TARGET: Final = "/cached-fallback/v1/messages"
MOCK_ANSWER: Final = "mocked fallback answer"
TOOL_PERMISSION: Final = "synthesized-tool-permission"
PIPELINE_REWRITE: Final = "synthesized-pipeline-rewrite"
SECRET: Final = "SECRET"
MASKED: Final = "[MASKED]"
REWRITE_CODE: Final = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    "    if input_type != 'response':\n"
    "        return allow()\n"
    f"    texts = [text.replace({SECRET!r}, {MASKED!r}) for text in inputs.get('texts') or []]\n"
    "    calls = [\n"
    "        {'id': call['id'], 'type': 'function', 'function': {'name': call['function']['name'],\n"
    f"        'arguments': call['function']['arguments'].replace({SECRET!r}, {MASKED!r})}}}}\n"
    "        for call in inputs.get('tool_calls') or []\n"
    "    ]\n"
    "    return modify(texts=texts, tool_calls=calls)\n"
)
WEB_SEARCH_TOOL: Final[Mapping[str, JsonValue]] = {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
READ_TOOL: Final[Mapping[str, JsonValue]] = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
}
NO_CACHE: Final[Mapping[str, JsonValue]] = {"cache": {"no-cache": True}}
REFUSAL_STOP_DETAILS: Final[Mapping[str, JsonValue]] = {
    "type": "refusal",
    "category": "cyber",
    "explanation": "This request was declined because it could enable cyber harm.",
}
_FRAME_BOUNDARY: Final = re.compile(rb"(?<=\n\n)")

LITELLM_SETTINGS: Final[Mapping[str, JsonValue]] = {
    "callbacks": ["websearch_interception"],
    "websearch_interception_params": {"enabled_providers": ["anthropic"], "search_tool_name": "tavily-search"},
}


def extra_models(wire_url: str) -> Sequence[Mapping[str, JsonValue]]:
    return (
        deployment(PRIMARY_MODEL, CLAUDE, f"{wire_url}/primary"),
        deployment(FALLBACK_MODEL, CLAUDE, f"{wire_url}/fallback"),
        deployment(REFUSED_MODEL, CLAUDE, f"{wire_url}/refused"),
        deployment(MOCK_MODEL, CLAUDE, f"{wire_url}/mock", mock_response=MOCK_ANSWER),
        deployment(PIPELINE_MODEL, CLAUDE, f"{wire_url}/pipeline"),
        deployment(AZURE_REFUSED_MODEL, "azure_ai/claude-sonnet-4-6", f"{wire_url}/azure/anthropic"),
        deployment(CACHED_FALLBACK_MODEL, CLAUDE, f"{wire_url}/cached-fallback"),
    )


def top_level(wire_url: str) -> Mapping[str, JsonValue]:
    tavily: Final[Mapping[str, JsonValue]] = {
        "search_tool_name": "tavily-search",
        "litellm_params": {
            "search_provider": "tavily",
            "api_key": "synthetic-tavily-key",
            "api_base": f"{wire_url}/tavily",
        },
    }
    tool_permission: Final[Mapping[str, JsonValue]] = {
        "guardrail_name": TOOL_PERMISSION,
        "litellm_params": {
            "guardrail": "tool_permission",
            "mode": "post_call",
            "default_on": False,
            "default_action": "allow",
            "on_disallowed_action": "rewrite",
            "rules": [{"id": "deny_bash", "tool_name": "^Bash$", "decision": "deny"}],
        },
    }
    pipeline_rewrite: Final[Mapping[str, JsonValue]] = {
        "guardrail_name": PIPELINE_REWRITE,
        "litellm_params": {
            "guardrail": "custom_code",
            "mode": "post_call",
            "default_on": False,
            "custom_code": REWRITE_CODE,
        },
    }
    masking_policy: Final[Mapping[str, JsonValue]] = {
        "guardrails": {"add": [PIPELINE_REWRITE]},
        "pipeline": {
            "mode": "post_call",
            "steps": [{"guardrail": PIPELINE_REWRITE, "on_pass": "allow", "on_fail": "block"}],
        },
    }
    return {
        "search_tools": [dict(tavily)],
        "guardrails": [dict(tool_permission), dict(pipeline_rewrite)],
        "policies": {"synthesized-masking": dict(masking_policy)},
        "policy_attachments": [{"policy": "synthesized-masking", "models": [PIPELINE_MODEL]}],
        "router_settings": {
            "disable_cooldowns": True,
            "content_policy_fallbacks": [
                {PRIMARY_MODEL: [FALLBACK_MODEL]},
                {REFUSED_MODEL: [MOCK_MODEL]},
                {AZURE_REFUSED_MODEL: [CACHED_FALLBACK_MODEL]},
            ],
        },
    }


@pytest.fixture(scope="module")
def synthesizing(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("synthesized-streams")
    with safeguards_rig(
        directory, litellm_settings=LITELLM_SETTINGS, top_level=top_level, extra_models=extra_models
    ) as rig:
        yield rig


def claude_backend(message: Mapping[str, JsonValue], delta_fields: Mapping[str, JsonValue]) -> Responder:
    def respond(request: Request) -> Reply:
        if request_json(request).get("stream") is True:
            return Reply(
                content_type="text/event-stream", chunks=native_stream_frames(message, delta_extra=delta_fields)
            )
        return Reply(body=json.dumps({**message, **delta_fields}).encode())

    return respond


def by_target(backends: Mapping[str, Responder]) -> Responder:
    def respond(request: Request) -> Reply:
        return backends[urlsplit(request.target).path](request)

    return respond


def refusal(identity: str) -> Mapping[str, JsonValue]:
    return native_message(identity, content=(), stop_reason="refusal", model="claude-sonnet-4-6")


def answer(identity: str, text: str) -> Mapping[str, JsonValue]:
    return native_message(identity, content=(text_block(text),), stop_reason="end_turn", model="claude-sonnet-4-6")


def refusal_fields() -> Mapping[str, JsonValue]:
    return {"stop_details": dict(REFUSAL_STOP_DETAILS), "safeguard_results": list(available({}))}


def search_body(model: str, request_nonce: str) -> Mapping[str, JsonValue]:
    return auto_mode_body(model, request_nonce, tools=[dict(WEB_SEARCH_TOOL)], prompt="search the web for", **NO_CACHE)


def message_delta(raw: bytes) -> Mapping[str, JsonValue]:
    deltas: Final = tuple(
        object_value(event["delta"]) for event in sse_events(raw) if event.get("type") == "message_delta"
    )
    assert len(deltas) == 1, raw
    return deltas[0]


def message_start(raw: bytes) -> Mapping[str, JsonValue]:
    starts: Final = tuple(
        object_value(event["message"]) for event in sse_events(raw) if event.get("type") == "message_start"
    )
    assert len(starts) == 1, raw
    return starts[0]


def streamed_text(raw: bytes) -> str:
    deltas: Final = tuple(
        object_value(event["delta"]) for event in sse_events(raw) if event.get("type") == "content_block_delta"
    )
    return "".join(str(delta["text"]) for delta in deltas if delta.get("type") == "text_delta")


def delta_frame(raw: bytes) -> bytes:
    frames: Final = tuple(frame for frame in _FRAME_BOUNDARY.split(raw) if b'"message_delta"' in frame)
    assert len(frames) == 1, raw
    return frames[0]


def targets(rig: Rig, request_nonce: str) -> tuple[str, ...]:
    return tuple(urlsplit(request.target).path for request in rig.received(request_nonce))


def test_intercepted_refusal_stream_delivers_the_upstream_stop_details(synthesizing: Rig) -> None:
    request_nonce: Final = nonce()
    synthesizing.register(request_nonce, claude_backend(refusal(tool_id(request_nonce)), refusal_fields()))
    status, raw = synthesizing.stream(search_body(CLAUDE_MODEL, request_nonce))
    assert status == 200, raw
    assert message_delta(raw) == {
        "stop_reason": "refusal",
        "stop_sequence": None,
        "stop_details": dict(REFUSAL_STOP_DETAILS),
        "safeguard_results": list(available({})),
    }, raw
    assert targets(synthesizing, request_nonce) == (CLAUDE_TARGET,)


def test_intercepted_refusal_stream_with_a_content_policy_fallback_delivers_the_refusal_without_falling_back(
    synthesizing: Rig,
) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    synthesizing.register(
        request_nonce,
        by_target(
            {
                PRIMARY_TARGET: claude_backend(refusal(identity), refusal_fields()),
                FALLBACK_TARGET: claude_backend(answer(f"{identity}f", f"fallback answer {request_nonce}"), {}),
            }
        ),
    )
    status, raw = synthesizing.stream(search_body(PRIMARY_MODEL, request_nonce))
    assert status == 200, raw
    assert message_delta(raw) == {
        "stop_reason": "refusal",
        "stop_sequence": None,
        "stop_details": dict(REFUSAL_STOP_DETAILS),
        "safeguard_results": list(available({})),
    }, raw
    assert targets(synthesizing, request_nonce) == (PRIMARY_TARGET,)


def test_tool_permission_rewrite_keeps_only_the_delivered_tool_verdict_with_its_explanation(synthesizing: Rig) -> None:
    request_nonce: Final = nonce()
    denied: Final = tool_id(request_nonce)
    allowed: Final = f"{denied}r"
    allowed_verdict: Final[Mapping[str, JsonValue]] = {
        "type": "evaluated",
        "outcome": "flagged",
        "explanation": f"reads a secret {allowed}",
    }
    verdicts: Final[Mapping[str, JsonValue]] = {
        denied: {"type": "evaluated", "outcome": "flagged", "explanation": f"destructive {denied}"},
        allowed: dict(allowed_verdict),
    }
    message: Final = native_message(
        denied,
        content=(
            text_block("Cleaning up."),
            tool_use_block(denied),
            tool_use_block(allowed, name="Read", arguments={"path": "/work/repo/.env"}),
        ),
        model="claude-sonnet-4-6",
    )
    synthesizing.register(request_nonce, claude_backend(message, {"safeguard_results": list(available(verdicts))}))
    body: Final = auto_mode_body(
        CLAUDE_MODEL, request_nonce, tools=[dict(BASH_TOOL), dict(READ_TOOL)], guardrails=[TOOL_PERMISSION], **NO_CACHE
    )
    status, raw = synthesizing.stream(body)
    assert status == 200, raw
    assert stream_tool_use_ids(raw) == (allowed,), raw
    assert stream_results(raw) == available({allowed: dict(allowed_verdict)}), raw
    assert targets(synthesizing, request_nonce) == (CLAUDE_TARGET,)


def test_intercepted_answer_stream_keeps_its_safeguard_results(synthesizing: Rig) -> None:
    request_nonce: Final = nonce()
    text: Final = f"found it {request_nonce}"
    results: Final = list(available({}))
    synthesizing.register(
        request_nonce,
        claude_backend(answer(tool_id(request_nonce), text), {"stop_details": None, "safeguard_results": results}),
    )
    status, raw = synthesizing.stream(search_body(CLAUDE_MODEL, request_nonce))
    assert status == 200, raw
    assert streamed_text(raw) == text, raw
    assert message_delta(raw) == {"stop_reason": "end_turn", "stop_sequence": None, "safeguard_results": results}, raw
    received: Final = targets(synthesizing, request_nonce)
    assert received == (CLAUDE_TARGET,) and TAVILY_TARGET not in received, received


def test_native_refusal_stream_without_a_guardrail_keeps_the_upstream_message_delta_bytes(synthesizing: Rig) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    message: Final = native_message(
        identity, content=(text_block("Partial."),), stop_reason="refusal", model="claude-sonnet-4-6"
    )
    fields: Final = {**refusal_fields(), "safeguard_results": list(unavailable((identity,), "error"))}
    sent: Final = native_stream_frames(message, delta_extra=fields)
    synthesizing.register(request_nonce, claude_backend(message, fields))
    status, raw = synthesizing.stream(auto_mode_body(CLAUDE_MODEL, request_nonce, **NO_CACHE))
    assert status == 200, raw
    assert delta_frame(raw) == delta_frame(b"".join(sent)), raw
    assert targets(synthesizing, request_nonce) == (CLAUDE_TARGET,)


def test_native_refusal_falling_back_to_a_mock_reply_renders_a_fresh_message_lifecycle(synthesizing: Rig) -> None:
    request_nonce: Final = nonce()
    synthesizing.register(request_nonce, claude_backend(refusal(tool_id(request_nonce)), refusal_fields()))
    status, raw = synthesizing.stream(auto_mode_body(REFUSED_MODEL, request_nonce, **NO_CACHE))
    assert status == 200, raw
    started: Final = message_start(raw)
    assert started["stop_reason"] is None, raw
    assert started.get("stop_details") is None and "safeguard_results" not in started, raw
    assert message_delta(raw) == {"stop_reason": "end_turn", "stop_sequence": None}, raw
    assert streamed_text(raw) == MOCK_ANSWER, raw
    assert b'"refusal"' not in raw, raw
    assert targets(synthesizing, request_nonce) == (REFUSED_TARGET,)


def test_native_refusal_falling_back_to_a_cached_intercepted_reply_streams_its_fields_in_message_delta(
    synthesizing: Rig,
) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    text: Final = f"cached answer {request_nonce}"
    results: Final = list(available({}))
    synthesizing.register(
        request_nonce,
        by_target(
            {
                AZURE_REFUSED_TARGET: claude_backend(refusal(identity), refusal_fields()),
                CACHED_FALLBACK_TARGET: claude_backend(
                    answer(f"{identity}c", text), {"stop_details": None, "safeguard_results": results}
                ),
            }
        ),
    )
    cacheable: Final = auto_mode_body(
        CACHED_FALLBACK_MODEL, request_nonce, tools=[dict(WEB_SEARCH_TOOL)], prompt="search the web for"
    )

    def prime() -> tuple[str, ...]:
        primed: Final = synthesizing.post({**cacheable, "stream": False})
        assert primed.status_code == 200, primed.text
        return targets(synthesizing, request_nonce)

    first_prime: Final = prime()
    assert first_prime == (CACHED_FALLBACK_TARGET,), first_prime
    eventually(prime, lambda received: received == ())
    status, raw = synthesizing.stream({**cacheable, "model": AZURE_REFUSED_MODEL})
    assert status == 200, raw
    assert streamed_text(raw) == text, raw
    started: Final = message_start(raw)
    assert started["stop_details"] is None and "safeguard_results" not in started, raw
    assert message_delta(raw) == {"stop_reason": "end_turn", "stop_sequence": None, "safeguard_results": results}, raw
    assert targets(synthesizing, request_nonce) == (AZURE_REFUSED_TARGET,)


def flagged_verdict(explanation: str) -> Mapping[str, JsonValue]:
    return {"type": "evaluated", "outcome": "flagged", "explanation": explanation}


FLAGGED_WITHOUT_EXPLANATION: Final[Mapping[str, JsonValue]] = {"type": "evaluated", "outcome": "flagged"}


def masked_refusal(identity: str) -> tuple[Mapping[str, JsonValue], Mapping[str, JsonValue], Mapping[str, JsonValue]]:
    message: Final = native_message(
        identity, content=(text_block(f"Partial {SECRET}."),), stop_reason="refusal", model="claude-sonnet-4-6"
    )
    expected: Final[Mapping[str, JsonValue]] = {
        "stop_reason": "refusal",
        "stop_sequence": None,
        "stop_details": {**REFUSAL_STOP_DETAILS, "explanation": None},
        "safeguard_results": list(available({})),
    }
    return message, refusal_fields(), expected


def masked_tool_turn(
    identity: str,
) -> tuple[Mapping[str, JsonValue], Mapping[str, JsonValue], Mapping[str, JsonValue]]:
    message: Final = native_message(
        identity,
        content=(text_block(f"Reading {SECRET}."), tool_use_block(identity, name="Read", arguments={"path": "/work"})),
        model="claude-sonnet-4-6",
    )
    sent: Final[Mapping[str, JsonValue]] = {
        "safeguard_results": list(available({identity: dict(flagged_verdict(f"reads {identity}"))}))
    }
    expected: Final[Mapping[str, JsonValue]] = {
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "safeguard_results": list(available({identity: dict(FLAGGED_WITHOUT_EXPLANATION)})),
    }
    return message, sent, expected


MASKED_STREAMS: Final = {"refusal_stop_details": masked_refusal, "verdict_explanation": masked_tool_turn}


@pytest.mark.parametrize("shape", tuple(MASKED_STREAMS))
def test_pipeline_text_rewrite_drops_explanations_from_the_upstream_message_delta(
    synthesizing: Rig, shape: str
) -> None:
    request_nonce: Final = nonce()
    message, sent, expected = MASKED_STREAMS[shape](tool_id(request_nonce))
    synthesizing.register(request_nonce, claude_backend(message, sent))
    body: Final = auto_mode_body(PIPELINE_MODEL, request_nonce, tools=[dict(READ_TOOL)], **NO_CACHE)
    status, raw = synthesizing.stream(body)
    assert status == 200, raw
    assert MASKED in streamed_text(raw) and SECRET not in streamed_text(raw), raw
    assert message_delta(raw) == expected, raw
    assert targets(synthesizing, request_nonce) == (PIPELINE_TARGET,)


def test_pipeline_tool_call_rewrite_keeps_only_the_untouched_tool_verdict(synthesizing: Rig) -> None:
    request_nonce: Final = nonce()
    rewritten: Final = tool_id(request_nonce)
    untouched: Final = f"{rewritten}r"
    message: Final = native_message(
        rewritten,
        content=(
            tool_use_block(rewritten, arguments={"command": f"cat {SECRET}"}),
            tool_use_block(untouched, name="Read", arguments={"path": "/work"}),
        ),
        model="claude-sonnet-4-6",
    )
    verdicts: Final[Mapping[str, JsonValue]] = {
        rewritten: dict(flagged_verdict(f"reads a file {rewritten}")),
        untouched: dict(flagged_verdict(f"reads a directory {untouched}")),
    }
    synthesizing.register(request_nonce, claude_backend(message, {"safeguard_results": list(available(verdicts))}))
    body: Final = auto_mode_body(PIPELINE_MODEL, request_nonce, tools=[dict(BASH_TOOL), dict(READ_TOOL)], **NO_CACHE)
    status, raw = synthesizing.stream(body)
    assert status == 200, raw
    assert stream_tool_use_ids(raw) == (rewritten, untouched), raw
    assert SECRET.encode() not in raw and MASKED.encode() in raw, raw
    assert stream_results(raw) == available({untouched: dict(FLAGGED_WITHOUT_EXPLANATION)}), raw
    assert targets(synthesizing, request_nonce) == (PIPELINE_TARGET,)
