import json
import uuid
from collections.abc import Callable, Mapping
from typing import Final, Literal

import pytest
from integration._support.anthropic_sse import CONTENT_BLOCK_START, CONTENT_TAIL, message_start, text_delta
from integration._support.client import Gateway, Scenario, string_value
from integration._support.openai_wire import answering_model_discovery
from integration._support.responses_stream import data_frames, spend_row, sse_events
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

Ending = Literal["ending-complete", "ending-clean-eof", "ending-done-without-finish", "ending-done-only"]
Family = Literal["openai", "anthropic", "hosted_vllm", "gemini", "langgraph", "agentcore"]

_ENDINGS: Final[tuple[Ending, ...]] = (
    "ending-complete",
    "ending-clean-eof",
    "ending-done-without-finish",
    "ending-done-only",
)
_PARTIAL: Final = ('{"findings":[{"title":', '"unfinished')
_BACKEND: Final = "gpt-5.6"
_CLAUDE: Final = "claude-sonnet-5-5"
_GEMINI: Final = "gemini-3.8-flash"
_AGENT_RUNTIME: Final = "arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/integration-agent"
_PROVIDER_KEY: Final = "synthetic-provider-key"
_CALL_ID_HEADER: Final = "x-litellm-call-id"
_USAGE: Final = {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16}
_TARGETS: Final[Mapping[Family, str]] = {
    "openai": "/chat/completions",
    "hosted_vllm": "/chat/completions",
    "anthropic": "/v1/messages",
    "gemini": f"/models/{_GEMINI}:streamGenerateContent",
    "langgraph": "/runs/stream",
    "agentcore": "/runtimes/",
}
_DEPLOYMENTS: Final[Mapping[Family, str]] = {
    "openai": f"openai/{_BACKEND}",
    "hosted_vllm": f"hosted_vllm/{_BACKEND}",
    "anthropic": f"anthropic/{_CLAUDE}",
    "gemini": f"gemini/{_GEMINI}",
    "langgraph": "langgraph/agent",
    "agentcore": f"bedrock/agentcore/{_AGENT_RUNTIME}",
}


def _data(frame: JsonValue) -> bytes:
    return b"data: " + json.dumps(frame).encode() + b"\n\n"


def _chat_frames(identity: str, ending: Ending) -> tuple[bytes, ...]:
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": _BACKEND}
    content: Final = (
        _data({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": _PARTIAL[0]}}]}),
        _data({**chunk, "choices": [{"index": 0, "delta": {"content": _PARTIAL[1]}, "finish_reason": None}]}),
    )
    match ending:
        case "ending-complete":
            finish: Final = _data({**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            usage: Final = _data({**chunk, "choices": [], "usage": _USAGE})
            return (*content, finish + usage + b"data: [DONE]\n\n")
        case "ending-clean-eof":
            return content
        case "ending-done-without-finish":
            return (*content, b"data: [DONE]\n\n")
        case "ending-done-only":
            return (b"data: [DONE]\n\n",)


def _anthropic_frames(identity: str, ending: Ending) -> tuple[bytes, ...]:
    content: Final = (message_start(identity, _CLAUDE), CONTENT_BLOCK_START, *map(text_delta, _PARTIAL))
    return (*content, CONTENT_TAIL) if ending == "ending-complete" else content


def _gemini_frames(ending: Ending) -> tuple[bytes, ...]:
    """Gemini puts finishReason on the last text chunk and can follow it with a usage-only chunk."""
    finished: Final = ending == "ending-complete"
    content: Final = tuple(
        _data(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": part}], "role": "model"},
                        "index": 0,
                        **({"finishReason": "STOP"} if finished and part == _PARTIAL[-1] else {}),
                    }
                ]
            }
        )
        for part in _PARTIAL
    )
    usage: Final = _data({"usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 7, "totalTokenCount": 16}})
    return (*content, usage) if finished else content


def _langgraph_frames() -> tuple[bytes, ...]:
    """LangGraph ends a run by closing the stream after its metadata event."""
    return (
        *(_data(["messages", [[{"type": "ai", "content": part}, {}]]]) for part in _PARTIAL),
        _data(["metadata", {"run_id": "run-1"}]),
    )


def _agentcore_frames(ending: Ending) -> tuple[bytes, ...]:
    """A runtime that is not Strands can end its turn on messageStop with no metadata or final message."""
    content: Final = tuple(_data({"event": {"contentBlockDelta": {"delta": {"text": part}}}}) for part in _PARTIAL)
    message_stop: Final = _data({"event": {"messageStop": {"stopReason": "end_turn"}}})
    return (*content, message_stop) if ending == "ending-complete" else content


def _frames(family: Family, identity: str, ending: Ending) -> tuple[bytes, ...]:
    match family:
        case "agentcore":
            return _agentcore_frames(ending)
        case "anthropic":
            return _anthropic_frames(identity, ending)
        case "gemini":
            return _gemini_frames(ending)
        case "langgraph":
            return _langgraph_frames()
        case "openai" | "hosted_vllm":
            return _chat_frames(identity, ending)


def _ending(request: Request) -> Ending:
    endings: Final = [ending for ending in _ENDINGS if ending.encode() in request.body]
    assert len(endings) == 1, request.body
    return endings[0]


def _upstream(family: Family) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and _TARGETS[family] in request.target, request.target
        frames: Final = _frames(family, f"upstream-{uuid.uuid4().hex}", _ending(request))
        return Reply(content_type="text/event-stream", chunks=frames)

    return answering_model_discovery(respond)


def _deploy(scenario: Scenario, family: Family, wire: Wire) -> str:
    if family == "agentcore":
        return scenario.model(
            model=_DEPLOYMENTS[family],
            aws_bedrock_runtime_endpoint=wire.url,
            aws_access_key_id="AKIAINTEGRATIONAGENT",
            aws_secret_access_key="synthetic-agentcore-secret",
            aws_region_name="us-west-2",
        )
    return scenario.model(model=_DEPLOYMENTS[family], api_base=wire.url, api_key=_PROVIDER_KEY)


def _prompt(ending: Ending) -> str:
    return f"Return JSON {ending} {uuid.uuid4().hex}"


def _post(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> tuple[str, str]:
    response: Final = gateway.request("POST", path, {**body, "stream": True, "num_retries": 0})
    assert response.status_code == 200, response.text
    return response.text, response.headers[_CALL_ID_HEADER]


def _chat(gateway: Gateway, model: str, ending: Ending) -> tuple[str, str]:
    return _post(
        gateway,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": _prompt(ending)}],
            "stream_options": {"include_usage": True},
        },
    )


def _finish_reasons(text: str) -> tuple[str, ...]:
    return tuple(
        string_value(choice["finish_reason"])
        for frame in data_frames(text)
        for choice in frame.get("choices") or []
        if isinstance(choice, dict) and choice.get("finish_reason")
    )


def _chat_text(text: str) -> str:
    deltas: Final = (
        choice.get("delta")
        for frame in data_frames(text)
        for choice in frame.get("choices") or []
        if isinstance(choice, dict)
    )
    return "".join(string_value(delta.get("content") or "") for delta in deltas if isinstance(delta, dict))


def _chat_errors(text: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(frame for frame in data_frames(text) if "error" in frame)


def _event_types(text: str) -> tuple[str, ...]:
    return tuple(event for event, _ in sse_events(text))


@pytest.mark.parametrize("family", ["openai", "anthropic", "agentcore"])
def test_chat_stream_that_ends_cleanly_without_a_finish_reason_reaches_the_client_as_an_error(
    gateway: Gateway, family: Family
) -> None:
    with wire_server(_upstream(family)) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, family, wire)
        text, call_id = _chat(gateway, model, "ending-clean-eof")
        assert _chat_text(text) == "".join(_PARTIAL), text
        assert _finish_reasons(text) == (), text
        assert len(_chat_errors(text)) == 1, text
        assert spend_row(call_id)["status"] == "failure"


def test_chat_stream_that_sends_done_without_a_finish_reason_reaches_the_client_as_an_error(gateway: Gateway) -> None:
    with wire_server(_upstream("openai")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_DEPLOYMENTS["openai"], api_base=wire.url, api_key=_PROVIDER_KEY)
        text, call_id = _chat(gateway, model, "ending-done-without-finish")
        assert _chat_text(text) == "".join(_PARTIAL), text
        assert _finish_reasons(text) == (), text
        assert len(_chat_errors(text)) == 1, text
        assert spend_row(call_id)["status"] == "failure"


def test_chat_stream_that_sends_only_done_still_completes_empty(gateway: Gateway) -> None:
    """An upstream can answer with only [DONE]: Gemini's OpenAI-compatible endpoint did on 2026-10-20 when
    reasoning used up max_tokens."""
    with wire_server(_upstream("openai")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_DEPLOYMENTS["openai"], api_base=wire.url, api_key=_PROVIDER_KEY)
        text, _ = _chat(gateway, model, "ending-done-only")
        assert _chat_text(text) == "", text
        assert _finish_reasons(text) == ("stop",), text
        assert _chat_errors(text) == (), text
        assert text.rstrip().endswith("data: [DONE]"), text


@pytest.mark.parametrize("family", ["openai", "anthropic", "gemini", "langgraph", "agentcore"])
def test_chat_stream_with_a_provider_finish_reason_still_completes(gateway: Gateway, family: Family) -> None:
    with wire_server(_upstream(family)) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, family, wire)
        text, call_id = _chat(gateway, model, "ending-complete")
        assert _chat_text(text) == "".join(_PARTIAL), text
        assert _finish_reasons(text) == ("stop",), text
        assert _chat_errors(text) == (), text
        assert text.rstrip().endswith("data: [DONE]"), text
        assert spend_row(call_id)["status"] == "success"


def test_chat_stream_from_a_mock_response_deployment_still_completes(gateway: Gateway) -> None:
    """mock_completion streams never carry a finish reason, so a mocked deployment must keep its synthetic stop."""
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model=_DEPLOYMENTS["openai"], mock_response="".join(_PARTIAL))
        text, call_id = _chat(gateway, model, "ending-clean-eof")
        assert _chat_text(text) == "".join(_PARTIAL), text
        assert _finish_reasons(text) == ("stop",), text
        assert _chat_errors(text) == (), text
        assert spend_row(call_id)["status"] == "success"


@pytest.mark.parametrize(("ending", "terminal"), [("ending-complete", True), ("ending-clean-eof", False)])
def test_messages_over_a_chat_bridged_deployment_reports_end_turn_only_for_a_finished_stream(
    gateway: Gateway, ending: Ending, terminal: bool
) -> None:
    with wire_server(_upstream("hosted_vllm")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_DEPLOYMENTS["hosted_vllm"], api_base=wire.url, api_key=_PROVIDER_KEY)
        text, call_id = _post(
            gateway,
            "/v1/messages",
            {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": _prompt(ending)}]},
        )
        types: Final = _event_types(text)
        stop_reasons: Final = tuple(
            data["delta"]["stop_reason"]
            for event, data in sse_events(text)
            if event == "message_delta" and isinstance(data.get("delta"), dict)
        )
        assert types[0] == "message_start", types
        assert ("message_stop" in types) is terminal, types
        assert ("error" in types) is not terminal, types
        assert stop_reasons == (("end_turn",) if terminal else ()), types
        assert spend_row(call_id)["status"] == ("success" if terminal else "failure")


@pytest.mark.parametrize(("ending", "terminal"), [("ending-complete", True), ("ending-clean-eof", False)])
def test_responses_over_a_chat_bridged_anthropic_deployment_completes_only_a_finished_stream(
    gateway: Gateway, ending: Ending, terminal: bool
) -> None:
    with wire_server(_upstream("anthropic")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_DEPLOYMENTS["anthropic"], api_base=wire.url, api_key=_PROVIDER_KEY)
        text, call_id = _post(gateway, "/v1/responses", {"model": model, "input": _prompt(ending)})
        types: Final = tuple(string_value(frame.get("type") or "") for frame in data_frames(text))
        assert types[:1] == ("response.created",), text
        assert ("response.completed" in types) is terminal, text
        assert (types[-1] in {"error", "response.failed"}) is not terminal, text
        assert spend_row(call_id)["status"] == ("success" if terminal else "failure")
