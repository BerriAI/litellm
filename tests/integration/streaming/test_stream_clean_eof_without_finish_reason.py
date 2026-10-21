import asyncio
import itertools
import json
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Final, Literal

import anthropic
import openai
import pytest
from integration._support.anthropic_sse import CONTENT_BLOCK_START, CONTENT_TAIL, message_start, text_delta
from integration._support.client import Gateway, Scenario, eventually, string_value
from integration._support.database import read_rows, scratch_database
from integration._support.database_relay import DatabaseRelay, database_relay
from integration._support.openai_wire import answering_model_discovery, posted_targets
from integration._support.process import owned_proxy
from integration._support.responses_stream import data_frames, spend_row, sse_events
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._count_tokens_system_lift import (
    anthropic_client,
    async_anthropic_client,
    async_openai_client,
    openai_client,
)
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


def _event(name: str, frame: JsonValue) -> bytes:
    return f"event: {name}\n".encode() + _data(frame)


def _langgraph_frames() -> tuple[bytes, ...]:
    """A LangGraph server's run stream in its default mode, its run metadata first and then the state after each step"""
    question: Final = {"type": "human", "content": "Answer in two steps."}
    answers: Final = tuple({"type": "ai", "content": part} for part in _PARTIAL)
    return (
        _event("metadata", {"run_id": "run-1", "attempt": 1}),
        *(_event("values", {"messages": [question, *answers[:step]]}) for step in range(len(answers) + 1)),
    )


def _agentcore_frames(ending: Ending) -> tuple[bytes, ...]:
    """A runtime that forwards Bedrock Converse events without the metadata event, so its turn ends on messageStop"""
    start: Final = _data({"event": {"messageStart": {"role": "assistant"}}})
    content: Final = tuple(
        _data({"event": {"contentBlockDelta": {"delta": {"text": part}, "contentBlockIndex": 0}}}) for part in _PARTIAL
    )
    end: Final = (
        _data({"event": {"contentBlockStop": {"contentBlockIndex": 0}}}),
        _data({"event": {"messageStop": {"stopReason": "end_turn"}}}),
    )
    return (start, *content, *end) if ending == "ending-complete" else (start, *content)


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


def _upstream(family: Family, ending: Ending | None = None) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and _TARGETS[family] in request.target, request.target
        frames: Final = _frames(family, f"upstream-{uuid.uuid4().hex}", ending or _ending(request))
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


def _choices(text: str) -> Iterator[Mapping[str, JsonValue]]:
    for frame in data_frames(text):
        yield from (choice for choice in frame.get("choices") or [] if isinstance(choice, dict))


def _finish_reasons(text: str) -> tuple[str, ...]:
    return tuple(string_value(choice["finish_reason"]) for choice in _choices(text) if choice.get("finish_reason"))


def _chat_text(text: str) -> str:
    deltas: Final = (choice.get("delta") for choice in _choices(text))
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


def _drain_openai_sdk(gateway: Gateway, model: str, sync_mode: bool) -> None:
    messages: Final = [{"role": "user", "content": _prompt("ending-clean-eof")}]
    if sync_mode:
        for _ in openai_client(gateway).chat.completions.create(model=model, messages=messages, stream=True):
            pass
        return

    async def drain() -> None:
        stream: Final = await async_openai_client(gateway).chat.completions.create(
            model=model, messages=messages, stream=True
        )
        async for _ in stream:
            pass

    asyncio.run(drain())


def _drain_anthropic_sdk(gateway: Gateway, model: str, sync_mode: bool) -> None:
    messages: Final = [{"role": "user", "content": _prompt("ending-clean-eof")}]
    if sync_mode:
        for _ in anthropic_client(gateway).messages.create(model=model, max_tokens=64, messages=messages, stream=True):
            pass
        return

    async def drain() -> None:
        stream: Final = await async_anthropic_client(gateway).messages.create(
            model=model, max_tokens=64, messages=messages, stream=True
        )
        async for _ in stream:
            pass

    asyncio.run(drain())


@pytest.mark.parametrize("sync_mode", [True, False])
def test_openai_sdk_stream_cut_before_a_finish_reason_raises(gateway: Gateway, sync_mode: bool) -> None:
    with wire_server(_upstream("openai")) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, "openai", wire)
        with pytest.raises(openai.APIError, match="Stream ended before a finish_reason was received"):
            _drain_openai_sdk(gateway, model, sync_mode)


@pytest.mark.parametrize("sync_mode", [True, False])
def test_anthropic_sdk_messages_stream_cut_before_a_finish_reason_raises(gateway: Gateway, sync_mode: bool) -> None:
    with wire_server(_upstream("hosted_vllm")) as wire, gateway.scenario() as scenario:
        model: Final = _deploy(scenario, "hosted_vllm", wire)
        with pytest.raises(anthropic.APIError, match="Stream ended before a finish_reason was received"):
            _drain_anthropic_sdk(gateway, model, sync_mode)


def test_responses_stream_cut_on_its_deployment_finishes_on_the_request_fallback(gateway: Gateway) -> None:
    with (
        wire_server(_upstream("anthropic")) as primary,
        wire_server(_upstream("anthropic", "ending-complete")) as backup,
        gateway.scenario() as scenario,
    ):
        model: Final = _deploy(scenario, "anthropic", primary)
        fallback: Final = _deploy(scenario, "anthropic", backup)
        text, _ = _post(
            gateway, "/v1/responses", {"model": model, "input": _prompt("ending-clean-eof"), "fallbacks": [fallback]}
        )
        types: Final = tuple(string_value(frame.get("type") or "") for frame in data_frames(text))
        assert "response.completed" in types and "response.failed" not in types, text
        assert (len(posted_targets(primary)), len(posted_targets(backup))) == (1, 1), text


def test_chat_stream_cut_after_its_content_is_not_replayed_on_the_request_fallback(gateway: Gateway) -> None:
    with (
        wire_server(_upstream("openai")) as primary,
        wire_server(_upstream("openai", "ending-complete")) as backup,
        gateway.scenario() as scenario,
    ):
        model: Final = _deploy(scenario, "openai", primary)
        fallback: Final = _deploy(scenario, "openai", backup)
        text, _ = _post(
            gateway,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": _prompt("ending-clean-eof")}],
                "fallbacks": [fallback],
            },
        )
        assert _chat_text(text) == "".join(_PARTIAL), text
        assert len(_chat_errors(text)) == 1, text
        assert posted_targets(backup) == (), text


Route = Literal["/v1/chat/completions", "/v1/messages", "/v1/responses"]


def _route_body(route: Route, model: str, ending: Ending) -> Mapping[str, JsonValue]:
    match route:
        case "/v1/chat/completions":
            return {"model": model, "messages": [{"role": "user", "content": _prompt(ending)}]}
        case "/v1/messages":
            return {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": _prompt(ending)}]}
        case "/v1/responses":
            return {"model": model, "input": _prompt(ending)}


def _finished(route: Route, text: str) -> bool:
    match route:
        case "/v1/chat/completions":
            return _finish_reasons(text) == ("stop",) and _chat_errors(text) == ()
        case "/v1/messages":
            return "message_stop" in _event_types(text) and "error" not in _event_types(text)
        case "/v1/responses":
            return "response.completed" in tuple(string_value(frame.get("type") or "") for frame in data_frames(text))


def _stream_outcome(gateway: Gateway, model: str, route: Route, ending: Ending) -> tuple[bool, str]:
    text, call_id = _post(gateway, route, _route_body(route, model, ending))
    return _finished(route, text), call_id


_BRIDGED_FAMILY: Final[Mapping[Route, Family]] = {
    "/v1/chat/completions": "openai",
    "/v1/messages": "hosted_vllm",
    "/v1/responses": "anthropic",
}


def test_concurrent_burst_ends_each_stream_by_whether_its_provider_finished_it(gateway: Gateway) -> None:
    endings: Final[tuple[Ending, ...]] = ("ending-complete", "ending-clean-eof")
    cases: Final = tuple(itertools.product(_BRIDGED_FAMILY, endings, range(5)))
    with ExitStack() as wires, gateway.scenario() as scenario:
        models: Final = {
            route: _deploy(scenario, family, wires.enter_context(wire_server(_upstream(family))))
            for route, family in _BRIDGED_FAMILY.items()
        }
        with ThreadPoolExecutor(max_workers=12) as pool:
            outcomes: Final = tuple(
                pool.map(lambda case: _stream_outcome(gateway, models[case[0]], case[0], case[1]), cases)
            )
        assert tuple(finished for finished, _ in outcomes) == tuple(
            ending == "ending-complete" for _, ending, _ in cases
        ), outcomes
        assert tuple(spend_row(call_id)["status"] for _, call_id in outcomes) == tuple(
            "success" if ending == "ending-complete" else "failure" for _, ending, _ in cases
        )


_SPEND_LOG_INSERT: Final = b'INSERT INTO "public"."LiteLLM_SpendLogs"'
Outcomes = tuple[tuple[bool, str], ...]


def _spend_statuses(database_url: str) -> Counter[tuple[str, str]]:
    rows: Final = read_rows('SELECT litellm_call_id, status FROM "LiteLLM_SpendLogs"', (), database_url=database_url)
    return Counter((string_value(row["litellm_call_id"]), string_value(row["status"])) for row in rows)


def _bursts_until_recovered(relay: DatabaseRelay, burst: Callable[[], Outcomes]) -> Iterator[tuple[bool, Outcomes]]:
    """Bursts until the proxy reconnects after the outage, each marked by whether the database was down when it began"""
    while not relay.reconnected.is_set():
        yield relay.tripped.is_set(), burst()


def test_spend_log_outage_mid_burst_keeps_each_stream_ending_and_lands_every_row_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    endings: Final[tuple[Ending, ...]] = ("ending-complete", "ending-clean-eof")
    cases: Final = tuple(itertools.product(_BRIDGED_FAMILY, endings))
    with (
        scratch_database() as database_url,
        database_relay(database_url, _SPEND_LOG_INSERT) as (relay, relayed_url),
        owned_proxy(gateway, tmp_path, {"DATABASE_URL": relayed_url}, workers=2) as candidate,
        ExitStack() as wires,
        candidate.scenario() as scenario,
        ThreadPoolExecutor(max_workers=len(cases)) as pool,
    ):
        models: Final = {
            route: _deploy(scenario, family, wires.enter_context(wire_server(_upstream(family))))
            for route, family in _BRIDGED_FAMILY.items()
        }

        def burst() -> Outcomes:
            return tuple(pool.map(lambda case: _stream_outcome(candidate, models[case[0]], case[0], case[1]), cases))

        relay.arm()
        waves: Final = tuple(itertools.islice(_bursts_until_recovered(relay, burst), 60))
        assert relay.reconnected.is_set(), "the proxy never reconnected to the database after the outage"
        assert relay.refused >= 1 and any(down for down, _ in waves), (relay.refused, len(waves))
        finished: Final = tuple(ending == "ending-complete" for _, ending in cases)
        assert all(tuple(done for done, _ in outcomes) == finished for _, outcomes in waves), waves
        sent: Final = tuple(itertools.chain.from_iterable(outcomes for _, outcomes in waves))
        expected: Final = Counter(
            (call_id, "success" if ending == "ending-complete" else "failure")
            for (_, ending), (_, call_id) in zip(cases * len(waves), sent, strict=True)
        )
        landed: Final = eventually(
            lambda: _spend_statuses(database_url),
            lambda rows: {call_id for call_id, _ in expected} <= {call_id for call_id, _ in rows},
            seconds=120,
        )
        assert landed == expected
