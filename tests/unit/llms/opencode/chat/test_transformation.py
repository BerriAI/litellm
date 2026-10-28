import asyncio
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import RESPONSE_COST_HEADER
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.opencode.common_utils import go_headers


@pytest.fixture
def go_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    for route in ("chat/completions", "messages", "responses", "gemini"):
        monkeypatch.setitem(
            litellm.model_cost,
            f"opencode_go/fixture-{route.split('/')[0]}",
            {
                "litellm_provider": "opencode_go",
                "mode": "responses" if route == "responses" else "chat",
                "supported_endpoints": ["/v1beta/models" if route == "gemini" else f"/v1/{route}"],
                "max_output_tokens": 128000,
                "max_input_tokens": 1000000,
                "input_cost_per_token": 0.000002,
                "output_cost_per_token": 0.000005,
                "supports_function_calling": True,
            },
        )
    monkeypatch.delenv("OPENCODE_GO_API_BASE", raising=False)


def upstream_reply(route: str, model: str) -> Mapping[str, object]:
    if route == "gemini":
        return {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        }
    if route == "messages":
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    if route == "responses":
        return {
            "id": "resp_test",
            "object": "response",
            "created_at": 1700000000,
            "model": model,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_test",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream", ["chat/completions", "messages", "responses", "gemini"])
@pytest.mark.parametrize("ingress", ["chat", "responses", "messages"])
async def test_all_routes_use_model_protocol_and_preserve_session(
    go_catalog: None, upstream: str, ingress: str
) -> None:
    model: Final = f"fixture-{upstream.split('/')[0]}"

    def respond(request: httpx.Request) -> httpx.Response:
        family: Final = "anthropic" if upstream == "messages" else "openai"
        expected_path: Final = (
            f"/inference/go/google/v1beta/models/{model}:generateContent"
            if upstream == "gemini"
            else f"/inference/go/{family}/v1/{upstream}"
        )
        assert request.url.path == expected_path
        assert request.headers["authorization"] == "Bearer fake-go-key"
        assert request.headers["x-opencode-session"] == "conversation-1"
        body: Final = json.loads(request.content)
        if upstream == "gemini":
            assert body["contents"][0]["parts"][0]["text"] == "hello"
        else:
            assert body["model"] == model
            assert "input" in body if upstream == "responses" else "messages" in body
        if upstream == "messages":
            assert body["max_tokens"] == 128000
        return httpx.Response(200, json=upstream_reply(upstream, model))

    with respx.mock(using="httpx") as transport:
        transport.post().mock(side_effect=respond)
        kwargs: Final = {
            "model": f"opencode_go/{model}",
            "api_key": "fake-go-key",
            "litellm_session_id": "conversation-1",
        }
        if ingress == "chat":
            response: Final = await litellm.acompletion(messages=[{"role": "user", "content": "hello"}], **kwargs)
            assert response.choices[0].message.content == "hello"
            assert response.usage.total_tokens == 15
        elif ingress == "responses":
            response_api: Final = await litellm.aresponses(input="hello", **kwargs)
            assert response_api.output[0].content[0].text == "hello"
            assert response_api.usage.total_tokens == 15
        else:
            message: Final = await litellm.anthropic.messages.acreate(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=128000,
                **kwargs,
            )
            assert message["content"][0]["text"] == "hello"
            assert message["usage"]["output_tokens"] == 5


def test_go_never_falls_back_to_other_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    with pytest.raises(litellm.AuthenticationError, match="OPENCODE_API_KEY"):
        go_headers({}, {}, None, "fixture-model")


@pytest.mark.parametrize("input_tokens,expected", [(100, 0.00025), (101, 0.000504)])
def test_go_uses_registered_context_tiers_not_underlying_provider_prices(
    monkeypatch: pytest.MonkeyPatch,
    input_tokens: int,
    expected: float,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "opencode_go/fixture-priced",
        {
            "litellm_provider": "opencode_go",
            "mode": "chat",
            "input_cost_per_token": 0.000002,
            "output_cost_per_token": 0.000005,
            "tiered_pricing": [
                {"range": [0, 100], "input_cost_per_token": 0.000002, "output_cost_per_token": 0.000005},
                {"range": [100, 1000], "input_cost_per_token": 0.000004, "output_cost_per_token": 0.000010},
            ],
        },
    )
    cost: Final = litellm.cost_per_token(
        model="opencode_go/fixture-priced",
        prompt_tokens=input_tokens,
        completion_tokens=10,
    )
    assert sum(cost) == pytest.approx(expected)


@pytest.mark.parametrize("suffix", ["", "/v1", "/v1/messages", "/v1/messages/"])
def test_messages_custom_base_does_not_duplicate_version_or_route(go_catalog: None, suffix: str) -> None:
    from litellm.llms.opencode.messages.transformation import OpenCodeGoMessagesConfig

    assert (
        OpenCodeGoMessagesConfig().get_complete_url(
            f"https://gateway.example/go{suffix}",
            None,
            "fixture-messages",
            {},
            {},
        )
        == "https://gateway.example/go/v1/messages"
    )


@pytest.mark.parametrize("base", [None, "https://gateway.example/go/v1"])
def test_discovery_returns_qualified_ids_and_accepts_response_metadata(base: str | None) -> None:
    url: Final = "https://gateway.example/go/v1/models" if base else "https://opencode.ai/inference/go/v1/models"
    with respx.mock(using="httpx") as transport:
        transport.get(url).respond(
            json={
                "object": "list",
                "data": [
                    {"id": "fixture-chat", "object": "model"},
                    {"id": "fixture-messages", "object": "model"},
                ],
            }
        )
        assert litellm.get_valid_models(
            check_provider_endpoint=True,
            custom_llm_provider="opencode_go",
            api_key="fake-go-key",
            api_base=base,
        ) == [
            "opencode_go/fixture-chat",
            "opencode_go/fixture-messages",
        ]


@pytest.mark.parametrize("explicit_base", [None, "https://explicit-gateway.example/go/v1"])
def test_discovery_honors_environment_base_and_explicit_override(
    monkeypatch: pytest.MonkeyPatch, explicit_base: str | None
) -> None:
    from litellm.llms.opencode.chat.transformation import OpenCodeGoChatConfig

    monkeypatch.setenv("OPENCODE_GO_API_BASE", "https://env-gateway.example/go/v1/")
    expected_base: Final = explicit_base or "https://env-gateway.example/go/v1"
    with respx.mock(using="httpx") as transport:
        transport.get(f"{expected_base}/models").respond(json={"data": [{"id": "gateway-only-model"}]})
        assert OpenCodeGoChatConfig().get_models(api_key="fake-go-key", api_base=explicit_base) == [
            "opencode_go/gateway-only-model"
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("ingress", ["chat", "responses", "messages"])
async def test_gemini_streams_text_and_usage_through_all_routes(go_catalog: None, ingress: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/inference/go/google/v1beta/models/fixture-gemini:streamGenerateContent"
        assert request.url.query == b"alt=sse"
        assert "stream" not in json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(upstream_reply("gemini", "fixture-gemini")) + "\n\n",
        )

    with respx.mock(using="httpx") as transport:
        transport.post().mock(side_effect=respond)
        kwargs: Final = {"model": "opencode_go/fixture-gemini", "api_key": "fake-go-key", "stream": True}
        if ingress == "chat":
            stream: Final = await litellm.acompletion(
                messages=[{"role": "user", "content": "hello"}], stream_options={"include_usage": True}, **kwargs
            )
            chunks: Final = [chunk async for chunk in stream]
            assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "hello"
            assert chunks[-1].usage.total_tokens == 15
        elif ingress == "responses":
            responses_stream: Final = await litellm.aresponses(input="hello", **kwargs)
            events: Final = [event async for event in responses_stream]
            assert "".join(event.delta for event in events if event.type == "response.output_text.delta") == "hello"
            assert events[-1].response.usage.total_tokens == 15
        else:
            messages_stream: Final = await litellm.anthropic.messages.acreate(
                messages=[{"role": "user", "content": "hello"}], max_tokens=128000, **kwargs
            )
            wire_events: Final = b"".join([event async for event in messages_stream]).decode()
            message_events: Final = [
                json.loads(line[6:]) for line in wire_events.splitlines() if line.startswith("data: ")
            ]
            assert (
                "".join(
                    event["delta"].get("text", "") for event in message_events if event["type"] == "content_block_delta"
                )
                == "hello"
            )
            assert (
                next(event["usage"]["output_tokens"] for event in message_events if event["type"] == "message_delta")
                == 5
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("factory_path", [False, True])
@pytest.mark.parametrize("async_stream", [False, True])
async def test_gemini_streams_with_dispatcher_and_factory_configs(
    go_catalog: None, factory_path: bool, async_stream: bool
) -> None:
    messages: Final = [{"role": "user", "content": "hello"}]
    logging_obj: Final = Logging(
        model="fixture-gemini",
        messages=messages,
        stream=True,
        call_type="acompletion" if async_stream else "completion",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="go-stream-regression",
        function_id="go-stream-regression",
    )
    logging_obj.update_environment_variables(
        model="fixture-gemini",
        user="",
        optional_params={"stream_options": {"include_usage": True}},
        litellm_params={},
        custom_llm_provider="opencode_go",
        stream_options={"include_usage": True},
    )
    with respx.mock(using="httpx") as transport:
        transport.post(
            "https://opencode.ai/inference/go/google/v1beta/models/fixture-gemini:streamGenerateContent?alt=sse"
        ).respond(
            headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(upstream_reply("gemini", "fixture-gemini")) + "\n\n",
        )
        if factory_path:
            result: Final = BaseLLMHTTPHandler().completion(
                model="fixture-gemini",
                messages=messages,
                api_base=None,
                custom_llm_provider="opencode_go",
                model_response=litellm.ModelResponse(),
                encoding=None,
                logging_obj=logging_obj,
                optional_params={"stream_options": {"include_usage": True}},
                timeout=30,
                litellm_params={},
                acompletion=async_stream,
                stream=True,
                api_key="fake-go-key",
            )
        else:
            completion: Final = litellm.acompletion if async_stream else litellm.completion
            result: Final = completion(
                model="opencode_go/fixture-gemini",
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                api_key="fake-go-key",
            )
        if async_stream:
            stream: Final = await result
            chunks: Final = [chunk async for chunk in stream]
        else:
            chunks: Final = tuple(result)
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "hello"
        assert chunks[-1].usage.total_tokens == 15


@pytest.mark.parametrize("suffix", ["", "/v1beta", "/v1beta/models", "/v1beta/models/"])
def test_gemini_custom_base_normalizes_version_and_models(go_catalog: None, suffix: str) -> None:
    with respx.mock(using="httpx") as transport:
        transport.post("https://gateway.example/google/v1beta/models/fixture-gemini:generateContent").respond(
            json=upstream_reply("gemini", "fixture-gemini")
        )
        response: Final = litellm.completion(
            model="opencode_go/fixture-gemini",
            api_key="fake-go-key",
            api_base=f"https://gateway.example/google{suffix}",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert response.choices[0].message.content == "hello"


def test_gemini_sync_stream_preserves_tool_calls_and_usage(go_catalog: None) -> None:
    chunk: Final = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
                            "thoughtSignature": "synthetic-signature",
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
    }
    with respx.mock(using="httpx") as transport:
        transport.post(
            "https://opencode.ai/inference/go/google/v1beta/models/fixture-gemini:streamGenerateContent?alt=sse"
        ).respond(headers={"content-type": "text/event-stream"}, text="data: " + json.dumps(chunk) + "\n\n")
        stream: Final = litellm.completion(
            model="opencode_go/fixture-gemini",
            api_key="fake-go-key",
            stream=True,
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            stream_options={"include_usage": True},
        )
        chunks: Final = tuple(stream)
        tool: Final = next(
            c.choices[0].delta.tool_calls[0] for c in chunks if c.choices and c.choices[0].delta.tool_calls
        )
        assert tool.function.name == "get_weather"
        assert json.loads(tool.function.arguments) == {"city": "Paris"}
        assert any(c.choices[0].finish_reason == "tool_calls" for c in chunks if c.choices)
        assert chunks[-1].usage.total_tokens == 15


GO_REPORTED_USAGE: Final = {
    "cost": 0,
    "opencode": {"billing_source": "go", "list_cost": 0.00412, "allowance": 0.00412},
}
ROW_PRICE_FOR_FIXTURE_USAGE: Final = 10 * 0.000002 + 5 * 0.000005


def go_priced_reply(route: str, model: str, usage_extra: Mapping[str, object]) -> Mapping[str, object]:
    reply: Final = upstream_reply(route, model)
    key: Final = "usageMetadata" if route == "gemini" else "usage"
    usage: Final = TypeAdapter(Mapping[str, object]).validate_python(reply[key])
    return {**reply, key: {**usage, **usage_extra}}


class CostRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.costs: list[float | None] = []  # mutable-ok: test probe accumulates logged costs

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:  # noqa: ANN001
        self.costs.append(kwargs.get("response_cost"))


async def logged_cost(recorder: CostRecorder) -> float | None:
    for _ in range(50):
        if recorder.costs:
            return recorder.costs[-1]
        await asyncio.sleep(0.05)
    raise AssertionError("success callback never ran")


@pytest.fixture
def cost_recorder(monkeypatch: pytest.MonkeyPatch) -> CostRecorder:
    recorder: Final = CostRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    return recorder


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream", ["chat/completions", "messages", "responses", "gemini"])
@pytest.mark.parametrize("ingress", ["chat", "responses"])
async def test_reported_list_cost_is_billed_instead_of_rows(
    go_catalog: None, cost_recorder: CostRecorder, upstream: str, ingress: str
) -> None:
    model: Final = f"fixture-{upstream.split('/')[0]}"
    with respx.mock(using="httpx") as transport:
        transport.post().mock(
            return_value=httpx.Response(200, json=go_priced_reply(upstream, model, GO_REPORTED_USAGE))
        )
        kwargs: Final = {"model": f"opencode_go/{model}", "api_key": "fake-go-key"}
        if ingress == "chat":
            response: Final = await litellm.acompletion(messages=[{"role": "user", "content": "hello"}], **kwargs)
        else:
            response = await litellm.aresponses(input="hello", **kwargs)
    assert response._hidden_params["additional_headers"][RESPONSE_COST_HEADER] == pytest.approx(0.00412)
    assert await logged_cost(cost_recorder) == pytest.approx(0.00412)


@pytest.mark.asyncio
@pytest.mark.parametrize("usage_extra", [{}, {"cost": 0}, {"opencode": {"billing_source": "go"}}])
async def test_rows_price_the_request_when_no_list_cost_is_reported(
    go_catalog: None, cost_recorder: CostRecorder, usage_extra: Mapping[str, object]
) -> None:
    with respx.mock(using="httpx") as transport:
        transport.post().mock(
            return_value=httpx.Response(200, json=go_priced_reply("chat/completions", "fixture-chat", usage_extra))
        )
        response: Final = await litellm.acompletion(
            model="opencode_go/fixture-chat", api_key="fake-go-key", messages=[{"role": "user", "content": "hello"}]
        )
    assert RESPONSE_COST_HEADER not in response._hidden_params.get("additional_headers", {})
    assert await logged_cost(cost_recorder) == pytest.approx(ROW_PRICE_FOR_FIXTURE_USAGE)


def sse(*events: Mapping[str, object]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_chat_stream_bills_the_final_usage_chunks_list_cost(
    go_catalog: None, cost_recorder: CostRecorder
) -> None:
    chunk: Final = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "fixture-chat",
    }
    with respx.mock(using="httpx") as transport:
        transport.post().mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=sse(
                    {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}}]},
                    {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {
                        **chunk,
                        "choices": [],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, **GO_REPORTED_USAGE},
                    },
                ),
            )
        )
        stream: Final = await litellm.acompletion(
            model="opencode_go/fixture-chat",
            api_key="fake-go-key",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks: Final = [piece async for piece in stream]
    assert "".join(piece.choices[0].delta.content or "" for piece in chunks if piece.choices) == "hello"
    assert await logged_cost(cost_recorder) == pytest.approx(0.00412)


@pytest.mark.asyncio
async def test_responses_stream_bills_the_completed_events_list_cost(
    go_catalog: None, cost_recorder: CostRecorder
) -> None:
    completed: Final = go_priced_reply("responses", "fixture-responses", GO_REPORTED_USAGE)
    with respx.mock(using="httpx") as transport:
        transport.post().mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=sse(
                    {
                        "type": "response.output_text.delta",
                        "delta": "hello",
                        "item_id": "msg_test",
                        "output_index": 0,
                        "content_index": 0,
                        "sequence_number": 1,
                    },
                    {"type": "response.completed", "response": completed, "sequence_number": 2},
                ),
            )
        )
        stream: Final = await litellm.aresponses(
            model="opencode_go/fixture-responses", api_key="fake-go-key", input="hello", stream=True
        )
        events: Final = [event async for event in stream]
    assert events[-1].type == "response.completed"
    assert events[-1].response._hidden_params["additional_headers"][RESPONSE_COST_HEADER] == pytest.approx(0.00412)
    assert await logged_cost(cost_recorder) == pytest.approx(0.00412)


def test_stream_chunk_builder_reads_list_cost_from_dict_chunks():
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj = Logging(
        model="opencode_go/glm-5.3",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="dict-chunks",
        function_id="1",
    )
    logging_obj.model_call_details["custom_llm_provider"] = "opencode_go"
    chunk = {
        "id": "chatcmpl-go",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "glm-5.3",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    }
    final_usage = {**chunk, "choices": [], "usage": dict(GO_REPORTED_USAGE)}

    assembled = litellm.stream_chunk_builder(chunks=[chunk, final_usage], logging_obj=logging_obj)

    assert assembled is not None
    assert assembled._hidden_params["response_cost"] == GO_REPORTED_USAGE["opencode"]["list_cost"]
