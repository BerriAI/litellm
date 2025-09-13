import json
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.router_utils.pattern_match_deployments import PatternMatchRouter
import pytest
from typing import Any, AsyncIterator


from litellm.proxy.anthropic_endpoints.service import AnthropicMessagesService


class FakeProxyLogging:
    async def pre_call_hook(self, user_api_key_dict, data, call_type):
        return data

    async def during_call_hook(self, data, user_api_key_dict, call_type):
        return None

    def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict, response, request_data
    ):
        return response

    async def async_post_call_streaming_hook(
        self, user_api_key_dict, response, data, str_so_far=""
    ):
        return response

    async def post_call_success_hook(self, data, user_api_key_dict, response):
        return response


class FakeModelResponse:
    def __init__(self, content: str = "hello") -> None:
        self._hidden_params = {
            "model_id": "fake-model",
            "cache_key": "cache-1",
            "api_base": "http://example.com",
            "response_cost": "0.001",
        }
        self.choices = [{"message": {"role": "assistant", "content": content}}]


class FakeStream:
    def __init__(self) -> None:
        self._hidden_params = {
            "model_id": "fake-model",
            "cache_key": "cache-2",
            "api_base": "http://example.com",
            "response_cost": "0.002",
        }
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._sent:
            self._sent = True
            return {"delta": "hello "}
        raise StopAsyncIteration


class FakeRouter:
    def __init__(self) -> None:
        self.model_names = {"test-openai-model"}
        self.deployment_names: set[str] = set()
        self.model_group_alias = None

        self.pattern_router = PatternMatchRouter()
        self._last_kwargs = None

    def get_model_ids(self):
        return set(self.model_names)

    async def acompletion(self, **kwargs):
        self._last_kwargs = kwargs
        if kwargs.get("stream"):
            return FakeStream()
        return FakeModelResponse(content="world")


class FakeAnthropicAdapter:
    def translate_completion_input_params(self, data: dict):
        msgs = data.get("messages") or []
        content = ""
        if msgs and isinstance(msgs[0], dict):
            c = msgs[0].get("content")
            if isinstance(c, list):
                content = "".join([x.get("text", "") for x in c if isinstance(x, dict)])
            elif isinstance(c, str):
                content = c
        return {
            "model": "test-openai-model",
            "messages": [{"role": "user", "content": content}],
            "stream": data.get("stream", False),
        }


class FakeAnthropicMessagesAdapter:
    def translate_openai_response_to_anthropic(self, response):
        text = ""
        try:
            text = response.choices[0]["message"]["content"]
        except Exception:
            text = "ok"
        return {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-fake",
            "content": [{"type": "text", "text": text}],
        }


class FakeAnthropicStreamWrapper:
    def __init__(self, completion_stream: AsyncIterator[Any], model: str):
        self._stream = completion_stream
        self._model = model

    async def async_anthropic_sse_wrapper(self):
        yield b"event: message_start\ndata: {}\n\n"
        async for c in self._stream:
            payload = json.dumps(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": str(c.get("delta", ""))},
                }
            )
            yield ("data: " + payload + "\n\n").encode("utf-8")
        yield b"event: message_stop\ndata: {}\n\n"


@pytest.mark.asyncio
async def test_build_openai_request_and_route_non_stream(monkeypatch):
    service = AnthropicMessagesService(
        anthropic_adapter_factory=lambda: FakeAnthropicAdapter(),
        anthropic_messages_adapter_factory=lambda: FakeAnthropicMessagesAdapter(),
        anthropic_stream_wrapper_cls=FakeAnthropicStreamWrapper,
    )

    fake_logging = FakeProxyLogging()
    llm_router = FakeRouter()

    data = {
        "model": "claude-4-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "stop_sequences": ["END"],
    }

    openai_data = await service.build_openai_request(
        data,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
        proxy_logging_obj=fake_logging,
    )
    assert openai_data["model"] == "test-openai-model"
    assert openai_data.get("stop") == ["END"]

    response = await service.route_and_call(
        openai_data,
        proxy_logging_obj=fake_logging,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
        llm_router=llm_router,
        user_model=None,
    )
    anth = service.to_anthropic_response(openai_response=response)
    assert anth.get("type") == "message"
    assert anth.get("role") == "assistant"
    assert isinstance(anth.get("content"), list)


@pytest.mark.asyncio
async def test_wrap_openai_stream_as_anthropic(monkeypatch):
    service = AnthropicMessagesService(
        anthropic_adapter_factory=lambda: FakeAnthropicAdapter(),
        anthropic_messages_adapter_factory=lambda: FakeAnthropicMessagesAdapter(),
        anthropic_stream_wrapper_cls=FakeAnthropicStreamWrapper,
    )

    fake_logging = FakeProxyLogging()
    llm_router = FakeRouter()

    data = {
        "model": "claude-4-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    openai_data = await service.build_openai_request(
        data,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
        proxy_logging_obj=fake_logging,
    )

    response_stream = await service.route_and_call(
        openai_data,
        proxy_logging_obj=fake_logging,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
        llm_router=llm_router,
        user_model=None,
    )

    # Now wrap as Anthropic SSE text iterator
    sse_texts = []
    async for s in service.wrap_openai_stream_as_anthropic(
        openai_stream=response_stream,
        model=data["model"],
        proxy_logging_obj=fake_logging,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="1234",
        ),
        request_data=openai_data,
    ):
        sse_texts.append(s)

    assert any("event: message_start" in t for t in sse_texts)
    assert any("event: message_stop" in t for t in sse_texts)
