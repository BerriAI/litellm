import asyncio
import json
from copy import deepcopy
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.caching.caching import Cache
from litellm.caching.caching_handler import _PENDING_CACHE_WRITES

URL: Final = "https://reasoning-test.invalid/v1/chat/completions"
MODEL: Final = "hosted_vllm/reasoning-test"
REASONING: Final = "Inspect both tool results before answering."


def _messages():
    return [
        {"role": "user", "content": "Compare both records"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": REASONING,
            "tool_calls": [
                {"id": f"call_{index}", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                for index in (1, 2)
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "first record"},
        {"role": "tool", "tool_call_id": "call_2", "content": "second record"},
    ]


def _route(mock: respx.MockRouter):
    return mock.post(URL).respond(
        200,
        json={
            "id": "chatcmpl-reasoning-test",
            "object": "chat.completion",
            "created": 1,
            "model": "reasoning-test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Compared"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        },
    )


def _assert_wire(request: httpx.Request, enabled: bool):
    body: Final = json.loads(request.content)
    assert body["model"] == "reasoning-test"
    assert "forward_reasoning_content" not in body
    assert "forward_reasoning_content" not in request.content.decode()
    messages: Final = body["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "tool"]
    assert [tool["id"] for tool in messages[1]["tool_calls"]] == ["call_1", "call_2"]
    assert [message["tool_call_id"] for message in messages[2:]] == ["call_1", "call_2"]
    assert [message["content"] for message in messages[2:]] == ["first record", "second record"]
    assert messages[1].get("reasoning_content") == (REASONING if enabled else None)
    assert request.content.decode().count(REASONING) == int(enabled)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["completion", "acompletion"])
@pytest.mark.parametrize("forward", [None, False, True], ids=["absent", "false", "true"])
async def test_direct_completion_reasoning_flag_reaches_final_wire(
    async_mode: bool, forward: bool | None, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    messages: Final = _messages()
    original: Final = deepcopy(messages)
    kwargs: Final = {
        "model": MODEL,
        "api_base": URL.removesuffix("/chat/completions"),
        "api_key": "test-key",
        "messages": messages,
        **({} if forward is None else {"forward_reasoning_content": forward}),
    }
    with respx.mock(assert_all_called=True) as mock:
        route: Final = _route(mock)
        response: Final = await litellm.acompletion(**kwargs) if async_mode else litellm.completion(**kwargs)
        assert response.choices[0].message.content == "Compared"
        assert route.call_count == 1
        _assert_wire(route.calls[0].request, forward is True)
    assert messages == original


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["completion", "acompletion"])
async def test_router_aliases_isolate_reasoning_flag_on_same_backend(async_mode: bool, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    model_list: Final = [
        {
            "model_name": alias,
            "litellm_params": {
                "model": MODEL,
                "api_base": URL.removesuffix("/chat/completions"),
                "api_key": "test-key",
                **({} if forward is None else {"forward_reasoning_content": forward}),
            },
        }
        for alias, forward in (("default", None), ("disabled", False), ("enabled", True))
    ]
    original_models: Final = deepcopy(model_list)
    router: Final = Router(model_list=model_list, num_retries=0)
    messages: Final = _messages()
    original_messages: Final = deepcopy(messages)
    with respx.mock(assert_all_called=True) as mock:
        route: Final = _route(mock)
        for alias in ("enabled", "default", "disabled", "enabled"):
            response = (
                await router.acompletion(model=alias, messages=messages)
                if async_mode
                else router.completion(model=alias, messages=messages)
            )
            assert response.choices[0].message.content == "Compared"
            _assert_wire(route.calls[-1].request, alias == "enabled")
            assert messages == original_messages
        assert route.call_count == 4
    assert model_list == original_models


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("surface", ["sdk", "router", "responses", "router-responses"])
@pytest.mark.parametrize("provider", ["hosted_vllm", "openai"])
@pytest.mark.parametrize("normalize", [False, True])
async def test_local_cache_separates_forwarded_reasoning_history(
    async_mode: bool, surface: str, provider: str, normalize: bool, monkeypatch: pytest.MonkeyPatch
):
    if provider == "openai" and not normalize:
        pytest.skip("OpenAI does not apply hosted_vllm forwarding policy")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "cache", Cache(type="local", namespace="reasoning-cache-test"))
    messages: Final = _messages()
    original_messages: Final = deepcopy(messages)
    input_items: Final = [
        {"role": "user", "content": "Compare both records"},
        {
            "type": "reasoning",
            "id": "rs_previous",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": REASONING}],
        },
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call", "call_id": "call_2", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "first record"},
        {"type": "function_call_output", "call_id": "call_2", "output": "second record"},
    ]
    original_input: Final = deepcopy(input_items)
    router: Final = Router(
        model_list=[
            {
                "model_name": alias,
                "litellm_params": {
                    "model": f"{provider}/reasoning-test",
                    "api_base": URL.removesuffix("/chat/completions"),
                    "api_key": "test-key",
                    "use_chat_completions_api": True,
                    **(
                        {
                            "forward_reasoning_content": True,
                            **(
                                {}
                                if enabled is None
                                else {"reasoning_content_field": "reasoning" if enabled else "reasoning_content"}
                            ),
                        }
                        if normalize
                        else ({} if enabled is None else {"forward_reasoning_content": enabled})
                    ),
                },
            }
            for alias, enabled in (("default", None), ("disabled", False), ("enabled", True))
        ],
        cache_responses=True,
        caching_groups=[("default", "disabled", "enabled")],
        num_retries=0,
    )

    def backend(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        assert "forward_reasoning_content" not in body
        assert "reasoning_content_field" not in body
        enabled: Final = body["messages"][1].get("reasoning" if normalize else "reasoning_content") == REASONING
        return httpx.Response(
            200,
            json={
                "id": "provider-cache-reasoning",
                "object": "chat.completion",
                "created": 1,
                "model": "reasoning-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "forwarded" if enabled else "omitted"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    with respx.mock(assert_all_called=True) as mock:
        route: Final = mock.post(URL).mock(side_effect=backend)
        for alias, forward, expected_calls in (
            ("default", None, 1),
            ("disabled", False, 1),
            ("enabled", True, 2),
            ("disabled", False, 2),
            ("enabled", True, 2),
        ):
            kwargs: Final = {
                "model": f"{provider}/reasoning-test",
                "api_base": URL.removesuffix("/chat/completions"),
                "api_key": "test-key",
                "caching": True,
                **(
                    {
                        "forward_reasoning_content": True,
                        **(
                            {}
                            if forward is None
                            else {"reasoning_content_field": "reasoning" if forward else "reasoning_content"}
                        ),
                    }
                    if normalize
                    else ({} if forward is None else {"forward_reasoning_content": forward})
                ),
            }
            if surface == "router-responses":
                response = (
                    await router.aresponses(model=alias, input=input_items)
                    if async_mode
                    else router.responses(model=alias, input=input_items)
                )
            elif surface == "router":
                response = (
                    await router.acompletion(model=alias, messages=messages)
                    if async_mode
                    else router.completion(model=alias, messages=messages)
                )
            elif surface == "responses":
                response = (
                    await litellm.aresponses(input=input_items, use_chat_completions_api=True, **kwargs)
                    if async_mode
                    else litellm.responses(input=input_items, use_chat_completions_api=True, **kwargs)
                )
            else:
                response = (
                    await litellm.acompletion(messages=messages, **kwargs)
                    if async_mode
                    else litellm.completion(messages=messages, **kwargs)
                )
            await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
            content: Final = (
                response.output[0].content[0].text
                if surface in ("responses", "router-responses")
                else response.choices[0].message.content
            )
            assert content == ("forwarded" if forward is True else "omitted")
            assert route.call_count == expected_calls
            assert messages == original_messages
            assert input_items == original_input
