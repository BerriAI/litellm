"""Tests for litellm.litellm_core_utils.fallback_utils."""

import asyncio
import contextlib
import json
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Final, NoReturn

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.caching.caching import Cache, DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils import litellm_logging
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.fallback_utils import (
    async_completion_with_fallbacks,
)
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.chatgpt.common_utils import CHATGPT_DEVICE_CODE_URL
from litellm.main import acompletion as python_acompletion
from litellm.main import completion as python_completion
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral, Choices, CredentialItem, LLMResponseTypes, Message, ModelResponse
from litellm.vector_stores.vector_store_registry import LiteLLM_ManagedVectorStore, VectorStoreRegistry

LITELLM_PARAMS: Final = TypeAdapter(dict[str, object])


@pytest.mark.asyncio
async def test_fallback_dict_not_mutated(monkeypatch):
    fallback_dict = {"model": "fallback-model", "temperature": 0.2}
    original_fallback_dict = dict(fallback_dict)

    attempted_models: list[str] = []

    async def _fake_acompletion(*, model: str, **kwargs):
        attempted_models.append(model)
        if model == "primary-model":
            raise Exception("primary failed")
        return {"model": model, "temperature": kwargs.get("temperature")}

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)

    # Call 1: primary fails, fallback dict succeeds
    response_1 = await async_completion_with_fallbacks(
        model="primary-model",
        kwargs={"fallbacks": [fallback_dict]},
    )
    assert response_1["model"] == "fallback-model"
    assert fallback_dict == original_fallback_dict

    # Call 2: re-use the same dict object; it should still work and remain unchanged
    response_2 = await async_completion_with_fallbacks(
        model="primary-model",
        kwargs={"fallbacks": [fallback_dict]},
    )
    assert response_2["model"] == "fallback-model"
    assert fallback_dict == original_fallback_dict

    assert attempted_models == [
        "primary-model",
        "fallback-model",
        "primary-model",
        "fallback-model",
    ]


@pytest.mark.asyncio
async def test_async_completion_with_fallbacks_sets_attempted_fallbacks_header():
    """
    When a fallback succeeds, the response must carry the
    `x-litellm-attempted-fallbacks` header so the proxy and other callers can
    detect that a fallback occurred. Without it,
    `_override_openai_response_model` stamps the requested model back over the
    fallback model used. See issue #28241.
    """
    response = await async_completion_with_fallbacks(
        model="openai/primary-llm",
        messages=[{"role": "user", "content": "hi"}],
        api_key="fake-key",
        mock_response=Exception("forced failure"),
        kwargs={
            "fallbacks": [
                {
                    "model": "openai/backup-llm",
                    "api_key": "fake-key",
                    "mock_response": "backup-resp",
                }
            ]
        },
    )

    hidden_params = getattr(response, "_hidden_params", None)
    assert isinstance(hidden_params, dict)
    headers = hidden_params.get("additional_headers") or {}
    assert headers.get("x-litellm-attempted-fallbacks") == 1


@pytest.mark.asyncio
async def test_async_completion_with_fallbacks_header_is_zero_when_primary_succeeds():
    """
    When the primary model succeeds on the first attempt, the header should be
    `0` (no fallback was used). This mirrors the existing router-level
    semantics in `async_function_with_fallbacks`.
    """
    response = await async_completion_with_fallbacks(
        model="openai/primary-llm",
        messages=[{"role": "user", "content": "hi"}],
        api_key="fake-key",
        mock_response="primary-resp",
        kwargs={
            "fallbacks": [
                {
                    "model": "openai/backup-llm",
                    "api_key": "fake-key",
                    "mock_response": "backup-resp",
                }
            ]
        },
    )

    hidden_params = getattr(response, "_hidden_params", None)
    assert isinstance(hidden_params, dict)
    headers = hidden_params.get("additional_headers") or {}
    assert headers.get("x-litellm-attempted-fallbacks") == 0
    assert response.choices[0].message.content == "primary-resp"


def test_process_response_headers_preserves_x_litellm_headers_when_internal():
    """
    `process_response_headers` must not add the `llm_provider-` prefix to
    LiteLLM's own internal headers (anything starting with `x-litellm-`) when
    the caller has marked the input as LiteLLM-owned. These are markers set by
    LiteLLM (e.g. fallback / retry headers); the proxy and other callers look
    up the bare key.
    """
    result = process_response_headers(
        {
            "x-litellm-attempted-fallbacks": 1,
            "x-litellm-model-group": "gpt-4",
            "x-stainless-arch": "arm64",
        },
        preserve_litellm_internal_headers=True,
    )
    assert result["x-litellm-attempted-fallbacks"] == 1
    assert result["x-litellm-model-group"] == "gpt-4"
    assert result["llm_provider-x-stainless-arch"] == "arm64"


def test_process_response_headers_prefixes_x_litellm_from_raw_provider():
    """
    On raw upstream-provider headers (default `preserve_litellm_internal_headers=False`),
    a header whose name starts with `x-litellm-` MUST still get the
    `llm_provider-` prefix. Otherwise a malicious provider could return
    `x-litellm-attempted-fallbacks` and spoof a LiteLLM-internal marker,
    bypassing the proxy model-override guard.
    """
    result = process_response_headers(
        {
            "x-litellm-attempted-fallbacks": 99,
            "x-stainless-arch": "arm64",
        }
    )
    assert "x-litellm-attempted-fallbacks" not in result
    assert result["llm_provider-x-litellm-attempted-fallbacks"] == 99
    assert result["llm_provider-x-stainless-arch"] == "arm64"


def test_process_response_headers_ignores_preserve_flag_for_httpx_headers():
    """
    Some providers store raw httpx.Headers directly in _hidden_params["additional_headers"]
    without a prior normalization pass. If preserve_litellm_internal_headers=True were
    honored for httpx.Headers inputs, a provider returning x-litellm-attempted-fallbacks
    could spoof it as a bare LiteLLM-internal marker and make the proxy skip
    stamping the correct response model. The flag must be ignored for httpx.Headers.
    """
    raw = httpx.Headers(
        {
            "x-litellm-attempted-fallbacks": "1",
            "content-type": "application/json",
        }
    )
    result = process_response_headers(raw, preserve_litellm_internal_headers=True)
    assert "x-litellm-attempted-fallbacks" not in result
    assert result["llm_provider-x-litellm-attempted-fallbacks"] == "1"


@pytest.mark.asyncio
async def test_acompletion_fallbacks_bad_models(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=(
            httpx.Response(
                404,
                json={
                    "error": {
                        "message": "model unavailable",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "model_not_found",
                    }
                },
            ),
            httpx.Response(
                404,
                json={
                    "error": {
                        "message": "fallback unavailable",
                        "type": "authentication_error",
                        "param": None,
                        "code": "model_not_found",
                    }
                },
            ),
        )
    )

    with pytest.raises(Exception, match=r"fallback unavailable\. All fallback attempts failed"):
        await python_acompletion(
            model="openai/first",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            fallbacks=["openai/second"],
        )

    assert len(route.calls) == 2


@pytest.mark.asyncio
async def test_acompletion_fallbacks_basic(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=(
            httpx.Response(
                404,
                json={
                    "error": {
                        "message": "model unavailable",
                        "type": "authentication_error",
                        "param": None,
                        "code": "model_not_found",
                    }
                },
            ),
            httpx.Response(
                200,
                json={
                    "id": "chatcmpl-fallback",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "second",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "fallback answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            ),
        )
    )

    response: Final = await python_acompletion(
        model="openai/first",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        fallbacks=["openai/second"],
    )

    assert response.choices[0].message.content == "fallback answer"
    assert tuple(json.loads(call.request.content)["model"] for call in route.calls) == ("first", "second")


@pytest.mark.asyncio
async def test_acompletion_fallbacks_empty_list(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            404,
            json={
                "error": {
                    "message": "model unavailable",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "model_not_found",
                }
            },
        )
    )

    with pytest.raises(litellm.NotFoundError, match="model unavailable"):
        await python_acompletion(
            model="openai/first",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            fallbacks=[],
        )

    assert len(route.calls) == 1


@pytest.mark.asyncio
async def test_acompletion_fallbacks_none_response(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=(
            httpx.Response(
                500,
                json={
                    "error": {
                        "message": "temporary failure",
                        "type": "server_error",
                        "param": None,
                        "code": "server_error",
                    }
                },
            ),
            httpx.Response(
                200,
                json={
                    "id": "chatcmpl-fallback",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "second",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "recovered answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            ),
        )
    )

    response: Final = await python_acompletion(
        model="openai/first",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        fallbacks=["openai/second"],
    )

    assert response.choices[0].message.content == "recovered answer"
    assert len(route.calls) == 2


@pytest.mark.asyncio
async def test_acompletion_fallbacks_with_dict_config(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=(
            httpx.Response(
                401,
                json={
                    "error": {
                        "message": "invalid key",
                        "type": "authentication_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            ),
            httpx.Response(
                200,
                json={
                    "id": "chatcmpl-fallback",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "first",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "authenticated answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            ),
        )
    )

    response: Final = await python_acompletion(
        model="openai/first",
        messages=[{"role": "user", "content": "hello"}],
        api_key="invalid-key",
        fallbacks=[{"api_key": "fallback-key"}],
    )

    assert response.choices[0].message.content == "authenticated answer"
    assert tuple(call.request.headers["authorization"] for call in route.calls) == (
        "Bearer invalid-key",
        "Bearer fallback-key",
    )


def test_completion_fallbacks_sync(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=(
            httpx.Response(
                404,
                json={
                    "error": {
                        "message": "model unavailable",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "model_not_found",
                    }
                },
            ),
            httpx.Response(
                200,
                json={
                    "id": "chatcmpl-fallback",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "second",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "sync fallback answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            ),
        )
    )

    response: Final = python_completion(
        model="openai/first",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        fallbacks=["openai/second"],
    )

    assert response.choices[0].message.content == "sync fallback answer"
    assert tuple(json.loads(call.request.content)["model"] for call in route.calls) == ("first", "second")


@pytest.mark.timeout(30, method="thread")
def test_completion_fallbacks_name_the_last_error_when_every_model_fails(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    fallback_unavailable: Final = {
        "error": {"message": "fallback unavailable", "type": "invalid_request_error", "param": None, "code": None}
    }
    _ = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=(_model_unavailable(), httpx.Response(404, json=fallback_unavailable))
    )

    with pytest.raises(Exception, match=r"fallback unavailable\. All fallback attempts failed"):
        python_completion(
            model="openai/first",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            fallbacks=["openai/second"],
        )


CACHE_POINTS: Final = ({"location": "message", "role": "system"},)
CACHED_MESSAGES: Final = (
    {"role": "system", "content": "You are a careful assistant."},
    {"role": "user", "content": "hello"},
)
OPENAI_ANSWER: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-6.1-sol",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
ANTHROPIC_ANSWER: Final = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def _cached_request(**kwargs: object) -> dict[str, object]:
    return {
        "messages": [dict(message) for message in CACHED_MESSAGES],
        "cache_control_injection_points": [dict(point) for point in CACHE_POINTS],
        **kwargs,
    }


@pytest.mark.asyncio
async def test_acompletion_fallback_sends_the_fallback_model_the_request_a_direct_call_sends(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    openai_route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={"error": {"message": "invalid key", "type": "authentication_error", "param": None, "code": None}},
        )
    )
    anthropic_route: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_ANSWER)
    )

    await python_acompletion(**_cached_request(model="anthropic/claude-haiku-4-5", api_key="anthropic-key"))
    await python_acompletion(
        **_cached_request(
            model="openai/gpt-6.1-sol",
            api_key="openai-key",
            fallbacks=[{"model": "anthropic/claude-haiku-4-5", "api_key": "anthropic-key"}],
        )
    )

    assert "prompt_cache_options" in json.loads(openai_route.calls[0].request.content)
    direct_body, fallback_body = (json.loads(call.request.content) for call in anthropic_route.calls)
    assert fallback_body == direct_body


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_cache_dialect_follows_base_url_over_an_api_base_the_call_does_not_use(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_ANSWER)
    )

    async def send(**kwargs: object) -> None:
        request: Final = _cached_request(
            model="openai/gpt-6.1-sol", api_key="openai-key", base_url="https://api.openai.com/v1", **kwargs
        )
        if is_async:
            await python_acompletion(**request)
        else:
            python_completion(**request)

    await send()
    await send(api_base="https://unused-gateway.example.com/v1")

    base_url_only, with_unused_api_base = (json.loads(call.request.content) for call in route.calls)
    assert with_unused_api_base == base_url_only


@pytest.mark.asyncio
async def test_acompletion_fallback_sends_the_context_a_vector_store_tool_retrieves(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(
        litellm,
        "vector_store_registry",
        VectorStoreRegistry(
            vector_stores=[
                LiteLLM_ManagedVectorStore(
                    vector_store_id="vs_docs", custom_llm_provider="openai", litellm_params={"api_key": "openai-key"}
                )
            ]
        ),
    )
    search_route: Final = respx_mock.post("https://api.openai.com/v1/vector_stores/vs_docs/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "vector_store.search_results.page",
                "search_query": "what is litellm?",
                "data": [{"score": 1.0, "content": [{"type": "text", "text": "litellm is an LLM gateway"}]}],
            },
        )
    )
    openai_route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={"error": {"message": "invalid key", "type": "authentication_error", "param": None, "code": None}},
        )
    )
    anthropic_route: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_ANSWER)
    )

    async def send(**kwargs: object) -> None:
        await python_acompletion(
            messages=[{"role": "user", "content": "what is litellm?"}],
            tools=[{"type": "file_search", "vector_store_ids": ["vs_docs"]}],
            **kwargs,
        )

    await send(model="anthropic/claude-haiku-4-5", api_key="anthropic-key")
    await send(
        model="openai/gpt-6.1-sol",
        api_key="openai-key",
        fallbacks=[{"model": "anthropic/claude-haiku-4-5", "api_key": "anthropic-key"}],
    )

    direct_body, fallback_body = (json.loads(call.request.content) for call in anthropic_route.calls)
    assert "litellm is an LLM gateway" in json.dumps(direct_body)
    assert fallback_body == direct_body
    assert search_route.call_count <= openai_route.call_count + anthropic_route.call_count


@pytest.mark.asyncio
async def test_acompletion_fallback_sends_the_prompt_the_primary_model_prefix_template_renders(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    _ = (tmp_path / "brief.prompt").write_text(
        "---\ninput:\n  schema:\n    question: string\n---\nSystem: Answer in one word.\n\nUser: {{question}}\n"
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "global_prompt_directory", str(tmp_path))
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [])
    _ = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={"error": {"message": "invalid key", "type": "authentication_error", "param": None, "code": None}},
        )
    )
    anthropic_route: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_ANSWER)
    )

    async def send(**kwargs: object) -> None:
        await python_acompletion(
            prompt_id="brief",
            prompt_variables={"question": "What color is the sky?"},
            messages=[{"role": "user", "content": "placeholder"}],
            **kwargs,
        )

    await send(model="dotprompt/anthropic/claude-haiku-4-5", api_key="anthropic-key")
    await send(
        model="dotprompt/openai/gpt-6.1-sol",
        api_key="openai-key",
        fallbacks=[{"model": "anthropic/claude-haiku-4-5", "api_key": "anthropic-key"}],
    )

    direct_body, fallback_body = (json.loads(call.request.content) for call in anthropic_route.calls)
    assert "What color is the sky?" in json.dumps(direct_body)
    assert fallback_body == direct_body


def _model_unavailable() -> httpx.Response:
    return httpx.Response(
        404,
        json={
            "error": {
                "message": "model unavailable",
                "type": "invalid_request_error",
                "param": None,
                "code": "model_not_found",
            }
        },
    )


def _replies(contents: tuple[str | None, ...]) -> tuple[httpx.Response, ...]:
    return tuple(
        _model_unavailable()
        if content is None
        else httpx.Response(
            200,
            json={
                **OPENAI_ANSWER,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
            },
        )
        for content in contents
    )


def _requested_models(route: respx.Route) -> tuple[str, ...]:
    return tuple(json.loads(call.request.content)["model"] for call in route.calls)


GLOBAL_CHAIN_OUTCOMES: Final = (
    pytest.param(("first answer",), ("first",), id="primary-answers"),
    pytest.param((None, None, "third answer"), ("first", "second", "third"), id="falls-through-in-order"),
)


@pytest.mark.timeout(30, method="thread")
@pytest.mark.parametrize(("contents", "expected_models"), GLOBAL_CHAIN_OUTCOMES)
@pytest.mark.asyncio
async def test_acompletion_tries_the_global_model_fallbacks_once_each_in_order(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    contents: tuple[str | None, ...],
    expected_models: tuple[str, ...],
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_fallbacks", ["openai/second", "openai/third"])
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_replies(contents))

    response: Final = await python_acompletion(
        model="openai/first", messages=[{"role": "user", "content": "hello"}], api_key="test-key"
    )

    assert response.choices[0].message.content == contents[-1]
    assert _requested_models(route) == expected_models


@pytest.mark.timeout(30, method="thread")
@pytest.mark.parametrize(("contents", "expected_models"), GLOBAL_CHAIN_OUTCOMES)
def test_completion_tries_the_global_model_fallbacks_once_each_in_order(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    contents: tuple[str | None, ...],
    expected_models: tuple[str, ...],
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_fallbacks", ["openai/second", "openai/third"])
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_replies(contents))

    response: Final = python_completion(
        model="openai/first", messages=[{"role": "user", "content": "hello"}], api_key="test-key"
    )

    assert response.choices[0].message.content == contents[-1]
    assert _requested_models(route) == expected_models


@pytest.mark.timeout(30, method="thread")
@pytest.mark.asyncio
async def test_acompletion_per_call_fallbacks_win_over_the_global_model_fallbacks(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_fallbacks", ["openai/global-fallback"])
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=_replies((None, "per-call answer"))
    )

    response: Final = await python_acompletion(
        model="openai/first",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        fallbacks=["openai/second"],
    )

    assert response.choices[0].message.content == "per-call answer"
    assert _requested_models(route) == ("first", "second")


@pytest.mark.timeout(30, method="thread")
@pytest.mark.asyncio
async def test_acompletion_empty_per_call_fallbacks_opt_out_of_the_global_model_fallbacks(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_fallbacks", ["openai/second"])
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=_replies((None, "fallback answer"))
    )

    with pytest.raises(litellm.NotFoundError, match="model unavailable"):
        await python_acompletion(
            model="openai/first", messages=[{"role": "user", "content": "hello"}], api_key="test-key", fallbacks=[]
        )

    assert _requested_models(route) == ("first",)


async def _flush_success_logs() -> None:
    await asyncio.sleep(0)
    GLOBAL_LOGGING_WORKER.start()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)


class _SuccessRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.models: list[object] = []
        self.call_ids: list[object] = []

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.models.append(kwargs.get("model"))
        self.call_ids.append(LITELLM_PARAMS.validate_python(kwargs.get("litellm_params")).get("litellm_call_id"))


@pytest.mark.timeout(30, method="thread")
@pytest.mark.asyncio
async def test_acompletion_through_the_global_model_fallbacks_logs_one_success(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder: Final = _SuccessRecorder()
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_fallbacks", ["openai/second"])
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    _ = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=_replies((None, "fallback answer"))
    )

    await python_acompletion(
        model="openai/first",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        litellm_call_id="caller-call-id",
    )
    await _flush_success_logs()

    assert recorder.models == ["second"]
    assert recorder.call_ids == ["caller-call-id"]


@pytest.mark.timeout(30, method="thread")
@pytest.mark.parametrize(
    "global_fallbacks",
    (pytest.param(["openai/second"], id="global-set"), pytest.param(None, id="global-unset")),
)
@pytest.mark.asyncio
async def test_acompletion_with_empty_per_call_fallbacks_logs_one_success(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, global_fallbacks: list[str] | None
) -> None:
    recorder: Final = _SuccessRecorder()
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_fallbacks", global_fallbacks)
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    _ = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_replies(("first answer",)))

    await python_acompletion(
        model="openai/first", messages=[{"role": "user", "content": "hello"}], api_key="test-key", fallbacks=[]
    )
    await _flush_success_logs()

    assert recorder.models == ["first"]


CREDENTIAL_ROUTES: Final = (
    pytest.param(
        "openai/second",
        "https://gateway.example/v1/chat/completions",
        ("authorization", "Bearer caller-key"),
        id="same-provider-keeps-the-callers-credentials",
    ),
    pytest.param(
        "anthropic/claude-haiku-5-5",
        "https://api.anthropic.com/v1/messages",
        ("x-api-key", "anthropic-env-key"),
        id="other-provider-uses-its-own-credentials",
    ),
)


@pytest.mark.timeout(30, method="thread")
@pytest.mark.parametrize(("fallback", "fallback_url", "credential"), CREDENTIAL_ROUTES)
@pytest.mark.asyncio
async def test_acompletion_fallback_sends_credentials_only_to_the_provider_they_belong_to(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    fallback: str,
    fallback_url: str,
    credential: tuple[str, str],
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_fallbacks", [fallback])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-env-key")
    _ = respx_mock.post("https://gateway.example/v1/chat/completions").mock(
        side_effect=(_model_unavailable(), httpx.Response(200, json=OPENAI_ANSWER))
    )
    _ = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_ANSWER)
    )

    response: Final = await python_acompletion(
        model="openai/first",
        messages=[{"role": "user", "content": "hello"}],
        api_key="caller-key",
        api_base="https://gateway.example/v1",
    )

    header, value = credential
    assert response.choices[0].message.content == "ok"
    assert str(respx_mock.calls.last.request.url) == fallback_url
    assert respx_mock.calls.last.request.headers[header] == value


GATEWAY_CREDENTIAL_CARRIERS: Final = (
    pytest.param("openai/first", {"litellm_credential_name": "gateway"}, id="named-credential"),
    pytest.param(
        "openai/first",
        {
            "api_key": "gateway-key",
            "api_base": "https://gateway.example/v1",
            "headers": {"authorization": "Bearer gateway-key"},
        },
        id="request-headers",
    ),
    pytest.param(
        "first-custom-model",
        {"custom_llm_provider": "openai", "api_key": "gateway-key", "api_base": "https://gateway.example/v1"},
        id="explicit-provider",
    ),
)


@pytest.mark.timeout(30, method="thread")
@pytest.mark.parametrize(("model", "gateway_credentials"), GATEWAY_CREDENTIAL_CARRIERS)
@pytest.mark.asyncio
async def test_acompletion_fallback_to_another_provider_never_reuses_the_first_providers_credentials(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    gateway_credentials: dict[str, object],
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(
        litellm,
        "credential_list",
        [
            CredentialItem(
                credential_name="gateway",
                credential_values={"api_key": "gateway-key", "api_base": "https://gateway.example/v1"},
                credential_info={},
            )
        ],
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-env-key")
    _ = respx_mock.post("https://gateway.example/v1/chat/completions").mock(return_value=_model_unavailable())
    _ = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_ANSWER)
    )

    response: Final = await python_acompletion(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        fallbacks=["anthropic/claude-haiku-5-5"],
        **gateway_credentials,
    )

    fallback_request: Final = respx_mock.calls.last.request
    assert response.choices[0].message.content == "ok"
    assert str(fallback_request.url) == "https://api.anthropic.com/v1/messages"
    assert fallback_request.headers["x-api-key"] == "anthropic-env-key"
    assert not any("gateway-key" in value for value in fallback_request.headers.values())


@pytest.mark.timeout(30, method="thread")
@pytest.mark.asyncio
async def test_acompletion_fallback_to_an_unprefixed_model_keeps_the_requests_provider_and_credentials(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://models.example/v1/chat/completions").mock(
        side_effect=_replies((None, "fallback answer"))
    )

    response: Final = await python_acompletion(
        model="first-custom-model",
        custom_llm_provider="openai",
        api_base="https://models.example/v1",
        api_key="models-key",
        messages=[{"role": "user", "content": "hello"}],
        fallbacks=["second-custom-model"],
    )

    assert response.choices[0].message.content == "fallback answer"
    assert _requested_models(route) == ("first-custom-model", "second-custom-model")
    assert route.calls.last.request.headers["authorization"] == "Bearer models-key"


@pytest.mark.timeout(30, method="thread")
def test_completion_fallback_from_a_model_without_a_provider_never_reuses_its_credentials(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-env-key")
    route: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_ANSWER)
    )

    response: Final = python_completion(
        model="first-without-a-provider",
        messages=[{"role": "user", "content": "hello"}],
        api_key="first-key",
        fallbacks=["anthropic/claude-haiku-5-5"],
    )

    assert response.choices[0].message.content == "ok"
    assert route.calls.last.request.headers["x-api-key"] == "anthropic-env-key"


async def _router_completion(router: litellm.Router, *, is_async: bool) -> object:
    messages: Final = [{"role": "user", "content": "hello"}]
    if is_async:
        return await router.acompletion(model="first", messages=messages)
    return router.completion(model="first", messages=messages)


@pytest.mark.timeout(30, method="thread")
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_router_leaves_fallbacks_to_the_router_when_global_model_fallbacks_are_set(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_fallbacks", ["openai/second"])
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=_replies((None, "global fallback answer"))
    )
    router: Final = litellm.Router(
        model_list=[{"model_name": "first", "litellm_params": {"model": "openai/first", "api_key": "test-key"}}],
        num_retries=0,
    )

    with pytest.raises(litellm.NotFoundError, match="model unavailable"):
        await _router_completion(router, is_async=is_async)

    assert _requested_models(route) == ("first",)


STREAMED_ANSWER: Final = (
    "".join(
        f"data: {json.dumps(chunk)}\n\n"
        for chunk in (
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "first",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "streamed answer"}}],
            },
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "first",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        )
    )
    + "data: [DONE]\n\n"
)


@pytest.mark.timeout(30, method="thread")
def test_completion_streams_through_the_global_model_fallbacks(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_fallbacks", ["openai/second"])
    _ = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=STREAMED_ANSWER, headers={"content-type": "text/event-stream"})
    )

    stream: Final = python_completion(
        model="openai/first", messages=[{"role": "user", "content": "hello"}], api_key="test-key", stream=True
    )

    assert "".join(chunk.choices[0].delta.content or "" for chunk in stream) == "streamed answer"


class _ConcurrentSyncSuccessRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.models: list[object] = []
        self._pair_concurrent_logs: Final = threading.Barrier(2, timeout=0.5)

    def logging_hook(
        self, kwargs: dict[str, object], result: object, call_type: str
    ) -> tuple[dict[str, object], object]:
        with contextlib.suppress(threading.BrokenBarrierError):
            _ = self._pair_concurrent_logs.wait()
        return kwargs, result

    def log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.models.append(kwargs.get("model"))


@pytest.mark.timeout(30, method="thread")
def test_completion_through_the_global_model_fallbacks_logs_one_success(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder: Final = _ConcurrentSyncSuccessRecorder()
    logging_executor: Final = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(litellm.utils, "executor", logging_executor)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_fallbacks", ["openai/second"])
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    _ = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=_replies((None, "fallback answer"))
    )

    python_completion(model="openai/first", messages=[{"role": "user", "content": "hello"}], api_key="test-key")
    logging_executor.shutdown(wait=True)

    assert recorder.models == ["second"]


class _RedactingGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(
            guardrail_name="redact", event_hook=[GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]
        )

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: CallTypesLiteral
    ) -> dict[str, object]:
        return {**data, "messages": [{"role": "user", "content": "[REDACTED]"}]}

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response: LLMResponseTypes
    ) -> ModelResponse:
        return ModelResponse(model="checked", choices=[Choices(message=Message(content="[CHECKED]"))])


@pytest.mark.timeout(30, method="thread")
def test_completion_with_fallbacks_runs_the_requested_guardrail_before_every_attempt(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [_RedactingGuardrail()])
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=_replies((None, "fallback answer"))
    )

    response: Final = python_completion(
        model="openai/first",
        messages=[{"role": "user", "content": "my card is 4111 1111 1111 1111"}],
        api_key="test-key",
        fallbacks=["openai/second"],
        guardrails=["redact"],
    )

    assert response.choices[0].message.content == "[CHECKED]"
    assert [json.loads(call.request.content)["messages"] for call in route.calls] == [
        [{"role": "user", "content": "[REDACTED]"}]
    ] * 2


class _RejectingGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(guardrail_name="reject", event_hook=GuardrailEventHooks.post_call)

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response: LLMResponseTypes
    ) -> NoReturn:
        raise ValueError("answer rejected")


def _completion_checked_by_the_rejecting_guardrail() -> object:
    return python_completion(
        model="openai/first",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        fallbacks=["openai/second"],
        guardrails=["reject"],
    )


@pytest.mark.timeout(30, method="thread")
def test_completion_with_fallbacks_never_caches_an_answer_its_guardrail_rejected(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "cache", Cache(type="local"))
    monkeypatch.setattr(litellm, "callbacks", [_RejectingGuardrail()])
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=_replies(("rejected answer",) * 4)
    )

    with pytest.raises(Exception, match="answer rejected"):
        _ = _completion_checked_by_the_rejecting_guardrail()
    with pytest.raises(Exception, match="answer rejected"):
        _ = _completion_checked_by_the_rejecting_guardrail()

    assert _requested_models(route) == ("first", "second", "first", "second")


class _SyncAnswerRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.answers: list[object] = []

    def log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.answers.append(
            response_obj.choices[0].message.content if isinstance(response_obj, ModelResponse) else None
        )


@pytest.mark.parametrize(
    ("guardrail_class", "logged_answers"),
    [(_RedactingGuardrail, ["[CHECKED]"]), (_RejectingGuardrail, [])],
    ids=["rewritten", "rejected"],
)
@pytest.mark.timeout(30, method="thread")
def test_completion_with_fallbacks_logs_only_the_answer_its_guardrail_returned(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    guardrail_class: type[CustomGuardrail],
    logged_answers: list[str],
) -> None:
    guardrail: Final = guardrail_class()
    recorder: Final = _SyncAnswerRecorder()
    logging_executor: Final = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(litellm.utils, "executor", logging_executor)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [guardrail, recorder])
    _ = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_replies((None, "raw answer")))

    with contextlib.suppress(Exception):
        _ = python_completion(
            model="openai/first",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            fallbacks=["openai/second"],
            guardrails=[guardrail.guardrail_name],
        )
    logging_executor.shutdown(wait=True)

    assert recorder.answers == logged_answers


@pytest.mark.timeout(30, method="thread")
def test_completion_streams_through_fallbacks_with_the_requested_guardrail(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [_RedactingGuardrail()])
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=STREAMED_ANSWER, headers={"content-type": "text/event-stream"})
    )

    stream: Final = python_completion(
        model="openai/first",
        messages=[{"role": "user", "content": "my card is 4111 1111 1111 1111"}],
        api_key="test-key",
        stream=True,
        fallbacks=["openai/second"],
        guardrails=["redact"],
    )

    assert "".join(chunk.choices[0].delta.content or "" for chunk in stream) == "streamed answer"
    assert json.loads(route.calls.last.request.content)["messages"] == [{"role": "user", "content": "[REDACTED]"}]


@pytest.mark.timeout(30, method="thread")
@pytest.mark.asyncio
async def test_acompletion_fallback_to_another_model_is_priced_like_a_direct_call_to_it(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _ = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=(
            _model_unavailable(),
            httpx.Response(200, json=OPENAI_ANSWER),
            httpx.Response(200, json=OPENAI_ANSWER),
        )
    )
    primary_price: Final = {"input_cost_per_token": 1.0, "output_cost_per_token": 1.0}

    through_fallback: Final = await python_acompletion(
        model="openai/first",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
        fallbacks=["openai/gpt-5.4-nano"],
        **primary_price,
    )
    direct: Final = await python_acompletion(
        model="openai/gpt-5.4-nano", messages=[{"role": "user", "content": "hello"}], api_key="test-key"
    )

    assert through_fallback._hidden_params["response_cost"] == direct._hidden_params["response_cost"]


@pytest.mark.timeout(30, method="thread")
def test_completion_never_signs_in_to_a_chatgpt_fallback_the_call_does_not_use(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_fallbacks", ["chatgpt/gpt-5.4"])
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    device_login: Final = respx_mock.post(CHATGPT_DEVICE_CODE_URL).mock(return_value=httpx.Response(503))
    _ = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(side_effect=_replies(("first answer",)))

    response: Final = python_completion(
        model="openai/first", messages=[{"role": "user", "content": "hello"}], api_key="test-key"
    )

    assert response.choices[0].message.content == "first answer"
    assert not device_login.called


@pytest.mark.timeout(30, method="thread")
@pytest.mark.asyncio
async def test_acompletion_falls_back_from_a_chatgpt_model_that_is_not_signed_in(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "model_fallbacks", ["openai/second"])
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key")
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=_replies(("fallback answer",))
    )

    response: Final = await python_acompletion(
        model="gpt-5.4", custom_llm_provider="chatgpt", messages=[{"role": "user", "content": "hello"}]
    )

    assert response.choices[0].message.content == "fallback answer"
    assert _requested_models(route) == ("second",)
