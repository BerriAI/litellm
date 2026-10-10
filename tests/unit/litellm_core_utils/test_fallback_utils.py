"""Tests for litellm.litellm_core_utils.fallback_utils."""

import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.fallback_utils import (
    async_completion_with_fallbacks,
)
from litellm.main import acompletion as python_acompletion
from litellm.main import completion as python_completion


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
async def test_acompletion_fallbacks_bad_models(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    with pytest.raises(Exception, match="All fallback attempts failed"):
        await python_acompletion(
            model="openai/first",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            fallbacks=["openai/second"],
        )

    assert len(route.calls) == 2


@pytest.mark.asyncio
async def test_acompletion_fallbacks_basic(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
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
async def test_acompletion_fallbacks_empty_list(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_completion_fallbacks_sync(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
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
