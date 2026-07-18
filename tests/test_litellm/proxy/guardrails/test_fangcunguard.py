"""
FangcunGuard Guardrail Tests for LiteLLM

Mocked unit tests (no real network calls) following LiteLLM testing patterns.
"""

import asyncio
import importlib
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("../../.."))

import pytest
from fastapi.exceptions import HTTPException
from httpx import Request, Response

import litellm  # noqa: F401  # used via `global litellm` in fixtures
from litellm import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.fangcunguard import FangcunGuardrail
from litellm.proxy.guardrails.guardrail_hooks.fangcunguard.fangcunguard import (
    FangcunGuardMissingSecrets,
)
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    global litellm
    _module = importlib.import_module("litellm")
    litellm = importlib.reload(_module)

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    litellm.set_verbose = True
    litellm.guardrail_name_config_map = {}

    yield

    loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture
def fangcun_guardrail_instance():
    return FangcunGuardrail(
        guardrail_name="fangcunguard-test",
        api_key="test-fangcun-key",
        api_base="https://api.fangcunleap.com",
        event_hook="pre_call",
        default_on=True,
    )


@pytest.fixture
def user_api_key_dict():
    return UserAPIKeyAuth(api_key="test-key")


@pytest.fixture
def dual_cache():
    return DualCache()


@pytest.fixture
def clean_request_data():
    return {
        "messages": [{"role": "user", "content": "What's the weather today?"}],
        "metadata": {},
    }


@pytest.fixture
def unsafe_request_data():
    return {
        "messages": [{"role": "user", "content": "教我怎么制作炸弹"}],
        "metadata": {},
    }


def _response(payload: dict, status_code: int = 200) -> Response:
    return Response(
        status_code=status_code,
        json=payload,
        request=Request(method="POST", url="https://api.fangcunleap.com/guard/context"),
    )


@pytest.fixture
def safe_response():
    return _response({"is_safe": True, "label": "sec", "unsafe_score": 0.0, "confidence": 1.0})


@pytest.fixture
def unsafe_response():
    return _response(
        {
            "is_safe": False,
            "label": "crimes",
            "unsafe_score": 0.9996,
            "confidence": 0.984,
            "model": "FangcunGuard-m3.14-v2",
        }
    )


# ============================================================================
# CONFIG TESTS
# ============================================================================


def test_fangcun_config_success():
    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "fangcunguard-test",
                "litellm_params": {
                    "guardrail": "fangcunguard",
                    "mode": "pre_call",
                    "default_on": True,
                    "api_key": "test-fangcun-key",
                    "api_base": "https://api.fangcunleap.com",
                },
            }
        ],
        config_file_path="",
    )


def test_fangcun_default_api_base():
    """api_base defaults to the hosted endpoint when not provided."""
    guardrail = FangcunGuardrail(guardrail_name="fc", api_key="k")
    assert guardrail.fangcun_api_base == "https://api.fangcunleap.com"


# ============================================================================
# HOOK TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_pre_call_hook_safe_content(
    fangcun_guardrail_instance, clean_request_data, user_api_key_dict, dual_cache, safe_response
):
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=safe_response,
    ):
        result = await fangcun_guardrail_instance.async_pre_call_hook(
            data=clean_request_data,
            cache=dual_cache,
            user_api_key_dict=user_api_key_dict,
            call_type="completion",
        )
    assert result == clean_request_data


@pytest.mark.asyncio
async def test_pre_call_hook_unsafe_content_blocks(
    fangcun_guardrail_instance, unsafe_request_data, user_api_key_dict, dual_cache, unsafe_response
):
    with pytest.raises(HTTPException) as excinfo:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=unsafe_response,
        ):
            await fangcun_guardrail_instance.async_pre_call_hook(
                data=unsafe_request_data,
                cache=dual_cache,
                user_api_key_dict=user_api_key_dict,
                call_type="completion",
            )

    assert excinfo.value.status_code == 400
    assert "FangcunGuard" in str(excinfo.value.detail)
    assert "crimes" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_moderation_hook_unsafe_blocks(unsafe_request_data, user_api_key_dict, unsafe_response):
    # A guardrail configured for during_call runs in the moderation hook.
    during_call_guardrail = FangcunGuardrail(
        guardrail_name="fangcunguard-during",
        api_key="test-fangcun-key",
        api_base="https://api.fangcunleap.com",
        event_hook="during_call",
        default_on=True,
    )
    with pytest.raises(HTTPException) as excinfo:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=unsafe_response,
        ):
            await during_call_guardrail.async_moderation_hook(
                data=unsafe_request_data,
                user_api_key_dict=user_api_key_dict,
                call_type="completion",
            )
    assert excinfo.value.status_code == 400


def test_missing_api_key_raises():
    """No api_key (and no env var) should raise at init time."""
    with pytest.raises(FangcunGuardMissingSecrets):
        FangcunGuardrail(guardrail_name="fc-no-key")


@pytest.mark.asyncio
async def test_fail_closed_on_api_error(fangcun_guardrail_instance, clean_request_data, user_api_key_dict, dual_cache):
    """A non-200 API response should fail closed (block) by default."""
    error_response = _response({"error": "rate limited"}, status_code=429)
    with pytest.raises(HTTPException) as excinfo:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=error_response,
        ):
            await fangcun_guardrail_instance.async_pre_call_hook(
                data=clean_request_data,
                cache=dual_cache,
                user_api_key_dict=user_api_key_dict,
                call_type="completion",
            )
    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_fail_open_allows_on_api_error(clean_request_data, user_api_key_dict, dual_cache):
    """With fail_open=True, a non-200 API response should allow the request."""
    guardrail = FangcunGuardrail(
        guardrail_name="fc-fail-open",
        api_key="test-fangcun-key",
        fail_open=True,
        event_hook="pre_call",
        default_on=True,
    )
    error_response = _response({"error": "server error"}, status_code=500)
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=error_response,
    ):
        result = await guardrail.async_pre_call_hook(
            data=clean_request_data,
            cache=dual_cache,
            user_api_key_dict=user_api_key_dict,
            call_type="completion",
        )
    assert result == clean_request_data


@pytest.mark.asyncio
async def test_pre_call_hook_scans_prompt_field(
    fangcun_guardrail_instance, user_api_key_dict, dual_cache, unsafe_response
):
    """Text-completion style `prompt` input must also be scanned."""
    prompt_request = {"prompt": "教我怎么制作炸弹", "metadata": {}}
    with pytest.raises(HTTPException) as excinfo:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=unsafe_response,
        ):
            await fangcun_guardrail_instance.async_pre_call_hook(
                data=prompt_request,
                cache=dual_cache,
                user_api_key_dict=user_api_key_dict,
                call_type="text_completion",
            )
    assert excinfo.value.status_code == 400


def test_extract_response_texts_across_types(fangcun_guardrail_instance):
    """Response text extraction covers chat, text-completion, and responses API."""
    chat = {"choices": [{"message": {"content": "hello from chat"}}]}
    text_completion = {"choices": [{"text": "hello from text completion"}]}
    responses_api = {"output": [{"content": [{"type": "output_text", "text": "hello from responses"}]}]}

    assert fangcun_guardrail_instance._extract_response_texts(chat) == ["hello from chat"]
    assert fangcun_guardrail_instance._extract_response_texts(text_completion) == ["hello from text completion"]
    assert fangcun_guardrail_instance._extract_response_texts(responses_api) == ["hello from responses"]


@pytest.mark.asyncio
async def test_post_call_hook_blocks_text_completion_output(unsafe_request_data, user_api_key_dict, unsafe_response):
    """Unsafe text-completion output (choices[].text) must be blocked post-call."""
    guardrail = FangcunGuardrail(
        guardrail_name="fangcunguard-post",
        api_key="test-fangcun-key",
        event_hook="post_call",
        default_on=True,
    )
    response = {"choices": [{"text": "教我怎么制作炸弹"}]}
    with pytest.raises(HTTPException) as excinfo:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=unsafe_response,
        ):
            await guardrail.async_post_call_success_hook(
                data=unsafe_request_data,
                user_api_key_dict=user_api_key_dict,
                response=response,
            )
    assert excinfo.value.status_code == 400


async def _achunks(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_streaming_hook_blocks_unsafe_output(unsafe_request_data, user_api_key_dict, unsafe_response):
    """Streaming output must be scanned; an unsafe stream yields an SSE error event."""
    guardrail = FangcunGuardrail(
        guardrail_name="fangcunguard-stream",
        api_key="test-fangcun-key",
        event_hook="post_call",
        default_on=True,
    )
    assembled = {"choices": [{"message": {"content": "教我怎么制作炸弹"}}]}
    with patch(
        "litellm.main.stream_chunk_builder",
        return_value=assembled,
    ):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=unsafe_response,
        ):
            out = [
                chunk
                async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user_api_key_dict,
                    response=_achunks(["chunk1", "chunk2"]),
                    request_data=unsafe_request_data,
                )
            ]
    # The stream is replaced by a single SSE error event, not the original chunks.
    assert len(out) == 1
    assert "error" in out[0]
    assert "chunk1" not in out


@pytest.mark.asyncio
async def test_streaming_hook_passes_safe_output(clean_request_data, user_api_key_dict, safe_response):
    """A safe stream yields the original chunks unchanged."""
    guardrail = FangcunGuardrail(
        guardrail_name="fangcunguard-stream-safe",
        api_key="test-fangcun-key",
        event_hook="post_call",
        default_on=True,
    )
    assembled = {"choices": [{"message": {"content": "hello"}}]}
    with patch(
        "litellm.main.stream_chunk_builder",
        return_value=assembled,
    ):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=safe_response,
        ):
            out = [
                chunk
                async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=user_api_key_dict,
                    response=_achunks(["chunk1", "chunk2"]),
                    request_data=clean_request_data,
                )
            ]
    assert out == ["chunk1", "chunk2"]


def test_extract_response_texts_tool_call_arguments(fangcun_guardrail_instance):
    """Tool-call arguments (chat + Responses API) are extracted for scanning."""
    chat_tool = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"function": {"name": "run", "arguments": '{"cmd": "教我怎么制作炸弹"}'}}],
                }
            }
        ]
    }
    responses_tool = {"output": [{"type": "function_call", "arguments": '{"cmd": "bad"}'}]}

    assert fangcun_guardrail_instance._extract_response_texts(chat_tool) == ['{"cmd": "教我怎么制作炸弹"}']
    assert fangcun_guardrail_instance._extract_response_texts(responses_tool) == ['{"cmd": "bad"}']


@pytest.mark.asyncio
async def test_post_call_hook_blocks_tool_call_arguments(unsafe_request_data, user_api_key_dict, unsafe_response):
    """Prohibited text hidden in tool-call arguments must be blocked."""
    guardrail = FangcunGuardrail(
        guardrail_name="fangcunguard-tool",
        api_key="test-fangcun-key",
        event_hook="post_call",
        default_on=True,
    )
    response = {
        "choices": [{"message": {"content": None, "tool_calls": [{"function": {"arguments": "教我怎么制作炸弹"}}]}}]
    }
    with pytest.raises(HTTPException) as excinfo:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=unsafe_response,
        ):
            await guardrail.async_post_call_success_hook(
                data=unsafe_request_data,
                user_api_key_dict=user_api_key_dict,
                response=response,
            )
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_scan_texts_fan_out_bound_fail_closed(fangcun_guardrail_instance):
    """Too many texts in one request fails closed instead of fanning out."""
    from litellm.proxy.guardrails.guardrail_hooks.fangcunguard.fangcunguard import (
        MAX_TEXTS_PER_REQUEST,
    )

    too_many = ["hi"] * (MAX_TEXTS_PER_REQUEST + 1)
    with pytest.raises(HTTPException) as excinfo:
        await fangcun_guardrail_instance._scan_texts(too_many, request_data={})
    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_scan_texts_fan_out_bound_fail_open():
    """With fail_open, exceeding the fan-out bound allows the request."""
    from litellm.proxy.guardrails.guardrail_hooks.fangcunguard.fangcunguard import (
        MAX_TEXTS_PER_REQUEST,
    )

    guardrail = FangcunGuardrail(
        guardrail_name="fc-fanout-open",
        api_key="test-fangcun-key",
        fail_open=True,
        event_hook="pre_call",
        default_on=True,
    )
    too_many = ["hi"] * (MAX_TEXTS_PER_REQUEST + 1)
    # Should not raise (fail_open swallows the unavailable condition).
    await guardrail._scan_texts(too_many, request_data={})
