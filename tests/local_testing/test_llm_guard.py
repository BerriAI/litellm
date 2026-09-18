# What is this?
## This tests the llm guard integration

# What is this?
## Unit test for presidio pii masking
import sys, os, asyncio, time, random
from datetime import datetime
from typing import Final, Literal
import traceback
from dotenv import load_dotenv

load_dotenv()

import pytest
from fastapi import HTTPException

import litellm
from litellm_enterprise.enterprise_callbacks.llm_guard import _ENTERPRISE_LLMGuard
from litellm import Router, mock_completion
from litellm.proxy.utils import ProxyLogging, hash_token
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache
from litellm.types.utils import CallTypesLiteral

### UNIT TESTS FOR LLM GUARD ###


@pytest.mark.asyncio
async def test_llm_guard_valid_response():
    """
    A valid (is_valid=True) LLM Guard response must apply the returned
    sanitized_prompt back onto the request data so the provider receives the
    redacted content.
    """
    litellm.llm_guard_mode = "all"
    input_a_anonymizer_results = {
        "sanitized_prompt": "hello world",
        "is_valid": True,
        "scanners": {"Regex": 0.0},
    }
    llm_guard = _ENTERPRISE_LLMGuard(
        mock_testing=True, mock_redacted_text=input_a_anonymizer_results
    )

    _api_key = "sk-12345"
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    data = {
        "messages": [
            {
                "role": "user",
                "content": "hello world, my name is Jane Doe. My number is: 23r323r23r2wwkl",
            }
        ]
    }

    result = await llm_guard.async_moderation_hook(
        data=data,
        user_api_key_dict=user_api_key_dict,
        call_type="completion",
    )

    assert result is data
    assert data["messages"][0]["content"] == "hello world"


@pytest.mark.asyncio
async def test_llm_guard_sanitizes_multimodal_and_input():
    """
    Sanitization must reach text parts of multimodal message content and the
    ``input`` field (embeddings/moderation) while leaving non-text parts intact.
    """
    litellm.llm_guard_mode = "all"
    llm_guard = _ENTERPRISE_LLMGuard(
        mock_testing=True,
        mock_redacted_text={
            "sanitized_prompt": "email: [REDACTED]",
            "is_valid": True,
            "scanners": {"Regex": 0.0},
        },
    )
    user_api_key_dict = UserAPIKeyAuth(api_key=hash_token("sk-12345"))

    image_part = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "email: person@example.com"},
                    image_part,
                ],
            }
        ]
    }
    result = await llm_guard.async_moderation_hook(
        data=data, user_api_key_dict=user_api_key_dict, call_type="completion"
    )
    assert result["messages"][0]["content"][0]["text"] == "email: [REDACTED]"
    assert result["messages"][0]["content"][1] == image_part

    input_data = {"input": ["email: person@example.com", "another prompt"]}
    input_result = await llm_guard.async_moderation_hook(
        data=input_data, user_api_key_dict=user_api_key_dict, call_type="embeddings"
    )
    assert input_result["input"] == ["email: [REDACTED]", "email: [REDACTED]"]


@pytest.mark.parametrize(
    "call_type, payload_key",
    (
        ("completion", "messages"),
        ("acompletion", "messages"),
        ("text_completion", "prompt"),
        ("atext_completion", "prompt"),
        ("embeddings", "input"),
        ("embedding", "input"),
        ("aembedding", "input"),
        ("moderation", "input"),
        ("amoderation", "input"),
        ("image_generation", "prompt"),
        ("aimage_generation", "prompt"),
        ("audio_transcription", "prompt"),
        ("transcription", "prompt"),
        ("atranscription", "prompt"),
    ),
)
@pytest.mark.parametrize("is_valid", (True, False))
@pytest.mark.asyncio
async def test_llm_guard_call_type_aliases(
    call_type: CallTypesLiteral,
    payload_key: Literal["messages", "input", "prompt"],
    is_valid: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "llm_guard_mode", "all")
    llm_guard: Final = _ENTERPRISE_LLMGuard(
        mock_testing=True,
        mock_redacted_text={
            "sanitized_prompt": "email: [REDACTED]",
            "is_valid": is_valid,
        },
    )
    user_api_key_dict: Final = UserAPIKeyAuth(api_key=hash_token("sk-12345"))
    data: Final = {
        payload_key: [{"role": "user", "content": "email: person@example.com"}]
        if payload_key == "messages"
        else "email: person@example.com"
    }

    if not is_valid:
        with pytest.raises(HTTPException) as exc_info:
            await llm_guard.async_moderation_hook(
                data=data, user_api_key_dict=user_api_key_dict, call_type=call_type
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == {"error": "Violated content safety policy"}
        return

    result: Final = await llm_guard.async_moderation_hook(
        data=data, user_api_key_dict=user_api_key_dict, call_type=call_type
    )
    assert result is data
    assert data[payload_key] == (
        [{"role": "user", "content": "email: [REDACTED]"}]
        if payload_key == "messages"
        else "email: [REDACTED]"
    )


@pytest.mark.parametrize(
    "call_type",
    (
        "responses",
        "aresponses",
        "anthropic_messages",
        "aanthropic_messages",
        "aspeech",
        "aimage_edit",
        "pass_through_endpoint",
    ),
)
@pytest.mark.asyncio
async def test_llm_guard_skips_unsupported_call_types(
    call_type: CallTypesLiteral, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "llm_guard_mode", "all")
    llm_guard: Final = _ENTERPRISE_LLMGuard(
        mock_testing=True,
        mock_redacted_text={"is_valid": False},
    )
    data: Final = {"messages": [{"role": "user", "content": "unchanged"}]}
    result: Final = await llm_guard.async_moderation_hook(
        data=data, user_api_key_dict=UserAPIKeyAuth(), call_type=call_type
    )
    assert result is data
    assert data == {"messages": [{"role": "user", "content": "unchanged"}]}


@pytest.mark.asyncio
async def test_llm_guard_error_raising():
    """
    Tests to see llm guard raises an error for a flagged response
    """

    input_b_anonymizer_results = {
        "sanitized_prompt": "hello world",
        "is_valid": False,
        "scanners": {"Regex": 0.0},
    }
    llm_guard = _ENTERPRISE_LLMGuard(
        mock_testing=True, mock_redacted_text=input_b_anonymizer_results
    )

    _api_key = "sk-12345"
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    with pytest.raises(HTTPException) as exc_info:
        await llm_guard.async_moderation_hook(
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": "hello world, my name is Jane Doe. My number is: 23r323r23r2wwkl",
                    }
                ]
            },
            user_api_key_dict=user_api_key_dict,
            call_type="completion",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {"error": "Violated content safety policy"}


def test_llm_guard_key_specific_mode():
    """
    Tests to see if llm guard 'key-specific' permissions work
    """
    litellm.llm_guard_mode = "key-specific"

    llm_guard = _ENTERPRISE_LLMGuard(mock_testing=True)

    _api_key = "sk-12345"
    # NOT ENABLED
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
    )

    request_data = {}
    should_proceed = llm_guard.should_proceed(
        user_api_key_dict=user_api_key_dict, data=request_data
    )

    assert should_proceed == False

    # ENABLED
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key, permissions={"enable_llm_guard_check": True}
    )

    request_data = {}

    should_proceed = llm_guard.should_proceed(
        user_api_key_dict=user_api_key_dict, data=request_data
    )

    assert should_proceed == True


def test_llm_guard_request_specific_mode():
    """
    Tests to see if llm guard 'request-specific' permissions work
    """
    litellm.llm_guard_mode = "request-specific"

    llm_guard = _ENTERPRISE_LLMGuard(mock_testing=True)

    _api_key = "sk-12345"
    # NOT ENABLED
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
    )

    request_data = {}

    should_proceed = llm_guard.should_proceed(
        user_api_key_dict=user_api_key_dict, data=request_data
    )

    assert should_proceed == False

    # ENABLED
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key, permissions={"enable_llm_guard_check": True}
    )

    request_data = {"metadata": {"permissions": {"enable_llm_guard_check": True}}}

    should_proceed = llm_guard.should_proceed(
        user_api_key_dict=user_api_key_dict, data=request_data
    )

    assert should_proceed == True
