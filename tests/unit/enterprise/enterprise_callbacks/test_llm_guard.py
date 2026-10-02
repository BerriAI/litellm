from typing import Final, Literal

import pytest
from litellm_enterprise.enterprise_callbacks.llm_guard import _ENTERPRISE_LLMGuard
from starlette.exceptions import HTTPException

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import hash_token
from litellm.types.utils import CallTypesLiteral


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
        ("image_generation", "prompt"),
        ("aimage_generation", "prompt"),
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
            await llm_guard.async_moderation_hook(data=data, user_api_key_dict=user_api_key_dict, call_type=call_type)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == {"error": "Violated content safety policy"}
        return

    result: Final = await llm_guard.async_moderation_hook(
        data=data, user_api_key_dict=user_api_key_dict, call_type=call_type
    )
    assert result is data
    assert data[payload_key] == (
        [{"role": "user", "content": "email: [REDACTED]"}] if payload_key == "messages" else "email: [REDACTED]"
    )


@pytest.mark.parametrize("call_type", ("amoderation", "atranscription", "aresponses", "aanthropic_messages"))
@pytest.mark.asyncio
async def test_llm_guard_ignores_call_types_the_proxy_never_moderates(
    call_type: CallTypesLiteral, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "llm_guard_mode", "all")
    llm_guard: Final = _ENTERPRISE_LLMGuard(
        mock_testing=True,
        mock_redacted_text={"sanitized_prompt": "[REDACTED]", "is_valid": False},
    )
    data: Final = {"input": "email: person@example.com"}
    result: Final = await llm_guard.async_moderation_hook(
        data=data, user_api_key_dict=UserAPIKeyAuth(), call_type=call_type
    )
    assert result is data
    assert data["input"] == "email: person@example.com"


@pytest.mark.parametrize("call_type", ("text_completion", "atext_completion"))
@pytest.mark.parametrize("is_valid", (True, False))
@pytest.mark.asyncio
async def test_llm_guard_scans_list_prompt(
    call_type: CallTypesLiteral, is_valid: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "llm_guard_mode", "all")
    llm_guard: Final = _ENTERPRISE_LLMGuard(
        mock_testing=True,
        mock_redacted_text={"sanitized_prompt": "[REDACTED]", "is_valid": is_valid},
    )
    data: Final = {"prompt": ["email: person@example.com", "say ok", [1, 2, 3]]}

    if not is_valid:
        with pytest.raises(HTTPException) as exc_info:
            await llm_guard.async_moderation_hook(data=data, user_api_key_dict=UserAPIKeyAuth(), call_type=call_type)
        assert exc_info.value.status_code == 400
        return

    result: Final = await llm_guard.async_moderation_hook(
        data=data, user_api_key_dict=UserAPIKeyAuth(), call_type=call_type
    )
    assert result is data
    assert data["prompt"] == ["[REDACTED]", "[REDACTED]", [1, 2, 3]]


@pytest.mark.parametrize("call_type", ("aembedding", "atext_completion"))
@pytest.mark.parametrize("is_valid", (True, False))
@pytest.mark.asyncio
async def test_llm_guard_scans_input_and_prompt_alongside_messages(
    call_type: CallTypesLiteral, is_valid: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "llm_guard_mode", "all")
    llm_guard: Final = _ENTERPRISE_LLMGuard(
        mock_testing=True,
        mock_redacted_text={"sanitized_prompt": "[REDACTED]", "is_valid": is_valid},
    )
    data: Final = {
        "messages": [],
        "input": "email: person@example.com",
        "prompt": ["say ok"],
    }

    if not is_valid:
        with pytest.raises(HTTPException) as exc_info:
            await llm_guard.async_moderation_hook(data=data, user_api_key_dict=UserAPIKeyAuth(), call_type=call_type)
        assert exc_info.value.status_code == 400
        return

    result: Final = await llm_guard.async_moderation_hook(
        data=data, user_api_key_dict=UserAPIKeyAuth(), call_type=call_type
    )
    assert result is data
    assert data["messages"] == []
    assert data["input"] == "[REDACTED]"
    assert data["prompt"] == ["[REDACTED]"]


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
