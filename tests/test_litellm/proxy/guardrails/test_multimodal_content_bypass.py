"""
Regression tests for the multimodal-content guardrail bypass.

Each guardrail covered here previously short-circuited on
`isinstance(content, str)` and silently skipped multimodal list-shaped
message content (e.g. `[{"type": "text", "text": "..."}]`), allowing users
to evade scanning. These tests assert that the same logical payload triggers
the same guardrail behavior whether it arrives as a plain string or as a
multimodal list.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


def _string_messages(payload: str):
    return [{"role": "user", "content": payload}]


def _multimodal_messages(payload: str):
    return [{"role": "user", "content": [{"type": "text", "text": payload}]}]


@pytest.fixture(params=["string", "multimodal"], ids=["string", "multimodal"])
def messages_factory(request):
    return _string_messages if request.param == "string" else _multimodal_messages


# ---------------------------------------------------------------------------
# secret_detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_secret_detection_redacts_in_both_shapes(messages_factory):
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm_enterprise.enterprise_callbacks.secret_detection import (
        _ENTERPRISE_SecretDetection,
    )

    secret_instance = _ENTERPRISE_SecretDetection()
    payload = "API_KEY = 'sk_1234567890abcdef'"
    data = {"messages": messages_factory(payload), "model": "gpt-3.5-turbo"}

    await secret_instance.async_pre_call_hook(
        cache=DualCache(),
        data=data,
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-12345"),
        call_type="completion",
    )

    content = data["messages"][0]["content"]
    if isinstance(content, str):
        assert "[REDACTED]" in content
        assert "sk_1234567890abcdef" not in content
    else:
        assert "[REDACTED]" in content[0]["text"]
        assert "sk_1234567890abcdef" not in content[0]["text"]


@pytest.mark.asyncio
async def test_secret_detection_redacts_prompt_list_writes_back():
    """Regression for the latent bug where `for item in data['prompt']: item = ...`
    rebound the loop variable but never wrote back."""
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm_enterprise.enterprise_callbacks.secret_detection import (
        _ENTERPRISE_SecretDetection,
    )

    secret_instance = _ENTERPRISE_SecretDetection()
    data = {
        "prompt": [
            "first prompt with API_KEY = 'sk_1234567890abcdef'",
            "second clean prompt",
        ]
    }

    await secret_instance.async_pre_call_hook(
        cache=DualCache(),
        data=data,
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-12345"),
        call_type="completion",
    )

    assert "[REDACTED]" in data["prompt"][0]
    assert "sk_1234567890abcdef" not in data["prompt"][0]
    assert data["prompt"][1] == "second clean prompt"


# ---------------------------------------------------------------------------
# banned_keywords
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_banned_keywords_blocks_in_both_shapes(messages_factory):
    import litellm
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth

    litellm.banned_keywords_list = ["forbidden"]
    from enterprise.enterprise_hooks.banned_keywords import _ENTERPRISE_BannedKeywords

    instance = _ENTERPRISE_BannedKeywords()
    data = {"messages": messages_factory("please say forbidden out loud")}

    with pytest.raises(HTTPException) as exc:
        await instance.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-12345"),
            cache=DualCache(),
            call_type="completion",
            data=data,
        )
    assert "forbidden" in str(exc.value.detail)


# ---------------------------------------------------------------------------
# openai_moderation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_moderation_sends_text_in_both_shapes(messages_factory):
    from enterprise.enterprise_hooks.openai_moderation import (
        _ENTERPRISE_OpenAI_Moderation,
    )
    from litellm.proxy._types import UserAPIKeyAuth

    instance = _ENTERPRISE_OpenAI_Moderation()
    payload = "scan-this-text"
    data = {"messages": messages_factory(payload)}

    fake_router = MagicMock()
    fake_response = MagicMock()
    fake_response.results = [MagicMock(flagged=False)]
    fake_router.amoderation = AsyncMock(return_value=fake_response)

    with patch("litellm.proxy.proxy_server.llm_router", fake_router):
        await instance.async_moderation_hook(
            data=data,
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-12345"),
            call_type="completion",
        )

    fake_router.amoderation.assert_awaited_once()
    sent_input = fake_router.amoderation.await_args.kwargs["input"]
    assert payload in sent_input


# ---------------------------------------------------------------------------
# google_text_moderation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_google_text_moderation_sends_text_in_both_shapes(messages_factory):
    pytest.importorskip("google.cloud.language_v1")
    from enterprise.enterprise_hooks.google_text_moderation import (
        _ENTERPRISE_GoogleTextModeration,
    )
    from litellm.proxy._types import UserAPIKeyAuth

    # Skip __init__ to avoid spinning up a real Google client.
    instance = _ENTERPRISE_GoogleTextModeration.__new__(
        _ENTERPRISE_GoogleTextModeration
    )
    captured = {}

    def fake_document(content, type_):
        captured["content"] = content
        return {"content": content}

    def fake_request(document):
        return {"document": document}

    fake_client = MagicMock()
    fake_client.moderate_text = MagicMock(
        return_value=MagicMock(moderation_categories=[])
    )
    instance.client = fake_client
    instance.language_document = fake_document
    instance.moderate_text_request = fake_request
    instance.document_type = "plain"

    payload = "scan-this-text"
    await instance.async_moderation_hook(
        data={"messages": messages_factory(payload)},
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-12345"),
        call_type="completion",
    )

    assert payload in captured["content"]


# ---------------------------------------------------------------------------
# azure_content_safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_azure_content_safety_scans_in_both_shapes(messages_factory):
    pytest.importorskip("azure.ai.contentsafety")
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.azure_content_safety import _PROXY_AzureContentSafety

    # Skip __init__ to avoid the real Azure client.
    instance = _PROXY_AzureContentSafety.__new__(_PROXY_AzureContentSafety)
    instance.test_violation = AsyncMock()

    payload = "anything"
    await instance.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-12345"),
        cache=DualCache(),
        data={"messages": messages_factory(payload)},
        call_type="completion",
    )

    instance.test_violation.assert_awaited_once_with(content=payload, source="input")


# ---------------------------------------------------------------------------
# ibm_detector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ibm_detector_scans_in_both_shapes(messages_factory):
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.guardrails.guardrail_hooks.ibm_guardrails.ibm_detector import (
        IBMGuardrailDetector,
    )

    instance = IBMGuardrailDetector(
        guardrail_name="ibm_test",
        auth_token="t",
        base_url="https://x",
        detector_id="d",
        is_detector_server=True,
    )
    instance.should_run_guardrail = MagicMock(return_value=True)
    instance._call_detector_server = AsyncMock(return_value=[[]])
    instance._filter_detections_by_threshold = MagicMock(return_value=[])

    payload = "scan-me"
    await instance.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-12345"),
        cache=MagicMock(),
        data={"messages": messages_factory(payload)},
        call_type="completion",
    )

    instance._call_detector_server.assert_awaited_once()
    sent_contents = instance._call_detector_server.await_args.kwargs["contents"]
    assert sent_contents == [payload]


# ---------------------------------------------------------------------------
# lakera_ai_v2
# ---------------------------------------------------------------------------


def test_lakera_flatten_messages_for_lakera_collapses_multimodal():
    from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import LakeraAIGuardrail

    messages = [
        {"role": "user", "content": "plain"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first "},
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "second"},
            ],
        },
    ]
    flattened = LakeraAIGuardrail._flatten_messages_for_lakera(messages)
    assert flattened[0]["content"] == "plain"
    assert flattened[1]["content"] == "first second"
    # original is not mutated
    assert isinstance(messages[1]["content"], list)


def test_lakera_mask_pii_writes_back_into_multimodal_first_text_part():
    from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import LakeraAIGuardrail

    instance = LakeraAIGuardrail.__new__(LakeraAIGuardrail)
    # The flattened text Lakera saw: "my secret is hunter2"
    # Lakera reports an offset for "hunter2" (positions 13..20 in the flattened string).
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "my secret is "},
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "hunter2"},
            ],
        }
    ]
    lakera_response = {
        "payload": [
            {
                "message_id": 0,
                "start": 13,
                "end": 20,
                "detector_type": "pii/password",
            }
        ]
    }
    masked = instance._mask_pii_in_messages(messages, lakera_response, {})
    parts = masked[0]["content"]
    # First text part holds the masked result; remaining text parts cleared;
    # image part preserved.
    assert parts[0]["text"] == "my secret is [MASKED PASSWORD]"
    assert parts[1] == {"type": "image_url", "image_url": {"url": "x"}}
    assert parts[2]["text"] == ""
