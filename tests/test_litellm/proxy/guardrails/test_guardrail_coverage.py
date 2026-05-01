"""
Regression tests for guardrail-coverage gaps.

Each test confirms that a previously-bypassable input shape now triggers
inspection by the relevant guardrail hook:

- VERIA-11: multimodal list-format ``content`` is inspected (no longer
  silently skipped because of an ``isinstance(content, str)`` check).
- fniVO9-F: Responses-API ``data["input"]`` is inspected (no longer
  silently skipped because the hook only looked at ``data["messages"]``).
- yVS0wMDO: Aim's post-call hook inspects every choice when ``n>1``,
  not just ``choices[0]``.
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import Request, Response

from litellm import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import Choices, Message, ModelResponse


@pytest.fixture
def user_api_key():
    return UserAPIKeyAuth(api_key="hashed", user_id="u", key_alias=None)


# ── Aim ───────────────────────────────────────────────────────────────────────


def _aim_no_action_response() -> Response:
    return Response(
        status_code=200,
        json={"required_action": None},
        request=Request("POST", "https://api.aim.security/fw/v1/analyze"),
    )


@pytest.mark.asyncio
async def test_aim_inspects_multimodal_list_content(user_api_key, monkeypatch):
    monkeypatch.setenv("AIM_API_KEY", "hs-aim-key")
    from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail

    guard = AimGuardrail()
    sent_payload: Dict[str, Any] = {}

    async def capture(url, headers, json):
        sent_payload.update(json)
        return _aim_no_action_response()

    with patch.object(guard.async_handler, "post", side_effect=capture):
        await guard.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=DualCache(),
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "secret payload"},
                            {"type": "image_url", "image_url": {"url": "..."}},
                        ],
                    }
                ]
            },
            call_type="acompletion",
        )

    # The multimodal text part must be visible to Aim.
    assert sent_payload["messages"] == [{"role": "user", "content": "secret payload"}]


@pytest.mark.asyncio
async def test_aim_inspects_responses_api_input(user_api_key, monkeypatch):
    monkeypatch.setenv("AIM_API_KEY", "hs-aim-key")
    from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail

    guard = AimGuardrail()
    sent_payload: Dict[str, Any] = {}

    async def capture(url, headers, json):
        sent_payload.update(json)
        return _aim_no_action_response()

    with patch.object(guard.async_handler, "post", side_effect=capture):
        await guard.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=DualCache(),
            data={"input": "responses-api content"},
            call_type="acompletion",
        )

    assert sent_payload["messages"] == [
        {"role": "user", "content": "responses-api content"}
    ]


@pytest.mark.asyncio
async def test_aim_post_call_inspects_all_choices(user_api_key, monkeypatch):
    """yVS0wMDO: ``n>1`` no longer bypasses Aim by hiding violations in
    ``choices[1+]``."""
    monkeypatch.setenv("AIM_API_KEY", "hs-aim-key")
    from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail

    guard = AimGuardrail()
    inspected_outputs = []

    async def capture(request_data, output, hook, key_alias):
        inspected_outputs.append(output)
        return {"redacted_output": output}

    response = ModelResponse(
        choices=[
            Choices(index=0, message=Message(role="assistant", content="first")),
            Choices(index=1, message=Message(role="assistant", content="second")),
            Choices(index=2, message=Message(role="assistant", content="third")),
        ]
    )

    with patch.object(guard, "call_aim_guardrail_on_output", side_effect=capture):
        await guard.async_post_call_success_hook(
            data={"messages": [{"role": "user", "content": "hi"}]},
            user_api_key_dict=user_api_key,
            response=response,
        )

    # ``asyncio.gather`` is used for parallelism, so order of inspection is
    # not guaranteed.
    assert sorted(inspected_outputs) == ["first", "second", "third"]


# ── Lakera v2 ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lakera_v2_inspects_responses_api_input(user_api_key, monkeypatch):
    monkeypatch.setenv("LAKERA_API_KEY", "lk-test")
    from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import (
        LakeraAIGuardrail,
    )

    guard = LakeraAIGuardrail(api_key="lk-test", on_flagged="monitor")

    seen_messages = []

    async def fake_call_v2_guard(messages, request_data, event_type):
        seen_messages.append(messages)
        return {"flagged": False}, {}

    with patch.object(guard, "call_v2_guard", side_effect=fake_call_v2_guard):
        await guard.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=DualCache(),
            data={"input": "responses-api content"},
            call_type="responses",
        )

    assert seen_messages == [[{"role": "user", "content": "responses-api content"}]]


@pytest.mark.asyncio
async def test_lakera_v2_responses_api_input_redacted_writeback(
    user_api_key, monkeypatch
):
    """Greptile P1: when input arrives via Responses-API ``data["input"]``
    (string) and Lakera flags PII, the redacted content must be written
    back to ``data["input"]`` — the Responses-API backend reads from
    ``input``, so writing only to ``messages`` would let unredacted PII
    reach the LLM."""
    monkeypatch.setenv("LAKERA_API_KEY", "lk-test")
    from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import (
        LakeraAIGuardrail,
    )

    guard = LakeraAIGuardrail(api_key="lk-test", on_flagged="block")

    async def fake_call_v2_guard(messages, request_data, event_type):
        return ({"flagged": True, "payload": []}, {"EMAIL": 1})

    def fake_mask(messages, lakera_response, masked_entity_count):
        return [{"role": "user", "content": "[REDACTED EMAIL]"}]

    with (
        patch.object(guard, "call_v2_guard", side_effect=fake_call_v2_guard),
        patch.object(guard, "_is_only_pii_violation", return_value=True),
        patch.object(guard, "_mask_pii_in_messages", side_effect=fake_mask),
    ):
        data = {"input": "user@example.com leaked"}
        await guard.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=DualCache(),
            data=data,
            call_type="responses",
        )

    assert data["input"] == "[REDACTED EMAIL]"


@pytest.mark.asyncio
async def test_aim_responses_api_input_anonymize_writeback(user_api_key, monkeypatch):
    """Greptile P1: Aim's anonymize action must redact ``data["input"]``
    for Responses-API requests, not just ``data["messages"]``."""
    monkeypatch.setenv("AIM_API_KEY", "hs-aim-key")
    from litellm.proxy.guardrails.guardrail_hooks.aim.aim import AimGuardrail

    guard = AimGuardrail()

    aim_response_body = {
        "required_action": {"action_type": "anonymize_action"},
        "redacted_chat": {
            "all_redacted_messages": [
                {"role": "user", "content": "[REDACTED] anonymised"}
            ]
        },
    }

    async def capture(url, headers, json):
        return Response(
            status_code=200,
            json=aim_response_body,
            request=Request("POST", "https://api.aim.security/fw/v1/analyze"),
        )

    with patch.object(guard.async_handler, "post", side_effect=capture):
        data = {"input": "user@example.com leaked"}
        await guard.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=DualCache(),
            data=data,
            call_type="responses",
        )

    assert data["input"] == "[REDACTED] anonymised"


@pytest.mark.asyncio
async def test_lakera_v2_multimodal_pii_degrades_to_block(user_api_key, monkeypatch):
    """Mask-in-place uses Lakera offsets and cannot preserve image/audio
    parts of multimodal input. When PII is detected on a multimodal
    request, the hook must raise the block exception instead of silently
    flattening ``data["messages"]`` to text-only."""
    monkeypatch.setenv("LAKERA_API_KEY", "lk-test")
    from fastapi import HTTPException

    from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import (
        LakeraAIGuardrail,
    )

    guard = LakeraAIGuardrail(api_key="lk-test", on_flagged="block")

    async def fake_call_v2_guard(messages, request_data, event_type):
        return (
            {
                "flagged": True,
                "payload": [{"detector_type": "pii/email", "start": 0, "end": 5}],
            },
            {"EMAIL": 1},
        )

    with (
        patch.object(guard, "call_v2_guard", side_effect=fake_call_v2_guard),
        patch.object(guard, "_is_only_pii_violation", return_value=True),
        patch.object(
            guard,
            "_get_http_exception_for_blocked_guardrail",
            return_value=HTTPException(status_code=400, detail="blocked"),
        ),
    ):
        with pytest.raises(HTTPException):
            await guard.async_pre_call_hook(
                user_api_key_dict=user_api_key,
                cache=DualCache(),
                data={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "leak"},
                                {"type": "image_url", "image_url": {"url": "..."}},
                            ],
                        }
                    ]
                },
                call_type="acompletion",
            )


@pytest.mark.asyncio
async def test_lakera_v2_inspects_multimodal_list_content(user_api_key, monkeypatch):
    monkeypatch.setenv("LAKERA_API_KEY", "lk-test")
    from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import (
        LakeraAIGuardrail,
    )

    guard = LakeraAIGuardrail(api_key="lk-test", on_flagged="monitor")
    seen_messages = []

    async def fake_call_v2_guard(messages, request_data, event_type):
        seen_messages.append(messages)
        return {"flagged": False}, {}

    with patch.object(guard, "call_v2_guard", side_effect=fake_call_v2_guard):
        await guard.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=DualCache(),
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "AKIAEXAMPLE"},
                            {"type": "image_url", "image_url": {"url": "..."}},
                        ],
                    }
                ]
            },
            call_type="acompletion",
        )

    assert seen_messages == [[{"role": "user", "content": "AKIAEXAMPLE"}]]


# ── Lasso ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lasso_multimodal_falls_back_to_classify(user_api_key, monkeypatch):
    """Lasso's classifix (mask) endpoint returns text that overwrites
    ``data["messages"]``. For multimodal input that would silently strip
    image parts — the hook must use the classify endpoint instead and
    leave the original payload intact."""
    monkeypatch.setenv("LASSO_API_KEY", "ls-test")
    from litellm.proxy.guardrails.guardrail_hooks.lasso.lasso import LassoGuardrail

    guard = LassoGuardrail(lasso_api_key="ls-test", mask=True)

    masking_called = False
    classify_called = False

    async def fake_masking(data, cache, message_type, messages):
        nonlocal masking_called
        masking_called = True
        return data

    async def fake_classification(data, cache, message_type, messages):
        nonlocal classify_called
        classify_called = True
        return data

    with (
        patch.object(guard, "_handle_masking", side_effect=fake_masking),
        patch.object(guard, "_handle_classification", side_effect=fake_classification),
    ):
        await guard._run_lasso_guardrail(
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "image_url", "image_url": {"url": "..."}},
                        ],
                    }
                ]
            },
            cache=DualCache(),
            message_type="PROMPT",
        )

    assert classify_called is True
    assert masking_called is False


@pytest.mark.asyncio
async def test_lasso_inspects_responses_api_input(user_api_key, monkeypatch):
    monkeypatch.setenv("LASSO_API_KEY", "ls-test")
    from litellm.proxy.guardrails.guardrail_hooks.lasso.lasso import LassoGuardrail

    guard = LassoGuardrail(lasso_api_key="ls-test")

    seen_messages = []

    async def fake_handle_classification(data, cache, message_type, messages):
        seen_messages.append(messages)
        return data

    with patch.object(
        guard, "_handle_classification", side_effect=fake_handle_classification
    ):
        await guard._run_lasso_guardrail(
            data={"input": "responses-api content"},
            cache=DualCache(),
            message_type="PROMPT",
        )

    assert seen_messages == [[{"role": "user", "content": "responses-api content"}]]


# ── Banned Keywords ───────────────────────────────────────────────────────────


def test_banned_keywords_blocks_multimodal_content(monkeypatch):
    """VERIA-11: a banned word hidden in a multimodal text part is now caught."""
    monkeypatch.setattr("litellm.banned_keywords_list", ["forbidden"], raising=False)
    from enterprise.enterprise_hooks.banned_keywords import _ENTERPRISE_BannedKeywords
    from fastapi import HTTPException

    guard = _ENTERPRISE_BannedKeywords()

    async def _run():
        await guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed", user_id="u"),
            cache=DualCache(),
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "forbidden word here"},
                            {"type": "image_url", "image_url": {"url": "..."}},
                        ],
                    }
                ]
            },
            call_type="completion",
        )

    import asyncio

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_run())
    assert "forbidden" in str(exc.value.detail).lower()


def test_banned_keywords_blocks_responses_api_input(monkeypatch):
    monkeypatch.setattr("litellm.banned_keywords_list", ["forbidden"], raising=False)
    from enterprise.enterprise_hooks.banned_keywords import _ENTERPRISE_BannedKeywords
    from fastapi import HTTPException

    guard = _ENTERPRISE_BannedKeywords()

    async def _run():
        await guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed", user_id="u"),
            cache=DualCache(),
            data={"input": "this contains forbidden content"},
            call_type="completion",
        )

    import asyncio

    with pytest.raises(HTTPException):
        asyncio.run(_run())


# ── Secret Detection ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_secret_detection_redacts_multimodal_text_parts(user_api_key):
    from enterprise.litellm_enterprise.enterprise_callbacks.secret_detection import (
        _ENTERPRISE_SecretDetection,
    )

    guard = _ENTERPRISE_SecretDetection()
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "AKIAIOSFODNN7EXAMPLE is the key",
                    },
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]
    }

    await guard.async_pre_call_hook(
        user_api_key_dict=user_api_key,
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    parts = data["messages"][0]["content"]
    assert "AKIAIOSFODNN7EXAMPLE" not in parts[0]["text"]
    assert "[REDACTED]" in parts[0]["text"]
    # Non-text part is preserved untouched.
    assert parts[1] == {"type": "image_url", "image_url": {"url": "..."}}


@pytest.mark.asyncio
async def test_secret_detection_redacts_responses_api_input(user_api_key):
    from enterprise.litellm_enterprise.enterprise_callbacks.secret_detection import (
        _ENTERPRISE_SecretDetection,
    )

    guard = _ENTERPRISE_SecretDetection()
    data = {"input": "leak: AKIAIOSFODNN7EXAMPLE"}

    await guard.async_pre_call_hook(
        user_api_key_dict=user_api_key,
        cache=DualCache(),
        data=data,
        call_type="moderation",
    )

    assert "AKIAIOSFODNN7EXAMPLE" not in data["input"]
    assert "[REDACTED]" in data["input"]


# ── OpenAI Moderation ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_moderation_inspects_multimodal_content(monkeypatch, user_api_key):
    """The aggregated text passed to ``llm_router.amoderation`` must include
    list-format text parts and Responses-API input — without this, multimodal
    content silently passed moderation."""
    from enterprise.enterprise_hooks.openai_moderation import (
        _ENTERPRISE_OpenAI_Moderation,
    )

    guard = _ENTERPRISE_OpenAI_Moderation()

    seen_inputs = []

    class FakeModeration:
        results = [type("R", (), {"flagged": False})()]

    async def fake_amoderation(model, input):
        seen_inputs.append(input)
        return FakeModeration()

    fake_router = MagicMock()
    fake_router.amoderation = AsyncMock(side_effect=fake_amoderation)

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.llm_router", fake_router, raising=False
    )

    await guard.async_moderation_hook(
        data={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "alpha "},
                        {"type": "image_url", "image_url": {"url": "..."}},
                        {"type": "text", "text": "beta"},
                    ],
                }
            ]
        },
        user_api_key_dict=user_api_key,
        call_type="acompletion",
    )

    assert seen_inputs == ["alpha beta"]


# ── Google Text Moderation ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_google_text_moderation_inspects_multimodal_content(user_api_key):
    """The text passed to Google's moderation client must include list-format
    text parts."""
    from enterprise.enterprise_hooks.google_text_moderation import (
        _ENTERPRISE_GoogleTextModeration,
    )

    guard = _ENTERPRISE_GoogleTextModeration.__new__(_ENTERPRISE_GoogleTextModeration)
    seen_documents = []

    def fake_language_document(content, type_):
        seen_documents.append(content)
        return MagicMock()

    fake_response = MagicMock()
    fake_response.moderation_categories = []

    guard.language_document = fake_language_document
    guard.moderate_text_request = MagicMock(return_value=MagicMock())
    guard.document_type = MagicMock()
    guard.client = MagicMock()
    guard.client.moderate_text = MagicMock(return_value=fake_response)

    await guard.async_moderation_hook(
        data={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "image_url", "image_url": {"url": "..."}},
                        {"type": "text", "text": "world"},
                    ],
                }
            ]
        },
        user_api_key_dict=user_api_key,
        call_type="acompletion",
    )

    assert seen_documents == ["hello world"]
