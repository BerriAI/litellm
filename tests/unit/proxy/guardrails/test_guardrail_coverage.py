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

import json
import sys
from collections.abc import Mapping
from functools import reduce
from types import SimpleNamespace
from typing import Any, Dict, Final, NamedTuple
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException
from httpx import Request, Response
from pydantic import JsonValue

import litellm
from enterprise.enterprise_hooks.google_text_moderation import ENTERPRISE_GoogleTextModeration
from litellm import DualCache
from litellm.constants import DEFAULT_OPENAI_MODERATIONS_MODEL
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

    async def capture(url, headers, json, **_kwargs):
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

    async def capture(url, headers, json, **_kwargs):
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

    assert seen_messages == [({"role": "user", "content": "responses-api content"},)]


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

    async def capture(url, headers, json, **_kwargs):
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

    assert seen_messages == [({"role": "user", "content": "AKIAEXAMPLE"},)]


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


@pytest.mark.asyncio
async def test_lasso_masking_writes_back_responses_api_input(user_api_key, monkeypatch):
    """Krrish blocker: Lasso classifix masking must update ``data["input"]``
    for Responses-API requests, not only ``data["messages"]``."""
    monkeypatch.setenv("LASSO_API_KEY", "ls-test")
    from litellm.proxy.guardrails.guardrail_hooks.lasso.lasso import LassoGuardrail

    guard = LassoGuardrail(lasso_api_key="ls-test", mask=True)
    lasso_response = {
        "violations_detected": True,
        "deputies": {"pii": True},
        "findings": {"pii": [{"action": "AUTO_MASKING"}]},
        "messages": [{"role": "user", "content": "[REDACTED]"}],
    }

    async def fake_call_lasso_api(headers, payload, api_url=None):
        return lasso_response

    data = {"input": "user@example.com leaked"}

    with patch.object(guard, "_call_lasso_api", side_effect=fake_call_lasso_api):
        await guard._run_lasso_guardrail(
            data=data,
            cache=DualCache(),
            message_type="PROMPT",
        )

    assert data["input"] == "[REDACTED]"


# ── Banned Keywords ───────────────────────────────────────────────────────────


def test_banned_keywords_blocks_multimodal_content(monkeypatch):
    """VERIA-11: a banned word hidden in a multimodal text part is now caught.

    Uses ``acompletion`` — the value the proxy ingress actually passes
    for ``/v1/chat/completions``. Asserting against the literal sync
    ``"completion"`` would pass even if the hook's call-type gate were
    misaligned with the runtime, so the test wouldn't catch regressions.
    """
    monkeypatch.setattr("litellm.banned_keywords_list", ["forbidden"], raising=False)
    from enterprise.enterprise_hooks.banned_keywords import ENTERPRISE_BannedKeywords
    from fastapi import HTTPException

    guard = ENTERPRISE_BannedKeywords()

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
            call_type="acompletion",
        )

    import asyncio

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_run())
    assert "forbidden" in str(exc.value.detail).lower()


def test_banned_keywords_blocks_responses_api_input(monkeypatch):
    monkeypatch.setattr("litellm.banned_keywords_list", ["forbidden"], raising=False)
    from enterprise.enterprise_hooks.banned_keywords import ENTERPRISE_BannedKeywords
    from fastapi import HTTPException

    guard = ENTERPRISE_BannedKeywords()

    async def _run():
        await guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed", user_id="u"),
            cache=DualCache(),
            data={"input": "this contains forbidden content"},
            call_type="aresponses",
        )

    import asyncio

    with pytest.raises(HTTPException):
        asyncio.run(_run())


def test_banned_keywords_still_checks_messages_after_a_deep_tool_result_chain(monkeypatch):
    monkeypatch.setattr("litellm.banned_keywords_list", ["forbidden"], raising=False)
    from enterprise.enterprise_hooks.banned_keywords import ENTERPRISE_BannedKeywords

    chain: Final = reduce(
        lambda content, _: [{"type": "tool_result", "tool_use_id": "tool", "content": content}],
        range(1500),
        [{"type": "text", "text": "inner"}],
    )
    data: Final = {"messages": [{"role": "assistant", "content": chain}, {"role": "user", "content": "forbidden"}]}

    import asyncio

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            ENTERPRISE_BannedKeywords().async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="hashed", user_id="u"),
                cache=DualCache(),
                data=data,
                call_type="acompletion",
            )
        )
    assert "forbidden" in str(exc.value.detail).lower()


@pytest.mark.parametrize("call_type", ["completion", "acompletion", "aresponses"])
def test_banned_keywords_fires_on_text_content_call_types(monkeypatch, call_type):
    """Locks the call-type gate to the runtime ``route_type`` values the
    proxy actually emits — pinning a regression where the hook had
    ``call_type == "completion"`` and silently no-op'd both
    ``acompletion`` (chat completions) and ``aresponses`` (Responses API).
    """
    monkeypatch.setattr("litellm.banned_keywords_list", ["forbidden"], raising=False)
    from enterprise.enterprise_hooks.banned_keywords import ENTERPRISE_BannedKeywords
    from fastapi import HTTPException

    guard = ENTERPRISE_BannedKeywords()

    import asyncio

    with pytest.raises(HTTPException):
        asyncio.run(
            guard.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="hashed", user_id="u"),
                cache=DualCache(),
                data={
                    "messages": [{"role": "user", "content": "forbidden text"}],
                    "input": "forbidden text",
                },
                call_type=call_type,
            )
        )


def test_banned_keywords_skips_non_text_call_types(monkeypatch):
    """Embedding / moderation / audio paths don't carry chat text and
    aren't in the text-guardrail scope. They must not trigger the hook
    even when the request body otherwise looks like a chat payload.
    """
    monkeypatch.setattr("litellm.banned_keywords_list", ["forbidden"], raising=False)
    from enterprise.enterprise_hooks.banned_keywords import ENTERPRISE_BannedKeywords

    guard = ENTERPRISE_BannedKeywords()

    import asyncio

    for call_type in ("aembedding", "amoderation", "aspeech", "atranscription"):
        # Should return without raising, even though the data carries the banned word.
        asyncio.run(
            guard.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="hashed", user_id="u"),
                cache=DualCache(),
                data={"input": "forbidden text"},
                call_type=call_type,
            )
        )


@pytest.mark.asyncio
async def test_banned_keywords_post_call_checks_all_choices(monkeypatch, user_api_key):
    """Krrish blocker: ``n>1`` responses must not bypass post-call checks by
    placing the banned text in ``choices[1+]``."""
    monkeypatch.setattr("litellm.banned_keywords_list", ["forbidden"], raising=False)
    from enterprise.enterprise_hooks.banned_keywords import ENTERPRISE_BannedKeywords
    from fastapi import HTTPException

    guard = ENTERPRISE_BannedKeywords()
    response = ModelResponse(
        choices=[
            Choices(index=0, message=Message(role="assistant", content="clean")),
            Choices(index=1, message=Message(role="assistant", content="forbidden")),
        ]
    )

    with pytest.raises(HTTPException) as exc:
        await guard.async_post_call_success_hook(
            data={},
            user_api_key_dict=user_api_key,
            response=response,
        )

    assert "forbidden" in str(exc.value.detail).lower()


# ── Azure Content Safety ──────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type, data",
    [
        (
            "acompletion",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "scan me"},
                            {"type": "image_url", "image_url": {"url": "..."}},
                        ],
                    }
                ]
            },
        ),
        ("aresponses", {"input": "scan me"}),
    ],
)
async def test_azure_content_safety_pre_call_fires_on_runtime_call_types(
    user_api_key, call_type, data
):
    """The proxy ingress passes ``route_type`` straight through as
    ``call_type`` — ``acompletion`` for chat completions and
    ``aresponses`` for the Responses API. The hook must inspect text
    fragments under both, not only the literal ``"completion"`` string
    used by some SDK callers."""
    from litellm.proxy.hooks.azure_content_safety import PROXY_AzureContentSafety

    guard = PROXY_AzureContentSafety.__new__(PROXY_AzureContentSafety)
    seen = []

    async def fake_test_violation(content, source=None):
        seen.append((content, source))

    guard.test_violation = fake_test_violation
    await guard.async_pre_call_hook(
        user_api_key_dict=user_api_key,
        cache=DualCache(),
        data=data,
        call_type=call_type,
    )
    assert ("scan me", "input") in seen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type, data",
    [
        ("aresponses", {"instructions": "x" * 10001, "input": [{"role": "user", "content": "flagged text"}]}),
        (
            "acompletion",
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "write_file", "arguments": json.dumps({"content": "x" * 10001})},
                            }
                        ],
                    },
                    {"role": "user", "content": "flagged text"},
                ]
            },
        ),
    ],
    ids=("responses-instructions", "chat-tool-arguments"),
)
async def test_azure_content_safety_checks_message_text_before_oversized_tool_or_system_text(
    user_api_key: UserAPIKeyAuth, call_type: str, data: dict[str, JsonValue]
) -> None:
    from litellm.proxy.hooks.azure_content_safety import PROXY_AzureContentSafety

    guard: Final = PROXY_AzureContentSafety.__new__(PROXY_AzureContentSafety)
    seen: Final[list[str]] = []

    async def fake_test_violation(content: str, source: str | None = None) -> None:
        seen.append(content)
        if len(content) > 10000:
            raise RuntimeError("(InvalidRequestBody) The length of given text exceeds the limit 10000.")
        if "flagged" in content:
            raise HTTPException(status_code=400, detail={"error": "Violated content safety policy"})

    guard.test_violation = fake_test_violation
    with pytest.raises(HTTPException):
        await guard.async_pre_call_hook(
            user_api_key_dict=user_api_key,
            cache=DualCache(),
            data=data,
            call_type=call_type,
        )
    assert seen == ["flagged text"]


@pytest.mark.asyncio
async def test_azure_content_safety_post_call_checks_all_choices(user_api_key):
    """Krrish blocker: ``n>1`` responses must not bypass Azure Content Safety
    by placing the unsafe text in ``choices[1+]``."""
    from fastapi import HTTPException
    from litellm.proxy.hooks.azure_content_safety import PROXY_AzureContentSafety

    guard = PROXY_AzureContentSafety.__new__(PROXY_AzureContentSafety)
    seen_outputs = []

    async def fake_test_violation(content, source=None):
        seen_outputs.append((content, source))
        if "unsafe" in content:
            raise HTTPException(status_code=400, detail={"error": "unsafe"})

    guard.test_violation = fake_test_violation
    response = ModelResponse(
        choices=[
            Choices(index=0, message=Message(role="assistant", content="clean")),
            Choices(index=1, message=Message(role="assistant", content="unsafe")),
            Choices(index=2, message=Message(role="assistant", content="later")),
        ]
    )

    with pytest.raises(HTTPException):
        await guard.async_post_call_success_hook(
            data={},
            user_api_key_dict=user_api_key,
            response=response,
        )

    assert seen_outputs == [("clean", "output"), ("unsafe", "output")]


# ── Secret Detection ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_secret_detection_redacts_multimodal_text_parts(user_api_key):
    from litellm_enterprise.enterprise_callbacks.secret_detection import (
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
    from litellm_enterprise.enterprise_callbacks.secret_detection import (
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
        ENTERPRISE_OpenAI_Moderation,
    )

    guard = ENTERPRISE_OpenAI_Moderation()

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


_MODERATION_FLAG: Final = "I am going to kill you tonight."
_MODERATION_LIMIT: Final = 24_000
_LONG_BENIGN: Final = ("Follow the repository conventions and write tests. " * 600)[:30_000]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data", "added"),
    [
        ({"instructions": _LONG_BENIGN, "input": _MODERATION_FLAG}, _LONG_BENIGN),
        ({"system": _LONG_BENIGN, "messages": [{"role": "user", "content": _MODERATION_FLAG}]}, _LONG_BENIGN),
        (
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "write_file", "arguments": json.dumps({"content": _LONG_BENIGN})},
                            }
                        ],
                    },
                    {"role": "user", "content": _MODERATION_FLAG},
                ]
            },
            _LONG_BENIGN,
        ),
    ],
    ids=("responses-instructions", "messages-system", "chat-tool-arguments"),
)
async def test_openai_moderation_sends_tool_and_system_text_in_its_own_call(
    monkeypatch: pytest.MonkeyPatch, user_api_key: UserAPIKeyAuth, data: dict[str, JsonValue], added: str
) -> None:
    from enterprise.enterprise_hooks.openai_moderation import ENTERPRISE_OpenAI_Moderation

    seen: Final[list[str]] = []

    async def amoderation(model: str, input: str) -> SimpleNamespace:
        seen.append(input)
        flagged: Final = _MODERATION_FLAG in input and len(input) <= _MODERATION_LIMIT
        return SimpleNamespace(results=[SimpleNamespace(flagged=flagged)])

    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", SimpleNamespace(amoderation=amoderation))
    with pytest.raises(HTTPException) as rejection:
        await ENTERPRISE_OpenAI_Moderation().async_moderation_hook(data, user_api_key, "acompletion")
    assert rejection.value.status_code == 403
    assert seen == [_MODERATION_FLAG]

    seen.clear()
    clean: Final = json.loads(json.dumps(data).replace(_MODERATION_FLAG, "say hi"))
    assert await ENTERPRISE_OpenAI_Moderation().async_moderation_hook(clean, user_api_key, "acompletion") is None
    assert seen == ["say hi", added]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data", "inputs"),
    [
        ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}, [""]),
        (
            {
                "system": "be brief",
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
            },
            ["be brief"],
        ),
    ],
    ids=("no-text", "system-only"),
)
async def test_openai_moderation_makes_one_call_when_one_side_has_no_text(
    monkeypatch: pytest.MonkeyPatch, user_api_key: UserAPIKeyAuth, data: dict[str, JsonValue], inputs: list[str]
) -> None:
    from enterprise.enterprise_hooks.openai_moderation import ENTERPRISE_OpenAI_Moderation

    seen: Final[list[str]] = []

    async def amoderation(model: str, input: str) -> SimpleNamespace:
        seen.append(input)
        return SimpleNamespace(results=[SimpleNamespace(flagged=False)])

    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", SimpleNamespace(amoderation=amoderation))
    assert await ENTERPRISE_OpenAI_Moderation().async_moderation_hook(data, user_api_key, "acompletion") is None
    assert seen == inputs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_after_init", "expected_model"),
    [("omni-moderation-2024-09-26", "omni-moderation-2024-09-26"), (None, DEFAULT_OPENAI_MODERATIONS_MODEL)],
)
async def test_openai_moderation_reads_model_name_at_call_time(
    monkeypatch, user_api_key, configured_after_init, expected_model
):
    """``litellm_settings`` applies ``callbacks`` and ``openai_moderations_model_name`` in YAML
    order, so the hook must resolve the model when it runs, not when it is constructed."""
    from enterprise.enterprise_hooks.openai_moderation import (
        ENTERPRISE_OpenAI_Moderation,
    )

    monkeypatch.setattr(litellm, "openai_moderations_model_name", None)
    guard = ENTERPRISE_OpenAI_Moderation()
    monkeypatch.setattr(litellm, "openai_moderations_model_name", configured_after_init)

    class FakeModeration:
        results = [type("R", (), {"flagged": False})()]

    fake_router = MagicMock()
    fake_router.amoderation = AsyncMock(return_value=FakeModeration())
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", fake_router, raising=False)

    await guard.async_moderation_hook(
        data={"messages": [{"role": "user", "content": "hello"}]},
        user_api_key_dict=user_api_key,
        call_type="acompletion",
    )

    fake_router.amoderation.assert_awaited_once_with(model=expected_model, input="hello")


@pytest.mark.asyncio
async def test_openai_moderation_error_raising(monkeypatch, respx_mock):
    from enterprise.enterprise_hooks.openai_moderation import (
        ENTERPRISE_OpenAI_Moderation,
    )
    from litellm.proxy.utils import hash_token

    monkeypatch.setattr(litellm, "openai_moderations_model_name", "omni-moderation-latest")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    openai_mod: Final = ENTERPRISE_OpenAI_Moderation()
    user_api_key_dict: Final = UserAPIKeyAuth(api_key=hash_token("sk-98765"))

    llm_router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "omni-moderation-latest",
                "litellm_params": {
                    "model": "omni-moderation-latest",
                    "api_key": "fake-key",
                },
            }
        ]
    )

    moderation_route: Final = respx_mock.post("https://api.openai.com/v1/moderations").respond(
        200,
        json={
            "id": "modr-123",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "flagged": True,
                    "categories": {"harassment": True},
                    "category_scores": {"harassment": 0.97},
                }
            ],
        },
    )

    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "llm_router", llm_router)

    with pytest.raises(Exception, match="Violated content safety policy") as exc_info:
        await openai_mod.async_moderation_hook(
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": "fuck off you're the worst",
                    }
                ]
            },
            user_api_key_dict=user_api_key_dict,
            call_type="completion",
        )
    assert "Violated content safety policy" in str(exc_info.value)
    assert moderation_route.call_count == 1


# ── Google Text Moderation ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_google_text_moderation_inspects_multimodal_content(user_api_key):
    """The text passed to Google's moderation client must include list-format
    text parts."""
    from enterprise.enterprise_hooks.google_text_moderation import (
        ENTERPRISE_GoogleTextModeration,
    )

    guard = ENTERPRISE_GoogleTextModeration.__new__(ENTERPRISE_GoogleTextModeration)
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


class _GoogleDocument(NamedTuple):
    content: str
    type_: object


class _GoogleRequest(NamedTuple):
    document: _GoogleDocument


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", (False, True), ids=("allow", "block"))
@pytest.mark.parametrize(
    ("data", "documents"),
    (
        pytest.param(
            {"messages": [{"role": "user", "content": "ordinary blocked fragment"}]},
            ("ordinary blocked fragment",),
            id="ordinary-text-control",
        ),
        pytest.param(
            {
                "system": [{"type": "text", "text": "system blocked fragment"}],
                "messages": [{"role": "user", "content": "ordinary control"}],
            },
            ("ordinary control", "system blocked fragment"),
            id="native-system",
        ),
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "ordinary control"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "ignored-id",
                                "name": "ignored-name",
                                "input": {
                                    "first": "input first ",
                                    "nested": [7, False, None, {"value": "input blocked fragment"}],
                                },
                            }
                        ],
                    },
                ]
            },
            ("ordinary control", "input first \ninput blocked fragment"),
            id="native-tool-input",
        ),
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "ordinary control"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "ignored-id",
                                "content": [
                                    {"type": "text", "text": "result first "},
                                    {"type": "image", "source": {"data": "ignored-image"}},
                                    {"type": "text", "text": "result blocked fragment"},
                                ],
                            }
                        ],
                    },
                ]
            },
            ("ordinary control", "result first result blocked fragment"),
            id="native-tool-result",
        ),
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "ordinary control"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "ignored-id",
                                "type": "function",
                                "function": {
                                    "name": "ignored-name",
                                    "arguments": json.dumps(
                                        {
                                            "first": "argument first ",
                                            "nested": [7, False, None, {"value": "argument blocked fragment"}],
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                ]
            },
            ("ordinary control", "argument first \nargument blocked fragment"),
            id="chat-tool-arguments",
        ),
    ),
)
async def test_google_text_moderation_inspects_tool_and_system_payloads(
    user_api_key: UserAPIKeyAuth,
    data: Mapping[str, JsonValue],
    documents: tuple[str, ...],
    blocked: bool,
) -> None:
    sdk: Final = MagicMock()
    sdk.types.Document.side_effect = _GoogleDocument
    sdk.ModerateTextRequest.side_effect = _GoogleRequest

    def moderate(*, request: _GoogleRequest) -> SimpleNamespace:
        flagged: Final = blocked and "blocked fragment" in request.document.content
        return SimpleNamespace(moderation_categories=[SimpleNamespace(name="Toxic", confidence=float(flagged))])

    client: Final = sdk.LanguageServiceClient.return_value
    client.moderate_text.side_effect = moderate
    cloud: Final = SimpleNamespace(language_v1=sdk)
    request_data: Final = dict(data)
    with patch.dict(sys.modules, {"google.cloud": cloud, "google.cloud.language_v1": sdk}):
        guard: Final = ENTERPRISE_GoogleTextModeration()
        if blocked:
            with pytest.raises(HTTPException, match="Violated content safety policy") as rejection:
                await guard.async_moderation_hook(request_data, user_api_key, "acompletion")
            assert rejection.value.status_code == 400
        else:
            assert await guard.async_moderation_hook(request_data, user_api_key, "acompletion") is request_data
    sdk.LanguageServiceClient.assert_called_once_with()
    plain_text: Final = sdk.types.Document.Type.PLAIN_TEXT
    assert sdk.types.Document.call_args_list == [call(content=text, type_=plain_text) for text in documents]
    assert client.moderate_text.call_args_list == [
        call(request=_GoogleRequest(_GoogleDocument(text, plain_text))) for text in documents
    ]
