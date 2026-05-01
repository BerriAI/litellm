import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.peyeeye.peyeeye import (
    PEyeEyeGuardrail,
    PEyeEyeGuardrailAPIError,
    PEyeEyeGuardrailMissingSecrets,
)
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2


def _ok(json_payload: dict):
    resp = MagicMock()
    resp.json.return_value = json_payload
    resp.raise_for_status = MagicMock()
    return resp


def test_peyeeye_init_requires_api_key():
    for var in ("PEYEEYE_API_KEY", "PEYEEYE_API_BASE"):
        os.environ.pop(var, None)
    with pytest.raises(PEyeEyeGuardrailMissingSecrets):
        PEyeEyeGuardrail(guardrail_name="t")


def test_peyeeye_init_reads_env():
    os.environ["PEYEEYE_API_KEY"] = "pk_test"
    try:
        g = PEyeEyeGuardrail(guardrail_name="t")
        assert g.peyeeye_api_key == "pk_test"
        assert g.api_base == "https://api.peyeeye.ai"
        assert g.peyeeye_session_mode == "stateful"
    finally:
        del os.environ["PEYEEYE_API_KEY"]


def test_peyeeye_init_explicit_args():
    g = PEyeEyeGuardrail(
        peyeeye_api_key="pk_x",
        api_base="https://api.example/",
        peyeeye_locale="en",
        peyeeye_entities=["EMAIL"],
        peyeeye_session_mode="stateless",
        guardrail_name="t",
    )
    assert g.peyeeye_api_key == "pk_x"
    assert g.api_base == "https://api.example"
    assert g.peyeeye_locale == "en"
    assert g.peyeeye_entities == ["EMAIL"]
    assert g.peyeeye_session_mode == "stateless"


def test_peyeeye_guardrail_config_via_init():
    litellm.guardrail_name_config_map = {}
    os.environ["PEYEEYE_API_KEY"] = "pk_test"
    try:
        init_guardrails_v2(
            all_guardrails=[
                {
                    "guardrail_name": "peyeeye-pre",
                    "litellm_params": {
                        "guardrail": "peyeeye",
                        "mode": "pre_call",
                        "default_on": True,
                    },
                }
            ],
            config_file_path="",
        )
    finally:
        del os.environ["PEYEEYE_API_KEY"]


@pytest.mark.asyncio
async def test_pre_call_redacts_messages_and_caches_session():
    g = PEyeEyeGuardrail(peyeeye_api_key="pk", guardrail_name="t")
    g.async_handler = MagicMock()
    g.async_handler.post = AsyncMock(
        return_value=_ok(
            {"text": ["hi [EMAIL_1]"], "session_id": "ses_abc"}
        )
    )

    cache = DualCache()
    data = {
        "messages": [{"role": "user", "content": "hi alice@acme.com"}],
        "litellm_call_id": "call-1",
    }
    user = UserAPIKeyAuth(api_key="x")
    out = await g.async_pre_call_hook(user, cache, data, "completion")

    from litellm.proxy.guardrails.guardrail_hooks.peyeeye.peyeeye import (
        global_cache,
    )
    assert out["messages"][0]["content"] == "hi [EMAIL_1]"
    assert global_cache.get_cache("peyeeye_session:call-1") == "ses_abc"
    global_cache.delete_cache("peyeeye_session:call-1")


@pytest.mark.asyncio
async def test_pre_call_stateless_returns_skey():
    g = PEyeEyeGuardrail(
        peyeeye_api_key="pk",
        peyeeye_session_mode="stateless",
        guardrail_name="t",
    )
    g.async_handler = MagicMock()
    g.async_handler.post = AsyncMock(
        return_value=_ok(
            {"text": ["[EMAIL_1]"], "rehydration_key": "skey_xyz"}
        )
    )

    cache = DualCache()
    data = {
        "messages": [{"role": "user", "content": "alice@acme.com"}],
        "litellm_call_id": "call-2",
    }
    await g.async_pre_call_hook(UserAPIKeyAuth(api_key="x"), cache, data, "completion")

    from litellm.proxy.guardrails.guardrail_hooks.peyeeye.peyeeye import (
        global_cache,
    )
    sent_body = g.async_handler.post.call_args.kwargs["json"]
    assert sent_body["session"] == "stateless"
    assert global_cache.get_cache("peyeeye_session:call-2") == "skey_xyz"
    global_cache.delete_cache("peyeeye_session:call-2")


@pytest.mark.asyncio
async def test_pre_call_skips_when_no_messages():
    g = PEyeEyeGuardrail(peyeeye_api_key="pk", guardrail_name="t")
    g.async_handler = MagicMock()
    g.async_handler.post = AsyncMock()

    cache = DualCache()
    data = {"messages": [], "litellm_call_id": "x"}
    out = await g.async_pre_call_hook(
        UserAPIKeyAuth(api_key="x"), cache, data, "completion"
    )
    assert out is data
    g.async_handler.post.assert_not_called()


@pytest.mark.asyncio
async def test_pre_and_post_call_roundtrip_uses_shared_cache():
    """End-to-end: pre-call seeds session id, post-call retrieves & rehydrates."""
    from litellm.proxy.guardrails.guardrail_hooks.peyeeye.peyeeye import (
        global_cache,
    )

    g = PEyeEyeGuardrail(peyeeye_api_key="pk", guardrail_name="t")
    g.async_handler = MagicMock()
    g.async_handler.post = AsyncMock(
        side_effect=[
            _ok({"text": ["hi [EMAIL_1]"], "session_id": "ses_abc"}),
            _ok({"text": "Reply to alice@acme.com", "replaced": 1}),
        ]
    )
    g.async_handler.delete = AsyncMock()

    cache = DualCache()
    data = {
        "messages": [{"role": "user", "content": "hi alice@acme.com"}],
        "litellm_call_id": "rt-1",
    }
    user = UserAPIKeyAuth(api_key="x")
    await g.async_pre_call_hook(user, cache, data, "completion")

    response = litellm.ModelResponse()
    response.choices = [
        litellm.utils.Choices(
            finish_reason="stop",
            index=0,
            message=litellm.utils.Message(
                content="Reply to [EMAIL_1]", role="assistant"
            ),
        )
    ]
    out = await g.async_post_call_success_hook(
        {"litellm_call_id": "rt-1"}, user, response
    )
    assert out.choices[0].message.content == "Reply to alice@acme.com"
    g.async_handler.delete.assert_awaited()
    assert global_cache.get_cache("peyeeye_session:rt-1") is None


@pytest.mark.asyncio
async def test_post_call_noop_without_session():
    g = PEyeEyeGuardrail(peyeeye_api_key="pk", guardrail_name="t")
    g.async_handler = MagicMock()
    g.async_handler.post = AsyncMock()

    response = litellm.ModelResponse()
    response.choices = [
        litellm.utils.Choices(
            finish_reason="stop",
            index=0,
            message=litellm.utils.Message(content="hello", role="assistant"),
        )
    ]
    out = await g.async_post_call_success_hook(
        {"litellm_call_id": "no-session"}, UserAPIKeyAuth(api_key="x"), response
    )
    assert out.choices[0].message.content == "hello"
    g.async_handler.post.assert_not_called()


@pytest.mark.asyncio
async def test_redact_api_error_raises_typed():
    g = PEyeEyeGuardrail(peyeeye_api_key="pk", guardrail_name="t")
    g.async_handler = MagicMock()

    bad = MagicMock()
    bad.raise_for_status.side_effect = RuntimeError("boom")
    g.async_handler.post = AsyncMock(return_value=bad)

    cache = DualCache()
    with pytest.raises(PEyeEyeGuardrailAPIError):
        await g.async_pre_call_hook(
            UserAPIKeyAuth(api_key="x"),
            cache,
            {
                "messages": [{"role": "user", "content": "hi"}],
                "litellm_call_id": "c",
            },
            "completion",
        )


@pytest.mark.asyncio
async def test_redact_json_decode_error_raises_typed():
    """If response.json() blows up, surface a typed PEyeEyeGuardrailAPIError."""
    g = PEyeEyeGuardrail(peyeeye_api_key="pk", guardrail_name="t")
    g.async_handler = MagicMock()

    bad = MagicMock()
    bad.raise_for_status = MagicMock()
    bad.json.side_effect = ValueError("not json")
    g.async_handler.post = AsyncMock(return_value=bad)

    cache = DualCache()
    with pytest.raises(PEyeEyeGuardrailAPIError):
        await g.async_pre_call_hook(
            UserAPIKeyAuth(api_key="x"),
            cache,
            {
                "messages": [{"role": "user", "content": "hi"}],
                "litellm_call_id": "json-err",
            },
            "completion",
        )


@pytest.mark.asyncio
async def test_pre_call_raises_on_length_mismatch():
    """If /v1/redact returns fewer texts than sent, refuse to forward."""
    g = PEyeEyeGuardrail(peyeeye_api_key="pk", guardrail_name="t")
    g.async_handler = MagicMock()
    g.async_handler.post = AsyncMock(
        return_value=_ok({"text": ["[EMAIL_1]"], "session_id": "ses_x"})
    )

    cache = DualCache()
    data = {
        "messages": [
            {"role": "user", "content": "hi alice@acme.com"},
            {"role": "user", "content": "hi bob@acme.com"},
        ],
        "litellm_call_id": "len-1",
    }
    with pytest.raises(PEyeEyeGuardrailAPIError, match="partially-redacted"):
        await g.async_pre_call_hook(
            UserAPIKeyAuth(api_key="x"), cache, data, "completion"
        )


@pytest.mark.asyncio
async def test_pre_call_raises_on_unexpected_response_shape():
    """If /v1/redact returns neither str nor list for `text`, refuse to forward."""
    g = PEyeEyeGuardrail(peyeeye_api_key="pk", guardrail_name="t")
    g.async_handler = MagicMock()
    g.async_handler.post = AsyncMock(
        return_value=_ok({"text": 42, "session_id": "ses_x"})
    )

    cache = DualCache()
    data = {
        "messages": [{"role": "user", "content": "hi alice@acme.com"}],
        "litellm_call_id": "shape-1",
    }
    with pytest.raises(PEyeEyeGuardrailAPIError, match="unexpected response shape"):
        await g.async_pre_call_hook(
            UserAPIKeyAuth(api_key="x"), cache, data, "completion"
        )
