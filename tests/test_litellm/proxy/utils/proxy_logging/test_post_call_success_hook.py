"""Pin ``ProxyLogging.post_call_success_hook``."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ProxyLogging._callback_capabilities_cache.clear()
    yield
    ProxyLogging._callback_capabilities_cache.clear()


def _make_guardrail(name="g", should_run=True, override=None):
    cb = MagicMock(spec=CustomGuardrail)
    cb.__class__ = CustomGuardrail
    cb.guardrail_name = name
    cb.event_hook = GuardrailEventHooks.post_call
    cb.should_run_guardrail = MagicMock(return_value=should_run)
    cb.async_post_call_success_hook = AsyncMock(return_value=override)
    cb.run_in_parallel = False
    return cb


@pytest.mark.asyncio
async def test_post_call_success_hook_returns_response_when_no_callbacks(proxy_logging, make_user_api_key_auth, mock_callbacks_disabled):
    response = {"original": True, "model": "m", "choices": []}
    out = await proxy_logging.post_call_success_hook(
        data={}, response=response, user_api_key_dict=make_user_api_key_auth()
    )
    assert out == {"original": True, "model": "m", "choices": []}


@pytest.mark.asyncio
async def test_post_call_success_hook_runs_other_callback_and_replaces_response(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    new_response = {"modified": True, "kept": "yes", "final": "v"}

    class _CL(CustomLogger):
        async def async_post_call_success_hook(self, **kwargs):  # type: ignore[override]
            return new_response

    monkeypatch.setattr(litellm, "callbacks", [_CL()])
    out = await proxy_logging.post_call_success_hook(
        data={}, response={"original": True}, user_api_key_dict=make_user_api_key_auth()
    )
    assert out == new_response


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_should_not_run_skipped(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    g = _make_guardrail(should_run=False)
    monkeypatch.setattr(litellm, "callbacks", [g])
    response = MagicMock()
    out = await proxy_logging.post_call_success_hook(
        data={}, response=response, user_api_key_dict=make_user_api_key_auth()
    )
    g.async_post_call_success_hook.assert_not_called()
    assert out is response


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_error_raises(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    g = _make_guardrail()
    g.async_post_call_success_hook = AsyncMock(side_effect=RuntimeError("blocked"))
    monkeypatch.setattr(litellm, "callbacks", [g])
    with pytest.raises(RuntimeError):
        await proxy_logging.post_call_success_hook(
            data={}, response=MagicMock(), user_api_key_dict=make_user_api_key_auth()
        )


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_returns_modified_response(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    modified = {"a": 1, "b": 2, "c": 3}
    g = _make_guardrail(override=modified)
    monkeypatch.setattr(litellm, "callbacks", [g])
    out = await proxy_logging.post_call_success_hook(
        data={}, response={"orig": True}, user_api_key_dict=make_user_api_key_auth()
    )
    assert out == modified


# ---------------------------------------------------------------------------
# hook_filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_skipped_when_hook_filters_model_excludes(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    from litellm.litellm_core_utils.hook_filter_utils import parse_hook_filters

    monkeypatch.setattr(litellm, "enable_hook_filters", True)
    g = _make_guardrail()
    g.hook_filters = parse_hook_filters("g", {"async_post_call_success_hook": {"models": ["claude-*"]}})
    monkeypatch.setattr(litellm, "callbacks", [g])

    await proxy_logging.post_call_success_hook(
        data={"model": "gpt-4o"}, response=MagicMock(), user_api_key_dict=make_user_api_key_auth()
    )
    g.async_post_call_success_hook.assert_not_called()


@pytest.mark.asyncio
async def test_post_call_success_hook_guardrail_runs_when_hook_filters_model_matches(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    from litellm.litellm_core_utils.hook_filter_utils import parse_hook_filters

    monkeypatch.setattr(litellm, "enable_hook_filters", True)
    g = _make_guardrail()
    g.hook_filters = parse_hook_filters("g", {"async_post_call_success_hook": {"models": ["gpt-4o*"]}})
    monkeypatch.setattr(litellm, "callbacks", [g])

    await proxy_logging.post_call_success_hook(
        data={"model": "gpt-4o"}, response=MagicMock(), user_api_key_dict=make_user_api_key_auth()
    )
    g.async_post_call_success_hook.assert_called_once()


@pytest.mark.asyncio
async def test_post_call_success_hook_plain_callback_skipped_when_hook_filters_model_excludes(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    from litellm.litellm_core_utils.hook_filter_utils import parse_hook_filters

    monkeypatch.setattr(litellm, "enable_hook_filters", True)
    calls = []

    class _CL(CustomLogger):
        async def async_post_call_success_hook(self, **kwargs):  # type: ignore[override]
            calls.append(kwargs)
            return None

    cb = _CL()
    cb.hook_filters = parse_hook_filters("cl", {"async_post_call_success_hook": {"models": ["claude-*"]}})
    monkeypatch.setattr(litellm, "callbacks", [cb])

    await proxy_logging.post_call_success_hook(
        data={"model": "gpt-4o"}, response={"original": True}, user_api_key_dict=make_user_api_key_auth()
    )
    assert calls == []


@pytest.mark.asyncio
async def test_post_call_success_hook_plain_callback_runs_when_hook_filters_model_matches(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    from litellm.litellm_core_utils.hook_filter_utils import parse_hook_filters

    monkeypatch.setattr(litellm, "enable_hook_filters", True)
    calls = []

    class _CL(CustomLogger):
        async def async_post_call_success_hook(self, **kwargs):  # type: ignore[override]
            calls.append(kwargs)
            return None

    cb = _CL()
    cb.hook_filters = parse_hook_filters("cl", {"async_post_call_success_hook": {"models": ["gpt-4o*"]}})
    monkeypatch.setattr(litellm, "callbacks", [cb])

    await proxy_logging.post_call_success_hook(
        data={"model": "gpt-4o"}, response={"original": True}, user_api_key_dict=make_user_api_key_auth()
    )
    assert len(calls) == 1
