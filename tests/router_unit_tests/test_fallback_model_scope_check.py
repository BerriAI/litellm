"""Regression: server-side fallbacks re-check model authorization.

A key restricted to model-a should not be served by model-b via a
router-configured fallback, regardless of which metadata bucket carries
the auth object (metadata vs litellm_metadata).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.router_utils.fallback_event_handlers import (
    _proxy_key_allows_fallback_model,
)


def _make_kwargs(metadata=None, litellm_metadata=None):
    kwargs = {}
    if metadata is not None:
        kwargs["metadata"] = metadata
    if litellm_metadata is not None:
        kwargs["litellm_metadata"] = litellm_metadata
    return kwargs


def _make_auth(team_models=None, key_models=None):
    auth = MagicMock()
    auth.team_models = team_models or []
    auth.team_model_aliases = None
    auth.team_id = None
    auth.models = key_models or []
    auth.config = None
    return auth


@pytest.mark.asyncio
async def test_no_auth_returns_true():
    """SDK callers (no auth object) are unaffected."""
    kwargs = _make_kwargs(metadata={})
    assert await _proxy_key_allows_fallback_model(
        litellm_router=MagicMock(), kwargs=kwargs, fallback_entry="model-b"
    )


@pytest.mark.asyncio
async def test_metadata_bucket_auth(monkeypatch):
    """Auth resolved from kwargs['metadata'] (the standard proxy path)."""
    from litellm.proxy.auth import auth_checks

    check = AsyncMock(side_effect=auth_checks.ProxyException(message="denied"))
    monkeypatch.setattr(auth_checks, "can_key_call_resolved_model", check)

    kwargs = _make_kwargs(metadata={"user_api_key_auth": _make_auth()})
    result = await _proxy_key_allows_fallback_model(
        litellm_router=MagicMock(), kwargs=kwargs, fallback_entry="model-b"
    )
    assert result is False
    check.assert_called_once()


@pytest.mark.asyncio
async def test_litellm_metadata_bucket_auth(monkeypatch):
    """Auth resolved from kwargs['litellm_metadata'] (the /v1/messages and
    /v1/responses path). Without this, those routes skip the check entirely."""
    from litellm.proxy.auth import auth_checks

    check = AsyncMock(side_effect=auth_checks.ProxyException(message="denied"))
    monkeypatch.setattr(auth_checks, "can_key_call_resolved_model", check)

    kwargs = _make_kwargs(litellm_metadata={"user_api_key_auth": _make_auth()})
    result = await _proxy_key_allows_fallback_model(
        litellm_router=MagicMock(), kwargs=kwargs, fallback_entry="model-b"
    )
    assert result is False
    check.assert_called_once()


@pytest.mark.asyncio
async def test_metadata_takes_precedence_over_litellm_metadata(monkeypatch):
    """When both buckets carry auth, the metadata bucket wins (standard path)."""
    from litellm.proxy.auth import auth_checks

    check = AsyncMock()  # no exception = allowed
    monkeypatch.setattr(auth_checks, "can_key_call_resolved_model", check)

    kwargs = _make_kwargs(
        metadata={"user_api_key_auth": _make_auth()},
        litellm_metadata={"user_api_key_auth": _make_auth()},
    )
    result = await _proxy_key_allows_fallback_model(
        litellm_router=MagicMock(), kwargs=kwargs, fallback_entry="model-b"
    )
    assert result is True
    check.assert_called_once()


@pytest.mark.asyncio
async def test_allowed_model_returns_true(monkeypatch):
    """When the resolved-model check passes, the fallback proceeds."""
    from litellm.proxy.auth import auth_checks

    check = AsyncMock()  # no exception = allowed
    monkeypatch.setattr(auth_checks, "can_key_call_resolved_model", check)

    kwargs = _make_kwargs(metadata={"user_api_key_auth": _make_auth()})
    result = await _proxy_key_allows_fallback_model(
        litellm_router=MagicMock(), kwargs=kwargs, fallback_entry="model-b"
    )
    assert result is True
    check.assert_called_once_with(
        model="model-b", llm_model_list=None, valid_token=kwargs["metadata"]["user_api_key_auth"], llm_router=pytest.ANY
    )
