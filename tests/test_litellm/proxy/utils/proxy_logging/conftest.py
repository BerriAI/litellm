"""Shared fixtures for tests/test_litellm/proxy/utils/proxy_logging/.

All fixtures used by PR1 of the proxy/utils.py behavior-pinning project
live here. Tests should not declare fixtures inline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


VOLATILE_KEYS = frozenset(
    {
        "created_at",
        "updated_at",
        "id",
        "request_id",
        "token",
        "expires",
        "expires_at",
        "litellm_call_id",
        "key_alias",
        "created",
        "start_time",
        "end_time",
        "duration",
        "guardrail_start_time",
        "guardrail_end_time",
        "guardrail_duration",
    }
)


def normalize(data: Any, volatile: frozenset = VOLATILE_KEYS) -> Any:
    if isinstance(data, dict):
        return {
            k: ("<VOLATILE>" if k in volatile else normalize(v, volatile))
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [normalize(v, volatile) for v in data]
    return data


@pytest.fixture
def mock_dual_cache():
    from litellm.caching.caching import DualCache

    cache = DualCache(default_in_memory_ttl=1)
    return cache


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.guardrail_list = []
    router.get_available_guardrail = MagicMock(return_value={"callback": None})
    return router


@pytest.fixture
def mock_callbacks_disabled(monkeypatch):
    """Disable all litellm callbacks for the duration of a test."""
    import litellm

    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    yield


@pytest.fixture
def make_user_api_key_auth():
    from litellm.proxy._types import UserAPIKeyAuth

    def _make(**overrides) -> UserAPIKeyAuth:
        defaults: Dict[str, Any] = {
            "api_key": "sk-test-1234",
            "user_id": "test-user",
            "team_id": "test-team",
            "user_role": None,
            "max_budget": None,
            "spend": 0.0,
        }
        defaults.update(overrides)
        return UserAPIKeyAuth(**defaults)

    return _make


@pytest.fixture
def proxy_logging(mock_callbacks_disabled):
    """A wired-up ProxyLogging instance backed by a fresh DualCache.

    The fixture leaves it un-started; tests that need ``startup_event``
    should call it explicitly with the deps they want to control.
    """
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import ProxyLogging

    return ProxyLogging(user_api_key_cache=UserApiKeyCache())


@pytest.fixture
def normalize_fn():
    return normalize


@pytest.fixture
def make_mcp_request_obj():
    from litellm.types.llms.base import HiddenParams
    from litellm.types.mcp import MCPPreCallRequestObject

    def _make(
        tool_name: str = "calculator",
        arguments: Optional[dict] = None,
        server_name: Optional[str] = "math-server",
    ) -> MCPPreCallRequestObject:
        return MCPPreCallRequestObject(
            tool_name=tool_name,
            arguments=arguments if arguments is not None else {"x": 1, "y": 2},
            server_name=server_name,
            user_api_key_auth={},
            hidden_params=HiddenParams(),
        )

    return _make
