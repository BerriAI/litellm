"""
Tests for web search model redirect in common_processing_pre_call_logic.

Covers per-deployment force_websearch_model and global websearch_fallback_model.
The redirect logic lives inline in common_processing_pre_call_logic, so tests
construct a ProxyBaseLLMRequestProcessing instance and call the method with
mocked dependencies, then verify self.data["model"] after the redirect block.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}
REGULAR_TOOL = {
    "type": "custom",
    "name": "get_weather",
    "description": "Get weather",
    "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
}


def _make_mock_router(deployments):
    """Build a mock router that returns the given deployments for get_model_list."""
    router = MagicMock()
    router.get_model_list.return_value = deployments
    router.get_model_group_info.return_value = None
    return router


def _make_mock_request():
    request = MagicMock()
    request.headers = {}
    request.url = MagicMock()
    request.url.path = "/v1/messages"
    return request


def _make_user_api_key_dict():
    user_api_key_dict = MagicMock()
    user_api_key_dict.aliases = {}
    user_api_key_dict.models = []
    user_api_key_dict.api_key = "sk-test"
    user_api_key_dict.user_id = "test"
    user_api_key_dict.team_id = None
    user_api_key_dict.metadata = {}
    return user_api_key_dict


async def _run_pre_call_until_redirect(data, llm_router):
    """Run common_processing_pre_call_logic far enough to trigger the redirect,
    then let it fail on subsequent steps — we only care about data['model']."""
    proc = ProxyBaseLLMRequestProcessing(data=data)

    async def _passthrough_add_litellm_data(data, **kwargs):
        return data

    with patch(
        "litellm.proxy.common_request_processing.add_litellm_data_to_request",
        side_effect=_passthrough_add_litellm_data,
    ):
        try:
            await proc.common_processing_pre_call_logic(
                request=_make_mock_request(),
                user_api_key_dict=_make_user_api_key_dict(),
                llm_router=llm_router,
                proxy_config=MagicMock(),
                general_settings={},
                proxy_logging_obj=MagicMock(),
                route_type="anthropic_messages",
                version=None,
            )
        except Exception:
            pass
    return proc.data.get("model")


class TestForceWebsearchModel:
    @pytest.mark.asyncio
    async def test_pure_websearch_redirected(self):
        router = _make_mock_router(
            [
                {
                    "litellm_params": {
                        "model": "openai/gpt-5.4",
                        "force_websearch_model": "gpt-5.5",
                    }
                }
            ]
        )
        data = {
            "model": "my-model",
            "messages": [{"role": "user", "content": "search"}],
            "tools": [WEB_SEARCH_TOOL],
        }
        result = await _run_pre_call_until_redirect(data, router)
        assert result == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_mixed_tools_not_redirected(self):
        router = _make_mock_router(
            [
                {
                    "litellm_params": {
                        "model": "openai/gpt-5.4",
                        "force_websearch_model": "gpt-5.5",
                    }
                }
            ]
        )
        data = {
            "model": "my-model",
            "messages": [{"role": "user", "content": "search"}],
            "tools": [WEB_SEARCH_TOOL, REGULAR_TOOL],
        }
        result = await _run_pre_call_until_redirect(data, router)
        assert result == "my-model"

    @pytest.mark.asyncio
    async def test_multiple_websearch_tools_redirected(self):
        router = _make_mock_router(
            [
                {
                    "litellm_params": {
                        "model": "openai/gpt-5.4",
                        "force_websearch_model": "gpt-5.5",
                    }
                }
            ]
        )
        data = {
            "model": "my-model",
            "messages": [{"role": "user", "content": "search"}],
            "tools": [
                WEB_SEARCH_TOOL,
                {"name": "WebSearch", "description": "search"},
            ],
        }
        result = await _run_pre_call_until_redirect(data, router)
        assert result == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_no_force_no_redirect(self):
        router = _make_mock_router([{"litellm_params": {"model": "openai/gpt-5.5"}}])
        data = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "search"}],
            "tools": [WEB_SEARCH_TOOL],
        }
        with patch("litellm.supports_web_search", return_value=True):
            result = await _run_pre_call_until_redirect(data, router)
        assert result == "gpt-5.5"


class TestWebsearchFallbackModel:
    @pytest.mark.asyncio
    async def test_fallback_when_model_lacks_search(self):
        router = _make_mock_router(
            [{"litellm_params": {"model": "openai/my-local-llm"}}]
        )
        data = {
            "model": "my-local-llm",
            "messages": [{"role": "user", "content": "search"}],
            "tools": [WEB_SEARCH_TOOL],
        }
        with (
            patch("litellm.supports_web_search", return_value=False),
            patch.object(
                litellm,
                "websearch_fallback_model",
                "gpt-5.4-mini",
                create=True,
            ),
        ):
            result = await _run_pre_call_until_redirect(data, router)
        assert result == "gpt-5.4-mini"

    @pytest.mark.asyncio
    async def test_no_fallback_when_model_supports_search(self):
        router = _make_mock_router([{"litellm_params": {"model": "openai/gpt-5.5"}}])
        data = {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "search"}],
            "tools": [WEB_SEARCH_TOOL],
        }
        with patch("litellm.supports_web_search", return_value=True):
            result = await _run_pre_call_until_redirect(data, router)
        assert result == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_no_fallback_when_setting_not_configured(self):
        router = _make_mock_router(
            [{"litellm_params": {"model": "openai/my-local-llm"}}]
        )
        data = {
            "model": "my-local-llm",
            "messages": [{"role": "user", "content": "search"}],
            "tools": [WEB_SEARCH_TOOL],
        }
        with (
            patch("litellm.supports_web_search", return_value=False),
            patch.object(litellm, "websearch_fallback_model", None, create=True),
        ):
            result = await _run_pre_call_until_redirect(data, router)
        assert result == "my-local-llm"


class TestWebsearchForceOverFallbackPriority:
    @pytest.mark.asyncio
    async def test_force_takes_priority_over_fallback(self):
        router = _make_mock_router(
            [
                {
                    "litellm_params": {
                        "model": "openai/gpt-5.4",
                        "force_websearch_model": "gpt-5.5",
                    }
                }
            ]
        )
        data = {
            "model": "my-model",
            "messages": [{"role": "user", "content": "search"}],
            "tools": [WEB_SEARCH_TOOL],
        }
        with (
            patch("litellm.supports_web_search", return_value=False),
            patch.object(
                litellm,
                "websearch_fallback_model",
                "gpt-5.4-mini",
                create=True,
            ),
        ):
            result = await _run_pre_call_until_redirect(data, router)
        assert result == "gpt-5.5"


class TestWebsearchRedirectGuardConditions:
    @pytest.mark.asyncio
    async def test_no_tools(self):
        router = _make_mock_router([])
        data = {
            "model": "my-model",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = await _run_pre_call_until_redirect(data, router)
        assert result == "my-model"

    @pytest.mark.asyncio
    async def test_no_router(self):
        data = {
            "model": "my-model",
            "messages": [{"role": "user", "content": "search"}],
            "tools": [WEB_SEARCH_TOOL],
        }
        result = await _run_pre_call_until_redirect(data, None)
        assert result == "my-model"

    @pytest.mark.asyncio
    async def test_empty_tools(self):
        router = _make_mock_router([])
        data = {
            "model": "my-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
        }
        result = await _run_pre_call_until_redirect(data, router)
        assert result == "my-model"
