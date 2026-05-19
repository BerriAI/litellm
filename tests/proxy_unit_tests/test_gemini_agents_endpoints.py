"""
Unit tests for litellm/proxy/google_endpoints/agents_endpoints.py

Focus: verify that list_gemini_agents, get_gemini_agent, delete_gemini_agent,
and list_gemini_agent_versions correctly forward per-request credentials
(api_key, api_base, …) supplied via query parameters.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.datastructures import Headers, QueryParams

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.google_endpoints.agents_endpoints import (
    _merge_query_params_into_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(query_string: str = "") -> MagicMock:
    """Build a minimal mock Request whose query_params match *query_string*."""
    req = MagicMock(spec=Request)
    req.query_params = QueryParams(query_string)
    req.headers = Headers({})
    return req


# ---------------------------------------------------------------------------
# _merge_query_params_into_data – unit tests for the helper
# ---------------------------------------------------------------------------


class TestMergeQueryParamsIntoData:
    def test_no_query_params_leaves_data_unchanged(self):
        data = {"custom_llm_provider": "gemini"}
        request = _make_request("")
        result = _merge_query_params_into_data(data, request)
        assert result == {"custom_llm_provider": "gemini"}

    def test_flat_api_key_is_merged(self):
        data = {"custom_llm_provider": "gemini"}
        request = _make_request("api_key=AIzaSyTest123")
        _merge_query_params_into_data(data, request)
        assert data["api_key"] == "AIzaSyTest123"
        assert data["custom_llm_provider"] == "gemini"

    def test_flat_params_do_not_overwrite_existing_keys(self):
        """Path params like 'name' must never be clobbered by query params."""
        data = {"name": "my-agent", "custom_llm_provider": "gemini"}
        request = _make_request("name=INJECTED&api_key=AIzaSyTest")
        _merge_query_params_into_data(data, request)
        assert data["name"] == "my-agent"
        assert data["api_key"] == "AIzaSyTest"

    def test_litellm_params_template_json_is_expanded(self):
        template = json.dumps(
            {"api_key": "AIzaFromTemplate", "api_base": "https://example.com"}
        )
        from urllib.parse import quote

        request = _make_request(f"litellm_params_template={quote(template)}")
        data = {"custom_llm_provider": "gemini"}
        _merge_query_params_into_data(data, request)
        assert data["api_key"] == "AIzaFromTemplate"
        assert data["api_base"] == "https://example.com"
        # The raw template key itself must NOT appear in data
        assert "litellm_params_template" not in data

    def test_litellm_params_template_does_not_overwrite_existing(self):
        template = json.dumps(
            {"api_key": "FromTemplate", "custom_llm_provider": "openai"}
        )
        from urllib.parse import quote

        request = _make_request(f"litellm_params_template={quote(template)}")
        data = {"custom_llm_provider": "gemini"}
        _merge_query_params_into_data(data, request)
        # custom_llm_provider was already set; template must not override it
        assert data["custom_llm_provider"] == "gemini"
        assert data["api_key"] == "FromTemplate"

    def test_invalid_litellm_params_template_json_is_ignored(self):
        request = _make_request("litellm_params_template=NOT_VALID_JSON")
        data = {"custom_llm_provider": "gemini"}
        _merge_query_params_into_data(data, request)
        # Bad JSON is silently skipped; other data stays intact
        assert data == {"custom_llm_provider": "gemini"}

    def test_flat_and_template_combined(self):
        """Template and a flat param can coexist."""
        template = json.dumps({"api_key": "FromTemplate"})
        from urllib.parse import quote

        qs = f"litellm_params_template={quote(template)}&vertex_project=my-project"
        request = _make_request(qs)
        data = {"custom_llm_provider": "gemini"}
        _merge_query_params_into_data(data, request)
        assert data["api_key"] == "FromTemplate"
        assert data["vertex_project"] == "my-project"
        assert "litellm_params_template" not in data


# ---------------------------------------------------------------------------
# Endpoint-level smoke tests: data dict is populated before the processor call
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_srv():
    """Patch _proxy_server_imports to return lightweight fakes."""
    srv = {
        "general_settings": {},
        "llm_router": MagicMock(),
        "proxy_config": MagicMock(),
        "proxy_logging_obj": MagicMock(),
        "select_data_generator": MagicMock(),
        "user_api_base": None,
        "user_max_tokens": None,
        "user_model": None,
        "user_request_timeout": None,
        "user_temperature": None,
        "version": "0.0.0",
    }
    with patch(
        "litellm.proxy.google_endpoints.agents_endpoints._proxy_server_imports",
        return_value=srv,
    ):
        yield srv


@pytest.fixture
def user_api_key_dict():
    from litellm.proxy._types import UserAPIKeyAuth

    return UserAPIKeyAuth(api_key="test-key")


def _make_endpoint_request(query_string: str = "") -> MagicMock:
    req = MagicMock(spec=Request)
    req.query_params = QueryParams(query_string)
    req.headers = Headers({})
    req.scope = {}

    async def _body():
        return b""

    req.body = _body
    return req


@pytest.mark.asyncio
async def test_list_gemini_agents_passes_api_key_to_processor(
    mock_srv, user_api_key_dict
):
    from litellm.proxy.google_endpoints.agents_endpoints import list_gemini_agents

    captured: dict = {}

    async def fake_process(**kwargs):
        captured.update({"data": kwargs.get("data") or {}})
        return MagicMock()

    with patch(
        "litellm.proxy.google_endpoints.agents_endpoints.ProxyBaseLLMRequestProcessing"
    ) as MockProcessor:
        instance = MockProcessor.return_value
        instance.base_process_llm_request = AsyncMock(side_effect=fake_process)

        request = _make_endpoint_request("api_key=AIzaListTest")
        await list_gemini_agents(
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

        init_data = MockProcessor.call_args[1]["data"]
        assert init_data.get("api_key") == "AIzaListTest"
        assert init_data.get("custom_llm_provider") == "gemini"


@pytest.mark.asyncio
async def test_get_gemini_agent_passes_api_key_to_processor(
    mock_srv, user_api_key_dict
):
    from litellm.proxy.google_endpoints.agents_endpoints import get_gemini_agent

    with patch(
        "litellm.proxy.google_endpoints.agents_endpoints.ProxyBaseLLMRequestProcessing"
    ) as MockProcessor:
        instance = MockProcessor.return_value
        instance.base_process_llm_request = AsyncMock(return_value=MagicMock())

        request = _make_endpoint_request("api_key=AIzaGetTest")
        await get_gemini_agent(
            request=request,
            name="my-agent",
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

        init_data = MockProcessor.call_args[1]["data"]
        assert init_data.get("api_key") == "AIzaGetTest"
        assert init_data.get("name") == "my-agent"
        assert init_data.get("custom_llm_provider") == "gemini"


@pytest.mark.asyncio
async def test_delete_gemini_agent_passes_api_key_to_processor(
    mock_srv, user_api_key_dict
):
    from litellm.proxy.google_endpoints.agents_endpoints import delete_gemini_agent

    with patch(
        "litellm.proxy.google_endpoints.agents_endpoints.ProxyBaseLLMRequestProcessing"
    ) as MockProcessor:
        instance = MockProcessor.return_value
        instance.base_process_llm_request = AsyncMock(return_value=MagicMock())

        request = _make_endpoint_request("api_key=AIzaDeleteTest")
        await delete_gemini_agent(
            request=request,
            name="my-agent",
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

        init_data = MockProcessor.call_args[1]["data"]
        assert init_data.get("api_key") == "AIzaDeleteTest"
        assert init_data.get("name") == "my-agent"
        assert init_data.get("custom_llm_provider") == "gemini"


@pytest.mark.asyncio
async def test_list_gemini_agent_versions_passes_api_key_to_processor(
    mock_srv, user_api_key_dict
):
    from litellm.proxy.google_endpoints.agents_endpoints import (
        list_gemini_agent_versions,
    )

    with patch(
        "litellm.proxy.google_endpoints.agents_endpoints.ProxyBaseLLMRequestProcessing"
    ) as MockProcessor:
        instance = MockProcessor.return_value
        instance.base_process_llm_request = AsyncMock(return_value=MagicMock())

        request = _make_endpoint_request("api_key=AIzaVersionsTest")
        await list_gemini_agent_versions(
            request=request,
            name="my-agent",
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

        init_data = MockProcessor.call_args[1]["data"]
        assert init_data.get("api_key") == "AIzaVersionsTest"
        assert init_data.get("name") == "my-agent"
        assert init_data.get("custom_llm_provider") == "gemini"


@pytest.mark.asyncio
async def test_get_gemini_agent_name_not_overwritten_by_query_param(
    mock_srv, user_api_key_dict
):
    """Path-param ``name`` must not be replaced by an attacker-controlled query param."""
    from litellm.proxy.google_endpoints.agents_endpoints import get_gemini_agent

    with patch(
        "litellm.proxy.google_endpoints.agents_endpoints.ProxyBaseLLMRequestProcessing"
    ) as MockProcessor:
        instance = MockProcessor.return_value
        instance.base_process_llm_request = AsyncMock(return_value=MagicMock())

        request = _make_endpoint_request("name=INJECTED&api_key=AIzaKey")
        await get_gemini_agent(
            request=request,
            name="real-agent",
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

        init_data = MockProcessor.call_args[1]["data"]
        assert init_data["name"] == "real-agent"
        assert init_data["api_key"] == "AIzaKey"


@pytest.mark.asyncio
async def test_list_agents_template_via_query_param(mock_srv, user_api_key_dict):
    """litellm_params_template in query string is expanded."""
    from litellm.proxy.google_endpoints.agents_endpoints import list_gemini_agents
    from urllib.parse import quote

    template = json.dumps({"api_key": "TemplateKey", "vertex_project": "proj-x"})

    with patch(
        "litellm.proxy.google_endpoints.agents_endpoints.ProxyBaseLLMRequestProcessing"
    ) as MockProcessor:
        instance = MockProcessor.return_value
        instance.base_process_llm_request = AsyncMock(return_value=MagicMock())

        request = _make_endpoint_request(f"litellm_params_template={quote(template)}")
        await list_gemini_agents(
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=user_api_key_dict,
        )

        init_data = MockProcessor.call_args[1]["data"]
        assert init_data["api_key"] == "TemplateKey"
        assert init_data["vertex_project"] == "proj-x"
        assert "litellm_params_template" not in init_data
