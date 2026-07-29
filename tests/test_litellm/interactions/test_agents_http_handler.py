"""
Unit tests for litellm/interactions/agents/http_handler.py

These tests exercise both the sync and async branches of every CRUD method
on AgentsHTTPHandler using stub httpx clients, plus the _is_async dispatch
branches, error mapping, and pre/post logging hooks.

No real HTTP traffic is made.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.interactions.agents.http_handler import (
    AgentsHTTPHandler,
    agents_http_handler,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.gemini.agents.transformation import GeminiAgentsConfig
from litellm.llms.gemini.common_utils import GeminiError
from litellm.types.agents import (
    AgentCreateResponse,
    AgentDeleteResult,
    AgentListResponse,
    AgentVersionsResponse,
)
from litellm.types.router import GenericLiteLLMParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status_code: int = 200, json_data=None, text: str = "") -> MagicMock:
    """Build a stub httpx-like response."""
    response = MagicMock()
    response.status_code = status_code
    response.text = text or (str(json_data) if json_data is not None else "")
    response.headers = {}
    if json_data is not None:
        response.json.return_value = json_data
    else:
        response.json.return_value = {}
    return response


def _make_sync_client() -> MagicMock:
    client = MagicMock(spec=HTTPHandler)
    return client


def _make_async_client() -> MagicMock:
    client = MagicMock(spec=AsyncHTTPHandler)
    client.post = AsyncMock()
    client.get = AsyncMock()
    client.delete = AsyncMock()
    return client


def _make_logging_obj() -> MagicMock:
    return MagicMock()


@pytest.fixture
def handler() -> AgentsHTTPHandler:
    return AgentsHTTPHandler()


@pytest.fixture
def config() -> GeminiAgentsConfig:
    return GeminiAgentsConfig()


@pytest.fixture
def litellm_params() -> GenericLiteLLMParams:
    return GenericLiteLLMParams(api_key="AIza-test")


# ---------------------------------------------------------------------------
# Module-level singleton sanity check
# ---------------------------------------------------------------------------


def test_module_singleton_is_agents_http_handler_instance():
    assert isinstance(agents_http_handler, AgentsHTTPHandler)


# ---------------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------------


class TestCreateAgent:
    def test_sync_returns_parsed_create_response(self, handler, config, litellm_params):
        client = _make_sync_client()
        client.post.return_value = _make_response(
            200, json_data={"id": "agent-x", "base_agent": "gemini-2.5-flash"}
        )
        logging_obj = _make_logging_obj()

        result = handler.create_agent(
            agents_api_config=config,
            name="agent-x",
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            extra_headers={"X-Test": "1"},
            extra_body={"foo": "bar"},
            client=client,
        )

        assert isinstance(result, AgentCreateResponse)
        assert result.id == "agent-x"
        client.post.assert_called_once()
        kwargs = client.post.call_args.kwargs
        assert kwargs["url"].endswith("/v1beta/agents")
        assert kwargs["json"]["name"] == "agent-x"
        assert kwargs["json"]["foo"] == "bar"
        assert kwargs["headers"]["X-Test"] == "1"
        logging_obj.pre_call.assert_called_once()
        logging_obj.post_call.assert_called_once()

    def test_sync_dispatches_to_async_when_is_async(
        self, handler, config, litellm_params
    ):
        client = _make_sync_client()

        result = handler.create_agent(
            agents_api_config=config,
            name="agent-x",
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            client=client,
            _is_async=True,
        )

        import asyncio

        assert asyncio.iscoroutine(result)
        result.close()

    def test_sync_maps_http_error_via_config(self, handler, config, litellm_params):
        client = _make_sync_client()
        bad = _make_response(404, text="not found")
        client.post.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=bad
        )

        with pytest.raises(GeminiError):
            handler.create_agent(
                agents_api_config=config,
                name="agent-x",
                litellm_params=litellm_params,
                logging_obj=_make_logging_obj(),
                client=client,
            )

    @pytest.mark.asyncio
    async def test_async_returns_parsed_create_response(
        self, handler, config, litellm_params
    ):
        client = _make_async_client()
        client.post.return_value = _make_response(
            200, json_data={"id": "agent-y", "base_agent": "gemini-2.5-flash"}
        )

        result = await handler.async_create_agent(
            agents_api_config=config,
            name="agent-y",
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            extra_body={"baz": "qux"},
            client=client,
        )

        assert isinstance(result, AgentCreateResponse)
        assert result.id == "agent-y"
        client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_maps_http_error_via_config(
        self, handler, config, litellm_params
    ):
        client = _make_async_client()
        bad = _make_response(500, text="server error")
        client.post.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=bad
        )

        with pytest.raises(GeminiError):
            await handler.async_create_agent(
                agents_api_config=config,
                name="agent-y",
                litellm_params=litellm_params,
                logging_obj=_make_logging_obj(),
                client=client,
            )


# ---------------------------------------------------------------------------
# LIST
# ---------------------------------------------------------------------------


class TestListAgents:
    def test_sync_returns_list_response(self, handler, config, litellm_params):
        client = _make_sync_client()
        client.get.return_value = _make_response(
            200,
            json_data={
                "agents": [{"id": "a-1"}, {"id": "a-2"}],
                "nextPageToken": "tok",
            },
        )

        result = handler.list_agents(
            agents_api_config=config,
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            client=client,
        )

        assert isinstance(result, AgentListResponse)
        assert len(result.agents) == 2
        assert result.next_page_token == "tok"
        client.get.assert_called_once()

    def test_sync_dispatches_to_async_when_is_async(
        self, handler, config, litellm_params
    ):
        client = _make_sync_client()

        result = handler.list_agents(
            agents_api_config=config,
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            client=client,
            _is_async=True,
        )

        import asyncio

        assert asyncio.iscoroutine(result)
        result.close()

    def test_sync_maps_http_error_via_config(self, handler, config, litellm_params):
        client = _make_sync_client()
        bad = _make_response(403, text="forbidden")
        client.get.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=bad
        )

        with pytest.raises(GeminiError):
            handler.list_agents(
                agents_api_config=config,
                litellm_params=litellm_params,
                logging_obj=_make_logging_obj(),
                client=client,
            )

    @pytest.mark.asyncio
    async def test_async_returns_list_response(self, handler, config, litellm_params):
        client = _make_async_client()
        client.get.return_value = _make_response(
            200, json_data={"agents": [{"id": "a-1"}]}
        )

        result = await handler.async_list_agents(
            agents_api_config=config,
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            client=client,
        )

        assert isinstance(result, AgentListResponse)
        assert len(result.agents) == 1
        client.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_maps_http_error_via_config(
        self, handler, config, litellm_params
    ):
        client = _make_async_client()
        bad = _make_response(429, text="rate limited")
        client.get.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=bad
        )

        with pytest.raises(GeminiError):
            await handler.async_list_agents(
                agents_api_config=config,
                litellm_params=litellm_params,
                logging_obj=_make_logging_obj(),
                client=client,
            )


# ---------------------------------------------------------------------------
# GET
# ---------------------------------------------------------------------------


class TestGetAgent:
    def test_sync_returns_get_response(self, handler, config, litellm_params):
        client = _make_sync_client()
        client.get.return_value = _make_response(200, json_data={"id": "agent-x"})

        result = handler.get_agent(
            agents_api_config=config,
            name="agent-x",
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            client=client,
        )

        assert isinstance(result, AgentCreateResponse)
        assert result.id == "agent-x"
        kwargs = client.get.call_args.kwargs
        assert kwargs["url"].endswith("/v1beta/agents/agent-x")

    def test_sync_dispatches_to_async_when_is_async(
        self, handler, config, litellm_params
    ):
        result = handler.get_agent(
            agents_api_config=config,
            name="agent-x",
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            client=_make_sync_client(),
            _is_async=True,
        )
        import asyncio

        assert asyncio.iscoroutine(result)
        result.close()

    def test_sync_maps_http_error_via_config(self, handler, config, litellm_params):
        client = _make_sync_client()
        bad = _make_response(404, text="not found")
        client.get.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=bad
        )

        with pytest.raises(GeminiError):
            handler.get_agent(
                agents_api_config=config,
                name="agent-x",
                litellm_params=litellm_params,
                logging_obj=_make_logging_obj(),
                client=client,
            )

    @pytest.mark.asyncio
    async def test_async_returns_get_response(self, handler, config, litellm_params):
        client = _make_async_client()
        client.get.return_value = _make_response(200, json_data={"id": "agent-y"})

        result = await handler.async_get_agent(
            agents_api_config=config,
            name="agent-y",
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            client=client,
        )

        assert isinstance(result, AgentCreateResponse)
        assert result.id == "agent-y"

    @pytest.mark.asyncio
    async def test_async_maps_http_error_via_config(
        self, handler, config, litellm_params
    ):
        client = _make_async_client()
        bad = _make_response(404, text="not found")
        client.get.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=bad
        )

        with pytest.raises(GeminiError):
            await handler.async_get_agent(
                agents_api_config=config,
                name="agent-x",
                litellm_params=litellm_params,
                logging_obj=_make_logging_obj(),
                client=client,
            )


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------


class TestDeleteAgent:
    def test_sync_returns_delete_result(self, handler, config, litellm_params):
        client = _make_sync_client()
        client.delete.return_value = _make_response(200, json_data={})

        result = handler.delete_agent(
            agents_api_config=config,
            name="agent-x",
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            client=client,
        )

        assert isinstance(result, AgentDeleteResult)
        assert result.name == "agent-x"
        assert result.deleted is True
        kwargs = client.delete.call_args.kwargs
        assert kwargs["url"].endswith("/v1beta/agents/agent-x")

    def test_sync_dispatches_to_async_when_is_async(
        self, handler, config, litellm_params
    ):
        result = handler.delete_agent(
            agents_api_config=config,
            name="agent-x",
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            client=_make_sync_client(),
            _is_async=True,
        )
        import asyncio

        assert asyncio.iscoroutine(result)
        result.close()

    def test_sync_maps_http_error_via_config(self, handler, config, litellm_params):
        client = _make_sync_client()
        bad = _make_response(403, text="forbidden")
        client.delete.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=bad
        )

        with pytest.raises(GeminiError):
            handler.delete_agent(
                agents_api_config=config,
                name="agent-x",
                litellm_params=litellm_params,
                logging_obj=_make_logging_obj(),
                client=client,
            )

    @pytest.mark.asyncio
    async def test_async_returns_delete_result(self, handler, config, litellm_params):
        client = _make_async_client()
        client.delete.return_value = _make_response(200, json_data={})

        result = await handler.async_delete_agent(
            agents_api_config=config,
            name="agent-y",
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            client=client,
        )

        assert isinstance(result, AgentDeleteResult)
        assert result.name == "agent-y"
        assert result.deleted is True

    @pytest.mark.asyncio
    async def test_async_maps_http_error_via_config(
        self, handler, config, litellm_params
    ):
        client = _make_async_client()
        bad = _make_response(500, text="server error")
        client.delete.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=bad
        )

        with pytest.raises(GeminiError):
            await handler.async_delete_agent(
                agents_api_config=config,
                name="agent-x",
                litellm_params=litellm_params,
                logging_obj=_make_logging_obj(),
                client=client,
            )


# ---------------------------------------------------------------------------
# LIST VERSIONS
# ---------------------------------------------------------------------------


class TestListAgentVersions:
    def test_sync_returns_versions_response(self, handler, config, litellm_params):
        client = _make_sync_client()
        client.get.return_value = _make_response(
            200,
            json_data={
                "agentVersions": [
                    {"agent": "agent-x", "name": "agents/agent-x/versions/v1"}
                ],
                "nextPageToken": "tok",
            },
        )

        result = handler.list_agent_versions(
            agents_api_config=config,
            name="agent-x",
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            client=client,
        )

        assert isinstance(result, AgentVersionsResponse)
        assert len(result.agent_versions) == 1
        assert result.next_page_token == "tok"
        kwargs = client.get.call_args.kwargs
        assert kwargs["url"].endswith("/v1beta/agents/agent-x/versions")

    def test_sync_dispatches_to_async_when_is_async(
        self, handler, config, litellm_params
    ):
        result = handler.list_agent_versions(
            agents_api_config=config,
            name="agent-x",
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            client=_make_sync_client(),
            _is_async=True,
        )
        import asyncio

        assert asyncio.iscoroutine(result)
        result.close()

    def test_sync_maps_http_error_via_config(self, handler, config, litellm_params):
        client = _make_sync_client()
        bad = _make_response(404, text="not found")
        client.get.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=bad
        )

        with pytest.raises(GeminiError):
            handler.list_agent_versions(
                agents_api_config=config,
                name="agent-x",
                litellm_params=litellm_params,
                logging_obj=_make_logging_obj(),
                client=client,
            )

    @pytest.mark.asyncio
    async def test_async_returns_versions_response(
        self, handler, config, litellm_params
    ):
        client = _make_async_client()
        client.get.return_value = _make_response(200, json_data={"agentVersions": []})

        result = await handler.async_list_agent_versions(
            agents_api_config=config,
            name="agent-y",
            litellm_params=litellm_params,
            logging_obj=_make_logging_obj(),
            client=client,
        )

        assert isinstance(result, AgentVersionsResponse)
        assert result.agent_versions == []

    @pytest.mark.asyncio
    async def test_async_maps_http_error_via_config(
        self, handler, config, litellm_params
    ):
        client = _make_async_client()
        bad = _make_response(500, text="server error")
        client.get.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=bad
        )

        with pytest.raises(GeminiError):
            await handler.async_list_agent_versions(
                agents_api_config=config,
                name="agent-x",
                litellm_params=litellm_params,
                logging_obj=_make_logging_obj(),
                client=client,
            )
