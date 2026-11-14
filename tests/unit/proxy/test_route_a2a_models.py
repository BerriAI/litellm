"""
Test A2A model routing in proxy.

Maps to: litellm/proxy/agent_endpoints/a2a_routing.py
"""

import asyncio
import base64
import datetime
import json
import logging
import math
import uuid
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, Mock, patch

from tests.unit.proxy.router_entry_point_stubs import route_entry_points_to_stubs

import httpx
import pytest
import respx
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.proxy._types import SpendLogsPayload
from litellm.proxy.agent_endpoints.a2a_routing import route_a2a_agent_request
from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
from litellm.proxy.agent_endpoints.databricks_oauth import databricks_app_oauth_token_cache
from litellm.proxy.common_request_processing import cost_headers
from litellm.proxy.route_llm_request import route_request
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.types.agents import AgentResponse
from litellm.types.utils import ModelResponse


@pytest.mark.asyncio
async def test_route_a2a_model_bypasses_router():
    """Test that a2a/ prefixed models bypass router and go directly to litellm with api_base"""

    # Mock data for chat completion with a2a model
    data = {
        "model": "a2a/test-agent",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Mock router that doesn't have the a2a model
    mock_router = route_entry_points_to_stubs(Mock())
    mock_router.model_names = ["gpt-4", "gpt-3.5-turbo"]
    mock_router.deployment_names = []
    mock_router.has_model_id = Mock(return_value=False)
    mock_router.is_recognized_model = Mock(return_value=False)
    mock_router.model_group_alias = None
    mock_router.router_general_settings = Mock(pass_through_all_models=False)
    mock_router.default_deployment = None
    mock_router.pattern_router = Mock(patterns=[])
    mock_router.map_team_model = Mock(return_value=None)

    # Mock agent in registry
    from litellm.types.agents import AgentResponse

    mock_agent = AgentResponse(
        agent_id="test-agent-id",
        agent_name="test-agent",
        agent_card_params={"url": "http://agent.example.com"},
        litellm_params=None,
    )

    mock_registry = Mock()
    mock_registry.get_agent_by_id = Mock(return_value=None)
    mock_registry.get_agent_by_name = Mock(return_value=mock_agent)

    # Mock litellm.acompletion to verify it's called
    mock_acompletion = AsyncMock(return_value={"id": "test-response"})

    with patch("litellm.acompletion", mock_acompletion):
        with patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            mock_registry,
        ):
            result = await route_request(
                data=data,
                llm_router=mock_router,
                user_model=None,
                route_type="acompletion",
            )

            # Verify litellm.acompletion was called with api_base injected
            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["model"] == "a2a/test-agent"
            assert call_kwargs["api_base"] == "http://agent.example.com"


@pytest.mark.asyncio
async def test_route_non_a2a_model_raises_error_if_not_in_router():
    """Test that non-a2a models that aren't in router raise an error"""

    # Mock data for chat completion with model not in router
    data = {
        "model": "unknown-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Mock router without the model
    mock_router = route_entry_points_to_stubs(Mock())
    mock_router.model_names = ["gpt-4", "gpt-3.5-turbo"]
    mock_router.deployment_names = []
    mock_router.has_model_id = Mock(return_value=False)
    mock_router.is_recognized_model = Mock(return_value=False)
    mock_router.model_group_alias = None
    mock_router.router_general_settings = Mock(pass_through_all_models=False)
    mock_router.default_deployment = None
    mock_router.pattern_router = Mock(patterns=[])
    mock_router.map_team_model = Mock(return_value=None)

    # Should raise ProxyModelNotFoundError
    from litellm.proxy.route_llm_request import ProxyModelNotFoundError

    with pytest.raises(ProxyModelNotFoundError):
        await route_request(
            data=data,
            llm_router=mock_router,
            user_model=None,
            route_type="acompletion",
        )


class _DbAgentRow:
    def __init__(self, agent_id: str, agent_name: str) -> None:
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.object_permission = None
        self.spend = 0.0

    def model_dump(self):
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_card_params": {"name": self.agent_name, "url": "http://sibling-db-agent.example.com"},
            "litellm_params": {},
            "object_permission": None,
            "spend": self.spend,
        }


def _router_without_models():
    mock_router = route_entry_points_to_stubs(Mock())
    mock_router.model_names = []
    mock_router.deployment_names = []
    mock_router.has_model_id = Mock(return_value=False)
    mock_router.model_group_alias = None
    mock_router.router_general_settings = Mock(pass_through_all_models=False)
    mock_router.default_deployment = None
    mock_router.pattern_router = Mock(patterns=[])
    mock_router.map_team_model = Mock(return_value=None)
    mock_router.is_recognized_model = Mock(return_value=False)
    mock_router.team_public_model_names = []
    return mock_router


@pytest.mark.asyncio
async def test_route_a2a_model_read_through_recovers_agent_created_on_sibling_replica(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    agent_name = "a2a-sibling-replica-agent"
    prisma_client = Mock()
    prisma_client.db.litellm_agentstable.find_unique = AsyncMock(
        side_effect=[None, _DbAgentRow("a2a-sibling-replica-agent-id", agent_name)]
    )
    prisma_client.writer_db.litellm_agentstable.find_unique = AsyncMock(
        return_value=_DbAgentRow("a2a-sibling-replica-agent-id", agent_name)
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)
    monkeypatch.setattr(proxy_server, "store_model_in_db", True)

    original_agents = list(global_agent_registry.agent_list)
    original_config_agents = getattr(global_agent_registry, "config_agents", ())
    global_agent_registry.agent_list = []
    global_agent_registry.config_agents = ()

    data = {
        "model": f"a2a/{agent_name}",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    mock_acompletion = AsyncMock(return_value={"id": "read-through-response"})

    try:
        with patch("litellm.acompletion", mock_acompletion):
            await route_request(
                data=data,
                llm_router=_router_without_models(),
                user_model=None,
                route_type="acompletion",
            )
    finally:
        global_agent_registry.agent_list = original_agents
        global_agent_registry.config_agents = original_config_agents

    mock_acompletion.assert_called_once()
    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["model"] == f"a2a/{agent_name}"
    assert call_kwargs["api_base"] == "http://sibling-db-agent.example.com"
    prisma_client.db.litellm_agentstable.find_unique.assert_awaited()


WORKSPACE = "https://adb-1.azuredatabricks.net"
APP_URL = "https://my-app-1.azure.databricksapps.com/responses"
AGENT_REPLY = {
    "output": [
        {"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "pong"}]}
    ]
}
A2A_REPLY = {
    "jsonrpc": "2.0",
    "id": "1",
    "result": {"kind": "message", "role": "agent", "messageId": "m-1", "parts": [{"kind": "text", "text": "pong"}]},
}


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    databricks_app_oauth_token_cache.flush_cache()
    yield
    databricks_app_oauth_token_cache.flush_cache()


def _registry_with(agent: AgentResponse) -> AgentRegistry:
    registry = AgentRegistry()
    registry.register_agent(agent)
    return registry


def _token_route(workspace: str) -> respx.Route:
    return respx.post(f"{workspace}/oidc/v1/token").mock(
        return_value=httpx.Response(200, json={"access_token": "minted-oauth", "expires_in": 3600})
    )


@respx.mock
async def test_route_a2a_bridge_agent_calls_the_endpoint_with_the_agent_params(httpx_transport: None) -> None:
    agent = AgentResponse(
        agent_id="dbx-agent-id",
        agent_name="dbx-agent",
        agent_card_params={"url": WORKSPACE},
        litellm_params={
            "custom_llm_provider": "databricks_agent",
            "model": "my-agent",
            "api_base": WORKSPACE,
            "api_key": "pat-1",
            "custom_inputs": {"tenant": "t-1"},
            "is_public": True,
        },
    )
    endpoint = respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    data = {"model": "a2a/dbx-agent", "messages": [{"role": "user", "content": "ping"}]}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        response = await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    sent = endpoint.calls.last.request
    assert sent.headers["Authorization"] == "Bearer pat-1"
    assert json.loads(sent.content) == {
        "input": [{"role": "user", "content": "ping"}],
        "custom_inputs": {"tenant": "t-1"},
    }
    assert response.choices[0].message.content == "pong"


@respx.mock
async def test_route_a2a_bridge_agent_merges_client_static_and_agent_extra_headers(httpx_transport: None) -> None:
    agent = AgentResponse(
        agent_id="dbx-agent-id",
        agent_name="dbx-agent",
        agent_card_params={"url": WORKSPACE},
        static_headers={"X-Static": "static", "x-shared": "from-static", "X-Both": "from-static"},
        litellm_params={
            "custom_llm_provider": "databricks_agent",
            "model": "my-agent",
            "api_base": WORKSPACE,
            "api_key": "pat-1",
            "extra_headers": {"X-Agent": "agent", "x-both": "from-agent"},
        },
    )
    endpoint = respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    data = {
        "model": "a2a/dbx-agent",
        "messages": [{"role": "user", "content": "ping"}],
        "extra_headers": {"X-Trace": "trace-1", "X-Shared": "from-client", "X-LiteLLM-User-Id": "spoofed-user"},
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    sent = endpoint.calls.last.request
    assert "X-LiteLLM-User-Id" not in sent.headers
    assert sent.headers["X-Trace"] == "trace-1"
    assert sent.headers["X-Static"] == "static"
    assert sent.headers["X-Agent"] == "agent"
    assert sent.headers.get_list("X-Shared") == ["from-static"]
    assert sent.headers.get_list("X-Both") == ["from-agent"]
    assert sent.headers["Authorization"] == "Bearer pat-1"


async def test_route_a2a_bridge_agent_lets_the_request_win_except_for_the_agent_credentials() -> None:
    agent = AgentResponse(
        agent_id="dbx-agent-id",
        agent_name="dbx-agent",
        agent_card_params={"url": WORKSPACE},
        litellm_params={
            "custom_llm_provider": "databricks_agent",
            "model": "my-agent",
            "api_base": WORKSPACE,
            "api_key": "pat-1",
            "metadata": {"tags": ["agent-tag"]},
            "custom_inputs": {"tenant": "t-1"},
            "timeout": 30,
        },
    )
    proxy_metadata = {"user_api_key_hash": "key-hash", "user_api_key_team_id": "team-1"}
    data = {
        "model": "a2a/dbx-agent",
        "messages": [{"role": "user", "content": "ping"}],
        "metadata": proxy_metadata,
        "api_base": "https://elsewhere.example.com",
        "api_key": "client-key",
    }
    completion = Mock(return_value="reply")

    with (
        patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)),
        patch.object(litellm, "acompletion", completion),
    ):
        await route_a2a_agent_request(data=data, route_type="acompletion")

    sent = completion.call_args.kwargs
    assert sent["metadata"] == {**proxy_metadata, "model_group": "a2a/dbx-agent"}
    assert (sent["api_base"], sent["api_key"]) == (WORKSPACE, "pat-1")
    assert (sent["custom_inputs"], sent["timeout"]) == ({"tenant": "t-1"}, 30)
    assert sent["model"] == "databricks_agent/my-agent"


@respx.mock
async def test_route_a2a_bridge_agent_keeps_its_custom_inputs_and_extra_body_keys_over_the_callers(
    httpx_transport: None,
) -> None:
    agent = AgentResponse(
        agent_id="dbx-agent-id",
        agent_name="dbx-agent",
        agent_card_params={"url": WORKSPACE},
        litellm_params={
            "custom_llm_provider": "databricks_agent",
            "model": "my-agent",
            "api_base": WORKSPACE,
            "api_key": "pat-1",
            "custom_inputs": {"tenant": "owner"},
            "extra_body": {"metadata": {"tenant": "owner"}},
        },
    )
    endpoint = respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    data = {
        "model": "a2a/dbx-agent",
        "messages": [{"role": "user", "content": "ping"}],
        "custom_inputs": {"tenant": "caller", "session": "s-1"},
        "extra_body": {"metadata": {"tenant": "caller", "turn": "2"}},
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    assert json.loads(endpoint.calls.last.request.content) == {
        "input": [{"role": "user", "content": "ping"}],
        "custom_inputs": {"tenant": "owner", "session": "s-1"},
        "metadata": {"tenant": "owner", "turn": "2"},
    }


@respx.mock
async def test_route_a2a_bridge_agent_with_an_empty_oauth_block_fails_before_calling_the_app(
    httpx_transport: None,
) -> None:
    agent = AgentResponse(
        agent_id="dbx-app-id",
        agent_name="dbx-app",
        agent_card_params={"url": APP_URL},
        litellm_params={"custom_llm_provider": "databricks_agent", "databricks_oauth": {}},
    )
    app = respx.post(APP_URL).mock(return_value=httpx.Response(200, json=AGENT_REPLY))
    data = {"model": "a2a/dbx-app", "messages": [{"role": "user", "content": "ping"}]}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        with pytest.raises(litellm.BadRequestError, match="missing required field") as info:
            await route_a2a_agent_request(data=data, route_type="acompletion")

    assert info.value.status_code == 400
    assert not app.called


@respx.mock
@pytest.mark.parametrize(
    "oauth_params",
    [
        {"databricks_oauth": {"client_id": "sp", "client_secret": "s", "workspace_url": WORKSPACE}},
        {"client_id": "sp", "client_secret": "s", "workspace_url": WORKSPACE, "scope": "all-apis"},
    ],
)
async def test_route_a2a_bridge_agent_with_databricks_oauth_sends_the_minted_token(
    httpx_transport: None, oauth_params: dict[str, object]
) -> None:
    agent = AgentResponse(
        agent_id="dbx-app-id",
        agent_name="dbx-app",
        agent_card_params={"url": APP_URL},
        litellm_params={"custom_llm_provider": "databricks_agent", **oauth_params},
    )
    token = _token_route(WORKSPACE)
    app = respx.post(APP_URL).mock(return_value=httpx.Response(200, json=AGENT_REPLY))
    data = {
        "model": "a2a/dbx-app",
        "messages": [{"role": "user", "content": "ping"}],
        "extra_headers": {"X-Tenant": "t"},
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        response = await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    assert token.calls.last.request.headers["Authorization"] == f"Basic {base64.b64encode(b'sp:s').decode()}"
    sent = app.calls.last.request
    assert sent.headers["Authorization"] == "Bearer minted-oauth"
    assert sent.headers["X-Tenant"] == "t"
    assert json.loads(sent.content) == {"input": [{"role": "user", "content": "ping"}]}
    assert response.choices[0].message.content == "pong"


@respx.mock
@pytest.mark.parametrize(
    "oauth_block",
    [
        {"client_id": "sp", "client_secret": "s", "workspace_url": WORKSPACE},
        {"client_id": "sp"},
        "not-a-mapping",
        {"client_id": "sp", "client_secret": "s", "workspace_url": "https://my-app-1.azure.databricksapps.com"},
    ],
    ids=["complete", "partial", "not-a-mapping", "app-host-as-workspace"],
)
async def test_route_a2a_url_agent_chat_keeps_the_client_authorization_over_its_databricks_oauth(
    httpx_transport: None, oauth_block: object
) -> None:
    agent_url = "https://my-a2a-app.azure.databricksapps.com"
    agent = AgentResponse(
        agent_id="a2a-app-id",
        agent_name="a2a-app",
        agent_card_params={"url": agent_url},
        litellm_params={"databricks_oauth": oauth_block},
    )
    token = _token_route(WORKSPACE)
    app = respx.post(agent_url).mock(return_value=httpx.Response(200, json=A2A_REPLY))
    data = {
        "model": "a2a/a2a-app",
        "messages": [{"role": "user", "content": "ping"}],
        "extra_headers": {"Authorization": "Bearer client-token"},
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        response = await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    sent = app.calls.last.request
    assert sent.headers["Authorization"] == "Bearer client-token"
    assert json.loads(sent.content)["method"] == "message/send"
    assert not token.called
    assert response.choices[0].message.content == "pong"


@respx.mock
async def test_route_a2a_agent_with_another_provider_keeps_calling_its_card_url_over_a2a(
    httpx_transport: None,
) -> None:
    agent_url = "https://pydantic-agent.example.com"
    agent = AgentResponse(
        agent_id="pydantic-agent-id",
        agent_name="pydantic-agent",
        agent_card_params={"url": agent_url},
        litellm_params={"custom_llm_provider": "pydantic_ai_agents", "model": "pydantic-agent"},
    )
    agent_endpoint = respx.post(agent_url).mock(return_value=httpx.Response(200, json=A2A_REPLY))
    data = {"model": "a2a/pydantic-agent", "messages": [{"role": "user", "content": "ping"}]}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        response = await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    assert json.loads(agent_endpoint.calls.last.request.content)["method"] == "message/send"
    assert response.choices[0].message.content == "pong"


@respx.mock
@pytest.mark.parametrize("client_target", ["api_base", "base_url"])
async def test_route_a2a_bridge_agent_ignores_a_client_target_over_its_card_url(
    httpx_transport: None, client_target: str
) -> None:
    agent = AgentResponse(
        agent_id="dbx-agent-id",
        agent_name="dbx-agent",
        agent_card_params={"url": WORKSPACE},
        litellm_params={"custom_llm_provider": "databricks_agent", "model": "my-agent", "api_key": "pat-1"},
    )
    endpoint = respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    data = {
        "model": "a2a/dbx-agent",
        client_target: "https://collector.example.com",
        "messages": [{"role": "user", "content": "ping"}],
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        response = await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    assert endpoint.call_count == 1
    assert endpoint.calls.last.request.headers["Authorization"] == "Bearer pat-1"
    assert response.choices[0].message.content == "pong"


@respx.mock
@pytest.mark.parametrize("endpoint_name", [{"model": ""}, {"model": None}, {}])
async def test_route_a2a_bridge_agent_without_an_endpoint_name_answers_400_before_calling_databricks(
    httpx_transport: None, endpoint_name: dict[str, object]
) -> None:
    agent = AgentResponse(
        agent_id="dbx-agent-id",
        agent_name="dbx-agent",
        agent_card_params={"url": WORKSPACE},
        litellm_params={
            "custom_llm_provider": "databricks_agent",
            "api_base": WORKSPACE,
            "api_key": "p",
            **endpoint_name,
        },
    )
    workspace = respx.route(host="adb-1.azuredatabricks.net").mock(return_value=httpx.Response(200, json=AGENT_REPLY))
    data = {"model": "a2a/dbx-agent", "messages": [{"role": "user", "content": "ping"}]}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        with pytest.raises(litellm.BadRequestError, match="must name the Databricks serving endpoint"):
            await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    assert not workspace.called


@respx.mock
async def test_route_a2a_bridge_app_agent_without_an_endpoint_name_calls_the_app(httpx_transport: None) -> None:
    agent = AgentResponse(
        agent_id="dbx-app-id",
        agent_name="dbx-app",
        agent_card_params={"url": APP_URL},
        litellm_params={"custom_llm_provider": "databricks_agent", "api_key": "p"},
    )
    app = respx.post(APP_URL).mock(return_value=httpx.Response(200, json=AGENT_REPLY))
    data = {"model": "a2a/dbx-app", "messages": [{"role": "user", "content": "ping"}]}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        response = await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    assert app.called
    assert response.choices[0].message.content == "pong"


@respx.mock
@pytest.mark.parametrize(
    ("credential", "env_api_key", "expected_authorization"),
    [
        ({"api_key": "pat-1"}, None, "Bearer pat-1"),
        ({}, "env-pat", "Bearer env-pat"),
        ({"client_id": "sp", "client_secret": "s", "workspace_url": WORKSPACE}, "env-pat", "Bearer minted-oauth"),
        (
            {"client_id": "sp", "client_secret": "s", "workspace_url": WORKSPACE, "api_key": "pat-1"},
            None,
            "Bearer minted-oauth",
        ),
        ({}, None, "Bearer client-token"),
    ],
)
async def test_route_a2a_bridge_agent_credential_wins_over_a_client_authorization(
    httpx_transport: None,
    monkeypatch: pytest.MonkeyPatch,
    credential: dict[str, object],
    env_api_key: str | None,
    expected_authorization: str,
) -> None:
    if env_api_key is None:
        monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    else:
        monkeypatch.setenv("DATABRICKS_API_KEY", env_api_key)
    agent = AgentResponse(
        agent_id="dbx-app-id",
        agent_name="dbx-app",
        agent_card_params={"url": APP_URL},
        litellm_params={"custom_llm_provider": "databricks_agent", **credential},
    )
    _token_route(WORKSPACE)
    app = respx.post(APP_URL).mock(return_value=httpx.Response(200, json=AGENT_REPLY))
    data = {
        "model": "a2a/dbx-app",
        "messages": [{"role": "user", "content": "ping"}],
        "extra_headers": {"authorization": "Bearer client-token", "X-Tenant": "t"},
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    sent = app.calls.last.request
    assert sent.headers.get_list("Authorization") == [expected_authorization]
    assert sent.headers["X-Tenant"] == "t"


@respx.mock
@pytest.mark.parametrize("extra_headers", [5, ["X-A"], {"X-N": 5}, "x"])
async def test_route_a2a_bridge_agent_answers_400_for_malformed_client_extra_headers(
    httpx_transport: None, extra_headers: object
) -> None:
    agent = AgentResponse(
        agent_id="dbx-app-id",
        agent_name="dbx-app",
        agent_card_params={"url": APP_URL},
        litellm_params={"custom_llm_provider": "databricks_agent", "api_key": "p"},
    )
    app = respx.post(APP_URL).mock(return_value=httpx.Response(200, json=AGENT_REPLY))
    data = {"model": "a2a/dbx-app", "messages": [{"role": "user", "content": "ping"}], "extra_headers": extra_headers}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        with pytest.raises(HTTPException) as info:
            await route_a2a_agent_request(data=data, route_type="acompletion")

    assert info.value.status_code == 400
    assert "extra_headers" in str(info.value.detail)
    assert not app.called


@respx.mock
@pytest.mark.parametrize(
    ("extra_headers", "drop_params"),
    [
        (5, {"drop_params": True}),
        ({"X-N": 5}, {"drop_params": True}),
        ("", {}),
        ({}, {}),
        (None, {}),
    ],
)
async def test_route_a2a_bridge_agent_calls_without_client_extra_headers_it_drops_or_finds_empty(
    httpx_transport: None, extra_headers: object, drop_params: dict[str, object]
) -> None:
    agent = AgentResponse(
        agent_id="dbx-app-id",
        agent_name="dbx-app",
        agent_card_params={"url": APP_URL},
        litellm_params={"custom_llm_provider": "databricks_agent", "api_key": "p"},
    )
    app = respx.post(APP_URL).mock(return_value=httpx.Response(200, json=AGENT_REPLY))
    data = {
        "model": "a2a/dbx-app",
        "messages": [{"role": "user", "content": "ping"}],
        "extra_headers": extra_headers,
        **drop_params,
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        response = await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    assert app.calls.last.request.headers["Authorization"] == "Bearer p"
    assert "X-N" not in app.calls.last.request.headers
    assert response.choices[0].message.content == "pong"


PER_QUERY_PRICE: Final = {"cost_per_query": 0.05}
PER_TOKEN_PRICE: Final = {"input_cost_per_token": 0.0001, "output_cost_per_token": 0.0002}


def _priced_dbx_agent(name: str, price: Mapping[str, object]) -> AgentResponse:
    return AgentResponse(
        agent_id=f"{name}-id",
        agent_name=name,
        litellm_params={
            "custom_llm_provider": "databricks_agent",
            "model": "my-agent",
            "api_base": WORKSPACE,
            "api_key": "pat-1",
            **price,
        },
        agent_card_params={"url": WORKSPACE},
    )


def _proxy_logging_obj(stream: bool = False) -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="a2a/agent",
        messages=[{"role": "user", "content": "ping"}],
        stream=stream,
        call_type="acompletion",
        start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="a2a-chat-call",
        function_id="a2a-chat",
    )


async def _billed_chat_call(
    registry: AgentRegistry, agent_name: str, request: Mapping[str, object] = MappingProxyType({})
) -> tuple[float | None, ModelResponse, LiteLLMLoggingObj]:
    logging_obj: Final = _proxy_logging_obj()
    data: Final = {
        "model": f"a2a/{agent_name}",
        "messages": [{"role": "user", "content": "ping"}],
        "metadata": {"user_api_key_hash": "key-hash", "model_group": "client-says"},
        "litellm_logging_obj": logging_obj,
        **request,
    }
    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        response: Final = await (await route_a2a_agent_request(data=data, route_type="acompletion"))
    return logging_obj.response_cost_calculator(result=response), response, logging_obj


@respx.mock
async def test_route_a2a_bridge_agents_bill_their_own_price_whatever_another_agent_was_billed(
    httpx_transport: None,
) -> None:
    respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    registry: Final = AgentRegistry()
    for agent in (
        _priced_dbx_agent("per-token", PER_TOKEN_PRICE),
        _priced_dbx_agent("per-query", PER_QUERY_PRICE),
        _priced_dbx_agent("unpriced", {}),
    ):
        registry.register_agent(agent)

    per_token_cost, per_token_response, _ = await _billed_chat_call(registry, "per-token")
    per_query_cost, _, _ = await _billed_chat_call(registry, "per-query")
    unpriced_cost, _, _ = await _billed_chat_call(registry, "unpriced")

    usage: Final = per_token_response.usage
    assert usage.prompt_tokens > 0
    assert per_token_cost == pytest.approx(usage.prompt_tokens * 0.0001 + usage.completion_tokens * 0.0002)
    assert per_query_cost == pytest.approx(0.05)
    assert unpriced_cost == 0.0


@respx.mock
async def test_route_a2a_bridge_agent_price_wins_over_the_callers_prices(httpx_transport: None) -> None:
    respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    client_prices: Final = {"cost_per_query": 0, "input_cost_per_token": 0, "output_cost_per_token": 0}

    cost, _, _ = await _billed_chat_call(
        _registry_with(_priced_dbx_agent("per-query", PER_QUERY_PRICE)), "per-query", client_prices
    )

    assert cost == pytest.approx(0.05)


@respx.mock
async def test_route_a2a_bridge_agent_call_is_attributed_to_the_agent(httpx_transport: None) -> None:
    respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )

    _, _, logging_obj = await _billed_chat_call(
        _registry_with(_priced_dbx_agent("per-query", PER_QUERY_PRICE)), "per-query"
    )

    assert logging_obj.model_call_details["agent_id"] == "per-query-id"
    metadata: Final = logging_obj.model_call_details["litellm_params"]["metadata"]
    assert (metadata["model_group"], metadata["user_api_key_hash"]) == ("a2a/per-query", "key-hash")


@respx.mock
@pytest.mark.parametrize(
    ("route_type", "request_body"),
    [
        ("aresponses", {"input": "ping"}),
        ("anthropic_messages", {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 16}),
    ],
)
async def test_route_a2a_bridge_agent_call_names_the_agent_on_litellm_metadata_routes(
    httpx_transport: None, route_type: str, request_body: Mapping[str, object]
) -> None:
    respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    logging_obj: Final = _proxy_logging_obj()
    data: Final = {
        "model": "a2a/per-query",
        **request_body,
        "litellm_metadata": {"user_api_key_hash": "key-hash"},
        "litellm_logging_obj": logging_obj,
    }
    registry: Final = _registry_with(_priced_dbx_agent("per-query", PER_QUERY_PRICE))
    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        await (await route_a2a_agent_request(data=data, route_type=route_type))

    metadata: Final = logging_obj.model_call_details["litellm_params"]["metadata"]
    assert (metadata["model_group"], metadata["user_api_key_hash"]) == ("a2a/per-query", "key-hash")


@respx.mock
async def test_route_a2a_bridge_agent_price_gets_the_proxys_discount_and_margin(
    httpx_transport: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    monkeypatch.setattr(litellm, "cost_discount_config", {"databricks_agent": 0.2})
    monkeypatch.setattr(litellm, "cost_margin_config", {"global": 0.1})

    cost, _, logging_obj = await _billed_chat_call(
        _registry_with(_priced_dbx_agent("per-query", PER_QUERY_PRICE)), "per-query"
    )

    assert cost == pytest.approx(0.05 * 0.8 * 1.1)
    assert logging_obj.cost_breakdown is not None
    assert logging_obj.cost_breakdown["original_cost"] == pytest.approx(0.05)


@respx.mock
@pytest.mark.parametrize(
    ("price", "expected_cost"),
    [({"cost_per_query": "five cents"}, None), ({"cost_per_query": ""}, 0.0), ({"cost_per_query": "nan"}, 0.0)],
)
async def test_route_a2a_bridge_agent_with_an_unusable_price_still_answers(
    httpx_transport: None, price: Mapping[str, object], expected_cost: float | None
) -> None:
    respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )

    cost, response, logging_obj = await _billed_chat_call(
        _registry_with(_priced_dbx_agent("odd-price", price)), "odd-price"
    )

    assert response.choices[0].message.content == "pong"
    assert cost == expected_cost
    header_values: Final = cost_headers(response_cost=cost, litellm_logging_obj=logging_obj).values()
    assert all(math.isfinite(float(value)) for value in header_values if value not in (None, "None"))


@respx.mock
async def test_route_a2a_bridge_token_priced_agent_reports_its_input_and_output_cost(
    httpx_transport: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    monkeypatch.setattr(litellm, "cost_margin_config", {"global": 0.1})

    cost, response, logging_obj = await _billed_chat_call(
        _registry_with(_priced_dbx_agent("per-token", PER_TOKEN_PRICE)), "per-token"
    )

    headers: Final = cost_headers(response_cost=cost, litellm_logging_obj=logging_obj)
    input_cost: Final = response.usage.prompt_tokens * 0.0001
    output_cost: Final = response.usage.completion_tokens * 0.0002
    assert float(headers["x-litellm-response-cost-input"]) == pytest.approx(input_cost)
    assert float(headers["x-litellm-response-cost-output"]) == pytest.approx(output_cost)
    assert cost == pytest.approx((input_cost + output_cost) * 1.1)


@respx.mock
async def test_route_a2a_bridge_per_query_agent_reports_no_input_or_output_cost(httpx_transport: None) -> None:
    respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )

    cost, _, logging_obj = await _billed_chat_call(
        _registry_with(_priced_dbx_agent("per-query", PER_QUERY_PRICE)), "per-query"
    )

    headers: Final = cost_headers(response_cost=cost, litellm_logging_obj=logging_obj)
    assert float(headers["x-litellm-response-cost"]) == pytest.approx(0.05)
    assert (headers["x-litellm-response-cost-input"], headers["x-litellm-response-cost-output"]) == (None, None)


@respx.mock
@pytest.mark.parametrize(
    ("config_name", "config"),
    [
        ("cost_margin_config", {"global": {"percentage": "ten"}}),
        ("cost_margin_config", {"global": {"fixed_amount": None}}),
        ("cost_discount_config", {"databricks_agent": "a fifth"}),
        ("cost_discount_config", ["databricks_agent"]),
    ],
)
async def test_route_a2a_bridge_agent_with_a_malformed_cost_config_still_answers(
    httpx_transport: None, monkeypatch: pytest.MonkeyPatch, config_name: str, config: object
) -> None:
    respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    monkeypatch.setattr(litellm, config_name, config)

    cost, response, _ = await _billed_chat_call(
        _registry_with(_priced_dbx_agent("per-query", PER_QUERY_PRICE)), "per-query"
    )

    assert response.choices[0].message.content == "pong"
    assert cost is None


@respx.mock
async def test_route_a2a_bridge_agent_cost_adds_the_calls_additional_response_cost(httpx_transport: None) -> None:
    respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    _, response, logging_obj = await _billed_chat_call(
        _registry_with(_priced_dbx_agent("per-query", PER_QUERY_PRICE)), "per-query"
    )

    logging_obj.model_call_details["additional_response_cost"] = 0.01

    assert logging_obj.response_cost_calculator(result=response) == pytest.approx(0.06)


@respx.mock
async def test_route_a2a_bridge_agent_responses_call_returns_the_callers_metadata_unchanged(
    httpx_transport: None,
) -> None:
    respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    data: Final = {
        "model": "a2a/per-query",
        "input": "ping",
        "metadata": {"session": "s-1"},
        "litellm_metadata": {"user_api_key_hash": "key-hash"},
        "litellm_logging_obj": _proxy_logging_obj(),
    }
    registry: Final = _registry_with(_priced_dbx_agent("per-query", PER_QUERY_PRICE))
    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", registry):
        response: Final = await (await route_a2a_agent_request(data=data, route_type="aresponses"))

    assert response.metadata == {"session": "s-1"}


@respx.mock
async def test_route_a2a_bridge_agent_warns_once_about_prices_agent_billing_ignores(
    httpx_transport: None, caplog: pytest.LogCaptureFixture
) -> None:
    respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    per_second_agent: Final = f"per-second-{uuid.uuid4().hex}"
    per_query_agent: Final = f"per-query-{uuid.uuid4().hex}"
    registry: Final = AgentRegistry()
    registry.register_agent(_priced_dbx_agent(per_second_agent, {"input_cost_per_second": 0.5}))
    registry.register_agent(_priced_dbx_agent(per_query_agent, PER_QUERY_PRICE))

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        await _billed_chat_call(registry, per_second_agent)
        await _billed_chat_call(registry, per_second_agent)
        await _billed_chat_call(registry, per_query_agent)

    warnings: Final = [record.getMessage() for record in caplog.records if "agent billing ignores" in record.getMessage()]
    assert len(warnings) == 1
    assert per_second_agent in warnings[0]
    assert "input_cost_per_second" in warnings[0]


class _SpendRowCapture(CustomLogger):
    def __init__(self, row: asyncio.Future[SpendLogsPayload]) -> None:
        super().__init__()
        self._row = row

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self._row.set_result(
            get_logging_payload(kwargs=kwargs, response_obj=response_obj, start_time=start_time, end_time=end_time)
        )


async def _logged_spend_row(
    agent: AgentResponse, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[SpendLogsPayload, str | None]:
    row: Final[asyncio.Future[SpendLogsPayload]] = asyncio.get_running_loop().create_future()
    capture: Final = _SpendRowCapture(row)
    monkeypatch.setattr(litellm, "callbacks", [capture])
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])
    data: Final = {
        "model": f"a2a/{agent.agent_name}",
        "messages": [{"role": "user", "content": "ping"}],
        "metadata": {"user_api_key_hash": "key-hash"},
        "litellm_logging_obj": _proxy_logging_obj(stream=stream),
        "stream": stream,
    }
    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        response: Final = await (await route_a2a_agent_request(data=data, route_type="acompletion"))
        text: Final = (
            "".join([chunk.choices[0].delta.content or "" async for chunk in response])
            if isinstance(response, CustomStreamWrapper)
            else response.choices[0].message.content
        )
    return await asyncio.wait_for(row, timeout=5), text


def _billed_fields(row: SpendLogsPayload) -> tuple[float, str | None, str | None, int, int]:
    return row["spend"], row["agent_id"], row["model_group"], row["prompt_tokens"], row["completion_tokens"]


@respx.mock
@pytest.mark.parametrize(
    ("price", "expected_spend"), [(PER_QUERY_PRICE, 0.05), (PER_TOKEN_PRICE, 12 * 0.0001 + 34 * 0.0002)]
)
async def test_route_a2a_bridge_agent_streamed_call_logs_the_same_spend_row_as_a_plain_call(
    httpx_transport: None, monkeypatch: pytest.MonkeyPatch, price: Mapping[str, float], expected_spend: float
) -> None:
    respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json={**AGENT_REPLY, "usage": {"input_tokens": 12, "output_tokens": 34}})
    )
    agent: Final = _priced_dbx_agent("priced", price)

    plain_row, plain_text = await _logged_spend_row(agent, False, monkeypatch)
    streamed_row, streamed_text = await _logged_spend_row(agent, True, monkeypatch)

    assert streamed_text == plain_text == "pong"
    assert _billed_fields(streamed_row) == _billed_fields(plain_row)
    assert _billed_fields(streamed_row) == (pytest.approx(expected_spend), "priced-id", "a2a/priced", 12, 34)
