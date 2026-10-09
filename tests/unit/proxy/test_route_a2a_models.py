"""
Test A2A model routing in proxy.

Maps to: litellm/proxy/agent_endpoints/a2a_routing.py
"""

import base64
import json
from collections.abc import Iterator
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import respx

import litellm
from litellm.proxy.agent_endpoints.a2a_routing import route_a2a_agent_request
from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
from litellm.proxy.agent_endpoints.databricks_oauth import databricks_app_oauth_token_cache
from litellm.proxy.route_llm_request import route_request
from litellm.types.agents import AgentResponse


@pytest.mark.asyncio
async def test_route_a2a_model_bypasses_router():
    """Test that a2a/ prefixed models bypass router and go directly to litellm with api_base"""

    # Mock data for chat completion with a2a model
    data = {
        "model": "a2a/test-agent",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Mock router that doesn't have the a2a model
    mock_router = Mock()
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
    mock_router = Mock()
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
    mock_router = Mock()
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
async def test_route_a2a_bridge_agent_merges_client_and_agent_extra_headers(httpx_transport: None) -> None:
    agent = AgentResponse(
        agent_id="dbx-agent-id",
        agent_name="dbx-agent",
        agent_card_params={"url": WORKSPACE},
        litellm_params={
            "custom_llm_provider": "databricks_agent",
            "model": "my-agent",
            "api_base": WORKSPACE,
            "api_key": "pat-1",
            "extra_headers": {"X-Agent": "agent", "X-Shared": "from-agent"},
        },
    )
    endpoint = respx.post(f"{WORKSPACE}/serving-endpoints/my-agent/invocations").mock(
        return_value=httpx.Response(200, json=AGENT_REPLY)
    )
    data = {
        "model": "a2a/dbx-agent",
        "messages": [{"role": "user", "content": "ping"}],
        "extra_headers": {"X-Trace": "trace-1", "x-shared": "from-client"},
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    sent = endpoint.calls.last.request
    assert sent.headers["X-Trace"] == "trace-1"
    assert sent.headers["X-Agent"] == "agent"
    assert sent.headers.get_list("X-Shared") == ["from-agent"]
    assert sent.headers["Authorization"] == "Bearer pat-1"


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
        with pytest.raises(ValueError, match="missing required field"):
            await route_a2a_agent_request(data=data, route_type="acompletion")

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
async def test_route_a2a_url_agent_with_databricks_oauth_sends_the_minted_token(httpx_transport: None) -> None:
    agent_url = "https://my-a2a-app.azure.databricksapps.com"
    agent = AgentResponse(
        agent_id="a2a-app-id",
        agent_name="a2a-app",
        agent_card_params={"url": agent_url},
        litellm_params={"databricks_oauth": {"client_id": "sp", "client_secret": "s", "workspace_url": WORKSPACE}},
    )
    _token_route(WORKSPACE)
    a2a_reply = {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {"kind": "message", "role": "agent", "messageId": "m-1", "parts": [{"kind": "text", "text": "pong"}]},
    }
    app = respx.post(agent_url).mock(return_value=httpx.Response(200, json=a2a_reply))
    data = {"model": "a2a/a2a-app", "messages": [{"role": "user", "content": "ping"}]}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        response = await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    sent = app.calls.last.request
    assert sent.headers["Authorization"] == "Bearer minted-oauth"
    assert json.loads(sent.content)["method"] == "message/send"
    assert response.choices[0].message.content == "pong"
