"""
Test A2A model routing in proxy.

Maps to: litellm/proxy/agent_endpoints/a2a_routing.py
"""

import asyncio
import base64
import json
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import respx
from fastapi import HTTPException

import litellm
from litellm.a2a_protocol.providers.base import BaseA2AProviderConfig
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy.agent_endpoints.a2a_routing import route_a2a_agent_request
from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
from litellm.proxy.agent_endpoints.databricks_oauth import databricks_app_oauth_token_cache
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.route_llm_request import ProxyModelNotFoundError, route_request
from litellm.responses.streaming_iterator import drain_logging_tasks
from litellm.types.agents import AgentResponse
from litellm.types.utils import CallTypes
from tests.unit.proxy.router_entry_point_stubs import route_entry_points_to_stubs


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
A2A_TASK_REPLY = {
    "jsonrpc": "2.0",
    "id": "1",
    "result": {
        "id": "task-1",
        "kind": "task",
        "status": {"state": "completed"},
        "history": [
            {
                "kind": "message",
                "role": "agent",
                "messageId": "m-1",
                "parts": [{"kind": "text", "text": "pong"}],
            }
        ],
    },
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
    assert sent["metadata"] == proxy_metadata
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
@pytest.mark.parametrize("agent_reply", [A2A_REPLY, A2A_TASK_REPLY], ids=["message", "task"])
async def test_route_a2a_agent_with_another_provider_keeps_calling_its_card_url_over_a2a(
    httpx_transport: None, agent_reply: dict[str, object]
) -> None:
    agent_endpoint: Final = respx.post("https://pydantic-agent.example.com").mock(
        return_value=httpx.Response(200, json=agent_reply)
    )
    data: Final = {"model": "a2a/pydantic-agent", "timeout": 5, "messages": [{"role": "user", "content": "ping"}]}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(_pydantic_agent())):
        response: Final = await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    sent: Final = agent_endpoint.calls.last.request
    sent_body: Final = json.loads(sent.content)
    assert sent_body["method"] == "message/send"
    assert sent_body["params"]["message"]["kind"] == "message"
    assert sent_body["params"]["configuration"] == {"acceptedOutputModes": ["text"], "blocking": True}
    assert sent.extensions["timeout"]["read"] == 5
    assert response.choices[0].message.content == "pong"


@respx.mock
@pytest.mark.parametrize("agent_reply", [A2A_REPLY, A2A_TASK_REPLY], ids=["message", "task"])
async def test_route_a2a_agent_with_another_provider_streams_its_answer_as_openai_chunks(
    httpx_transport: None, agent_reply: dict[str, object]
) -> None:
    respx.post("https://pydantic-agent.example.com").mock(return_value=httpx.Response(200, json=agent_reply))
    data: Final = {"model": "a2a/pydantic-agent", "stream": True, "messages": [{"role": "user", "content": "ping"}]}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(_pydantic_agent())):
        stream: Final = await (await route_a2a_agent_request(data=data, route_type="acompletion"))
        chunks: Final = [chunk async for chunk in stream]

    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "pong"
    assert chunks[-1].choices[0].finish_reason == "stop"


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


@respx.mock
async def test_route_a2a_agent_with_another_provider_sends_its_registered_api_key(
    httpx_transport: None,
) -> None:
    agent_url = "https://pydantic-agent.example.com"
    agent = AgentResponse(
        agent_id="pydantic-agent-id",
        agent_name="pydantic-agent",
        agent_card_params={"url": agent_url},
        litellm_params={
            "custom_llm_provider": "pydantic_ai_agents",
            "model": "pydantic-agent",
            "api_key": "agent-token",
            "headers": {"X-Tenant": "t"},
        },
    )
    agent_endpoint = respx.post(agent_url).mock(return_value=httpx.Response(200, json=A2A_TASK_REPLY))
    data = {"model": "a2a/pydantic-agent", "messages": [{"role": "user", "content": "ping"}]}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        response = await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    sent = agent_endpoint.calls.last.request
    assert sent.headers["Authorization"] == "Bearer agent-token"
    assert sent.headers["X-Tenant"] == "t"
    assert response.choices[0].message.content == "pong"


@respx.mock
async def test_route_a2a_agent_with_another_provider_raises_on_a_json_rpc_error(
    httpx_transport: None,
) -> None:
    agent_url = "https://pydantic-agent.example.com"
    agent = AgentResponse(
        agent_id="pydantic-agent-id",
        agent_name="pydantic-agent",
        agent_card_params={"url": agent_url},
        litellm_params={"custom_llm_provider": "pydantic_ai_agents", "model": "pydantic-agent"},
    )
    respx.post(agent_url).mock(
        return_value=httpx.Response(
            200, json={"jsonrpc": "2.0", "id": "1", "error": {"code": -32602, "message": "Invalid params"}}
        )
    )
    data = {"model": "a2a/pydantic-agent", "messages": [{"role": "user", "content": "ping"}]}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        with pytest.raises(Exception, match="Invalid params"):
            await (await route_a2a_agent_request(data=data, route_type="acompletion"))


def _pydantic_agent(litellm_params: dict[str, object] | None = None) -> AgentResponse:
    return AgentResponse(
        agent_id="pydantic-agent-id",
        agent_name="pydantic-agent",
        agent_card_params={"url": "https://pydantic-agent.example.com"},
        litellm_params={
            "custom_llm_provider": "pydantic_ai_agents",
            "model": "pydantic-agent",
            **(litellm_params or {}),
        },
    )


class _SuccessRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.responses: Final[list[object]] = []  # mutable-ok: records what each success callback received
        self.failures: Final[list[object]] = []  # mutable-ok: records what each failure callback received

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.responses.append(response_obj)

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.failures.append(kwargs.get("exception"))


def _request_logging_obj(stream: bool, recorder: _SuccessRecorder) -> LiteLLMLogging:
    logging_obj: Final = LiteLLMLogging(
        litellm_call_id=str(uuid.uuid4()),
        call_type=CallTypes.acompletion.value,
        model="a2a/pydantic-agent",
        messages=[{"role": "user", "content": "ping"}],
        function_id=str(uuid.uuid4()),
        stream=stream,
        start_time=datetime(2026, 1, 1),  # noqa: DTZ001  # Logging subtracts it from naive end times
        dynamic_async_success_callbacks=[recorder],
        dynamic_async_failure_callbacks=[recorder],
    )
    logging_obj.defer_async_logging = not stream
    return logging_obj


async def _settle_logging() -> None:
    await drain_logging_tasks()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)


@respx.mock
@pytest.mark.parametrize("guardrail_blocked", [False, True])
async def test_route_a2a_agent_with_another_provider_logs_success_only_after_the_post_call_guardrails(
    httpx_transport: None, guardrail_blocked: bool
) -> None:
    respx.post("https://pydantic-agent.example.com").mock(return_value=httpx.Response(200, json=A2A_TASK_REPLY))
    recorder: Final = _SuccessRecorder()
    logging_obj: Final = _request_logging_obj(stream=False, recorder=recorder)
    data: Final = {
        "model": "a2a/pydantic-agent",
        "messages": [{"role": "user", "content": "ping"}],
        "litellm_logging_obj": logging_obj,
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(_pydantic_agent())):
        response: Final = await (await route_a2a_agent_request(data=data, route_type="acompletion"))
    await _settle_logging()
    logged_before_guardrails: Final = tuple(recorder.responses)

    ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(
        logging_obj=logging_obj, exception_raised=guardrail_blocked
    )
    await _settle_logging()

    assert logged_before_guardrails == ()
    assert recorder.responses == ([] if guardrail_blocked else [response])


@respx.mock
async def test_route_a2a_agent_with_another_provider_logs_the_whole_streamed_answer_once(
    httpx_transport: None,
) -> None:
    respx.post("https://pydantic-agent.example.com").mock(return_value=httpx.Response(200, json=A2A_TASK_REPLY))
    recorder: Final = _SuccessRecorder()
    data: Final = {
        "model": "a2a/pydantic-agent",
        "stream": True,
        "messages": [{"role": "user", "content": "ping"}],
        "litellm_logging_obj": _request_logging_obj(stream=True, recorder=recorder),
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(_pydantic_agent())):
        stream: Final = await (await route_a2a_agent_request(data=data, route_type="acompletion"))
        chunks: Final = [chunk async for chunk in stream]
    await _settle_logging()

    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "pong"
    assert [response.choices[0].message.content for response in recorder.responses] == ["pong"]


@respx.mock
async def test_route_a2a_agent_with_another_provider_fails_a_stream_before_it_opens_on_a_json_rpc_error(
    httpx_transport: None,
) -> None:
    respx.post("https://pydantic-agent.example.com").mock(
        return_value=httpx.Response(
            200, json={"jsonrpc": "2.0", "id": "1", "error": {"code": -32602, "message": "Invalid params"}}
        )
    )
    data: Final = {"model": "a2a/pydantic-agent", "stream": True, "messages": [{"role": "user", "content": "ping"}]}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(_pydantic_agent())):
        with pytest.raises(litellm.InternalServerError, match="Invalid params"):
            await (await route_a2a_agent_request(data=data, route_type="acompletion"))


async def test_route_a2a_agent_with_another_provider_and_no_url_is_not_found() -> None:
    agent: Final = AgentResponse(
        agent_id="pydantic-agent-id",
        agent_name="pydantic-agent",
        agent_card_params={},
        litellm_params={"custom_llm_provider": "pydantic_ai_agents", "model": "pydantic-agent"},
    )
    data: Final = {"model": "a2a/pydantic-agent", "messages": [{"role": "user", "content": "ping"}]}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(agent)):
        with pytest.raises(ProxyModelNotFoundError):
            await (await route_a2a_agent_request(data=data, route_type="acompletion"))


@respx.mock
async def test_route_a2a_agent_with_another_provider_logs_a_failed_call_with_the_agents_status(
    httpx_transport: None,
) -> None:
    respx.post("https://pydantic-agent.example.com").mock(return_value=httpx.Response(503, json={"detail": "busy"}))
    recorder: Final = _SuccessRecorder()
    data: Final = {
        "model": "a2a/pydantic-agent",
        "messages": [{"role": "user", "content": "ping"}],
        "litellm_logging_obj": _request_logging_obj(stream=False, recorder=recorder),
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(_pydantic_agent())):
        with pytest.raises(litellm.ServiceUnavailableError) as raised:
            await (await route_a2a_agent_request(data=data, route_type="acompletion"))
    await _settle_logging()

    assert raised.value.status_code == 503
    assert recorder.failures == [raised.value]
    assert recorder.responses == []


@respx.mock
async def test_route_a2a_agent_with_another_provider_reports_estimated_usage(httpx_transport: None) -> None:
    respx.post("https://pydantic-agent.example.com").mock(return_value=httpx.Response(200, json=A2A_TASK_REPLY))
    data: Final = {"model": "a2a/pydantic-agent", "messages": [{"role": "user", "content": "ping"}]}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(_pydantic_agent())):
        response: Final = await (await route_a2a_agent_request(data=data, route_type="acompletion"))

    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


@respx.mock
async def test_route_a2a_agent_with_another_provider_answers_the_responses_api(httpx_transport: None) -> None:
    agent_endpoint: Final = respx.post("https://pydantic-agent.example.com").mock(
        return_value=httpx.Response(200, json=A2A_TASK_REPLY)
    )
    data: Final = {"model": "a2a/pydantic-agent", "input": "ping"}

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(_pydantic_agent())):
        response: Final = await (await route_a2a_agent_request(data=data, route_type="aresponses"))

    sent_parts: Final = json.loads(agent_endpoint.calls.last.request.content)["params"]["message"]["parts"]
    assert "ping" in sent_parts[0]["text"]
    assert response.output_text == "pong"


class _RecordingAgentProvider(BaseA2AProviderConfig):
    def __init__(self, events: Sequence[Mapping[str, object]]) -> None:
        self.events: Final = events
        self.calls: Final[list[str]] = []  # mutable-ok: records which agent method each request used
        self.stream_closed: Final[list[bool]] = []  # mutable-ok: records whether the agent stream was closed

    async def handle_non_streaming(
        self, request_id: str, params: dict[str, object], api_base: str | None = None, **kwargs: object
    ) -> dict[str, object]:
        self.calls.append("send")
        return dict(self.events[-1])

    async def handle_streaming(
        self, request_id: str, params: dict[str, object], api_base: str | None = None, **kwargs: object
    ) -> AsyncIterator[dict[str, object]]:
        self.calls.append("stream")
        try:
            for event in self.events:
                yield dict(event)
        finally:
            self.stream_closed.append(True)


def _agent_with_card(capabilities: Mapping[str, object]) -> AgentResponse:
    return AgentResponse(
        agent_id="pydantic-agent-id",
        agent_name="pydantic-agent",
        agent_card_params={"url": "https://pydantic-agent.example.com", "capabilities": capabilities},
        litellm_params={"custom_llm_provider": "pydantic_ai_agents", "model": "pydantic-agent"},
    )


_PONG_EVENT: Final = {"jsonrpc": "2.0", "id": "1", "result": A2A_REPLY["result"]}


async def test_route_a2a_agent_with_another_provider_fakes_the_stream_for_an_agent_that_cannot_stream() -> None:
    provider: Final = _RecordingAgentProvider([_PONG_EVENT])
    data: Final = {"model": "a2a/pydantic-agent", "stream": True, "messages": [{"role": "user", "content": "ping"}]}

    with (
        patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            _registry_with(_agent_with_card({"streaming": False})),
        ),
        patch(
            "litellm.a2a_protocol.providers.config_manager.A2AProviderConfigManager.get_provider_config",
            return_value=provider,
        ),
    ):
        stream: Final = await (await route_a2a_agent_request(data=data, route_type="acompletion"))
        chunks: Final = [chunk async for chunk in stream]

    assert provider.calls == ["send"]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "pong"


async def test_route_a2a_agent_with_another_provider_closes_the_agent_stream_when_the_caller_stops_reading() -> None:
    provider: Final = _RecordingAgentProvider([_PONG_EVENT, _PONG_EVENT, _PONG_EVENT])
    data: Final = {"model": "a2a/pydantic-agent", "stream": True, "messages": [{"role": "user", "content": "ping"}]}

    with (
        patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            _registry_with(_agent_with_card({"streaming": True})),
        ),
        patch(
            "litellm.a2a_protocol.providers.config_manager.A2AProviderConfigManager.get_provider_config",
            return_value=provider,
        ),
    ):
        stream: Final = await (await route_a2a_agent_request(data=data, route_type="acompletion"))
        first: Final = await anext(aiter(stream))
        await stream.aclose()

    assert first.choices[0].delta.content == "pong"
    assert (provider.calls, provider.stream_closed) == (["stream"], [True])


@respx.mock
async def test_route_a2a_agent_with_another_provider_asks_a_streamed_agent_to_wait_for_its_answer(
    httpx_transport: None,
) -> None:
    agent_endpoint: Final = respx.post("https://pydantic-agent.example.com").mock(
        return_value=httpx.Response(200, json=A2A_TASK_REPLY)
    )
    data: Final = {
        "model": "a2a/pydantic-agent",
        "stream": True,
        "messages": [{"role": "user", "content": "ping"}],
    }

    with patch("litellm.proxy.agent_endpoints.agent_registry.global_agent_registry", _registry_with(_pydantic_agent())):
        stream: Final = await (await route_a2a_agent_request(data=data, route_type="acompletion"))
        chunks: Final = [chunk async for chunk in stream]

    sent: Final = agent_endpoint.calls.last.request
    assert json.loads(sent.content)["params"]["configuration"] == {"acceptedOutputModes": ["text"], "blocking": True}
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "pong"
