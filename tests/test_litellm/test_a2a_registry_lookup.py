"""
Test A2A provider registry lookup functionality.

Maps to: litellm/llms/a2a/chat/transformation.py
"""

import json
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.a2a.chat.transformation import A2AConfig


def test_resolve_agent_config_from_registry_static_method():
    """Test the static helper method for registry resolution"""

    # Test 1: Unregistered agent name keeps the explicit config
    api_base, api_key, headers = A2AConfig.resolve_agent_config_from_registry(
        agent_name="not-registered",
        api_base="http://test.com",
        api_key=None,
        headers=None,
        optional_params={},
    )
    assert api_base == "http://test.com"
    assert api_key is None

    # Test 2: All params provided - should not lookup registry
    api_base, api_key, headers = A2AConfig.resolve_agent_config_from_registry(
        agent_name="test-agent",
        api_base="http://explicit.com",
        api_key="explicit-key",
        headers={"X-Test": "value"},
        optional_params={},
    )
    assert api_base == "http://explicit.com"
    assert api_key == "explicit-key"


def test_a2a_registry_integration():
    """A chat call for a registered agent must post to the registered url with the registered key as the
    bearer even though completion() strips the a2a/ prefix before the lookup runs."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
    from litellm.types.agents import AgentResponse

    test_agent = AgentResponse(
        agent_id="test-id",
        agent_name="test-agent",
        agent_card_params={"url": "http://registry-url.example.com:9999"},
        litellm_params={"api_key": "registry-key", "headers": {"X-Agent": "static"}},
    )
    client = HTTPHandler()
    agent_reply = httpx.Response(
        200,
        json={"jsonrpc": "2.0", "id": "1", "result": {"kind": "message", "parts": [{"kind": "text", "text": "4"}]}},
    )
    original_agents = global_agent_registry.agent_list.copy()
    global_agent_registry.register_agent(test_agent)

    try:
        with patch.object(client, "post", return_value=agent_reply) as post:  # test-quality-ok: injected client
            response = litellm.completion(
                model="a2a/test-agent", messages=[{"role": "user", "content": "What is 2+2?"}], client=client
            )
    finally:
        global_agent_registry.agent_list = original_agents

    assert response.choices[0].message.content == "4"
    assert post.call_args.kwargs["url"] == "http://registry-url.example.com:9999"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer registry-key"
    assert post.call_args.kwargs["headers"]["X-Agent"] == "static"


def _foundry_card_stored_through_the_agents_api() -> dict:
    from litellm.proxy.a2a.agent_card import merge_agent_card

    return merge_agent_card(
        {"name": "Foundry", "url": "https://foundry.example.com/a2a", "capabilities": {"streaming": False}},
        proxy_url="http://localhost:4000/a2a/foundry-agent",
        proxy_base_url="http://localhost:4000",
    )


@pytest.mark.parametrize(
    "agent_card_params",
    [
        {"url": "https://foundry.example.com/a2a", "capabilities": {"streaming": False}},
        _foundry_card_stored_through_the_agents_api(),
    ],
    ids=["card registered verbatim from config.yaml", "card stored through POST /v1/agents"],
)
def test_streaming_chat_to_an_agent_whose_card_declines_streaming_uses_a_blocking_send(agent_card_params: dict):
    """Microsoft Foundry agents publish `capabilities.streaming: false` and answer message/stream with a
    JSON-RPC error. A streaming chat call to such an agent must post a blocking message/send and hand the
    caller the answer as a stream, whether the card was registered verbatim from config.yaml or stored
    through POST /v1/agents, which keeps only truthy capabilities and so drops the `false` itself."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
    from litellm.types.agents import AgentResponse

    foundry_agent = AgentResponse(
        agent_id="foundry-id",
        agent_name="foundry-agent",
        agent_card_params=agent_card_params,
        litellm_params={"api_key": "registry-key"},
    )
    client = HTTPHandler()
    agent_reply = httpx.Response(
        200,
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "result": {
                "kind": "task",
                "status": {"state": "completed"},
                "artifacts": [{"parts": [{"kind": "text", "text": "4"}]}],
            },
        },
    )
    original_agents = global_agent_registry.agent_list.copy()
    global_agent_registry.register_agent(foundry_agent)

    try:
        with patch.object(client, "post", return_value=agent_reply) as post:  # test-quality-ok: injected client
            chunks = list(
                litellm.completion(
                    model="a2a/foundry-agent",
                    messages=[{"role": "user", "content": "What is 2+2?"}],
                    stream=True,
                    client=client,
                )
            )
    finally:
        global_agent_registry.agent_list = original_agents

    posted = json.loads(post.call_args.kwargs["data"])
    assert posted["method"] == "message/send"
    assert posted["params"]["configuration"] == {"blocking": True}
    assert post.call_args.kwargs.get("stream", False) is False
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "4"
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.parametrize(
    "agent_card_params",
    [
        {"url": "https://agent.example.com/a2a"},
        {"url": "https://agent.example.com/a2a", "capabilities": {"streaming": True}},
    ],
    ids=["card without a capabilities block", "card says streaming true"],
)
def test_registry_lookup_leaves_streaming_alone_when_the_card_does_not_decline_it(agent_card_params: dict):
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
    from litellm.types.agents import AgentResponse

    silent_agent = AgentResponse(
        agent_id="silent-id",
        agent_name="silent-agent",
        agent_card_params=agent_card_params,
        litellm_params={"api_key": "registry-key"},
    )
    original_agents = global_agent_registry.agent_list.copy()
    global_agent_registry.register_agent(silent_agent)
    optional_params: dict = {"stream": True}

    try:
        A2AConfig.resolve_agent_config_from_registry(
            agent_name="silent-agent", api_base=None, api_key=None, headers=None, optional_params=optional_params
        )
    finally:
        global_agent_registry.agent_list = original_agents

    assert optional_params == {"stream": True}


def test_registry_entra_agent_authenticates_with_the_entra_token_and_keeps_its_secrets_private():
    """An agent registered with Entra credentials has no api_key, so the chat route must resolve the
    bearer from those credentials, and the credential fields must not ride along into optional_params
    where they would reach spend logs and callbacks."""
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
    from litellm.types.agents import AgentResponse

    entra_agent = AgentResponse(
        agent_id="entra-id",
        agent_name="entra-agent",
        agent_card_params={"url": "https://foundry.example.com/a2a"},
        litellm_params={"azure_ad_token": "entra-token", "tenant_id": "tenant", "timeout": 30},
    )
    original_agents = global_agent_registry.agent_list.copy()
    global_agent_registry.register_agent(entra_agent)
    optional_params: dict = {}

    try:
        api_base, api_key, _headers = A2AConfig.resolve_agent_config_from_registry(
            agent_name="entra-agent",
            api_base=None,
            api_key=None,
            headers=None,
            optional_params=optional_params,
        )
    finally:
        global_agent_registry.agent_list = original_agents

    assert api_base == "https://foundry.example.com/a2a"
    assert api_key == "entra-token"
    assert optional_params == {"timeout": 30}


def test_registry_entra_agent_with_an_unresolvable_credential_fails_the_chat_call(monkeypatch):
    """The chat route mints the Foundry bearer from the registered credentials; when they resolve to
    nothing the caller must get the credential error instead of an unauthenticated backend call."""
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
    from litellm.types.agents import AgentResponse

    monkeypatch.delenv("LITELLM_TEST_UNSET_FOUNDRY_TOKEN", raising=False)
    entra_agent = AgentResponse(
        agent_id="entra-unset-id",
        agent_name="entra-unset-agent",
        agent_card_params={"url": "https://foundry.example.com/a2a"},
        litellm_params={"azure_ad_token": "os.environ/LITELLM_TEST_UNSET_FOUNDRY_TOKEN"},
    )
    original_agents = global_agent_registry.agent_list.copy()
    global_agent_registry.register_agent(entra_agent)

    try:
        with pytest.raises(litellm.APIConnectionError, match="client_secret"):
            litellm.completion(model="a2a/entra-unset-agent", messages=[{"role": "user", "content": "hi"}])
    finally:
        global_agent_registry.agent_list = original_agents
