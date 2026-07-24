"""Tests for litellm/llms/a2a/chat/transformation.py response transform."""

from unittest.mock import MagicMock

import pytest

from litellm.llms.a2a.chat.transformation import A2AConfig
from litellm.types.utils import ModelResponse


def _raw_response(text: str) -> MagicMock:
    raw = MagicMock()
    raw.status_code = 200
    raw.headers = {}
    raw.json.return_value = {
        "jsonrpc": "2.0",
        "id": "resp-1",
        "result": {
            "kind": "message",
            "parts": [{"kind": "text", "text": text}],
        },
    }
    return raw


def test_transform_response_sets_usage():
    """Regression: A2AConfig.transform_response must populate usage so per-token
    pricing computes real cost and callers don't get usage 0/0/0."""
    result = A2AConfig().transform_response(
        model="a2a/test-agent",
        raw_response=_raw_response("hello from the agent"),
        model_response=ModelResponse(),
        logging_obj=MagicMock(),
        request_data={},
        messages=[{"role": "user", "content": "hi there agent"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert result.usage is not None
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.total_tokens == (result.usage.prompt_tokens + result.usage.completion_tokens)


@pytest.fixture
def registered_agent():
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
    from litellm.types.agents import AgentResponse

    def _register(**kwargs) -> AgentResponse:
        agent = AgentResponse(
            agent_id="static-headers-agent-id",
            agent_name="static-headers-agent",
            agent_card_params={"url": "http://agent.example.com:9999"},
            **kwargs,
        )
        global_agent_registry.register_agent(agent)
        return agent

    original = list(global_agent_registry.agent_list)
    try:
        yield _register
    finally:
        global_agent_registry.agent_list = original


@pytest.mark.parametrize("model", ["a2a/static-headers-agent", "static-headers-agent"])
def test_resolve_agent_config_applies_static_headers(registered_agent, model):
    """Regression for #32608: the /chat/completions bridge must forward an agent's
    static_headers, consistent with the native /a2a/{agent_id} route.

    get_llm_provider strips the "a2a/" prefix before dispatch, so the resolver
    must find the agent whether the model still carries the prefix or not."""
    registered_agent(static_headers={"x-api-key": "secret-value"})

    api_base, _, headers = A2AConfig.resolve_agent_config_from_registry(
        model=model,
        api_base=None,
        api_key=None,
        headers=None,
        optional_params={},
    )

    assert headers == {"x-api-key": "secret-value"}
    assert api_base == "http://agent.example.com:9999"


def test_static_headers_win_over_request_headers_case_insensitive(registered_agent):
    """static_headers must override a request-supplied header of the same name
    (case-insensitively), mirroring merge_agent_headers on the native route."""
    registered_agent(static_headers={"X-Api-Key": "admin-secret"})

    _, _, headers = A2AConfig.resolve_agent_config_from_registry(
        model="a2a/static-headers-agent",
        api_base=None,
        api_key=None,
        headers={"x-api-key": "caller-supplied", "x-other": "keep-me"},
        optional_params={},
    )

    assert headers == {"X-Api-Key": "admin-secret", "x-other": "keep-me"}


def test_no_static_headers_leaves_request_headers_untouched(registered_agent):
    registered_agent(litellm_params=None)

    _, _, headers = A2AConfig.resolve_agent_config_from_registry(
        model="a2a/static-headers-agent",
        api_base=None,
        api_key=None,
        headers={"x-caller": "value"},
        optional_params={},
    )

    assert headers == {"x-caller": "value"}
