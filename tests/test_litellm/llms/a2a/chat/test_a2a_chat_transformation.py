"""Tests for litellm/llms/a2a/chat/transformation.py response transform."""

from unittest.mock import MagicMock

import pytest

from litellm.llms.a2a.chat.transformation import A2A_PROTOCOL_VERSION_PARAM, A2AConfig
from litellm.types.utils import ModelResponse

_MESSAGES = [{"role": "user", "content": "hi there agent"}]


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


def _v1_raw_response(text: str) -> MagicMock:
    """A2A 1.0 (a2a-sdk 1.x) protobuf-JSON send response: no ``kind`` fields."""
    raw = MagicMock()
    raw.status_code = 200
    raw.headers = {}
    raw.json.return_value = {
        "jsonrpc": "2.0",
        "id": "resp-1",
        "result": {"message": {"messageId": "m1", "role": "ROLE_AGENT", "parts": [{"text": text}]}},
    }
    return raw


def test_transform_request_defaults_to_0_3():
    """Without a pinned protocol version the bridge emits the legacy 0.3 method."""
    request = A2AConfig().transform_request("a2a/agent", _MESSAGES, {}, {}, {})
    assert request["method"] == "message/send"
    assert request["params"]["message"]["parts"][0]["kind"] == "text"


def test_transform_request_v1_uses_send_message():
    """Regression for #32609: a 1.0-pinned agent must get the 1.0 method and the
    protobuf-JSON envelope (uppercase role enum, parts without ``kind``)."""
    pytest.importorskip("a2a")
    request = A2AConfig().transform_request(
        "a2a/agent", _MESSAGES, {A2A_PROTOCOL_VERSION_PARAM: "1.0"}, {}, {}
    )
    assert request["method"] == "SendMessage"
    message = request["params"]["message"]
    assert message["role"] == "ROLE_USER"
    assert message["parts"] == [{"text": "user: hi there agent"}]
    assert "kind" not in message["parts"][0]


def test_transform_request_v1_streaming_uses_send_streaming_message():
    pytest.importorskip("a2a")
    request = A2AConfig().transform_request(
        "a2a/agent", _MESSAGES, {A2A_PROTOCOL_VERSION_PARAM: "1.0", "stream": True}, {}, {}
    )
    assert request["method"] == "SendStreamingMessage"


def test_transform_request_v1_from_litellm_params():
    """The pinned version may arrive via litellm_params (direct SDK usage)."""
    pytest.importorskip("a2a")
    request = A2AConfig().transform_request(
        "a2a/agent", _MESSAGES, {}, {A2A_PROTOCOL_VERSION_PARAM: "1.0"}, {}
    )
    assert request["method"] == "SendMessage"


def test_transform_request_reads_pinned_version_from_registry():
    """Regression for #32609: the completion bridge strips the ``a2a/`` prefix and
    injects only api_base, so transform_request must resolve the pinned version from
    the registry (keyed by the bare agent name) rather than defaulting to 0.3."""
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
        from litellm.types.agents import AgentResponse
    except ImportError:
        import pytest

        pytest.skip("Registry not available (not in proxy context)")

    agent = AgentResponse(
        agent_id="v1-id",
        agent_name="v1-agent",
        agent_card_params={"url": "http://agent.example.com:9999", "protocolVersion": "1.0"},
        litellm_params={},
    )
    original_agents = global_agent_registry.agent_list.copy()
    global_agent_registry.register_agent(agent)
    try:
        request = A2AConfig().transform_request("v1-agent", _MESSAGES, {}, {}, {})
        assert request["method"] == "SendMessage"
        assert "kind" not in request["params"]["message"]["parts"][0]
    finally:
        global_agent_registry.agent_list = original_agents


def test_transform_request_registry_agent_without_pinned_version_uses_0_3():
    """An agent whose card omits ``protocolVersion`` keeps the legacy 0.3 method."""
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
        from litellm.types.agents import AgentResponse
    except ImportError:
        import pytest

        pytest.skip("Registry not available (not in proxy context)")

    agent = AgentResponse(
        agent_id="legacy-id",
        agent_name="legacy-agent",
        agent_card_params={"url": "http://agent.example.com:9999"},
        litellm_params={},
    )
    original_agents = global_agent_registry.agent_list.copy()
    global_agent_registry.register_agent(agent)
    try:
        request = A2AConfig().transform_request("legacy-agent", _MESSAGES, {}, {}, {})
        assert request["method"] == "message/send"
    finally:
        global_agent_registry.agent_list = original_agents


def test_transform_response_extracts_text_from_v1_message():
    """Regression for #32609: text must be extracted from a 1.0 protobuf-JSON
    message result whose parts carry no ``kind`` field."""
    result = A2AConfig().transform_response(
        model="a2a/test-agent",
        raw_response=_v1_raw_response("hello from a 1.0 agent"),
        model_response=ModelResponse(),
        logging_obj=MagicMock(),
        request_data={},
        messages=_MESSAGES,
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert result.choices[0].message.content == "hello from a 1.0 agent"


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
