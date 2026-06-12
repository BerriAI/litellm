"""
Tests for LangGraph provider integration.

These tests require a LangGraph server running locally on port 2024.
To start a LangGraph server, follow the LangGraph documentation.

Example test server curl commands:
Streaming:
  curl -s --request POST \
    --url "http://localhost:2024/runs/stream" \
    --header 'Content-Type: application/json' \
    --data '{"assistant_id": "agent", "input": {"messages": [{"role": "human", "content": "What is 25 * 4?"}]}, "stream_mode": "messages-tuple"}'

Non-streaming:
  curl -s --request POST \
    --url "http://localhost:2024/runs/wait" \
    --header 'Content-Type: application/json' \
    --data '{"assistant_id": "agent", "input": {"messages": [{"role": "human", "content": "What is 25 * 4?"}]}}'
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))




def test_langgraph_config_get_complete_url():
    """
    Test that LangGraphConfig correctly generates URLs for streaming and non-streaming.
    """
    from litellm.llms.langgraph.chat.transformation import LangGraphConfig

    config = LangGraphConfig()

    non_streaming_url = config.get_complete_url(
        api_base="http://localhost:2024",
        api_key=None,
        model="agent",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert non_streaming_url == "http://localhost:2024/runs/wait"

    streaming_url = config.get_complete_url(
        api_base="http://localhost:2024",
        api_key=None,
        model="agent",
        optional_params={},
        litellm_params={},
        stream=True,
    )
    assert streaming_url == "http://localhost:2024/runs/stream"


def test_langgraph_config_transform_request():
    """
    Test that LangGraphConfig correctly transforms requests.
    """
    from litellm.llms.langgraph.chat.transformation import LangGraphConfig

    config = LangGraphConfig()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    request = config.transform_request(
        model="langgraph/agent",
        messages=messages,
        optional_params={},
        litellm_params={"stream": False},
        headers={},
    )

    assert request["assistant_id"] == "agent"
    assert "input" in request
    assert "messages" in request["input"]
    assert len(request["input"]["messages"]) == 2
    assert request["input"]["messages"][0]["role"] == "system"
    assert request["input"]["messages"][1]["role"] == "human"

    streaming_request = config.transform_request(
        model="langgraph/agent",
        messages=messages,
        optional_params={},
        litellm_params={"stream": True},
        headers={},
    )

    assert streaming_request["stream_mode"] == "messages-tuple"


def test_langgraph_provider_detection():
    """
    Test that the langgraph provider is correctly detected from model name.
    """
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="langgraph/agent",
        api_base="http://localhost:2024",
    )

    assert provider == "langgraph"
    assert model == "agent"
