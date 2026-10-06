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

import asyncio
import importlib
import os

import pytest

import litellm
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from tests.fake_openai_endpoint import ensure_fake_openai_endpoint


@pytest.mark.asyncio
async def test_langgraph_acompletion_non_streaming():
    """
    Test non-streaming acompletion call to LangGraph server.
    Uses the /runs/wait endpoint for synchronous response.
    """
    api_base = os.environ.get("LANGGRAPH_API_BASE", "http://localhost:2024")

    try:
        response = await litellm.acompletion(
            model="langgraph/agent",
            messages=[{"role": "user", "content": "What is 25 * 4?"}],
            api_base=api_base,
            stream=False,
        )

        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message is not None
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    except Exception as e:
        pytest.skip(f"LangGraph server not available: {e}")


@pytest.mark.asyncio
async def test_langgraph_acompletion_streaming():
    """
    Test streaming acompletion call to LangGraph server.
    Uses the /runs/stream endpoint with stream_mode="messages-tuple".
    """
    api_base = os.environ.get("LANGGRAPH_API_BASE", "http://localhost:2024")

    try:
        response = await litellm.acompletion(
            model="langgraph/agent",
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
            api_base=api_base,
            stream=True,
        )

        full_content = ""
        chunk_count = 0

        async for chunk in response:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content

        assert chunk_count > 0, "Should receive at least one chunk"

    except Exception as e:
        pytest.skip(f"LangGraph server not available: {e}")


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


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def fake_openai_endpoint():
    ensure_fake_openai_endpoint()
    yield


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown(event_loop):
    import litellm

    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    asyncio.set_event_loop(event_loop)
    yield
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}
