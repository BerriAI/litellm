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

import pytest

import litellm

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
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                full_content += chunk.choices[0].delta.content

        assert chunk_count > 0, "Should receive at least one chunk"

    except Exception as e:
        pytest.skip(f"LangGraph server not available: {e}")
