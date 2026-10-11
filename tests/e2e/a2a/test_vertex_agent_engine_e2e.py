"""Vertex AI Agent Engine end to end via litellm.acompletion, real Vertex creds."""

import pytest

import litellm

pytestmark = pytest.mark.e2e

VERTEX_AGENT_RESOURCE_NAME = (
    "projects/1060139831167/locations/us-central1/reasoningEngines/8263861224643493888"
)


@pytest.mark.asyncio
async def test_vertex_agent_engine_non_streaming():
    """Non-streaming completion against a hosted Reasoning Engine returns text."""
    response = await litellm.acompletion(
        model=f"vertex_ai/agent_engine/{VERTEX_AGENT_RESOURCE_NAME}",
        messages=[{"role": "user", "content": "Hello! What can you do?"}],
        stream=False,
    )

    assert response is not None
    assert len(response.choices) > 0
    content = response.choices[0].message.content
    assert content is not None
    assert isinstance(content, str)
    assert len(content) > 0
    assert response.choices[0].finish_reason is not None


@pytest.mark.asyncio
async def test_vertex_agent_engine_streaming():
    """Streaming completion against a hosted Reasoning Engine yields text chunks."""
    response = await litellm.acompletion(
        model=f"vertex_ai/agent_engine/{VERTEX_AGENT_RESOURCE_NAME}",
        messages=[{"role": "user", "content": "Hello! What can you do?"}],
        stream=True,
    )

    chunks = []
    full_content = ""
    async for chunk in response:
        chunks.append(chunk)
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                full_content += delta.content

    assert len(chunks) > 0
    assert len(full_content) > 0
