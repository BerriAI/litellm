import os
import sys
import json

import pytest

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.types.llms.openai import ResponsesAPIResponse


@pytest.mark.asyncio
async def test_responses_api_compaction_with_conversation_history():
    """
    1. Request should not fail with compaction param
    """
    litellm._turn_on_debug()
    litellm.set_verbose = True

    response = await litellm.aresponses(
        model="groq/llama-3.3-70b-versatile",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": "Help me debug a production incident.",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "What symptoms are you seeing?",
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": "We are seeing intermittent 502s from one provider path.",
            },
        ],
        context_management=[{"type": "compaction", "compact_threshold": 200000}],
        store=False,
        max_output_tokens=200,
    )

    print("response=", json.dumps(response, indent=4, default=str))

    assert response is not None
    assert isinstance(response, ResponsesAPIResponse)
    assert response.id is not None
    assert response.output is not None
    assert len(response.output) > 0


@pytest.mark.asyncio
async def test_responses_api_compaction_with_trimming():
    """
    2. Request should be trimmed to the compaction threshold
    """
    litellm._turn_on_debug()
    litellm.set_verbose = True

    response = await litellm.aresponses(
        model="groq/llama-3.3-70b-versatile",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": "Help me debug a production incident.",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "What symptoms are you seeing?" * 1000,
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": "We are seeing intermittent 502s from one provider path.",
            },
        ],
        context_management=[{"type": "compaction", "compact_threshold": 100}],
        store=False,
        max_output_tokens=200,
    )

    print("response=", json.dumps(response, indent=4, default=str))

    assert response is not None
    assert isinstance(response, ResponsesAPIResponse)
    assert response.id is not None
    assert response.output is not None
    assert len(response.output) > 0

