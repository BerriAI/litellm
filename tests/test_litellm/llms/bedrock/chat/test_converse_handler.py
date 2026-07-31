import json
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.bedrock.chat.converse_handler import make_sync_call


def _tool_use_body():
    return {
        "metrics": {"latencyMs": 100.0},
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "json_tool_call",
                            "input": {"result": {"name": "Claude"}},
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 8, "outputTokens": 3, "totalTokens": 11},
    }


def _sync_fake_stream_text(**extra):
    body = _tool_use_body()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = body
    response.text = json.dumps(body)
    client = MagicMock()
    client.post = MagicMock(return_value=response)

    completion_stream = make_sync_call(
        client=client,
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse",
        headers={},
        data="{}",
        model="anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[],
        logging_obj=MagicMock(),
        fake_stream=True,
        json_mode=True,
        **extra,
    )
    return "".join(chunk["text"] for chunk in completion_stream)


def test_make_sync_call_unwraps_json_object_wrapper_on_fake_stream():
    """Sync streaming must strip the json_object wrapper, like the async path.

    The conversion from tool call to content happens in MockResponseIterator, so
    the unwrap key has to reach it; otherwise a non-async caller streaming with
    `response_format={"type": "json_object"}` receives the raw
    `{"result": {...}}` wrapper instead of the object they asked for.
    """
    text = _sync_fake_stream_text(json_object_unwrap_key="result")
    assert json.loads(text) == {"name": "Claude"}


def test_make_sync_call_leaves_tool_arguments_alone_without_unwrap_key():
    """A caller-supplied json_schema is never unwrapped."""
    text = _sync_fake_stream_text()
    assert json.loads(text) == {"result": {"name": "Claude"}}
