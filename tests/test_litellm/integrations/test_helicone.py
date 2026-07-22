import os
import sys
import types

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.helicone import HeliconeLogger


def _claude_mapping(messages, response_obj):
    logger = HeliconeLogger.__new__(HeliconeLogger)
    return logger.claude_mapping(model="gpt-5.6", messages=messages, response_obj=response_obj)


def test_claude_mapping_serializes_custom_tool_calls(monkeypatch):
    try:
        import anthropic  # noqa: F401
    except ImportError:
        stub = types.ModuleType("anthropic")
        stub.HUMAN_PROMPT = "\n\nHuman:"
        stub.AI_PROMPT = "\n\nAssistant:"
        monkeypatch.setitem(sys.modules, "anthropic", stub)
    response_obj = {
        "id": "chatcmpl-1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_c",
                            "type": "custom",
                            "custom": {"name": "ApplyPatch", "input": "*** Begin Patch"},
                        },
                        {
                            "id": "call_f",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }
    mapped = _claude_mapping([{"role": "user", "content": "hi"}], response_obj)
    tool_use_blocks = [b for b in mapped["content"] if b["type"] == "tool_use"]
    assert {"type": "tool_use", "id": "call_c", "name": "ApplyPatch", "input": "*** Begin Patch"} in tool_use_blocks
    assert {"type": "tool_use", "id": "call_f", "name": "read_file", "input": '{"path": "a.py"}'} in tool_use_blocks
