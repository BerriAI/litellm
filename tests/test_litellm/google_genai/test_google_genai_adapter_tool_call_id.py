"""
Tests for unique tool_call_id generation in the Google GenAI adapter.

Covers:
- Unique IDs for repeated calls to the same function
- FIFO matching between functionCall and functionResponse
- Multi-turn conversations with interleaved tool calls
- Fallback ID generation when no preceding functionCall exists

Related issue: functionCall/functionResponse parts in Gemini-native
contents caused tool_call_id collisions when the same function was
called multiple times (e.g. get_weather for two cities).  The adapter
now generates uuid-based IDs and matches responses via FIFO ordering.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter


@pytest.fixture
def adapter():
    return GoogleGenAIAdapter()


class TestToolCallIdUniqueness:
    """tool_call_ids must be globally unique, even for repeated function names."""

    def test_single_function_call_gets_unique_id(self, adapter):
        """A single functionCall should produce a unique call_* id."""
        contents = [
            {"role": "user", "parts": [{"text": "What's the weather?"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "London"},
                        }
                    }
                ],
            },
        ]
        messages = adapter._transform_contents_to_messages(contents)
        assistant_msg = messages[1]

        assert assistant_msg["role"] == "assistant"
        tool_calls = assistant_msg.get("tool_calls", [])
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"].startswith("call_")
        assert len(tool_calls[0]["id"]) > len("call_")

    def test_duplicate_function_names_get_distinct_ids(self, adapter):
        """Two calls to the same function in one turn must have different IDs."""
        contents = [
            {"role": "user", "parts": [{"text": "Weather in London and Paris"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "London"},
                        }
                    },
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                        }
                    },
                ],
            },
        ]
        messages = adapter._transform_contents_to_messages(contents)
        assistant_msg = messages[1]

        tool_calls = assistant_msg.get("tool_calls", [])
        assert len(tool_calls) == 2
        id_set = {tc["id"] for tc in tool_calls}
        assert len(id_set) == 2, "Duplicate tool_call_ids detected"

    def test_different_functions_get_distinct_ids(self, adapter):
        """Calls to different functions must also produce distinct IDs."""
        contents = [
            {"role": "user", "parts": [{"text": "Weather and time"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "London"},
                        }
                    },
                    {
                        "functionCall": {
                            "name": "get_time",
                            "args": {"timezone": "UTC"},
                        }
                    },
                ],
            },
        ]
        messages = adapter._transform_contents_to_messages(contents)
        assistant_msg = messages[1]

        tool_calls = assistant_msg.get("tool_calls", [])
        assert len(tool_calls) == 2
        id_set = {tc["id"] for tc in tool_calls}
        assert len(id_set) == 2


class TestFunctionResponseIdMatching:
    """functionResponse tool_call_ids must match the preceding functionCall."""

    def test_response_matches_call_id(self, adapter):
        """A functionResponse should carry the same id as its functionCall."""
        contents = [
            {"role": "user", "parts": [{"text": "Weather?"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "London"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"temp": "15C"},
                        }
                    }
                ],
            },
        ]
        messages = adapter._transform_contents_to_messages(contents)

        # messages[1] = assistant with tool_calls
        call_id = messages[1]["tool_calls"][0]["id"]
        # messages[2] = tool response
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == call_id

    def test_fifo_matching_for_duplicate_function_names(self, adapter):
        """When the same function is called twice, responses match in order."""
        contents = [
            {"role": "user", "parts": [{"text": "Two cities"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "London"},
                        }
                    },
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"temp": "15C"},
                        }
                    },
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"temp": "18C"},
                        }
                    },
                ],
            },
        ]
        messages = adapter._transform_contents_to_messages(contents)

        call_ids = [tc["id"] for tc in messages[1]["tool_calls"]]
        assert len(call_ids) == 2
        assert call_ids[0] != call_ids[1]

        # Tool messages should match in FIFO order
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == call_ids[0]
        assert tool_msgs[1]["tool_call_id"] == call_ids[1]

    def test_mixed_functions_match_correctly(self, adapter):
        """Multiple different functions match their responses correctly."""
        contents = [
            {"role": "user", "parts": [{"text": "Weather and time"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "London"},
                        }
                    },
                    {
                        "functionCall": {
                            "name": "get_time",
                            "args": {"tz": "UTC"},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"temp": "15C"},
                        }
                    },
                    {
                        "functionResponse": {
                            "name": "get_time",
                            "response": {"time": "12:00"},
                        }
                    },
                ],
            },
        ]
        messages = adapter._transform_contents_to_messages(contents)

        weather_call_id = messages[1]["tool_calls"][0]["id"]
        time_call_id = messages[1]["tool_calls"][1]["id"]

        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        # get_weather response matches get_weather call
        assert tool_msgs[0]["tool_call_id"] == weather_call_id
        assert json.loads(tool_msgs[0]["content"]) == {"temp": "15C"}
        # get_time response matches get_time call
        assert tool_msgs[1]["tool_call_id"] == time_call_id
        assert json.loads(tool_msgs[1]["content"]) == {"time": "12:00"}


class TestMultiTurnToolCalling:
    """End-to-end multi-turn conversations with tool use."""

    def test_full_multi_turn_tool_conversation(self, adapter):
        """
        Simulate: user asks -> model calls tool -> user sends result ->
        model calls another tool -> user sends result -> model answers.
        """
        contents = [
            {"role": "user", "parts": [{"text": "Add 2+3 then multiply by 4"}]},
            # Turn 1: model calls add
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "add",
                            "args": {"a": 2, "b": 3},
                        }
                    }
                ],
            },
            # Turn 1 response
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "add",
                            "response": {"result": 5},
                        }
                    }
                ],
            },
            # Turn 2: model calls multiply
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "multiply",
                            "args": {"a": 5, "b": 4},
                        }
                    }
                ],
            },
            # Turn 2 response
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "multiply",
                            "response": {"result": 20},
                        }
                    }
                ],
            },
            # Final answer
            {
                "role": "model",
                "parts": [{"text": "The result is 20."}],
            },
        ]
        messages = adapter._transform_contents_to_messages(contents)

        # Verify structure: user, assistant+tool_calls, tool, assistant+tool_calls, tool, assistant
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["tool_calls"]) == 1
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == messages[1]["tool_calls"][0]["id"]
        assert messages[3]["role"] == "assistant"
        assert len(messages[3]["tool_calls"]) == 1
        assert messages[4]["role"] == "tool"
        assert messages[4]["tool_call_id"] == messages[3]["tool_calls"][0]["id"]
        assert messages[5]["role"] == "assistant"
        assert messages[5]["content"] == "The result is 20."

        # All tool_call_ids must be distinct
        all_ids = {
            messages[1]["tool_calls"][0]["id"],
            messages[3]["tool_calls"][0]["id"],
        }
        assert len(all_ids) == 2

    def test_same_function_reused_across_separate_turns(self, adapter):
        """
        The same function called in turn 1 AND turn 2 must produce distinct
        IDs, and each turn's response must match its own turn's call.
        """
        contents = [
            {"role": "user", "parts": [{"text": "Step 1"}]},
            # Turn 1: model calls get_weather
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "London"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"temp": "15C"},
                        }
                    }
                ],
            },
            # Turn 2: model calls get_weather AGAIN
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"temp": "18C"},
                        }
                    }
                ],
            },
        ]
        messages = adapter._transform_contents_to_messages(contents)

        # Turn 1: assistant[1] -> tool[2]
        turn1_call_id = messages[1]["tool_calls"][0]["id"]
        assert messages[2]["tool_call_id"] == turn1_call_id

        # Turn 2: assistant[3] -> tool[4]
        turn2_call_id = messages[3]["tool_calls"][0]["id"]
        assert messages[4]["tool_call_id"] == turn2_call_id

        # IDs across turns must be distinct
        assert turn1_call_id != turn2_call_id

    def test_orphan_function_response_gets_fresh_id(self, adapter):
        """
        A functionResponse with no preceding functionCall should still
        produce a valid (generated) tool_call_id, not crash.
        """
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "unknown_func",
                            "response": {"data": "value"},
                        }
                    }
                ],
            },
        ]
        messages = adapter._transform_contents_to_messages(contents)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"].startswith("call_")
        assert len(messages[0]["tool_call_id"]) > len("call_")

    def test_function_response_content_serialization(self, adapter):
        """functionResponse.response should be JSON-serialized as content."""
        contents = [
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "search",
                            "args": {"q": "test"},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "search",
                            "response": {"results": [1, 2, 3], "total": 3},
                        }
                    }
                ],
            },
        ]
        messages = adapter._transform_contents_to_messages(contents)

        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        parsed = json.loads(tool_msg["content"])
        assert parsed == {"results": [1, 2, 3], "total": 3}

    def test_inline_data_and_string_parts(self, adapter):
        """inline_data and bare-string parts are handled in user turns."""
        contents = [
            {
                "role": "user",
                "parts": [
                    "bare string part",
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": "iVBORw0KGgo=",
                        }
                    },
                ],
            },
            {
                "role": "model",
                "parts": ["model bare string"],
            },
        ]
        messages = adapter._transform_contents_to_messages(contents)

        user_msg = messages[0]
        assert user_msg["role"] == "user"
        assert len(user_msg["content"]) == 2
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][0]["text"] == "bare string part"
        assert user_msg["content"][1]["type"] == "image_url"
        assert "data:image/png;base64,iVBORw0KGgo=" in user_msg["content"][1]["image_url"]["url"]

        assistant_msg = messages[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "model bare string"
