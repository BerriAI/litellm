import json
from unittest.mock import MagicMock

import pytest


import base64

from litellm.llms.bedrock.realtime.transformation import (
    TRIGGER_LEADING_SILENCE,
    TRIGGER_TRAILING_SILENCE,
    BedrockRealtimeConfig,
)
from litellm.llms.bedrock.realtime.trigger_audio import ready_trigger_pcm

# The OpenAI realtime function-call lifecycle a single Nova Sonic toolUse must expand into,
# when an assistant block already opened the response the tool call belongs to.
_TOOL_CALL_EVENT_SEQUENCE = [
    "response.output_item.added",
    "conversation.item.added",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.output_item.done",
    "response.done",
]

# Nova Sonic opens tool turns with contentStart role TOOL, which opens no response, so the
# tool call has to open one itself before it can close it.
_TOOL_CALL_EVENT_SEQUENCE_NEW_RESPONSE = ["response.created", *_TOOL_CALL_EVENT_SEQUENCE]


def _only(events, event_type):
    matches = [event for event in events if event["type"] == event_type]
    assert len(matches) == 1, f"expected exactly one {event_type}, got {len(matches)}"
    return matches[0]


def _response_id_of(event):
    """The response id an event is bound to, or None for events that carry no response id."""
    if event["type"] in ("response.created", "response.done"):
        return event["response"]["id"]
    return event.get("response_id")


class TestBedrockRealtimeConfig:
    """Test suite for BedrockRealtimeConfig class"""

    def test_initialization(self):
        """Test that BedrockRealtimeConfig initializes with correct defaults"""
        config = BedrockRealtimeConfig()

        assert config is not None
        assert config.max_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.voice_id == "matthew"
        assert config.output_sample_rate_hertz == 24000
        assert config.input_sample_rate_hertz == 16000
        assert config.text_media_type == "text/plain"

    def test_session_configuration_request(self):
        """Test session configuration request generation"""
        config = BedrockRealtimeConfig()

        session_config = config.session_configuration_request("amazon.nova-sonic-v1:0")
        session_dict = json.loads(session_config)

        assert "session_start" in session_dict
        assert "prompt_start" in session_dict

        # Check session start
        session_start = session_dict["session_start"]["event"]["sessionStart"]
        assert session_start["inferenceConfiguration"]["maxTokens"] == 1024
        assert session_start["inferenceConfiguration"]["temperature"] == 0.7

        # Check prompt start
        prompt_start = session_dict["prompt_start"]["event"]["promptStart"]
        assert prompt_start["audioOutputConfiguration"]["voiceId"] == "matthew"
        assert prompt_start["audioOutputConfiguration"]["sampleRateHertz"] == 24000

    def test_session_configuration_with_tools(self):
        """Test session configuration with tools"""
        config = BedrockRealtimeConfig()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        session_config = config.session_configuration_request("amazon.nova-sonic-v1:0", tools=tools)
        session_dict = json.loads(session_config)

        prompt_start = session_dict["prompt_start"]["event"]["promptStart"]
        assert "toolConfiguration" in prompt_start
        assert "tools" in prompt_start["toolConfiguration"]
        assert len(prompt_start["toolConfiguration"]["tools"]) == 1
        assert prompt_start["toolConfiguration"]["tools"][0]["toolSpec"]["name"] == "get_weather"

    def test_transform_tools_to_bedrock_format(self):
        """Test OpenAI tool format to Bedrock format transformation"""
        config = BedrockRealtimeConfig()

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string", "description": "City name"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        bedrock_tools = config._transform_tools_to_bedrock_format(openai_tools)

        assert len(bedrock_tools) == 1
        assert bedrock_tools[0]["toolSpec"]["name"] == "get_weather"
        assert bedrock_tools[0]["toolSpec"]["description"] == "Get current weather"
        assert "inputSchema" in bedrock_tools[0]["toolSpec"]

        # Verify the schema is properly JSON stringified
        schema = json.loads(bedrock_tools[0]["toolSpec"]["inputSchema"]["json"])
        assert schema["type"] == "object"
        assert "location" in schema["properties"]

    def test_audio_format_mapping(self):
        """Test audio format to sample rate mapping"""
        config = BedrockRealtimeConfig()

        # Test PCM16 format
        assert config._map_audio_format_to_sample_rate("pcm16", is_output=True) == 24000
        assert config._map_audio_format_to_sample_rate("pcm16", is_output=False) == 16000

        # Test G.711 formats
        assert config._map_audio_format_to_sample_rate("g711_ulaw", is_output=True) == 8000
        assert config._map_audio_format_to_sample_rate("g711_alaw", is_output=False) == 8000

    def test_transform_session_update_event(self):
        """Test session.update event transformation"""
        config = BedrockRealtimeConfig()

        session_update = {
            "type": "session.update",
            "session": {
                "temperature": 0.9,
                "voice": "joanna",
                "max_response_output_tokens": 2048,
                "output_audio_format": "pcm16",
            },
        }

        messages = config.transform_session_update_event(session_update)

        assert len(messages) >= 2  # At least session start and prompt start

        # Verify attributes were updated
        assert config.temperature == 0.9
        assert config.voice_id == "joanna"
        assert config.max_tokens == 2048

        # Verify session start message
        session_start = json.loads(messages[0])
        assert session_start["event"]["sessionStart"]["inferenceConfiguration"]["temperature"] == 0.9

    def test_transform_session_update_with_tools(self):
        """Test session.update with tools"""
        config = BedrockRealtimeConfig()

        session_update = {
            "type": "session.update",
            "session": {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "description": "Get current time",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]
            },
        }

        messages = config.transform_session_update_event(session_update)

        # Find prompt start message
        prompt_start = json.loads(messages[1])
        assert "toolConfiguration" in prompt_start["event"]["promptStart"]

    def test_transform_conversation_item_create_text(self):
        """Test conversation.item.create with text"""
        config = BedrockRealtimeConfig()

        item_create = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello, how are you?"}],
            },
        }

        messages = config.transform_conversation_item_create_event(item_create)

        # Should have content start, text input, and content end
        assert len(messages) == 3

        content_start = json.loads(messages[0])
        assert content_start["event"]["contentStart"]["type"] == "TEXT"
        assert content_start["event"]["contentStart"]["role"] == "USER"

        text_input = json.loads(messages[1])
        assert text_input["event"]["textInput"]["content"] == "Hello, how are you?"

    def test_transform_conversation_item_create_tool_result(self):
        """Test conversation.item.create with tool result"""
        config = BedrockRealtimeConfig()

        tool_result = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": json.dumps({"temperature": 72, "conditions": "sunny"}),
            },
        }

        messages = config.transform_conversation_item_create_event(tool_result)

        # Should have content start, tool result, and content end
        assert len(messages) == 3

        content_start = json.loads(messages[0])
        assert content_start["event"]["contentStart"]["type"] == "TOOL"
        assert content_start["event"]["contentStart"]["role"] == "TOOL"
        assert content_start["event"]["contentStart"]["toolResultInputConfiguration"]["toolUseId"] == "call_123"

    def test_transform_input_audio_buffer_append(self):
        """Test input_audio_buffer.append transformation"""
        config = BedrockRealtimeConfig()

        audio_append = {
            "type": "input_audio_buffer.append",
            "audio": "base64_audio_data_here",
        }

        messages = config.transform_input_audio_buffer_append_event(audio_append)

        # First call should include content start
        assert len(messages) == 2

        content_start = json.loads(messages[0])
        assert content_start["event"]["contentStart"]["type"] == "AUDIO"
        assert content_start["event"]["contentStart"]["audioInputConfiguration"]["sampleRateHertz"] == 16000

        audio_input = json.loads(messages[1])
        assert audio_input["event"]["audioInput"]["content"] == "base64_audio_data_here"

    def test_transform_input_audio_buffer_commit(self):
        """Test input_audio_buffer.commit transformation"""
        config = BedrockRealtimeConfig()

        # First append to set the flag
        config._audio_content_started = True

        commit = {"type": "input_audio_buffer.commit"}

        messages = config.transform_input_audio_buffer_commit_event(commit)

        assert len(messages) == 1
        content_end = json.loads(messages[0])
        assert "contentEnd" in content_end["event"]


class TestBedrockRealtimeResponseCreate:
    """response.create must trigger Nova Sonic generation (LIT-2239 regression)"""

    def _start_session(self, config):
        config.transform_realtime_request(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"instructions": "You are a helpful assistant."},
                }
            ),
            "amazon.nova-sonic-v1:0",
        )

    def test_response_create_before_session_update_is_noop(self):
        config = BedrockRealtimeConfig()

        messages = config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")

        assert messages == []

    def test_response_create_emits_spoken_trigger_audio(self):
        config = BedrockRealtimeConfig()
        self._start_session(config)

        messages = config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")

        assert len(messages) > 1

        content_start = json.loads(messages[0])["event"]["contentStart"]
        assert content_start["promptName"] == config.prompt_name
        assert content_start["contentName"] == config.audio_content_name
        assert content_start["type"] == "AUDIO"
        assert content_start["interactive"] is True
        assert content_start["role"] == "USER"
        assert content_start["audioInputConfiguration"]["sampleRateHertz"] == 16000

        audio_events = [json.loads(message)["event"]["audioInput"] for message in messages[1:]]
        assert all(event["promptName"] == config.prompt_name for event in audio_events)
        assert all(event["contentName"] == config.audio_content_name for event in audio_events)

        sent_pcm = b"".join(base64.b64decode(event["content"]) for event in audio_events)
        assert sent_pcm == TRIGGER_LEADING_SILENCE + ready_trigger_pcm() + TRIGGER_TRAILING_SILENCE

    def test_second_response_create_reuses_open_audio_content(self):
        config = BedrockRealtimeConfig()
        self._start_session(config)

        first = config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")
        second = config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")

        assert len(second) == len(first) - 1
        assert all("audioInput" in json.loads(message)["event"] for message in second)

    def test_response_create_is_noop_when_client_streams_audio(self):
        config = BedrockRealtimeConfig()
        self._start_session(config)
        config.transform_realtime_request(
            json.dumps({"type": "input_audio_buffer.append", "audio": "c2lsZW5jZQ=="}),
            "amazon.nova-sonic-v1:0",
        )

        messages = config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")

        assert messages == []

    def test_client_audio_after_trigger_reopens_block_at_client_sample_rate(self):
        config = BedrockRealtimeConfig()
        config.transform_realtime_request(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "instructions": "You are a helpful assistant.",
                        "input_audio_format": "g711_ulaw",
                    },
                }
            ),
            "amazon.nova-sonic-v1:0",
        )
        config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")
        trigger_content_name = config.audio_content_name

        messages = config.transform_realtime_request(
            json.dumps({"type": "input_audio_buffer.append", "audio": "c2lsZW5jZQ=="}),
            "amazon.nova-sonic-v1:0",
        )

        events = [json.loads(message)["event"] for message in messages]
        assert [next(iter(event)) for event in events] == [
            "contentEnd",
            "contentStart",
            "audioInput",
        ]
        assert events[0]["contentEnd"]["contentName"] == trigger_content_name
        new_content_start = events[1]["contentStart"]
        assert new_content_start["contentName"] == config.audio_content_name
        assert new_content_start["contentName"] != trigger_content_name
        assert new_content_start["audioInputConfiguration"]["sampleRateHertz"] == 8000
        assert events[2]["audioInput"]["contentName"] == config.audio_content_name

    def test_client_audio_after_trigger_reuses_block_at_matching_sample_rate(self):
        config = BedrockRealtimeConfig()
        self._start_session(config)
        config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")
        trigger_content_name = config.audio_content_name

        messages = config.transform_realtime_request(
            json.dumps({"type": "input_audio_buffer.append", "audio": "c2lsZW5jZQ=="}),
            "amazon.nova-sonic-v1:0",
        )

        assert len(messages) == 1
        audio_input = json.loads(messages[0])["event"]["audioInput"]
        assert audio_input["contentName"] == trigger_content_name

    def test_session_close_messages_close_audio_prompt_and_session(self):
        config = BedrockRealtimeConfig()
        self._start_session(config)
        config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")

        close_messages = [json.loads(message)["event"] for message in config.session_close_messages()]

        assert [next(iter(event)) for event in close_messages] == [
            "contentEnd",
            "promptEnd",
            "sessionEnd",
        ]
        assert close_messages[0]["contentEnd"]["contentName"] == config.audio_content_name
        assert close_messages[1]["promptEnd"]["promptName"] == config.prompt_name
        assert config.session_close_messages() == []

    def test_session_close_messages_before_session_update_is_empty(self):
        config = BedrockRealtimeConfig()

        assert config.session_close_messages() == []


class TestBedrockRealtimeResponseTransformation:
    """Test suite for response transformation"""

    def test_bedrock_session_start_does_not_emit_duplicate_session_created(self):
        """A Bedrock output sessionStart must not forward a second session.created to the
        client; session.created is sent exactly once on connect (LIT-4655)"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        bedrock_message = {
            "event": {"sessionStart": {"inferenceConfiguration": {"maxTokens": 1024, "temperature": 0.7}}}
        }

        result = config.transform_realtime_response(
            json.dumps(bedrock_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": None,
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )

        assert result["response"] == []
        assert result["session_configuration_request"] == json.dumps({"configured": True})

    def test_transform_text_output_response(self):
        """Test textOutput response transformation"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        # First create a content start to initialize IDs
        content_start_message = {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}

        result1 = config.transform_realtime_response(
            json.dumps(content_start_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )

        # Now send text output
        text_output_message = {"event": {"textOutput": {"content": "Hello, world!"}}}

        result2 = config.transform_realtime_response(
            json.dumps(text_output_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": result1["current_output_item_id"],
                "current_response_id": result1["current_response_id"],
                "current_conversation_id": result1["current_conversation_id"],
                "current_delta_chunks": result1["current_delta_chunks"],
                "current_item_chunks": [],
                "current_delta_type": result1["current_delta_type"],
            },
        )

        # Check for text delta
        text_deltas = [msg for msg in result2["response"] if msg["type"] == "response.text.delta"]
        assert len(text_deltas) == 1
        assert text_deltas[0]["delta"] == "Hello, world!"

        # Check that delta chunks are accumulated
        assert len(result2["current_delta_chunks"]) == 1

    def test_transform_audio_output_response(self):
        """Test audioOutput response transformation"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        # First create a content start for audio
        content_start_message = {"event": {"contentStart": {"role": "ASSISTANT", "type": "AUDIO"}}}

        result1 = config.transform_realtime_response(
            json.dumps(content_start_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )

        # Now send audio output
        audio_output_message = {"event": {"audioOutput": {"content": "base64_audio_content"}}}

        result2 = config.transform_realtime_response(
            json.dumps(audio_output_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": result1["current_output_item_id"],
                "current_response_id": result1["current_response_id"],
                "current_conversation_id": result1["current_conversation_id"],
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": result1["current_delta_type"],
            },
        )

        # Check for audio delta
        audio_deltas = [msg for msg in result2["response"] if msg["type"] == "response.audio.delta"]
        assert len(audio_deltas) == 1
        assert audio_deltas[0]["delta"] == "base64_audio_content"

    def test_transform_tool_use_response(self):
        """Test toolUse response transformation"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        tool_use_message = {
            "event": {
                "toolUse": {
                    "toolUseId": "tool_call_123",
                    "toolName": "get_weather",
                    "input": json.dumps({"location": "San Francisco"}),
                }
            }
        }

        result = config.transform_realtime_response(
            json.dumps(tool_use_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": "item_123",
                "current_response_id": "resp_123",
                "current_conversation_id": "conv_123",
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": "text",
            },
        )

        assert [msg["type"] for msg in result["response"]] == _TOOL_CALL_EVENT_SEQUENCE
        function_call = _only(result["response"], "response.function_call_arguments.done")
        assert function_call["call_id"] == "tool_call_123"
        assert function_call["name"] == "get_weather"
        assert json.loads(function_call["arguments"]) == {"location": "San Francisco"}

    def test_transform_tool_use_response_with_content_field(self):
        """Test toolUse response transformation with Nova 2 Sonic `content` field"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        tool_use_message = {
            "event": {
                "toolUse": {
                    "toolUseId": "tool_call_123",
                    "toolName": "get_weather",
                    "content": json.dumps({"location": "San Francisco"}),
                }
            }
        }

        result = config.transform_realtime_response(
            json.dumps(tool_use_message),
            "amazon.nova-2-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": "item_123",
                "current_response_id": "resp_123",
                "current_conversation_id": "conv_123",
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": "text",
            },
        )

        assert [msg["type"] for msg in result["response"]] == _TOOL_CALL_EVENT_SEQUENCE
        function_call = _only(result["response"], "response.function_call_arguments.done")
        assert function_call["call_id"] == "tool_call_123"
        assert function_call["name"] == "get_weather"
        assert json.loads(function_call["arguments"]) == {"location": "San Francisco"}

        done = _only(result["response"], "response.done")
        assert done["response"]["id"] == "resp_123"
        assert done["response"]["output"] == [
            {
                "id": "item_123",
                "object": "realtime.item",
                "type": "function_call",
                "status": "completed",
                "call_id": "tool_call_123",
                "name": "get_weather",
                "arguments": function_call["arguments"],
            }
        ]
        assert result["current_response_id"] is None
        assert result["current_output_item_id"] is None

    def test_transform_tool_use_event_directly(self):
        """transform_tool_use_event emits the full OpenAI function-call lifecycle"""
        config = BedrockRealtimeConfig()

        # Missing IDs are minted (Nova Sonic starts tool turns with contentStart role=TOOL)
        events, tool_call_id, tool_name = config.transform_tool_use_event(
            {
                "toolUse": {
                    "toolUseId": "tool_call_no_ids",
                    "toolName": "get_weather",
                    "content": json.dumps({"location": "Seattle"}),
                }
            },
            None,
            None,
            "conv_1",
        )
        assert [event["type"] for event in events] == _TOOL_CALL_EVENT_SEQUENCE_NEW_RESPONSE
        function_call = _only(events, "response.function_call_arguments.done")
        assert _only(events, "response.created")["response"]["id"] == function_call["response_id"]
        assert _only(events, "response.created")["response"]["conversation_id"] == "conv_1"
        assert function_call["call_id"] == "tool_call_no_ids"
        assert function_call["name"] == "get_weather"
        assert function_call["response_id"].startswith("resp_")
        assert function_call["item_id"].startswith("item_")
        assert json.loads(function_call["arguments"]) == {"location": "Seattle"}
        assert tool_call_id == "tool_call_no_ids"
        assert tool_name == "get_weather"

        # Every event in the turn shares the minted response/item ids
        assert {_response_id_of(event) for event in events} - {None} == {function_call["response_id"]}
        assert _only(events, "response.output_item.added")["item"]["id"] == function_call["item_id"]
        assert _only(events, "response.output_item.done")["item"]["id"] == function_call["item_id"]

        # The added item is in_progress with empty args; the done item carries the parsed args
        assert _only(events, "response.output_item.added")["item"]["status"] == "in_progress"
        assert _only(events, "response.output_item.added")["item"]["arguments"] == ""
        assert _only(events, "conversation.item.added")["item"]["arguments"] == ""
        assert _only(events, "response.function_call_arguments.delta")["delta"] == function_call["arguments"]
        assert _only(events, "response.output_item.done")["item"]["status"] == "completed"
        assert _only(events, "response.output_item.done")["item"]["arguments"] == function_call["arguments"]

        # response.done closes the turn and carries the call so spend logging can harvest it
        done = _only(events, "response.done")
        assert done["response"]["id"] == function_call["response_id"]
        assert done["response"]["conversation_id"] == "conv_1"
        assert done["response"]["status"] == "completed"
        assert done["response"]["output"][0]["call_id"] == "tool_call_no_ids"
        assert done["response"]["output"][0]["type"] == "function_call"

        # Explicit ids are reused rather than minted
        events, _, _ = config.transform_tool_use_event(
            {
                "toolUse": {
                    "toolUseId": "tool_call_123",
                    "toolName": "get_weather",
                    "content": json.dumps({"location": "San Francisco"}),
                }
            },
            "item_123",
            "resp_123",
            "conv_1",
        )
        function_call = _only(events, "response.function_call_arguments.done")
        assert function_call["response_id"] == "resp_123"
        assert function_call["item_id"] == "item_123"
        assert json.loads(function_call["arguments"]) == {"location": "San Francisco"}

        # Legacy `input` field is still honoured when `content` is absent
        events, _, _ = config.transform_tool_use_event(
            {
                "toolUse": {
                    "toolUseId": "tool_call_legacy",
                    "toolName": "get_weather",
                    "input": json.dumps({"location": "Boston"}),
                }
            },
            "item_123",
            "resp_123",
            "conv_1",
        )
        assert json.loads(_only(events, "response.function_call_arguments.done")["arguments"]) == {"location": "Boston"}

        # Invalid JSON content falls back to empty arguments
        events, _, _ = config.transform_tool_use_event(
            {
                "toolUse": {
                    "toolUseId": "tool_call_124",
                    "toolName": "get_weather",
                    "content": "not valid json",
                }
            },
            "item_123",
            "resp_123",
            "conv_1",
        )
        assert json.loads(_only(events, "response.function_call_arguments.done")["arguments"]) == {}

    def test_transform_realtime_response_persists_minted_tool_ids(self):
        """TOOL-first turns must write minted response/item ids into session state"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        state = {
            "session_configuration_request": json.dumps({"configured": True}),
            "current_output_item_id": None,
            "current_response_id": None,
            "current_conversation_id": "conv_123",
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        }

        content_start_result = config.transform_realtime_response(
            json.dumps({"event": {"contentStart": {"role": "TOOL", "type": "TOOL"}}}),
            "amazon.nova-2-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=state,
        )
        assert content_start_result["response"] == []
        assert content_start_result["current_delta_type"] is None
        state.update(
            {
                "current_output_item_id": content_start_result["current_output_item_id"],
                "current_response_id": content_start_result["current_response_id"],
                "current_conversation_id": content_start_result["current_conversation_id"],
                "current_delta_chunks": content_start_result["current_delta_chunks"],
                "current_item_chunks": content_start_result["current_item_chunks"],
                "current_delta_type": content_start_result["current_delta_type"],
            }
        )

        tool_use_message = {
            "event": {
                "toolUse": {
                    "toolUseId": "tool_call_state",
                    "toolName": "get_weather",
                    "content": json.dumps({"location": "Seattle"}),
                }
            }
        }

        result = config.transform_realtime_response(
            json.dumps(tool_use_message),
            "amazon.nova-2-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=state,
        )

        assert [msg["type"] for msg in result["response"]] == _TOOL_CALL_EVENT_SEQUENCE_NEW_RESPONSE
        function_call = _only(result["response"], "response.function_call_arguments.done")
        assert _only(result["response"], "response.created")["response"]["id"] == function_call["response_id"]
        assert function_call["response_id"].startswith("resp_")
        assert function_call["item_id"].startswith("item_")
        assert json.loads(function_call["arguments"]) == {"location": "Seattle"}

        # The tool turn closes the response it minted, so no in-progress response is orphaned
        # and the ids cannot leak into the post-tool assistant turn.
        tool_done = _only(result["response"], "response.done")
        assert tool_done["response"]["id"] == function_call["response_id"]
        assert result["current_response_id"] is None
        assert result["current_output_item_id"] is None

        content_end_message = {
            "event": {
                "contentEnd": {
                    "stopReason": "TOOL_USE",
                    "type": "TOOL",
                }
            }
        }
        follow_up = config.transform_realtime_response(
            json.dumps(content_end_message),
            "amazon.nova-2-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": result["session_configuration_request"],
                "current_output_item_id": result["current_output_item_id"],
                "current_response_id": result["current_response_id"],
                "current_conversation_id": result["current_conversation_id"],
                "current_delta_chunks": result["current_delta_chunks"],
                "current_item_chunks": result["current_item_chunks"],
                "current_delta_type": result["current_delta_type"],
            },
        )
        assert follow_up["current_response_id"] is None
        assert follow_up["current_output_item_id"] is None
        assert follow_up["current_delta_type"] is None
        # The tool turn already emitted response.done; TOOL contentEnd must not emit a second
        # one, nor an unpaired message-shaped output_item.done.
        assert follow_up["response"] == []

        post_tool_state = {
            "session_configuration_request": follow_up["session_configuration_request"],
            "current_output_item_id": follow_up["current_output_item_id"],
            "current_response_id": follow_up["current_response_id"],
            "current_conversation_id": follow_up["current_conversation_id"],
            "current_delta_chunks": follow_up["current_delta_chunks"],
            "current_item_chunks": follow_up["current_item_chunks"],
            "current_delta_type": follow_up["current_delta_type"],
        }
        assistant_start = config.transform_realtime_response(
            json.dumps({"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}),
            "amazon.nova-2-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=post_tool_state,
        )
        assert assistant_start["current_response_id"] is not None
        assert assistant_start["current_output_item_id"] is not None
        assert assistant_start["current_response_id"] != function_call["response_id"]
        assert assistant_start["current_output_item_id"] != function_call["item_id"]
        created = [msg for msg in assistant_start["response"] if msg["type"] == "response.created"][0]
        added = [msg for msg in assistant_start["response"] if msg["type"] == "response.output_item.added"][0]
        assert created["response"]["id"] == assistant_start["current_response_id"]
        assert added["item"]["id"] == assistant_start["current_output_item_id"]
        assert created["response"]["id"] != function_call["response_id"]
        assert added["item"]["id"] != function_call["item_id"]

    def test_tool_content_end_does_not_emit_message_output_item_done(self):
        """
        TOOL contentEnd must stay silent: the tool turn already closed its own response, and a
        message-shaped output_item.done here would have no matching output_item.added.
        """
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        content_end_message = {
            "event": {
                "contentEnd": {
                    "stopReason": "TOOL_USE",
                    "type": "TOOL",
                }
            }
        }
        result = config.transform_realtime_response(
            json.dumps(content_end_message),
            "amazon.nova-2-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": "item_open_assistant_turn",
                "current_response_id": "resp_open_assistant_turn",
                "current_conversation_id": "conv_123",
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": "text",
            },
        )

        assert result["response"] == []
        assert result["current_output_item_id"] is None
        assert result["current_delta_type"] is None
        # A TOOL block that produced no toolUse leaves the assistant response open rather than
        # dropping its id, so the next assistant block reuses it instead of orphaning it.
        assert result["current_response_id"] == "resp_open_assistant_turn"

    def test_transform_content_end_text(self):
        """Test contentEnd for text response"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        # Create some delta chunks first
        delta_chunks = [
            {"delta": "Hello, ", "type": "response.text.delta"},
            {"delta": "world!", "type": "response.text.delta"},
        ]

        content_end_message = {"event": {"contentEnd": {}}}

        result = config.transform_realtime_response(
            json.dumps(content_end_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": "item_123",
                "current_response_id": "resp_123",
                "current_conversation_id": "conv_123",
                "current_delta_chunks": delta_chunks,
                "current_item_chunks": [],
                "current_delta_type": "text",
            },
        )

        # Should have text.done, content_part.done, and output_item.done
        assert len(result["response"]) == 3

        text_done = [msg for msg in result["response"] if msg["type"] == "response.text.done"][0]
        assert text_done["text"] == "Hello, world!"

        # Delta chunks should be reset
        assert result["current_delta_chunks"] is None

    def test_content_end_end_turn_emits_response_done(self):
        """END_TURN contentEnd must produce response.done (LIT-2239 regression)"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        content_end_message = {"event": {"contentEnd": {"stopReason": "END_TURN", "type": "AUDIO"}}}

        result = config.transform_realtime_response(
            json.dumps(content_end_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": "item_123",
                "current_response_id": "resp_123",
                "current_conversation_id": "conv_123",
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": "audio",
            },
        )

        response_done_events = [msg for msg in result["response"] if msg["type"] == "response.done"]
        assert len(response_done_events) == 1
        assert response_done_events[0]["response"]["status"] == "completed"
        assert result["current_output_item_id"] is None
        assert result["current_response_id"] is None
        assert result["current_delta_type"] is None

    def test_content_end_partial_turn_does_not_emit_response_done(self):
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        content_end_message = {"event": {"contentEnd": {"stopReason": "PARTIAL_TURN", "type": "TEXT"}}}

        result = config.transform_realtime_response(
            json.dumps(content_end_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": "item_123",
                "current_response_id": "resp_123",
                "current_conversation_id": "conv_123",
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": "text",
            },
        )

        assert all(msg["type"] != "response.done" for msg in result["response"])
        assert result["current_response_id"] == "resp_123"

    def test_transform_prompt_end_response(self):
        """Test promptEnd response transformation"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        prompt_end_message = {"event": {"promptEnd": {}}}

        result = config.transform_realtime_response(
            json.dumps(prompt_end_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": "item_123",
                "current_response_id": "resp_123",
                "current_conversation_id": "conv_123",
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": "text",
            },
        )

        # Should have response.done
        assert len(result["response"]) == 1
        assert result["response"][0]["type"] == "response.done"
        assert result["response"][0]["response"]["status"] == "completed"

        # State should be reset
        assert result["current_output_item_id"] is None
        assert result["current_response_id"] is None
        assert result["current_delta_type"] is None

    def test_event_id_uniqueness(self):
        """Test that all event_ids are unique"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        # Create a sequence of messages
        content_start = {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}
        text_output1 = {"event": {"textOutput": {"content": "Hello"}}}
        text_output2 = {"event": {"textOutput": {"content": " world"}}}

        all_events = []
        state = {
            "session_configuration_request": json.dumps({"configured": True}),
            "current_output_item_id": None,
            "current_response_id": None,
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        }

        # Process all messages
        for msg in [content_start, text_output1, text_output2]:
            result = config.transform_realtime_response(
                json.dumps(msg),
                "amazon.nova-sonic-v1:0",
                logging_obj,
                realtime_response_transform_input=state,
            )
            all_events.extend(result["response"])
            # Update state for next iteration
            state.update(
                {
                    "current_output_item_id": result["current_output_item_id"],
                    "current_response_id": result["current_response_id"],
                    "current_conversation_id": result["current_conversation_id"],
                    "current_delta_chunks": result["current_delta_chunks"],
                    "current_delta_type": result["current_delta_type"],
                }
            )

        # Check all event_ids are unique
        event_ids = [event["event_id"] for event in all_events if "event_id" in event]
        assert len(event_ids) == len(set(event_ids)), "Event IDs should be unique"

    def test_response_id_consistency(self):
        """Test that response_id remains consistent across related events"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        # Create a sequence of messages
        content_start = {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}
        text_output = {"event": {"textOutput": {"content": "Hello"}}}

        all_events = []
        state = {
            "session_configuration_request": json.dumps({"configured": True}),
            "current_output_item_id": None,
            "current_response_id": None,
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        }

        # Process messages
        for msg in [content_start, text_output]:
            result = config.transform_realtime_response(
                json.dumps(msg),
                "amazon.nova-sonic-v1:0",
                logging_obj,
                realtime_response_transform_input=state,
            )
            all_events.extend(result["response"])
            state.update(
                {
                    "current_output_item_id": result["current_output_item_id"],
                    "current_response_id": result["current_response_id"],
                    "current_conversation_id": result["current_conversation_id"],
                    "current_delta_chunks": result["current_delta_chunks"],
                    "current_delta_type": result["current_delta_type"],
                }
            )

        # Check all response_ids are the same
        response_ids = [event["response_id"] for event in all_events if "response_id" in event]
        assert len(set(response_ids)) == 1, "Response IDs should be consistent"


class TestBedrockRealtimeSessionEvents:
    """session.created / session.updated builders produce spec-shaped events (LIT-4655)"""

    @staticmethod
    def _logging():
        from types import SimpleNamespace

        return SimpleNamespace(litellm_trace_id="trace_123")

    def test_session_created_event_shape(self):
        event = BedrockRealtimeConfig().session_created_event("amazon.nova-sonic-v1:0", self._logging())
        assert event["type"] == "session.created"
        assert event["session"]["id"] == "trace_123"
        assert event["session"]["model"] == "amazon.nova-sonic-v1:0"
        assert event["session"]["modalities"] == ["text", "audio"]
        assert event["event_id"]

    def test_session_updated_event_shape(self):
        event = BedrockRealtimeConfig().session_updated_event("amazon.nova-sonic-v1:0", self._logging())
        assert event["type"] == "session.updated"
        assert event["session"]["id"] == "trace_123"
        assert event["session"]["model"] == "amazon.nova-sonic-v1:0"
        assert event["event_id"]

    def test_created_and_updated_have_distinct_event_ids(self):
        config = BedrockRealtimeConfig()
        logging_obj = self._logging()
        created = config.session_created_event("amazon.nova-sonic-v1:0", logging_obj)
        updated = config.session_updated_event("amazon.nova-sonic-v1:0", logging_obj)
        assert created["event_id"] != updated["event_id"]

    def test_session_updated_reflects_requested_modalities(self):
        event = BedrockRealtimeConfig().session_updated_event(
            "amazon.nova-sonic-v1:0", self._logging(), modalities=["text"]
        )
        assert event["session"]["modalities"] == ["text"]

    def test_session_updated_defaults_modalities_when_unspecified(self):
        event = BedrockRealtimeConfig().session_updated_event("amazon.nova-sonic-v1:0", self._logging())
        assert event["session"]["modalities"] == ["text", "audio"]


class TestBedrockRealtimeContentBlockLifecycle:
    """
    Bedrock streams discrete content blocks. Session state must follow block
    boundaries so text/audio/tool blocks cannot leak into each other.
    """

    def _state(self, **overrides):
        base = {
            "session_configuration_request": json.dumps({"configured": True}),
            "current_output_item_id": None,
            "current_response_id": None,
            "current_conversation_id": "conv_1",
            "current_delta_chunks": None,
            "current_item_chunks": [],
            "current_delta_type": None,
        }
        base.update(overrides)
        return base

    def _apply(self, config, logging_obj, state, message):
        result = config.transform_realtime_response(
            json.dumps(message),
            "amazon.nova-2-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=state,
        )
        state.update(
            {
                "current_output_item_id": result["current_output_item_id"],
                "current_response_id": result["current_response_id"],
                "current_conversation_id": result["current_conversation_id"],
                "current_delta_chunks": result["current_delta_chunks"],
                "current_item_chunks": result["current_item_chunks"],
                "current_delta_type": result["current_delta_type"],
            }
        )
        return result

    def test_tool_block_does_not_leak_prior_text_into_next_assistant_turn(self):
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"
        state = self._state()

        self._apply(config, logging_obj, state, {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}})
        first_response_id = state["current_response_id"]
        self._apply(
            config,
            logging_obj,
            state,
            {"event": {"textOutput": {"content": "I will check the weather."}}},
        )
        assert state["current_delta_chunks"] is not None
        assert len(state["current_delta_chunks"]) == 1

        self._apply(
            config,
            logging_obj,
            state,
            {"event": {"contentEnd": {"stopReason": "PARTIAL_TURN", "type": "TEXT"}}},
        )
        assert state["current_delta_chunks"] is None
        assert state["current_delta_type"] is None
        assert state["current_output_item_id"] is None
        assert state["current_response_id"] == first_response_id

        self._apply(config, logging_obj, state, {"event": {"contentStart": {"role": "TOOL", "type": "TOOL"}}})
        assert state["current_delta_chunks"] is None
        assert state["current_response_id"] == first_response_id

        tool_result = self._apply(
            config,
            logging_obj,
            state,
            {
                "event": {
                    "toolUse": {
                        "toolUseId": "tool_1",
                        "toolName": "get_weather",
                        "content": json.dumps({"location": "Seattle"}),
                    }
                }
            },
        )
        assert [msg["type"] for msg in tool_result["response"]] == _TOOL_CALL_EVENT_SEQUENCE
        # The response the assistant text block opened is closed by the tool turn instead of
        # being left in_progress forever once the ids are cleared.
        tool_done = _only(tool_result["response"], "response.done")
        assert tool_done["response"]["id"] == first_response_id
        assert tool_done["response"]["output"][0]["call_id"] == "tool_1"
        assert state["current_response_id"] is None
        assert state["current_output_item_id"] is None
        assert state["current_delta_chunks"] is None
        assert state["current_delta_type"] is None

        tool_content_end = self._apply(
            config,
            logging_obj,
            state,
            {"event": {"contentEnd": {"stopReason": "TOOL_USE", "type": "TOOL"}}},
        )
        assert tool_content_end["response"] == []
        assert state["current_response_id"] is None
        assert state["current_output_item_id"] is None

        post_tool = self._apply(
            config,
            logging_obj,
            state,
            {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}},
        )
        assert state["current_response_id"] != first_response_id
        assert state["current_delta_chunks"] is None
        assert [msg["type"] for msg in post_tool["response"]].count("response.created") == 1

        self._apply(
            config,
            logging_obj,
            state,
            {"event": {"textOutput": {"content": "It is sunny in Seattle."}}},
        )
        done = self._apply(
            config,
            logging_obj,
            state,
            {"event": {"contentEnd": {"stopReason": "END_TURN", "type": "TEXT"}}},
        )
        text_done = [msg for msg in done["response"] if msg["type"] == "response.text.done"][0]
        assert text_done["text"] == "It is sunny in Seattle."
        assert "I will check the weather." not in text_done["text"]
        assert any(msg["type"] == "response.done" for msg in done["response"])
        assert state["current_response_id"] is None
        assert state["current_delta_chunks"] is None

    def test_every_created_response_is_closed_across_a_tool_turn(self):
        """
        Realtime clients track in-progress responses by id. A tool turn that drops the
        response id without a matching response.done leaves one open forever.
        """
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"
        state = self._state()

        turn = [
            {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}},
            {"event": {"textOutput": {"content": "Let me check."}}},
            {"event": {"contentEnd": {"stopReason": "PARTIAL_TURN", "type": "TEXT"}}},
            {"event": {"contentStart": {"role": "TOOL", "type": "TOOL"}}},
            {
                "event": {
                    "toolUse": {
                        "toolUseId": "tool_1",
                        "toolName": "get_weather",
                        "content": json.dumps({"location": "Seattle"}),
                    }
                }
            },
            {"event": {"contentEnd": {"stopReason": "TOOL_USE", "type": "TOOL"}}},
            {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}},
            {"event": {"textOutput": {"content": "It is sunny."}}},
            {"event": {"contentEnd": {"stopReason": "END_TURN", "type": "TEXT"}}},
        ]
        emitted = [msg for event in turn for msg in self._apply(config, logging_obj, state, event)["response"]]

        created = [msg["response"]["id"] for msg in emitted if msg["type"] == "response.created"]
        done = [msg["response"]["id"] for msg in emitted if msg["type"] == "response.done"]
        assert len(created) == 2
        assert created == done
        assert state["current_response_id"] is None

    def test_every_response_is_opened_and_closed_on_a_tool_first_turn(self):
        """
        Nova Sonic opens tool turns with contentStart role TOOL, which emits nothing, so the
        tool call is the first thing in the session. It has to open the response it closes,
        or the client sees a response.done for an id it never saw created.
        """
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"
        state = self._state(current_conversation_id=None)

        turn = [
            {"event": {"contentStart": {"role": "TOOL", "type": "TOOL"}}},
            {
                "event": {
                    "toolUse": {
                        "toolUseId": "tool_1",
                        "toolName": "get_weather",
                        "content": json.dumps({"location": "Seattle"}),
                    }
                }
            },
            {"event": {"contentEnd": {"stopReason": "TOOL_USE", "type": "TOOL"}}},
        ]
        emitted = [msg for event in turn for msg in self._apply(config, logging_obj, state, event)["response"]]

        created = [msg["response"]["id"] for msg in emitted if msg["type"] == "response.created"]
        done = [msg["response"]["id"] for msg in emitted if msg["type"] == "response.done"]
        assert len(created) == 1
        assert created == done
        # Every event in the turn is bound to that one response.
        assert {_response_id_of(msg) for msg in emitted} - {None} == set(created)
        assert state["current_response_id"] is None

    def test_second_assistant_content_block_reuses_response_not_item(self):
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"
        state = self._state()

        first = self._apply(
            config, logging_obj, state, {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}
        )
        response_id = state["current_response_id"]
        first_item = state["current_output_item_id"]
        assert sum(1 for msg in first["response"] if msg["type"] == "response.created") == 1

        self._apply(
            config,
            logging_obj,
            state,
            {"event": {"contentEnd": {"stopReason": "PARTIAL_TURN", "type": "TEXT"}}},
        )
        second = self._apply(
            config, logging_obj, state, {"event": {"contentStart": {"role": "ASSISTANT", "type": "AUDIO"}}}
        )
        assert state["current_response_id"] == response_id
        assert state["current_output_item_id"] != first_item
        assert sum(1 for msg in second["response"] if msg["type"] == "response.created") == 0
        assert sum(1 for msg in second["response"] if msg["type"] == "response.output_item.added") == 1


class TestBedrockRealtimeUsageAccounting:
    def _usage_event(
        self,
        *,
        input_speech: int,
        input_text: int,
        output_speech: int,
        output_text: int,
        total_input: int | None = None,
        total_output: int | None = None,
        total: int | None = None,
    ) -> dict:
        resolved_input = total_input if total_input is not None else input_speech + input_text
        resolved_output = total_output if total_output is not None else output_speech + output_text
        resolved_total = total if total is not None else resolved_input + resolved_output
        return {
            "event": {
                "usageEvent": {
                    "completionId": "completion_1",
                    "details": {
                        "total": {
                            "input": {"speechTokens": input_speech, "textTokens": input_text},
                            "output": {"speechTokens": output_speech, "textTokens": output_text},
                        }
                    },
                    "promptName": "prompt_1",
                    "sessionId": "session_1",
                    "totalInputTokens": resolved_input,
                    "totalOutputTokens": resolved_output,
                    "totalTokens": resolved_total,
                }
            }
        }

    def test_usage_event_fills_response_done_turn_delta(self):
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"
        state = {
            "session_configuration_request": json.dumps({"configured": True}),
            "current_output_item_id": "item_1",
            "current_response_id": "resp_1",
            "current_conversation_id": "conv_1",
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": "audio",
        }

        config.transform_realtime_response(
            json.dumps(self._usage_event(input_speech=10, input_text=2, output_speech=20, output_text=3)),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=state,
        )
        first_done = config.transform_realtime_response(
            json.dumps({"event": {"contentEnd": {"stopReason": "END_TURN", "type": "AUDIO"}}}),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=state,
        )
        done_events = [msg for msg in first_done["response"] if msg["type"] == "response.done"]
        assert len(done_events) == 1
        usage = done_events[0]["response"]["usage"]
        assert usage["input_tokens"] == 12
        assert usage["output_tokens"] == 23
        assert usage["total_tokens"] == 35
        assert usage["input_token_details"]["audio_tokens"] == 10
        assert usage["input_token_details"]["text_tokens"] == 2
        assert usage["output_token_details"]["audio_tokens"] == 20
        assert usage["output_token_details"]["text_tokens"] == 3

        config.transform_realtime_response(
            json.dumps(
                self._usage_event(
                    input_speech=15,
                    input_text=2,
                    output_speech=30,
                    output_text=3,
                    total_input=17,
                    total_output=33,
                    total=50,
                )
            ),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                **state,
                "current_output_item_id": "item_2",
                "current_response_id": "resp_2",
            },
        )
        second_done = config.transform_realtime_response(
            json.dumps({"event": {"contentEnd": {"stopReason": "END_TURN", "type": "AUDIO"}}}),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                **state,
                "current_output_item_id": "item_2",
                "current_response_id": "resp_2",
            },
        )
        second_usage = [msg for msg in second_done["response"] if msg["type"] == "response.done"][0]["response"][
            "usage"
        ]
        assert second_usage["input_tokens"] == 5
        assert second_usage["output_tokens"] == 10
        assert second_usage["total_tokens"] == 15
        assert second_usage["input_token_details"]["audio_tokens"] == 5
        assert second_usage["output_token_details"]["audio_tokens"] == 10

    def test_late_usage_event_after_response_id_cleared_emits_response_done(self):
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_late_usage"
        state = {
            "session_configuration_request": json.dumps({"configured": True}),
            "current_output_item_id": "item_1",
            "current_response_id": "resp_1",
            "current_conversation_id": "conv_1",
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": "audio",
        }

        end_turn = config.transform_realtime_response(
            json.dumps({"event": {"contentEnd": {"stopReason": "END_TURN", "type": "AUDIO"}}}),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=state,
        )
        assert any(msg["type"] == "response.done" for msg in end_turn["response"])
        assert end_turn["current_response_id"] is None
        state["current_response_id"] = end_turn["current_response_id"]
        state["current_output_item_id"] = end_turn["current_output_item_id"]
        state["current_conversation_id"] = end_turn["current_conversation_id"]

        late_usage = config.transform_realtime_response(
            json.dumps(self._usage_event(input_speech=10, input_text=2, output_speech=20, output_text=3)),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=state,
        )
        done_events = [msg for msg in late_usage["response"] if msg["type"] == "response.done"]
        assert len(done_events) == 1
        usage = done_events[0]["response"]["usage"]
        assert usage["input_tokens"] == 12
        assert usage["output_tokens"] == 23
        assert usage["total_tokens"] == 35
        assert not config.has_unbilled_usage()

    def test_tool_content_end_with_end_turn_stop_reason_emits_response_done(self):
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_tool_end_turn"
        state = {
            "session_configuration_request": json.dumps({"configured": True}),
            "current_output_item_id": "item_tool",
            "current_response_id": "resp_tool",
            "current_conversation_id": "conv_tool",
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        }

        config.transform_realtime_response(
            json.dumps(self._usage_event(input_speech=4, input_text=1, output_speech=0, output_text=2)),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=state,
        )
        tool_end = config.transform_realtime_response(
            json.dumps({"event": {"contentEnd": {"stopReason": "END_TURN", "type": "TOOL"}}}),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=state,
        )
        done_events = [msg for msg in tool_end["response"] if msg["type"] == "response.done"]
        assert len(done_events) == 1
        assert done_events[0]["response"]["id"] == "resp_tool"
        assert done_events[0]["response"]["usage"]["input_tokens"] == 5
        assert done_events[0]["response"]["usage"]["output_tokens"] == 2
        assert tool_end["current_response_id"] is None
        assert not config.has_unbilled_usage()

    def test_tool_turn_does_not_double_bill_across_late_usage_and_completion_end(self):
        """The tool response.done, a later usageEvent, and completionEnd must each bill once."""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_tool_usage"
        state = {
            "session_configuration_request": json.dumps({"configured": True}),
            "current_output_item_id": "item_1",
            "current_response_id": "resp_1",
            "current_conversation_id": "conv_1",
            "current_delta_chunks": None,
            "current_item_chunks": [],
            "current_delta_type": None,
        }

        config.transform_realtime_response(
            json.dumps(self._usage_event(input_speech=4, input_text=0, output_speech=6, output_text=0)),
            "amazon.nova-2-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=state,
        )
        tool_result = config.transform_realtime_response(
            json.dumps(
                {
                    "event": {
                        "toolUse": {
                            "toolUseId": "tool_1",
                            "toolName": "get_weather",
                            "content": json.dumps({"location": "Seattle"}),
                        }
                    }
                }
            ),
            "amazon.nova-2-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=state,
        )
        tool_usage = _only(tool_result["response"], "response.done")["response"]["usage"]
        assert tool_usage["input_tokens"] == 4
        assert tool_usage["output_tokens"] == 6
        assert not config.has_unbilled_usage()
        state["current_response_id"] = tool_result["current_response_id"]
        state["current_output_item_id"] = tool_result["current_output_item_id"]

        # Cumulative usage grows after the tool turn; only the delta may be billed again.
        late = config.transform_realtime_response(
            json.dumps(self._usage_event(input_speech=10, input_text=0, output_speech=15, output_text=0)),
            "amazon.nova-2-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=state,
        )
        late_usage = _only(late["response"], "response.done")["response"]["usage"]
        assert late_usage["input_tokens"] == 6
        assert late_usage["output_tokens"] == 9
        state["current_response_id"] = late["current_response_id"]

        completion_end = config.transform_realtime_response(
            json.dumps({"event": {"completionEnd": {}}}),
            "amazon.nova-2-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input=state,
        )
        assert completion_end["response"] == []
        assert not config.has_unbilled_usage()
        assert config.flush_pending_usage_as_response_done(None, None) == []

    def test_flush_pending_usage_on_session_close(self):
        config = BedrockRealtimeConfig()
        config.record_usage_event(
            self._usage_event(input_speech=8, input_text=1, output_speech=16, output_text=2)["event"]["usageEvent"]
        )
        assert config.has_unbilled_usage()

        flushed = config.flush_pending_usage_as_response_done(None, None)
        assert len(flushed) == 1
        assert flushed[0]["type"] == "response.done"
        usage = flushed[0]["response"]["usage"]
        assert usage["input_tokens"] == 9
        assert usage["output_tokens"] == 18
        assert usage["total_tokens"] == 27
        assert not config.has_unbilled_usage()
        assert config.flush_pending_usage_as_response_done(None, None) == []

    def test_completion_end_with_unbilled_usage_mints_response_done(self):
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_completion_end"
        config.record_usage_event(
            self._usage_event(input_speech=3, input_text=0, output_speech=6, output_text=0)["event"]["usageEvent"]
        )

        result = config.transform_realtime_response(
            json.dumps({"event": {"completionEnd": {}}}),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": None,
                "current_item_chunks": None,
                "current_delta_type": None,
            },
        )
        done_events = [msg for msg in result["response"] if msg["type"] == "response.done"]
        assert len(done_events) == 1
        assert done_events[0]["response"]["usage"]["input_tokens"] == 3
        assert done_events[0]["response"]["usage"]["output_tokens"] == 6
        assert not config.has_unbilled_usage()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
