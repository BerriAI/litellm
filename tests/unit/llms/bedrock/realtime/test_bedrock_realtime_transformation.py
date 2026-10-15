import json
from typing import Final
from unittest.mock import MagicMock

import pytest


import base64

from litellm.llms.bedrock.realtime.transformation import (
    TRIGGER_LEADING_SILENCE,
    TRIGGER_TRAILING_SILENCE,
    BedrockRealtimeConfig,
)
from litellm.llms.bedrock.realtime.trigger_audio import ready_trigger_pcm
from litellm.types.llms.openai import OpenAIRealtimeEventTypes

OBJECT_PARAMETERS: Final = {"type": "object"}
CONFLICT_TOOL: Final = {"type": "function", "name": "get_weather", "parameters": OBJECT_PARAMETERS}


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

    def test_transform_tools_flat_realtime_shape(self):
        """The flat OpenAI Realtime tool shape {type: function, name, parameters}
        produces the same toolSpec as the nested chat-completions shape (LIT-9465)"""
        config = BedrockRealtimeConfig()

        nested = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        flat = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]

        assert config._transform_tools_to_bedrock_format(flat) == config._transform_tools_to_bedrock_format(nested)

        flat_tools = config._transform_tools_to_bedrock_format(flat)
        assert flat_tools[0]["toolSpec"]["name"] == "get_weather"
        assert flat_tools[0]["toolSpec"]["description"] == "Get current weather for a city"
        schema = json.loads(flat_tools[0]["toolSpec"]["inputSchema"]["json"])
        assert schema["required"] == ["location"]

        session_config = config.session_configuration_request("amazon.nova-2-sonic-v1:0", tools=flat)
        prompt_start = json.loads(session_config)["prompt_start"]["event"]["promptStart"]
        assert prompt_start["toolConfiguration"]["tools"][0]["toolSpec"]["name"] == "get_weather"

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

    def test_session_update_without_instructions_sends_empty_system_prompt(self):
        config = BedrockRealtimeConfig()
        session_update = {
            "type": "session.update",
            "session": {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]
            },
        }

        messages = config.transform_session_update_event(session_update)
        event_names = [next(iter(json.loads(message)["event"])) for message in messages]
        assert event_names[:3] == ["sessionStart", "promptStart", "contentStart"]
        system_content_start = json.loads(messages[2])["event"]["contentStart"]
        assert system_content_start["role"] == "SYSTEM"
        text_input = json.loads(messages[3])["event"]["textInput"]
        assert text_input["content"] == ""
        assert json.loads(messages[4])["event"]["contentEnd"]["contentName"] == system_content_start["contentName"]

    def test_session_update_instructions_become_system_prompt(self):
        config = BedrockRealtimeConfig()
        session_update = {"type": "session.update", "session": {"instructions": "Answer tersely."}}

        messages = config.transform_session_update_event(session_update)
        text_input = json.loads(messages[3])["event"]["textInput"]
        assert text_input["content"] == "Answer tersely."

    def test_second_session_update_sends_no_duplicate_prompt_start(self):
        config = BedrockRealtimeConfig()
        session_update = {"type": "session.update", "session": {"instructions": "You are helpful."}}

        first = config.transform_session_update_event(session_update)
        assert any("promptStart" in message for message in first)
        assert config.transform_session_update_event(session_update) == []

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
        """toolUse is forwarded with its arguments even when no assistant response is
        active yet (Bedrock Nova Sonic sends the call inside a role TOOL block before
        any ASSISTANT contentStart, LIT-9465)"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        tool_use_message = {
            "event": {
                "toolUse": {
                    "completionId": "comp_1",
                    "content": '{"location":"New York"}',
                    "contentId": "content_1",
                    "promptName": "prompt_1",
                    "role": "TOOL",
                    "sessionId": "sess_1",
                    "toolName": "get_weather",
                    "toolUseId": "tool_call_123",
                }
            }
        }

        result = config.transform_realtime_response(
            json.dumps(tool_use_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": "item_message_1",
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )

        assert [e["type"] for e in result["response"]] == [
            "response.created",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
        ]
        response_created = result["response"][0]
        assert response_created["response"]["id"] == result["current_response_id"]

        arguments_delta = result["response"][2]
        assert arguments_delta["delta"] == '{"location":"New York"}'
        assert arguments_delta["call_id"] == "tool_call_123"
        assert arguments_delta["item_id"] == result["response"][1]["item"]["id"]
        assert arguments_delta["response_id"] == result["current_response_id"]

        function_call = result["response"][3]
        assert function_call["call_id"] == "tool_call_123"
        assert function_call["name"] == "get_weather"
        assert function_call["arguments"] == '{"location":"New York"}'
        assert function_call["response_id"] == result["current_response_id"]
        assert function_call["item_id"] != "item_message_1"
        assert function_call["item_id"] == result["response"][1]["item"]["id"]
        assert result["response"][1]["item"]["type"] == "function_call"

        done_item = result["response"][4]["item"]
        assert done_item["type"] == "function_call"
        assert done_item["status"] == "completed"
        assert done_item["call_id"] == "tool_call_123"
        assert done_item["arguments"] == '{"location":"New York"}'

    def test_tool_use_reuses_active_response(self):
        """A toolUse arriving while an assistant response is in flight must not emit a
        second response.created and must reuse the active response id"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        tool_use_message = {
            "event": {
                "toolUse": {
                    "content": '{"location":"New York"}',
                    "toolName": "get_weather",
                    "toolUseId": "tool_call_123",
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

        assert [e["type"] for e in result["response"]] == [
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
        ]
        assert result["current_response_id"] == "resp_123"
        assert all(e.get("response_id") == "resp_123" for e in result["response"])

    def test_transform_content_end_text(self):
        """Test contentEnd for text response"""
        config = BedrockRealtimeConfig()
        config._assistant_block_open = True
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


class TestBedrockRealtimeUserEventsAndUsage:
    """Regression tests for #38346 (USER ASR transcripts, speech boundary events,
    usage propagation, duplicate response.created) and for FINAL assistant text
    blocks replaying every sentence after the audio turn ended"""

    @staticmethod
    def _run(config, messages):
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"
        state = {
            "session_configuration_request": json.dumps({"configured": True}),
            "current_output_item_id": None,
            "current_response_id": None,
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        }
        all_events = []
        for msg in messages:
            result = config.transform_realtime_response(
                json.dumps(msg),
                "amazon.nova-2-sonic-v1:0",
                logging_obj,
                realtime_response_transform_input=dict(state),
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
        return all_events

    def test_user_speech_start_and_stop_events(self):
        events = self._run(
            BedrockRealtimeConfig(),
            [{"event": {"userSpeechStart": {}}}, {"event": {"userSpeechEnd": {}}}],
        )
        assert [e["type"] for e in events] == [
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
        ]
        assert all(e["event_id"] and e["item_id"] for e in events)
        assert events[0]["item_id"] == events[1]["item_id"]

    def test_utterance_lifecycle_shares_one_item_id(self):
        events = self._run(
            BedrockRealtimeConfig(),
            [
                {"event": {"userSpeechStart": {}}},
                {"event": {"userSpeechEnd": {}}},
                {
                    "event": {
                        "contentStart": {
                            "role": "USER",
                            "type": "TEXT",
                            "additionalModelFields": json.dumps({"generationStage": "FINAL"}),
                        }
                    }
                },
                {"event": {"textOutput": {"content": "ready"}}},
                {"event": {"contentEnd": {"stopReason": "PARTIAL_TURN"}}},
            ],
        )
        item_ids = {e["item_id"] for e in events if "item_id" in e}
        assert len(item_ids) == 1

    def test_new_utterance_gets_new_item_id(self):
        config = BedrockRealtimeConfig()
        first = self._run(config, [{"event": {"userSpeechStart": {}}}, {"event": {"userSpeechEnd": {}}}])
        second = self._run(config, [{"event": {"userSpeechStart": {}}}, {"event": {"userSpeechEnd": {}}}])
        assert first[0]["item_id"] == first[1]["item_id"]
        assert second[0]["item_id"] == second[1]["item_id"]
        assert first[0]["item_id"] != second[0]["item_id"]

    def test_user_transcript_emits_input_audio_transcription_events(self):
        events = self._run(
            BedrockRealtimeConfig(),
            [
                {
                    "event": {
                        "contentStart": {
                            "role": "USER",
                            "type": "TEXT",
                            "additionalModelFields": json.dumps({"generationStage": "FINAL"}),
                        }
                    }
                },
                {"event": {"textOutput": {"content": "ready"}}},
                {"event": {"contentEnd": {"stopReason": "PARTIAL_TURN"}}},
            ],
        )
        deltas = [e for e in events if e["type"] == "conversation.item.input_audio_transcription.delta"]
        completed = [e for e in events if e["type"] == "conversation.item.input_audio_transcription.completed"]
        assert len(deltas) == 1 and deltas[0]["delta"] == "ready"
        assert len(completed) == 1 and completed[0]["transcript"] == "ready"
        assert deltas[0]["item_id"] == completed[0]["item_id"]
        assert not any(e["type"] == "response.text.delta" for e in events)

    def test_speculative_user_transcript_emits_delta_only(self):
        events = self._run(
            BedrockRealtimeConfig(),
            [
                {
                    "event": {
                        "contentStart": {
                            "role": "USER",
                            "type": "TEXT",
                            "additionalModelFields": json.dumps({"generationStage": "SPECULATIVE"}),
                        }
                    }
                },
                {"event": {"textOutput": {"content": "rea"}}},
            ],
        )
        assert [e["type"] for e in events] == ["conversation.item.input_audio_transcription.delta"]

    def test_user_transcript_state_resets_on_content_end(self):
        events = self._run(
            BedrockRealtimeConfig(),
            [
                {"event": {"contentStart": {"role": "USER", "type": "TEXT"}}},
                {"event": {"contentEnd": {"stopReason": "PARTIAL_TURN"}}},
                {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}},
                {"event": {"textOutput": {"content": "Hi there"}}},
            ],
        )
        text_deltas = [e for e in events if e["type"] == "response.text.delta"]
        assert len(text_deltas) == 1 and text_deltas[0]["delta"] == "Hi there"
        assert not any(e["type"].startswith("conversation.item.input_audio_transcription") for e in events)

    def test_response_created_emitted_once_per_response(self):
        events = self._run(
            BedrockRealtimeConfig(),
            [
                {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}},
                {"event": {"textOutput": {"content": "Hi"}}},
                {"event": {"contentEnd": {"stopReason": "PARTIAL_TURN"}}},
                {"event": {"contentStart": {"role": "ASSISTANT", "type": "AUDIO"}}},
            ],
        )
        assert sum(1 for e in events if e["type"] == "response.created") == 1

    def test_usage_event_propagates_to_response_done(self):
        events = self._run(
            BedrockRealtimeConfig(),
            [
                {
                    "event": {
                        "usageEvent": {
                            "totalInputTokens": 25,
                            "totalOutputTokens": 40,
                            "totalTokens": 65,
                            "details": {
                                "total": {
                                    "input": {"speechTokens": 20, "textTokens": 5},
                                    "output": {"speechTokens": 30, "textTokens": 10},
                                }
                            },
                        }
                    }
                },
                {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}},
                {"event": {"textOutput": {"content": "Hi"}}},
                {"event": {"contentEnd": {"stopReason": "END_TURN"}}},
            ],
        )
        done_events = [e for e in events if e["type"] == "response.done"]
        assert len(done_events) == 1
        usage = done_events[0]["response"]["usage"]
        assert usage["input_tokens"] == 25
        assert usage["output_tokens"] == 40
        assert usage["total_tokens"] == 65
        assert usage["input_token_details"]["audio_tokens"] == 20
        assert usage["input_token_details"]["text_tokens"] == 5
        assert usage["output_token_details"]["audio_tokens"] == 30
        assert usage["output_token_details"]["text_tokens"] == 10

    def test_response_done_without_usage_event_reports_zero_usage(self):
        events = self._run(
            BedrockRealtimeConfig(),
            [
                {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}},
                {"event": {"textOutput": {"content": "Hi"}}},
                {"event": {"contentEnd": {"stopReason": "END_TURN"}}},
            ],
        )
        done_events = [e for e in events if e["type"] == "response.done"]
        assert len(done_events) == 1
        usage = done_events[0]["response"]["usage"]
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["total_tokens"] == 0

    @staticmethod
    def _usage_event(total_input, total_output, in_speech, in_text, out_speech, out_text):
        return {
            "event": {
                "usageEvent": {
                    "totalInputTokens": total_input,
                    "totalOutputTokens": total_output,
                    "totalTokens": total_input + total_output,
                    "details": {
                        "total": {
                            "input": {"speechTokens": in_speech, "textTokens": in_text},
                            "output": {"speechTokens": out_speech, "textTokens": out_text},
                        }
                    },
                }
            }
        }

    _ASSISTANT_TURN = (
        {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}},
        {"event": {"textOutput": {"content": "Hi"}}},
        {"event": {"contentEnd": {"stopReason": "END_TURN"}}},
    )

    def test_multi_turn_usage_reports_per_response_deltas_not_cumulative_totals(self):
        events = self._run(
            BedrockRealtimeConfig(),
            [
                self._usage_event(25, 40, in_speech=20, in_text=5, out_speech=30, out_text=10),
                *self._ASSISTANT_TURN,
                self._usage_event(40, 100, in_speech=30, in_text=10, out_speech=75, out_text=25),
                *self._ASSISTANT_TURN,
            ],
        )
        usages = [e["response"]["usage"] for e in events if e["type"] == "response.done"]
        assert len(usages) == 2
        assert (usages[0]["input_tokens"], usages[0]["output_tokens"], usages[0]["total_tokens"]) == (25, 40, 65)
        assert (usages[1]["input_tokens"], usages[1]["output_tokens"], usages[1]["total_tokens"]) == (15, 60, 75)
        assert usages[1]["input_token_details"] == {"audio_tokens": 10, "text_tokens": 5, "cached_tokens": 0}
        assert usages[1]["output_token_details"] == {"audio_tokens": 45, "text_tokens": 15}
        assert sum(u["total_tokens"] for u in usages) == 140

    def test_usage_reported_after_last_response_done_flushes_as_logged_only_done(self):
        config = BedrockRealtimeConfig()
        self._run(
            config,
            [
                self._usage_event(25, 40, in_speech=20, in_text=5, out_speech=30, out_text=10),
                *self._ASSISTANT_TURN,
            ],
        )
        assert config.leftover_usage_done_events() == ()

        self._run(config, [self._usage_event(25, 46, in_speech=20, in_text=5, out_speech=30, out_text=16)])
        leftover = config.leftover_usage_done_events()
        assert len(leftover) == 1
        assert leftover[0]["type"] == "response.done"
        usage = leftover[0]["response"]["usage"]
        assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (0, 6, 6)
        assert usage["output_token_details"] == {"audio_tokens": 0, "text_tokens": 6}
        assert config.leftover_usage_done_events() == ()

    def test_final_transcript_fragments_emit_one_completed_with_full_transcript(self):
        events = self._run(
            BedrockRealtimeConfig(),
            [
                {
                    "event": {
                        "contentStart": {
                            "role": "USER",
                            "type": "TEXT",
                            "additionalModelFields": json.dumps({"generationStage": "FINAL"}),
                        }
                    }
                },
                {"event": {"textOutput": {"content": "What is the "}}},
                {"event": {"textOutput": {"content": "capital of France?"}}},
                {"event": {"contentEnd": {"stopReason": "PARTIAL_TURN"}}},
            ],
        )
        deltas = [e for e in events if e["type"] == "conversation.item.input_audio_transcription.delta"]
        completed = [e for e in events if e["type"] == "conversation.item.input_audio_transcription.completed"]
        assert [d["delta"] for d in deltas] == ["What is the ", "capital of France?"]
        assert len(completed) == 1
        assert completed[0]["transcript"] == "What is the capital of France?"
        assert {e["item_id"] for e in deltas + completed} == {completed[0]["item_id"]}

    def test_speculative_transcript_block_end_emits_no_completed(self):
        events = self._run(
            BedrockRealtimeConfig(),
            [
                {
                    "event": {
                        "contentStart": {
                            "role": "USER",
                            "type": "TEXT",
                            "additionalModelFields": json.dumps({"generationStage": "SPECULATIVE"}),
                        }
                    }
                },
                {"event": {"textOutput": {"content": "rea"}}},
                {"event": {"contentEnd": {"stopReason": "PARTIAL_TURN"}}},
            ],
        )
        assert [e["type"] for e in events] == ["conversation.item.input_audio_transcription.delta"]

    @staticmethod
    def _assistant_text_block(stage: str, text: str, stop_reason: str) -> list[dict[str, object]]:
        return [
            {
                "event": {
                    "contentStart": {
                        "role": "ASSISTANT",
                        "type": "TEXT",
                        "additionalModelFields": json.dumps({"generationStage": stage}),
                    }
                }
            },
            {"event": {"textOutput": {"content": text}}},
            {"event": {"contentEnd": {"stopReason": stop_reason}}},
        ]

    @staticmethod
    def _assistant_audio_block(stop_reason: str) -> list[dict[str, object]]:
        return [
            {"event": {"contentStart": {"role": "ASSISTANT", "type": "AUDIO"}}},
            {"event": {"audioOutput": {"content": base64.b64encode(b"\x00\x01" * 8).decode()}}},
            {"event": {"contentEnd": {"stopReason": stop_reason}}},
        ]

    def test_final_assistant_text_blocks_do_not_replay_the_spoken_sentences(self):
        sentences: Final = ["I can't check live weather.", " Try a weather app.", " It will have the forecast."]
        events: Final = self._run(
            BedrockRealtimeConfig(),
            self._assistant_text_block("SPECULATIVE", sentences[0], "PARTIAL_TURN")
            + self._assistant_audio_block("PARTIAL_TURN")
            + self._assistant_text_block("SPECULATIVE", sentences[1], "PARTIAL_TURN")
            + self._assistant_audio_block("PARTIAL_TURN")
            + self._assistant_text_block("SPECULATIVE", sentences[2], "PARTIAL_TURN")
            + self._assistant_audio_block("END_TURN")
            + self._assistant_text_block("FINAL", sentences[0], "END_TURN")
            + self._assistant_text_block("FINAL", sentences[1], "PARTIAL_TURN")
            + self._assistant_text_block("FINAL", sentences[2], "PARTIAL_TURN"),
        )
        types: Final = [e["type"] for e in events]
        assert [e["text"] for e in events if e["type"] == "response.text.done"] == sentences
        assert types.count("response.created") == 1
        assert types.count("response.done") == 1
        assert types[-1] == "response.done"
        response_ids: Final = {e["response_id"] for e in events if e["type"] == "response.text.delta"}
        assert response_ids == {events[0]["response"]["id"]}

    def test_final_assistant_text_after_end_turn_opens_no_response(self):
        events: Final = self._run(
            BedrockRealtimeConfig(),
            self._assistant_text_block("FINAL", "Already spoken.", "END_TURN"),
        )
        assert events == []

    def test_speculative_text_after_a_final_block_is_forwarded(self):
        config: Final = BedrockRealtimeConfig()
        self._run(config, self._assistant_text_block("FINAL", "Already spoken.", "END_TURN"))
        events: Final = self._run(config, self._assistant_text_block("SPECULATIVE", "Next turn.", "PARTIAL_TURN"))
        assert [e["type"] for e in events][:1] == ["response.created"]
        assert [e["text"] for e in events if e["type"] == "response.text.done"] == ["Next turn."]

    def test_barge_in_marker_in_final_block_is_dropped_and_the_next_answer_still_flows(self):
        events: Final = self._run(
            BedrockRealtimeConfig(),
            self._assistant_text_block("SPECULATIVE", "I can't check live weather.", "PARTIAL_TURN")
            + self._assistant_audio_block("INTERRUPTED")
            + self._assistant_text_block("FINAL", '{ "interrupted" : true }', "INTERRUPTED")
            + self._assistant_text_block("FINAL", "I can't check live weather.", "INTERRUPTED")
            + self._assistant_text_block("SPECULATIVE", "Still no live weather.", "PARTIAL_TURN")
            + self._assistant_audio_block("END_TURN"),
        )
        types: Final = [e["type"] for e in events]
        assert [e["text"] for e in events if e["type"] == "response.text.done"] == [
            "I can't check live weather.",
            "Still no live weather.",
        ]
        assert types.count("response.created") == 1
        assert types.count("response.done") == 1

    def test_nova2_sonic_tool_call_sequence_forwards_each_call_with_arguments(self):
        """The real Nova 2 Sonic event sequence for one question: a USER transcript
        block, two role TOOL blocks each carrying a toolUse with a distinct toolUseId,
        then the post-tool-result ASSISTANT answer (LIT-9465)"""
        args_json: Final = '{"location":"New York"}'
        events: Final = self._run(
            BedrockRealtimeConfig(),
            [
                {
                    "event": {
                        "contentStart": {
                            "role": "USER",
                            "type": "TEXT",
                            "additionalModelFields": json.dumps({"generationStage": "FINAL"}),
                        }
                    }
                },
                {"event": {"textOutput": {"content": "ready"}}},
                {"event": {"contentEnd": {"stopReason": "PARTIAL_TURN", "type": "TEXT"}}},
            ]
            + self._tool_block("tooluse_A", args_json)
            + self._tool_block("tooluse_B", args_json)
            + self._assistant_text_block(
                "SPECULATIVE", "The current weather in New York is sunny and 72 degrees.", "END_TURN"
            ),
        )

        function_calls: Final = [e for e in events if e["type"] == "response.function_call_arguments.done"]
        assert [e["call_id"] for e in function_calls] == ["tooluse_A", "tooluse_B"]
        assert all(e["name"] == "get_weather" for e in function_calls)
        assert all(e["arguments"] == args_json for e in function_calls)
        assert function_calls[0]["item_id"] != function_calls[1]["item_id"]

        response_dones: Final = [e for e in events if e["type"] == "response.done"]
        assert len(response_dones) == 3
        tool_call_dones: Final = [
            d for d in response_dones if any(o["type"] == "function_call" for o in d["response"]["output"])
        ]
        assert len(tool_call_dones) == 2
        assert tool_call_dones[0]["response"]["output"][0]["call_id"] == "tooluse_A"
        assert tool_call_dones[1]["response"]["output"][0]["call_id"] == "tooluse_B"
        for done, call in zip(tool_call_dones, function_calls):
            assert len(done["response"]["output"]) == 1
            assert done["response"]["output"][0]["type"] == "function_call"
            assert done["response"]["output"][0]["status"] == "completed"
            assert done["response"]["id"] == call["response_id"]
        final_output: Final = response_dones[2]["response"]["output"]
        assert [o["type"] for o in final_output] == ["message"]
        assert final_output[0]["status"] == "completed"

        response_ids: Final = {d["response"]["id"] for d in response_dones}
        assert len(response_ids) == 3

    @staticmethod
    def _tool_block(tool_use_id: str, arguments: str = '{"city":"Paris"}') -> list[dict[str, object]]:
        return [
            {"event": {"contentStart": {"role": "TOOL", "type": "TOOL", "contentId": f"c_{tool_use_id}"}}},
            {
                "event": {
                    "toolUse": {
                        "content": arguments,
                        "contentId": f"c_{tool_use_id}",
                        "role": "TOOL",
                        "toolName": "get_weather",
                        "toolUseId": tool_use_id,
                    }
                }
            },
            {"event": {"contentEnd": {"stopReason": "TOOL_USE", "type": "TOOL", "contentId": f"c_{tool_use_id}"}}},
        ]

    @staticmethod
    def _item_id_of(event) -> str | None:
        if event["type"].startswith("response.output_item."):
            return event["item"]["id"]
        return event.get("item_id")

    def test_tool_call_after_assistant_message_uses_output_index_one(self):
        events: Final = self._run(
            BedrockRealtimeConfig(),
            self._assistant_text_block("SPECULATIVE", "Let me check.", "PARTIAL_TURN") + self._tool_block("tooluse_A"),
        )
        types: Final = [e["type"] for e in events]
        assert types.count("response.created") == 1

        added_items: Final = [e["item"]["id"] for e in events if e["type"] == "response.output_item.added"]
        message_id, call_id = added_items

        message_events: Final = [e for e in events if self._item_id_of(e) == message_id]
        call_events: Final = [e for e in events if self._item_id_of(e) == call_id]
        assert message_events and all(e["output_index"] == 0 for e in message_events)
        assert call_events and all(e["output_index"] == 1 for e in call_events)

        response_dones: Final = [e for e in events if e["type"] == "response.done"]
        assert len(response_dones) == 1
        assert [o["id"] for o in response_dones[0]["response"]["output"]] == [message_id, call_id]

    def test_tool_call_inside_open_assistant_block_defers_response_done(self):
        config: Final = BedrockRealtimeConfig()
        events: Final = self._run(
            config,
            [
                {
                    "event": {
                        "contentStart": {
                            "role": "ASSISTANT",
                            "type": "TEXT",
                            "additionalModelFields": json.dumps({"generationStage": "SPECULATIVE"}),
                        }
                    }
                },
                {"event": {"textOutput": {"content": "Let me check."}}},
            ]
            + self._tool_block("tooluse_A")
            + [
                {"event": {"textOutput": {"content": " It is 72."}}},
                {"event": {"contentEnd": {"stopReason": "PARTIAL_TURN"}}},
            ],
        )
        types: Final = [e["type"] for e in events]
        assert types.count("response.done") == 1
        assert types[-1] == "response.done"
        assert [e["text"] for e in events if e["type"] == "response.text.done"] == ["Let me check. It is 72."]

        next_events: Final = self._run(config, self._assistant_text_block("SPECULATIVE", "Sunny.", "END_TURN"))
        assert [e["text"] for e in next_events if e["type"] == "response.text.done"] == ["Sunny."]
        assert all(e["output_index"] == 0 for e in next_events if "output_index" in e)
        assert [e["response"]["id"] for e in next_events if e["type"] == "response.created"] != [
            e["response"]["id"] for e in events if e["type"] == "response.created"
        ]

    def test_response_done_message_item_carries_the_text_part(self):
        events: Final = self._run(
            BedrockRealtimeConfig(),
            self._assistant_text_block("SPECULATIVE", "It is 72.", "END_TURN"),
        )
        (done,) = [e for e in events if e["type"] == "response.done"]
        (message_item,) = [o for o in done["response"]["output"] if o["type"] == "message"]
        assert message_item["content"] == [{"type": "text", "text": "It is 72."}]

    def test_response_done_message_item_merges_text_then_audio_parts_in_order(self):
        events: Final = self._run(
            BedrockRealtimeConfig(),
            self._assistant_text_block("SPECULATIVE", "Hi.", "PARTIAL_TURN") + self._assistant_audio_block("END_TURN"),
        )
        (done,) = [e for e in events if e["type"] == "response.done"]
        assert [o["type"] for o in done["response"]["output"]] == ["message"]
        assert done["response"]["output"][0]["content"] == [
            {"type": "text", "text": "Hi."},
            {"type": "audio", "transcript": ""},
        ]

    def test_session_update_rejection_before_prompt_start(self):
        assert (
            BedrockRealtimeConfig().session_update_rejection(
                {"type": "session.update", "session": {"tools": [CONFLICT_TOOL]}}
            )
            is None
        )

    @pytest.mark.parametrize(
        "session,expects_conflict",
        [
            ({"modalities": ["text"]}, False),
            ({"tools": [CONFLICT_TOOL]}, False),
            ({"tools": [{"parameters": OBJECT_PARAMETERS, "name": "get_weather", "type": "function"}]}, False),
            ({"tools": [{"type": "function", "name": "get_time", "parameters": OBJECT_PARAMETERS}]}, True),
            ({"tools": []}, True),
            ({"instructions": "Be brief."}, True),
            ({"instructions": None}, False),
            ({"instructions": ""}, False),
            ({"voice": "amy"}, True),
            ({"output_audio_format": "g711_ulaw"}, True),
            ({"output_sample_rate_hertz": 8000}, True),
            ({"temperature": 0.1}, True),
            ({"max_response_output_tokens": 512}, True),
            ({"voice": "matthew", "temperature": 0.7}, False),
            ({"input_audio_format": "g711_alaw"}, False),
        ],
    )
    def test_session_update_rejection_locked_settings(self, session, expects_conflict):
        config: Final = BedrockRealtimeConfig()
        config.transform_session_update_event({"type": "session.update", "session": {"tools": [CONFLICT_TOOL]}})
        assert config.prompt_started

        conflict: Final = config.session_update_rejection({"type": "session.update", "session": session})
        assert (conflict.code if conflict else None) == ("session_update_not_applied" if expects_conflict else None)

    def test_session_update_with_inf_max_tokens_omits_max_tokens_from_session_start(self):
        config: Final = BedrockRealtimeConfig()
        messages: Final = config.transform_session_update_event(
            {"type": "session.update", "session": {"max_response_output_tokens": "inf", "modalities": ["text"]}}
        )
        session_start: Final = json.loads(messages[0])["event"]["sessionStart"]["inferenceConfiguration"]
        assert "maxTokens" not in session_start
        assert session_start["topP"] == 0.9
        assert session_start["temperature"] == 0.7

    @pytest.mark.parametrize(
        "first_session,later_session,expects_conflict",
        [
            ({"modalities": ["text"]}, {"max_response_output_tokens": "inf"}, True),
            ({"max_response_output_tokens": "inf"}, {"max_response_output_tokens": "inf"}, False),
            ({"max_response_output_tokens": "inf"}, {"max_response_output_tokens": 2048}, True),
        ],
    )
    def test_session_update_rejection_inf_max_tokens(self, first_session, later_session, expects_conflict):
        config: Final = BedrockRealtimeConfig()
        config.transform_session_update_event({"type": "session.update", "session": first_session})
        assert config.prompt_started

        conflict: Final = config.session_update_rejection({"type": "session.update", "session": later_session})
        assert (conflict is not None) == expects_conflict

    def test_session_update_after_start_leaves_locked_attributes_unchanged(self):
        config: Final = BedrockRealtimeConfig()
        config.transform_session_update_event({"type": "session.update", "session": {"modalities": ["text"]}})

        config.transform_session_update_event(
            {
                "type": "session.update",
                "session": {"voice": "amy", "output_audio_format": "g711_ulaw", "temperature": 0.1},
            }
        )
        assert config.voice_id == "matthew"
        assert config.output_sample_rate_hertz == 24000
        assert config.temperature == 0.7

        # Input audio settings still apply mid-session
        config.transform_session_update_event(
            {"type": "session.update", "session": {"input_audio_format": "g711_alaw"}}
        )
        assert config.input_sample_rate_hertz == 8000
        config.transform_session_update_event({"type": "session.update", "session": {"input_sample_rate_hertz": 16000}})
        assert config.input_sample_rate_hertz == 16000

    def test_stray_non_assistant_content_end_emits_no_duplicate_done_events(self):
        events: Final = self._run(
            BedrockRealtimeConfig(),
            self._assistant_text_block("SPECULATIVE", "Hi.", "PARTIAL_TURN")
            + [
                {"event": {"contentStart": {"role": "USER", "type": "AUDIO"}}},
                {"event": {"contentEnd": {}}},
            ],
        )
        assert len([e for e in events if e["type"] == "response.text.done"]) == 1
        message_item_dones: Final = [
            e for e in events if e["type"] == "response.output_item.done" and e["item"]["type"] == "message"
        ]
        assert len(message_item_dones) == 1

    @staticmethod
    def _inference_configuration(config: BedrockRealtimeConfig, session: dict[str, object]) -> dict[str, object]:
        messages: Final = config.transform_session_update_event({"type": "session.update", "session": session})
        return json.loads(messages[0])["event"]["sessionStart"]["inferenceConfiguration"]

    def test_integral_float_max_tokens_reaches_session_start_as_an_integer(self):
        inference: Final = self._inference_configuration(
            BedrockRealtimeConfig(), {"max_response_output_tokens": 4096.0, "temperature": 1}
        )
        assert inference["maxTokens"] == 4096
        assert isinstance(inference["maxTokens"], int)
        assert inference["temperature"] == 1.0

    @pytest.mark.parametrize(
        "session,param",
        [
            ({"temperature": "0.2"}, "session.temperature"),
            ({"temperature": True}, "session.temperature"),
            ({"max_response_output_tokens": "4096"}, "session.max_response_output_tokens"),
            ({"max_response_output_tokens": 4096.5}, "session.max_response_output_tokens"),
            ({"max_response_output_tokens": True}, "session.max_response_output_tokens"),
        ],
    )
    def test_malformed_number_setting_is_rejected_without_drop_params(self, session, param):
        rejection: Final = BedrockRealtimeConfig().session_update_rejection(
            {"type": "session.update", "session": session}
        )
        assert rejection is not None
        assert (rejection.code, rejection.param) == ("invalid_type", param)

    @pytest.mark.parametrize(
        "session",
        [
            {"temperature": "0.2", "max_response_output_tokens": "4096"},
            {"temperature": True, "max_response_output_tokens": 4096.5},
        ],
    )
    def test_malformed_number_setting_is_dropped_with_drop_params(self, session):
        config: Final = BedrockRealtimeConfig(drop_params=True)
        assert config.session_update_rejection({"type": "session.update", "session": session}) is None
        inference: Final = self._inference_configuration(config, session)
        assert (inference["temperature"], inference["maxTokens"]) == (0.7, 1024)

    def test_null_number_settings_count_as_unset(self):
        session: Final = {"temperature": None, "max_response_output_tokens": None}
        config: Final = BedrockRealtimeConfig()
        assert config.session_update_rejection({"type": "session.update", "session": session}) is None
        inference: Final = self._inference_configuration(config, session)
        assert (inference["temperature"], inference["maxTokens"]) == (0.7, 1024)

    def test_tool_use_without_tool_use_id_is_forwarded_with_empty_call_id(self):
        events: Final = self._run(
            BedrockRealtimeConfig(),
            [{"event": {"toolUse": {"content": '{"city":"Paris"}', "toolName": "get_weather"}}}],
        )
        (function_call,) = [e for e in events if e["type"] == "response.function_call_arguments.done"]
        assert (function_call["call_id"], function_call["arguments"]) == ("", '{"city":"Paris"}')

    def test_tool_call_without_assistant_message_stays_output_index_zero(self):
        events: Final = self._run(BedrockRealtimeConfig(), self._tool_block("tooluse_A"))
        call_events: Final = [
            e
            for e in events
            if e["type"]
            in (
                "response.output_item.added",
                "response.output_item.done",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            )
        ]
        assert call_events and all(e["output_index"] == 0 for e in call_events)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
