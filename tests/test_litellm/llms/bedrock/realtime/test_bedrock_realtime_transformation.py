import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

import base64

from litellm.llms.bedrock.realtime.transformation import (
    TRIGGER_LEADING_SILENCE,
    TRIGGER_TRAILING_SILENCE,
    BedrockRealtimeConfig,
)
from litellm.llms.bedrock.realtime.trigger_audio import ready_trigger_pcm
from litellm.types.llms.openai import OpenAIRealtimeEventTypes


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

    def test_transform_session_start_response(self):
        """Test sessionStart response transformation"""
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

        assert len(result["response"]) == 1
        assert result["response"][0]["type"] == "session.created"
        assert result["response"][0]["session"]["id"] == "trace_123"
        assert "model" in result["response"][0]["session"]

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

        # Check for function call event
        assert len(result["response"]) == 1
        function_call = result["response"][0]
        assert function_call["type"] == "response.function_call_arguments.done"
        assert function_call["call_id"] == "tool_call_123"
        assert function_call["name"] == "get_weather"

        # Verify arguments are properly formatted
        args = json.loads(function_call["arguments"])
        assert args["location"] == "San Francisco"

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
