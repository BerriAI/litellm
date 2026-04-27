import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.gemini.realtime.transformation import GeminiRealtimeConfig
from litellm.types.llms.openai import OpenAIRealtimeStreamSessionEvents


def test_gemini_realtime_transformation_session_created():
    config = GeminiRealtimeConfig()
    assert config is not None

    session_configuration_request = {
        "model": "gemini-1.5-flash",
        "generationConfig": {"responseModalities": ["TEXT"]},
    }
    session_configuration_request_str = json.dumps(session_configuration_request)
    session_created_message = {"setupComplete": {}}

    session_created_message_str = json.dumps(session_created_message)
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "123"

    transformed_message = config.transform_realtime_response(
        session_created_message_str,
        "gemini-1.5-flash",
        logging_obj,
        realtime_response_transform_input={
            "session_configuration_request": session_configuration_request_str,
            "current_output_item_id": None,
            "current_response_id": None,
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        },
    )

    print(transformed_message)
    assert transformed_message["response"][0]["type"] == "session.created"


def test_session_created_does_not_overwrite_session_configuration_request():
    config = GeminiRealtimeConfig()

    session_configuration_request_str = json.dumps(
        {
            "setup": {
                "model": "models/gemini-2.5-flash-native-audio",
                "generationConfig": {"responseModalities": ["AUDIO"]},
            }
        }
    )
    setup_complete_message = {"setupComplete": {}}

    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_123"

    transformed = config.transform_realtime_response(
        json.dumps(setup_complete_message),
        "gemini-2.5-flash-native-audio",
        logging_obj,
        realtime_response_transform_input={
            "session_configuration_request": session_configuration_request_str,
            "current_output_item_id": None,
            "current_response_id": None,
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        },
    )

    # Must keep original setup payload (with "setup"), not overwrite with session.created event.
    assert transformed["session_configuration_request"] == session_configuration_request_str

    # Also verify emitted session.created reflects audio modality from setup payload.
    session_created = transformed["response"][0]
    assert session_created["type"] == "session.created"
    assert "audio" in session_created["session"]["modalities"]


def test_gemini_realtime_transformation_content_delta():
    config = GeminiRealtimeConfig()
    assert config is not None

    session_configuration_request = {
        "model": "gemini-1.5-flash",
        "generationConfig": {"responseModalities": ["TEXT"]},
    }
    session_configuration_request_str = json.dumps(session_configuration_request)
    session_created_message = {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {"text": "Hello, world!"},
                    {"text": "How are you?"},
                ]
            }
        }
    }

    session_created_message_str = json.dumps(session_created_message)
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id.return_value = "123"

    returned_object = config.transform_realtime_response(
        session_created_message_str,
        "gemini-1.5-flash",
        logging_obj,
        realtime_response_transform_input={
            "session_configuration_request": session_configuration_request_str,
            "current_output_item_id": None,
            "current_response_id": None,
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        },
    )
    transformed_message = returned_object["response"]
    assert isinstance(transformed_message, list)
    print(transformed_message)
    transformed_message_str = json.dumps(transformed_message)
    assert "Hello, world" in transformed_message_str
    assert "How are you?" in transformed_message_str
    print(transformed_message)

    ## assert all instances of 'event_id' are unique
    event_ids = [
        event["event_id"] for event in transformed_message if "event_id" in event
    ]
    assert len(event_ids) == len(set(event_ids))
    ## assert all instances of 'response_id' are the same
    response_ids = [
        event["response_id"] for event in transformed_message if "response_id" in event
    ]
    assert len(set(response_ids)) == 1
    ## assert all instances of 'output_item_id' are the same
    output_item_ids = [
        event["item_id"] for event in transformed_message if "item_id" in event
    ]
    assert len(set(output_item_ids)) == 1


def test_gemini_model_turn_event_mapping():
    from litellm.types.llms.openai import OpenAIRealtimeEventTypes

    config = GeminiRealtimeConfig()
    assert config is not None

    model_turn_event = {"parts": [{"text": "Hello, world!"}]}
    openai_event = config.map_model_turn_event(model_turn_event)
    assert openai_event == OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA

    model_turn_event = {
        "parts": [{"inlineData": {"mimeType": "audio/pcm", "data": "..."}}]
    }
    openai_event = config.map_model_turn_event(model_turn_event)
    assert openai_event == OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DELTA

    model_turn_event = {
        "parts": [
            {
                "text": "Hello, world!",
                "inlineData": {"mimeType": "audio/pcm", "data": "..."},
            }
        ]
    }
    openai_event = config.map_model_turn_event(model_turn_event)
    assert openai_event == OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA


def test_gemini_realtime_transformation_audio_delta():
    from litellm.types.llms.openai import OpenAIRealtimeEventTypes

    config = GeminiRealtimeConfig()
    assert config is not None

    session_configuration_request = {
        "model": "gemini-1.5-flash",
        "generationConfig": {"responseModalities": ["AUDIO"]},
    }
    session_configuration_request_str = json.dumps(session_configuration_request)

    audio_delta_event = {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {"inlineData": {"mimeType": "audio/pcm", "data": "my-audio-data"}}
                ]
            }
        }
    }

    result = config.transform_realtime_response(
        json.dumps(audio_delta_event),
        "gemini-1.5-flash",
        MagicMock(),
        realtime_response_transform_input={
            "session_configuration_request": session_configuration_request_str,
            "current_output_item_id": None,
            "current_response_id": None,
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        },
    )

    print(result)

    responses = result["response"]

    contains_audio_delta = False
    for response in responses:
        if response["type"] == OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DELTA.value:
            contains_audio_delta = True
            break
    assert contains_audio_delta, "Expected audio delta event"


def test_gemini_realtime_transformation_generation_complete():
    from litellm.types.llms.openai import OpenAIRealtimeEventTypes

    config = GeminiRealtimeConfig()
    assert config is not None

    session_configuration_request = {
        "model": "gemini-1.5-flash",
        "generationConfig": {"responseModalities": ["AUDIO"]},
    }
    session_configuration_request_str = json.dumps(session_configuration_request)

    audio_delta_event = {"serverContent": {"generationComplete": True}}

    result = config.transform_realtime_response(
        json.dumps(audio_delta_event),
        "gemini-1.5-flash",
        MagicMock(),
        realtime_response_transform_input={
            "session_configuration_request": session_configuration_request_str,
            "current_output_item_id": "my-output-item-id",
            "current_response_id": "my-response-id",
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": "audio",
        },
    )

    print(result)

    responses = result["response"]

    contains_audio_done_event = False
    for response in responses:
        if response["type"] == OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DONE.value:
            contains_audio_delta = True
            break
    assert contains_audio_delta, "Expected audio delta event"


def test_gemini_3_1_flash_live_preview_model_cost_map_entry():
    for key in (
        "gemini-3.1-flash-live-preview",
        "gemini/gemini-3.1-flash-live-preview",
    ):
        assert key in litellm.model_cost
        info = litellm.model_cost[key]
        assert "/v1/realtime" in info.get("supported_endpoints", [])
        assert info.get("max_input_tokens") == 131072
        assert info.get("max_output_tokens") == 65536
        assert "video" in info.get("supported_modalities", [])
        assert info.get("supports_function_calling") is True


def test_gemini_realtime_tool_call_transformation():
    """Test transformation of Gemini toolCall to OpenAI function_call_arguments.done format."""
    config = GeminiRealtimeConfig()
    
    # Gemini toolCall message format
    gemini_tool_call = {
        "toolCall": {
            "functionCalls": [
                {
                    "id": "call_123",
                    "name": "get_weather",
                    "args": {"location": "San Francisco", "unit": "fahrenheit"},
                }
            ]
        }
    }
    
    gemini_tool_call_str = json.dumps(gemini_tool_call)
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "test-trace-123"
    
    # Transform the toolCall message
    result = config.transform_realtime_response(
        gemini_tool_call_str,
        "gemini-2.5-flash",
        logging_obj,
        realtime_response_transform_input={
            "session_configuration_request": None,
            "current_output_item_id": "item_123",
            "current_response_id": "resp_123",
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        },
    )
    
    print("Tool call transformation result:", json.dumps(result, indent=2))
    
    # Verify the transformation
    responses = result["response"]
    assert len(responses) > 0, "Expected at least one response event"
    
    # Find the function_call_arguments.done event
    function_call_event = None
    for event in responses:
        if event.get("type") == "response.function_call_arguments.done":
            function_call_event = event
            break
    
    assert function_call_event is not None, "Expected function_call_arguments.done event"
    assert function_call_event["call_id"] == "call_123"
    assert function_call_event["name"] == "get_weather"
    assert function_call_event["response_id"] == "resp_123"
    assert function_call_event["item_id"] == "item_123_tool_0"
    assert function_call_event["output_index"] == 0
    
    # Verify arguments are properly serialized as JSON string
    args = json.loads(function_call_event["arguments"])
    assert args["location"] == "San Francisco"
    assert args["unit"] == "fahrenheit"


def test_gemini_realtime_session_update_with_tools():
    """Test transformation of OpenAI session.update with tools to Gemini setup format."""
    config = GeminiRealtimeConfig()
    
    # OpenAI format session update with tools
    session_update = {
        "type": "session.update",
        "session": {
            "instructions": "You are a helpful assistant with weather tools.",
            "temperature": 0.7,
            "max_response_output_tokens": 1024,
            "modalities": ["audio"],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a location.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city name",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["fahrenheit", "celsius"],
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
        },
    }
    
    # Transform to Gemini format (first session.update, so setup should be sent)
    messages = config.transform_realtime_request(
        json.dumps(session_update), "gemini-2.5-flash", session_configuration_request=None
    )
    
    assert len(messages) == 1, "Expected one setup message"
    
    gemini_setup = json.loads(messages[0])
    assert "setup" in gemini_setup
    
    setup_config = gemini_setup["setup"]
    
    # Verify tools are at top level, not in generationConfig
    assert "tools" in setup_config
    assert "tools" not in setup_config.get("generationConfig", {})
    
    # Verify tool structure matches Gemini format
    tools = setup_config["tools"]
    assert len(tools) == 1
    assert "function_declarations" in tools[0]
    
    function_decl = tools[0]["function_declarations"][0]
    assert function_decl["name"] == "get_weather"
    assert "Get the current weather" in function_decl["description"]
    assert "parameters" in function_decl


def test_gemini_session_update_defaults_to_audio_modality():
    config = GeminiRealtimeConfig()

    session_update = {
        "type": "session.update",
        "session": {
            "instructions": "You are a helpful assistant.",
            # No modalities on purpose
        },
    }

    messages = config.transform_realtime_request(
        json.dumps(session_update), "gemini-2.5-flash", session_configuration_request=None
    )

    assert len(messages) == 1
    setup_payload = json.loads(messages[0])["setup"]
    assert setup_payload["generationConfig"]["responseModalities"] == ["AUDIO"]


def test_gemini_requires_session_configuration_feature_flag(monkeypatch):
    config = GeminiRealtimeConfig()

    # Default behavior remains backwards-compatible (auto setup on connect)
    monkeypatch.setattr(litellm, "gemini_live_defer_setup", False, raising=False)
    assert config.requires_session_configuration() is True

    # Opt-in behavior: defer setup until client sends session.update
    monkeypatch.setattr(litellm, "gemini_live_defer_setup", True, raising=False)
    assert config.requires_session_configuration() is False


def test_gemini_realtime_function_call_output_transformation():
    """Test transformation of OpenAI function_call_output to Gemini toolResponse format."""
    config = GeminiRealtimeConfig()
    
    # OpenAI format function call output
    function_output = {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": json.dumps({
                "location": "San Francisco",
                "temperature": 72,
                "unit": "fahrenheit",
                "conditions": "sunny",
            }),
        },
    }
    
    # Transform to Gemini format
    messages = config.transform_realtime_request(
        json.dumps(function_output), "gemini-2.5-flash", session_configuration_request="existing"
    )
    
    assert len(messages) == 1, "Expected one toolResponse message"
    
    gemini_response = json.loads(messages[0])
    assert "toolResponse" in gemini_response
    
    tool_response = gemini_response["toolResponse"]
    assert "functionResponses" in tool_response
    assert len(tool_response["functionResponses"]) == 1
    
    func_response = tool_response["functionResponses"][0]
    assert func_response["id"] == "call_123"
    assert "response" in func_response
    assert func_response["response"]["temperature"] == 72
    assert func_response["response"]["conditions"] == "sunny"


def test_gemini_realtime_user_text_transformation():
    """Test transformation of OpenAI user message to Gemini clientContent format."""
    config = GeminiRealtimeConfig()
    
    # OpenAI format user message
    user_message = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What's the weather in London?"}],
        },
    }
    
    # Transform to Gemini format
    messages = config.transform_realtime_request(
        json.dumps(user_message), "gemini-2.5-flash", session_configuration_request="existing"
    )
    
    assert len(messages) == 1, "Expected one clientContent message"
    
    gemini_message = json.loads(messages[0])
    assert "clientContent" in gemini_message
    
    client_content = gemini_message["clientContent"]
    assert "turns" in client_content
    assert len(client_content["turns"]) == 1
    
    turn = client_content["turns"][0]
    assert turn["role"] == "user"
    assert len(turn["parts"]) == 1
    assert turn["parts"][0]["text"] == "What's the weather in London?"
    assert client_content["turnComplete"] is True


def test_return_new_content_delta_events_without_session_config_does_not_error():
    config = GeminiRealtimeConfig()

    events = config.return_new_content_delta_events(
        response_id="resp_1",
        output_item_id="item_1",
        conversation_id="conv_1",
        delta_type="text",
        session_configuration_request=None,
    )

    assert len(events) >= 1
    assert events[0]["type"] == "response.created"


def test_gemini_realtime_multi_tool_calls_have_unique_item_ids():
    config = GeminiRealtimeConfig()
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "test-trace-123"

    gemini_tool_call = {
        "toolCall": {
            "functionCalls": [
                {
                    "id": "call_1",
                    "name": "get_weather",
                    "args": {"location": "SF"},
                },
                {
                    "id": "call_2",
                    "name": "get_weather",
                    "args": {"location": "NYC"},
                },
            ]
        }
    }

    result = config.transform_realtime_response(
        json.dumps(gemini_tool_call),
        "gemini-2.5-flash",
        logging_obj,
        realtime_response_transform_input={
            "session_configuration_request": None,
            "current_output_item_id": "item_123",
            "current_response_id": "resp_123",
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        },
    )

    responses = [
        ev
        for ev in result["response"]
        if ev.get("type") == "response.function_call_arguments.done"
    ]
    assert len(responses) == 2
    assert responses[0]["response_id"] == "resp_123"
    assert responses[1]["response_id"] == "resp_123"
    assert responses[0]["item_id"] == "item_123_tool_0"
    assert responses[1]["item_id"] == "item_123_tool_1"
    assert responses[0]["item_id"] != responses[1]["item_id"]
    assert responses[0]["output_index"] == 0
    assert responses[1]["output_index"] == 1


def test_gemini_session_update_includes_input_audio_transcription_default():
    """Verify _handle_session_update includes inputAudioTranscription default."""
    config = GeminiRealtimeConfig()
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        },
    }

    result = config.transform_realtime_request(
        json.dumps(session_update),
        "gemini-2.5-flash",
        session_configuration_request=None,
    )

    assert len(result) == 1
    setup = json.loads(result[0])
    assert "setup" in setup
    assert "inputAudioTranscription" in setup["setup"]
    assert setup["setup"]["inputAudioTranscription"] == {}


def test_gemini_tool_call_emits_response_created_preamble():
    """Verify response.created is emitted before tool call events when response_id is None."""
    config = GeminiRealtimeConfig()
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_123"
    
    gemini_tool_call = {
        "toolCall": {
            "functionCalls": [
                {
                    "id": "call_123",
                    "name": "get_weather",
                    "args": {"location": "San Francisco", "unit": "fahrenheit"}
                }
            ]
        }
    }
    
    # Transform with current_response_id=None to trigger preamble emission
    result = config.transform_realtime_response(
        json.dumps(gemini_tool_call),
        "gemini-2.5-flash",
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
    
    responses = result["response"]
    # Should have: response.created, output_item.added, function_call_arguments.done, output_item.done, conversation.item.created, response.done
    assert len(responses) >= 6
    assert responses[0]["type"] == "response.created"
    assert "response" in responses[0]
    assert responses[0]["response"]["status"] == "in_progress"
    assert responses[1]["type"] == "response.output_item.added"
    assert responses[1]["item"]["type"] == "function_call"
    assert responses[1]["item"]["status"] == "in_progress"
    assert responses[2]["type"] == "response.function_call_arguments.done"
    assert responses[3]["type"] == "response.output_item.done"
    assert responses[3]["item"]["type"] == "function_call"
    assert responses[3]["item"]["status"] == "completed"
    assert responses[4]["type"] == "conversation.item.created"
    assert responses[4]["item"]["type"] == "function_call"
    assert responses[4]["item"]["status"] == "completed"
    assert responses[5]["type"] == "response.done"
    assert responses[5]["response"]["status"] == "completed"
    assert len(responses[5]["response"]["output"]) == 1
    assert responses[5]["response"]["output"][0]["type"] == "function_call"
