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
        "setup": {
            "model": "gemini-1.5-flash",
            "generationConfig": {"responseModalities": ["TEXT"]},
        }
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

    session_created = transformed_message["response"][0]
    assert session_created["type"] == "session.created"
    # Verify the setup-wrapped configuration reaches the modality lookup so
    # the synthetic session.created reflects the cached responseModalities.
    assert session_created["session"]["modalities"] == ["text"]


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
    assert (
        transformed["session_configuration_request"]
        == session_configuration_request_str
    )

    # Also verify emitted session.created reflects audio modality from setup payload.
    session_created = transformed["response"][0]
    assert session_created["type"] == "session.created"
    assert "audio" in session_created["session"]["modalities"]


def test_gemini_realtime_transformation_content_delta():
    config = GeminiRealtimeConfig()
    assert config is not None

    session_configuration_request = {
        "setup": {
            "model": "gemini-1.5-flash",
            "generationConfig": {"responseModalities": ["TEXT"]},
        }
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
        "setup": {
            "model": "gemini-1.5-flash",
            "generationConfig": {"responseModalities": ["AUDIO"]},
        }
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
        "setup": {
            "model": "gemini-1.5-flash",
            "generationConfig": {"responseModalities": ["AUDIO"]},
        }
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
            contains_audio_done_event = True
            break
    assert contains_audio_done_event, "Expected audio done event"


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

    assert (
        function_call_event is not None
    ), "Expected function_call_arguments.done event"
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
        json.dumps(session_update),
        "gemini-2.5-flash",
        session_configuration_request=None,
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
        json.dumps(session_update),
        "gemini-2.5-flash",
        session_configuration_request=None,
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
    """Test transformation of OpenAI function_call_output to Gemini toolResponse format.

    Exercises the full production round-trip: a Gemini toolCall arrives first
    and populates the call_id -> name mapping, then the OpenAI
    function_call_output is transformed and must carry the function name back
    to Gemini in functionResponses.
    """
    config = GeminiRealtimeConfig()

    # Receive a toolCall from Gemini first to populate the call_id -> name mapping.
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_func_output"
    config.transform_realtime_response(
        json.dumps(
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call_123",
                            "name": "get_weather",
                            "args": {"location": "San Francisco"},
                        }
                    ]
                }
            }
        ),
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
    assert config._tool_call_id_to_name.get("call_123") == "get_weather"

    # OpenAI format function call output
    function_output = {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": json.dumps(
                {
                    "location": "San Francisco",
                    "temperature": 72,
                    "unit": "fahrenheit",
                    "conditions": "sunny",
                }
            ),
        },
    }

    # Transform to Gemini format
    messages = config.transform_realtime_request(
        json.dumps(function_output),
        "gemini-2.5-flash",
        session_configuration_request="existing",
    )

    assert len(messages) == 1, "Expected one toolResponse message"

    gemini_response = json.loads(messages[0])
    assert "toolResponse" in gemini_response

    tool_response = gemini_response["toolResponse"]
    assert "functionResponses" in tool_response
    assert len(tool_response["functionResponses"]) == 1

    func_response = tool_response["functionResponses"][0]
    assert func_response["id"] == "call_123"
    assert func_response["name"] == "get_weather"
    assert "response" in func_response
    assert func_response["response"]["temperature"] == 72
    assert func_response["response"]["conditions"] == "sunny"

    # A retry of the same function_call_output (e.g. a client SDK that
    # re-sends the result) must still produce a functionResponses payload
    # carrying ``name`` — the call_id → name mapping must not be evicted
    # after the first lookup.
    retry_messages = config.transform_realtime_request(
        json.dumps(function_output),
        "gemini-2.5-flash",
        session_configuration_request="existing",
    )
    retry_response = json.loads(retry_messages[0])["toolResponse"]["functionResponses"][
        0
    ]
    assert retry_response["name"] == "get_weather"


def test_gemini_realtime_user_text_transformation():
    """Test transformation of OpenAI user message to Gemini clientContent format."""
    config = GeminiRealtimeConfig()

    # OpenAI format user message
    user_message = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What's the weather in London?"}
            ],
        },
    }

    # Transform to Gemini format
    messages = config.transform_realtime_request(
        json.dumps(user_message),
        "gemini-2.5-flash",
        session_configuration_request="existing",
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
                    "args": {"location": "San Francisco", "unit": "fahrenheit"},
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
    # response.created on the tool-call path mirrors the audio/text preamble:
    # modalities/temperature/max_output_tokens are present so spec-compliant
    # clients see consistent response metadata regardless of payload type.
    assert "modalities" in responses[0]["response"]
    assert "temperature" in responses[0]["response"]
    assert "max_output_tokens" in responses[0]["response"]
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
    assert result["current_output_item_id"] is None
    assert result["current_response_id"] is None


def test_gemini_tool_call_resets_ids_for_post_tool_model_turn():
    """After tool-call response.done, a subsequent modelTurn must emit response.created."""
    config = GeminiRealtimeConfig()
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_123"

    session_configuration_request = json.dumps(
        {
            "model": "gemini-1.5-flash",
            "generationConfig": {"responseModalities": ["TEXT"]},
        }
    )

    tool_result = config.transform_realtime_response(
        json.dumps(
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call_123",
                            "name": "get_weather",
                            "args": {"location": "San Francisco"},
                        }
                    ]
                }
            }
        ),
        "gemini-2.5-flash",
        logging_obj,
        realtime_response_transform_input={
            "session_configuration_request": session_configuration_request,
            "current_output_item_id": None,
            "current_response_id": None,
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        },
    )

    tool_response_id = tool_result["response"][0]["response"]["id"]
    assert tool_result["current_output_item_id"] is None
    assert tool_result["current_response_id"] is None

    post_tool_result = config.transform_realtime_response(
        json.dumps(
            {
                "serverContent": {
                    "modelTurn": {"parts": [{"text": "The weather is sunny."}]}
                }
            }
        ),
        "gemini-2.5-flash",
        logging_obj,
        realtime_response_transform_input={
            "session_configuration_request": session_configuration_request,
            "current_output_item_id": tool_result["current_output_item_id"],
            "current_response_id": tool_result["current_response_id"],
            "current_conversation_id": tool_result["current_conversation_id"],
            "current_delta_chunks": tool_result["current_delta_chunks"],
            "current_item_chunks": tool_result["current_item_chunks"],
            "current_delta_type": tool_result["current_delta_type"],
        },
    )

    post_tool_events = post_tool_result["response"]
    assert post_tool_events[0]["type"] == "response.created"
    assert post_tool_events[0]["response"]["id"] != tool_response_id
    assert (
        post_tool_result["current_response_id"] == post_tool_events[0]["response"]["id"]
    )


def test_gemini_empty_tool_call_does_not_crash_websocket():
    """A toolCall payload with no functionCalls must not raise the
    'Unknown message type' guard — that would terminate the WebSocket session
    on what is at worst a benign no-op from Gemini."""
    config = GeminiRealtimeConfig()
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_empty_tool_call"

    result = config.transform_realtime_response(
        json.dumps({"toolCall": {"functionCalls": []}}),
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

    assert result["response"] == []
    assert result["current_response_id"] is None
    assert result["current_output_item_id"] is None


def test_gemini_function_call_output_includes_name():
    """Verify function_call_output includes name field from stored mapping."""
    config = GeminiRealtimeConfig()

    # First, receive a toolCall from Gemini (this stores the call_id → name mapping)
    gemini_tool_call = {
        "toolCall": {
            "functionCalls": [
                {
                    "id": "call_123",
                    "name": "get_weather",
                    "args": {"location": "San Francisco"},
                }
            ]
        }
    }

    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_123"

    config.transform_realtime_response(
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

    # Verify mapping was stored
    assert "call_123" in config._tool_call_id_to_name
    assert config._tool_call_id_to_name["call_123"] == "get_weather"

    # Now send a function_call_output back (this should include the name)
    function_output = {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": json.dumps({"result": "72 degrees"}),
        },
    }

    result = config.transform_realtime_request(
        json.dumps(function_output),
        "gemini-2.5-flash",
        session_configuration_request="{}",
    )

    assert len(result) == 1
    tool_response = json.loads(result[0])
    assert "toolResponse" in tool_response
    assert "functionResponses" in tool_response["toolResponse"]
    assert len(tool_response["toolResponse"]["functionResponses"]) == 1

    function_response = tool_response["toolResponse"]["functionResponses"][0]
    assert function_response["id"] == "call_123"
    assert function_response["name"] == "get_weather"  # ✅ Name is included
    assert "response" in function_response


def test_gemini_subsequent_session_update_forwards_tools_merged_with_original_setup():
    """A client session.update sent after the auto-setup must forward tools/
    instructions as a follow-up setup, merged with the original setup so we
    don't drop the pre-existing config (model, generationConfig, etc.)."""
    config = GeminiRealtimeConfig()

    original_setup = {
        "setup": {
            "model": "models/gemini-2.5-flash-native-audio",
            "generationConfig": {"responseModalities": ["AUDIO"]},
            "inputAudioTranscription": {},
            "systemInstruction": {"role": "user", "parts": [{"text": "original"}]},
        }
    }

    session_update = {
        "type": "session.update",
        "session": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            "instructions": "Be concise.",
        },
    }

    messages = config.transform_realtime_request(
        json.dumps(session_update),
        "gemini-2.5-flash-native-audio",
        session_configuration_request=json.dumps(original_setup),
    )

    assert len(messages) == 1
    follow_up = json.loads(messages[0])["setup"]
    assert "tools" in follow_up
    assert follow_up["tools"][0]["function_declarations"][0]["name"] == "get_weather"
    # systemInstruction overwritten by client's instructions
    assert follow_up["systemInstruction"]["parts"][0]["text"] == "Be concise."
    # Original generationConfig / model / inputAudioTranscription preserved
    assert follow_up["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert follow_up["model"] == "models/gemini-2.5-flash-native-audio"
    assert follow_up["inputAudioTranscription"] == {}


def test_gemini_subsequent_session_update_with_turn_detection_only_preserves_original_tools():
    """A subsequent session.update carrying only turn_detection (the
    guardrail-injected disable) must keep the original tools/generationConfig."""
    config = GeminiRealtimeConfig()

    original_setup = {
        "setup": {
            "model": "models/gemini-2.5-flash-native-audio",
            "generationConfig": {"responseModalities": ["AUDIO"]},
            "inputAudioTranscription": {},
            "tools": [
                {
                    "function_declarations": [
                        {"name": "lookup", "description": "x", "parameters": {}}
                    ]
                }
            ],
        }
    }

    session_update = {
        "type": "session.update",
        "session": {"turn_detection": {"create_response": False}},
    }

    messages = config.transform_realtime_request(
        json.dumps(session_update),
        "gemini-2.5-flash-native-audio",
        session_configuration_request=json.dumps(original_setup),
    )

    assert len(messages) == 1
    follow_up = json.loads(messages[0])["setup"]
    assert follow_up["tools"] == original_setup["setup"]["tools"]
    assert (
        follow_up["realtimeInputConfig"]["automaticActivityDetection"]["disabled"]
        is True
    )


def test_gemini_follow_up_session_update_preserves_response_modalities_on_partial_generation_config():
    """A follow-up session.update that only sets `temperature` (or any other
    generationConfig sub-field) must not wipe `responseModalities` from the
    original setup."""
    config = GeminiRealtimeConfig()

    original_setup = {
        "setup": {
            "model": "models/gemini-2.5-flash-native-audio",
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "maxOutputTokens": 2048,
            },
            "inputAudioTranscription": {},
        }
    }

    session_update = {
        "type": "session.update",
        "session": {"temperature": 0.7},
    }

    messages = config.transform_realtime_request(
        json.dumps(session_update),
        "gemini-2.5-flash-native-audio",
        session_configuration_request=json.dumps(original_setup),
    )

    follow_up = json.loads(messages[0])["setup"]
    assert follow_up["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert follow_up["generationConfig"]["maxOutputTokens"] == 2048
    assert follow_up["generationConfig"]["temperature"] == 0.7


def test_gemini_subsequent_session_update_preserves_automatic_activity_detection_subfields():
    config = GeminiRealtimeConfig()

    original_setup = {
        "setup": {
            "model": "models/gemini-2.5-flash-native-audio",
            "generationConfig": {"responseModalities": ["AUDIO"]},
            "realtimeInputConfig": {
                "automaticActivityDetection": {
                    "disabled": False,
                    "silenceDurationMs": 500,
                    "prefixPaddingMs": 100,
                }
            },
        }
    }

    session_update = {
        "type": "session.update",
        "session": {"turn_detection": {"create_response": False}},
    }

    messages = config.transform_realtime_request(
        json.dumps(session_update),
        "gemini-2.5-flash-native-audio",
        session_configuration_request=json.dumps(original_setup),
    )

    automatic_activity_detection = json.loads(messages[0])["setup"][
        "realtimeInputConfig"
    ]["automaticActivityDetection"]
    assert automatic_activity_detection["disabled"] is True
    assert automatic_activity_detection["silenceDurationMs"] == 500
    assert automatic_activity_detection["prefixPaddingMs"] == 100


def test_gemini_tool_call_id_to_name_evicts_oldest_when_capped():
    """The call_id → name LRU must evict the oldest entry once the cap is
    reached so long sessions with many tool calls don't grow unboundedly,
    while keeping recently-seen call_ids resolvable for retried
    function_call_output messages."""
    config = GeminiRealtimeConfig()
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_lru"

    config._TOOL_CALL_ID_TO_NAME_MAX = 4

    for idx in range(8):
        config.transform_realtime_response(
            json.dumps(
                {
                    "toolCall": {
                        "functionCalls": [
                            {
                                "id": f"call_{idx}",
                                "name": f"fn_{idx}",
                                "args": {},
                            }
                        ]
                    }
                }
            ),
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

    assert len(config._tool_call_id_to_name) == 4
    # Most recent 4 retained; oldest 4 evicted.
    assert list(config._tool_call_id_to_name) == [
        "call_4",
        "call_5",
        "call_6",
        "call_7",
    ]
