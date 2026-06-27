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
        if (
            response["type"]
            == OpenAIRealtimeEventTypes.RESPONSE_OUTPUT_AUDIO_DELTA.value
        ):
            contains_audio_delta = True
            break
    assert contains_audio_delta, "Expected audio delta event"


def test_gemini_output_audio_transcript_delta_uses_active_response_ids():
    config = GeminiRealtimeConfig()

    session_configuration_request = {
        "setup": {
            "model": "gemini-1.5-flash",
            "generationConfig": {"responseModalities": ["AUDIO"]},
        }
    }
    session_configuration_request_str = json.dumps(session_configuration_request)
    event = {
        "serverContent": {
            "outputTranscription": {"text": "Hello from Gemini."},
            "modelTurn": {
                "parts": [
                    {"inlineData": {"mimeType": "audio/pcm", "data": "my-audio-data"}}
                ]
            },
        }
    }

    result = config.transform_realtime_response(
        json.dumps(event),
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

    responses = result["response"]
    response_created = next(
        response for response in responses if response["type"] == "response.created"
    )
    transcript_delta = next(
        response
        for response in responses
        if response["type"] == "response.output_audio_transcript.delta"
    )
    audio_delta = next(
        response
        for response in responses
        if response["type"] == "response.output_audio.delta"
    )

    assert transcript_delta["response_id"] == response_created["response"]["id"]
    assert transcript_delta["response_id"] == audio_delta["response_id"]
    assert transcript_delta["item_id"] == audio_delta["item_id"]
    assert result["current_response_id"] == transcript_delta["response_id"]
    assert result["current_output_item_id"] == transcript_delta["item_id"]


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
        if (
            response["type"]
            == OpenAIRealtimeEventTypes.RESPONSE_OUTPUT_AUDIO_DONE.value
        ):
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


@pytest.mark.parametrize(
    "model",
    [
        "gemini-2.5-flash-native-audio",
        "gemini-3.1-flash-live-preview",
        "gemini/gemini-3.1-flash-live-preview",
    ],
)
def test_gemini_audio_only_live_models_coerce_text_modality_to_audio(model, patch_gemini_audio_cost_map_entries):
    """Regression: TEXT-only responseModalities causes 1007 on audio-only Live models."""
    config = GeminiRealtimeConfig()
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["text"],
            "instructions": "You are a terse assistant.",
        },
    }

    messages = config.transform_realtime_request(
        json.dumps(session_update),
        model,
        session_configuration_request=None,
    )

    setup = json.loads(messages[0])["setup"]
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]


def test_gemini_audio_only_live_models_drop_text_from_text_audio_combo(patch_gemini_audio_cost_map_entries):
    config = GeminiRealtimeConfig()
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": "Be concise.",
        },
    }

    messages = config.transform_realtime_request(
        json.dumps(session_update),
        "gemini-3.1-flash-live-preview",
        session_configuration_request=None,
    )

    setup = json.loads(messages[0])["setup"]
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]


def test_gemini_non_live_model_preserves_text_modality():
    config = GeminiRealtimeConfig()
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["text"],
            "instructions": "You are a terse assistant.",
        },
    }

    messages = config.transform_realtime_request(
        json.dumps(session_update),
        "gemini-2.5-flash",
        session_configuration_request=None,
    )

    setup = json.loads(messages[0])["setup"]
    assert setup["generationConfig"]["responseModalities"] == ["TEXT"]


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
    # Expected sequence:
    #   0: response.created
    #   1: response.output_item.added  (item status=in_progress)
    #   2: conversation.item.added     (registers call_id in Pipecat's _pending_function_calls)
    #   3: response.function_call_arguments.delta
    #   4: response.function_call_arguments.done
    #   5: response.output_item.done
    #   6: response.done
    assert len(responses) >= 7
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
    assert responses[2]["type"] == "conversation.item.added"
    assert responses[2]["item"]["type"] == "function_call"
    assert responses[2]["item"]["call_id"] == "call_123"
    assert responses[3]["type"] == "response.function_call_arguments.delta"
    assert responses[3]["call_id"] == "call_123"
    assert responses[3]["delta"] == responses[4]["arguments"]
    assert responses[4]["type"] == "response.function_call_arguments.done"
    assert responses[5]["type"] == "response.output_item.done"
    assert responses[5]["item"]["type"] == "function_call"
    assert responses[5]["item"]["status"] == "completed"
    assert responses[6]["type"] == "response.done"
    assert responses[6]["response"]["status"] == "completed"
    assert len(responses[6]["response"]["output"]) == 1
    assert responses[6]["response"]["output"][0]["type"] == "function_call"
    assert result["current_output_item_id"] is None
    assert result["current_response_id"] is None


def test_gemini_tool_call_resets_ids_for_post_tool_model_turn():
    """After tool-call response.done, a subsequent modelTurn must emit response.created."""
    config = GeminiRealtimeConfig()
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_123"

    session_configuration_request = json.dumps(
        {
            "setup": {
                "model": "gemini-1.5-flash",
                "generationConfig": {"responseModalities": ["TEXT"]},
            }
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


def test_gemini_empty_tool_call_with_sibling_usage_metadata_does_not_crash():
    """A toolCall with empty functionCalls alongside a sibling key (e.g.
    ``usageMetadata``) must still be handled as a benign no-op: the empty
    toolCall is consumed and the metadata sibling is skipped, without
    raising ``Unknown message type``."""
    config = GeminiRealtimeConfig()
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_empty_tool_call_with_sibling"

    result = config.transform_realtime_response(
        json.dumps(
            {
                "toolCall": {"functionCalls": []},
                "usageMetadata": {"totalTokenCount": 7},
            }
        ),
        "gemini-2.5-flash",
        logging_obj,
        realtime_response_transform_input={
            "session_configuration_request": None,
            "current_output_item_id": "item_existing",
            "current_response_id": "resp_existing",
            "current_conversation_id": "conv_existing",
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        },
    )

    assert result["response"] == []
    # In-flight response IDs must survive the benign no-op.
    assert result["current_response_id"] == "resp_existing"
    assert result["current_output_item_id"] == "item_existing"


def test_gemini_tool_call_response_done_includes_usage_from_sibling_metadata():
    """A ``toolCall`` frame with a sibling ``usageMetadata`` must propagate the
    real token counts onto the emitted ``response.done`` so spend/budget
    accounting records tokens consumed by the tool-call turn — otherwise an
    authenticated client can repeatedly drive tool calls with zero spend."""
    config = GeminiRealtimeConfig()
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_tool_call_usage"

    result = config.transform_realtime_response(
        json.dumps(
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call_usage",
                            "name": "get_weather",
                            "args": {"location": "NYC"},
                        }
                    ]
                },
                "usageMetadata": {
                    "promptTokenCount": 17,
                    "responseTokenCount": 4,
                    "totalTokenCount": 21,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 17},
                    ],
                    "responseTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 4},
                    ],
                },
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

    response_done = next(
        ev for ev in result["response"] if ev.get("type") == "response.done"
    )
    usage = response_done["response"]["usage"]
    assert usage["input_tokens"] == 17
    assert usage["output_tokens"] == 4
    assert usage["total_tokens"] == 21
    assert usage["input_token_details"]["text_tokens"] == 17
    assert usage["output_token_details"]["text_tokens"] == 4


def test_gemini_tool_call_response_done_falls_back_to_empty_usage():
    """Without sibling ``usageMetadata`` the tool-call ``response.done`` still
    carries a valid empty usage block so OpenAI-compatible clients (which
    expect ``usage`` on every ``response.done``) don't break."""
    config = GeminiRealtimeConfig()
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_tool_call_no_usage"

    result = config.transform_realtime_response(
        json.dumps(
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call_no_usage",
                            "name": "get_weather",
                            "args": {"location": "NYC"},
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

    response_done = next(
        ev for ev in result["response"] if ev.get("type") == "response.done"
    )
    usage = response_done["response"]["usage"]
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["total_tokens"] == 0


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


def test_gemini_realtime_pipecat_ga_session_voice_and_tools(patch_gemini_audio_cost_map_entries):
    """Pipecat OpenAIRealtimeSessionProperties: output_modalities, nested tools,
    and audio.output.voice (e.g. Kore) must map into Gemini setup."""
    config = GeminiRealtimeConfig()

    session_update = {
        "type": "session.update",
        "session": {
            "output_modalities": ["audio"],
            "instructions": "Follow system instructions.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "terminate_call",
                        "description": "End the call.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "Kore",
                },
            },
            "temperature": 0,
        },
    }

    messages = config.transform_realtime_request(
        json.dumps(session_update),
        "gemini-2.5-flash-native-audio",
        session_configuration_request=None,
    )

    assert len(messages) == 1
    setup = json.loads(messages[0])["setup"]
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    # Native-audio Live rejects speechConfig on setup (see _finalize_gemini_live_setup).
    assert "speechConfig" not in setup.get("generationConfig", {})
    assert setup["tools"][0]["function_declarations"][0]["name"] == "terminate_call"
    assert (
        setup["realtimeInputConfig"]["automaticActivityDetection"]["disabled"] is False
    )


def test_gemini_realtime_pipecat_semantic_vad_omits_realtime_input_config():
    """Pipecat SemanticTurnDetection (semantic_vad) must not map to disabled VAD."""
    config = GeminiRealtimeConfig()
    session_update = {
        "type": "session.update",
        "session": {
            "output_modalities": ["audio"],
            "instructions": "test",
            "audio": {
                "input": {"turn_detection": {"type": "semantic_vad"}},
            },
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "terminate_call",
                        "description": "End call.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    }
    messages = config.transform_realtime_request(
        json.dumps(session_update),
        "gemini-live-2.5-flash-native-audio",
        session_configuration_request=None,
    )
    setup = json.loads(messages[0])["setup"]
    assert "realtimeInputConfig" not in setup
    assert setup["tools"][0]["function_declarations"][0]["name"] == "terminate_call"


def test_gemini_input_audio_buffer_commit_maps_to_audio_stream_end():
    config = GeminiRealtimeConfig()
    setup = {
        "setup": {
            "realtimeInputConfig": {
                "automaticActivityDetection": {"disabled": False},
            }
        }
    }
    messages = config.transform_realtime_request(
        json.dumps({"type": "input_audio_buffer.commit"}),
        "gemini-live-2.5-flash-native-audio",
        session_configuration_request=json.dumps(setup),
    )
    assert len(messages) == 1
    assert json.loads(messages[0]) == {"realtimeInput": {"audioStreamEnd": True}}


def test_gemini_input_audio_buffer_end_maps_to_audio_stream_end():
    config = GeminiRealtimeConfig()
    messages = config.transform_realtime_request(
        json.dumps({"type": "input_audio_buffer.end"}),
        "gemini-live-2.5-flash-native-audio",
        session_configuration_request=None,
    )
    assert len(messages) == 1
    assert json.loads(messages[0]) == {"realtimeInput": {"audioStreamEnd": True}}


def test_gemini_input_audio_buffer_clear_is_local_noop():
    config = GeminiRealtimeConfig()
    messages = config.transform_realtime_request(
        json.dumps({"type": "input_audio_buffer.clear"}),
        "gemini-live-2.5-flash-native-audio",
        session_configuration_request=None,
    )
    assert messages == []


def test_gemini_input_audio_buffer_commit_maps_to_activity_end_when_manual_vad():
    config = GeminiRealtimeConfig()
    setup = {
        "setup": {
            "realtimeInputConfig": {
                "automaticActivityDetection": {"disabled": True},
            }
        }
    }
    messages = config.transform_realtime_request(
        json.dumps({"type": "input_audio_buffer.commit"}),
        "gemini-live-2.5-flash-native-audio",
        session_configuration_request=json.dumps(setup),
    )
    assert len(messages) == 1
    assert json.loads(messages[0]) == {"realtimeInput": {"activityEnd": True}}


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


def test_gemini_standalone_usage_metadata_does_not_crash_websocket():
    """A Gemini frame containing only sibling metadata (e.g. a standalone
    ``usageMetadata`` block emitted between turns) must not trip the
    ``Unknown message type`` guard — that would terminate the WebSocket
    session on a benign no-op frame."""
    config = GeminiRealtimeConfig()
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_usage_only"

    result = config.transform_realtime_response(
        json.dumps(
            {
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "responseTokenCount": 34,
                    "totalTokenCount": 46,
                }
            }
        ),
        "gemini-2.5-flash",
        logging_obj,
        realtime_response_transform_input={
            "session_configuration_request": None,
            "current_output_item_id": "item_existing",
            "current_response_id": "resp_existing",
            "current_conversation_id": "conv_existing",
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        },
    )

    assert result["response"] == []
    # State must be returned unchanged so subsequent frames continue the
    # in-flight response correctly.
    assert result["current_output_item_id"] == "item_existing"
    assert result["current_response_id"] == "resp_existing"
    assert result["current_conversation_id"] == "conv_existing"


def test_gemini_standalone_usage_metadata_is_attributed_to_next_tool_call_response_done():
    """A standalone ``usageMetadata`` frame emitted between turns must not
    silently drop the consumed tokens. The next tool-call ``response.done``
    must carry those token counts so an authenticated client cannot drive
    tool-call turns whose token usage is recorded as zero, bypassing
    spend/budget accounting."""
    config = GeminiRealtimeConfig()
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_standalone_usage_then_tool_call"

    standalone_result = config.transform_realtime_response(
        json.dumps(
            {
                "usageMetadata": {
                    "promptTokenCount": 31,
                    "responseTokenCount": 9,
                    "totalTokenCount": 40,
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
    assert standalone_result["response"] == []

    tool_call_result = config.transform_realtime_response(
        json.dumps(
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call_buffered",
                            "name": "get_weather",
                            "args": {"location": "NYC"},
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

    response_done = next(
        ev for ev in tool_call_result["response"] if ev.get("type") == "response.done"
    )
    usage = response_done["response"]["usage"]
    assert usage["input_tokens"] == 31
    assert usage["output_tokens"] == 9
    assert usage["total_tokens"] == 40
    # Buffer must be cleared after attribution so a subsequent tool-call
    # turn without its own usage does not double-count the previous frame.
    assert config._pending_usage_metadata is None


def test_gemini_standalone_usage_metadata_is_attributed_to_next_response_done():
    """A standalone ``usageMetadata`` frame must also flow into the normal
    (non-tool-call) ``response.done`` path so audio/text turns whose usage
    arrives in a separate frame are still billed correctly."""
    config = GeminiRealtimeConfig()
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_standalone_usage_then_turn_complete"

    config.transform_realtime_response(
        json.dumps(
            {
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "responseTokenCount": 11,
                    "totalTokenCount": 16,
                    "promptTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 5},
                    ],
                    "responseTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 11},
                    ],
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

    turn_complete_result = config.transform_realtime_response(
        json.dumps({"serverContent": {"turnComplete": True}}),
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

    response_done = next(
        ev
        for ev in turn_complete_result["response"]
        if ev.get("type") == "response.done"
    )
    usage = response_done["response"]["usage"]
    assert usage["input_tokens"] == 5
    assert usage["output_tokens"] == 11
    assert usage["total_tokens"] == 16
    assert usage["input_token_details"]["text_tokens"] == 5
    assert usage["output_token_details"]["text_tokens"] == 11
    assert config._pending_usage_metadata is None


def test_gemini_in_frame_usage_metadata_clears_pending_buffer():
    """When ``usageMetadata`` arrives in the same frame as the closing
    ``toolCall`` / ``turnComplete``, the in-frame counts are authoritative
    and any buffered standalone metadata must be discarded so a later
    turn's ``response.done`` does not double-count tokens."""
    config = GeminiRealtimeConfig()
    config._pending_usage_metadata = {
        "promptTokenCount": 99,
        "responseTokenCount": 99,
        "totalTokenCount": 198,
    }
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_in_frame_clears_buffer"

    result = config.transform_realtime_response(
        json.dumps(
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call_in_frame",
                            "name": "get_weather",
                            "args": {"location": "NYC"},
                        }
                    ]
                },
                "usageMetadata": {
                    "promptTokenCount": 3,
                    "responseTokenCount": 2,
                    "totalTokenCount": 5,
                },
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

    response_done = next(
        ev for ev in result["response"] if ev.get("type") == "response.done"
    )
    usage = response_done["response"]["usage"]
    assert usage["input_tokens"] == 3
    assert usage["output_tokens"] == 2
    assert usage["total_tokens"] == 5
    assert config._pending_usage_metadata is None


def test_gemini_post_tool_bare_turn_complete_followed_by_answer():
    """After a tool call, Gemini Live can emit a bare ``turnComplete`` (with
    usage but no model content) before the follow-up answer stream. That bare
    ``turnComplete`` may produce an extra ``response.done``; Pipecat is tolerant
    of that because ``_process_completed_function_calls`` is idempotent (the
    pending call queue is empty by the time the second ``response.done`` arrives).
    The important thing is that the post-tool answer is correctly generated."""
    config = GeminiRealtimeConfig()
    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace_post_tool_bare_turn_complete"

    session_configuration_request = json.dumps(
        {
            "setup": {
                "model": "gemini-2.5-flash-native-audio",
                "generationConfig": {"responseModalities": ["AUDIO"]},
            }
        }
    )
    base_input = {
        "session_configuration_request": session_configuration_request,
        "current_output_item_id": None,
        "current_response_id": None,
        "current_conversation_id": None,
        "current_delta_chunks": [],
        "current_item_chunks": [],
        "current_delta_type": None,
    }

    tool_result = config.transform_realtime_response(
        json.dumps(
            {
                "toolCall": {
                    "functionCalls": [
                        {
                            "id": "call_post_tool",
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                        }
                    ]
                }
            }
        ),
        "gemini-2.5-flash-native-audio",
        logging_obj,
        realtime_response_transform_input=base_input,
    )
    assert tool_result["response"][-1]["type"] == "response.done"

    bare_turn_complete = config.transform_realtime_response(
        json.dumps(
            {
                "serverContent": {"turnComplete": True},
                "usageMetadata": {
                    "promptTokenCount": 30,
                    "responseTokenCount": 5,
                    "totalTokenCount": 35,
                },
            }
        ),
        "gemini-2.5-flash-native-audio",
        logging_obj,
        realtime_response_transform_input={
            **base_input,
            "current_output_item_id": tool_result["current_output_item_id"],
            "current_response_id": tool_result["current_response_id"],
            "current_conversation_id": tool_result["current_conversation_id"],
            "current_delta_chunks": tool_result["current_delta_chunks"],
            "current_item_chunks": tool_result["current_item_chunks"],
            "current_delta_type": tool_result["current_delta_type"],
        },
    )
    # The bare turnComplete must not surface as a response.done because clients
    # that use collect_until("response.done") would stop collecting prematurely
    # before the real follow-up answer arrives.
    assert bare_turn_complete["response"] == []

    post_tool_answer = config.transform_realtime_response(
        json.dumps(
            {
                "serverContent": {
                    "outputTranscription": {"text": "The temperature is 72."},
                    "modelTurn": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/pcm",
                                    "data": "audio-chunk",
                                }
                            }
                        ]
                    },
                }
            }
        ),
        "gemini-2.5-flash-native-audio",
        logging_obj,
        realtime_response_transform_input={
            **base_input,
            "current_output_item_id": bare_turn_complete["current_output_item_id"],
            "current_response_id": bare_turn_complete["current_response_id"],
            "current_conversation_id": bare_turn_complete["current_conversation_id"],
            "current_delta_chunks": bare_turn_complete["current_delta_chunks"],
            "current_item_chunks": bare_turn_complete["current_item_chunks"],
            "current_delta_type": bare_turn_complete["current_delta_type"],
        },
    )
    assert post_tool_answer["response"][0]["type"] == "response.created"
    transcript_delta = next(
        event
        for event in post_tool_answer["response"]
        if event["type"] == "response.output_audio_transcript.delta"
    )
    assert "72" in transcript_delta["delta"]

    final_turn = config.transform_realtime_response(
        json.dumps({"serverContent": {"turnComplete": True}}),
        "gemini-2.5-flash-native-audio",
        logging_obj,
        realtime_response_transform_input={
            **base_input,
            "current_output_item_id": post_tool_answer["current_output_item_id"],
            "current_response_id": post_tool_answer["current_response_id"],
            "current_conversation_id": post_tool_answer["current_conversation_id"],
            "current_delta_chunks": post_tool_answer["current_delta_chunks"],
            "current_item_chunks": post_tool_answer["current_item_chunks"],
            "current_delta_type": post_tool_answer["current_delta_type"],
        },
    )
    response_done = next(
        event
        for event in final_turn["response"]
        if event["type"] == "response.done"
    )
    assert response_done["response"]["status"] == "completed"


@pytest.fixture(autouse=False)
def patch_gemini_audio_cost_map_entries(monkeypatch):
    """Inject gemini_native_audio / gemini_audio_only_live into the cost map.

    litellm.model_cost is fetched from main branch at import time, so in CI
    the fields may not exist yet. Patch locally so these tests are
    self-contained.
    """
    native_audio_models = [
        "gemini-2.5-flash-native-audio",
        "gemini-2.5-flash-native-audio-latest",
        "gemini/gemini-2.5-flash-native-audio-latest",
    ]
    flash_live_models = [
        "gemini-3.1-flash-live-preview",
        "gemini/gemini-3.1-flash-live-preview",
    ]
    for m in native_audio_models:
        entry = dict(litellm.model_cost.get(m, {}))
        entry["gemini_native_audio"] = True
        monkeypatch.setitem(litellm.model_cost, m, entry)
    for m in flash_live_models:
        entry = dict(litellm.model_cost.get(m, {}))
        entry["gemini_audio_only_live"] = True
        monkeypatch.setitem(litellm.model_cost, m, entry)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini-3.1-flash-live-preview", True),
        ("gemini/gemini-3.1-flash-live-preview", True),
        ("gemini-2.5-flash-native-audio-latest", True),
        ("gemini/gemini-2.5-flash-native-audio-latest", True),
        ("gemini-2.0-flash", False),
        ("gemini-2.5-flash", False),
    ],
)
def test_is_audio_only_live_model_uses_cost_map(
    model, expected, patch_gemini_audio_cost_map_entries
):
    assert GeminiRealtimeConfig._is_audio_only_live_model(model) == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini-2.5-flash-native-audio-latest", True),
        ("gemini/gemini-2.5-flash-native-audio-latest", True),
        ("gemini-3.1-flash-live-preview", False),
        ("gemini/gemini-3.1-flash-live-preview", False),
        ("gemini-2.0-flash", False),
    ],
)
def test_is_native_audio_model_uses_cost_map(
    model, expected, patch_gemini_audio_cost_map_entries
):
    assert GeminiRealtimeConfig._is_native_audio_model(model) == expected


def test_is_setup_message_and_is_content_message():
    config = GeminiRealtimeConfig()
    assert config.is_setup_message({"setup": {}}) is True
    assert config.is_setup_message({"realtimeInput": {}}) is False
    assert config.is_content_message({"realtimeInput": {}}) is True
    assert config.is_content_message({"clientContent": {}}) is True
    assert config.is_content_message({"toolResponse": {}}) is True
    assert config.is_content_message({"setup": {}}) is False
