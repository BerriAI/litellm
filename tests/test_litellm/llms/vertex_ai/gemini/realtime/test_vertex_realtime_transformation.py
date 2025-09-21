import json
import os
import sys
from typing import Any, Dict, List, cast

import pytest
from unittest.mock import MagicMock

# Add project root so relative imports work
sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.vertex_ai.gemini.realtime.transformation import VertexLiveRealtimeConfig
from litellm.types.llms.openai import (
    OpenAIRealtimeEventTypes,
    OpenAIRealtimeEvents,
)
from litellm.types.llms.vertex_ai import ContentType
from litellm.types.realtime import RealtimeResponseTypedDict


# ---------- helpers ----------
def _vertex_session_config(
    *,
    modalities: List[str] = ["TEXT"],
    model: str = "projects/p/locations/l/publishers/google/models/gemini-2.0-flash-live-preview-04-09",
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """
    Vertex Live setup payload passed through LiteLLM's realtime layer.
    NOTE: Vertex impl reads snake keys inside `generationConfig`.
    """
    generation_config: Dict[str, Any] = {"response_modalities": list(modalities)}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if max_output_tokens is not None:
        generation_config["max_output_tokens"] = max_output_tokens

    return json.dumps(
        {
            "setup": {
                "model": model,
                "generationConfig": generation_config,
            }
        }
    )


def _types_of(resp: RealtimeResponseTypedDict) -> List[str]:
    events = cast(List[Dict[str, Any]], resp["response"])
    return [cast(str, e.get("type")) for e in events]


# ---------- tests ----------
def test_vertex_realtime_transformation_session_created():
    config = VertexLiveRealtimeConfig()
    assert config is not None

    session_configuration_request_str = _vertex_session_config(modalities=["TEXT"])
    session_created_frame = {"setupComplete": {"sessionId": "abc"}}

    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace-123"

    transformed = config.transform_realtime_response(
        json.dumps(session_created_frame),
        "vertex_ai/gemini-2.0-flash-live-preview-04-09",
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
    types = _types_of(transformed)
    assert types and types[0] == "session.created"


def test_vertex_realtime_transformation_content_delta_text():
    config = VertexLiveRealtimeConfig()
    assert config is not None

    session_configuration_request_str = _vertex_session_config(modalities=["TEXT"])
    # Vertex server frame uses camelCase: serverContent/modelTurn/parts/text
    frame = {
        "serverContent": {
            "modelTurn": {
                "role": "model",
                "parts": [{"text": "Hello, "}, {"text": "world!"}],
            }
        }
    }

    logging_obj = MagicMock()
    logging_obj.litellm_trace_id = "trace-456"

    returned = config.transform_realtime_response(
        json.dumps(frame),
        "vertex_ai/gemini-2.0-flash-live-preview-04-09",
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

    responses = cast(List[OpenAIRealtimeEvents], returned["response"])
    has_text_delta = any(
        cast(Dict[str, Any], r).get("type") == OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA.value for r in responses
    )
    assert has_text_delta, "Expected text delta event"

    # Collect all delta payloads and ensure our text is present
    deltas = [
        cast(Dict[str, Any], r).get("delta", "")
        for r in responses
        if cast(Dict[str, Any], r).get("type") == OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA.value
    ]
    joined = "".join(deltas)
    assert "Hello, " in joined and "world!" in joined


def test_vertex_realtime_transformation_audio_delta():
    config = VertexLiveRealtimeConfig()
    assert config is not None

    session_configuration_request_str = _vertex_session_config(modalities=["AUDIO"])
    frame = {
        "serverContent": {
            "modelTurn": {
                "role": "model",
                "parts": [{"inlineData": {"mimeType": "audio/pcm", "data": "BASE64_AUDIO"}}],
            }
        }
    }

    returned = config.transform_realtime_response(
        json.dumps(frame),
        "vertex_ai/gemini-2.0-flash-live-preview-04-09",
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
    responses = cast(List[OpenAIRealtimeEvents], returned["response"])
    assert any(
        cast(Dict[str, Any], r).get("type") == OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DELTA.value for r in responses
    ), "Expected audio delta event"


def test_vertex_realtime_transformation_generation_complete_audio_done():
    config = VertexLiveRealtimeConfig()
    assert config is not None

    session_configuration_request_str = _vertex_session_config(modalities=["AUDIO"])
    frame = {"serverContent": {"generationComplete": True}}

    returned = config.transform_realtime_response(
        json.dumps(frame),
        "vertex_ai/gemini-2.0-flash-live-preview-04-09",
        MagicMock(),
        realtime_response_transform_input={
            "session_configuration_request": session_configuration_request_str,
            "current_output_item_id": "item-1",
            "current_response_id": "resp-1",
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": "audio",
        },
    )
    responses = cast(List[OpenAIRealtimeEvents], returned["response"])
    assert any(
        cast(Dict[str, Any], r).get("type") == OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DONE.value for r in responses
    ), "Expected audio done event"


def test_vertex_model_turn_event_mapping_unit():
    """
    Unit-test the tiny mapper directly.
    NOTE: Vertex mapper expects snake_case for the internal Part typing.
    """
    config = VertexLiveRealtimeConfig()
    assert config is not None

    event: Dict[str, Any] = {"parts": [{"text": "Hello, world!"}]}
    assert config.map_model_turn_event(cast(ContentType, event)) == OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA

    event = {"parts": [{"inline_data": {"mime_type": "audio/pcm", "data": "..."}}]}
    assert config.map_model_turn_event(cast(ContentType, event)) == OpenAIRealtimeEventTypes.RESPONSE_AUDIO_DELTA

    event = {
        "parts": [
            {"text": "Hello"},
            {"inline_data": {"mime_type": "audio/pcm", "data": "..."}},  # ignored due to existing text
        ]
    }
    assert config.map_model_turn_event(cast(ContentType, event)) == OpenAIRealtimeEventTypes.RESPONSE_TEXT_DELTA