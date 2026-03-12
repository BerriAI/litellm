from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel
from typing_extensions import TypedDict  # noqa: F401 – re-exported

from .llms.openai import (
    OpenAIRealtimeEvents,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeResponseDelta,
)

ALL_DELTA_TYPES = Literal["text", "audio"]


class RealtimeResponseTransformInput(TypedDict):
    session_configuration_request: Optional[str]
    current_output_item_id: Optional[
        str
    ]  # used to check if this is a new content.delta or a continuation of a previous content.delta
    current_response_id: Optional[
        str
    ]  # used to check if this is a new content.delta or a continuation of a previous content.delta
    current_delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]]
    current_item_chunks: Optional[List[OpenAIRealtimeOutputItemDone]]
    current_conversation_id: Optional[str]
    current_delta_type: Optional[ALL_DELTA_TYPES]


class RealtimeResponseTypedDict(TypedDict):
    response: Union[OpenAIRealtimeEvents, List[OpenAIRealtimeEvents]]
    current_output_item_id: Optional[str]
    current_response_id: Optional[str]
    current_delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]]
    current_conversation_id: Optional[str]
    current_item_chunks: Optional[List[OpenAIRealtimeOutputItemDone]]
    current_delta_type: Optional[ALL_DELTA_TYPES]
    session_configuration_request: Optional[str]


class RealtimeModalityResponseTransformOutput(TypedDict):
    returned_message: List[OpenAIRealtimeEvents]
    current_output_item_id: Optional[str]
    current_response_id: Optional[str]
    current_conversation_id: Optional[str]
    current_delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]]
    current_delta_type: Optional[ALL_DELTA_TYPES]


class RealtimeQueryParams(TypedDict, total=False):
    model: str
    intent: Optional[str]
    # Add more fields as needed


# ---------------------------------------------------------------------------
# WebRTC / client_secrets types  (POST /v1/realtime/client_secrets)
# ---------------------------------------------------------------------------


class RealtimeExpiresAfter(BaseModel):
    """Expiration config for a client secret."""

    anchor: Optional[str] = "created_at"
    seconds: Optional[int] = None


class RealtimeSessionConfig(BaseModel):
    """
    Session configuration nested inside the client_secrets request body.

    Mirrors OpenAI's RealtimeSessionCreateRequest (type=realtime) and
    RealtimeTranscriptionSessionCreateRequest (type=transcription).
    Extra/unknown fields are passed through unchanged.
    """

    model_config = {"extra": "allow"}

    type: Optional[str] = None
    model: Optional[str] = None
    instructions: Optional[str] = None
    audio: Optional[Dict[str, Any]] = None
    include: Optional[List[str]] = None
    max_output_tokens: Optional[Union[int, str]] = None
    output_modalities: Optional[List[str]] = None
    tool_choice: Optional[Any] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tracing: Optional[Any] = None
    truncation: Optional[Any] = None
    prompt: Optional[Dict[str, Any]] = None


class RealtimeClientSecretRequest(BaseModel):
    """
    Request body for POST /v1/realtime/client_secrets.

    LiteLLM also accepts a top-level `model` field for routing when
    session.model is absent (LiteLLM extension, not forwarded to OpenAI).
    """

    expires_after: Optional[RealtimeExpiresAfter] = None
    session: Optional[RealtimeSessionConfig] = None
    # LiteLLM-only routing hint — stripped before forwarding upstream
    model: Optional[str] = None


class RealtimeClientSecretResponse(BaseModel):
    """
    Response from POST /v1/realtime/client_secrets.

    Both the top-level `value` and `session.client_secret.value`
    will contain the encrypted token instead of the raw ephemeral key.
    The `session` field is kept as a raw dict so unknown fields pass through.
    """

    expires_at: int
    value: str
    session: Optional[Dict[str, Any]] = None
