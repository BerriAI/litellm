from typing import Any, Literal

from pydantic import BaseModel
from typing_extensions import ReadOnly, TypedDict

from .llms.openai import (
    OpenAIRealtimeEvents,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeResponseDelta,
)

ALL_DELTA_TYPES = Literal["text", "audio"]


class RealtimeResponseTransformInput(TypedDict):
    session_configuration_request: str | None
    current_output_item_id: (
        str | None
    )  # used to check if this is a new content.delta or a continuation of a previous content.delta
    current_response_id: (
        str | None
    )  # used to check if this is a new content.delta or a continuation of a previous content.delta
    current_delta_chunks: list[OpenAIRealtimeResponseDelta] | None
    current_item_chunks: list[OpenAIRealtimeOutputItemDone] | None
    current_conversation_id: str | None
    current_delta_type: ALL_DELTA_TYPES | None


class RealtimeResponseTypedDict(TypedDict):
    response: OpenAIRealtimeEvents | list[OpenAIRealtimeEvents]
    current_output_item_id: str | None
    current_response_id: str | None
    current_delta_chunks: list[OpenAIRealtimeResponseDelta] | None
    current_conversation_id: str | None
    current_item_chunks: list[OpenAIRealtimeOutputItemDone] | None
    current_delta_type: ALL_DELTA_TYPES | None
    session_configuration_request: str | None


class RealtimeModalityResponseTransformOutput(TypedDict):
    returned_message: list[OpenAIRealtimeEvents]
    current_output_item_id: str | None
    current_response_id: str | None
    current_conversation_id: str | None
    current_delta_chunks: list[OpenAIRealtimeResponseDelta] | None
    current_delta_type: ALL_DELTA_TYPES | None


class RealtimeQueryParams(TypedDict, total=False):
    model: str
    intent: str | None
    # Add more fields as needed


# ---------------------------------------------------------------------------
# WebRTC / client_secrets types  (POST /v1/realtime/client_secrets)
# ---------------------------------------------------------------------------


class RealtimeExpiresAfter(BaseModel):
    """Expiration config for a client secret."""

    anchor: str | None = "created_at"
    seconds: int | None = None


class RealtimeSessionConfig(BaseModel):
    """
    Session configuration nested inside the client_secrets request body.

    Mirrors OpenAI's RealtimeSessionCreateRequest (type=realtime) and
    RealtimeTranscriptionSessionCreateRequest (type=transcription).
    Extra/unknown fields are passed through unchanged.
    """

    model_config = {"extra": "allow"}

    type: str | None = None
    model: str | None = None
    instructions: str | None = None
    audio: dict[str, Any] | None = None
    include: list[str] | None = None
    max_output_tokens: int | str | None = None
    output_modalities: list[str] | None = None
    tool_choice: Any | None = None
    tools: list[dict[str, Any]] | None = None
    tracing: Any | None = None
    truncation: Any | None = None
    prompt: dict[str, Any] | None = None


class RealtimeClientSecretRequest(BaseModel):
    """
    Request body for POST /v1/realtime/client_secrets.

    LiteLLM also accepts a top-level `model` field for routing when
    session.model is absent (LiteLLM extension, not forwarded to OpenAI).
    """

    expires_after: RealtimeExpiresAfter | None = None
    session: RealtimeSessionConfig | None = None
    # LiteLLM-only routing hint — stripped before forwarding upstream
    model: str | None = None


class RealtimeClientSecretResponse(BaseModel):
    """
    Response from POST /v1/realtime/client_secrets.

    Both the top-level `value` and `session.client_secret.value`
    will contain the encrypted token instead of the raw ephemeral key.
    The `session` field is kept as a raw dict so unknown fields pass through.
    """

    expires_at: int | None = None
    value: str
    session: dict[str, Any] | None = None


class RealtimeTranscriptionSessionRequest(BaseModel):
    """
    Request body for POST /v1/realtime/transcription_sessions.

    Mirrors OpenAI's RealtimeTranscriptionSessionCreateRequest. The model used
    for routing is taken from the LiteLLM-only top-level `model` hint, falling
    back to `input_audio_transcription.model`. All other fields pass through
    unchanged to the provider.
    """

    model_config = {"extra": "allow"}

    # LiteLLM-only routing hint — stripped before forwarding upstream.
    model: str | None = None
    input_audio_transcription: dict[str, Any] | None = None

    def resolved_model(self) -> str | None:
        if self.model:
            return self.model
        if self.input_audio_transcription:
            return self.input_audio_transcription.get("model")
        return None


class RealtimeTranscriptionSessionResponse(BaseModel):
    """
    Response from POST /v1/realtime/transcription_sessions.

    `client_secret.value` contains the encrypted token instead of the raw
    ephemeral key. Unknown fields pass through unchanged.
    """

    model_config = {"extra": "allow"}

    client_secret: dict[str, Any] | None = None


class RealtimeErrorDetail(TypedDict):
    type: ReadOnly[str]
    message: ReadOnly[str]


class RealtimeErrorEvent(TypedDict):
    type: ReadOnly[Literal["error"]]
    error: ReadOnly[RealtimeErrorDetail]


class RealtimeInputAudioTranscriptionUsageInputTokenDetails(TypedDict):
    text_tokens: ReadOnly[int]
    audio_tokens: ReadOnly[int]


class RealtimeInputAudioTranscriptionUsage(TypedDict):
    type: ReadOnly[Literal["tokens"]]
    input_tokens: ReadOnly[int]
    output_tokens: ReadOnly[int]
    total_tokens: ReadOnly[int]
    input_token_details: ReadOnly[RealtimeInputAudioTranscriptionUsageInputTokenDetails]
