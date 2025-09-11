# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union, Optional
from typing_extensions import Literal, Annotated, TypeAlias

from ...._utils import PropertyInfo
from ...._models import BaseModel
from .session_update_event import SessionUpdateEvent
from .response_cancel_event import ResponseCancelEvent
from .response_create_event import ResponseCreateEvent
from .transcription_session_update import TranscriptionSessionUpdate
from .conversation_item_create_event import ConversationItemCreateEvent
from .conversation_item_delete_event import ConversationItemDeleteEvent
from .input_audio_buffer_clear_event import InputAudioBufferClearEvent
from .input_audio_buffer_append_event import InputAudioBufferAppendEvent
from .input_audio_buffer_commit_event import InputAudioBufferCommitEvent
from .conversation_item_retrieve_event import ConversationItemRetrieveEvent
from .conversation_item_truncate_event import ConversationItemTruncateEvent

__all__ = ["RealtimeClientEvent", "OutputAudioBufferClear"]


class OutputAudioBufferClear(BaseModel):
    type: Literal["output_audio_buffer.clear"]
    """The event type, must be `output_audio_buffer.clear`."""

    event_id: Optional[str] = None
    """The unique ID of the client event used for error handling."""


RealtimeClientEvent: TypeAlias = Annotated[
    Union[
        ConversationItemCreateEvent,
        ConversationItemDeleteEvent,
        ConversationItemRetrieveEvent,
        ConversationItemTruncateEvent,
        InputAudioBufferAppendEvent,
        InputAudioBufferClearEvent,
        OutputAudioBufferClear,
        InputAudioBufferCommitEvent,
        ResponseCancelEvent,
        ResponseCreateEvent,
        SessionUpdateEvent,
        TranscriptionSessionUpdate,
    ],
    PropertyInfo(discriminator="type"),
]
