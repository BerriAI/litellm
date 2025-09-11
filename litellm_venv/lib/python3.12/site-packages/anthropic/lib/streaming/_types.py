from typing import Union
from typing_extensions import List, Literal, Annotated

from ...types import (
    Message,
    ContentBlock,
    MessageDeltaEvent as RawMessageDeltaEvent,
    MessageStartEvent as RawMessageStartEvent,
    RawMessageStopEvent,
    ContentBlockDeltaEvent as RawContentBlockDeltaEvent,
    ContentBlockStartEvent as RawContentBlockStartEvent,
    RawContentBlockStopEvent,
)
from ..._models import BaseModel
from ..._utils._transform import PropertyInfo
from ...types.citations_delta import Citation


class TextEvent(BaseModel):
    type: Literal["text"]

    text: str
    """The text delta"""

    snapshot: str
    """The entire accumulated text"""


class CitationEvent(BaseModel):
    type: Literal["citation"]

    citation: Citation
    """The new citation"""

    snapshot: List[Citation]
    """All of the accumulated citations"""


class ThinkingEvent(BaseModel):
    type: Literal["thinking"]

    thinking: str
    """The thinking delta"""

    snapshot: str
    """The accumulated thinking so far"""


class SignatureEvent(BaseModel):
    type: Literal["signature"]

    signature: str
    """The signature of the thinking block"""


class InputJsonEvent(BaseModel):
    type: Literal["input_json"]

    partial_json: str
    """A partial JSON string delta

    e.g. `'"San Francisco,'`
    """

    snapshot: object
    """The currently accumulated parsed object.


    e.g. `{'location': 'San Francisco, CA'}`
    """


class MessageStopEvent(RawMessageStopEvent):
    type: Literal["message_stop"]

    message: Message


class ContentBlockStopEvent(RawContentBlockStopEvent):
    type: Literal["content_block_stop"]

    content_block: ContentBlock


MessageStreamEvent = Annotated[
    Union[
        TextEvent,
        CitationEvent,
        ThinkingEvent,
        SignatureEvent,
        InputJsonEvent,
        RawMessageStartEvent,
        RawMessageDeltaEvent,
        MessageStopEvent,
        RawContentBlockStartEvent,
        RawContentBlockDeltaEvent,
        ContentBlockStopEvent,
    ],
    PropertyInfo(discriminator="type"),
]
