from typing import Union
from typing_extensions import List, Literal, Annotated

from ..._models import BaseModel
from ...types.beta import (
    BetaMessage,
    BetaContentBlock,
    BetaRawMessageStopEvent,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStartEvent,
    BetaRawContentBlockStopEvent,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
)
from ..._utils._transform import PropertyInfo
from ...types.beta.beta_citations_delta import Citation


class BetaTextEvent(BaseModel):
    type: Literal["text"]

    text: str
    """The text delta"""

    snapshot: str
    """The entire accumulated text"""


class BetaCitationEvent(BaseModel):
    type: Literal["citation"]

    citation: Citation
    """The new citation"""

    snapshot: List[Citation]
    """All of the accumulated citations"""


class BetaThinkingEvent(BaseModel):
    type: Literal["thinking"]

    thinking: str
    """The thinking delta"""

    snapshot: str
    """The accumulated thinking so far"""


class BetaSignatureEvent(BaseModel):
    type: Literal["signature"]

    signature: str
    """The signature of the thinking block"""


class BetaInputJsonEvent(BaseModel):
    type: Literal["input_json"]

    partial_json: str
    """A partial JSON string delta

    e.g. `'"San Francisco,'`
    """

    snapshot: object
    """The currently accumulated parsed object.


    e.g. `{'location': 'San Francisco, CA'}`
    """


class BetaMessageStopEvent(BetaRawMessageStopEvent):
    type: Literal["message_stop"]

    message: BetaMessage


class BetaContentBlockStopEvent(BetaRawContentBlockStopEvent):
    type: Literal["content_block_stop"]

    content_block: BetaContentBlock


BetaMessageStreamEvent = Annotated[
    Union[
        BetaTextEvent,
        BetaCitationEvent,
        BetaThinkingEvent,
        BetaSignatureEvent,
        BetaInputJsonEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageDeltaEvent,
        BetaMessageStopEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockDeltaEvent,
        BetaContentBlockStopEvent,
    ],
    PropertyInfo(discriminator="type"),
]
