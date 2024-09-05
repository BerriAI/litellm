from enum import Enum
from typing import Optional, TypedDict


class EndpointType(str, Enum):
    VERTEX_AI = "vertex-ai"
    GENERIC = "generic"


class PassthroughStandardLoggingPayload(TypedDict, total=False):
    """
    Standard logging payload for all pass through endpoints
    """

    url: str
    request_body: Optional[dict]
    response_body: Optional[dict]  # only tracked for non-streaming responses
