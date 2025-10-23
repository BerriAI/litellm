from enum import Enum
from typing import Optional

from typing_extensions import TypedDict


class EndpointType(str, Enum):
    VERTEX_AI = "vertex-ai"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GENERIC = "generic"


class PassthroughStandardLoggingPayload(TypedDict, total=False):
    """
    Standard logging payload for all pass through endpoints
    """

    url: str
    """
    The full url of the request
    """

    request_method: Optional[str]
    """
    The method of the request
    "GET", "POST", "PUT", "DELETE", etc.
    """

    request_body: Optional[dict]
    """
    The body of the request
    """
    response_body: Optional[dict]  # only tracked for non-streaming responses
    """
    The body of the response
    """

    cost_per_request: Optional[float]
    """
    The cost per request to the target endpoint

    Optional field, we use this for cost tracking only if it's set.
    """
