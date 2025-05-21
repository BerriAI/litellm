from enum import Enum
from typing import Optional, TypedDict

from litellm.types.utils import CallTypes as LiteLLMCallTypes


class EndpointType(str, Enum):
    VERTEX_AI = "vertex-ai"
    ANTHROPIC = "anthropic"
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


class PassthroughEndpointInitKwargs(TypedDict, total=False):
    """
    Kwargs for the pass-through endpoint
    """

    litellm_params: dict
    model: Optional[str]
    call_type: LiteLLMCallTypes
    litellm_call_id: Optional[str]
    passthrough_logging_payload: Optional[PassthroughStandardLoggingPayload]
