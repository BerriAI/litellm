from enum import Enum
from typing import Optional

from typing_extensions import TypedDict

# Request.state key for programmatic pass-through callers (e.g. Bedrock proxy) that attach
# JSON without a FastAPI `custom_body` parameter (which would consume the HTTP body).
LITELLM_PASS_THROUGH_CUSTOM_BODY_STATE_KEY = "litellm_pass_through_custom_body"

# Request.state key for programmatic pass-through callers that must preserve an
# exact byte/string body, such as AWS SigV4-signed requests.
LITELLM_PASS_THROUGH_RAW_BODY_STATE_KEY = "litellm_pass_through_raw_body"

# Attribute set on the FastAPI endpoint function of every user-defined pass-through
# route. Auth reads it off the dispatched endpoint (``request.scope["endpoint"]``) to
# decide whether a request body ``model`` names an upstream model rather than a
# LiteLLM-managed one. Keying off the resolved endpoint (not the request path) means a
# custom path that collides with a built-in route never suppresses model-access checks:
# on a collision FastAPI dispatches the built-in handler, which does not carry this flag.
LITELLM_PASS_THROUGH_ENDPOINT_MARKER = "__litellm_pass_through_endpoint__"


class EndpointType(str, Enum):
    VERTEX_AI = "vertex-ai"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    # Streamed Cohere chat. Without its own member `api.cohere.com` classified
    # as GENERIC, which reassembles nothing and costs nothing — every streamed
    # Cohere pass-through was billed upstream and recorded at $0.
    COHERE = "cohere"
    GENERIC = "generic"


class PassthroughStandardLoggingPayload(TypedDict, total=False):
    """
    Standard logging payload for all pass through endpoints
    """

    url: str
    """
    The full url of the request
    """

    request_method: str | None
    """
    The method of the request
    "GET", "POST", "PUT", "DELETE", etc.
    """

    request_body: dict | None
    """
    The body of the request
    """
    response_body: dict | None  # only tracked for non-streaming responses
    """
    The body of the response
    """

    cost_per_request: float | None
    """
    The cost per request to the target endpoint

    Optional field, we use this for cost tracking only if it's set.
    """
