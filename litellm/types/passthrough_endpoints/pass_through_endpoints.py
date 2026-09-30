from enum import Enum
from typing import Final

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

# Request.state key for programmatic pass-through callers (e.g. Bedrock proxy) that attach
# JSON without a FastAPI `custom_body` parameter (which would consume the HTTP body).
LITELLM_PASS_THROUGH_CUSTOM_BODY_STATE_KEY: Final = "litellm_pass_through_custom_body"

# Request.state key for programmatic pass-through callers that must preserve an
# exact byte/string body, such as AWS SigV4-signed requests.
LITELLM_PASS_THROUGH_RAW_BODY_STATE_KEY: Final = "litellm_pass_through_raw_body"

# `model_info` of the router deployment a provider route (e.g. Vertex) resolved for this request.
LITELLM_PASS_THROUGH_DEPLOYMENT_MODEL_INFO_STATE_KEY: Final = "litellm_pass_through_deployment_model_info"

# Attribute set on the FastAPI endpoint function of every user-defined pass-through
# route. Auth reads it off the dispatched endpoint (``request.scope["endpoint"]``) to
# decide whether a request body ``model`` names an upstream model rather than a
# LiteLLM-managed one. Keying off the resolved endpoint (not the request path) means a
# custom path that collides with a built-in route never suppresses model-access checks:
# on a collision FastAPI dispatches the built-in handler, which does not carry this flag.
LITELLM_PASS_THROUGH_ENDPOINT_MARKER: Final = "__litellm_pass_through_endpoint__"


class EndpointType(str, Enum):
    VERTEX_AI = "vertex-ai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TINYFISH = "tinyfish"
    GENERIC = "generic"


class PassThroughAuthMode(str, Enum):
    PUBLIC = "public"
    ANY_KEY = "any_key"
    GRANTED_KEYS = "granted_keys"


_AUTH_FLAG: Final = TypeAdapter(bool)


def pass_through_auth_mode(auth: object) -> PassThroughAuthMode:
    try:
        enforced: Final = _AUTH_FLAG.validate_python(auth.strip() if isinstance(auth, str) else auth)
    except ValidationError:
        return PassThroughAuthMode.ANY_KEY
    return PassThroughAuthMode.GRANTED_KEYS if enforced else PassThroughAuthMode.PUBLIC


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
