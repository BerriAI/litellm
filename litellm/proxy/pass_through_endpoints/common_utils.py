from collections.abc import Mapping
from typing import Final

from fastapi import Request
from pydantic import TypeAdapter, ValidationError

_AUTH_FLAG: Final = TypeAdapter(bool)


def get_litellm_virtual_key(request: Request) -> str:
    """
    Extract and format API key from request headers.
    Prioritizes x-litellm-api-key over Authorization header.


    Vertex JS SDK uses `Authorization` header, we use `x-litellm-api-key` to pass litellm virtual key

    """
    litellm_api_key: Final = request.headers.get("x-litellm-api-key")
    if litellm_api_key:
        return f"Bearer {litellm_api_key}"
    return request.headers.get("Authorization", "")


def pass_through_auth_enforced(endpoint: Mapping[str, object]) -> bool:
    """
    Read a generic pass-through endpoint's ``auth`` the way ``PassThroughGenericEndpoint.auth`` does.

    A missing key or a value pydantic cannot read as a bool keeps auth on.
    """
    try:
        return _AUTH_FLAG.validate_python(endpoint.get("auth", True))
    except ValidationError:
        return True
