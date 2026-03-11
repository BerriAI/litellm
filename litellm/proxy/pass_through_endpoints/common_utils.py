from fastapi import Request


def get_litellm_virtual_key(request: Request) -> str:
    """
    Extract and format API key from request headers.
    Prioritizes x-litellm-api-key over Authorization header.


    Vertex JS SDK uses `Authorization` header, we use `x-litellm-api-key` to pass litellm virtual key

    """
    litellm_api_key = request.headers.get("x-litellm-api-key")
    if litellm_api_key:
        return f"Bearer {litellm_api_key}"
    return request.headers.get("Authorization", "")
