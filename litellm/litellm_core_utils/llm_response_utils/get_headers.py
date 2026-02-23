from typing import Optional


def get_response_headers(_response_headers: Optional[dict] = None) -> dict:
    """

    Sets the Appropriate OpenAI headers for the response and forward all headers as llm_provider-{header}

    Note: _response_headers Passed here should be OpenAI compatible headers

    Args:
        _response_headers (Optional[dict], optional): _response_headers. Defaults to None.

    Returns:
        dict: _response_headers with OpenAI headers and llm_provider-{header}

    """
    if _response_headers is None:
        return {}

    openai_headers = {}
    if "x-ratelimit-limit-requests" in _response_headers:
        openai_headers["x-ratelimit-limit-requests"] = _response_headers[
            "x-ratelimit-limit-requests"
        ]
    if "x-ratelimit-remaining-requests" in _response_headers:
        openai_headers["x-ratelimit-remaining-requests"] = _response_headers[
            "x-ratelimit-remaining-requests"
        ]
    if "x-ratelimit-limit-tokens" in _response_headers:
        openai_headers["x-ratelimit-limit-tokens"] = _response_headers[
            "x-ratelimit-limit-tokens"
        ]
    if "x-ratelimit-remaining-tokens" in _response_headers:
        openai_headers["x-ratelimit-remaining-tokens"] = _response_headers[
            "x-ratelimit-remaining-tokens"
        ]
    llm_provider_headers = _get_llm_provider_headers(_response_headers)
    return {**llm_provider_headers, **openai_headers}


def _get_llm_provider_headers(response_headers: dict) -> dict:
    """
    Adds a llm_provider-{header} to all headers that are not already prefixed with llm_provider

    Forward all headers as llm_provider-{header}

    """
    llm_provider_headers = {}
    for k, v in response_headers.items():
        if "llm_provider" not in k:
            _key = "{}-{}".format("llm_provider", k)
            llm_provider_headers[_key] = v
        else:
            llm_provider_headers[k] = v
    return llm_provider_headers
