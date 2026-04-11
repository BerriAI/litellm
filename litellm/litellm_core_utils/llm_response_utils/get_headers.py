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
    Forward all headers as llm_provider-{header} while also preserving originals.

    Every vendor header is stored twice:
      1. Under its original key (e.g. ``TRAFFIC-TYPE``) so consumers
         can look it up by the name the vendor actually sent.
      2. Under a ``llm_provider-`` prefixed key (e.g.
         ``llm_provider-TRAFFIC-TYPE``) for backwards-compatibility with
         existing litellm consumers that rely on the prefix convention.

    Headers that are already prefixed with ``llm_provider`` are kept as-is.
    """
    llm_provider_headers = {}
    for k, v in response_headers.items():
        if "llm_provider" not in k:
            llm_provider_headers[k] = v
            _key = "{}-{}".format("llm_provider", k)
            llm_provider_headers[_key] = v
        else:
            llm_provider_headers[k] = v
    return llm_provider_headers
