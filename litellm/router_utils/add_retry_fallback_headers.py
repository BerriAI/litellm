from typing import Any, Optional, Union

from pydantic import BaseModel

from litellm.types.utils import HiddenParams


def _add_headers_to_response(response: Any, headers: dict) -> Any:
    """
    Helper function to add headers to a response's hidden params
    """
    if response is None or not isinstance(response, BaseModel):
        return response

    hidden_params: Optional[Union[dict, HiddenParams]] = getattr(
        response, "_hidden_params", {}
    )

    if hidden_params is None:
        hidden_params_dict = {}
    elif isinstance(hidden_params, HiddenParams):
        hidden_params_dict = hidden_params.model_dump()
    else:
        hidden_params_dict = hidden_params

    hidden_params_dict.setdefault("additional_headers", {})
    hidden_params_dict["additional_headers"].update(headers)

    setattr(response, "_hidden_params", hidden_params_dict)
    return response


def add_retry_headers_to_response(
    response: Any,
    attempted_retries: int,
    max_retries: Optional[int] = None,
) -> Any:
    """
    Add retry headers to the request
    """
    retry_headers = {
        "x-litellm-attempted-retries": attempted_retries,
    }
    if max_retries is not None:
        retry_headers["x-litellm-max-retries"] = max_retries

    return _add_headers_to_response(response, retry_headers)


def add_fallback_headers_to_response(
    response: Any,
    attempted_fallbacks: int,
) -> Any:
    """
    Add fallback headers to the response

    Args:
        response: The response to add the headers to
        attempted_fallbacks: The number of fallbacks attempted

    Returns:
        The response with the headers added

    Note: It's intentional that we don't add max_fallbacks in response headers
    Want to avoid bloat in the response headers for performance.
    """
    fallback_headers = {
        "x-litellm-attempted-fallbacks": attempted_fallbacks,
    }
    return _add_headers_to_response(response, fallback_headers)
