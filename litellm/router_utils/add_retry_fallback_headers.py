import json
from typing import Protocol, TypedDict, cast

from pydantic import BaseModel


class FallbackErrorInfo(TypedDict):
    message: str
    type: str
    param: str | None
    code: str | None


class _HiddenParamsHost(Protocol):
    _hidden_params: dict[str, object]


def get_hidden_params_dict(response: object) -> dict[str, object]:
    hidden_params: object = cast(object, getattr(response, "_hidden_params", None))
    if isinstance(hidden_params, BaseModel):
        return cast("dict[str, object]", hidden_params.model_dump())
    if isinstance(hidden_params, dict):
        return cast("dict[str, object]", hidden_params)
    return {}


def _ensure_additional_headers_dict(
    hidden_params: dict[str, object],
) -> dict[str, object]:
    additional_headers = hidden_params.get("additional_headers")
    if isinstance(additional_headers, dict):
        return cast("dict[str, object]", additional_headers)
    return {}


def get_fallback_error_info(error: Exception) -> FallbackErrorInfo:
    message = cast(object, getattr(error, "message", str(error)))
    error_type = cast(object, getattr(error, "type", error.__class__.__name__))
    param = cast(object, getattr(error, "param", None))
    code = cast(object, getattr(error, "status_code", getattr(error, "code", None)))
    return FallbackErrorInfo(
        message=str(message),
        type=str(error_type),
        param=str(param) if param is not None else None,
        code=str(code) if code is not None else None,
    )


def _coerce_error_dicts(items: list[object]) -> list[dict[str, object]]:
    return [cast("dict[str, object]", item) for item in items if isinstance(item, dict)]


def get_fallback_errors_from_headers(
    additional_headers: dict[str, object],
) -> list[dict[str, object]]:
    existing_errors = additional_headers.get("x-litellm-fallback-errors")
    if isinstance(existing_errors, list):
        return _coerce_error_dicts(cast("list[object]", existing_errors))
    if isinstance(existing_errors, str):
        try:
            parsed_errors: object = cast(object, json.loads(existing_errors))
        except json.JSONDecodeError:
            return []
        if isinstance(parsed_errors, list):
            return _coerce_error_dicts(cast("list[object]", parsed_errors))
    return []


def _add_headers_to_response(response: object, headers: dict[str, object]) -> object:
    """
    Helper function to add headers to a response's hidden params
    """
    if response is None:
        return response

    if not isinstance(response, BaseModel) and not hasattr(response, "_hidden_params"):
        return response

    hidden_params = get_hidden_params_dict(response)
    additional_headers = _ensure_additional_headers_dict(hidden_params)
    additional_headers.update(headers)
    hidden_params["additional_headers"] = additional_headers

    cast(_HiddenParamsHost, response)._hidden_params = hidden_params
    return response


def add_retry_headers_to_response(
    response: object,
    attempted_retries: int,
    max_retries: int | None = None,
) -> object:
    """
    Add retry headers to the request
    """
    retry_headers: dict[str, object] = {
        "x-litellm-attempted-retries": attempted_retries,
    }
    if max_retries is not None:
        retry_headers["x-litellm-max-retries"] = max_retries

    return _add_headers_to_response(response, retry_headers)


def add_fallback_headers_to_response(
    response: object,
    attempted_fallbacks: int,
    fallback_errors: list[FallbackErrorInfo] | None = None,
) -> object:
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
    fallback_headers: dict[str, object] = {
        "x-litellm-attempted-fallbacks": attempted_fallbacks,
    }
    response = _add_headers_to_response(response, fallback_headers)
    if fallback_errors is None or response is None:
        return response

    hidden_params = get_hidden_params_dict(response)
    additional_headers = _ensure_additional_headers_dict(hidden_params)
    merged_errors = get_fallback_errors_from_headers(additional_headers) + [
        cast("dict[str, object]", error) for error in fallback_errors
    ]
    additional_headers["x-litellm-fallback-errors"] = json.dumps(merged_errors)
    hidden_params["additional_headers"] = additional_headers
    cast(_HiddenParamsHost, response)._hidden_params = hidden_params
    return response
