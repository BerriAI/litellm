import json
from typing import Any, Protocol, TypedDict, cast

from pydantic import BaseModel


class FallbackErrorInfo(TypedDict):
    message: str
    type: str
    param: str | None
    code: str | None


class _HiddenParamsHost(Protocol):
    _hidden_params: dict[str, object]


class HiddenParamsAsyncIteratorWrapper:
    """
    Wraps a bare async generator/iterator (e.g. a provider's raw SSE
    streaming response) that cannot itself hold a ``_hidden_params``
    attribute, so router-derived headers (ITPM/OTPM, model-group, retry,
    fallback) can attach to a streaming response the same way they attach
    to object-based responses (e.g. ``CustomStreamWrapper``).
    """

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self._hidden_params: dict[str, object] = {}

    def __aiter__(self) -> "HiddenParamsAsyncIteratorWrapper":
        return self

    async def __anext__(self) -> object:
        return await cast(Any, self._inner).__anext__()

    async def aclose(self) -> None:
        aclose = getattr(self._inner, "aclose", None)
        if callable(aclose):
            await aclose()


def prepare_response_for_header_attachment(response: object) -> object | None:
    if response is None:
        return None
    if isinstance(response, dict) or hasattr(response, "_hidden_params"):
        return response
    if hasattr(response, "__anext__"):
        return HiddenParamsAsyncIteratorWrapper(response)
    return response


def ensure_response_additional_headers(response: object) -> dict[str, object]:
    hidden_params = get_hidden_params_dict(response, create=isinstance(response, dict))
    _write_hidden_params(response, hidden_params)
    additional_headers = hidden_params.get("additional_headers")
    if not isinstance(additional_headers, dict):
        additional_headers = {}
        hidden_params["additional_headers"] = additional_headers
    return additional_headers


def apply_quality_router_decision_headers(
    additional_headers: dict[str, object],
    request_kwargs: object,
) -> None:
    metadata = (request_kwargs.get("metadata") or {}) if isinstance(request_kwargs, dict) else {}
    decision = metadata.get("quality_router_decision") if isinstance(metadata, dict) else None
    if not isinstance(decision, dict):
        return
    quality_header_fields = (
        ("routed_model", "x-litellm-quality-router-model"),
        ("quality_tier", "x-litellm-quality-router-tier"),
        ("routed_via", "x-litellm-quality-router-via"),
        ("matched_keyword", "x-litellm-quality-router-keyword"),
        ("complexity_tier", "x-litellm-quality-router-complexity"),
    )
    for field, header in quality_header_fields:
        if decision.get(field) is not None:
            additional_headers[header] = str(decision[field])


def response_in_flight_token_count(response: object) -> int:
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return 0
    if isinstance(usage, dict):
        total = int(usage.get("total_tokens") or 0)
        if total:
            return total
        return int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    return int(getattr(usage, "total_tokens", 0) or 0)


def apply_remaining_usage_headers(
    additional_headers: dict[str, object],
    remaining_usage: dict[str, int],
    in_flight_tokens: int,
) -> None:
    in_flight_delta = {
        "x-ratelimit-remaining-tokens": in_flight_tokens,
        "x-ratelimit-remaining-requests": 1,
    }
    for header, value in remaining_usage.items():
        if value is not None and header not in additional_headers:
            additional_headers[header] = value - in_flight_delta.get(header, 0)


def _normalize_hidden_params(hidden_params: object) -> dict[str, object]:
    if isinstance(hidden_params, BaseModel):
        return cast("dict[str, object]", hidden_params.model_dump())
    if isinstance(hidden_params, dict):
        return cast("dict[str, object]", hidden_params)
    return {}


def get_hidden_params_dict(
    response: object,
    *,
    create: bool = False,
) -> dict[str, object]:
    if isinstance(response, dict):
        hidden_params = _normalize_hidden_params(response.get("_hidden_params"))
        if not hidden_params and create:
            hidden_params = {}
            response["_hidden_params"] = hidden_params
        return hidden_params

    hidden_params = _normalize_hidden_params(cast(object, getattr(response, "_hidden_params", None)))
    return hidden_params


def _write_hidden_params(response: object, hidden_params: dict[str, object]) -> None:
    if isinstance(response, dict):
        response["_hidden_params"] = hidden_params
    elif hasattr(response, "_hidden_params"):
        cast(_HiddenParamsHost, response)._hidden_params = hidden_params


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

    if (
        not isinstance(response, BaseModel)
        and not isinstance(response, dict)
        and not hasattr(response, "_hidden_params")
    ):
        return response

    hidden_params = get_hidden_params_dict(response, create=isinstance(response, dict))
    additional_headers = _ensure_additional_headers_dict(hidden_params)
    additional_headers.update(headers)
    hidden_params["additional_headers"] = additional_headers

    _write_hidden_params(response, hidden_params)
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

    hidden_params = get_hidden_params_dict(response, create=isinstance(response, dict))
    additional_headers = _ensure_additional_headers_dict(hidden_params)
    merged_errors = get_fallback_errors_from_headers(additional_headers) + [
        cast("dict[str, object]", error) for error in fallback_errors
    ]
    additional_headers["x-litellm-fallback-errors"] = json.dumps(merged_errors)
    hidden_params["additional_headers"] = additional_headers
    _write_hidden_params(response, hidden_params)
    return response
