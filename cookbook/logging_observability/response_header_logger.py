from collections.abc import Mapping
from typing import Final, cast

import httpx

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth


def _as_dict(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _child(parent: dict[str, object], key: str) -> dict[str, object]:
    child: Final = _as_dict(parent.get(key))
    parent[key] = child  # rebind-ok: callback metadata must be updated in place
    return child


def _headers(value: object) -> dict[str, str]:
    if isinstance(value, httpx.Headers):
        return dict(value)
    return {name: header for name, header in _as_dict(value).items() if isinstance(header, str)}


def get_error_headers(error: BaseException | None, seen: tuple[int, ...] = ()) -> dict[str, str]:
    if error is None or id(error) in seen:
        return {}
    response: Final[object] = getattr(error, "response", None)
    headers: Final = _headers(getattr(response, "headers", None)) or _headers(getattr(error, "headers", None))
    if headers:
        return headers
    visited: Final = (*seen, id(error))
    return get_error_headers(error.__cause__, visited) or get_error_headers(error.__context__, visited)


def save_headers(metadata: dict[str, object], headers: Mapping[str, object]) -> None:
    existing: Final = _as_dict(metadata.get("spend_logs_metadata"))
    metadata["spend_logs_metadata"] = {  # rebind-ok: callback metadata must be updated in place
        **existing,
        "upstream_response_headers": {
            **_as_dict(existing.get("upstream_response_headers")),
            **headers,
        },
    }


class ResponseHeaderLogger(CustomLogger):
    async def async_post_call_response_headers_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
        request_headers: dict[str, str] | None = None,
        litellm_call_info: dict[str, object] | None = None,
    ) -> None:
        headers: Final = _headers(getattr(response, "_response_headers", None))
        spend_metadata: Final = _child(_child(data, "metadata"), "spend_logs_metadata")
        spend_metadata["upstream_response_headers"] = {f"llm_provider-{name}": value for name, value in headers.items()}

    async def async_logging_hook(
        self, kwargs: dict[str, object], result: object, call_type: str
    ) -> tuple[dict[str, object], object]:
        payload: Final = _as_dict(kwargs.get("standard_logging_object"))
        hidden: Final = _as_dict(payload.get("hidden_params"))
        headers: Final = _as_dict(hidden.get("additional_headers"))
        params: Final = _child(kwargs, "litellm_params")
        key: Final = "litellm_metadata" if params.get("litellm_metadata") else "metadata"
        save_headers(_child(params, key), headers)
        return kwargs, result

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        headers: Final = get_error_headers(original_exception)
        save_headers(
            _child(request_data, "metadata"),
            {f"llm_provider-{name}": value for name, value in headers.items()},
        )


proxy_handler_instance: Final = ResponseHeaderLogger()
