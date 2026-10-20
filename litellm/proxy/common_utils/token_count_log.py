import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeVar

from fastapi import status

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.openai_error_payload import error_status_code
from litellm.types.token_count import TokenCountSource
from litellm.types.utils import TokenCountResponse


def _logfmt_value(value: str | None) -> str:
    if value is None:
        return "None"
    bare: Final = value != "" and value.isprintable() and not any(ch.isspace() or ch in '"=' for ch in value)
    return value if bare else json.dumps(value)


_Counted: Final = TypeVar("_Counted")
_ProviderCount: Final = TypeVar("_ProviderCount", bound=Mapping[str, object])


@dataclass(frozen=True, slots=True)
class _CountFields:
    model_used: str | None
    input_tokens: int | None
    count_source: TokenCountSource | None


_FAILED_COUNT: Final = _CountFields(model_used=None, input_tokens=None, count_source=None)


def _log_token_count(litellm_call_id: str, request_model: str, fields: _CountFields, status_code: int) -> None:
    verbose_proxy_logger.info(
        "token count - litellm_call_id=%s request_model=%s model_used=%s input_tokens=%s count_source=%s "
        "status_code=%s",
        _logfmt_value(litellm_call_id),
        _logfmt_value(request_model),
        _logfmt_value(fields.model_used),
        fields.input_tokens,
        fields.count_source,
        status_code,
        extra=MappingProxyType(
            {
                "litellm_call_id": litellm_call_id,
                "request_model": request_model,
                "model_used": fields.model_used,
                "input_tokens": fields.input_tokens,
                "count_source": fields.count_source,
                "status_code": status_code,
            }
        ),
    )


async def _logged(
    count: Awaitable[_Counted],
    litellm_call_id: str,
    request_model: str,
    fields_of: Callable[[_Counted], _CountFields],
) -> _Counted:
    try:
        counted: Final = await count
    except Exception as e:
        _log_token_count(
            litellm_call_id, request_model, _FAILED_COUNT, error_status_code(e, status.HTTP_500_INTERNAL_SERVER_ERROR)
        )
        raise
    _log_token_count(litellm_call_id, request_model, fields_of(counted), status.HTTP_200_OK)
    return counted


def _token_count_response_fields(counted: TokenCountResponse) -> _CountFields:
    return _CountFields(
        model_used=counted.model_used, input_tokens=counted.total_tokens, count_source=counted.count_source
    )


async def logged_token_count(
    count: Awaitable[TokenCountResponse], litellm_call_id: str, request_model: str
) -> TokenCountResponse:
    return await _logged(count, litellm_call_id, request_model, _token_count_response_fields)


async def logged_provider_token_count(
    count: Awaitable[_ProviderCount], litellm_call_id: str, request_model: str, model_used: str
) -> _ProviderCount:
    def provider_fields(counted: _ProviderCount) -> _CountFields:
        input_tokens: Final = counted.get("input_tokens")
        return _CountFields(
            model_used=model_used,
            input_tokens=input_tokens if isinstance(input_tokens, int) and not isinstance(input_tokens, bool) else None,
            count_source="provider",
        )

    return await _logged(count, litellm_call_id, request_model, provider_fields)
