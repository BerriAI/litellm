import json
import re
from asyncio import CancelledError
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeVar

from fastapi import status

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.openai_error_payload import error_status_code
from litellm.types.token_count import TokenCountSource
from litellm.types.utils import TokenCountResponse

_BARE_LOGFMT_VALUE: Final = re.compile(r"[\w.:/@+-]+", re.ASCII)
_JSON_ESCAPED_BRACES: Final = str.maketrans({"{": "\\u007b", "}": "\\u007d"})


def _logfmt_value(value: str | None) -> str:
    if value is None:
        return "None"
    if _BARE_LOGFMT_VALUE.fullmatch(value):
        return value
    return json.dumps(value).translate(_JSON_ESCAPED_BRACES)


_Counted: Final = TypeVar("_Counted")
_Rejection: Final = TypeVar("_Rejection")
_ProviderCount: Final = TypeVar("_ProviderCount", bound=Mapping[str, object])


@dataclass(frozen=True, slots=True)
class _CountFields:
    model_used: str | None
    input_tokens: int | None
    count_source: TokenCountSource | None


_FAILED_COUNT: Final = _CountFields(model_used=None, input_tokens=None, count_source=None)


@dataclass(frozen=True, slots=True)
class _CountRequest:
    litellm_call_id: str
    request_model: str | None
    model_id: str | None


def deployment_model_id(deployment: Mapping[str, object] | None) -> str | None:
    match deployment:
        case {"model_info": {"id": str() as model_id}}:
            return model_id
        case _:
            return None


def _log_token_count(request: _CountRequest, fields: _CountFields, status_code: int) -> None:
    verbose_proxy_logger.info(
        "token count - litellm_call_id=%s request_model=%s model_id=%s model_used=%s input_tokens=%s "
        "count_source=%s status_code=%s",
        _logfmt_value(request.litellm_call_id),
        _logfmt_value(request.request_model),
        _logfmt_value(request.model_id),
        _logfmt_value(fields.model_used),
        fields.input_tokens,
        fields.count_source,
        status_code,
        extra=MappingProxyType(
            {
                "litellm_call_id": request.litellm_call_id,
                "request_model": request.request_model,
                "model_id": request.model_id,
                "model_used": fields.model_used,
                "input_tokens": fields.input_tokens,
                "count_source": fields.count_source,
                "status_code": status_code,
            }
        ),
    )


def rejected_token_count(rejection: _Rejection, litellm_call_id: str, request_model: str | None) -> _Rejection:
    _log_token_count(
        _CountRequest(litellm_call_id, request_model, model_id=None),
        _FAILED_COUNT,
        error_status_code(rejection, status.HTTP_500_INTERNAL_SERVER_ERROR),
    )
    return rejection


async def _logged(
    count: Awaitable[_Counted], request: _CountRequest, fields_of: Callable[[_Counted], _CountFields]
) -> _Counted:
    try:
        counted: Final = await count
    except CancelledError:
        raise
    except BaseException as e:
        _log_token_count(request, _FAILED_COUNT, error_status_code(e, status.HTTP_500_INTERNAL_SERVER_ERROR))
        raise
    _log_token_count(request, fields_of(counted), status.HTTP_200_OK)
    return counted


def _token_count_response_fields(counted: TokenCountResponse) -> _CountFields:
    return _CountFields(
        model_used=counted.model_used, input_tokens=counted.total_tokens, count_source=counted.count_source
    )


async def logged_token_count(
    count: Awaitable[TokenCountResponse], litellm_call_id: str, request_model: str, model_id: str | None
) -> TokenCountResponse:
    return await _logged(count, _CountRequest(litellm_call_id, request_model, model_id), _token_count_response_fields)


async def logged_provider_token_count(
    count: Awaitable[_ProviderCount], litellm_call_id: str, request_model: str, model_id: str | None, model_used: str
) -> _ProviderCount:
    def provider_fields(counted: _ProviderCount) -> _CountFields:
        input_tokens: Final = counted.get("input_tokens")
        return _CountFields(
            model_used=model_used,
            input_tokens=input_tokens if isinstance(input_tokens, int) and not isinstance(input_tokens, bool) else None,
            count_source="provider",
        )

    return await _logged(count, _CountRequest(litellm_call_id, request_model, model_id), provider_fields)
