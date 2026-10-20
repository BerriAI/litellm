from collections.abc import Awaitable
from types import MappingProxyType
from typing import Final

from fastapi import status

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.openai_error_payload import error_status_code
from litellm.types.token_count import TokenCountSource
from litellm.types.utils import TokenCountResponse


def _log_token_count(
    litellm_call_id: str,
    request_model: str,
    counted: TokenCountResponse | None,
    status_code: int,
) -> None:
    model_used: Final = None if counted is None else counted.model_used
    input_tokens: Final = None if counted is None else counted.total_tokens
    count_source: Final[TokenCountSource | None] = None if counted is None else counted.count_source
    verbose_proxy_logger.info(
        "token count - litellm_call_id=%s request_model=%s model_used=%s input_tokens=%s count_source=%s "
        "status_code=%s",
        litellm_call_id,
        request_model,
        model_used,
        input_tokens,
        count_source,
        status_code,
        extra=MappingProxyType(
            {
                "litellm_call_id": litellm_call_id,
                "request_model": request_model,
                "model_used": model_used,
                "input_tokens": input_tokens,
                "count_source": count_source,
                "status_code": status_code,
            }
        ),
    )


async def logged_token_count(
    count: Awaitable[TokenCountResponse], litellm_call_id: str, request_model: str
) -> TokenCountResponse:
    try:
        counted: Final = await count
    except Exception as e:
        _log_token_count(
            litellm_call_id, request_model, None, error_status_code(e, status.HTTP_500_INTERNAL_SERVER_ERROR)
        )
        raise
    _log_token_count(litellm_call_id, request_model, counted, status.HTTP_200_OK)
    return counted
