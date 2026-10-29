import asyncio
from collections.abc import Callable
from typing import Final

import pytest
from fastapi import HTTPException, Response

from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.openai_error_payload import LITELLM_CALL_ID_HEADER
from litellm.proxy.common_utils.token_count_log import answered_with_litellm_call_id, logged_token_count
from litellm.types.utils import TokenCountResponse


class _TokenizerPanic(BaseException):
    pass


async def _count_ending_in(error: BaseException) -> TokenCountResponse:
    raise error


@pytest.mark.asyncio
async def test_a_count_ending_in_a_base_exception_logs_a_500(
    token_count_log_lines: Callable[[], tuple[dict[str, object], ...]],
) -> None:
    with pytest.raises(_TokenizerPanic):
        await logged_token_count(_count_ending_in(_TokenizerPanic()), "count-log-panic", "count-log-gpt", None)

    assert token_count_log_lines() == (
        {
            "litellm_call_id": "count-log-panic",
            "request_model": "count-log-gpt",
            "model_id": None,
            "model_used": None,
            "input_tokens": None,
            "count_source": None,
            "status_code": 500,
        },
    )


@pytest.mark.asyncio
async def test_a_cancelled_count_writes_no_line(
    token_count_log_lines: Callable[[], tuple[dict[str, object], ...]],
) -> None:
    with pytest.raises(asyncio.CancelledError):
        await logged_token_count(_count_ending_in(asyncio.CancelledError()), "count-log-cancelled", "count-log-gpt", None)

    assert token_count_log_lines() == ()


@pytest.mark.asyncio
async def test_an_http_error_answers_its_status_and_detail_with_the_call_id_beside_its_own_headers() -> None:
    rejection: Final = HTTPException(status_code=429, detail={"error": "slow down"}, headers={"retry-after": "3"})

    with pytest.raises(HTTPException) as answered:
        await answered_with_litellm_call_id(_count_ending_in(rejection), "count-echo-http", Response())

    assert (answered.value.status_code, answered.value.detail, dict(answered.value.headers or {})) == (
        429,
        {"error": "slow down"},
        {"retry-after": "3", LITELLM_CALL_ID_HEADER: "count-echo-http"},
    )


@pytest.mark.asyncio
async def test_an_unhandled_error_answers_a_500_with_the_call_id() -> None:
    with pytest.raises(ProxyException) as answered:
        await answered_with_litellm_call_id(
            _count_ending_in(RuntimeError("tokenizer crashed")), "count-echo-crash", Response()
        )

    assert (answered.value.code, answered.value.headers) == ("500", {LITELLM_CALL_ID_HEADER: "count-echo-crash"})


@pytest.mark.asyncio
async def test_a_cancelled_answer_stays_cancelled_and_names_no_call_id() -> None:
    cancelled: Final = asyncio.CancelledError()
    fastapi_response: Final = Response()

    with pytest.raises(asyncio.CancelledError) as raised:
        await answered_with_litellm_call_id(_count_ending_in(cancelled), "count-echo-cancelled", fastapi_response)

    assert raised.value is cancelled
    assert LITELLM_CALL_ID_HEADER not in fastapi_response.headers
