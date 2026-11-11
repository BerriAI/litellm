import asyncio
from collections.abc import Callable
from typing import Final

import pytest
from fastapi import HTTPException, Request, Response

from litellm.proxy.common_utils.openai_error_payload import LITELLM_CALL_ID_HEADER
from litellm.proxy.common_utils.token_count_log import answered_with_litellm_call_id, logged_token_count
from litellm.types.utils import TokenCountResponse


class _TokenizerPanic(BaseException):
    pass


async def _count_ending_in(error: BaseException) -> TokenCountResponse:
    raise error


def _count_request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/utils/token_counter", "headers": []})


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
        await answered_with_litellm_call_id(_count_ending_in(rejection), "count-echo-http", _count_request(), Response())

    assert (answered.value.status_code, answered.value.detail, dict(answered.value.headers or {})) == (
        429,
        {"error": "slow down"},
        {"retry-after": "3", LITELLM_CALL_ID_HEADER: "count-echo-http"},
    )


class _CountTooLarge(HTTPException):
    pass


@pytest.mark.asyncio
async def test_an_http_error_reaches_its_handler_as_the_same_error_with_the_call_id() -> None:
    too_large: Final = _CountTooLarge(status_code=413, detail="count body too large")

    with pytest.raises(_CountTooLarge) as answered:
        await answered_with_litellm_call_id(_count_ending_in(too_large), "count-echo-same", _count_request(), Response())

    assert answered.value is too_large
    assert dict(answered.value.headers or {}) == {LITELLM_CALL_ID_HEADER: "count-echo-same"}


@pytest.mark.asyncio
async def test_a_cancelled_answer_stays_cancelled_and_names_no_call_id() -> None:
    cancelled: Final = asyncio.CancelledError()
    fastapi_response: Final = Response()

    with pytest.raises(asyncio.CancelledError) as raised:
        await answered_with_litellm_call_id(
            _count_ending_in(cancelled), "count-echo-cancelled", _count_request(), fastapi_response
        )

    assert raised.value is cancelled
    assert LITELLM_CALL_ID_HEADER not in fastapi_response.headers
