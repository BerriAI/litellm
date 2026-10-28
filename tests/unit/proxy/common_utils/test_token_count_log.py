import asyncio
from collections.abc import Callable

import pytest

from litellm.proxy.common_utils.token_count_log import logged_token_count
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
