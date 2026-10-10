"""A streamed request bills exactly what the same request bills without streaming.

Streaming reconstructs the response, its usage and its model name from chunks after the fact, in a separate code
path from the non-streamed one (LIT-6241, LIT-6872, LIT-7632, LIT-7729, LIT-9044, LIT-9065). Each case sends one
request plain and streamed through the same deployment and key, and both LiteLLM_SpendLogs rows must equal the
case's expected row
"""

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.cost_calculation.stream_parity.bases import anthropic, openai, openai_responses
from integration.cost_calculation.stream_parity.case import StreamParityTestCase
from integration.cost_calculation.stream_parity.runner import assert_stream_parity

CASES: Final = (
    openai.GPT_6_1_SOL_CHAT_COMPLETIONS_TEST_CASE,
    openai.GPT_6_1_SOL_MESSAGES_TEST_CASE,
    openai_responses.GPT_6_1_SOL_RESPONSES_TEST_CASE,
    openai_responses.GPT_6_1_SOL_CHAT_COMPLETIONS_TEST_CASE,
    anthropic.CLAUDE_OPUS_4_8_MESSAGES_TEST_CASE,
    anthropic.CLAUDE_OPUS_4_8_CHAT_COMPLETIONS_TEST_CASE,
    anthropic.CLAUDE_OPUS_4_8_RESPONSES_TEST_CASE,
)


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
@pytest.mark.timeout(180)
def test_streamed_request_bills_the_same_row_as_the_non_streamed_request(
    case: StreamParityTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_stream_parity(case, gateway, provider)
