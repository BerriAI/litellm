"""A streamed request bills exactly what the same request bills without streaming.

Streaming reconstructs the response, its usage and its model name from chunks after the fact, in a separate code
path from the non-streamed one (LIT-6241, LIT-6872, LIT-7632, LIT-7729, LIT-9044, LIT-9065). Each case is a plain and
a streamed CostTrackingTestCase for one request, sent through the same key, and both LiteLLM_SpendLogs rows must
bill the case's expected row.

Three cells are the native routes (chat, /v1/messages, /v1/responses); the rest are the bridges that translate one
API into another before the provider call, each with its own stream reassembly. PARITY_CASES are the plain and
streamed twins migrated from cost_tracking_cases.json that bill the same row.
"""

from typing import Final

import pytest
from integration._support.client import Gateway
from integration.cost_calculation.catalog import PARITY_CASES
from integration.cost_calculation.stream_parity.bases import anthropic, bedrock, openai, openai_responses, xai
from integration.cost_calculation.stream_parity.case import StreamParityTestCase
from integration.cost_calculation.stream_parity.runner import assert_stream_parity

NATIVE: Final = (
    openai.GPT_6_1_SOL_CHAT_COMPLETIONS_PARITY,
    anthropic.CLAUDE_OPUS_5_5_MESSAGES_PARITY,
    openai_responses.GPT_6_1_SOL_RESPONSES_PARITY,
)
BRIDGES: Final = (
    xai.GROK_4_7_MESSAGES_PARITY,
    openai.GPT_6_1_SOL_MESSAGES_PARITY,
    bedrock.CLAUDE_OPUS_5_5_CONVERSE_MESSAGES_PARITY,
    openai_responses.GPT_5_5_PRO_CHAT_COMPLETIONS_PARITY,
    anthropic.CLAUDE_OPUS_5_5_RESPONSES_PARITY,
)
CASES: Final = (*NATIVE, *BRIDGES, *PARITY_CASES)


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
@pytest.mark.timeout(180)
def test_streamed_request_bills_the_same_row_as_the_non_streamed_request(
    case: StreamParityTestCase, gateway: Gateway
) -> None:
    assert_stream_parity(case, gateway)
