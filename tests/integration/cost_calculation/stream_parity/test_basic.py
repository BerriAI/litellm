"""A streamed request bills exactly what the same request bills without streaming.

Streaming reconstructs the response, its usage and its model name from chunks after the fact, in a separate code
path from the non-streamed one (LIT-6241, LIT-6872, LIT-7632, LIT-7729, LIT-9044, LIT-9065). Each case is a plain and
a streamed CostTrackingTestCase for one request, sent through the same key, and both LiteLLM_SpendLogs rows must
bill the case's expected row.

Three cells are the native routes (chat, /v1/messages, /v1/responses); the rest are the bridges that translate one
API into another before the provider call, each with its own stream reassembly
"""

from typing import Final

import pytest
from integration._support.client import Gateway
from integration.cost_calculation.stream_parity.bases import anthropic, bedrock, groq, openai, openai_responses
from integration.cost_calculation.stream_parity.case import StreamParityTestCase
from integration.cost_calculation.stream_parity.runner import assert_stream_parity

NATIVE: Final = (
    openai.GPT_5_4_MINI_CHAT_COMPLETIONS_PARITY,
    anthropic.CLAUDE_SONNET_5_MESSAGES_PARITY,
    openai_responses.GPT_5_3_CODEX_RESPONSES_PARITY,
)
BRIDGES: Final = (
    groq.QWEN_3_8_MESSAGES_PARITY,
    openai.GPT_5_4_MINI_MESSAGES_PARITY,
    bedrock.CLAUDE_OPUS_5_CONVERSE_MESSAGES_PARITY,
    openai_responses.GPT_5_3_CODEX_CHAT_COMPLETIONS_PARITY,
    anthropic.CLAUDE_SONNET_5_RESPONSES_PARITY,
)
CASES: Final = (*NATIVE, *BRIDGES)


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
@pytest.mark.timeout(180)
def test_streamed_request_bills_the_same_row_as_the_non_streamed_request(
    case: StreamParityTestCase, gateway: Gateway
) -> None:
    assert_stream_parity(case, gateway)
