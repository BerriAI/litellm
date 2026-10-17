from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.anthropic import CLAUDE_SONNET_4_6_TEST_CASE, CLAUDE_SONNET_5_5_TEST_CASE
from integration.translation.messages.bases.azure_ai import (
    CLAUDE_SONNET_5_5_TEST_CASE as AZURE_CLAUDE_SONNET_5_5_TEST_CASE,
)
from integration.translation.messages.bases.vertex_ai import (
    CLAUDE_SONNET_5_5_TEST_CASE as VERTEX_CLAUDE_SONNET_5_5_TEST_CASE,
)
from integration.translation.response_contracts import messages_structure
from integration.translation.runner import assert_translation
from pydantic import JsonValue

SIGNATURE_1: Final = (
    "EpECCqgBCBIYAipAivUPApu85FYYe3+cXal8EiJOza7QGqKyekC8vDSn4oyeqGa2CrarO4abiuG7dzBXjmYR8+daw4h50ZjKmak7czIRY2xh"
    "dWRlLXNvbm5ldC00LTY4AEIIdGhpbmtpbmdaJGQwMDgxZjJiLWQ5NjEtNGFhYi05ZTRjLTcxYmU3ZTA0ZTY3MJoBEwoRY2xhdWRlLXNvbm5l"
    "dC00LTaoAY3fhdYGEgwJHsjNkCTlV9k1jWQaDOmeP/z67YtLTojSqCIwhRXGrNzSuGfMD1HqA72lctQCy83Wkr0u8W5lBXXn+MD6WfJGTJqM"
    "1FW7qRmOMOKJKhbheeMpsTs7XvdvsiDQqgM4PAJt4cwgGAE="
)

CLAUDE_SONNET_4_6_THINKING_BUDGET_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="thinking_budget",
    litellm_request={
        **CLAUDE_SONNET_4_6_TEST_CASE.litellm_request,
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    },
    expected_provider_request={
        **CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request,
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    },
    mock_provider_response={
        **CLAUDE_SONNET_4_6_TEST_CASE.mock_provider_response,
        "id": "msg_011CffzUREgTzMm1dXRqP2LR",
        "content": [
            {"type": "thinking", "thinking": "Hello!", "signature": SIGNATURE_1},
            {"type": "text", "text": "Hello!"},
        ],
        "usage": {
            "input_tokens": 47,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 15,
            "output_tokens_details": {"thinking_tokens": 7},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **CLAUDE_SONNET_4_6_TEST_CASE.expected_litellm_response,
        "id": "msg_011CffzUREgTzMm1dXRqP2LR",
        "content": [
            {"type": "thinking", "thinking": "Hello!", "signature": SIGNATURE_1},
            {"type": "text", "text": "Hello!"},
        ],
        "usage": {
            "input_tokens": 47,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 15,
            "output_tokens_details": {"thinking_tokens": 7},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
)


@pytest.mark.parametrize(
    "case", [CLAUDE_SONNET_4_6_THINKING_BUDGET_TEST_CASE], ids=(CLAUDE_SONNET_4_6_THINKING_BUDGET_TEST_CASE.id,)
)
def test_messages_reasoning_anthropic(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)


@pytest.mark.parametrize(
    "case",
    (CLAUDE_SONNET_5_5_TEST_CASE, AZURE_CLAUDE_SONNET_5_5_TEST_CASE, VERTEX_CLAUDE_SONNET_5_5_TEST_CASE),
    ids=(CLAUDE_SONNET_5_5_TEST_CASE.id, AZURE_CLAUDE_SONNET_5_5_TEST_CASE.id, VERTEX_CLAUDE_SONNET_5_5_TEST_CASE.id),
)
@pytest.mark.parametrize(
    "thinking,signature,preserve",
    (
        pytest.param("I need the file before answering.", "scripted-signed-history", True, id="visible"),
        pytest.param("", "scripted-signed-history", True, id="signed_empty"),
        pytest.param("", "", False, id="empty_signature"),
        pytest.param("", None, False, id="unsigned_empty"),
        pytest.param("I need the file before answering.", None, True, id="unsigned_visible"),
    ),
)
def test_messages_replays_signed_thinking_and_tool_results_without_losing_blocks(
    case: TranslationTestCase,
    thinking: str,
    signature: str | None,
    preserve: bool,
    gateway: Gateway,
    provider: SharedProvider,
) -> None:
    content: Final[list[JsonValue]] = [
        {"type": "thinking", "thinking": thinking, **({"signature": signature} if signature is not None else {})},
        {"type": "redacted_thinking", "data": "scripted-redacted-history"},
        {
            "type": "tool_use",
            "id": "toolu_history",
            "name": "read_file",
            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 200},
        },
    ]
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": "Read the validation file."},
        {
            "role": "assistant",
            "content": content,
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_history", "content": "quantity must be positive"}
            ],
        },
    ]
    replay: Final = replace(
        case,
        scenario="signed_thinking_replay",
        litellm_request={**case.litellm_request, "messages": history},
        expected_provider_request={
            **case.expected_provider_request,
            "messages": [
                history[0],
                {"role": "assistant", "content": content if preserve else content[1:]},
                history[2],
            ],
        },
    )
    body: Final = assert_translation(replay, gateway, provider)
    messages_structure(body)
