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

CLAUDE_SONNET_5_5_THINKING_BUDGET_SIGNATURE_1: Final = (
    "CAQS2gkKEAgTGAI4AUIIdGhpbmtpbmcSDCKQrdbAsNSlkOf3zBoMeduhYaXR7h2XbdAKIjAtYK1x82HeQuy3OMDRyk8Ru1NL6Bo2p4jMGpeqJSUAqMBl"
    "hxlFnMpq2arnT1bWs3kq9wjg19gHCDS58DA+DUr3YdYx6gy1aX/Qn1XeFVB1L/kSAHlK0Wuxzn41ZM3D3xdGPeg9El2AIQryUlRs/POxJdqRzXes2H5U"
    "fUU1f/azCoeDJv9RRk234jIMkgBCWiLstzExNwxL+kvTlY2mrQXQ+U9mhXRJC90AilZlFIY8fuRhL8HMFA6d0rqMisr3wLd9M7ACwVP3cOZXx1Wa9Sfb"
    "2yiutDHkG4fI61xE8cDTYvOqtyY6ZQ9juiV6/rOPDSP+85sLVVmH+wJUJXIEfYBRCmytchPqR/Y/Q1EdMnmiLYfssjWsj8NkfsEBjOHdlIYDTzvplkEK"
    "y2SHSBwQeHcahHr1Y1napFobIRjVuQnPOMGvhNLUJZwN4zJLAQXzxO5hK2KrFOg66nX3A/+77CGVUr+CpC+/Wbkyc7LQNfI+xzsdRjC7epieVPbgVFDS"
    "Rt0a7nppVA7JbYzZh04UdCoD7z2dDvX3lhE3WDdyICUatVgsRaU0w1Qmiu9fOwrfU6I29PaXDGw1i4pth6uL8GAS/ZuUC6xFhggEoXs497MgrD3N0uLQ"
    "69x+IGmuRzIPbcgWjC3IlNMI2se/FpfWXJAsScnCVWRqfBL8crjsmdbuwy4L/usMNT87H3vngSbpd9dZuKpKBW18Y+OPLQiF2im/Pv+zZpA1RJkOvXPr"
    "ru8eiHu9z2XKVIJgdxKkRjLgi4/JVnN54q3bdkSz7KARIiRc4EzKsyU3/PzJJN1Fev10hxbU4V+Pkv6hLcjzIL9pMGAYjZXrjpxyw+PTVt7jK2k1C1SB"
    "BXpHdmntJJ30E94IpSISyNIW9Twi3KPYcYhj/rd9WA5UwvhDtbvril6ET10JlqzV36l8RxWUPhlXegJgtEMdnjNplhfDeGsMwg8e3OL1GNRUQoqQNd1Q"
    "jP3bpHx0yaxaRlkGwgtIGXj2bxuwLwn5otMEMGzQEnnQVNHdjAIpuHunLH9k01S86ir6s5cP05+U1Y1V1U4dZiFVnr4vSwWLIZLpRg2k/O4/inqzIjOY"
    "LD9xE+d+Mm55vKnbYtIv+DM1GHpKdvbzSfa1czjV6YT1PKotUZpg4ul+uyQqZYhpq9dGYUttevBz5tPIs+XUL6nx3Kj8WnPtkaAD+MuH0+yuALNdwzGb"
    "4sxxAu4sYsbxHCzpriX+JLjgUq2bO2Y7ve1nHVsh1B1bzyatgv9hnAQyO7uLMP0IICF3F43zJtgol3/gng5bf1T6EeFHDyZAUxGAqbWGjPwh5awEfdLe"
    "SYc2AtK84ieDLU0LfhvuGRdmAcVXrGmF1D/mhHg95Hs8fKt5QN7O1/gUSsC0TvXGrpbYeE5OH/O2aFFzimURUyUp7W8f6UFNNy4ZHIjdUqomnUL8XH+o"
    "Rf69GBqbv8BsfPQuq9D6nAvFk+suLYhLvWdFRy/sMA/gXh2mq21t2bFXQXOjgq/dQ7GUJx0vU86alotjdPovkksZAqzHi1s5h3WrQ1h3KxF4pfVO309h"
    "lQiFvg3fNPaDARqFFci+7ToS0kQXk4r95A6XzGMYAQ=="
)

CLAUDE_SONNET_5_5_THINKING_BUDGET_SIGNATURE_2: Final = (
    "CAQS1gcKEQgTGAI4AUIJbmFycmF0aW9uEgxdyD7sRZxxSBB6HUEaDB0dBrarQ/maHvvYRiIwNo20xJqSuL5tpy0fFBXRggkfQd3MO+ehcdkzkAP1bNQF"
    "I7P1/rfhUbkzeiZI7NDDKvIG0xpIrPlf76ufYrZ2d9+04GJUImemCjrKGxjniCVaBpqVqJp0K7qWu24HZqZjX/pcKUi//W0t73h1BFmkUEy/8XSvsuUs"
    "u7lJqmMXACCXhkXZhg/EnswdEzKhYwVDay6cGqfGUUo+NBTmhMjYjy69h7x1+RPuEWTNNMhusS2K9U1hFwZjqEdjo/4CsWZzZC6/9WC213voHd6ewjly"
    "QQLjEKJgoQ4Mnlyep7NVe3b4Hzu0pChITa/PUyhEGEoQhIZoMx3/hYAf3HwqveaPBQRmrjf8m3bvcbh7R0hTJHS3dpUrz5RE6d8zSy6EGWTDTYqb6JpQ"
    "DdL0+5UxQMZEc6JTcpcBpfxJvS2KFyxyZwOkt02OdjEoTkPxaQTK9fnjVSq9xp5qLm/RC03XnBdKBG5Elk/Dxj4UqPphkKN+UQCyYKivCpkU/N0/MVm1"
    "/Rp0x5OQuZ/xVSbdyuF8ijlmK/FZuZ8P3sR0v5gvfPGvDoijdWaljc/+ZiJ7CK9LtgoSk16IYrCbKtbL75S7yGTHnWgnHSUzuXorpkf5xGuGKucloLTf"
    "DPaBsxw7mWUEYcoUE1LxUqip9j2p6Zf6BkOHX5TPgXB14F6194/vwIfoNV7QK/uXFT2kZ/6m4YDfTzmvDCtUP9PcfbyjOIPHXukL/KzbaunT91spQf28"
    "7CfAbZp0Tl0ATcVut63UB/3e26zxhZ45ZOLyjahllcLY43FWmPLM+tTUPtNmKQ99+JVBS8A4rqKZwnovgFJG4wmjbeBClY691hNoNVsIfPQs0RM0zgIh"
    "F4rutOqnPIBbcS/5eFGucSB3sAplbqDTQtHVIagaTFAgLtEBPxGqb9jze+Wwwt2RQ9xXE36bEEsgL+jgrd7CfiTs78RCV/G6tKzyfO5TJjQoX/8XUIag"
    "C5zdRpNg7cTRcDIZPkTl0Jii/Pgrhd3xkBG5heg9YOnsm1xuHkmTjGGGx1sDMeBfLiRPNyYTDHVAqejbqB2dyGLvfsx3DZgSLnv4IBXcesQ0am673hqL"
    "WYYMT7md2Z9JKoYGnTXqkbmXDkuf6H+RasWhGYkNJtF+XBxUwHVvbMjQaO2tJpQhv8msR3yORQn2Lnt5OF2ws69Irqt5tdkQCyCGn+O6xxOftE3gmEMZ"
    "0Udpgod2UvJj1Vz2udQhXAkmNRmPgLmp6jRWU7bmGAE="
)

CLAUDE_SONNET_5_5_THINKING_BUDGET_CONTENT: Final = [
    {
        "type": "thinking",
        "thinking": "",
        "signature": CLAUDE_SONNET_5_5_THINKING_BUDGET_SIGNATURE_1,
    },
    {
        "type": "thinking",
        "thinking": "",
        "signature": CLAUDE_SONNET_5_5_THINKING_BUDGET_SIGNATURE_2,
    },
    {
        "type": "tool_use",
        "id": "toolu_01CndWUrVgzs4gR9gg1R6w44",
        "name": "read_file",
        "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
        "caller": {"type": "direct"},
    },
    {
        "type": "tool_use",
        "id": "toolu_014RmKHkVks5GvQY6qZoXboC",
        "name": "read_file",
        "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 40},
        "caller": {"type": "direct"},
    },
]

CLAUDE_SONNET_5_5_THINKING_BUDGET_USAGE: Final = {
    "input_tokens": 4,
    "cache_creation_input_tokens": 873,
    "cache_read_input_tokens": 0,
    "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
    "output_tokens": 449,
    "output_tokens_details": {"thinking_tokens": 253},
    "service_tier": "standard",
    "inference_geo": "global",
    "speed": "standard",
}

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

CLAUDE_SONNET_5_5_THINKING_BUDGET_TEST_CASE: Final = replace(
    CLAUDE_SONNET_5_5_TEST_CASE,
    scenario="thinking_budget",
    litellm_request={
        **CLAUDE_SONNET_5_5_TEST_CASE.litellm_request,
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    },
    expected_provider_request={
        **CLAUDE_SONNET_5_5_TEST_CASE.expected_provider_request,
        "max_tokens": 2048,
        "thinking": {"type": "adaptive"},
    },
    mock_provider_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.mock_provider_response,
        "id": "msg_011CgcUJqytdPvGPzVpJNKEQ",
        "content": CLAUDE_SONNET_5_5_THINKING_BUDGET_CONTENT,
        "usage": CLAUDE_SONNET_5_5_THINKING_BUDGET_USAGE,
    },
    expected_litellm_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.expected_litellm_response,
        "id": "msg_011CgcUJqytdPvGPzVpJNKEQ",
        "content": CLAUDE_SONNET_5_5_THINKING_BUDGET_CONTENT,
        "usage": CLAUDE_SONNET_5_5_THINKING_BUDGET_USAGE,
    },
)


@pytest.mark.parametrize(
    "case",
    [
        CLAUDE_SONNET_4_6_THINKING_BUDGET_TEST_CASE,
        CLAUDE_SONNET_5_5_THINKING_BUDGET_TEST_CASE,
    ],
    ids=(
        CLAUDE_SONNET_4_6_THINKING_BUDGET_TEST_CASE.id,
        CLAUDE_SONNET_5_5_THINKING_BUDGET_TEST_CASE.id,
    ),
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
