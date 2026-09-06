from __future__ import annotations

from typing import Final

from ...shared.parity.fixtures.inputs import generate_case_inputs
from .sdk.chat_completions.fixtures.record import input_strategy as chat_input_strategy
from .sdk.messages.fixtures.record import input_strategy as messages_input_strategy
from .sdk.responses.fixtures.record import input_strategy as responses_input_strategy

MODEL: Final = "anthropic/claude-sonnet-5"


def test_messages_recording_strategy_generates_replayable_inputs() -> None:
    cases: Final = generate_case_inputs(messages_input_strategy(MODEL), 8)

    assert cases
    assert all(case.model == MODEL and case.body["messages"] and not case.provider_responses for case in cases)


def test_chat_recording_strategy_stays_inside_rust_supported_surface() -> None:
    cases: Final = generate_case_inputs(chat_input_strategy(MODEL), 8)

    assert cases
    assert all(case.messages and case.messages[0]["role"] in {"system", "user"} for case in cases)
    assert all("stream" not in case.optional_params and not case.provider_responses for case in cases)


def test_responses_recording_strategy_uses_chat_bridge_compatible_inputs() -> None:
    cases: Final = generate_case_inputs(responses_input_strategy(MODEL), 8)

    assert cases
    assert all(case.sdk_input and "tools" not in case.params and not case.provider_responses for case in cases)
