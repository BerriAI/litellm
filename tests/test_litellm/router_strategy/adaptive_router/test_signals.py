import json
from pathlib import Path
from typing import List, Tuple

import pytest

from litellm.router_strategy.adaptive_router.config import TOOL_CALL_HISTORY_MAX
from litellm.router_strategy.adaptive_router.signals import (
    SessionState,
    SignalDelta,
    Turn,
    apply_turn,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load(name: str) -> list:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text())


def _replay(turns: list) -> Tuple[SessionState, List[SignalDelta]]:
    state = SessionState(
        session_id="s",
        router_name="r",
        model_name="m",
        classified_type="general",
    )
    deltas: List[SignalDelta] = []
    for t in turns:
        deltas.append(
            apply_turn(
                state,
                Turn(
                    user_content=t.get("user_content"),
                    assistant_content=t.get("assistant_content"),
                    tool_calls=t.get("tool_calls", []),
                    tool_results=t.get("tool_results", []),
                    response_status=t.get("response_status"),
                ),
            )
        )
    return state, deltas


def test_clean_satisfaction_fires_satisfaction_only():
    state, _ = _replay(_load("clean_satisfaction"))
    assert state.satisfaction_count >= 1
    assert state.failure_count == 0
    assert state.disengagement_count == 0


def test_misalignment_fires_on_rephrase():
    state, _ = _replay(_load("misalignment_rephrase"))
    assert state.misalignment_count >= 1


def test_stagnation_fires_on_repeated_assistant():
    state, _ = _replay(_load("stagnation_repeat"))
    assert state.stagnation_count >= 1


def test_disengagement_fires_on_giveup():
    state, _ = _replay(_load("disengagement_giveup"))
    assert state.disengagement_count >= 1


def test_failure_fires_on_tool_error():
    state, _ = _replay(_load("failure_tool_error"))
    assert state.failure_count == 1


def test_loop_fires_on_repeated_tool():
    state, _ = _replay(_load("loop_same_tool"))
    assert state.loop_count >= 1


@pytest.mark.parametrize("fixture", ["exhaustion_429", "exhaustion_context_overflow"])
def test_exhaustion_fires_on_infra_signal(fixture):
    state, _ = _replay(_load(fixture))
    assert state.exhaustion_count >= 1


def test_no_signals_on_clean_session():
    state, _ = _replay(_load("clean_no_signals"))
    assert state.misalignment_count == 0
    assert state.stagnation_count == 0
    assert state.disengagement_count == 0
    assert state.failure_count == 0
    assert state.loop_count == 0
    assert state.exhaustion_count == 0


def test_mixed_failure_then_satisfaction():
    state, _ = _replay(_load("mixed_failure_then_satisfaction"))
    assert state.failure_count >= 1
    assert state.satisfaction_count >= 1


def test_apply_turn_is_o1_does_not_grow_history_unbounded():
    state = SessionState(
        session_id="s",
        router_name="r",
        model_name="m",
        classified_type="general",
    )
    for i in range(100):
        apply_turn(
            state,
            Turn(tool_calls=[{"name": f"tool_{i}", "arguments": {}}]),
        )
    assert len(state.tool_call_history) <= TOOL_CALL_HISTORY_MAX
