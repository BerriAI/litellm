from typing import Final

import pytest
from pydantic import JsonValue

from litellm.llms.anthropic.beta_detection_models import BetaDetectionRequest


@pytest.mark.parametrize(
    "thinking",
    (
        {"type": "future", "display": "updates", "extension": {"keep": True}},
        "opaque thinking",
        ["unknown", 1],
        {"type": "enabled", "display": "future"},
        {"type": "adaptive", "display": ["updates"]},
    ),
)
def test_unknown_thinking_survives_projection(thinking: JsonValue) -> None:
    request: Final = BetaDetectionRequest.model_validate({"thinking": thinking, "messages": []})

    assert request.thinking is not None
    assert request.thinking.model_dump(mode="json") == thinking


def test_unknown_tool_target_survives_projection() -> None:
    target: Final = {"type": "future_tool", "extension": {"keep": True}}
    request: Final = BetaDetectionRequest.model_validate(
        {"messages": [{"role": "system", "content": [{"type": "tool_addition", "tool": target}]}]}
    )

    assert request.model_dump(mode="json")["messages"][0]["content"][0]["tool"] == target
