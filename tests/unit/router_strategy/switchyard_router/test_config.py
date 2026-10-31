from typing import Final

import pytest
from pydantic import ValidationError

from litellm.router_strategy.switchyard_router.config import SwitchyardRouterConfig


@pytest.mark.parametrize(
    "extra",
    [
        {"weights": [1]},
        {"weights": [-1, 2]},
        {"weights": [0, 0]},
        {"weights": [float("inf"), 1]},
        {"models": {"any": []}},
        {"models": {"any": [""]}},
        {"seed": -1},
        {"unknown": True},
    ],
)
def test_invalid_random_configuration_is_rejected(extra: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SwitchyardRouterConfig.model_validate({"algorithm": "random", "models": {"any": ["a", "b"]}, **extra})


@pytest.mark.parametrize(
    "extra",
    [
        {"capability": {"message_hash_fallback": True}},
        {"capability": {"base_threshold": 1.1}},
        {"escalation": {"deescalation": {"strong_min_calls": 5, "strong_max_calls": 2, "confirmations": 1}}},
        {"stage": {"deescalation_note": "return"}},
        {"stage": {"tool_semantics": {"unknown": ["tool"]}}},
    ],
)
def test_inconsistent_classifier_configuration_is_rejected(extra: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SwitchyardRouterConfig.model_validate(
            {
                "models": {"efficient": ["a"], "capable": ["b"], "judge": ["j"]},
                **extra,
            }
        )


def test_custom_configuration_roundtrip_keeps_schema_and_fallback_order() -> None:
    source: Final[dict[str, object]] = {
        "algorithm": "custom",
        "models": {"judge": ["j"], "code": ["a", "b"]},
        "custom": {
            "prompt": "Choose a target",
            "selector": "/target",
            "default_target": "code",
            "response_schema": {"type": "object", "properties": {"target": {"type": "string"}}},
        },
    }
    parsed: Final = SwitchyardRouterConfig.model_validate(source)
    restored: Final = SwitchyardRouterConfig.model_validate_json(parsed.model_dump_json())
    assert restored == parsed
    assert restored.models["code"] == ("a", "b")
