from typing import Final

import pytest
from pydantic import ValidationError

from litellm.router_strategy.switchyard_router.config import SwitchyardRouterConfig, validate_switchyard_config_write


@pytest.mark.parametrize("classifier", [None, {}])
def test_default_confidence_depends_on_classifier(classifier: dict[str, object] | None) -> None:
    config: Final = SwitchyardRouterConfig.model_validate(
        {
            "models": {
                "capable": "strong",
                "efficient": "fast",
                **({"judge": "judge"} if classifier is not None else {}),
            },
            "classifier": classifier,
        }
    )
    assert config.confidence_threshold == (0.7 if classifier is not None else 0.5)


@pytest.mark.parametrize("classifier", [None, {}])
def test_explicit_confidence_is_preserved(classifier: dict[str, object] | None) -> None:
    config: Final = SwitchyardRouterConfig.model_validate(
        {
            "models": {
                "capable": "strong",
                "efficient": "fast",
                **({"judge": "judge"} if classifier is not None else {}),
            },
            "classifier": classifier,
            "options": {"confidence_threshold": 0.63, "picker": "capable_first"},
        }
    )
    assert config.confidence_threshold == 0.63


@pytest.mark.parametrize(
    "option", ["escalation_note", "deescalation_note", "capable_system_prompt", "efficient_system_prompt", "unknown"]
)
def test_request_rewrite_and_unknown_options_are_rejected(option: str) -> None:
    with pytest.raises(ValidationError, match=option):
        SwitchyardRouterConfig.model_validate(
            {"models": {"capable": "strong", "efficient": "fast"}, "options": {option: "ignored"}}
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"algorithm": "random"},
        {"classifier": {}},
        {"options": {"confidence_threshold": 2}},
        {"models": {"capable": "", "efficient": "fast"}},
        {"models": {"capable": "same", "efficient": "same"}},
        {"models": {"capable": "strong", "efficient": "fast", "judge": "j"}, "classifier": {"session_affinity": True}},
        {"models": {"capable": "strong", "efficient": "fast", "judge": "judge"}},
        {"models": {"capable": "strong", "efficient": "fast", "judge": "j"}, "classifier": {"kind": "other"}},
        {
            "models": {"capable": "strong", "efficient": "fast", "judge": "j"},
            "classifier": {"kind": "decision", "cutoff": 1.5},
        },
        {
            "models": {"capable": "strong", "efficient": "fast", "judge": "j"},
            "classifier": {"kind": "decision", "base_threshold": 0.5},
        },
        {
            "models": {"capable": "strong", "efficient": "fast", "judge": "j"},
            "classifier": {"kind": "llm", "cutoff": 0.5},
        },
    ],
)
def test_invalid_config_cannot_be_saved(bad: dict[str, object]) -> None:
    violation: Final = validate_switchyard_config_write({"models": {"capable": "strong", "efficient": "fast"}, **bad})
    assert violation is not None
    assert "Invalid switchyard_router_config" in violation


def test_missing_package_rejects_save_with_install_instructions() -> None:
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"switchyard_rust.libsy": None}):
        violation: Final = validate_switchyard_config_write({"models": {"capable": "strong", "efficient": "fast"}})
    assert violation is not None
    assert "litellm[switchyard]" in violation


@pytest.mark.parametrize(
    "classifier", [{"kind": "decision"}, {"kind": "decision", "cutoff": 0.4, "evidence": {"a": 1}}]
)
def test_decision_classifier_config_can_be_saved(classifier: dict[str, object]) -> None:
    assert (
        validate_switchyard_config_write(
            {"models": {"capable": "strong", "efficient": "fast", "judge": "jev"}, "classifier": classifier}
        )
        is None
    )
