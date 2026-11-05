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
    assert config.default_model == "fast"


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
    assert config.default_model == "strong"


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
        {"models": {"capable": "strong", "efficient": "fast", "judge": "judge"}},
    ],
)
def test_invalid_config_cannot_be_saved(bad: dict[str, object]) -> None:
    violation: Final = validate_switchyard_config_write({"models": {"capable": "strong", "efficient": "fast"}, **bad})
    assert violation is not None
    assert "Invalid switchyard_router_config" in violation


def test_missing_package_rejects_save_with_install_instructions() -> None:
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"switchyard.libsy": None}):
        violation: Final = validate_switchyard_config_write({"models": {"capable": "strong", "efficient": "fast"}})
    assert violation is not None
    assert "litellm[switchyard]" in violation


def test_native_validation_rejects_invalid_session_affinity() -> None:
    violation: Final = validate_switchyard_config_write(
        {
            "models": {"capable": "strong", "efficient": "fast", "judge": "judge"},
            "classifier": {"session_affinity": False, "message_hash_fallback": True},
        }
    )
    assert violation is not None
    assert "Invalid switchyard_router_config" in violation
