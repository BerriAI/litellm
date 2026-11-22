from typing import Final

import pytest

import litellm
from litellm.rust_bridge.host.model_capabilities import anthropic_model_capabilities

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


def _flag_model(monkeypatch: pytest.MonkeyPatch, name: str, **flags: bool) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        name,
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
            **flags,
        },
    )


def test_capabilities_come_from_the_model_map_under_the_callers_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _flag_model(
        monkeypatch,
        "claude-test-adaptive",
        supports_reasoning=True,
        supports_adaptive_thinking=True,
        supports_output_config=True,
        supports_xhigh_reasoning_effort=True,
        supports_sampling_params=False,
        supports_mid_conversation_system=True,
        supports_anthropic_user_profile_id=True,
    )

    capabilities: Final = anthropic_model_capabilities("anthropic/claude-test-adaptive", None)

    assert capabilities["supports_adaptive_thinking"]
    assert capabilities["supports_output_config"]
    assert capabilities["supports_mid_conversation_system"]
    assert capabilities["supports_anthropic_user_profile_id"]
    assert not capabilities["supports_legacy_thinking"]
    assert not capabilities["supports_sampling_params"]
    assert capabilities["effort_tiers"] == {
        "minimal": False,
        "low": False,
        "medium": False,
        "high": False,
        "xhigh": True,
        "max": False,
    }


@pytest.mark.parametrize(
    ("base_model", "takes_profile_id"),
    [("anthropic/claude-test-profile", True), (None, False)],
    ids=["base_model_flagged", "no_base_model"],
)
def test_an_unlisted_model_takes_the_user_profile_id_its_base_model_is_flagged_for(
    monkeypatch: pytest.MonkeyPatch, base_model: str | None, takes_profile_id: bool
) -> None:
    _flag_model(monkeypatch, "claude-test-profile", supports_anthropic_user_profile_id=True)

    capabilities: Final = anthropic_model_capabilities("anthropic/claude-test-unlisted", None, base_model)

    assert capabilities["supports_anthropic_user_profile_id"] is takes_profile_id


def test_unmapped_model_keeps_sampling_params_and_no_reasoning_features() -> None:
    capabilities: Final = anthropic_model_capabilities("anthropic/not-a-real-model", None)

    assert capabilities["supports_sampling_params"]
    assert not capabilities["supports_reasoning"]
    assert not capabilities["supports_adaptive_thinking"]
    assert not capabilities["supports_mid_conversation_system"]
    assert not capabilities["supports_anthropic_user_profile_id"]
    assert capabilities["effort_tiers"] == dict.fromkeys(("minimal", "low", "medium", "high", "xhigh", "max"), False)


def test_capability_source_observes_runtime_registration_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    model: Final = "claude-test-runtime-registration"
    _flag_model(monkeypatch, model, supports_reasoning=True, supports_output_config=True)
    before: Final = anthropic_model_capabilities(model, "anthropic")

    _flag_model(monkeypatch, model, supports_reasoning=False, supports_output_config=False)
    after: Final = anthropic_model_capabilities(model, "anthropic")

    assert before["supports_reasoning"] is True
    assert before["supports_output_config"] is True
    assert after["supports_reasoning"] is False
    assert after["supports_output_config"] is False
