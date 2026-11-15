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
    )

    capabilities: Final = anthropic_model_capabilities("anthropic/claude-test-adaptive", None)

    assert capabilities["supports_adaptive_thinking"]
    assert capabilities["supports_output_config"]
    assert capabilities["supports_mid_conversation_system"]
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


def test_unmapped_model_keeps_sampling_params_and_no_reasoning_features() -> None:
    capabilities: Final = anthropic_model_capabilities("anthropic/not-a-real-model", None)

    assert capabilities["supports_sampling_params"]
    assert not capabilities["supports_reasoning"]
    assert not capabilities["supports_adaptive_thinking"]
    assert not capabilities["supports_mid_conversation_system"]
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


@pytest.mark.parametrize(
    "model, expected",
    [
        ("claude-sonnet-5", True),
        ("claude-haiku-4-5", False),
        ("claude-sonnet-5-5", True),
    ],
)
def test_default_thinking_capability_is_projected(model: str, expected: bool) -> None:
    capabilities: Final = anthropic_model_capabilities(model, "anthropic")
    assert capabilities["thinking_on_by_default"] is expected


def test_capabilities_are_probed_once_per_model_map_state(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    model: Final = "claude-test-probe-count"
    _flag_model(monkeypatch, model, supports_reasoning=True)
    probes: list[str] = []
    probe: Final = AnthropicModelInfo._supports_model_capability

    def counted(name: str, key: str, provider: str) -> bool:
        probes.append(key)
        return probe(name, key, provider)

    monkeypatch.setattr(AnthropicModelInfo, "_supports_model_capability", staticmethod(counted))

    first: Final = anthropic_model_capabilities(model, "anthropic")
    probed: Final = len(probes)
    second: Final = anthropic_model_capabilities(model, "anthropic")

    assert probed > 0
    assert len(probes) == probed
    assert second == first


def test_a_cached_answer_is_a_fresh_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    model: Final = "claude-test-fresh-copy"
    _flag_model(monkeypatch, model, supports_xhigh_reasoning_effort=True)
    first: Final = anthropic_model_capabilities(model, "anthropic")
    tiers: Final = first["effort_tiers"]
    assert isinstance(tiers, dict)
    first["supports_reasoning"] = "changed"
    tiers["xhigh"] = "changed"

    second: Final = anthropic_model_capabilities(model, "anthropic")

    assert second["supports_reasoning"] is False
    assert second["effort_tiers"] == {
        **dict.fromkeys(("minimal", "low", "medium", "high", "max"), False),
        "xhigh": True,
    }


def test_register_model_is_observed_after_a_cached_answer() -> None:
    model: Final = "claude-test-registered-later"
    try:
        litellm.register_model({model: {"litellm_provider": "anthropic", "mode": "chat", "supports_reasoning": False}})
        before: Final = anthropic_model_capabilities(model, "anthropic")
        litellm.register_model({model: {"litellm_provider": "anthropic", "mode": "chat", "supports_reasoning": True}})
        after: Final = anthropic_model_capabilities(model, "anthropic")
    finally:
        litellm.model_cost.pop(model, None)

    assert before["supports_reasoning"] is False
    assert after["supports_reasoning"] is True


def test_the_cache_key_holds_the_entries_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.rust_bridge.host.model_capabilities import _fingerprint  # pyright: ignore[reportPrivateUsage]

    model: Final = "claude-test-held-entry"
    _flag_model(monkeypatch, model, supports_reasoning=True)
    entry: Final = litellm.model_cost[model]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the model map is an untyped dict of rows
    held: Final = _fingerprint(model, "anthropic", model, "anthropic")
    _flag_model(monkeypatch, model, supports_reasoning=True)
    same_content: Final = _fingerprint(model, "anthropic", model, "anthropic")

    assert any(h.value is entry for h in held[5])
    assert held != same_content
