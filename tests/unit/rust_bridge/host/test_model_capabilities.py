import datetime
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.fallback_generalizations import get_fallback_generalization_rules
from litellm.litellm_core_utils.model_cost_table import changed_in_place
from litellm.rust_bridge.host.model_capabilities import anthropic_model_capabilities, model_cost_document

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


def test_model_cost_document_carries_the_live_rows_and_the_active_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    _flag_model(monkeypatch, "claude-test-document", supports_tool_search=True)
    changed_in_place()

    version, document = model_cost_document(None)

    assert document is not None
    assert document["claude-test-document"]["supports_tool_search"] is True
    assert document["fallback_generalizations"] == {"rules": list(get_fallback_generalization_rules())}
    assert model_cost_document(version) == (version, None)


def test_model_cost_document_carries_registered_rows_with_non_json_values(monkeypatch: pytest.MonkeyPatch) -> None:
    registered_at: Final = datetime.datetime(2026, 11, 15, tzinfo=datetime.timezone.utc)
    monkeypatch.setitem(
        litellm.model_cost,
        "bedrock/invoke/claude-test-registered",
        {"litellm_provider": "bedrock", "mode": "chat", "supports_tool_search": True, "created_at": registered_at},
    )
    changed_in_place()

    _, document = model_cost_document(None)

    assert document is not None
    assert document["bedrock/invoke/claude-test-registered"] == {
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supports_tool_search": True,
        "created_at": str(registered_at),
    }


def test_model_cost_document_moves_with_a_write(monkeypatch: pytest.MonkeyPatch) -> None:
    version, _ = model_cost_document(None)

    _flag_model(monkeypatch, "claude-test-document-write", supports_tool_search=True)
    changed_in_place()
    moved, document = model_cost_document(version)

    assert moved != version
    assert document is not None
    assert "claude-test-document-write" in document
