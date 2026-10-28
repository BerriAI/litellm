from typing import Final

import pytest

import litellm
from litellm._catalog import CATALOG
from litellm._catalog.provider_sets import models_for


@pytest.fixture
def live_map(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, object]]:
    rows: Final = dict(CATALOG.bundled().served())
    monkeypatch.setattr(litellm, "model_cost", rows)
    return rows


def test_a_row_added_dropped_or_moved_shows_on_the_next_read(live_map: dict[str, dict[str, object]]) -> None:
    """provider-sets-live. On main a moved key stayed in its old set and a direct write reached none"""
    live_map["acme/new-chat"] = {"litellm_provider": "cohere", "mode": "chat"}
    assert "acme/new-chat" in litellm.cohere_models
    assert "acme/new-chat" in litellm.models_by_provider["cohere"]

    live_map["acme/new-chat"] = {"litellm_provider": "mistral", "mode": "chat"}
    live_map["acme/len-change"] = {}
    assert "acme/new-chat" not in litellm.cohere_models
    assert "acme/new-chat" in litellm.mistral_chat_models

    del live_map["acme/new-chat"]
    assert "acme/new-chat" not in litellm.models_by_provider["mistral"]


def test_the_provider_sets_are_read_only() -> None:
    """provider-sets-readonly"""
    with pytest.raises(AttributeError):
        litellm.anthropic_models.add("x")  # pyright: ignore[reportAttributeAccessIssue]  # the mutation under test
    with pytest.raises(TypeError):
        litellm.models_by_provider["openai"] = frozenset()  # pyright: ignore[reportIndexIssue]  # the mutation under test


@pytest.mark.usefixtures("live_map")
def test_every_provider_in_the_map_lists_the_rows_that_name_it() -> None:
    providers: Final = frozenset(
        row["litellm_provider"] for row in litellm.model_cost.values() if isinstance(row.get("litellm_provider"), str)
    )
    for provider in providers:
        assert provider in litellm.models_by_provider
    assert "together_ai" in litellm.models_by_provider
    assert all(litellm.model_cost[key]["litellm_provider"] == "aihubmix" for key in models_for("aihubmix"))


def test_registering_an_openrouter_model_adds_no_stripped_name(live_map: dict[str, dict[str, object]]) -> None:
    litellm.register_model({"openrouter/acme/model": {"litellm_provider": "openrouter", "mode": "chat"}})

    assert "openrouter/acme/model" in litellm.openrouter_models
    assert "acme/model" not in litellm.openrouter_models


@pytest.mark.usefixtures("live_map")
def test_the_request_path_membership_check_agrees_with_every_derived_set() -> None:
    from litellm._catalog.provider_sets import LEGACY_NAMES, contains, legacy_set

    names: Final = frozenset(litellm.model_cost) | {key.replace("vertex_ai/", "") for key in litellm.model_cost}
    disagreements: Final = [
        (set_name, model)
        for set_name in LEGACY_NAMES
        for model in names  # comprehension-ok: every set against every candidate name
        if contains(set_name, model) != (model in legacy_set(set_name))
    ]

    assert disagreements == []
