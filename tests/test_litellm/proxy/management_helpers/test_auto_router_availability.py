from collections.abc import Mapping
from typing import Final
from types import SimpleNamespace

from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

import pytest

from litellm.proxy.management_helpers.auto_router_availability import (
    auto_router_availability,
    build_auto_router_catalog,
)
from litellm.router_utils.auto_router_tuning_baseline import snapshot_tuning_baselines


def deployment(
    model_id: str,
    classifier: str,
    *,
    model: str = "solver",
    tuned: bool = False,
    config: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    return {
        "model_name": model_id,
        "model_info": {"id": model_id, "db_model": True},
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {
                "classifier_type": classifier,
                "tiers": {"SIMPLE": [model]},
                **({"code_keywords": ["internal-api"]} if tuned else {}),
                **(config or {}),
            },
        },
    }


@pytest.mark.parametrize("classifier", ("heuristic_v2", "capability", "llm_v2"))
def test_occupied_allowance_blocks_new_router_but_not_owner(classifier: str) -> None:
    existing: Final = deployment("existing", classifier)
    candidate: Final = deployment("new", classifier)
    new: Final = auto_router_availability(others=(existing,), existing=None, candidate=candidate, baselines={}, limit=1)
    edit: Final = auto_router_availability(others=(), existing=existing, candidate=existing, baselines={}, limit=1)
    new_slot: Final = next(slot for slot in new.allowances if slot.key == classifier)
    edit_slot: Final = next(slot for slot in edit.allowances if slot.key == classifier)
    assert (new_slot.remaining, new_slot.used_by_this_router, new.error is not None) == (0, False, True)
    assert (edit_slot.remaining, edit_slot.used_by_this_router, edit.error) == (1, True, None)


def test_edit_does_not_exempt_another_classifier_allowance() -> None:
    existing: Final = deployment("existing", "capability")
    result: Final = auto_router_availability(
        others=(deployment("other", "llm_v2"),),
        existing=existing,
        candidate=deployment("existing", "llm_v2"),
        baselines={},
        limit=1,
    )
    assert result.error is not None
    assert next(slot for slot in result.allowances if slot.key == "llm_v2").remaining == 0


def test_model_selection_does_not_claim_occupied_scoring_allowance() -> None:
    original: Final = deployment("legacy", "heuristic")
    changed: Final = deployment("other", "heuristic", tuned=True)
    baselines: Final = snapshot_tuning_baselines((original,))
    unchanged: Final = auto_router_availability(
        others=(changed,),
        existing=original,
        candidate=original,
        baselines=baselines,
        limit=1,
    )
    edited: Final = auto_router_availability(
        others=(changed,),
        existing=original,
        candidate=deployment("legacy", "heuristic", model="new"),
        baselines=baselines,
        limit=1,
    )
    assert unchanged.error is None
    assert next(slot for slot in unchanged.allowances if slot.key == "heuristic_tuning").remaining == 0
    assert edited.error is None
    tuned: Final = auto_router_availability(
        others=(changed,),
        existing=original,
        candidate=deployment("legacy", "heuristic", tuned=True),
        baselines=baselines,
        limit=1,
    )
    assert tuned.error is not None
    assert "weights, thresholds, keywords, and custom dimensions" in tuned.error


def test_missing_baselines_are_reported_as_unknown() -> None:
    result: Final = auto_router_availability(
        others=(),
        existing=None,
        candidate=deployment("new", "heuristic"),
        baselines=None,
        limit=1,
    )
    slot: Final = next(slot for slot in result.allowances if slot.key == "heuristic_tuning")
    assert (slot.available, slot.remaining, slot.limit) == (False, None, 1)


def test_unlimited_entitlement_does_not_report_exhausted_allowances() -> None:
    result: Final = auto_router_availability(
        others=(deployment("other", "heuristic_v2"),),
        existing=None,
        candidate=deployment("new", "heuristic_v2"),
        baselines=None,
        limit=None,
    )
    assert all(slot.available and slot.limit is None and slot.remaining is None for slot in result.allowances)
    assert result.error is None


@pytest.mark.parametrize(
    "customization",
    (
        {"tier_definitions": [{"name": "SIMPLE"}, {"name": "AUDIT", "description": "Review risks"}]},
        {"classification_prompt": "Use the simplest sufficient tier"},
        {"classification_examples": "Review this code -> COMPLEX"},
        {"classifier_llm_config": {"model": "judge", "system_prompt": "Route by urgency"}},
    ),
)
def test_customization_owner_can_edit_models_and_restoring_defaults_clears_the_gate(
    customization: Mapping[str, object],
) -> None:
    owner: Final = deployment("owner", "llm", config=customization)
    blocked: Final = auto_router_availability(
        others=(owner,), existing=None, candidate=deployment("new", "llm", config=customization), baselines={}, limit=1
    )
    assert blocked.error is not None
    assert "Custom tiers or classifier instructions" in blocked.error
    edited: Final = auto_router_availability(
        others=(),
        existing=owner,
        candidate=deployment("owner", "llm", model="new", config=customization),
        baselines={},
        limit=1,
    )
    assert edited.error is None
    assert next(slot for slot in edited.allowances if slot.key == "tier_or_classifier_prompt").used_by_this_router
    restored: Final = auto_router_availability(
        others=(owner,), existing=None, candidate=deployment("new", "llm"), baselines={}, limit=1
    )
    assert restored.error is None
    assert next(slot for slot in restored.allowances if slot.key == "tier_or_classifier_prompt").remaining == 0


def test_restoring_tiers_does_not_exempt_a_retained_custom_prompt() -> None:
    prompt: Final = {"classification_prompt": "Use the simplest sufficient tier"}
    result: Final = auto_router_availability(
        others=(deployment("owner", "llm", config=prompt),),
        existing=None,
        candidate=deployment("new", "llm", config=prompt),
        baselines={},
        limit=1,
    )
    assert result.error is not None
    assert "Custom tiers or classifier instructions" in result.error


@pytest.mark.parametrize("blocked", (False, True))
def test_catalog_keeps_unloaded_routers_and_ownership_without_provider_credentials(blocked: bool, monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "catalog-test-key")
    source: Final = SimpleNamespace(
        model_id="saved",
        created_by="owner",
        model_info={"team_id": "team"},
        blocked=blocked,
        litellm_params={
            "model": encrypt_value_helper("auto_router/complexity_router"),
            "api_key": "private-key",
            "complexity_router_config": {"classifier_type": "heuristic_v2"},
        },
    )
    provider: Final = SimpleNamespace(model_id="provider", litellm_params={"model": "openai/model"})
    catalog: Final = build_auto_router_catalog((source, provider))
    assert catalog is not None and len(catalog) == 1
    assert (catalog[0].model_id, catalog[0].team_id, catalog[0].created_by) == ("saved", "team", "owner")
    assert catalog[0].deployment == {
        "model_info": {"id": "saved", "db_model": True},
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {"classifier_type": "heuristic_v2"},
        },
    }


def test_catalog_distinguishes_missing_data_from_an_empty_model_table() -> None:
    assert build_auto_router_catalog(()) == ()
    assert build_auto_router_catalog((SimpleNamespace(model_id="incomplete"),)) is None
