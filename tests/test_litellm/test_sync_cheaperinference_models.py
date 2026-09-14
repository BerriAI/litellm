import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "sync_cheaperinference_models.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cheaperinference_sync"

_spec = importlib.util.spec_from_file_location("sync_cheaperinference_models", SCRIPT)
assert _spec is not None and _spec.loader is not None
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)

RECORDED_CATALOG = sync.load_catalog(FIXTURES.joinpath("models.json").read_bytes())
COST_MAP = json.loads((ROOT / "model_prices_and_context_window.json").read_text())
REGISTERED_CHEAPERINFERENCE = {k: v for k, v in COST_MAP.items() if k.startswith(sync.PREFIX)}


@pytest.mark.parametrize(
    ("per_million", "expected"),
    [
        ("3.500000", 3.5e-06),
        ("0.239969", 2.39969e-07),
        ("0.012750", 1.275e-08),
        ("14.000000", 1.4e-05),
        ("0.000000", 0.0),
    ],
)
def test_per_token_matches_registry_precision(per_million: str, expected: float) -> None:
    assert sync.per_token(per_million) == expected


def test_load_catalog_raises_on_shape_change() -> None:
    with pytest.raises(sync.SyncError):
        sync.load_catalog(b'[{"id": "x", "type": "text"}]')


def test_recorded_catalog_has_expected_models() -> None:
    ids = {model.id for model in RECORDED_CATALOG}
    assert "kimi-k3" in ids
    assert "gpt-6-astra" in ids
    assert "seedance-2.0" in ids  # not registered: proves unregistered models don't crash the sync


def test_recorded_catalog_carries_tiered_pricing_for_the_four_threshold_models() -> None:
    tiered = {model.id for model in RECORDED_CATALOG if model.pricing.above_threshold is not None}
    assert tiered == {"gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}


def test_registry_already_matches_the_recorded_catalog() -> None:
    # If this fails, either the registry has drifted from the last live sync or the sync logic
    # itself broke: both are worth knowing about, not just "no crash"
    outcome = sync.compute_sync(REGISTERED_CHEAPERINFERENCE, RECORDED_CATALOG)
    assert outcome.updated == ()
    assert outcome.missing_from_catalog == ()


def test_a_perturbed_price_is_detected_and_corrected() -> None:
    perturbed = json.loads(json.dumps(REGISTERED_CHEAPERINFERENCE))
    perturbed["cheaperinference/kimi-k3"]["input_cost_per_token"] = 999.0
    outcome = sync.compute_sync(perturbed, RECORDED_CATALOG)
    assert outcome.updated == ("cheaperinference/kimi-k3: input_cost_per_token: 999.0 -> 2.1e-06",)
    assert outcome.cost_map["cheaperinference/kimi-k3"]["input_cost_per_token"] == 2.1e-06
    # untouched, curated fields survive the merge unchanged
    assert outcome.cost_map["cheaperinference/kimi-k3"]["supports_vision"] is True


def test_capability_flags_are_never_synced() -> None:
    tampered = json.loads(json.dumps(REGISTERED_CHEAPERINFERENCE))
    tampered["cheaperinference/kimi-k3"]["supports_vision"] = False
    tampered["cheaperinference/kimi-k3"]["mode"] = "embedding"
    outcome = sync.compute_sync(tampered, RECORDED_CATALOG)
    assert outcome.updated == ()
    assert outcome.cost_map["cheaperinference/kimi-k3"]["supports_vision"] is False
    assert outcome.cost_map["cheaperinference/kimi-k3"]["mode"] == "embedding"


def test_a_model_dropped_from_the_catalog_is_flagged_and_left_untouched() -> None:
    thin_catalog = [model for model in RECORDED_CATALOG if model.id != "kimi-k3"]
    outcome = sync.compute_sync(REGISTERED_CHEAPERINFERENCE, thin_catalog)
    assert outcome.missing_from_catalog == ("cheaperinference/kimi-k3",)
    assert outcome.cost_map["cheaperinference/kimi-k3"] == REGISTERED_CHEAPERINFERENCE["cheaperinference/kimi-k3"]


def test_a_new_catalog_model_is_flagged_but_never_auto_added() -> None:
    outcome = sync.compute_sync(REGISTERED_CHEAPERINFERENCE, RECORDED_CATALOG)
    assert "cheaperinference/seedance-2.0" in outcome.unpriced_new_models
    assert "cheaperinference/seedance-2.0" not in outcome.cost_map


def test_a_model_losing_its_pricing_tier_drops_the_above_272k_fields() -> None:
    with_stale_tier = json.loads(json.dumps(REGISTERED_CHEAPERINFERENCE))
    with_stale_tier["cheaperinference/kimi-k3"]["input_cost_per_token_above_272k_tokens"] = 5.0
    outcome = sync.compute_sync(with_stale_tier, RECORDED_CATALOG)
    assert "input_cost_per_token_above_272k_tokens" not in outcome.cost_map["cheaperinference/kimi-k3"]


def test_null_context_length_and_max_output_tokens_do_not_overwrite_curated_limits() -> None:
    # gpt-oss-120b reports null context_length/max_output_tokens in the live catalog; the curated
    # registry limits must survive rather than being wiped to None
    registered = json.loads(json.dumps(REGISTERED_CHEAPERINFERENCE))
    assert "cheaperinference/gpt-oss-120b" in registered
    curated_max_input = registered["cheaperinference/gpt-oss-120b"].get("max_input_tokens")
    outcome = sync.compute_sync(registered, RECORDED_CATALOG)
    assert outcome.cost_map["cheaperinference/gpt-oss-120b"].get("max_input_tokens") == curated_max_input
