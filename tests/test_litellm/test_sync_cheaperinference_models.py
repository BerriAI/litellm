"""Tests for scripts/sync_cheaperinference_models.py.

``fixtures/cheaperinference_sync/models.json`` is a recorded ``GET /v1/models``
response, trimmed to the fields the sync reads, with the gateway's own
``pricing_version`` and ``pricing_checked_at`` kept for provenance.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from litellm.litellm_core_utils.llm_cost_calc.utils import _parse_above_token_threshold

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
    assert "seedance-2.0" in ids


def test_recorded_catalog_carries_tiered_pricing_for_the_four_threshold_models() -> None:
    tiered = {model.id for model in RECORDED_CATALOG if model.pricing.above_threshold is not None}
    assert tiered == {"gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}


def test_registry_already_matches_the_recorded_catalog() -> None:
    """A failure here means either the registry drifted from the last live sync or the sync logic broke."""
    outcome = sync.compute_sync(REGISTERED_CHEAPERINFERENCE, RECORDED_CATALOG)
    assert outcome.updated == ()
    assert outcome.missing_from_catalog == ()


def test_a_perturbed_price_is_detected_and_corrected() -> None:
    perturbed = json.loads(json.dumps(REGISTERED_CHEAPERINFERENCE))
    perturbed["cheaperinference/kimi-k3"]["input_cost_per_token"] = 999.0
    outcome = sync.compute_sync(perturbed, RECORDED_CATALOG)
    assert outcome.updated == ("cheaperinference/kimi-k3: input_cost_per_token: 999.0 -> 2.1e-06",)
    assert outcome.cost_map["cheaperinference/kimi-k3"]["input_cost_per_token"] == 2.1e-06
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


def test_a_model_losing_its_pricing_tier_drops_its_above_threshold_fields() -> None:
    with_stale_tier = json.loads(json.dumps(REGISTERED_CHEAPERINFERENCE))
    with_stale_tier["cheaperinference/kimi-k3"]["input_cost_per_token_above_272k_tokens"] = 5.0
    outcome = sync.compute_sync(with_stale_tier, RECORDED_CATALOG)
    assert "input_cost_per_token_above_272k_tokens" not in outcome.cost_map["cheaperinference/kimi-k3"]


def test_a_null_limit_in_the_catalog_never_overwrites_the_registry() -> None:
    """gpt-oss-120b reports a null max_output_tokens, so no output limit may appear."""
    registered = json.loads(json.dumps(REGISTERED_CHEAPERINFERENCE))
    assert "max_output_tokens" not in registered["cheaperinference/gpt-oss-120b"]

    outcome = sync.compute_sync(registered, RECORDED_CATALOG)

    assert "max_output_tokens" not in outcome.cost_map["cheaperinference/gpt-oss-120b"]
    assert "max_tokens" not in outcome.cost_map["cheaperinference/gpt-oss-120b"]


@pytest.mark.parametrize(
    "threshold,band",
    [
        (272_000, "above_272k_tokens"),
        (200_000, "above_200k_tokens"),
        (128_000, "above_128k_tokens"),
        (512_000, "above_512k_tokens"),
        (271_999, "above_271999_tokens"),
        (300_000, "above_300000_tokens"),
    ],
)
def test_a_catalog_threshold_names_its_own_band(threshold: int, band: str) -> None:
    assert sync.band_for_threshold(threshold) == band


@pytest.mark.parametrize("threshold", [100_000, 128_000, 200_000, 271_999, 272_000, 300_000, 512_000, 999_999])
def test_litellm_reads_back_the_exact_threshold_from_the_generated_field(threshold: int) -> None:
    field = f"input_cost_per_token_{sync.band_for_threshold(threshold)}"

    assert _parse_above_token_threshold(field) == threshold


def _catalog_with_threshold(model_id: str, threshold: int) -> list:
    """The recorded catalog with one model's long-context threshold moved."""
    raw = json.loads(FIXTURES.joinpath("models.json").read_bytes())
    entries = raw["data"] if isinstance(raw, dict) else raw
    for entry in entries:
        above = entry["pricing"].get("above_threshold")
        if entry["id"] == model_id and above is not None:
            above["input_token_price_threshold"] = threshold
    return sync.load_catalog(json.dumps(entries).encode())


def test_a_threshold_outside_the_k_form_bands_is_written_at_its_exact_value() -> None:
    catalog = _catalog_with_threshold("gpt-5.6-luna", 300_000)

    outcome = sync.compute_sync(REGISTERED_CHEAPERINFERENCE, catalog)

    entry = outcome.cost_map["cheaperinference/gpt-5.6-luna"]
    assert entry["input_cost_per_token_above_300000_tokens"] == pytest.approx(0.16e-06)
    assert "input_cost_per_token_above_271999_tokens" not in entry


def test_a_moved_threshold_rewrites_the_band_and_drops_the_old_one() -> None:
    catalog = _catalog_with_threshold("gpt-5.6-luna", 200_000)

    outcome = sync.compute_sync(REGISTERED_CHEAPERINFERENCE, catalog)

    entry = outcome.cost_map["cheaperinference/gpt-5.6-luna"]
    assert entry["input_cost_per_token_above_200k_tokens"] == pytest.approx(0.16e-06)
    assert "input_cost_per_token_above_271999_tokens" not in entry


def test_warnings_name_every_kind_of_catalog_drift() -> None:
    outcome = sync.SyncOutcome(
        cost_map={},
        unpriced_new_models=("cheaperinference/brand-new",),
        missing_from_catalog=("cheaperinference/retired",),
    )

    assert outcome.warnings == (
        "new in the catalog, not registered: cheaperinference/brand-new",
        "registered but gone from the catalog: cheaperinference/retired",
    )


def test_the_warnings_block_is_written_even_when_no_prices_changed(tmp_path: Path) -> None:
    outcome = sync.SyncOutcome(cost_map={}, unpriced_new_models=("cheaperinference/brand-new",))

    assert not outcome.has_changes
    assert "cheaperinference/brand-new" in sync.render_warnings_block(outcome)


def test_a_quiet_run_says_so_instead_of_writing_an_empty_list() -> None:
    assert "No catalog drift" in sync.render_warnings_block(sync.SyncOutcome(cost_map={}))


def test_the_step_summary_is_the_default_destination_in_actions(tmp_path: Path, monkeypatch) -> None:
    summary = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert sync._github_step_summary() == summary


def test_no_summary_destination_outside_actions(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    assert sync._github_step_summary() is None
