from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import jsonschema
import pytest

REPO_ROOT = Path(__file__).parents[2]
GENERATOR_PATH = REPO_ROOT / "ci_cd" / "generate_model_prices_schema.py"
PRICES_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PRICES_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"
SCHEMA_PATH = REPO_ROOT / "model_prices_and_context_window.schema.json"


def build_validator(schema: dict) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER)


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_model_prices_schema", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def committed_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def prices() -> dict:
    return json.loads(PRICES_PATH.read_text())


def test_committed_schema_matches_generator_output(prices: dict, committed_schema: dict):
    generator = load_generator()
    regenerated = json.loads(generator.render(generator.build_schema(prices)))
    assert regenerated == committed_schema, (
        "model_prices_and_context_window.schema.json is out of sync; "
        "run `python ci_cd/generate_model_prices_schema.py` and commit the result"
    )


def test_prices_file_validates_against_committed_schema(prices: dict, committed_schema: dict):
    validator = build_validator(committed_schema)
    errors = [
        f"{'.'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in validator.iter_errors(prices)
    ]
    assert errors == []


@pytest.mark.parametrize(
    "entry",
    [
        {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": "0.01"},
        {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": -1},
        {"litellm_provider": "openai", "mode": "not_a_real_mode"},
        {"mode": "chat"},
        {"litellm_provider": "openai", "deprecation_date": "June 2026"},
        {"litellm_provider": "openai", "deprecation_date": "2026-99-99"},
        {"litellm_provider": "openai", "deprecation_date": "2026-13-01"},
        {"litellm_provider": "openai", "deprecation_date": "2026-01-32"},
        {"litellm_provider": "openai", "deprecation_date": "2026-01-00"},
        {"litellm_provider": "openai", "deprecation_date": "2026-02-31"},
        {"litellm_provider": "openai", "supported_modalities": ["smell"]},
        {"litellm_provider": "openai", "supports_vision": "yes"},
        {"litellm_provider": "openai", "max_tokens": 8191.5},
        {"litellm_provider": "openai", "tiered_pricing": [{"unknown_tier_field": 1}]},
    ],
    ids=[
        "cost_as_string",
        "negative_cost",
        "unknown_mode",
        "missing_provider",
        "non_iso_deprecation_date",
        "impossible_month_and_day",
        "month_out_of_range",
        "day_out_of_range",
        "day_zero",
        "calendar_impossible_day",
        "unknown_modality",
        "boolean_flag_as_string",
        "fractional_max_tokens",
        "unknown_tiered_pricing_field",
    ],
)
def test_schema_rejects_malformed_entries(committed_schema: dict, entry: dict):
    validator = build_validator(committed_schema)
    assert not validator.is_valid({"some-model": entry})


def test_schema_accepts_minimal_and_unknown_optional_fields(committed_schema: dict):
    validator = build_validator(committed_schema)
    assert validator.is_valid({"some-model": {"litellm_provider": "openai"}})
    assert validator.is_valid({"some-model": {"litellm_provider": "openai", "brand_new_field": {"nested": True}}})


def test_schema_accepts_cache_creation_cost_inside_a_pricing_tier(committed_schema: dict):
    validator = build_validator(committed_schema)
    entry = {
        "litellm_provider": "dashscope",
        "mode": "chat",
        "tiered_pricing": [
            {
                "range": [0, 256000],
                "input_cost_per_token": 3.25e-07,
                "output_cost_per_token": 1.95e-06,
                "cache_creation_input_token_cost": 4.063e-07,
                "cache_read_input_token_cost": 3.25e-08,
            }
        ],
    }
    assert validator.is_valid({"some-model": entry})


def find_duplicate_keys(path: Path) -> list[str]:
    duplicates: list[str] = []

    def record_duplicates(pairs):
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        return dict(pairs)

    json.loads(path.read_text(), object_pairs_hook=record_duplicates)
    return duplicates


@pytest.mark.parametrize("path", (PRICES_PATH, BACKUP_PRICES_PATH), ids=("main", "backup"))
def test_price_map_has_no_duplicate_keys(path: Path):
    assert find_duplicate_keys(path) == [], (
        f"{path.name} defines the same key twice; JSON parsers keep only the last "
        "occurrence, so the earlier entry's fields are silently dropped. This is what "
        "a clean text merge of two branches that both added a model looks like: "
        "deduplicate the keys into one entry"
    )


DATED_VARIANT = re.compile(r"^(.*?)-(\d{4}-\d{2}-\d{2})$")
SERVICE_TIER_SUFFIXES = ("_flex", "_priority")


def tier_anchor(tier_key: str) -> str:
    matched = next(suffix for suffix in SERVICE_TIER_SUFFIXES if tier_key.endswith(suffix))
    return tier_key[: -len(matched)]


def test_dated_variants_carry_base_alias_service_tier_pricing(prices: dict):
    drifted = [
        f"{name}: missing {tier_key}={base[tier_key]} (base alias {match.group(1)})"
        for name, entry in prices.items()
        if isinstance(entry, dict)
        for match in [DATED_VARIANT.match(name)]
        if match is not None
        for base in [prices.get(match.group(1))]
        if isinstance(base, dict)
        for tier_key in base
        if tier_key.endswith(SERVICE_TIER_SUFFIXES)
        and tier_anchor(tier_key) in base
        and entry.get(tier_anchor(tier_key)) == base[tier_anchor(tier_key)]
        and entry.get(tier_key) != base[tier_key]
    ]
    assert drifted == [], (
        "dated model variants are missing flex/priority pricing their base alias has; "
        "sync the tier keys so service-tier requests against pinned snapshots are not "
        "billed at standard rates:\n" + "\n".join(drifted)
    )
