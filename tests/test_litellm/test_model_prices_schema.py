from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jsonschema
import pytest

REPO_ROOT = Path(__file__).parents[2]
GENERATOR_PATH = REPO_ROOT / "ci_cd" / "generate_model_prices_schema.py"
PRICES_PATH = REPO_ROOT / "model_prices_and_context_window.json"
SCHEMA_PATH = REPO_ROOT / "model_prices_and_context_window.schema.json"


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
    validator = jsonschema.Draft202012Validator(committed_schema)
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
        "unknown_modality",
        "boolean_flag_as_string",
        "fractional_max_tokens",
        "unknown_tiered_pricing_field",
    ],
)
def test_schema_rejects_malformed_entries(committed_schema: dict, entry: dict):
    validator = jsonschema.Draft202012Validator(committed_schema)
    assert not validator.is_valid({"some-model": entry})


def test_schema_accepts_minimal_and_unknown_optional_fields(committed_schema: dict):
    validator = jsonschema.Draft202012Validator(committed_schema)
    assert validator.is_valid({"some-model": {"litellm_provider": "openai"}})
    assert validator.is_valid({"some-model": {"litellm_provider": "openai", "brand_new_field": {"nested": True}}})
