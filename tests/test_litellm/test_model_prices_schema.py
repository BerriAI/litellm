from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import MappingProxyType
from typing import Final

import jsonschema
import pytest

from litellm.llms.openai.chat.gpt_5_transformation import is_gpt_reasoning_series_name
from litellm.router_utils.reasoning_effort_capability import resolve_supported_reasoning_efforts

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


OPENAI_REASONING_FAMILY_MARKERS = ("codex", "deep-research", "chat-latest")


def is_openai_o_series(name: str) -> bool:
    return len(name) > 1 and name[0] == "o" and name[1].isdigit()


def is_openai_reasoning_family(name: str) -> bool:
    base = name.split("/")[-1].removeprefix("ft:")
    if "search-api" in base:
        return False
    return (
        is_openai_o_series(base)
        or is_gpt_reasoning_series_name(base)
        or any(marker in base for marker in OPENAI_REASONING_FAMILY_MARKERS)
    )


def test_openai_reasoning_family_entries_carry_supports_reasoning(prices: dict):
    unflagged = [
        name
        for name, entry in prices.items()
        if isinstance(entry, dict)
        and entry.get("litellm_provider") == "openai"
        and is_openai_reasoning_family(name)
        and entry.get("supports_reasoning") is not True
    ]
    assert unflagged == [], (
        "OpenAI o-series, gpt-5+, codex, deep-research, and chat-latest models are reasoning "
        "models, and the Responses API drops the `reasoning` param for any mapped OpenAI model "
        "whose entry lacks supports_reasoning; flag these entries:\n" + "\n".join(unflagged)
    )


def test_chat_latest_declares_the_one_effort_openai_accepts(prices: dict):
    """OpenAI rejects every reasoning.effort on chat-latest except medium, and a reasoning entry
    with no declared levels resolves to None, which lets /model_group/info and the dashboard offer
    levels the upstream will 400 on."""
    assert resolve_supported_reasoning_efforts(prices["chat-latest"], deployment_is_mapped=True) == ("medium",)


@pytest.mark.parametrize("key", ["azure/gpt-chat-latest", "azure/chat-latest", "azure/us/gpt-chat-latest"])
def test_azure_gpt_chat_latest_declares_the_one_effort_azure_accepts(prices: dict, key: str):
    """Azure answers every reasoning_effort on a gpt-chat-latest deployment except medium with
    "Unsupported value ... Supported values are: 'medium'", the same fixed level OpenAI's chat-latest
    carries, so the Foundry product name and the OpenAI API name both declare that one level."""
    assert resolve_supported_reasoning_efforts(prices[key], deployment_is_mapped=True) == ("medium",)


BEDROCK_OPENAI_GPT_MARKERS: Final = ("openai.gpt-5.4", "openai.gpt-5.5", "openai.gpt-5.6", "openai.gpt-6-astra")
BEDROCK_PROVIDERS: Final = frozenset(("bedrock", "bedrock_converse", "bedrock_mantle"))
BEDROCK_ROW_PREFIXES: Final = ("bedrock_mantle/", "us.", "global.")
GPT_5_4_BEDROCK_LADDER: Final = ("none", "low", "medium", "high", "xhigh")
GPT_5_6_BEDROCK_LADDER: Final = ("none", "low", "medium", "high", "xhigh", "max")
GPT_6_ASTRA_BEDROCK_LADDER: Final = ("low", "medium", "high", "xhigh", "max")
BEDROCK_OPENAI_GPT_LADDERS: Final = MappingProxyType(
    {
        "bedrock_mantle/openai.gpt-5.4": GPT_5_4_BEDROCK_LADDER,
        "bedrock_mantle/openai.gpt-5.5": GPT_5_4_BEDROCK_LADDER,
        **{
            f"{prefix}openai.gpt-5.6-{variant}": GPT_5_6_BEDROCK_LADDER
            for prefix in BEDROCK_ROW_PREFIXES
            for variant in ("luna", "sol", "terra")
        },
        **{f"{prefix}openai.gpt-6-astra": GPT_6_ASTRA_BEDROCK_LADDER for prefix in BEDROCK_ROW_PREFIXES},
    }
)


@pytest.mark.parametrize(
    ("name", "ladder"), tuple(BEDROCK_OPENAI_GPT_LADDERS.items()), ids=tuple(BEDROCK_OPENAI_GPT_LADDERS)
)
def test_bedrock_openai_gpt_rows_advertise_the_ladder_bedrock_accepts(prices: dict, name: str, ladder: tuple[str, ...]):
    """Each ladder is the set of levels Bedrock answered 200 to for that row through the proxy on
    2026-09-11 (PR #40740): the Mantle rows go out over its Responses endpoint and the Converse rows
    over inference profiles. Bedrock differs from the direct OpenAI rows in two places, gpt-5.6 and
    gpt-6-astra take max there, and gpt-6-astra refuses none; minimal is refused on every row.
    xhigh and max are opt-in for the resolver, so a row missing either flag silently drops that
    level from every group it belongs to."""
    assert resolve_supported_reasoning_efforts(prices[name], deployment_is_mapped=True) == ladder


def test_every_bedrock_openai_gpt_row_advertises_xhigh(prices: dict):
    """The GovCloud and gpt-5.6-cyber rows cannot be called from our account, so they carry the
    family's xhigh flag rather than a measured ladder."""
    missing: Final = [
        name
        for name, entry in prices.items()
        if isinstance(entry, dict)
        and entry.get("litellm_provider") in BEDROCK_PROVIDERS
        and any(marker in name for marker in BEDROCK_OPENAI_GPT_MARKERS)
        and "xhigh" not in (resolve_supported_reasoning_efforts(entry, deployment_is_mapped=True) or ())
    ]
    assert missing == []


STANDARD_RATE_KEYS: Final = ("input_cost_per_token", "output_cost_per_token")
DISCOUNT_TIER_SUFFIXES: Final = ("_batch", "_flex")
REGIONAL_AZURE_PREFIXES: Final = ("azure/eu/", "azure/us/")
REGIONAL_AZURE_RATE_KEYS: Final = (*STANDARD_RATE_KEYS, "cache_read_input_token_cost")
REGIONAL_UPLIFT_CEILING: Final = 2.0


def rate(entry: dict, key: str) -> float | None:
    value: Final = entry.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def price_entries(prices: dict) -> list[tuple[str, dict]]:
    return [(name, entry) for name, entry in prices.items() if isinstance(entry, dict)]


def test_cache_read_never_costs_more_than_a_fresh_input_token(prices: dict):
    pricier: Final = [
        f"{name}: cache_read={cached} > input={fresh}"
        for name, entry in price_entries(prices)
        for cached in [rate(entry, "cache_read_input_token_cost")]
        for fresh in [rate(entry, "input_cost_per_token")]
        if cached is not None and fresh is not None and cached > fresh * (1 + 1e-9)
    ]
    assert pricier == []


def test_cache_write_costs_at_least_as_much_as_cache_read_unless_free(prices: dict):
    inverted: Final = [
        f"{name}: cache_write={write} < cache_read={read}"
        for name, entry in price_entries(prices)
        for write in [rate(entry, "cache_creation_input_token_cost")]
        for read in [rate(entry, "cache_read_input_token_cost")]
        if write is not None and read is not None and 0 < write < read
    ]
    assert inverted == []


def test_one_hour_cache_write_costs_at_least_the_five_minute_write(prices: dict):
    inverted: Final = [
        f"{name}: 1h={long} < 5m={short}"
        for name, entry in price_entries(prices)
        for long in [rate(entry, "cache_creation_input_token_cost_above_1hr")]
        for short in [rate(entry, "cache_creation_input_token_cost")]
        if long is not None and short is not None and long < short
    ]
    assert inverted == []


def test_batch_and_flex_tiers_never_cost_more_than_standard(prices: dict):
    pricier: Final = [
        f"{name}: {key}{suffix}={discounted} > {key}={standard}"
        for name, entry in price_entries(prices)
        for key in STANDARD_RATE_KEYS
        for suffix in DISCOUNT_TIER_SUFFIXES
        for discounted in [rate(entry, f"{key}{suffix}")]
        for standard in [rate(entry, key)]
        if discounted is not None and standard is not None and discounted > standard
    ]
    assert pricier == []


def test_priority_tier_never_costs_less_than_standard(prices: dict):
    cheaper: Final = [
        f"{name}: {key}_priority={priority} < {key}={standard}"
        for name, entry in price_entries(prices)
        for key in STANDARD_RATE_KEYS
        for priority in [rate(entry, f"{key}_priority")]
        for standard in [rate(entry, key)]
        if priority is not None and standard is not None and priority < standard
    ]
    assert cheaper == []


def long_context_anchor(key: str) -> str:
    base, _, remainder = key.partition("_above_")
    _, _, tier = remainder.partition("_tokens")
    return f"{base}{tier}"


def test_long_context_rates_never_undercut_the_same_tier_base_rate(prices: dict):
    cheaper: Final = [
        f"{name}: {key}={above} < {long_context_anchor(key)}={base}"
        for name, entry in price_entries(prices)
        for key in entry
        if "_above_" in key and "cost_per_token" in key
        for above in [rate(entry, key)]
        for base in [rate(entry, long_context_anchor(key))]
        if above is not None and base is not None and above < base
    ]
    assert cheaper == []


def test_max_output_tokens_fit_inside_max_tokens(prices: dict):
    oversized: Final = [
        f"{name}: max_output_tokens={output} > max_tokens={total}"
        for name, entry in price_entries(prices)
        for output in [rate(entry, "max_output_tokens")]
        for total in [rate(entry, "max_tokens")]
        if output is not None and total is not None and output > total
    ]
    assert oversized == []


def test_regional_azure_rows_are_priced_between_1x_and_2x_the_global_row(prices: dict):
    """Data zone deployments carry a fixed uplift over the global row; a regional row priced below
    global, or more than double it, is a mis-keyed or mis-scaled sync, not a real price."""
    drifted: Final = [
        f"{name}: {key}={regional} vs azure/{suffix}: {key}={global_rate}"
        for name, entry in price_entries(prices)
        for prefix in REGIONAL_AZURE_PREFIXES
        if name.startswith(prefix)
        for suffix in [name[len(prefix) :]]
        for base in [prices.get(f"azure/{suffix}")]
        if isinstance(base, dict)
        for key in REGIONAL_AZURE_RATE_KEYS
        for regional in [rate(entry, key)]
        for global_rate in [rate(base, key)]
        if regional is not None
        and global_rate is not None
        and not global_rate * (1 - 1e-9) <= regional <= global_rate * REGIONAL_UPLIFT_CEILING * (1 + 1e-9)
    ]
    assert drifted == []
