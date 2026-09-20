from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import jsonschema
import pytest

import litellm
from litellm.llms.openai.chat.gpt_5_transformation import is_gpt_reasoning_series_name
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
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


OFF_PEAK_ENTRY: Final = MappingProxyType(
    {
        "litellm_provider": "openrouter",
        "mode": "chat",
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 8e-6,
        "off_peak_pricing": {
            "hours_utc": "16:30-00:30",
            "windows": [{"hours_utc": ["00:30-02:00"], "weekdays": [6, "Sunday", "mon", "THURS"]}],
            "weekday_timezone": "Asia/Shanghai",
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 4e-6,
            "cache_read_input_token_cost": 1e-7,
        },
    }
)


def test_generator_classifies_off_peak_pricing_as_a_windowed_rate_block():
    generator = load_generator()
    schema = json.loads(generator.render(generator.build_schema({"some-model": dict(OFF_PEAK_ENTRY)})))
    validator = build_validator(schema)
    assert validator.is_valid({"some-model": dict(OFF_PEAK_ENTRY)})


@pytest.mark.parametrize(
    "block",
    [
        {"hours_utc": "16:30-00:30", "input_cost_per_token": "1e-6"},
        {"hours_utc": "16:30-00:30", "input_cost_per_token": -1e-6},
        {"hours_utc": 1630, "input_cost_per_token": 1e-6},
        {"hours_utc": "16:30-00:30", "discount": 0.5},
        {"windows": [{"weekdays": [6]}], "input_cost_per_token": 1e-6},
        {"windows": [{"hours_utc": "00:30-02:00", "weekdays": [0]}], "input_cost_per_token": 1e-6},
        {"windows": [], "input_cost_per_token": 1e-6},
        {"input_cost_per_token": 1e-6},
        {"hours_utc": "16:30", "input_cost_per_token": 1e-6},
        {"hours_utc": "25:00-01:00", "input_cost_per_token": 1e-6},
        {"hours_utc": ["16:30-00:30", "4pm-midnight"], "input_cost_per_token": 1e-6},
        {"windows": [{"hours_utc": "00:30-02:00", "weekdays": ["Funday"]}], "input_cost_per_token": 1e-6},
    ],
)
def test_generated_off_peak_schema_rejects_malformed_blocks(block: dict):
    generator = load_generator()
    schema = json.loads(generator.render(generator.build_schema({"some-model": dict(OFF_PEAK_ENTRY)})))
    validator = build_validator(schema)
    assert not validator.is_valid({"some-model": {**OFF_PEAK_ENTRY, "off_peak_pricing": block}})


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


def is_active_priced_mistral_chat_row(name: str, entry: Mapping[str, object]) -> bool:
    input_cost: Final = entry.get("input_cost_per_token")
    return (
        name.startswith("mistral/")
        and entry.get("mode") == "chat"
        and entry.get("deprecation_date") is None
        and isinstance(input_cost, (int, float))
        and input_cost > 0
    )


def cache_read_is_tenth_of_input(entry: Mapping[str, object]) -> bool:
    cache_read: Final = entry.get("cache_read_input_token_cost")
    input_cost: Final = entry.get("input_cost_per_token")
    return (
        isinstance(cache_read, float)
        and isinstance(input_cost, (int, float))
        and 0 < cache_read < input_cost
        and cache_read == pytest.approx(input_cost / 10)
    )


@pytest.mark.parametrize("path", (PRICES_PATH, BACKUP_PRICES_PATH), ids=("main", "backup"))
def test_active_mistral_chat_rows_price_cache_reads_below_input(path: Path):
    """A Mistral chat row without a cache-read rate bills cached prompt tokens at zero, so every
    active priced row must carry one, and it must be cheaper than a fresh input token. Mistral
    bills cached tokens at 10% of the input price for every model (docs.mistral.ai/studio/
    conversations/advanced/prompt-caching, read 2026-09-18), so the ratio is checked as well."""
    rows: Mapping[str, object] = json.loads(path.read_text())
    drifted: Final = [
        f"{name}: cache_read={entry.get('cache_read_input_token_cost')} input={entry.get('input_cost_per_token')}"
        for name, entry in rows.items()
        if isinstance(entry, dict)
        and is_active_priced_mistral_chat_row(name, entry)
        and not cache_read_is_tenth_of_input(entry)
    ]
    assert drifted == []


DEEPSEEK_PRICED_ROWS: Final = tuple(
    f"{prefix}{name}"
    for name in ("deepseek-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp", "deepseek-v4-pro")
    for prefix in ("", "deepseek/")
)
DEEPSEEK_OFF_PEAK_WINDOWS: Final = (
    {"hours_utc": ["00:00-01:00", "04:00-06:00", "10:00-00:00"], "weekdays": [1, 2, 3, 4, 5]},
    {"hours_utc": "00:00-00:00", "weekdays": [6, 7]},
)
DEEPSEEK_HALVED_RATES: Final = ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost")


def deepseek_off_peak_drift(entry: Mapping[str, object]) -> str | None:
    block: Final = entry.get("off_peak_pricing")
    if not isinstance(block, dict):
        return "no off_peak_pricing block"
    if tuple(block.get("windows", ())) != DEEPSEEK_OFF_PEAK_WINDOWS:
        return f"windows={block.get('windows')}"
    halved: Final = {rate: block.get(rate) for rate in DEEPSEEK_HALVED_RATES}
    expected: Final = {rate: float(str(entry[rate])) / 2 for rate in DEEPSEEK_HALVED_RATES}
    mismatched: Final = {
        rate for rate in DEEPSEEK_HALVED_RATES if halved[rate] != pytest.approx(expected[rate], rel=1e-9)
    }
    return f"off-peak rates {halved} are not half of the listed rates" if mismatched else None


@pytest.mark.parametrize("path", (PRICES_PATH, BACKUP_PRICES_PATH), ids=("main", "backup"))
def test_deepseek_rows_bill_half_rate_outside_weekday_peak_hours(path: Path):
    """DeepSeek charges half its listed rate outside 01:00-04:00 and 06:00-10:00 UTC Monday to
    Friday (api-docs.deepseek.com/quick_start/pricing, read 2026-09-19), so every row on that
    pricing page carries an off_peak_pricing block with those windows and the halved rates."""
    rows: Mapping[str, object] = json.loads(path.read_text())
    drifted: Final = {
        name: deepseek_off_peak_drift(entry)
        for name in DEEPSEEK_PRICED_ROWS
        if isinstance(entry := rows.get(name), dict) and deepseek_off_peak_drift(entry) is not None
    }
    assert drifted == {}
    assert all(name in rows for name in DEEPSEEK_PRICED_ROWS)


PROVIDER_LABELS_WITHOUT_A_MODEL_SET: Final = frozenset({"sagemaker", "bedrock_converse"})
MODES_SERVED_OUTSIDE_THE_LLM_PROVIDER_REGISTRY: Final = frozenset({"search", "evaluation"})
VERTEX_FAMILIES_A_VERTEX_WILDCARD_GRANT_DOES_NOT_LIST: Final = frozenset(
    {
        "vertex_ai-ai21_models",
        "vertex_ai-embedding-models",
        "vertex_ai-image-models",
        "vertex_ai-llama_models",
        "vertex_ai-mistral_models",
        "vertex_ai-openai_models",
        "vertex_ai-qwen_models",
        "vertex_ai-video-models",
    }
)


def is_registered_provider(label: str, model_names: tuple[str, ...]) -> bool:
    if label in litellm.models_by_provider or JSONProviderRegistry.exists(label):
        return True
    family_root: Final = label.split("-", 1)[0]
    wildcard_models: Final = litellm.models_by_provider.get(family_root, ())
    return any(
        name in wildcard_models or name.removeprefix(f"{family_root}/") in wildcard_models for name in model_names
    )


def unregistered_providers(rows: Mapping[str, object]) -> list[str]:
    labelled_rows: Final = tuple(
        (name, entry["litellm_provider"])
        for name, entry in rows.items()
        if name != "sample_spec"
        and isinstance(entry, dict)
        and "litellm_provider" in entry
        and entry.get("mode") not in MODES_SERVED_OUTSIDE_THE_LLM_PROVIDER_REGISTRY
        and entry["litellm_provider"] not in PROVIDER_LABELS_WITHOUT_A_MODEL_SET
        and entry["litellm_provider"] not in VERTEX_FAMILIES_A_VERTEX_WILDCARD_GRANT_DOES_NOT_LIST
    )
    return sorted(
        label
        for label in {label for _, label in labelled_rows}
        if not is_registered_provider(label, tuple(name for name, row_label in labelled_rows if row_label == label))
    )


@pytest.mark.parametrize("path", (PRICES_PATH, BACKUP_PRICES_PATH), ids=("main", "backup"))
def test_every_cost_map_provider_is_registered(path: Path):
    assert unregistered_providers(json.loads(path.read_text())) == [], (
        f"{path.name} carries a litellm_provider whose models a `<provider>/*` grant does not list. A new provider "
        "needs a `<provider>_models` set in litellm/__init__.py, filled in _populate_provider_model_sets and listed "
        "in _build_models_by_provider. A new `<provider>-<family>` label needs its rows added to a set that "
        "`models_by_provider[<provider>]` includes"
    )


def test_unregistered_provider_guard_flags_only_labels_nobody_registered():
    wired_vertex_model: Final = sorted(litellm.vertex_language_models)[0]
    rows: Final = {
        "sample_spec": {"litellm_provider": "one of the supported providers", "mode": "chat"},
        "nobody_registered/StartJob": {"litellm_provider": "nobody_registered", "mode": "audio_transcription"},
        "gpt-4o": {"litellm_provider": "openai", "mode": "chat"},
        wired_vertex_model: {"litellm_provider": "vertex_ai-language-models", "mode": "chat"},
        "vertex_ai/new-family-model": {"litellm_provider": "vertex_ai-new_family_models", "mode": "chat"},
        "unknown_root/model": {"litellm_provider": "unknown_root-new_family_models", "mode": "chat"},
        "some_search/search": {"litellm_provider": "some_search", "mode": "search"},
    }
    assert unregistered_providers(rows) == [
        "nobody_registered",
        "unknown_root-new_family_models",
        "vertex_ai-new_family_models",
    ]
