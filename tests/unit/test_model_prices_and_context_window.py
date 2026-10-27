from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

import litellm
from litellm.litellm_core_utils.fallback_generalizations import match_capability_generalizations
from litellm.llms.openai.chat.gpt_reasoning_transformation import is_gpt_reasoning_series_name
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.router_utils.reasoning_effort_capability import resolve_supported_reasoning_efforts
from litellm.types.utils import SearchFreeRequests

REPO_ROOT = Path(__file__).parents[2]
PRICES_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PRICES_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"


@pytest.fixture(scope="module")
def prices() -> dict:
    return json.loads(PRICES_PATH.read_text())


@pytest.mark.parametrize("model", ("azure/gpt-audio-1.5-2026-02-23", "azure/gpt-audio-mini-2025-10-06"))
def test_azure_audio_snapshot_capabilities_match_declared_modalities(model: str, local_model_cost_map: None) -> None:
    catalog: Final = TypeAdapter(dict[str, JsonValue]).validate_json(PRICES_PATH.read_bytes())
    backup: Final = TypeAdapter(dict[str, JsonValue]).validate_json(BACKUP_PRICES_PATH.read_bytes())
    entry: Final = catalog[model]
    assert isinstance(entry, dict)
    assert entry == backup[model]
    input_modalities: Final = entry["supported_modalities"]
    output_modalities: Final = entry["supported_output_modalities"]
    assert isinstance(input_modalities, list)
    assert isinstance(output_modalities, list)
    expected: Final = ("audio" in input_modalities, "audio" in output_modalities)
    info: Final = litellm.get_model_info(model)
    assert (info["supports_audio_input"], info["supports_audio_output"]) == expected
    assert (litellm.supports_audio_input(model), litellm.supports_audio_output(model)) == expected


class _CopilotEndpointRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    mode: str
    max_input_tokens: int
    max_output_tokens: int
    supported_endpoints: tuple[str, ...]


class _CopilotThinkingCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    supports_reasoning: bool | None = None
    supports_adaptive_thinking: bool | None = None
    supports_legacy_thinking: bool | None = None
    thinking_always_on: bool | None = None


def _copilot_endpoint_rows(path: Path) -> Mapping[str, _CopilotEndpointRow]:
    prices: Final[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue]).validate_json(path.read_bytes())
    return MappingProxyType(
        {
            key: _CopilotEndpointRow.model_validate(value)
            for key, value in prices.items()
            if key.startswith("github_copilot/") and isinstance(value, dict) and "supported_endpoints" in value
        }
    )


def _thinking_capabilities(prices: Mapping[str, JsonValue], key: str) -> _CopilotThinkingCapabilities:
    return _CopilotThinkingCapabilities.model_validate(prices[key])


def _github_copilot_catalog_rows(path: Path) -> Mapping[str, JsonValue]:
    prices: Final[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue]).validate_json(path.read_bytes())
    return MappingProxyType({key: value for key, value in prices.items() if key.startswith("github_copilot/")})


_THINKING_KEYS: Final[tuple[str, ...]] = (
    "supports_reasoning",
    "supports_adaptive_thinking",
    "supports_legacy_thinking",
    "thinking_always_on",
)


def _anthropic_counterpart(key: str) -> str:
    return key.removeprefix("github_copilot/").removesuffix("-fast").replace(".", "-")


@pytest.mark.parametrize(("path",), [(PRICES_PATH,), (BACKUP_PRICES_PATH,)], ids=("main", "backup"))
def test_github_copilot_rows_derive_mode_from_their_endpoints(path: Path) -> None:
    rows: Final[Mapping[str, _CopilotEndpointRow]] = _copilot_endpoint_rows(path)
    assert rows, f"No GitHub Copilot endpoint rows found in {path}"
    assert {key: row.mode for key, row in rows.items()} == {
        key: "chat" if "/v1/chat/completions" in row.supported_endpoints else "responses" for key, row in rows.items()
    }


def _rows_carrying_legacy_max_tokens(prices: Mapping[str, Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(key for key, row in prices.items() if "max_tokens" in row)


def test_no_row_carries_the_retired_max_tokens_field(prices: dict) -> None:
    assert _rows_carrying_legacy_max_tokens(prices) == ()


def test_github_copilot_backup_rows_match_the_main_catalog() -> None:
    assert _github_copilot_catalog_rows(PRICES_PATH) == _github_copilot_catalog_rows(BACKUP_PRICES_PATH)


def test_github_copilot_rows_resolve_through_get_model_info() -> None:
    rows: Final[Mapping[str, _CopilotEndpointRow]] = _copilot_endpoint_rows(PRICES_PATH)
    assert {
        key: (litellm.get_model_info(key)["mode"], litellm.get_model_info(key)["max_input_tokens"]) for key in rows
    } == {key: (row.mode, row.max_input_tokens) for key, row in rows.items()}


def test_github_copilot_rows_keep_the_reasoning_flag_their_family_fallback_supplies() -> None:
    reasoning_rows: Final = tuple(
        key
        for key in _copilot_endpoint_rows(PRICES_PATH)
        if (match_capability_generalizations(key) or {}).get("supports_reasoning") is True
    )
    prices: Final[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue]).validate_json(PRICES_PATH.read_bytes())
    assert reasoning_rows
    assert {key: _thinking_capabilities(prices, key).supports_reasoning for key in reasoning_rows} == dict.fromkeys(
        reasoning_rows, True
    )
    assert {key: litellm.get_model_info(key).get("supports_reasoning") for key in reasoning_rows} == dict.fromkeys(
        reasoning_rows, True
    )
    assert {
        key: resolve_supported_reasoning_efforts(
            litellm.get_model_info(key),
            deployment_is_mapped=True,
        )
        != ()
        for key in reasoning_rows
    } == dict.fromkeys(reasoning_rows, True)


def test_github_copilot_messages_claude_rows_keep_their_anthropic_thinking_capabilities() -> None:
    prices: Final[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue]).validate_json(PRICES_PATH.read_bytes())
    claude_rows: Final = tuple(
        key
        for key, row in _copilot_endpoint_rows(PRICES_PATH).items()
        if key.startswith("github_copilot/claude-") and "/v1/messages" in row.supported_endpoints
    )
    assert claude_rows
    assert {key: _thinking_capabilities(prices, key) for key in claude_rows} == {
        key: _thinking_capabilities(prices, _anthropic_counterpart(key)) for key in claude_rows
    }
    assert {
        key: tuple(litellm.get_model_info(key).get(thinking_key) for thinking_key in _THINKING_KEYS)
        for key in claude_rows
    } == {
        key: tuple(
            litellm.get_model_info(_anthropic_counterpart(key)).get(thinking_key) for thinking_key in _THINKING_KEYS
        )
        for key in claude_rows
    }


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


# Azure Foundry GPT-6.1 Sol announcement, 2026-09-29: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-gpt-6-1-sol-in-microsoft-foundry-advanced-intelligence-optimized-for/4560811
AZURE_GPT_6_1_SOL_REGIONAL_PREMIUM: Final = MappingProxyType({"us": 1.1, "eu": 1.2, "apac": 1.2})


@pytest.mark.parametrize("region", sorted(AZURE_GPT_6_1_SOL_REGIONAL_PREMIUM))
def test_azure_gpt_6_1_sol_data_zone_rows_charge_the_global_web_search_price(prices: dict, region: str):
    global_search_cost = prices["azure/gpt-6.1-sol"]["search_context_cost_per_query"]
    assert global_search_cost
    assert prices[f"azure/{region}/gpt-6.1-sol"].get("search_context_cost_per_query") == global_search_cost


@pytest.mark.parametrize("region", sorted(AZURE_GPT_6_1_SOL_REGIONAL_PREMIUM))
def test_azure_gpt_6_1_sol_data_zone_rows_price_requests_at_the_regional_premium(region: str):
    premium = AZURE_GPT_6_1_SOL_REGIONAL_PREMIUM[region]
    usage = {"prompt_tokens": 300_000, "completion_tokens": 1_000, "cache_read_input_tokens": 100_000}

    def total_cost(model: str) -> float:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            custom_llm_provider="azure",
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            cache_read_input_tokens=usage["cache_read_input_tokens"],
        )
        return prompt_cost + completion_cost

    global_cost = total_cost("azure/gpt-6.1-sol")
    assert global_cost > 0
    assert total_cost(f"azure/{region}/gpt-6.1-sol") == pytest.approx(global_cost * premium)


AZURE_GPT_6_1_SOL_ROWS: Final = (
    "azure/gpt-6.1-sol",
    "azure/gpt-6.1-sol-2026-09-29",
    "azure/us/gpt-6.1-sol",
    "azure/eu/gpt-6.1-sol",
    "azure/apac/gpt-6.1-sol",
    "azure_ai/gpt-6.1-sol",
)


@pytest.mark.parametrize("row", AZURE_GPT_6_1_SOL_ROWS)
def test_azure_gpt_6_1_sol_rows_accept_the_same_reasoning_efforts_as_the_openai_row(prices: dict, row: str):
    openai_efforts: Final = resolve_supported_reasoning_efforts(prices["gpt-6.1-sol"], deployment_is_mapped=True)
    assert openai_efforts

    assert resolve_supported_reasoning_efforts(prices[row], deployment_is_mapped=True) == openai_efforts


# https://ai.google.dev/gemini-api/docs/models/deep-research-preview-04-2026
@pytest.mark.parametrize("model", ["gemini/deep-research-preview-04-2026", "gemini/deep-research-max-preview-04-2026"])
def test_gemini_deep_research_previews_accept_the_documented_input_window(prices: dict, model: str):
    assert prices[model]["max_input_tokens"] == 1_048_576
    assert prices[model]["max_output_tokens"] == 65_536


# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
VERTEX_GEMINI_FLASH_RETIREMENT_DATES: Final = MappingProxyType(
    {"gemini-3.6-flash": "2026-11-19", "gemini-3.7-flash": "2027-01-28"}
)


@pytest.mark.parametrize("model", sorted(VERTEX_GEMINI_FLASH_RETIREMENT_DATES))
@pytest.mark.parametrize("prefix", ["", "vertex_ai/"])
def test_vertex_gemini_flash_rows_carry_the_documented_retirement_date(prices: dict, model: str, prefix: str):
    assert prices[f"{prefix}{model}"]["deprecation_date"] == VERTEX_GEMINI_FLASH_RETIREMENT_DATES[model]


def _sampling_flags_across_provider_twins(prices: Mapping[str, Mapping[str, object]], key: str) -> frozenset[object]:
    bare: Final = key.split("/")[-1]
    twins: Final = tuple(twin for twin in (bare, f"vertex_ai/{bare}", f"gemini/{bare}") if twin in prices)
    return frozenset(prices[twin].get("supports_sampling_params") for twin in (key, *twins))


def test_gemini_rows_agree_on_supports_sampling_params_across_their_provider_twins(
    prices: Mapping[str, Mapping[str, object]],
) -> None:
    """The bare, vertex_ai/ and gemini/ rows of one Gemini model gate sampling params identically, so a request
    takes the same drop-or-400 path on the Vertex route and the AI Studio route"""
    flagged: Final = tuple(
        key
        for key, row in prices.items()
        if "supports_sampling_params" in row and key.split("/")[-1].startswith("gemini-")
    )
    assert flagged
    for key in flagged:
        assert len(_sampling_flags_across_provider_twins(prices, key)) == 1, key


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


_ABSENT: Final = object()

REASONING_ANNOTATION_KEYS: Final = (
    "supports_reasoning",
    "supports_minimal_reasoning_effort",
    "supports_none_reasoning_effort",
    "supports_xhigh_reasoning_effort",
    "default_reasoning_effort",
)


def chatgpt_openai_twins(prices: dict) -> list[tuple[str, str]]:
    """`chatgpt/<model>` rows paired with the bare `<model>` row served by the openai provider.

    Scoped to openai twins on purpose. `ChatGPTConfig` and `ChatGPTResponsesAPIConfig` subclass
    their openai counterparts, so a chatgpt row's reasoning behaviour is whatever the openai row
    describes. The azure rows are a separate registry that already diverges from openai here, and
    pinning them to each other would assert something this repository does not control.
    """
    pairs = []
    for name, entry in prices.items():
        if not isinstance(entry, dict) or not name.startswith("chatgpt/"):
            continue
        bare = name.split("/", 1)[1]
        twin = prices.get(bare)
        if isinstance(twin, dict) and twin.get("litellm_provider") == "openai":
            pairs.append((name, bare))
    return pairs


def test_chatgpt_rows_carry_their_openai_twin_reasoning_annotations(prices: dict):
    """A chatgpt row must not silently drop the reasoning annotations of the model it proxies.

    `litellm.utils._get_model_info_from_generalization` refuses to fall back when an exact cost-map
    key exists, so an unannotated `chatgpt/<model>` row wins over its annotated twin and
    `/model/info` reports the model as non-reasoning.
    """
    twins = chatgpt_openai_twins(prices)
    assert twins, "no chatgpt/* row has an openai twin any more; this guard has stopped guarding"

    mismatched = []
    for name, bare in twins:
        for key in REASONING_ANNOTATION_KEYS:
            if prices[name].get(key, _ABSENT) != prices[bare].get(key, _ABSENT):
                mismatched.append(
                    f"{name}.{key} is {prices[name].get(key)!r}, {bare}.{key} is {prices[bare].get(key)!r}"
                )

    assert mismatched == [], (
        "chatgpt/* entries proxy their openai twin through ChatGPTConfig, so they must carry the "
        "same reasoning annotations; an exact cost-map key blocks the generalization fallback, so "
        "a missing flag here is reported to callers as 'not a reasoning model':\n" + "\n".join(mismatched)
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


def cache_read_is_below_input(entry: Mapping[str, object]) -> bool:
    cache_read: Final = entry.get("cache_read_input_token_cost")
    input_cost: Final = entry.get("input_cost_per_token")
    return isinstance(cache_read, float) and isinstance(input_cost, (int, float)) and 0 < cache_read < input_cost


@pytest.mark.parametrize("path", (PRICES_PATH, BACKUP_PRICES_PATH), ids=("main", "backup"))
def test_active_mistral_chat_rows_price_cache_reads_below_input(path: Path):
    """A Mistral chat row without a cache-read rate bills cached prompt tokens at zero, so every
    active priced row must carry one, and it must be cheaper than a fresh input token."""
    rows: Mapping[str, object] = json.loads(path.read_text())
    drifted: Final = [
        f"{name}: cache_read={entry.get('cache_read_input_token_cost')} input={entry.get('input_cost_per_token')}"
        for name, entry in rows.items()
        if isinstance(entry, dict)
        and is_active_priced_mistral_chat_row(name, entry)
        and not cache_read_is_below_input(entry)
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
TENCENT_TOKENHUB_DEEPSEEK_PRICED_ROWS: Final = (
    "tencent/deepseek-v4-pro-0813",
    "tencent/deepseek-v4.1-flash",
)
TENCENT_TOKENHUB_GLM_ROWS: Final = MappingProxyType(
    {
        "tencent/glm-5.3-flashx": (3.7e-07, 1.25e-06, 7.5e-08, True, True),
        "tencent/glm-5.3-flash": (1.5e-07, 5e-07, 3e-08, True, True),
        "tencent/glm-5.3": (1.4e-06, 4.4e-06, 2.6e-07, False, False),
        "tencent/glm-5.2": (1.4e-06, 4.4e-06, 2.6e-07, False, False),
    }
)
TENCENT_TOKENHUB_KIMI_ROWS: Final = MappingProxyType(
    {
        "tencent/kimi-k3": (3e-06, 1.5e-05, 3e-07, 1000000, False, False),
        "tencent/kimi-k2.8-preview": (1e-06, 4e-06, 2.5e-07, 1000000, False, False),
        "tencent/kimi-k2.7-code-highspeed": (1.9e-06, 8e-06, 3.8e-07, 256000, False, False),
        "tencent/kimi-k2.7-code": (9.5e-07, 4e-06, 1.9e-07, 256000, False, False),
    }
)


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


@pytest.mark.parametrize("path", (PRICES_PATH, BACKUP_PRICES_PATH), ids=("main", "backup"))
def test_tencent_tokenhub_deepseek_rows_bill_half_rate_outside_weekday_peak_hours(path: Path):
    """Tencent TokenHub applies the DeepSeek weekday peak windows and half-rate off-peak prices
    to these models (tencentcloud.com/document/product/1300/78937, read 2026-10-13)."""
    rows: Mapping[str, object] = json.loads(path.read_text())
    drifted: Final = {
        name: deepseek_off_peak_drift(entry)
        for name in TENCENT_TOKENHUB_DEEPSEEK_PRICED_ROWS
        if isinstance(entry := rows.get(name), dict) and deepseek_off_peak_drift(entry) is not None
    }
    invalid_timezones: Final = tuple(
        name
        for name in TENCENT_TOKENHUB_DEEPSEEK_PRICED_ROWS
        if isinstance(entry := rows.get(name), dict)
        and isinstance(block := entry.get("off_peak_pricing"), dict)
        and block.get("weekday_timezone") != "Asia/Shanghai"
    )
    assert drifted == {}
    assert all(name in rows for name in TENCENT_TOKENHUB_DEEPSEEK_PRICED_ROWS)
    assert invalid_timezones == ()


@pytest.mark.parametrize("path", (PRICES_PATH, BACKUP_PRICES_PATH), ids=("main", "backup"))
def test_tencent_tokenhub_glm_rows_match_published_metadata(path: Path):
    """TokenHub publishes these prices and capabilities in its official model catalog and rate card
    (tencentcloud.com/document/product/1300/78934 and /78937, read 2026-10-13)."""
    rows: Mapping[str, object] = json.loads(path.read_text())
    actual: Final = MappingProxyType(
        {
            name: (
                entry.get("input_cost_per_token"),
                entry.get("output_cost_per_token"),
                entry.get("cache_read_input_token_cost"),
                entry.get("supports_vision", False),
                entry.get("supports_video_input", False),
            )
            for name in TENCENT_TOKENHUB_GLM_ROWS
            if isinstance(entry := rows.get(name), dict)
        }
    )
    invalid_metadata: Final = tuple(
        name
        for name in TENCENT_TOKENHUB_GLM_ROWS
        if not isinstance(entry := rows.get(name), dict)
        or entry.get("litellm_provider") != "tencent"
        or entry.get("max_input_tokens") != 1000000
        or entry.get("max_output_tokens") != 128000
        or entry.get("input_cost_per_token_cache_hit") != entry.get("cache_read_input_token_cost")
    )
    assert actual == TENCENT_TOKENHUB_GLM_ROWS
    assert invalid_metadata == ()


@pytest.mark.parametrize("path", (PRICES_PATH, BACKUP_PRICES_PATH), ids=("main", "backup"))
def test_tencent_tokenhub_kimi_rows_match_published_metadata(path: Path):
    """TokenHub publishes these Kimi prices and capabilities in its official model catalog and rate card
    (tencentcloud.com/document/product/1300/78934 and /78937, read 2026-10-13)."""
    rows: Mapping[str, object] = json.loads(path.read_text())
    actual: Final = MappingProxyType(
        {
            name: (
                entry.get("input_cost_per_token"),
                entry.get("output_cost_per_token"),
                entry.get("cache_read_input_token_cost"),
                entry.get("max_output_tokens"),
                entry.get("supports_vision", False),
                entry.get("supports_video_input", False),
            )
            for name in TENCENT_TOKENHUB_KIMI_ROWS
            if isinstance(entry := rows.get(name), dict)
        }
    )
    invalid_metadata: Final = tuple(
        name
        for name, expected in TENCENT_TOKENHUB_KIMI_ROWS.items()
        if not isinstance(entry := rows.get(name), dict)
        or entry.get("litellm_provider") != "tencent"
        or entry.get("max_input_tokens") != expected[3]
        or entry.get("max_output_tokens") != expected[3]
        or entry.get("input_cost_per_token_cache_hit") != entry.get("cache_read_input_token_cost")
        or not entry.get("supports_reasoning")
        or not entry.get("supports_response_schema")
        or not entry.get("supports_function_calling")
        or not entry.get("supports_prompt_caching")
    )
    assert actual == TENCENT_TOKENHUB_KIMI_ROWS
    assert invalid_metadata == ()


class TestBedrockClaude35SonnetContextWindow:
    @pytest.mark.parametrize("path", (PRICES_PATH, BACKUP_PRICES_PATH), ids=("main", "backup"))
    def test_bare_model_ids_match_us_and_eu_siblings(self, path: Path):
        rows = json.loads(path.read_text())
        for model in (
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        ):
            assert rows[model]["max_input_tokens"] == rows[f"us.{model}"]["max_input_tokens"]
            assert rows[model]["max_input_tokens"] == rows[f"eu.{model}"]["max_input_tokens"]

    @pytest.mark.parametrize("path", (PRICES_PATH, BACKUP_PRICES_PATH), ids=("main", "backup"))
    def test_bare_model_ids_have_200k_context_window(self, path: Path):
        rows = json.loads(path.read_text())
        # Claude 3.5 Sonnet context window is 200K tokens, https://www.anthropic.com/news/claude-3-5-sonnet (accessed 2026-10-13)
        for model in (
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        ):
            assert rows[model]["max_input_tokens"] == 200000

    @pytest.mark.parametrize("path", (PRICES_PATH, BACKUP_PRICES_PATH), ids=("main", "backup"))
    def test_bare_model_ids_have_no_above_200k_pricing(self, path: Path):
        rows = json.loads(path.read_text())
        above_200k_fields = (
            "input_cost_per_token_above_200k_tokens",
            "output_cost_per_token_above_200k_tokens",
            "cache_creation_input_token_cost_above_200k_tokens",
            "cache_read_input_token_cost_above_200k_tokens",
        )
        for model in (
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        ):
            assert not set(above_200k_fields).intersection(rows[model])


PROVIDER_LABELS_WITHOUT_A_MODEL_SET: Final = frozenset({"sagemaker", "bedrock_converse"})
MODES_SERVED_OUTSIDE_THE_LLM_PROVIDER_REGISTRY: Final = frozenset({"search", "decisions", "evaluation"})
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


def _search_free_requests_rows(path: Path) -> Mapping[str, dict[str, JsonValue]]:
    prices: Final[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue]).validate_json(path.read_bytes())
    return MappingProxyType(
        {key: row for key, row in prices.items() if isinstance(row, dict) and "search_free_requests" in row}
    )


@pytest.mark.parametrize("path", (PRICES_PATH, BACKUP_PRICES_PATH), ids=("main", "backup"))
def test_search_free_requests_rows_are_grounding_rows_with_one_allowance_per_group(path: Path) -> None:
    rows = _search_free_requests_rows(path)
    assert rows
    allowances = {
        key: TypeAdapter(SearchFreeRequests).validate_python(row["search_free_requests"]) for key, row in rows.items()
    }
    for key, row in rows.items():
        assert row.get("supports_web_search") is True, key
        assert row.get("search_context_cost_per_query"), key
        assert allowances[key]["count"] > 0, key
    for group in {allowance["shared_group"] for allowance in allowances.values()}:
        same_group = {(a["count"], a["period"]) for a in allowances.values() if a["shared_group"] == group}
        assert len(same_group) == 1, (group, same_group)


def test_search_free_requests_backup_rows_match_the_main_catalog() -> None:
    assert _search_free_requests_rows(BACKUP_PRICES_PATH) == _search_free_requests_rows(PRICES_PATH)


@pytest.mark.parametrize("path", (PRICES_PATH, BACKUP_PRICES_PATH), ids=("main", "backup"))
def test_volcengine_doubao_seed_2_1_pro_260915_row_matches_published_metadata(path: Path):
    """Volcengine Ark lists doubao-seed-2-1-pro-260915 at 6 / 30 CNY per million input / output tokens with
    cache hits at 1.2 (ark.volcengine.com model detail and docs.volcengine.com/docs/82379/1544106, read
    2026-10-20), a 1024k context window and a 256k output cap. USD rows use the 6.9565 CNY/USD rate of the
    sibling 260628 row."""
    rows: Mapping[str, object] = json.loads(path.read_text())
    entry = rows.get("volcengine/doubao-seed-2-1-pro-260915")
    assert isinstance(entry, dict)
    assert entry["litellm_provider"] == "volcengine"
    assert entry["mode"] == "chat"
    assert entry["max_input_tokens"] == 1024000
    assert entry["max_output_tokens"] == 256000
    # Volcengine Ark pricing https://www.volcengine.com/docs/82379/1544106 checked 2026-10-20
    assert entry["input_cost_per_token"] == pytest.approx(6.0 / 6.9565 / 1_000_000, rel=1e-4)
    assert entry["output_cost_per_token"] == pytest.approx(30.0 / 6.9565 / 1_000_000, rel=1e-4)
    assert entry["cache_read_input_token_cost"] == pytest.approx(1.2 / 6.9565 / 1_000_000, rel=1e-4)
    sibling = rows["volcengine/doubao-seed-2-1-pro-260628"]
    assert isinstance(sibling, dict)
    assert all(entry[key] == sibling[key] for key in ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost"))


def _bedrock_gpt_oss_rows(prices: Mapping[str, JsonValue]) -> tuple[str, ...]:
    return tuple(
        key
        for key, row in prices.items()
        if "gpt-oss" in key and isinstance(row, dict) and str(row.get("litellm_provider", "")).startswith("bedrock")
    )


@pytest.mark.parametrize("path", [PRICES_PATH, BACKUP_PRICES_PATH], ids=["main", "backup"])
def test_bedrock_gpt_oss_rows_declare_vision_support_like_their_safeguard_siblings(path: Path) -> None:
    prices: Final = json.loads(path.read_text())
    rows: Final = _bedrock_gpt_oss_rows(prices)
    assert rows, "no Bedrock gpt-oss rows found"
    reference: Final = prices["openai.gpt-oss-safeguard-120b"]["supports_vision"]
    undeclared: Final = tuple(key for key in rows if prices[key].get("supports_vision") is not reference)
    assert undeclared == (), f"modality routing only skips rows that declare supports_vision: {undeclared}"


def test_bedrock_gpt_oss_vision_is_explicitly_disabled_for_the_router_gate() -> None:
    assert litellm.utils.is_vision_explicitly_disabled("bedrock/openai.gpt-oss-120b-1:0") is True
    assert litellm.utils.is_vision_explicitly_disabled("bedrock_mantle/openai.gpt-oss-20b") is True
