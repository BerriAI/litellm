import asyncio
import itertools
import json
import math
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Final

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, ValidationError

COST_MAP_PATHS: Final = (
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
)
VERCEL_AI_GATEWAY_URL: Final = "https://ai-gateway.vercel.sh/v1/models"
VERCEL_TYPE_TO_MODE: Final = MappingProxyType({"language": "chat", "embedding": "embedding"})
VERCEL_SERVICE_TIERS: Final = frozenset({"flex", "priority", "ultrafast"})
VERCEL_TAG_TO_FLAG: Final = MappingProxyType(
    {
        "tool-use": "supports_function_calling",
        "vision": "supports_vision",
        "reasoning": "supports_reasoning",
        "structured-output": "supports_response_schema",
    }
)


class VercelRates(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    input: float | None = None
    output: float | None = None
    input_cache_read: float | None = None
    input_cache_write: float | None = None

    def cost_map_rates(self) -> dict[str, float]:
        candidate: Final = {
            "input_cost_per_token": self.input,
            "output_cost_per_token": self.output,
            "cache_read_input_token_cost": self.input_cache_read,
            "cache_creation_input_token_cost": self.input_cache_write,
        }
        return {key: cost for key, cost in candidate.items() if cost is not None}


class VercelPriceTier(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    cost: float
    min: int = 0


class VercelLongContextRates(VercelRates):
    threshold: int


class VercelServiceTierRates(VercelRates):
    long_context: VercelLongContextRates | None = None


class VercelPricing(VercelRates):
    input_tiers: tuple[VercelPriceTier, ...] = ()
    output_tiers: tuple[VercelPriceTier, ...] = ()
    input_cache_read_tiers: tuple[VercelPriceTier, ...] = ()
    input_cache_write_tiers: tuple[VercelPriceTier, ...] = ()
    service_tiers: Mapping[str, VercelServiceTierRates] = Field(default_factory=dict)

    def tiers_by_rate(self) -> tuple[tuple[str, tuple[VercelPriceTier, ...]], ...]:
        return (
            ("input_cost_per_token", self.input_tiers),
            ("output_cost_per_token", self.output_tiers),
            ("cache_read_input_token_cost", self.input_cache_read_tiers),
            ("cache_creation_input_token_cost", self.input_cache_write_tiers),
        )


class VercelModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: str
    context_window: int | None = None
    max_tokens: int | None = None
    tags: tuple[str, ...] = ()
    pricing: VercelPricing | None = None


# Asynchronously fetch data from a given URL
async def fetch_data(url):
    try:
        # Create an asynchronous session
        async with aiohttp.ClientSession() as session:
            # Send a GET request to the URL
            async with session.get(url) as resp:
                # Raise an error if the response status is not OK
                resp.raise_for_status()
                # Parse the response JSON
                resp_json = await resp.json()
                print("Fetch the data from URL.")
                # Return the 'data' field from the JSON response
                return resp_json['data']
    except Exception as e:
        # Print an error message if fetching data fails
        print("Error fetching data from URL:", e)
        return None


FRIENDLI_API_URL = "https://api.friendli.ai/serverless/v1/models"
FRIENDLI_PROVIDER = "friendliai"

INHERITABLE_BASE_KEYS = (
    "supports_pdf_input",
    "supports_assistant_prefill",
    "supports_adaptive_thinking",
    "supports_output_config",
)

REASONING_EFFORT_LEVEL_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def _find_base_model_entry(base_model: str, local_data: dict) -> str | None:
    if not base_model:
        return None
    bm_tail = base_model.split("/")[-1].lower()
    if base_model in local_data:
        return base_model
    for key in local_data:
        if key.startswith("sample_spec") or key == "fallback_generalizations":
            continue
        if key.split("/")[-1].lower() == bm_tail:
            return key
    return None


def _reasoning_effort_levels(reasoning_options: list) -> list:
    offered = {
        val
        for opt in reasoning_options or []
        if opt.get("type") == "effort"
        for val in opt.get("values", [])
    }
    return [level for level in REASONING_EFFORT_LEVEL_ORDER if level in offered]


def _valid_token_price(value: object) -> bool:
    try:
        price = float(value)  # pyright: ignore[reportArgumentType]  # non-numeric values are rejected via the except
    except (TypeError, ValueError):
        return False
    return math.isfinite(price) and price >= 0


def _has_valid_token_prices(pricing: dict | None) -> bool:
    prices = pricing or {}
    return _valid_token_price(prices.get("input")) and _valid_token_price(prices.get("output"))


def _pricing(pricing: dict) -> dict:
    out: dict[str, Any] = {}
    if not pricing:
        return out
    if "input" in pricing:
        out["input_cost_per_token"] = float(pricing["input"])
    if "output" in pricing:
        out["output_cost_per_token"] = float(pricing["output"])
    if "input_cache_read" in pricing and pricing["input_cache_read"] is not None:
        out["cache_read_input_token_cost"] = float(pricing["input_cache_read"])
    return out


def _modality_flags(input_mods: list) -> dict:
    mods = input_mods or []
    has_image = "image" in mods
    return {
        "supports_vision": has_image,
        "supports_image_input": has_image,
        "supports_video_input": "video" in mods,
    }


def transform_friendli_data(data: list, local_data: dict) -> dict:
    transformed: dict[str, dict] = {}
    if not data:
        return transformed
    for model in data:
        # An unpriced row must never wholesale-replace an already priced local entry:
        # missing prices cost-calculate as zero, silently zeroing tracked spend
        if not _has_valid_token_prices(model.get("pricing")):
            continue
        model_id = model["id"]
        base_model = model.get("base_model") or ""
        entry: dict[str, Any] = {
            "litellm_provider": FRIENDLI_PROVIDER,
        }

        base_key = _find_base_model_entry(base_model, local_data)
        if base_key:
            base_entry = local_data[base_key]
            for k in INHERITABLE_BASE_KEYS:
                if k in base_entry:
                    entry[k] = base_entry[k]

        ctx = model.get("context_length")
        if ctx is not None:
            entry["max_input_tokens"] = int(ctx)
        max_out = model.get("max_completion_tokens")
        if max_out is not None:
            entry["max_output_tokens"] = int(max_out)
            entry["max_tokens"] = int(max_out)

        pricing = _pricing(model.get("pricing", {}))
        entry.update(pricing)
        entry["supports_prompt_caching"] = "cache_read_input_token_cost" in pricing

        reasoning = model.get("reasoning") is True
        entry["supports_reasoning"] = reasoning
        if reasoning:
            entry["reasoning_effort_levels"] = _reasoning_effort_levels(
                model.get("reasoning_options", [])
            )

        func = model.get("functionality", {})
        entry["supports_function_calling"] = func.get("tool_call") is True
        entry["supports_parallel_function_calling"] = func.get("parallel_tool_call") is True
        is_struct = func.get("structured_output") is True
        entry["supports_response_schema"] = is_struct
        entry["supports_native_structured_output"] = is_struct
        entry["supports_system_messages"] = func.get("system_messages") is True
        entry["supports_tool_choice"] = func.get("tool_choice") is True

        entry.update(_modality_flags(model.get("input_modalities", [])))

        entry["mode"] = model.get("mode", "chat")

        desc = model.get("description")
        if desc:
            entry["comment"] = desc

        dep = model.get("deprecation_date")
        if dep:
            entry["deprecation_date"] = dep.split("T")[0]

        entry["source"] = FRIENDLI_API_URL

        transformed[f"{FRIENDLI_PROVIDER}/{model_id}"] = entry
    return transformed

# Synchronize local data with remote data
def sync_local_data_with_remote(local_data, remote_data, replace_keys=frozenset()):
    # Update existing keys in local_data with values from remote_data
    # (replace_keys entries are swapped wholesale so a field the remote catalog
    #  dropped, e.g. cache pricing, cannot survive as a stale value)
    for key in (set(local_data) & set(remote_data)):
        if key in replace_keys:
            local_data[key] = remote_data[key]
        else:
            local_data[key].update(remote_data[key])

    # Add new keys from remote_data to local_data
    for key in (set(remote_data) - set(local_data)):
        local_data[key] = remote_data[key]

def write_to_file(file_path: str, data: dict[str, object]) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(json.dumps(data, indent=4, ensure_ascii=False) + "\n")
    print(f"Wrote {file_path}.")


def _threshold_label(lower_bound: int) -> str:
    thousands: Final = round(lower_bound / 1000)
    return f"{thousands}k" if abs(lower_bound - thousands * 1000) <= 1 else str(lower_bound - 1)


def _rates_above(rates: dict[str, float], lower_bound: int, suffix: str = "") -> dict[str, float]:
    label: Final = _threshold_label(lower_bound)
    return {f"{key}_above_{label}_tokens{suffix}": cost for key, cost in rates.items()}


def _tier_rates(pricing: VercelPricing) -> Iterator[tuple[str, float]]:
    for key, tiers in pricing.tiers_by_rate():
        yield from ((f"{key}_above_{_threshold_label(tier.min)}_tokens", tier.cost) for tier in tiers if tier.min > 0)


def _service_tier_rates(name: str, rates: VercelServiceTierRates) -> dict[str, float]:
    standard: Final = {f"{key}_{name}": cost for key, cost in rates.cost_map_rates().items()}
    long_context: Final = (
        {}
        if rates.long_context is None
        else _rates_above(rates.long_context.cost_map_rates(), rates.long_context.threshold, f"_{name}")
    )
    return {**standard, **long_context}


def _vercel_entry(model: VercelModel) -> dict[str, object] | None:
    mode: Final = VERCEL_TYPE_TO_MODE[model.type]
    pricing: Final = model.pricing if model.pricing is not None else VercelPricing()
    input_price_unusable: Final = pricing.input is None or pricing.input < 0
    output_price_unusable: Final = mode == "chat" and (pricing.output is None or pricing.output < 0)
    if input_price_unusable or output_price_unusable:
        print(f"Skipping vercel_ai_gateway/{model.id}: the catalog lists no usable per-token price for it.")
        return None
    candidate: Final = {
        "max_tokens": model.max_tokens or None,
        "max_input_tokens": model.context_window or None,
        "max_output_tokens": model.max_tokens or None,
        "input_cost_per_token": pricing.input,
        "output_cost_per_token": pricing.output if pricing.output is not None else 0.0,
        "cache_read_input_token_cost": pricing.input_cache_read,
        "cache_creation_input_token_cost": pricing.input_cache_write,
        "litellm_provider": "vercel_ai_gateway",
        "mode": mode,
    }
    service_tiers: Final = (
        _service_tier_rates(name, rates).items()
        for name, rates in pricing.service_tiers.items()
        if name in VERCEL_SERVICE_TIERS
    )
    flags: Final = {flag: True for tag, flag in VERCEL_TAG_TO_FLAG.items() if tag in model.tags}
    return {
        **{key: value for key, value in candidate.items() if value is not None},
        **dict(_tier_rates(pricing)),
        **dict(itertools.chain.from_iterable(service_tiers)),
        **flags,
    }


def _vercel_model(row: dict[str, object]) -> VercelModel | None:
    try:
        return VercelModel.model_validate(row)
    except ValidationError as error:
        print(f"Skipping vercel_ai_gateway/{row.get('id')}: {error.error_count()} of its catalog fields do not parse.")
        return None


def transform_vercel_ai_gateway_data(data: list[dict[str, object]] | None) -> dict[str, dict[str, object]]:
    parsed: Final = (_vercel_model(row) for row in data or () if row.get("type") in VERCEL_TYPE_TO_MODE)
    entries: Final = ((f"vercel_ai_gateway/{model.id}", _vercel_entry(model)) for model in parsed if model is not None)
    return {key: entry for key, entry in entries if entry is not None}


# Load local data from a specified file
def load_local_data(file_path):
    try:
        # Open the file in read mode
        with open(file_path, "r") as file:
            # Load and return the JSON data
            return json.load(file)
    except FileNotFoundError:
        # Print an error message if the file is not found
        print("File not found:", file_path)
        return None
    except json.JSONDecodeError as e:
        # Print an error message if JSON decoding fails
        print("Error decoding JSON:", e)
        return None

def _fetch(url: str) -> list[dict[str, object]] | None:
    return asyncio.run(fetch_data(url))


def main(
    fetch: Callable[[str], list[dict[str, object]] | None] = _fetch,
    paths: tuple[str, ...] = COST_MAP_PATHS,
) -> None:
    local_data: Final = load_local_data(paths[0])
    vercel_rows: Final = fetch(VERCEL_AI_GATEWAY_URL)
    friendli_rows: Final = fetch(FRIENDLI_API_URL)
    if not local_data or vercel_rows is None or friendli_rows is None:
        raise SystemExit("Failed to fetch model data from either local file or URL.")

    friendli_data: Final = transform_friendli_data(friendli_rows, local_data)
    all_remote_data: Final = {**transform_vercel_ai_gateway_data(vercel_rows), **friendli_data}
    sync_local_data_with_remote(local_data, all_remote_data, replace_keys=frozenset(friendli_data))
    for path in paths:
        write_to_file(path, local_data)

# Entry point of the script
if __name__ == "__main__":
    main()
