import asyncio
import aiohttp
import json
import math
from typing import Any

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

# Public, keyless catalog of models callable on /v1/chat/completions and /v1/responses. The
# Iran-hosted API (gateway.dothoosh.com) is not reachable from GitHub runners, so this is the
# byte-for-byte public mirror. Only ids, limits, prices, modalities and capability flags are
# copied, never the catalog version or verification timestamps, which rotate.
DOTHOOSH_API_URL = "https://www.dothoosh.com/gateway/v1/models"
DOTHOOSH_PROVIDER = "dothoosh"
# Absence deletes a local entry, so a list shorter than this is treated as truncated.
DOTHOOSH_MIN_MODELS = 5
DOTHOOSH_ENDPOINTS = {
    "openai-chat-completions": "/v1/chat/completions",
    "openai-responses": "/v1/responses",
}

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

def _threshold_label(tokens: int) -> str:
    # LiteLLM reads both `_above_128k_tokens` and `_above_128_tokens` key styles.
    return f"{tokens // 1000}k" if tokens % 1000 == 0 else str(tokens)


def _dothoosh_protocols(row: dict) -> list:
    names = []
    for entry in row.get("protocols") or []:
        name = entry.get("protocol") if isinstance(entry, dict) else entry
        if name in DOTHOOSH_ENDPOINTS:
            names.append(name)
    return names


def _token_price(value: object) -> float | None:
    if value is None:
        return None
    return float(value) if _valid_token_price(value) else None


def transform_dothoosh_data(data: list) -> dict:
    """Turn the DotHoosh gateway's public /v1/models list into `dothoosh/<id>` entries.

    Prices are USD per token (decimal strings). `pricing_tiers` give the prices that apply when
    the prompt exceeds `above_input_tokens`; a tier omits the fields that do not change, so each
    emitted tier repeats the previous tier's value for them (LiteLLM falls back to the base price,
    not to the previous tier, for a missing key).
    """
    transformed: dict[str, dict] = {}
    if not data:
        return transformed
    for row in data:
        model_id = row.get("id")
        pricing = row.get("pricing") or {}
        input_cost = _token_price(pricing.get("prompt"))
        output_cost = _token_price(pricing.get("completion"))
        top = row.get("top_provider") or {}
        context = row.get("context_length") or top.get("context_length") or row.get("context_window")
        # An unpriced or unsized row must never replace a priced local entry: missing prices
        # cost-calculate as zero.
        if not model_id or input_cost is None or output_cost is None or not context:
            continue
        # protocols absent: the row is on the list's own route (chat completions)
        protocols = _dothoosh_protocols(row) or ["openai-chat-completions"]

        entry: dict[str, Any] = {
            "litellm_provider": DOTHOOSH_PROVIDER,
            "mode": "chat" if "openai-chat-completions" in protocols else "responses",
            "max_input_tokens": int(context),
            "input_cost_per_token": input_cost,
            "output_cost_per_token": output_cost,
        }
        max_out = top.get("max_completion_tokens") or row.get("max_output_tokens")
        if max_out:
            entry["max_output_tokens"] = int(max_out)
            entry["max_tokens"] = int(max_out)

        cache_read = _token_price(pricing.get("input_cache_read"))
        cache_write = _token_price(pricing.get("input_cache_write"))
        cache_write_1h = _token_price(pricing.get("input_cache_write_1h"))
        if cache_read is not None:
            entry["cache_read_input_token_cost"] = cache_read
        if cache_write is not None:
            entry["cache_creation_input_token_cost"] = cache_write
        if cache_write_1h is not None:
            entry["cache_creation_input_token_cost_above_1hr"] = cache_write_1h
        entry["supports_prompt_caching"] = cache_read is not None

        current = {
            "input_cost_per_token": input_cost,
            "output_cost_per_token": output_cost,
            "cache_read_input_token_cost": cache_read,
            "cache_creation_input_token_cost": cache_write if cache_write is not None else cache_write_1h,
        }
        for tier in sorted(row.get("pricing_tiers") or [], key=lambda t: t["above_input_tokens"]):
            label = _threshold_label(int(tier["above_input_tokens"]))
            for key, source in (
                ("input_cost_per_token", "prompt"),
                ("output_cost_per_token", "completion"),
                ("cache_read_input_token_cost", "input_cache_read"),
                ("cache_creation_input_token_cost", "input_cache_write"),
            ):
                value = _token_price(tier.get(source))
                if value is None and key == "cache_creation_input_token_cost":
                    value = _token_price(tier.get("input_cache_write_1h"))
                if value is not None:
                    current[key] = value
                if current[key] is not None:
                    entry[f"{key}_above_{label}_tokens"] = current[key]

        params = set(row.get("supported_parameters") or [])
        entry["supports_function_calling"] = "tools" in params
        entry["supports_tool_choice"] = "tool_choice" in params
        entry["supports_reasoning"] = "reasoning" in params
        structured = "structured_outputs" in params or "response_format" in params
        entry["supports_response_schema"] = structured
        entry.update(_modality_flags((row.get("architecture") or {}).get("input_modalities", [])))
        entry["supported_endpoints"] = [DOTHOOSH_ENDPOINTS[name] for name in protocols]
        entry["source"] = DOTHOOSH_API_URL

        transformed[f"{DOTHOOSH_PROVIDER}/{model_id}"] = entry
    return transformed


def drop_retired_dothoosh_entries(local_data: dict, dothoosh_data: dict) -> list:
    """Remove `dothoosh/*` entries the gateway no longer lists.

    Does nothing for a short list, which is more likely a truncated answer than a mass retirement.
    """
    if len(dothoosh_data) < DOTHOOSH_MIN_MODELS:
        return []
    prefix = f"{DOTHOOSH_PROVIDER}/"
    retired = [k for k in local_data if k.startswith(prefix) and k not in dothoosh_data]
    for key in retired:
        del local_data[key]
    return retired


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

# Write data to the json file
def write_to_file(file_path, data):
    try:
        # Open the file in write mode
        with open(file_path, "w") as file:
            # Dump the data as JSON into the file
            json.dump(data, file, indent=4)
        print("Values updated successfully.")
    except Exception as e:
        # Print an error message if writing to file fails
        print("Error updating JSON file:", e)

# Update the existing models and add the missing models for OpenRouter
def transform_openrouter_data(data):
    transformed = {}
    if not data:
        return transformed
    for row in data:
        # Add the fields 'max_tokens' and 'input_cost_per_token'
        obj = {
            "max_tokens": row["context_length"],
            "input_cost_per_token": float(row["pricing"]["prompt"]),
        }

        # Add 'max_output_tokens' as a field if it is not None
        if "top_provider" in row and "max_completion_tokens" in row["top_provider"] and row["top_provider"]["max_completion_tokens"] is not None:
            obj['max_output_tokens'] = int(row["top_provider"]["max_completion_tokens"])

        # Add the field 'output_cost_per_token'
        obj.update({
            "output_cost_per_token": float(row["pricing"]["completion"]),
        })

        # Add field 'input_cost_per_image' if it exists and is non-zero
        if "pricing" in row and "image" in row["pricing"] and float(row["pricing"]["image"]) != 0.0:
            obj['input_cost_per_image'] = float(row["pricing"]["image"])

        # Add the fields 'litellm_provider' and 'mode'
        obj.update({
            "litellm_provider": "openrouter",
            "mode": "chat"
        })

        # Add the 'supports_vision' field if the modality is 'multimodal'
        if row.get('architecture', {}).get('modality') == 'multimodal':
            obj['supports_vision'] = True

        # Use a composite key to store the transformed object
        transformed[f'openrouter/{row["id"]}'] = obj

    return transformed

# Update the existing models and add the missing models for Vercel AI Gateway
def transform_vercel_ai_gateway_data(data):
    transformed = {}
    if not data:
        return transformed
    for row in data:
        # Rows without token pricing or token limits (video/embedding models) previously KeyError'd the whole sync
        if any(row.get(k) is None for k in ("context_window", "max_tokens")) or any(
            row.get("pricing", {}).get(k) is None for k in ("input", "output")
        ):
            continue
        obj = {
            "max_tokens": row["context_window"],
            "input_cost_per_token": float(row["pricing"]["input"]),
            "output_cost_per_token": float(row["pricing"]["output"]),
            'max_output_tokens': row['max_tokens'],
            'max_input_tokens': row["context_window"],
        }

        # Handle cache pricing if available
        if "pricing" in row:
            if "input_cache_read" in row["pricing"] and row["pricing"]["input_cache_read"] is not None:
                obj['cache_read_input_token_cost'] = float(f"{float(row['pricing']['input_cache_read']):e}")
            
            if "input_cache_write" in row["pricing"] and row["pricing"]["input_cache_write"] is not None:
                obj['cache_creation_input_token_cost'] = float(f"{float(row['pricing']['input_cache_write']):e}")

        mode = "embedding" if "embedding" in row["id"].lower() else "chat"
        
        obj.update({"litellm_provider": "vercel_ai_gateway", "mode": mode})

        transformed[f'vercel_ai_gateway/{row["id"]}'] = obj

    return transformed


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

def main():
    local_file_path = "model_prices_and_context_window.json"  # Path to the local data file
    openrouter_url = "https://openrouter.ai/api/v1/models"  # URL to fetch OpenRouter data
    vercel_ai_gateway_url = "https://ai-gateway.vercel.sh/v1/models"  # URL to fetch Vercel AI Gateway data

    # Load local data from file
    local_data = load_local_data(local_file_path)
    
    # Fetch OpenRouter data
    openrouter_data = asyncio.run(fetch_data(openrouter_url))
    # Transform the fetched OpenRouter data
    openrouter_data = transform_openrouter_data(openrouter_data)
    
    # Fetch Vercel AI Gateway data
    vercel_data = asyncio.run(fetch_data(vercel_ai_gateway_url))
    # Transform the fetched Vercel AI Gateway data
    vercel_data = transform_vercel_ai_gateway_data(vercel_data)

    friendli_data = asyncio.run(fetch_data(FRIENDLI_API_URL))
    friendli_data = transform_friendli_data(friendli_data, local_data)
    
    dothoosh_data = asyncio.run(fetch_data(DOTHOOSH_API_URL))
    dothoosh_data = transform_dothoosh_data(dothoosh_data)

    # Combine all datasets
    all_remote_data = {**openrouter_data, **vercel_data, **friendli_data, **dothoosh_data}

    # If both local and openrouter data are available, synchronize and save
    if local_data and all_remote_data:
        sync_local_data_with_remote(
            local_data, all_remote_data, replace_keys=frozenset(friendli_data) | frozenset(dothoosh_data)
        )
        for key in drop_retired_dothoosh_entries(local_data, dothoosh_data):
            print("Removed retired DotHoosh entry:", key)
        write_to_file(local_file_path, local_data)
    else:
        print("Failed to fetch model data from either local file or URL.")

# Entry point of the script
if __name__ == "__main__":
    main()
