import asyncio
import aiohttp
import json
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

INHERITABLE_BASE_KEYS = (
    "supports_reasoning",
    "supports_function_calling",
    "supports_parallel_function_calling",
    "supports_response_schema",
    "supports_system_messages",
    "supports_tool_choice",
    "supports_vision",
    "supports_pdf_input",
    "supports_prompt_caching",
    "supports_assistant_prefill",
    "supports_low_reasoning_effort",
    "supports_minimal_reasoning_effort",
    "supports_max_reasoning_effort",
    "supports_xhigh_reasoning_effort",
    "supports_none_reasoning_effort",
    "supports_adaptive_thinking",
    "supports_output_config",
    "supports_native_structured_output",
)

EFFORT_FLAG_MAP = {
    "none": "supports_none_reasoning_effort",
    "minimal": "supports_minimal_reasoning_effort",
    "low": "supports_low_reasoning_effort",
    "medium": "supports_low_reasoning_effort",
    "high": "supports_max_reasoning_effort",
    "xhigh": "supports_xhigh_reasoning_effort",
    "max": "supports_max_reasoning_effort",
}


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


def _effort_flags(reasoning_options: list) -> dict:
    flags: dict[str, bool] = {flag: False for flag in EFFORT_FLAG_MAP.values()}
    for opt in reasoning_options or []:
        if opt.get("type") == "effort":
            for val in opt.get("values", []):
                flag = EFFORT_FLAG_MAP.get(val)
                if flag:
                    flags[flag] = True
    return flags


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
    has_image = "image" in (input_mods or [])
    return {
        "supports_vision": has_image,
        "supports_image_input": has_image,
    }


def transform_friendli_data(data: list, local_data: dict) -> dict:
    transformed: dict[str, dict] = {}
    for model in data:
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
            entry["max_tokens"] = int(ctx)
        max_out = model.get("max_completion_tokens")
        if max_out is not None:
            entry["max_output_tokens"] = int(max_out)

        entry.update(_pricing(model.get("pricing", {})))

        reasoning = model.get("reasoning") is True
        entry["supports_reasoning"] = reasoning
        if reasoning:
            entry.update(_effort_flags(model.get("reasoning_options", [])))

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
def sync_local_data_with_remote(local_data, remote_data):
    # Update existing keys in local_data with values from remote_data
    for key in (set(local_data) & set(remote_data)):
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
    for row in data:
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
    
    # Combine both datasets
    all_remote_data = {**openrouter_data, **vercel_data, **friendli_data}

    # If both local and openrouter data are available, synchronize and save
    if local_data and all_remote_data:
        sync_local_data_with_remote(local_data, all_remote_data)
        write_to_file(local_file_path, local_data)
    else:
        print("Failed to fetch model data from either local file or URL.")

# Entry point of the script
if __name__ == "__main__":
    main()
