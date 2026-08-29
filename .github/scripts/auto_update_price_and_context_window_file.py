import asyncio
import json
from types import MappingProxyType
from typing import Final

import aiohttp
from pydantic import BaseModel

COST_MAP_PATHS: Final = (
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
)
OPENROUTER_URL: Final = "https://openrouter.ai/api/v1/models"
VERCEL_AI_GATEWAY_URL: Final = "https://ai-gateway.vercel.sh/v1/models"
VERCEL_TYPE_TO_MODE: Final = MappingProxyType({"language": "chat", "embedding": "embedding"})


class VercelPricing(BaseModel):
    input: float | None = None
    output: float | None = None
    input_cache_read: float | None = None
    input_cache_write: float | None = None


class VercelModel(BaseModel):
    id: str
    type: str
    context_window: int | None = None
    max_tokens: int | None = None
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

# Synchronize local data with remote data
def sync_local_data_with_remote(local_data, remote_data):
    # Update existing keys in local_data with values from remote_data
    for key in (set(local_data) & set(remote_data)):
        local_data[key].update(remote_data[key])

    # Add new keys from remote_data to local_data
    for key in (set(remote_data) - set(local_data)):
        local_data[key] = remote_data[key]

def write_to_file(file_path: str, data: dict[str, object]) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(json.dumps(data, indent=4, ensure_ascii=False) + "\n")
    print(f"Wrote {file_path}.")


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

def _vercel_entry(model: VercelModel) -> dict[str, object] | None:
    mode: Final = VERCEL_TYPE_TO_MODE[model.type]
    pricing: Final = model.pricing if model.pricing is not None else VercelPricing()
    if pricing.input is None or (mode == "chat" and pricing.output is None):
        print(f"Skipping vercel_ai_gateway/{model.id}: the catalog lists no per-token price for it.")
        return None
    candidate: Final = {
        "max_tokens": model.max_tokens,
        "max_input_tokens": model.context_window,
        "max_output_tokens": model.max_tokens,
        "input_cost_per_token": pricing.input,
        "output_cost_per_token": pricing.output if pricing.output is not None else 0.0,
        "cache_read_input_token_cost": pricing.input_cache_read,
        "cache_creation_input_token_cost": pricing.input_cache_write,
        "litellm_provider": "vercel_ai_gateway",
        "mode": mode,
    }
    return {key: value for key, value in candidate.items() if value is not None}


def transform_vercel_ai_gateway_data(data: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    models: Final = tuple(VercelModel.model_validate(row) for row in data if row.get("type") in VERCEL_TYPE_TO_MODE)
    entries: Final = ((f"vercel_ai_gateway/{model.id}", _vercel_entry(model)) for model in models)
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

def main():
    local_data = load_local_data(COST_MAP_PATHS[0])

    openrouter_data = transform_openrouter_data(asyncio.run(fetch_data(OPENROUTER_URL)))
    vercel_data = transform_vercel_ai_gateway_data(asyncio.run(fetch_data(VERCEL_AI_GATEWAY_URL)))
    all_remote_data = {**openrouter_data, **vercel_data}

    if local_data and all_remote_data:
        sync_local_data_with_remote(local_data, all_remote_data)
        for path in COST_MAP_PATHS:
            write_to_file(path, local_data)
    else:
        print("Failed to fetch model data from either local file or URL.")

# Entry point of the script
if __name__ == "__main__":
    main()
