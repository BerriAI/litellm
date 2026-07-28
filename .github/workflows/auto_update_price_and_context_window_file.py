import asyncio
import os
import re
import aiohttp
import json
from pydantic import BaseModel


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
                return resp_json["data"]
    except Exception as e:
        # Print an error message if fetching data fails
        print("Error fetching data from URL:", e)
        return None


# Synchronize local data with remote data
def sync_local_data_with_remote(local_data, remote_data):
    # Update existing keys in local_data with values from remote_data
    for key in set(local_data) & set(remote_data):
        local_data[key].update(remote_data[key])

    # Add new keys from remote_data to local_data
    for key in set(remote_data) - set(local_data):
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
        if (
            "top_provider" in row
            and "max_completion_tokens" in row["top_provider"]
            and row["top_provider"]["max_completion_tokens"] is not None
        ):
            obj["max_output_tokens"] = int(row["top_provider"]["max_completion_tokens"])

        # Add the field 'output_cost_per_token'
        obj.update(
            {
                "output_cost_per_token": float(row["pricing"]["completion"]),
            }
        )

        # Add field 'input_cost_per_image' if it exists and is non-zero
        if (
            "pricing" in row
            and "image" in row["pricing"]
            and float(row["pricing"]["image"]) != 0.0
        ):
            obj["input_cost_per_image"] = float(row["pricing"]["image"])

        # Add the fields 'litellm_provider' and 'mode'
        obj.update({"litellm_provider": "openrouter", "mode": "chat"})

        # Add the 'supports_vision' field if the modality is 'multimodal'
        if row.get("architecture", {}).get("modality") == "multimodal":
            obj["supports_vision"] = True

        # Use a composite key to store the transformed object
        transformed[f"openrouter/{row['id']}"] = obj

    return transformed


# Update the existing models and add the missing models for Vercel AI Gateway
def transform_vercel_ai_gateway_data(data):
    transformed = {}
    for row in data:
        obj = {
            "max_tokens": row["context_window"],
            "input_cost_per_token": float(row["pricing"]["input"]),
            "output_cost_per_token": float(row["pricing"]["output"]),
            "max_output_tokens": row["max_tokens"],
            "max_input_tokens": row["context_window"],
        }

        # Handle cache pricing if available
        if "pricing" in row:
            if (
                "input_cache_read" in row["pricing"]
                and row["pricing"]["input_cache_read"] is not None
            ):
                obj["cache_read_input_token_cost"] = float(
                    f"{float(row['pricing']['input_cache_read']):e}"
                )

            if (
                "input_cache_write" in row["pricing"]
                and row["pricing"]["input_cache_write"] is not None
            ):
                obj["cache_creation_input_token_cost"] = float(
                    f"{float(row['pricing']['input_cache_write']):e}"
                )

        mode = "embedding" if "embedding" in row["id"].lower() else "chat"

        obj.update({"litellm_provider": "vercel_ai_gateway", "mode": mode})

        transformed[f"vercel_ai_gateway/{row['id']}"] = obj

    return transformed


class _SupportFlag(BaseModel):
    supported: bool = False


class _ThinkingTypes(BaseModel):
    adaptive: _SupportFlag = _SupportFlag()


class _Thinking(BaseModel):
    supported: bool = False
    types: _ThinkingTypes = _ThinkingTypes()


class _Effort(BaseModel):
    supported: bool = False
    xhigh: _SupportFlag = _SupportFlag()
    max: _SupportFlag = _SupportFlag()


class _Capabilities(BaseModel):
    image_input: _SupportFlag = _SupportFlag()
    pdf_input: _SupportFlag = _SupportFlag()
    structured_outputs: _SupportFlag = _SupportFlag()
    thinking: _Thinking = _Thinking()
    effort: _Effort = _Effort()


class AnthropicModel(BaseModel):
    id: str
    max_input_tokens: int | None = None
    max_tokens: int | None = None
    capabilities: _Capabilities = _Capabilities()


class OpenRouterPricing(BaseModel):
    prompt: float
    completion: float
    input_cache_read: float | None = None
    input_cache_write: float | None = None


class OpenRouterModel(BaseModel):
    id: str
    pricing: OpenRouterPricing


class AnthropicEntry(BaseModel):
    litellm_provider: str
    mode: str
    max_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_input_token_cost: float | None = None
    cache_creation_input_token_cost: float | None = None
    supports_vision: bool | None = None
    supports_pdf_input: bool | None = None
    supports_response_schema: bool | None = None
    supports_reasoning: bool | None = None
    supports_adaptive_thinking: bool | None = None
    supports_xhigh_reasoning_effort: bool | None = None
    supports_max_reasoning_effort: bool | None = None
    supports_function_calling: bool | None = None
    supports_tool_choice: bool | None = None
    supports_prompt_caching: bool | None = None


# Normalize a provider model id to a comparable key (drop prefix/suffix, dates, dot vs dash)
def canonical_model_id(model_id):
    base = model_id.split("/")[-1].split(":")[0].lower()
    base = re.sub(r"-\d{8}$", "", base)
    return base.replace(".", "-")


# Build a price lookup from OpenRouter's Anthropic models, preferring base ids over ":variant" ids
def build_anthropic_price_index(openrouter_rows):
    anthropic_rows = (
        OpenRouterModel.model_validate(row)
        for row in openrouter_rows
        if str(row.get("id", "")).startswith("anthropic/")
    )
    return {
        canonical_model_id(model.id): model.pricing
        for model in sorted(anthropic_rows, key=lambda model: ":" not in model.id)
    }


# Build native Anthropic entries for models that are new and have a known price
def transform_anthropic_data(anthropic_rows, price_index, existing_keys):
    entries = {}
    for row in anthropic_rows:
        model = AnthropicModel.model_validate(row)
        if model.id in existing_keys:
            continue
        pricing = price_index.get(canonical_model_id(model.id))
        if pricing is None:
            print(f"Skipping {model.id}: no OpenRouter price match")
            continue
        caps = model.capabilities
        entry = AnthropicEntry(
            litellm_provider="anthropic",
            mode="chat",
            max_tokens=model.max_tokens,
            max_input_tokens=model.max_input_tokens,
            max_output_tokens=model.max_tokens,
            input_cost_per_token=pricing.prompt,
            output_cost_per_token=pricing.completion,
            cache_read_input_token_cost=pricing.input_cache_read,
            cache_creation_input_token_cost=pricing.input_cache_write,
            supports_vision=caps.image_input.supported or None,
            supports_pdf_input=caps.pdf_input.supported or None,
            supports_response_schema=caps.structured_outputs.supported or None,
            supports_reasoning=caps.thinking.supported or None,
            supports_adaptive_thinking=caps.thinking.types.adaptive.supported or None,
            supports_xhigh_reasoning_effort=caps.effort.xhigh.supported or None,
            supports_max_reasoning_effort=caps.effort.max.supported or None,
            supports_function_calling=True,
            supports_tool_choice=True,
            supports_prompt_caching=(pricing.input_cache_read is not None) or None,
        )
        entries[model.id] = entry.model_dump(exclude_none=True)
    return entries


# Fetch the Anthropic models list (authoritative for existence, context window, capabilities)
async def fetch_anthropic_models(api_key):
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    url = "https://api.anthropic.com/v1/models?limit=1000"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                payload = await resp.json()
                if payload.get("has_more"):
                    print(
                        "Warning: Anthropic models response truncated; pagination not implemented"
                    )
                return payload["data"]
    except Exception as e:
        print("Error fetching Anthropic models:", e)
        return None


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
    local_file_path = (
        "model_prices_and_context_window.json"  # Path to the local data file
    )
    openrouter_url = (
        "https://openrouter.ai/api/v1/models"  # URL to fetch OpenRouter data
    )
    vercel_ai_gateway_url = (
        "https://ai-gateway.vercel.sh/v1/models"  # URL to fetch Vercel AI Gateway data
    )

    local_data = load_local_data(local_file_path)
    if not local_data:
        print("Failed to load local model data.")
        return
    existing_keys = frozenset(local_data)

    openrouter_rows = asyncio.run(fetch_data(openrouter_url))
    vercel_rows = asyncio.run(fetch_data(vercel_ai_gateway_url))

    remote_data = {}
    if openrouter_rows:
        remote_data.update(transform_openrouter_data(openrouter_rows))
    if vercel_rows:
        remote_data.update(transform_vercel_ai_gateway_data(vercel_rows))

    if not remote_data:
        print("Failed to fetch model data from remote URLs.")
        return

    sync_local_data_with_remote(local_data, remote_data)

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key and openrouter_rows:
        anthropic_rows = asyncio.run(fetch_anthropic_models(anthropic_api_key))
        if anthropic_rows:
            price_index = build_anthropic_price_index(openrouter_rows)
            for key, entry in transform_anthropic_data(
                anthropic_rows, price_index, existing_keys
            ).items():
                local_data.setdefault(key, entry)
    elif not anthropic_api_key:
        print("ANTHROPIC_API_KEY not set; skipping native Anthropic source.")

    write_to_file(local_file_path, local_data)


# Entry point of the script
if __name__ == "__main__":
    main()
