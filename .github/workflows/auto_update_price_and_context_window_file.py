import asyncio
import aiohttp
import json
import re

# Asynchronously fetch data from a given URL
async def fetch_data(url, result_key='data'):
    try:
        # Create an asynchronous session
        async with aiohttp.ClientSession() as session:
            # Send a GET request to the URL
            async with session.get(url) as resp:
                # Raise an error if the response status is not OK
                resp.raise_for_status()
                # Parse the response JSON
                resp_json = await resp.json()
                print(f"Fetched data from {url}")
                if result_key:
                    return resp_json[result_key]
                return resp_json
    except Exception as e:
        # Print an error message if fetching data fails
        print("Error fetching data from URL:", e)
        return None

_COST_FIELD_PATTERNS = (
    'input_cost_per_', 'output_cost_per_', 'cache_read_', 'cache_creation_',
)


def _is_cost_field(field_name):
    return any(field_name.startswith(p) for p in _COST_FIELD_PATTERNS)


# Synchronize local data with remote data
def sync_local_data_with_remote(local_data, remote_data):
    # Update existing keys: merge remote fields into local, preserving any
    # manually set fields that the remote API doesn't return (e.g.
    # supports_assistant_prefill, tool_use_system_prompt_tokens, etc.)
    #
    # For NIM cost fields: skip updates when the remote value is 0.0 and
    # the local value is non-zero. This prevents weekly runs from resetting
    # manually-curated NIM prices, since the NIM API returns 0.0 for all
    # costs by default. We scope this to NIM only because:
    #   1) NIM is the only provider where 0.0 is the API default, and
    #   2) other providers might legitimately set a price to 0.0, which
    #      should be respected (e.g., a model becoming free).
    for key in (set(local_data) & set(remote_data)):
        local_entry = local_data[key]
        remote_entry = remote_data[key]
        is_nim = local_entry.get("litellm_provider") == "nvidia_nim"

        # For NIM entries, remove cost fields that are no longer present in
        # the remote schema (e.g. stale input_cost_per_token on rerank models).
        if is_nim:
            stale_cost_fields = [f for f in list(local_entry) if _is_cost_field(f) and f not in remote_entry]
            for f in stale_cost_fields:
                del local_entry[f]

        for field, remote_val in remote_entry.items():
            if (
                is_nim
                and _is_cost_field(field)
                and remote_val == 0.0
                and isinstance(local_entry.get(field), (int, float))
                and local_entry[field] != 0.0
            ):
                continue  # preserve manually-curated non-zero NIM price
            local_entry[field] = remote_val

    # Add new keys from remote_data to local_data
    for key in (set(remote_data) - set(local_data)):
        local_data[key] = remote_data[key]

# Write data to the json file
def write_to_file(file_path, data):
    try:
        # Open the file in write mode
        with open(file_path, "w") as file:
            # Dump the data as JSON into the file, preserving trailing newline
            json.dump(data, file, indent=4)
            file.write("\n")
        print("Values updated successfully.")
    except Exception as e:
        # Print an error message if writing to file fails
        print("Error updating JSON file:", e)

# Update the existing models and add the missing models for OpenRouter
def transform_openrouter_data(data):
    transformed = {}
    for row in data:
        # context_length is the input context window, not the output limit.
        # max_completion_tokens (when present) is the per-request output cap;
        # that maps to both max_output_tokens and max_tokens in litellm's schema.
        obj = {
            "max_input_tokens": row["context_length"],
            "input_cost_per_token": float(row["pricing"]["prompt"]),
        }

        # Add max_output_tokens / max_tokens from top_provider if available
        max_completion = (
            row.get("top_provider", {}) or {}
        ).get("max_completion_tokens")
        if max_completion is not None:
            obj['max_output_tokens'] = int(max_completion)
            obj['max_tokens'] = int(max_completion)

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

        # Add the 'supports_vision' field if image is in input modalities
        architecture = row.get('architecture') or {}
        input_modalities = architecture.get('input_modalities') or []
        if 'image' in input_modalities:
            obj['supports_vision'] = True

        if 'audio' in input_modalities:
            obj['supports_audio_input'] = True

        # Use a composite key to store the transformed object
        transformed[f'openrouter/{row["id"]}'] = obj

    return transformed


# Detect mode for NVIDIA NIM models based on name patterns
_NIM_EMBED_PATTERNS = ['embed', 'bge-m3', 'arctic-embed', 'nemoretriever', 'nvclip']
_NIM_RERANK_PATTERNS = ['rerank']
_NIM_VLM_PATTERNS = ['vision', 'vl-', '-vl/', 'vila', 'paligemma', 'deplot', 'kosmos', 'multimodal', 'fuyu', 'neva']
_NIM_REASONING_PATTERNS = ['thinking', 'qwq', 'flash-reasoning', 'magistral']


def _nim_detect_mode(model_id):
    mid = model_id.lower()
    for p in _NIM_RERANK_PATTERNS:
        if p in mid:
            return 'rerank'
    for p in _NIM_EMBED_PATTERNS:
        if p in mid:
            return 'embedding'
    return 'chat'


def _nim_detect_vision(model_id):
    mid = model_id.lower()
    return any(p in mid for p in _NIM_VLM_PATTERNS)


def _nim_detect_reasoning(model_id):
    mid = model_id.lower()
    if '/deepseek-r1' in mid or '-r1-' in mid or mid.endswith('-r1'):
        return True
    return any(p in mid for p in _NIM_REASONING_PATTERNS)


def _nim_detect_context(model_id):
    m = re.search(r'[-_](\d+)k(?:[-_]|$)', model_id.lower())
    if m:
        return int(m.group(1)) * 1024
    return None


# Update the existing models and add missing models for NVIDIA NIM
def transform_nvidia_nim_data(data):
    transformed = {}
    seen = set()
    for row in data:
        mid = row['id']
        if mid in seen:
            continue
        seen.add(mid)

        mode = _nim_detect_mode(mid)
        obj = {
            'litellm_provider': 'nvidia_nim',
            'mode': mode,
        }

        if mode == 'chat':
            obj['input_cost_per_token'] = 0.0
            obj['output_cost_per_token'] = 0.0
            ctx = _nim_detect_context(mid)
            if ctx:
                obj['max_input_tokens'] = ctx
                obj['max_tokens'] = ctx
            if _nim_detect_vision(mid):
                obj['supports_vision'] = True
            if _nim_detect_reasoning(mid):
                obj['supports_reasoning'] = True
            # Only flag explicit instruction-tuned variants; base model family
            # names (granite, falcon, codellama, starcoder, …) do not reliably
            # support function calling without an instruction-tuning suffix.
            _instruct_indicators = ['-instruct', '-chat', '-it', '-tool', '_instruct', '_chat']
            if any(mid.lower().endswith(ind) or (ind + '-') in mid.lower() or (ind + '_') in mid.lower() for ind in _instruct_indicators):
                obj['supports_function_calling'] = True
                obj['supports_tool_choice'] = True
        elif mode == 'rerank':
            # Rerank models are priced per-query, not per-token.
            # Do not add input_cost_per_token to avoid misapplied token-based costs.
            obj['input_cost_per_query'] = 0.0
        elif mode == 'embedding':
            obj['input_cost_per_token'] = 0.0

        transformed[f'nvidia_nim/{mid}'] = obj

    return transformed

# Update the existing models and add the missing models for Vercel AI Gateway
def transform_vercel_ai_gateway_data(data):
    transformed = {}
    for row in data:
        pricing = row.get("pricing") or {}  # guard against explicit null
        model_id = row["id"]
        context_window = row.get("context_window", 0)
        max_output = row.get("max_tokens", context_window)

        # Detect mode from model type field or name patterns
        model_type = row.get("type", "")
        mid = model_id.lower()
        if model_type == "embedding" or "embed" in mid:
            mode = "embedding"
        elif model_type == "image" or any(x in mid for x in ["imagen", "flux", "recraft", "dall-e", "imagine"]):
            mode = "image_generation"
        elif model_type == "video" or any(x in mid for x in ["veo", "wan-v", "seedance", "kling", "sora"]):
            mode = "video_generation"
        else:
            mode = "chat"

        obj = {"litellm_provider": "vercel_ai_gateway", "mode": mode}

        if context_window:
            obj["max_input_tokens"] = context_window
        if max_output:
            obj["max_tokens"] = max_output
            obj["max_output_tokens"] = max_output

        # Token pricing (chat / embedding models only — not image/video generation)
        if mode not in ("image_generation", "video_generation"):
            if pricing.get("input") is not None:
                obj["input_cost_per_token"] = float(pricing["input"])
            if pricing.get("output") is not None:
                obj["output_cost_per_token"] = float(pricing["output"])

        # Per-image pricing
        if pricing.get("image") is not None:
            obj["input_cost_per_image"] = float(pricing["image"])

        # Cache pricing
        if pricing.get("input_cache_read") is not None:
            obj["cache_read_input_token_cost"] = float(f"{float(pricing['input_cache_read']):e}")
        if pricing.get("input_cache_write") is not None:
            obj["cache_creation_input_token_cost"] = float(f"{float(pricing['input_cache_write']):e}")

        transformed[f'vercel_ai_gateway/{model_id}'] = obj

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
    openrouter_url = "https://openrouter.ai/api/v1/models"
    vercel_ai_gateway_url = "https://ai-gateway.vercel.sh/v1/models"
    nvidia_nim_url = "https://integrate.api.nvidia.com/v1/models"

    # Load local data from file
    local_data = load_local_data(local_file_path)

    all_remote_data = {}

    # Fetch and transform OpenRouter data
    openrouter_raw = asyncio.run(fetch_data(openrouter_url, result_key='data'))
    if openrouter_raw:
        all_remote_data.update(transform_openrouter_data(openrouter_raw))
        print(f"OpenRouter: {len(openrouter_raw)} models")
    else:
        print("WARNING: Failed to fetch OpenRouter data")

    # Fetch and transform Vercel AI Gateway data
    vercel_raw = asyncio.run(fetch_data(vercel_ai_gateway_url, result_key='data'))
    if vercel_raw:
        all_remote_data.update(transform_vercel_ai_gateway_data(vercel_raw))
        print(f"Vercel AI Gateway: {len(vercel_raw)} models")
    else:
        print("WARNING: Failed to fetch Vercel AI Gateway data")

    # Fetch and transform NVIDIA NIM data
    nvidia_nim_raw = asyncio.run(fetch_data(nvidia_nim_url, result_key='data'))
    if nvidia_nim_raw:
        all_remote_data.update(transform_nvidia_nim_data(nvidia_nim_raw))
        print(f"NVIDIA NIM: {len(nvidia_nim_raw)} models")
    else:
        print("WARNING: Failed to fetch NVIDIA NIM data")

    print(f"Total remote entries to sync: {len(all_remote_data)}")

    if local_data and all_remote_data:
        sync_local_data_with_remote(local_data, all_remote_data)
        write_to_file(local_file_path, local_data)
    else:
        print("Failed to fetch model data from either local file or URL.")

# Entry point of the script
if __name__ == "__main__":
    main()
