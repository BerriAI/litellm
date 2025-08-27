import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any

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

# Asynchronously fetch model endpoint information from OpenRouter
async def fetch_model_endpoints(session: aiohttp.ClientSession, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed model information from OpenRouter endpoints API.
    Args:
        session: The aiohttp session to use for requests
        model_name: The model name (e.g., "openai/gpt-4")
    Returns:
        Dict with model endpoint information, or None if the request fails.
    """
    endpoints_url = f"https://openrouter.ai/api/v1/models/{model_name}/endpoints"
    
    try:
        async with session.get(endpoints_url) as resp:
            resp.raise_for_status()
            resp_json = await resp.json()
            
            # Extract model data
            model_data = resp_json.get("data")
            if model_data is None:
                return None
                
            # Get the first endpoint (usually the primary one)
            endpoints = model_data.get("endpoints", [])
            if not endpoints:
                return None
                
            return endpoints[0]  # Return the primary endpoint data
    except Exception as e:
        print(f"Error fetching endpoint data for {model_name}: {e}")
        return None

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

# Update the existing models and add the missing models with comprehensive data
async def transform_remote_data(data):
    transformed = {}
    
    # Create a session for making endpoint requests
    async with aiohttp.ClientSession() as session:
        for i, row in enumerate(data):
            model_name = row["id"]
            print(f"Processing model {i+1}/{len(data)}: {model_name}")
            
            # Add a small delay to avoid overwhelming the API
            if i > 0 and i % 10 == 0:
                print("Pausing to avoid rate limits...")
                await asyncio.sleep(1)
            
            # Start with basic information from the models list
            obj = {
                "max_tokens": row["context_length"],
                "input_cost_per_token": float(row["pricing"]["prompt"]),
                "output_cost_per_token": float(row["pricing"]["completion"]),
                "litellm_provider": "openrouter",
                "mode": "chat"
            }

            # Add 'max_output_tokens' as a field if it is not None
            if "top_provider" in row and "max_completion_tokens" in row["top_provider"] and row["top_provider"]["max_completion_tokens"] is not None:
                obj['max_output_tokens'] = int(row["top_provider"]["max_completion_tokens"])

            # Add field 'input_cost_per_image' if it exists and is non-zero
            if "pricing" in row and "image" in row["pricing"] and float(row["pricing"]["image"]) != 0.0:
                obj['input_cost_per_image'] = float(row["pricing"]["image"])

            # Add the 'supports_vision' field if the modality is 'multimodal'
            if row.get('architecture', {}).get('modality') == 'multimodal':
                obj['supports_vision'] = True

            # Fetch detailed endpoint information
            endpoint_data = await fetch_model_endpoints(session, model_name)
            
            if endpoint_data:
                # Add comprehensive pricing information
                pricing_info = endpoint_data.get("pricing", {})
                
                # Add context length information from endpoint if available
                if endpoint_data.get("context_length"):
                    obj["max_tokens"] = endpoint_data["context_length"]

                if endpoint_data.get("supports_implicit_caching"):
                    obj["supports_prompt_caching"] = True 
                                  
                # Add max_input_tokens if available
                if endpoint_data.get("max_prompt_tokens"):
                    obj["max_input_tokens"] = endpoint_data["max_prompt_tokens"]
                
                # Add max_output_tokens from endpoint if available and not already set
                if endpoint_data.get("max_completion_tokens") and "max_output_tokens" not in obj:
                    obj["max_output_tokens"] = endpoint_data["max_completion_tokens"]
                
                # Add additional pricing fields
                if pricing_info.get("audio") and float(pricing_info["audio"]) != 0.0:
                    obj["input_cost_per_audio_token"] = float(pricing_info["audio"])
                
                if pricing_info.get("input_cache_read") and float(pricing_info["input_cache_read"]) != 0.0:
                    obj["cache_read_input_token_cost"] = float(pricing_info["input_cache_read"])
                
                if pricing_info.get("internal_reasoning") and float(pricing_info["internal_reasoning"]) != 0.0:
                    obj["output_cost_per_reasoning_token"] = float(pricing_info["internal_reasoning"])

                # Add supported parameters as capability flags
                supported_params = endpoint_data.get("supported_parameters", [])
                
                if "tools" in supported_params:
                    obj["supports_function_calling"] = True
                    # Check for parallel function calling support
                    if "parallel_tool_calls" in supported_params:
                        obj["supports_parallel_function_calling"] = True
                
                if "tool_choice" in supported_params:
                    obj["supports_tool_choice"] = True
                
                if "response_format" in supported_params or "structured_outputs" in supported_params:
                    obj["supports_response_schema"] = True
                
                if "web_search_options" in supported_params:
                    obj["supports_web_search"] = True
                
                if "reasoning" in supported_params or "include_reasoning" in supported_params:
                    obj["supports_reasoning"] = True
                
                if "system" in supported_params:
                    obj["supports_system_messages"] = True
                
                if "stream" in supported_params:
                    obj["supports_native_streaming"] = True
                
                # Check for prompt caching support
                if "cache_control" in supported_params or "prompt_caching" in supported_params:
                    obj["supports_prompt_caching"] = True

                # Check for PDF input support
                if "pdf" in supported_params or "document" in supported_params:
                    obj["supports_pdf_input"] = True
                
                # Add supported endpoints if available
                if endpoint_data.get("supported_endpoints"):
                    obj["supported_endpoints"] = endpoint_data["supported_endpoints"]
                
                # Add modalities information
                if endpoint_data.get("supported_modalities"):
                    obj["supported_modalities"] = endpoint_data["supported_modalities"]
                    # Set vision support based on modalities
                    if "image" in endpoint_data["supported_modalities"]:
                        obj["supports_vision"] = True
                
                if endpoint_data.get("supported_output_modalities"):
                    obj["supported_output_modalities"] = endpoint_data["supported_output_modalities"]

            # Add source URL for reference
            obj["source"] = f"https://openrouter.ai/{model_name}"

            # Use a composite key to store the transformed object
            transformed[f'openrouter/{model_name}'] = obj

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

async def main():
    local_file_path = "model_prices_and_context_window.json"  # Path to the local data file
    url = "https://openrouter.ai/api/v1/models"  # URL to fetch remote data

    # Load local data from file
    local_data = load_local_data(local_file_path)
    # Fetch remote data asynchronously
    remote_data = await fetch_data(url)
    # Transform the fetched remote data with comprehensive endpoint information
    if remote_data:
        remote_data = await transform_remote_data(remote_data)

    # If both local and remote data are available, synchronize and save
    if local_data and remote_data:
        sync_local_data_with_remote(local_data, remote_data)
        write_to_file(local_file_path, local_data)
    else:
        print("Failed to fetch model data from either local file or URL.")

# Entry point of the script
if __name__ == "__main__":
    asyncio.run(main())