import requests
import json

def fetch_data(url):
    """
    Fetches data from the specified URL.

    Args:
        url (str): The URL to fetch data from.

    Returns:
        dict or None: The JSON response if successful, None otherwise.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Error fetching data from URL:", e)
        return None

def update_local_data(local_data, remote_data):
    """
    Updates local data with information fetched remotely.

    Args:
        local_data (dict): Local data to be updated.
        remote_data (dict): Remote data fetched from an API.

    """
    for model_name, model_info in local_data.items():
        if model_name.startswith("openrouter/"):
            model_suffix = model_name[len("openrouter/"):]
            for model in remote_data["data"]:
                if model["id"] == model_suffix:
                    model_info["max_tokens"] = model["context_length"]
                    model_info["input_cost_per_token"] = model["pricing"]["prompt"]
                    model_info["output_cost_per_token"] = model["pricing"]["completion"]
                    break

def write_to_file(file_path, data):
    """
    Writes data to a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        data (dict): The data to write to the file.
    """
    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
        print("Values updated successfully.")
    except Exception as e:
        print("Error updating JSON file:", e)

def main():
    """
    Main function to orchestrate the process.
    """
    local_file_path = "model_prices_and_context_window.json"
    url = "https://openrouter.ai/api/v1/models"

    local_data = load_local_data(local_file_path)
    remote_data = fetch_data(url)

    if local_data and remote_data:
        update_local_data(local_data, remote_data)
        write_to_file(local_file_path, local_data)
    else:
        print("Failed to fetch model data from either local file or URL.")

def load_local_data(file_path):
    """
    Loads data from a local JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict or None: The loaded data if successful, None otherwise.
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("File not found:", file_path)
        return None
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None

if __name__ == "__main__":
    main()
