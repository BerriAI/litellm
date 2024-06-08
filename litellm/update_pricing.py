import requests
import json
import os

# Fetch data from OpenRouter API
api_url = "https://openrouter.ai/api/v1/models"
response = requests.get(api_url)
openrouter_data = response.json()

def translate_openrouter_to_model_cost_map(openrouter_data):
    model_cost_map = {}
    for model in openrouter_data:
        model_id = model['id']
        model_cost_map[model_id] = {
            "max_tokens": model.get("context_length", 4096),
            "input_cost_per_token": float(model['pricing'].get('prompt', 0)),
            "output_cost_per_token": float(model['pricing'].get('completion', 0)),
            "litellm_provider": "openrouter",
            "mode": "chat"  # assuming mode as chat for this example
        }
        if model['architecture']['modality'] == 'multimodal':
            model_cost_map[model_id]["supports_vision"] = True
    return model_cost_map

# Load the existing JSON file
with open('model_prices_and_context_window.json', 'r') as file:
    model_cost_map = json.load(file)

# Update the JSON file with new data
new_model_cost_map = translate_openrouter_to_model_cost_map(openrouter_data)
model_cost_map.update(new_model_cost_map)

# Save the updated JSON file
with open('model_prices_and_context_window.json', 'w') as file:
    json.dump(model_cost_map, file, indent=4)

# Commit and push changes
os.system("git add model_prices_and_context_window.json")
os.system("git commit -m 'Auto-update model pricing from OpenRouter'")
os.system("git push origin update-pricing")

# Create a PR (assuming you have gh CLI installed and authenticated)
os.system("gh pr create --title 'Auto-update model pricing from OpenRouter' --body 'This PR updates the model pricing based on data fetched from OpenRouter API.'")
