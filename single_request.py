import json
import requests

# Load the request payload from bad_request.json
with open("bad_request.json", "r") as f:
    payload = json.load(f)

# Update the model name to match config.yaml
payload["model"] = "db-openai-endpoint"

# Set up the API key from the load test file
api_key = "sk-1234"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Make the request (assuming LiteLLM proxy is running on localhost:4000)
# Update the URL if your proxy is running on a different host/port
url = "http://localhost:4000/chat/completions"

print("Making request...")
print(f"Model: {payload['model']}")
print(f"API Key: {api_key}")

response = requests.post(url, json=payload, headers=headers)

print(f"\nStatus Code: {response.status_code}")
print(f"\nResponse:")
print(response.text)

# Save the response to a file
with open("response.json", "w") as f:
    f.write(response.text)

print("\nResponse saved to response.json")

