import os
import requests
from datetime import datetime

# GitHub API endpoints
GITHUB_API_URL = "https://api.github.com"
REPO_OWNER = "BerriAI"
REPO_NAME = "litellm"

# GitHub personal access token (required for uploading release assets)
GITHUB_ACCESS_TOKEN = os.environ.get("GITHUB_ACCESS_TOKEN")

# Headers for GitHub API requests
headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_ACCESS_TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
}

# Get the latest release
releases_url = f"{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
response = requests.get(releases_url, headers=headers)
latest_release = response.json()
print("Latest release:", latest_release)

# Upload an asset to the latest release
upload_url = latest_release["upload_url"].split("{?")[0]
asset_name = "results_stats.csv"
asset_path = os.path.join(os.getcwd(), asset_name)
print("upload_url:", upload_url)

with open(asset_path, "rb") as asset_file:
    asset_data = asset_file.read()

upload_payload = {
    "name": asset_name,
    "label": "Load test results",
    "created_at": datetime.utcnow().isoformat() + "Z",
}

upload_headers = headers.copy()
upload_headers["Content-Type"] = "application/octet-stream"

upload_response = requests.post(
    upload_url,
    headers=upload_headers,
    data=asset_data,
    params=upload_payload,
)

if upload_response.status_code == 201:
    print(f"Asset '{asset_name}' uploaded successfully to the latest release.")
else:
    print(f"Failed to upload asset. Response: {upload_response.text}")
