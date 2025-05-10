"""
Constants for Copilot integration
"""
# Copilot API endpoints
GITHUB_COPILOT_API_BASE = "https://api.github.com/copilot/v1"
CHAT_COMPLETION_ENDPOINT = "/chat/completions"

# Model names
GITHUB_COPILOT_MODEL = "gpt-4o"  # The model identifier for Copilot

# Request headers
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "litellm",
}


class GetDeviceCodeError(Exception):
    pass


class GetAccessTokenError(Exception):
    pass


class APIKeyExpiredError(Exception):
    pass


class RefreshAPIKeyError(Exception):
    pass


class GetAPIKeyError(Exception):
    pass
