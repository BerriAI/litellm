import litellm
import json

data = {
    "custom_providers": getattr(litellm, "custom_providers", "MISSING"),
    "_custom_providers": getattr(litellm, "_custom_providers", "MISSING"),
    "openai_compatible_providers": getattr(litellm, "openai_compatible_providers", "MISSING"),
    "_openai_like_providers": getattr(litellm, "_openai_like_providers", "MISSING"),
}

# If they are lists, check if publicai is in them
if isinstance(data["custom_providers"], list):
    data["publicai_in_custom_providers"] = "publicai" in data["custom_providers"]
if isinstance(data["_custom_providers"], list):
    data["publicai_in__custom_providers"] = "publicai" in data["_custom_providers"]
if isinstance(data["openai_compatible_providers"], list):
    data["publicai_in_openai_compatible_providers"] = "publicai" in data["openai_compatible_providers"]
if isinstance(data["_openai_like_providers"], list):
    data["publicai_in__openai_like_providers"] = "publicai" in data["_openai_like_providers"]

print(json.dumps(data, indent=2))
