"""
JSON-based provider configuration loader for OpenAI-compatible providers.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from litellm._logging import verbose_logger


class SimpleProviderConfig:
    """Simple data class for JSON provider config"""

    def __init__(self, slug: str, data: dict):
        self.slug = slug
        self.base_url = data["base_url"]
        self.api_key_env = data["api_key_env"]
        self.api_base_env = data.get("api_base_env")
        self.base_class = data.get("base_class", "openai_gpt")
        self.param_mappings = data.get("param_mappings", {})
        self.constraints = data.get("constraints", {})
        self.special_handling = data.get("special_handling", {})


class JSONProviderRegistry:
    """Load providers from JSON once on import"""

    _providers: Dict[str, SimpleProviderConfig] = {}
    _loaded = False

    @classmethod
    def load(cls):
        """Load providers from JSON configuration file"""
        if cls._loaded:
            return

        json_path = Path(__file__).parent / "providers.json"
        
        if not json_path.exists():
            # No JSON file yet, that's okay
            cls._loaded = True
            return

        try:
            with open(json_path) as f:
                data = json.load(f)

            for slug, config in data.items():
                cls._providers[slug] = SimpleProviderConfig(slug, config)

            cls._loaded = True
        except Exception as e:
            verbose_logger.warning(f"Warning: Failed to load JSON provider configs: {e}")
            cls._loaded = True

    @classmethod
    def get(cls, slug: str) -> Optional[SimpleProviderConfig]:
        """Get a provider configuration by slug"""
        return cls._providers.get(slug)

    @classmethod
    def exists(cls, slug: str) -> bool:
        """Check if a provider is defined via JSON"""
        return slug in cls._providers

    @classmethod
    def list_providers(cls) -> list:
        """List all registered provider slugs"""
        return list(cls._providers.keys())


# Load on import
JSONProviderRegistry.load()
