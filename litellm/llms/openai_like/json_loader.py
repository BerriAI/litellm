"""
JSON-based provider configuration loader for OpenAI-compatible providers.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import httpx

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
    def _load_from_dict(cls, data: dict, source: str = "unknown"):
        """Load providers from a dictionary"""
        for slug, config in data.items():
            if slug in cls._providers:
                verbose_logger.debug(
                    f"Provider '{slug}' from {source} overwrites existing definition"
                )
            cls._providers[slug] = SimpleProviderConfig(slug, config)

    @classmethod
    def _load_from_json_string(cls, json_string: str):
        """Load providers from a JSON string"""
        try:
            verbose_logger.debug("Attempting to load custom providers from JSON string")
            data = json.loads(json_string)
            cls._load_from_dict(data, source="LITELLM_CUSTOM_PROVIDERS env var")
            verbose_logger.info(
                f"Successfully loaded {len(data)} custom provider(s) from JSON string"
            )
        except json.JSONDecodeError as e:
            verbose_logger.warning(
                f"Failed to parse custom providers JSON string: {e}"
            )
        except Exception as e:
            verbose_logger.warning(
                f"Failed to load custom providers from JSON string: {e}"
            )

    @classmethod
    def _load_from_url(cls, url: str):
        """Load providers from a URL"""
        try:
            verbose_logger.debug(f"Attempting to load custom providers from URL: {url}")
            
            # Basic URL validation to prevent SSRF
            # Allow http/https schemes only
            if not url.startswith(("http://", "https://")):
                verbose_logger.warning(
                    f"Invalid URL scheme for custom providers: {url}. Only http:// and https:// are allowed."
                )
                return
            
            # Use httpx to fetch the JSON from URL
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
            
            cls._load_from_dict(data, source=f"URL: {url}")
            verbose_logger.info(
                f"Successfully loaded {len(data)} custom provider(s) from {url}"
            )
        except httpx.HTTPError as e:
            verbose_logger.warning(
                f"Failed to load custom providers from URL {url}: HTTP error - {e}"
            )
        except json.JSONDecodeError as e:
            verbose_logger.warning(
                f"Failed to parse custom providers JSON from URL {url}: {e}"
            )
        except Exception as e:
            verbose_logger.warning(
                f"Failed to load custom providers from URL {url}: {e}"
            )

    @classmethod
    def load(cls):
        """Load providers from JSON configuration file and optionally from environment variables"""
        if cls._loaded:
            return

        # Load local providers.json file first
        json_path = Path(__file__).parent / "providers.json"
        
        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)
                cls._load_from_dict(data, source="local providers.json")
            except Exception as e:
                verbose_logger.warning(
                    f"Warning: Failed to load local JSON provider configs: {e}"
                )

        # Load custom providers from JSON string if specified
        custom_providers_json = os.environ.get("LITELLM_CUSTOM_PROVIDERS")
        if custom_providers_json:
            cls._load_from_json_string(custom_providers_json)

        # Load custom providers from URL if specified
        custom_providers_url = os.environ.get("LITELLM_CUSTOM_PROVIDERS_URL")
        if custom_providers_url:
            cls._load_from_url(custom_providers_url)

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
