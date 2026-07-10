"""Configuration for the LLM Classifier Router."""

from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMClassifierRouterConfig(BaseModel):
    classifier_model: str = "ollama/qwen2.5:0.5b"

    tiers: Dict[str, str] = Field(default_factory=lambda: {"SIMPLE": "gpt-4o-mini", "COMPLEX": "gpt-4o"})

    classifier_system_prompt: Optional[str] = None

    classifier_timeout: float = 3.0
    classifier_temperature: float = 0.0
    classifier_max_tokens: int = 10
    classifier_max_input_chars: int = 2000

    enable_cache: bool = True
    cache_ttl_seconds: int = 300
    cache_max_size: int = 1000

    fallback_tier: str = "SIMPLE"
    fallback_to_complexity_router: bool = True

    model_config = ConfigDict(extra="allow")
