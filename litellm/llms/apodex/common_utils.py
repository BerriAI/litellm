"""
Shared helpers for the Apodex provider.

Apodex serves two model families on one base URL, and the model id picks which
contract applies. Core models (apodex-1.1, apodex-1.1-mini) are plain inference
with native sampling parameters. The Deep Research tiers run an agent that
plans, searches and iterates, so they ignore sampling parameters, reject
OpenAI-style tools, and keep server-side state.

Ref: https://platform.apodex.ai/docs/models
"""

from typing import Final

from litellm.secret_managers.main import get_secret_str

APODEX_API_BASE_URL: Final = "https://api.apodex.ai/v1"

_DEEP_RESEARCH_MARKER: Final = "-deep-"
_RESPONSES_ONLY_MODELS: Final = frozenset({"apodex-1-1-deep-discover"})


def strip_provider_prefix(model: str) -> str:
    return model.rpartition("/")[2]


def is_deep_research_model(model: str) -> bool:
    """True for the Deep Research / Solve / Discover tiers, e.g. apodex-1-1-deep-solve."""
    return _DEEP_RESEARCH_MARKER in strip_provider_prefix(model)


def is_responses_only_model(model: str) -> bool:
    return strip_provider_prefix(model) in _RESPONSES_ONLY_MODELS


def get_apodex_api_key(api_key: str | None = None) -> str | None:
    return api_key or get_secret_str("APODEX_API_KEY")


def get_apodex_api_base(api_base: str | None = None) -> str:
    return api_base or get_secret_str("APODEX_API_BASE") or APODEX_API_BASE_URL
