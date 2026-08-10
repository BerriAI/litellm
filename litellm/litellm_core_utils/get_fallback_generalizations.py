"""
Loads the fallback generalization rules that ``fallback_generalizations`` compiles.

The rules live in their own ``litellm/fallback_generalizations.json`` file, bundled with
the package and served from the repository, so adding a model to the cost map never
shuffles the rule block around and the two can be published independently.

Resolution order mirrors the model cost map: remote first, bundled copy on any failure.
Set ``LITELLM_LOCAL_FALLBACK_GENERALIZATIONS=True`` (or ``LITELLM_LOCAL_MODEL_COST_MAP=True``,
which already means "no registry network calls") to skip the fetch, and
``LITELLM_FALLBACK_GENERALIZATIONS_URL`` to point at a different remote file.
"""

import json
import os
from importlib.resources import files
from typing import Final

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.fallback_generalizations import (
    set_fallback_generalizations,
)

RULES_FIELD: Final = "rules"
LOCAL_FILE_NAME: Final = "fallback_generalizations.json"


def _is_local_only() -> bool:
    return any(
        os.getenv(name, "").lower() == "true"
        for name in ("LITELLM_LOCAL_FALLBACK_GENERALIZATIONS", "LITELLM_LOCAL_MODEL_COST_MAP")
    )


def load_local_fallback_generalizations() -> list:
    """Return the rules from the copy bundled with the package."""
    content: Final = json.loads(files("litellm").joinpath(LOCAL_FILE_NAME).read_text(encoding="utf-8"))
    return _extract_rules(content, source=LOCAL_FILE_NAME)


def _extract_rules(content: object, source: str) -> list:
    if not isinstance(content, dict):
        verbose_logger.warning(
            "LiteLLM: fallback generalizations from %s are not a dict (type=%s); ignoring them.",
            source,
            type(content).__name__,
        )
        return []
    rules: Final = content.get(RULES_FIELD)
    if not isinstance(rules, list):
        verbose_logger.warning(
            "LiteLLM: fallback generalizations from %s have no '%s' list; ignoring them.",
            source,
            RULES_FIELD,
        )
        return []
    return rules


def fetch_remote_fallback_generalizations(url: str, timeout: int = 5) -> list:
    """Fetch the rules from ``url``, raising on network/parse errors for the caller to handle."""
    response: Final = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    return _extract_rules(response.json(), source=url)


def get_fallback_generalizations(url: str) -> list:
    """Return the active rule list, falling back to the bundled copy when the fetch is unusable."""
    if _is_local_only():
        return load_local_fallback_generalizations()

    try:
        remote: Final = fetch_remote_fallback_generalizations(url)
    except Exception as e:
        verbose_logger.warning(
            "LiteLLM: Failed to fetch fallback generalizations from %s: %s. Falling back to local backup.",
            url,
            str(e),
        )
        return load_local_fallback_generalizations()

    if not remote:
        return load_local_fallback_generalizations()

    return remote


def install_fallback_generalizations() -> list:
    """Load the rules and install them into the generalizations registry."""
    from litellm import fallback_generalizations_url

    rules: Final = get_fallback_generalizations(url=fallback_generalizations_url)
    set_fallback_generalizations(rules)
    return rules
