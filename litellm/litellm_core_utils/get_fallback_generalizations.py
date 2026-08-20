"""
Loads the fallback generalization rules that ``fallback_generalizations`` compiles.

The rules live in their own ``litellm/fallback_generalizations.json``, bundled with the
package and served from the repository, so adding a model to the cost map never shuffles
the rule block around and the two files can be published independently.

Resolution mirrors the model cost map: remote first, bundled copy on any failure. Set
``LITELLM_LOCAL_FALLBACK_GENERALIZATIONS=True`` (or ``LITELLM_LOCAL_MODEL_COST_MAP=True``,
which already means "no registry network calls") to skip the fetch, and
``LITELLM_FALLBACK_GENERALIZATIONS_URL`` to point at a different remote file.
"""

import json
import os
from collections.abc import Callable, Mapping
from importlib.resources import files
from typing import Final, TypeAlias

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.fallback_generalizations import (
    set_fallback_generalizations,
)

FallbackRules: TypeAlias = tuple[Mapping[str, object], ...]
JsonFetcher: TypeAlias = Callable[[str], object]

RULES_FIELD: Final = "rules"
LOCAL_FILE_NAME: Final = "fallback_generalizations.json"
LOCAL_ONLY_ENV_VARS: Final = ("LITELLM_LOCAL_FALLBACK_GENERALIZATIONS", "LITELLM_LOCAL_MODEL_COST_MAP")
FETCH_TIMEOUT_SECONDS: Final = 5


def _is_local_only() -> bool:
    return any(os.getenv(name, "").lower() == "true" for name in LOCAL_ONLY_ENV_VARS)


def _extract_rules(content: object, source: str) -> FallbackRules:
    if not isinstance(content, Mapping):
        verbose_logger.warning(
            "LiteLLM: fallback generalizations from %s are not an object (type=%s); ignoring them.",
            source,
            type(content).__name__,
        )
        return ()

    raw: Final = content.get(RULES_FIELD)
    if not isinstance(raw, list):
        verbose_logger.warning(
            "LiteLLM: fallback generalizations from %s have no '%s' list; ignoring them.",
            source,
            RULES_FIELD,
        )
        return ()

    rules: Final = tuple(rule for rule in raw if isinstance(rule, Mapping))
    if len(rules) != len(raw):
        verbose_logger.warning(
            "LiteLLM: %s fallback generalization rule(s) from %s are not objects; skipping them.",
            len(raw) - len(rules),
            source,
        )
    return rules


def load_local_fallback_generalizations() -> FallbackRules:
    """Return the rules from the copy bundled with the package."""
    content: Final = json.loads(files("litellm").joinpath(LOCAL_FILE_NAME).read_text(encoding="utf-8"))
    return _extract_rules(content, source=LOCAL_FILE_NAME)


def fetch_remote_json(url: str) -> object:
    """Fetch and parse ``url``, raising on any network, status, or parse failure."""
    response: Final = httpx.get(url, timeout=FETCH_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def get_fallback_generalizations(url: str, fetch: JsonFetcher = fetch_remote_json) -> FallbackRules:
    """Return the rules to install, falling back to the bundled copy whenever the fetch is unusable.

    Routing and capability fallbacks must survive an unreachable, 404, or truncated remote
    file, so anything that does not yield at least one rule resolves to the bundled copy.
    """
    if _is_local_only():
        return load_local_fallback_generalizations()

    try:
        remote: Final = _extract_rules(fetch(url), source=url)
    except (httpx.HTTPError, ValueError) as e:
        verbose_logger.warning(
            "LiteLLM: Failed to fetch fallback generalizations from %s: %s. Falling back to local backup.",
            url,
            str(e),
        )
        return load_local_fallback_generalizations()

    return remote or load_local_fallback_generalizations()


def install_fallback_generalizations(url: str | None = None, fetch: JsonFetcher = fetch_remote_json) -> FallbackRules:
    """Load the rules and install them into the generalizations registry."""
    import litellm

    rules: Final = get_fallback_generalizations(url=url or litellm.fallback_generalizations_url, fetch=fetch)
    set_fallback_generalizations(list(rules))
    return rules
