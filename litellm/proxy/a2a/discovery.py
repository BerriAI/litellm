"""
Fetch an A2A agent's well-known card from the upstream agent.

Different agent runtimes publish the card at different URL shapes, so the
fetcher dispatches by ``discovery_mode``:

- ``well_known_fallback`` (pure A2A): the card lives at one of the standard
  well-known paths on the agent's own base URL. We try the canonical path,
  then the previous-spec path, then a non-standard root fallback.

- ``langgraph_platform``: LangGraph Platform mounts a single card endpoint at
  ``{base}/.well-known/agent-card.json`` and disambiguates assistants via the
  ``assistant_id`` query parameter. There is no per-assistant subpath, so the
  pure-A2A fallback strategy returns 404 for these deployments.
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.url_utils import SSRFError, async_safe_get
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider


class DiscoveryMode(str, Enum):
    """How to locate the upstream agent card.

    String-valued so it serializes cleanly over JSON / Pydantic.
    """

    WELL_KNOWN_FALLBACK = "well_known_fallback"
    LANGGRAPH_PLATFORM = "langgraph_platform"


# Paths the pure-A2A fetcher tries in order. The first two are the current and
# previous A2A spec locations; ``/agent.json`` is a non-standard root fallback
# some agents still serve.
AGENT_CARD_WELL_KNOWN_PATHS: Tuple[str, ...] = (
    "/.well-known/agent-card.json",
    "/.well-known/agent.json",
    "/agent.json",
)

DEFAULT_DISCOVERY_TIMEOUT_SECONDS = 10.0


class AgentCardDiscoveryError(Exception):
    """Raised when none of the well-known paths returned a usable agent card."""


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _build_langgraph_platform_paths(
    params: Optional[Dict[str, Any]],
) -> Tuple[str, ...]:
    """Build the paths to try for LangGraph Platform discovery.

    LangGraph serves the card at ``/.well-known/agent-card.json`` with the
    ``assistant_id`` carried as a query parameter. We still try the other
    A2A path variants (with the same query string appended) so we degrade
    gracefully if a deployment uses an older spec name.
    """
    assistant_id = (params or {}).get("assistant_id")
    if not assistant_id:
        raise AgentCardDiscoveryError(
            "langgraph_platform discovery requires params.assistant_id"
        )
    query = urlencode({"assistant_id": str(assistant_id)})
    return tuple(f"{path}?{query}" for path in AGENT_CARD_WELL_KNOWN_PATHS)


def _paths_for_mode(
    mode: DiscoveryMode, params: Optional[Dict[str, Any]]
) -> Tuple[str, ...]:
    if mode == DiscoveryMode.WELL_KNOWN_FALLBACK:
        return AGENT_CARD_WELL_KNOWN_PATHS
    if mode == DiscoveryMode.LANGGRAPH_PLATFORM:
        return _build_langgraph_platform_paths(params)
    raise AgentCardDiscoveryError(f"unsupported discovery_mode: {mode}")


async def fetch_well_known_card(
    base_url: str,
    *,
    discovery_mode: DiscoveryMode = DiscoveryMode.WELL_KNOWN_FALLBACK,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Fetch an agent card from ``base_url`` using the strategy chosen by
    ``discovery_mode``. Returns the parsed JSON from the first path that
    responds with a JSON body.

    Raises:
        AgentCardDiscoveryError: if every path fails (network error, non-2xx,
            or non-JSON body), or if the chosen mode is missing required params.
    """
    if not base_url:
        raise AgentCardDiscoveryError("base_url is required")

    normalized = _normalize_base_url(base_url)
    paths = _paths_for_mode(discovery_mode, params)
    client = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.A2A,
        params={"timeout": timeout},
    )

    last_error: Optional[str] = None
    for path in paths:
        url = f"{normalized}{path}"
        try:
            # ``async_safe_get`` validates the URL against the SSRF blocklist
            # (private/loopback IPs, cloud metadata endpoints, etc.) on every
            # redirect hop. Even though the discovery endpoint is admin-only,
            # we don't want a compromised admin key to be able to probe
            # internal infrastructure through this fetcher.
            # Pass ``headers or {}`` because ``async_safe_get`` (in the
            # URL-validation path) uses ``kwargs.pop("headers", {})`` which
            # returns ``None`` when the key is present-but-None, then crashes
            # on ``{**None, "Host": ...}``. Default the kwarg to an empty
            # dict so production (``user_url_validation=True``) doesn't 500.
            response = await async_safe_get(client, url, headers=headers or {})
        except SSRFError as exc:
            last_error = f"{url}: {exc!s}"
            verbose_proxy_logger.debug(
                "A2A discovery blocked by SSRF guard for %s: %s", url, exc
            )
            continue
        except Exception as exc:
            last_error = f"{url}: {exc!s}"
            verbose_proxy_logger.debug("A2A discovery failed for %s: %s", url, exc)
            continue

        if response.status_code >= 400:
            last_error = f"{url}: HTTP {response.status_code}"
            verbose_proxy_logger.debug(
                "A2A discovery HTTP %s for %s", response.status_code, url
            )
            continue

        try:
            card = response.json()
        except Exception as exc:
            last_error = f"{url}: invalid JSON ({exc!s})"
            continue

        if not isinstance(card, dict):
            last_error = f"{url}: expected JSON object, got {type(card).__name__}"
            continue

        verbose_proxy_logger.debug("A2A discovery succeeded at %s", url)
        return card

    raise AgentCardDiscoveryError(
        f"Could not fetch agent card from {base_url} (mode={discovery_mode.value}). "
        f"Last error: {last_error}"
    )
