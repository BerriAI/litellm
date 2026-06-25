"""
Databricks Unity AI Gateway routing helpers.

Unity AI Gateway exposes models under ``https://<workspace-host>/ai-gateway/...``
with both unified (OpenAI-compatible) and provider-native API surfaces. This
module provides the pure, side-effect-free routing primitives used by the
Databricks connector to pick the correct gateway path for a given model:

* :func:`detect_family` — name-pattern router (no network call).
* :data:`AI_GATEWAY_PATHS` — per-family path templates appended to the gateway base.
* :func:`normalize_gateway_base` — ``<host>`` -> ``<host>/ai-gateway``.
* :func:`gateway_chat_url` / :func:`gateway_responses_url` — full URL builders.

Reference: https://docs.databricks.com/aws/en/ai-gateway/query-endpoints-beta
Only the reverse-proxied ``<workspace>/ai-gateway`` form is supported (the
dedicated ``*.ai-gateway.cloud.databricks.com`` hostname is intentionally not
handled).
"""

import os
import re
import threading
import time
from enum import Enum
from typing import Dict, Literal, Optional, Tuple

# Prefixes stripped from a model id before name-pattern matching. Supports both
# "databricks/databricks-claude-..." and bare "databricks-claude-..." forms.
_MODEL_PREFIXES = ("databricks/", "databricks-")


class ProviderFamily(str, Enum):
    """Which provider-native API contract a gateway endpoint speaks.

    Routing (first match wins): ``*claude*`` -> ANTHROPIC, ``*gemini*`` -> GEMINI,
    ``gpt-<digit>*`` -> OPENAI_RESPONSES, everything else -> OPENAI (the
    always-safe unified MLflow chat default).
    """

    OPENAI = "openai"  # unified MLflow chat — universal default
    OPENAI_RESPONSES = "openai_responses"  # native OpenAI Responses — gpt-N (non-oss)
    ANTHROPIC = "anthropic"  # native Anthropic Messages — Claude
    GEMINI = "gemini"  # native Google Gemini generateContent


def _bare_name(model: str) -> str:
    name = model.lower().strip()
    for prefix in _MODEL_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def detect_family(model: str) -> ProviderFamily:
    """Resolve the provider family for a model by name pattern (no network call).

    Priority (first match wins):

    1. ``*claude*``      -> :attr:`ProviderFamily.ANTHROPIC`
    2. ``*gemini*``      -> :attr:`ProviderFamily.GEMINI`
    3. ``gpt-<digit>*``  -> :attr:`ProviderFamily.OPENAI_RESPONSES`
       The leading-digit requirement excludes ``gpt-oss-*`` (starts with
       ``gpt-o``) without an explicit blocklist; future ``gpt-6``/``gpt-7``
       inherit the rule automatically.
    4. everything else   -> :attr:`ProviderFamily.OPENAI` (unified MLflow chat).
    """
    name = _bare_name(model)
    if "claude" in name:
        return ProviderFamily.ANTHROPIC
    if "gemini" in name:
        return ProviderFamily.GEMINI
    if re.match(r"gpt-\d", name):
        return ProviderFamily.OPENAI_RESPONSES
    return ProviderFamily.OPENAI


# Path templates appended to the AI Gateway base ("<host>/ai-gateway").
# {endpoint} is substituted with the bare endpoint name (gemini only — its
# endpoint name is part of the URL; all other families carry it in the body).
AI_GATEWAY_PATHS = {
    "chat": "/mlflow/v1/chat/completions",
    "embeddings": "/mlflow/v1/embeddings",
    "supervisor_responses": "/mlflow/v1/responses",
    "openai_responses": "/openai/v1/responses",
    "anthropic_messages": "/anthropic/v1/messages",
    "gemini_generate_content": "/gemini/v1beta/models/{endpoint}:generateContent",
}

_AI_GATEWAY_SUFFIX = "/ai-gateway"
_SERVING_ENDPOINTS_SUFFIX = "/serving-endpoints"


def bare_endpoint_name(model: str) -> str:
    """Strip the ``databricks/`` provider prefix to get the bare endpoint name."""
    name = model.strip()
    if name.startswith("databricks/"):
        name = name[len("databricks/") :]
    return name


def normalize_gateway_base(host: str) -> str:
    """Return the AI Gateway base for a workspace host (``<host>/ai-gateway``).

    Idempotent: a host already ending in ``/ai-gateway`` is returned unchanged.
    A host ending in ``/serving-endpoints`` has that suffix replaced.
    """
    h = host.rstrip("/")
    if h.endswith(_AI_GATEWAY_SUFFIX):
        return h
    if h.endswith(_SERVING_ENDPOINTS_SUFFIX):
        h = h[: -len(_SERVING_ENDPOINTS_SUFFIX)]
    return h + _AI_GATEWAY_SUFFIX


def workspace_host_from_base(api_base: str) -> str:
    """Strip a known surface suffix (``/ai-gateway`` or ``/serving-endpoints``)
    to recover the bare workspace host."""
    h = api_base.rstrip("/")
    for suffix in (_AI_GATEWAY_SUFFIX, _SERVING_ENDPOINTS_SUFFIX):
        if h.endswith(suffix):
            return h[: -len(suffix)]
    return h


def is_gateway_base(api_base: Optional[str]) -> bool:
    """True if the base already targets the AI Gateway surface."""
    return api_base is not None and api_base.rstrip("/").endswith(_AI_GATEWAY_SUFFIX)


def is_serving_endpoints_base(api_base: Optional[str]) -> bool:
    """True if the base already targets the legacy serving-endpoints surface."""
    return api_base is not None and api_base.rstrip("/").endswith(
        _SERVING_ENDPOINTS_SUFFIX
    )


def has_explicit_custom_path(api_base: Optional[str]) -> bool:
    """True if ``api_base`` carries a user-defined path that is *not* a recognized
    surface suffix.

    Such a base is treated as an opaque, explicit endpoint and used verbatim
    (strict Q4 — never rewritten to the gateway). A bare workspace host
    (path ``""`` or ``"/"``) and the ``/serving-endpoints`` / ``/ai-gateway``
    surfaces are *not* custom paths.
    """
    if not api_base:
        return False
    if is_gateway_base(api_base) or is_serving_endpoints_base(api_base):
        return False
    from urllib.parse import urlsplit

    # A bare host has path "" and root "/" collapses to "" after rstrip.
    path = urlsplit(api_base).path.rstrip("/")
    return path != ""


# ---------------------------------------------------------------------------
# Surface resolution: AI Gateway (default) vs. legacy serving-endpoints
# ---------------------------------------------------------------------------

Surface = Literal["ai_gateway", "serving_endpoints"]


def parse_use_ai_gateway_flag(
    litellm_params: Optional[dict] = None,
    optional_params: Optional[dict] = None,
) -> Optional[bool]:
    """Resolve the ``databricks_use_ai_gateway`` override.

    Precedence: ``optional_params`` -> ``litellm_params`` -> ``DATABRICKS_USE_AI_GATEWAY``
    env var. Returns ``True`` (force gateway), ``False`` (force serving-endpoints),
    or ``None`` (auto-detect — the default).
    """
    for source in (optional_params, litellm_params):
        if source is not None and "databricks_use_ai_gateway" in source:
            return _coerce_tristate(source.get("databricks_use_ai_gateway"))

    env_val = os.getenv("DATABRICKS_USE_AI_GATEWAY")
    if env_val is not None:
        return _coerce_tristate(env_val)
    return None


def _coerce_tristate(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on", "force"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    if text in ("auto", ""):
        return None
    return None


def resolve_surface(
    api_base: Optional[str],
    use_ai_gateway: Optional[bool],
    host: str,
) -> Surface:
    """Decide which Databricks surface to target. Pure, no network call.

    Precedence (highest first):

    1. Explicit ``api_base`` ending in ``/serving-endpoints`` or ``/ai-gateway``
       always wins (strict Q4 — never auto-rewrite an explicit surface).
    2. The ``databricks_use_ai_gateway`` flag: ``True`` -> gateway, ``False`` ->
       serving-endpoints.
    3. Auto (flag is ``None``): **optimistic gateway-first**. We target the
       gateway unless this host is already *known* (cached) to not serve it. There
       is intentionally NO preflight probe — gateway absence is learned reactively
       from a real request (see :func:`is_gateway_absent_error` /
       :func:`mark_gateway_absent`) and cached per host, so the next request for a
       gateway-less host routes straight to serving-endpoints.

    This keeps the gateway the default while never blocking on (or even issuing) a
    network probe from the request hot path.
    """
    if is_serving_endpoints_base(api_base):
        return "serving_endpoints"
    if is_gateway_base(api_base):
        return "ai_gateway"

    if use_ai_gateway is True:
        return "ai_gateway"
    if use_ai_gateway is False:
        return "serving_endpoints"

    # Auto: optimistic gateway-first; only avoid it for a host known to lack it.
    return "serving_endpoints" if gateway_known_absent(host) else "ai_gateway"


def build_surface_base(host_or_base: str, surface: Surface) -> str:
    """Return the base URL (no trailing endpoint path) for the chosen surface.

    Accepts either a bare host or an already-suffixed base and normalizes it.
    """
    host = workspace_host_from_base(host_or_base)
    if surface == "ai_gateway":
        return host + _AI_GATEWAY_SUFFIX
    return host + _SERVING_ENDPOINTS_SUFFIX


def build_gemini_generate_content_url(
    host_or_base: str, model: str, stream: bool = False
) -> str:
    """Native Gemini generateContent URL on the gateway.

    ``<host>/ai-gateway/gemini/v1beta/models/<endpoint>:generateContent`` (or
    ``:streamGenerateContent?alt=sse`` when streaming). The Databricks serving
    endpoint name lives in the URL path (not the body). This surface exists only
    on the AI Gateway, so the workspace host is always used.
    """
    host = workspace_host_from_base(host_or_base)
    base = normalize_gateway_base(host)
    endpoint = bare_endpoint_name(model)
    action = "streamGenerateContent" if stream else "generateContent"
    url = f"{base}/gemini/v1beta/models/{endpoint}:{action}"
    if stream:
        url += "?alt=sse"
    return url


def build_anthropic_messages_url(host_or_base: str) -> str:
    """Native Anthropic Messages URL on the gateway:
    ``<host>/ai-gateway/anthropic/v1/messages``.

    This API surface exists *only* on the AI Gateway, so the gateway base is
    always used (the host is recovered from any provided base). Idempotent for a
    base that already points at the full messages path.
    """
    base = host_or_base.rstrip("/")
    if base.endswith(AI_GATEWAY_PATHS["anthropic_messages"]):
        return base
    host = workspace_host_from_base(base)
    return normalize_gateway_base(host) + AI_GATEWAY_PATHS["anthropic_messages"]


def build_chat_url(host_or_base: str, surface: Surface) -> str:
    """Full chat-completions URL for the chosen surface.

    * ai_gateway        -> ``<host>/ai-gateway/mlflow/v1/chat/completions``
    * serving_endpoints -> ``<host>/serving-endpoints/chat/completions``
    """
    base = build_surface_base(host_or_base, surface)
    if surface == "ai_gateway":
        return base + AI_GATEWAY_PATHS["chat"]
    return base + "/chat/completions"


# ---------------------------------------------------------------------------
# Per-host gateway capability cache (reactive — no preflight probe)
# ---------------------------------------------------------------------------
#
# We never probe the gateway proactively (that would put blocking network I/O on
# the request hot path, including the proxy's async path). Instead we route
# optimistically to the gateway and learn absence *reactively*: when a real
# request to the gateway surface comes back with a host-level "absent" signal
# (see :func:`is_gateway_absent_error`), the connector calls
# :func:`mark_gateway_absent` and retries serving-endpoints. Gateway presence is a
# host-level fact, so a single observation is cached and authoritative for the
# host until the TTL expires.

# host -> (available, epoch_seconds_cached_at)
_GATEWAY_CACHE: Dict[str, Tuple[bool, float]] = {}
_GATEWAY_CACHE_LOCK = threading.Lock()
# Cache lifetime. Gateway enablement changes rarely; an hour re-checks enablement
# within a session lifetime while keeping at most one reactive fallback per host/hour.
GATEWAY_CACHE_TTL_SECONDS = 3600.0
# HTTP status that unambiguously means the workspace does not implement the gateway
# path. (A bare 404 is intentionally NOT here: it conflates "no gateway" with "no
# such model" — see :func:`is_gateway_absent_error`.)
GATEWAY_ABSENT_STATUSES = (501,)
# Error-body markers (lowercased) that indicate the gateway path itself is not served
# (vs. a model-level RESOURCE_DOES_NOT_EXIST).
GATEWAY_ABSENT_MARKERS = ("endpoint_not_found",)


def _cache_get(host: str) -> Optional[bool]:
    with _GATEWAY_CACHE_LOCK:
        entry = _GATEWAY_CACHE.get(host)
        if entry is None:
            return None
        available, cached_at = entry
        if (time.time() - cached_at) > GATEWAY_CACHE_TTL_SECONDS:
            _GATEWAY_CACHE.pop(host, None)
            return None
        return available


def _cache_set(host: str, available: bool) -> None:
    with _GATEWAY_CACHE_LOCK:
        _GATEWAY_CACHE[workspace_host_from_base(host)] = (available, time.time())


def clear_gateway_cache() -> None:
    """Clear the per-host capability cache (primarily for tests)."""
    with _GATEWAY_CACHE_LOCK:
        _GATEWAY_CACHE.clear()


def mark_gateway_absent(host: str) -> None:
    """Record that ``host`` does not serve the AI Gateway surface (reactive)."""
    _cache_set(host, False)


def mark_gateway_present(host: str) -> None:
    """Record that ``host`` serves the AI Gateway surface (reactive)."""
    _cache_set(host, True)


def gateway_known_absent(host: str) -> bool:
    """True only if ``host`` is *cached* as not serving the gateway. Unknown hosts
    return ``False`` (so :func:`resolve_surface` stays optimistic gateway-first)."""
    return _cache_get(workspace_host_from_base(host)) is False


def is_gateway_absent_error(exc: Exception) -> bool:
    """True if ``exc`` from a real gateway request indicates the workspace does not
    serve the AI Gateway *path* (a host-level signal), so the caller should mark the
    host absent and retry serving-endpoints.

    Keyed on the **unambiguous** signals only — HTTP ``501`` or an
    ``endpoint_not_found`` body marker. A bare ``404`` is deliberately NOT treated as
    host-absence: it conflates "no gateway here" with "no such model", and demoting
    the whole host on a model-level 404 would route every model to serving-endpoints
    for the cache TTL.

    Grounded in live verification against a Databricks workspace:
      - absent gateway path  -> ``404 ENDPOINT_NOT_FOUND`` ("Invalid path")
      - missing model        -> ``404 NOT_FOUND`` / ``RESOURCE_DOES_NOT_EXIST``
    so the marker reliably catches a missing gateway while excluding a model-level
    404 (which must not demote the whole host).

    Note: a *markerless* bare ``404`` returns ``False`` here, so it is surfaced
    directly to the caller (treated as a model/other error) rather than triggering a
    serving-endpoints retry. This is intentional given the verified behavior above;
    if a future/edge deployment returned a markerless ``404`` for an absent gateway,
    route around it with the explicit escape hatch
    (``databricks_use_ai_gateway=False`` / ``DATABRICKS_USE_AI_GATEWAY=false``).
    """
    status = getattr(exc, "status_code", None)
    try:
        status_int = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_int = None
    if status_int in GATEWAY_ABSENT_STATUSES:
        return True
    msg = (str(getattr(exc, "message", "") or "") + " " + str(exc)).lower()
    return any(marker in msg for marker in GATEWAY_ABSENT_MARKERS)
