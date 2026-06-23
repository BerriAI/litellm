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
from typing import Callable, Dict, Literal, Optional, Tuple

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
    gateway_available: Callable[[], bool],
) -> Surface:
    """Decide which Databricks surface to target. Pure given ``gateway_available``.

    Precedence (highest first):

    1. Explicit ``api_base`` ending in ``/serving-endpoints`` or ``/ai-gateway``
       always wins (strict Q4 — never auto-rewrite an explicit surface).
    2. The ``databricks_use_ai_gateway`` flag: ``True`` -> gateway, ``False`` ->
       serving-endpoints (no probe).
    3. Auto (flag is ``None``): gateway by default, probing once per host and
       falling back to serving-endpoints only when the gateway is unavailable.

    ``gateway_available`` is a zero-arg callable so the (possibly networked)
    probe is evaluated lazily — only when auto mode actually needs it.
    """
    if is_serving_endpoints_base(api_base):
        return "serving_endpoints"
    if is_gateway_base(api_base):
        return "ai_gateway"

    if use_ai_gateway is True:
        return "ai_gateway"
    if use_ai_gateway is False:
        return "serving_endpoints"

    # Auto: default to the gateway, fall back only if the probe says it's absent.
    return "ai_gateway" if gateway_available() else "serving_endpoints"


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
# Per-host gateway capability cache + probe
# ---------------------------------------------------------------------------

# host -> (available, epoch_seconds_cached_at)
_GATEWAY_CACHE: Dict[str, Tuple[bool, float]] = {}
_GATEWAY_CACHE_LOCK = threading.Lock()
# Cache lifetime. Gateway enablement changes rarely, so an hour keeps the probe
# cost negligible while still picking up enablement within a session lifetime.
GATEWAY_CACHE_TTL_SECONDS = 3600.0
# Statuses that indicate the gateway surface is NOT served by this workspace.
_GATEWAY_ABSENT_STATUSES = (404, 501)


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
        _GATEWAY_CACHE[host] = (available, time.time())


def clear_gateway_cache() -> None:
    """Clear the per-host capability cache (primarily for tests)."""
    with _GATEWAY_CACHE_LOCK:
        _GATEWAY_CACHE.clear()


def gateway_available(
    host: str,
    headers: Optional[dict] = None,
    *,
    probe_fn: Optional[Callable[[str, Optional[dict]], bool]] = None,
) -> bool:
    """Return whether the AI Gateway surface is served by ``host`` (cached per host).

    On a cache miss the (injectable) ``probe_fn`` is invoked and its result
    cached. ``probe_fn`` defaults to :func:`default_gateway_probe`.
    """
    host = workspace_host_from_base(host)
    cached = _cache_get(host)
    if cached is not None:
        return cached

    probe = probe_fn or default_gateway_probe
    try:
        available = probe(host, headers)
    except Exception:
        # Network/transport failure during probe: assume the gateway is present
        # rather than silently degrading every host on a transient blip. A real
        # 404/501 (handled inside the probe) is what marks it absent.
        available = True
    _cache_set(host, available)
    return available


def default_gateway_probe(host: str, headers: Optional[dict]) -> bool:
    """Probe ``<host>/ai-gateway/mlflow/v1/chat/completions`` to detect the gateway.

    A bare ``GET`` against the chat path returns 404/501 when the gateway is not
    served by the workspace, and 4xx (e.g. 400/401/405) when it *is* served but
    the request is rejected for other reasons. We therefore treat 404/501 as
    "absent" and everything else as "present".
    """
    from litellm.llms.custom_httpx.http_handler import _get_httpx_client

    probe_url = normalize_gateway_base(host) + AI_GATEWAY_PATHS["chat"]
    timeout = float(os.getenv("DATABRICKS_GATEWAY_PROBE_TIMEOUT", "3.0"))
    client = _get_httpx_client()
    response = client.get(url=probe_url, headers=headers or {}, timeout=timeout)
    return response.status_code not in _GATEWAY_ABSENT_STATUSES
