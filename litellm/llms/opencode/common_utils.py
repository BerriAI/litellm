from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Final

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str


class OpenCodeException(BaseLLMException):
    """Exception for OpenCode API errors."""


# Models the gateway serves over the OpenAI Responses wire format, per surface.
#
# The generic responses bridge decides takeover from the cost map's ``mode``
# field, but ``litellm.model_cost`` is fetched from the published remote map at
# import and a provider's entries only land there once released. Relying on it
# alone sends every model below to chat completions — the wrong endpoint — on
# any install whose cost map predates this provider. This table is the routing
# source; the cost map remains the source of pricing.
OPENCODE_RESPONSES_MODELS: Final = MappingProxyType(
    {
        "zen": frozenset(
            {
                "gemini-3-flash",
                "gemini-3.1-pro",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.6-flash",
                "gpt-5",
                "gpt-5-codex",
                "gpt-5-nano",
                "gpt-5.1",
                "gpt-5.1-codex",
                "gpt-5.1-codex-max",
                "gpt-5.1-codex-mini",
                "gpt-5.2",
                "gpt-5.2-codex",
                "gpt-5.3-codex",
                "gpt-5.3-codex-spark",
                "gpt-5.4",
                "gpt-5.4-mini",
                "gpt-5.4-nano",
                "gpt-5.4-pro",
                "gpt-5.5",
                "gpt-5.5-pro",
                "gpt-5.6-luna",
                "gpt-5.6-sol",
                "gpt-5.6-terra",
                "grok-build-0.1",
            }
        ),
        "go": frozenset({"gpt-5.6-luna"}),
    }
)


def opencode_surface(custom_llm_provider: str) -> str | None:
    """Return the OpenCode surface for *custom_llm_provider*, or None."""
    if custom_llm_provider == "opencode_go":
        return "go"
    if custom_llm_provider == "opencode_zen":
        return "zen"
    return None


def is_responses_model(surface: str, model: str) -> bool:
    """Return True when *model* is served over the Responses wire format."""
    bare: Final = model.rsplit("/", 1)[-1]
    return bare in OPENCODE_RESPONSES_MODELS.get(surface, frozenset())


@lru_cache(maxsize=1)
def _bundled_opencode_pricing() -> Mapping[str, Mapping[str, object]]:
    """OpenCode entries from the cost map bundled with the package, parsed once.

    Only the provider's own entries are kept, so the rest of the backup is not
    held in memory.
    """
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    try:
        backup: Final[Mapping[str, Mapping[str, object]]] = GetModelCostMap.load_local_model_cost_map()
    except Exception:  # noqa: BLE001  # an unreadable bundled map means "no pricing to add", never a failed call
        return MappingProxyType({})
    return MappingProxyType({k: v for k, v in backup.items() if k.startswith("opencode_")})


def _is_priced(value: object) -> bool:
    """True when *value* is a rate the cost calculator would bill against."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _carries_pricing(entry: object) -> bool:
    """True when *entry* holds pricing ``cost_per_token`` can actually use.

    Mirrors that function's own test rather than asking whether the key exists.
    ``Router`` registers a bare placeholder for every deployment when it starts,
    so a model in use through the proxy is already present in ``model_cost`` as
    an empty entry before any request runs; a presence check would read that as
    priced and skip the fallback entirely.
    """
    if not isinstance(entry, dict):
        return False
    return (
        _is_priced(entry.get("input_cost_per_token"))
        or _is_priced(entry.get("output_cost_per_token"))
        or entry.get("tiered_pricing") is not None
    )


def ensure_opencode_pricing(custom_llm_provider: str, model: str) -> None:
    """Register bundled pricing for *model* when the runtime cost map lacks it.

    ``litellm.model_cost`` is fetched from the published remote map at import,
    and a provider's entries only land there once released. Until then a call
    resolves no pricing, ``response_cost`` is None, and a proxy recording spend
    reads that as zero. The pricing shipped with the package covers the gap.

    Registered with ``persist_across_reloads=False`` deliberately: a cost map
    that later carries the model must win over the bundled copy, which can go
    stale. If a refresh still lacks the entry, the next call registers it again.
    """
    import litellm

    key: Final = f"{custom_llm_provider}/{model.rsplit('/', 1)[-1]}"
    if _carries_pricing(litellm.model_cost.get(key)):
        return
    entry: Final = _bundled_opencode_pricing().get(key)
    if entry is not None:
        litellm.register_model(
            {key: entry},  # mutable-ok: register_model's contract takes a mutable dict
            persist_across_reloads=False,
        )


def resolve_opencode_api_key(surface: str, api_key: str | None = None) -> str | None:
    """Resolve the OpenCode key for *surface*, most specific source first.

    Explicit argument, then the surface-specific module attribute and env var,
    then the shared OpenCode module attribute and env var, and finally the
    generic ``litellm.api_key``. Deliberately never falls back to
    ANTHROPIC_API_KEY: the messages arm speaks the Anthropic wire format but
    terminates at opencode.ai, so an Anthropic-shaped fallback would hand the
    caller's first-party Anthropic key to a third party.
    """
    import litellm

    surface_upper: Final = surface.upper()
    return (
        api_key
        or getattr(litellm, f"opencode_{surface}_api_key", None)
        or get_secret_str(f"OPENCODE_{surface_upper}_API_KEY")
        or litellm.opencode_api_key
        or get_secret_str("OPENCODE_API_KEY")
        or litellm.api_key
    )


def cost_map_max_output_tokens(surface: str, model: str) -> int | None:
    """Return the cost-map ``max_output_tokens`` for an OpenCode model.

    ``model`` arrives bare (e.g. ``qwen3.7-plus``); qualify it with the surface
    prefix for the ``litellm.model_cost`` lookup. Returns ``None`` when the
    model has no cost-map entry.
    """
    import litellm

    qualified: Final = f"opencode_{surface}/{model}"
    entry: Final = litellm.model_cost.get(qualified)
    if entry is None:
        return None
    if "max_output_tokens" in entry:
        return entry["max_output_tokens"]
    if "max_tokens" in entry:
        return entry["max_tokens"]
    return None


def resolve_opencode_api_base(surface: str, api_base: str | None = None) -> str | None:
    """Resolve the configured OpenCode base URL for *surface*, or None.

    Mirrors :func:`resolve_opencode_api_key`'s precedence so every arm honours
    the same variables. Returns None when nothing is configured, leaving the
    caller to apply its own surface default.
    """
    import litellm

    surface_upper: Final = surface.upper()
    return (
        api_base
        or getattr(litellm, f"opencode_{surface}_api_base", None)
        or get_secret_str(f"OPENCODE_{surface_upper}_API_BASE")
        or litellm.opencode_api_base
        or litellm.api_base
    )
