from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue

from .config import AUTOROUTER_MODEL_NAME

ENV_KEY: Final = "env"
PERMISSIONS_KEY: Final = "permissions"
ALLOW_KEY: Final = "allow"
API_KEY_HELPER_KEY: Final = "apiKeyHelper"
ANTHROPIC_API_KEY_KEY: Final = "ANTHROPIC_API_KEY"
ANTHROPIC_AUTH_TOKEN_KEY: Final = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_BASE_URL_KEY: Final = "ANTHROPIC_BASE_URL"
ENABLE_TOOL_SEARCH_KEY: Final = "ENABLE_TOOL_SEARCH"
ENABLE_TOOL_SEARCH_VALUE: Final = "true"
# Force every one of Claude Code's own model tiers to request the auto-router by name.
# Router's auto-router registry is keyed by the literal requested model string
# (litellm/router.py:10711-10717) with no wildcard/pattern resolution, so a bare "*"
# model_name can never work as a catch-all -- these overrides are what actually makes
# Claude Code send "autorouter" regardless of /model or its own version-specific defaults.
ANTHROPIC_DEFAULT_MODEL_ENV_KEYS: Final = (
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
)

# Reduces prompts for the curl commands the shunt guardrail's Read/Bash rewrite generates
# (litellm/proxy/guardrails/auto_router_shunt.py); it is NOT a security boundary. Claude Code
# matches an allow rule's text before its first "*" verbatim, with no argument- or host-aware
# matching, so a rule narrow enough to name "curl" at all cannot also pin the destination the
# trailing "*" is free to name any URL. A real boundary needs a PreToolUse hook instead, which
# is the docs' own recommendation for exactly this case.
SHUNT_BASH_ALLOW_RULES: Final = (
    "Bash(curl -sS -F question=*)",
    "Bash(curl -sS -F spec=*)",
)

_NO_MAPPING: Final[Mapping[str, JsonValue]] = MappingProxyType({})
_NO_RULES: Final[tuple[JsonValue, ...]] = ()


def merge_claude_settings_static_token(
    settings: dict[str, JsonValue], base_url: str, auth_token: str
) -> dict[str, JsonValue]:
    """Return a new settings dict wired to a local ephemeral proxy with a static token.

    Unlike up.py's merge_claude_settings (which sets apiKeyHelper for a long-lived, real
    remote proxy needing refreshable SSO tokens), this proxy is ephemeral and its key is the
    locally persisted autoroute master key, so a plain env var is simpler and correct. Any
    existing apiKeyHelper is cleared so it can't fight with the static token.
    """
    raw_env: Final = settings.get(ENV_KEY, {})
    base_env: Final = raw_env if isinstance(raw_env, dict) else {}
    env: Final[dict[str, JsonValue]] = {
        ENABLE_TOOL_SEARCH_KEY: ENABLE_TOOL_SEARCH_VALUE,
        **base_env,
        ANTHROPIC_BASE_URL_KEY: base_url.rstrip("/"),
        ANTHROPIC_AUTH_TOKEN_KEY: auth_token,
        **{key: AUTOROUTER_MODEL_NAME for key in ANTHROPIC_DEFAULT_MODEL_ENV_KEYS},
    }
    env.pop(ANTHROPIC_API_KEY_KEY, None)
    merged: Final[dict[str, JsonValue]] = {**settings, ENV_KEY: env}
    merged.pop(API_KEY_HELPER_KEY, None)
    return merged


def merge_claude_settings_shunt_permissions(settings: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    """Add the shunt allow rules to `permissions.allow`, preserving every other permissions key.

    A naive top-level `{**settings, "permissions": {...}}` would replace the whole `permissions`
    object, silently dropping any `deny`/`ask` rules the caller already has -- exactly the trap
    `merge_claude_settings_static_token` avoids for `env` by merging that key explicitly instead
    of replacing it. This does the same for `permissions.allow`: read what's there, add only the
    two shunt rules if they're not already present, and leave `deny`/`ask`/anything else alone.

    Returns a `MappingProxyType` nested at every level rather than a plain dict, so the caller's
    `json.dump(..., default=dict)` is what converts it back to something the JSON encoder accepts
    -- the one place a concrete mutable mapping is genuinely needed, kept out of this function.
    """
    raw_permissions: Final = settings.get(PERMISSIONS_KEY, _NO_MAPPING)
    base_permissions: Final = raw_permissions if isinstance(raw_permissions, Mapping) else _NO_MAPPING
    raw_allow: Final = base_permissions.get(ALLOW_KEY, _NO_RULES)
    base_allow: Final = raw_allow if isinstance(raw_allow, (list, tuple)) else _NO_RULES
    new_allow: Final = (*base_allow, *(rule for rule in SHUNT_BASH_ALLOW_RULES if rule not in base_allow))
    permissions: Final = MappingProxyType({**base_permissions, ALLOW_KEY: new_allow})
    return MappingProxyType({**settings, PERMISSIONS_KEY: permissions})


__all__ = [  # mutable-ok: __all__ must be a list per Python convention
    "merge_claude_settings_shunt_permissions",
    "merge_claude_settings_static_token",
]
