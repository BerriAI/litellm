from typing import Dict

from pydantic import JsonValue

ENV_KEY = "env"
API_KEY_HELPER_KEY = "apiKeyHelper"
ANTHROPIC_API_KEY_KEY = "ANTHROPIC_API_KEY"
ANTHROPIC_AUTH_TOKEN_KEY = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_BASE_URL_KEY = "ANTHROPIC_BASE_URL"


def merge_claude_settings_static_token(
    settings: Dict[str, JsonValue], base_url: str, auth_token: str
) -> Dict[str, JsonValue]:
    """Return a new settings dict wired to a local ephemeral proxy with a static token.

    Unlike up.py's merge_claude_settings (which sets apiKeyHelper for a long-lived, real
    remote proxy needing refreshable SSO tokens), this proxy is ephemeral and its key was just
    minted for this session, so a plain env var is simpler and correct. Any existing
    apiKeyHelper is cleared so it can't fight with the static token.
    """
    raw_env = settings.get(ENV_KEY, {})
    base_env = raw_env if isinstance(raw_env, dict) else {}
    env: Dict[str, JsonValue] = {
        **base_env,
        ANTHROPIC_BASE_URL_KEY: base_url.rstrip("/"),
        ANTHROPIC_AUTH_TOKEN_KEY: auth_token,
    }
    env.pop(ANTHROPIC_API_KEY_KEY, None)
    merged: Dict[str, JsonValue] = {**settings, ENV_KEY: env}
    merged.pop(API_KEY_HELPER_KEY, None)
    return merged


__all__ = ["merge_claude_settings_static_token"]
