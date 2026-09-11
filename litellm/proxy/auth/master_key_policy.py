from collections.abc import Mapping
from typing import Final, Literal

from typing_extensions import assert_never

INSECURE_MASTER_KEYS: Final = frozenset({"sk-1234"})

InsecureMasterKeyReason = Literal["example_key", "missing"]

_ALTERNATIVE_AUTH_SETTINGS: Final = ("enable_jwt_auth", "enable_oauth2_auth", "enable_oauth2_proxy_auth", "custom_auth")


def alternative_auth_enabled(general_settings: Mapping[str, object]) -> bool:
    return any(general_settings.get(k, False) for k in _ALTERNATIVE_AUTH_SETTINGS)


def insecure_master_key_reason(
    master_key: str | None, alternative_auth_enabled: bool
) -> InsecureMasterKeyReason | None:
    if master_key in INSECURE_MASTER_KEYS:
        return "example_key"
    if (master_key is None or master_key == "") and not alternative_auth_enabled:
        return "missing"
    return None


def insecure_master_key_warning(master_key: str | None, alternative_auth_enabled: bool) -> str | None:
    reason: Final = insecure_master_key_reason(master_key, alternative_auth_enabled)
    match reason:
        case "example_key":
            return (
                "LITELLM_MASTER_KEY is set to the example key 'sk-1234' from the docs. "
                "Anyone who has read the docs can administer this gateway, and publicly reachable "
                "gateways using this key have been compromised. Set a strong random master key "
                "(e.g. `python -c \"import secrets; print('sk-' + secrets.token_urlsafe(32))\"`). "
                "A future release will refuse to start with this key."
            )
        case "missing":
            return (
                "No master key is set (LITELLM_MASTER_KEY or general_settings.master_key). "
                "Every request to this proxy is accepted without authentication, including "
                'admin routes. Set a strong random master key (e.g. `python -c "import secrets; '
                "print('sk-' + secrets.token_urlsafe(32))\"`) before exposing it to a network."
            )
        case None:
            return None
    assert_never(reason)
