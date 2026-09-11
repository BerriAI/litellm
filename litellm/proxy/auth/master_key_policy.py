from typing import Final

INSECURE_MASTER_KEYS: Final = frozenset({"sk-1234"})


def insecure_master_key_warning(master_key: str | None, alternative_auth_enabled: bool) -> str | None:
    if master_key in INSECURE_MASTER_KEYS:
        return (
            "LITELLM_MASTER_KEY is set to the example key 'sk-1234' from the docs. "
            "Anyone who has read the docs can administer this gateway, and publicly reachable "
            "gateways using this key have been compromised. Set a strong random master key "
            "(e.g. `python -c \"import secrets; print('sk-' + secrets.token_urlsafe(32))\"`). "
            "A future release will refuse to start with this key."
        )
    if (master_key is None or master_key == "") and not alternative_auth_enabled:
        return (
            "No master key is set (LITELLM_MASTER_KEY or general_settings.master_key). "
            "Every request to this proxy is accepted without authentication, including "
            'admin routes. Set a strong random master key (e.g. `python -c "import secrets; '
            "print('sk-' + secrets.token_urlsafe(32))\"`) before exposing it to a network."
        )
    return None
