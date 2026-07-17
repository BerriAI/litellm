from __future__ import annotations

from typing import Optional


def safe_relay_state(target: Optional[str], default: str) -> str:
    """Return ``target`` only if it's a safe same-site path, else ``default``.

    Guards the post-login redirect against open-redirect: the target must be a
    relative path (single leading slash, no scheme, no protocol-relative ``//``
    or backslash tricks).
    """
    if target and target.startswith("/") and not target.startswith("//") and "://" not in target and "\\" not in target:
        return target
    return default
