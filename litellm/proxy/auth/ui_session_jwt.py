"""The one place a UI session JWT is signed.

The ``token`` cookie the dashboard carries is an HS256 JWT signed with the proxy ``master_key``.
It is minted from six places (SSO callback, ``/login``, ``/v2/login``, ``/v3/login``, and the two
onboarding links), and every one of them used to call :func:`jwt.encode` directly with no ``exp``,
so the cookie never expired: a leaked one stayed valid until the master key rotated. Worse, readers
that (correctly) require a bounded lifetime rejected every real cookie, because none of the mints
stamped the claim they were checking for.

Routing all six through :func:`encode_ui_session_jwt` makes the lifetime a property of the
credential rather than of whichever endpoint happened to issue it. The window is
``LITELLM_UI_SESSION_DURATION`` (default 24h), the same setting that already bounds the virtual key
carried inside the payload, so the cookie and the key it wraps expire together instead of the
cookie outliving its own contents.

This module deliberately imports nothing from the proxy package: ``login_utils`` already imports
from ``ui_sso``, so a shared helper living in either one would close an import cycle for the other.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt

from litellm.constants import LITELLM_UI_SESSION_DURATION
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.types.proxy.ui_sso import ReturnedUITokenObject, UISessionJWTClaims

_UI_SESSION_JWT_ALGORITHM = "HS256"


def ui_session_expires_at(now: datetime) -> datetime:
    """When a UI session minted at ``now`` expires."""
    return now + timedelta(seconds=duration_in_seconds(LITELLM_UI_SESSION_DURATION))


def encode_ui_session_jwt(
    token_object: ReturnedUITokenObject,
    master_key: str,
    now: datetime | None = None,
) -> str:
    """Sign ``token_object`` as the UI session cookie, stamping a bounded ``exp``.

    ``now`` is injectable so tests can pin the expiry without patching the clock.
    """
    issued_at = now if now is not None else datetime.now(timezone.utc)
    claims: UISessionJWTClaims = {
        **token_object,
        "exp": int(ui_session_expires_at(issued_at).timestamp()),
    }
    return jwt.encode(dict(claims), master_key, algorithm=_UI_SESSION_JWT_ALGORITHM)
