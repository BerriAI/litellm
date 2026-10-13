"""Which dataplane routes the passthrough may carry. Management only, never auth bootstrap or inference."""

from dataclasses import dataclass
from typing import Final
from urllib.parse import unquote

MANAGEMENT_PREFIXES: Final = (
    "/key/",
    "/v2/key/",
    "/user/",
    "/v2/user/",
    "/team/",
    "/v2/team/",
    "/organization/",
    "/v2/organization/",
    "/customer/",
    "/end_user/",
    "/budget/",
    "/tag/",
    "/model/",
    "/v1/model/info",
    "/v2/model/",
    "/model_group/",
    "/model_access_group/",
    "/access_group/",
    "/credentials",
    "/credential",
    "/guardrails/",
    "/v2/guardrails/",
    "/v1/mcp/server",
    "/vector_store/",
    "/spend/",
    "/global/spend",
    "/user/daily/activity",
    "/team/daily/activity",
    "/tag/daily/activity",
    "/audit",
    "/management/v1/",
    "/health",
    "/active/callbacks",
    "/callbacks",
    "/alerting/",
    "/router_settings",
    "/settings",
)

DENIED_PREFIXES: Final = (
    "/health/drain",
    "/login",
    "/logout",
    "/sso/",
    "/token",
    "/jwt/",
    "/oauth/",
    "/invitation/",
    "/onboarding/",
    "/session/",
    "/key/generate/service_account",
    "/config/",
    "/debug/",
    "/utils/",
)


@dataclass(frozen=True, slots=True)
class Allowed:
    path: str


@dataclass(frozen=True, slots=True)
class Denied:
    path: str
    reason: str


def _matches(path: str, prefix: str) -> bool:
    """Route boundary aware: ``/audit`` matches ``/audit`` and ``/audit/x`` but never ``/auditor``."""
    base: Final = prefix.rstrip("/")
    return path == base or path.startswith(base + "/")


def _normalize(raw_path: str) -> str | None:
    """Decode once and refuse anything still encoded, so the path we check is the path httpx sends."""
    decoded: Final = unquote(raw_path)
    if any(reserved in decoded for reserved in ("%", "?", "#", "\\")):
        return None
    return "/" + decoded.lstrip("/")


def decide(raw_path: str) -> Allowed | Denied:
    path: Final = _normalize(raw_path)
    if path is None:
        return Denied("/" + raw_path.lstrip("/"), "encoded or reserved path characters are not allowed")
    segments: Final = path.split("/")[1:]
    if any(segment in {".", ".."} for segment in segments) or "" in segments[:-1]:
        return Denied(path, "path traversal is not allowed")
    if any(_matches(path, prefix) for prefix in DENIED_PREFIXES):
        return Denied(path, "auth bootstrap, debug and lifecycle routes never go through the control plane")
    if not any(_matches(path, prefix) for prefix in MANAGEMENT_PREFIXES):
        return Denied(path, "only management routes are proxied, inference goes straight to the dataplane")
    return Allowed(path)
