"""Variable interpolation for MCP server `static_headers` and authentication.

Admins define `${VAR_NAME}` placeholders in a server's `static_headers` (or
auth value) and a sibling `env_vars` list that declares each variable's scope:

    [{"name": "DB_HOSTNAME",   "scope": "instance",  "value": "db.corp.internal"},
     {"name": "CORP_PASSWORD", "scope": "per_user",  "value": null}]

At request time we resolve each `${NAME}` to:
    1. The calling user's stored value for that var (per_user scope), OR
    2. The instance value baked into the server config (instance scope).

If a required per_user var has no stored value, we raise `MissingEnvVarsError`,
which the MCP handler converts into a `tools/call` error containing a deep
link back to the dashboard's fill-in-credentials modal.

Per-user values live in `LiteLLM_MCPUserCredentials` next to BYOK / OAuth
credentials, distinguished by a `"type": "vars"` field in the JSON payload
(matching the existing convention from `CLAUDE.md`).
"""

import json
import os
import re
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.db import (
    _decode_user_credential,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.utils import PrismaClient

EnvVarScope = Literal["instance", "per_user"]

# Matches `${NAME}` where NAME is UPPER_SNAKE_CASE. Same rule as the UI's
# `EnvVarsSection` warningOnly validator, so anything the form accepts is
# resolved here and vice-versa.
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

# JSON-payload tag that distinguishes per-user env vars from BYOK / OAuth.
_VARS_PAYLOAD_TYPE = "vars"


class EnvVarDefinition(BaseModel):
    """One row of the per-server env-var definition list."""

    name: str
    scope: EnvVarScope
    # Only meaningful for `instance` scope. `per_user` entries store the
    # admin-declared placeholder (no value); the value comes from each user.
    value: Optional[str] = None


class MissingEnvVarsError(Exception):
    """Raised when a server's static_headers reference per-user vars the
    caller has not yet stored. The MCP handler catches this and returns a
    `CallToolResult(isError=True)` so Claude Code prints the message.
    """

    def __init__(
        self,
        *,
        server_alias: str,
        server_name: Optional[str],
        missing: List[str],
    ):
        self.server_alias = server_alias
        self.server_name = server_name or server_alias
        self.missing = list(missing)
        super().__init__(
            f"Missing per-user MCP vars for server={server_alias!r}: {missing}"
        )

    def deep_link(self) -> str:
        """URL the user clicks to land on the MCP Servers page with the fill
        modal auto-opened. The modal handler in `mcp_servers.tsx` only mounts
        on `/tools/mcp-servers`, so the path matters.

        Resolution order:
            1. ``PROXY_UI_BASE_URL`` — full URL to the UI root (use this in
               dev, e.g. ``http://localhost:3000``).
            2. ``PROXY_BASE_URL/ui`` — the proxy-served static export path.
            3. ``http://localhost:4000/ui`` — final fallback.
        """
        ui_base = os.environ.get("PROXY_UI_BASE_URL")
        if not ui_base:
            proxy_base = os.environ.get(
                "PROXY_BASE_URL", "http://localhost:4000"
            ).rstrip("/")
            ui_base = f"{proxy_base}/ui"
        return (
            f"{ui_base.rstrip('/')}/tools/mcp-servers"
            f"?fill_fields={self.server_alias}"
        )

    def to_user_message(self) -> str:
        """Single-string error rendered into the terminal by the MCP client.

        The shape mirrors the prototype's `MockClaudeCodeModal` copy so the
        real and mock experiences read the same.
        """
        bullets = "\n".join(f"  - {name}" for name in self.missing)
        return (
            f'Cannot connect to MCP server "{self.server_name}".\n\n'
            f"Your administrator configured this server to require per-user "
            f"credentials, but you haven't set the following yet:\n"
            f"{bullets}\n\n"
            f"Set your credentials here:\n{self.deep_link()}"
        )


def parse_env_var_definitions(raw: Any) -> List[EnvVarDefinition]:
    """Normalize whatever shape the DB / config gave us into typed objects.

    Accepts:
        - None / empty → []
        - list of dicts (DB JSON) → parsed
        - list of EnvVarDefinition (in-memory) → returned as-is
    Silently drops malformed entries; they would be unresolvable anyway.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    parsed: List[EnvVarDefinition] = []
    for entry in raw:
        if isinstance(entry, EnvVarDefinition):
            parsed.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        try:
            parsed.append(EnvVarDefinition.model_validate(entry))
        except Exception as e:  # noqa: BLE001 — config data, log and skip
            verbose_proxy_logger.debug(
                "env_vars: dropping malformed entry %r: %s", entry, e
            )
    return parsed


def _placeholders_in(text: str) -> List[str]:
    return _PLACEHOLDER_RE.findall(text or "")


def collect_placeholders(values: Iterable[Optional[str]]) -> List[str]:
    """Return the union of `${NAME}` references found across `values`."""
    seen: List[str] = []
    for v in values:
        if not v:
            continue
        for name in _placeholders_in(v):
            if name not in seen:
                seen.append(name)
    return seen


def resolve_values(
    defs: List[EnvVarDefinition],
    per_user_values: Dict[str, str],
    *,
    referenced: Iterable[str],
) -> Dict[str, str]:
    """Build the `{name: value}` map for substitution.

    Per-user wins over instance for the same name (defensive — admins should
    not declare both, but if they do, the per-user value is the more specific).
    Only resolves vars actually referenced; unreferenced defs are ignored so
    a stray missing per-user value never blocks an unrelated request.
    """
    by_name = {d.name: d for d in defs}
    referenced_set = set(referenced)
    resolved: Dict[str, str] = {}
    for name in referenced_set:
        if name in per_user_values and per_user_values[name]:
            resolved[name] = per_user_values[name]
            continue
        d = by_name.get(name)
        if d is None or d.scope != "instance":
            continue
        if d.value is None or d.value == "":
            continue
        resolved[name] = d.value
    return resolved


def missing_required(
    defs: List[EnvVarDefinition],
    per_user_values: Dict[str, str],
    *,
    referenced: Iterable[str],
) -> List[str]:
    """Names that are referenced AND defined-as-per_user AND not yet set.

    instance vars without a value are *not* reported as missing here — that
    is admin misconfiguration, surfaced at server-edit time. Only per-user
    gaps generate the dashboard deep-link error.
    """
    by_name = {d.name: d for d in defs}
    out: List[str] = []
    for name in referenced:
        d = by_name.get(name)
        if d is None or d.scope != "per_user":
            continue
        if not per_user_values.get(name):
            out.append(name)
    return out


def _interpolate_string(value: str, resolved: Dict[str, str]) -> str:
    def _sub(match: "re.Match[str]") -> str:
        name = match.group(1)
        return resolved.get(name, match.group(0))

    return _PLACEHOLDER_RE.sub(_sub, value)


def interpolate_headers(
    headers: Optional[Dict[str, str]],
    resolved: Dict[str, str],
) -> Optional[Dict[str, str]]:
    """Return a new dict with `${NAME}` placeholders replaced in each value.

    Unresolved placeholders are left in place (caller already checked
    `missing_required`); we don't want to silently strip them and ship a
    broken header upstream.
    """
    if not headers:
        return headers
    return {k: _interpolate_string(v or "", resolved) for k, v in headers.items()}


def interpolate_value(value: Optional[str], resolved: Dict[str, str]) -> Optional[str]:
    """`interpolate_headers` for a scalar (auth token, etc.)."""
    if value is None:
        return None
    return _interpolate_string(value, resolved)


# ---------------------------------------------------------------------------
# Storage helpers — per-user values in LiteLLM_MCPUserCredentials
# ---------------------------------------------------------------------------


def _encode_vars_payload(values: Dict[str, str]) -> str:
    payload = {"type": _VARS_PAYLOAD_TYPE, "values": values}
    return encrypt_value_helper(json.dumps(payload))


def _decode_vars_payload(stored: str) -> Optional[Dict[str, str]]:
    decoded = _decode_user_credential(stored)
    if decoded is None:
        return None
    try:
        parsed = json.loads(decoded)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict) or parsed.get("type") != _VARS_PAYLOAD_TYPE:
        return None
    values = parsed.get("values")
    if not isinstance(values, dict):
        return None
    # Drop any non-string values defensively.
    return {k: v for k, v in values.items() if isinstance(v, str)}


async def get_user_env_vars(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
) -> Dict[str, str]:
    """Return the per-user `{NAME: value}` map for (user, server), or {}."""
    row = await prisma_client.db.litellm_mcpusercredentials.find_unique(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
    )
    if row is None:
        return {}
    return _decode_vars_payload(row.credential_b64) or {}


async def store_user_env_vars(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
    values: Dict[str, str],
) -> None:
    """Upsert the per-user values payload. Replaces any existing vars payload
    in full (the UI form sends the complete map on save).

    Caveat — shares the (user_id, server_id) row with BYOK / OAuth payloads.
    If a row already exists with a different `"type"`, this overwrites it.
    In practice a server that uses per-user vars does not also need a per-user
    BYOK key (the var *is* the credential), so the collision is acceptable for
    the prototype. If we ever need both, split this into its own table.
    """
    encoded = _encode_vars_payload(values)
    await prisma_client.db.litellm_mcpusercredentials.upsert(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}},
        data={
            "create": {
                "user_id": user_id,
                "server_id": server_id,
                "credential_b64": encoded,
            },
            "update": {"credential_b64": encoded},
        },
    )
