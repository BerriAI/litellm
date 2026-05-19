"""Helpers for resolving admin-declared MCP user fields at request time.

User fields are per-user values (e.g. bearer tokens, workspace IDs) the
admin declares when adding an MCP server. End-users fill them in via the
dashboard; this module retrieves the stored values and injects them into
outbound MCP requests as HTTP headers (http/sse) or env vars (stdio).

The retrieval result is cached in process so a tool call does not pay a
DB round-trip for every step. See ``_user_fields_cache`` in
``server.py`` for the cache itself; this module only contains the pure
resolution / injection logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_logger
from litellm.types.mcp_server.mcp_server_manager import MCPServer

if TYPE_CHECKING:
    from litellm.proxy._types import LiteLLM_MCPServerTable

    UserFieldServer = Union[MCPServer, LiteLLM_MCPServerTable]
else:
    UserFieldServer = MCPServer


def _entry_to_dict(entry: Any) -> Optional[Dict[str, Any]]:
    """Coerce a single user-field entry to a plain dict, or None if malformed.

    Accepts both raw dicts (as Prisma hands JSONB columns back) and Pydantic
    ``MCPUserField`` model instances (as a fully-typed
    ``LiteLLM_MCPServerTable`` would carry).
    """
    if isinstance(entry, dict):
        return entry
    model_dump = getattr(entry, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:  # noqa: BLE001 — drop malformed entries silently
            return None
        if isinstance(dumped, dict):
            return dumped
    return None


def coerce_user_fields(server: UserFieldServer) -> List[Dict[str, Any]]:
    """Return the server's declared user fields as a list of plain dicts.

    The column is stored as JSONB but Prisma sometimes hands it back as a
    string; fully-typed ``LiteLLM_MCPServerTable`` instances carry it as a
    list of ``MCPUserField`` Pydantic models. This normalises all shapes
    and silently drops malformed entries (the admin form filters these
    out, but DB writes from external tooling might not).
    """
    raw = getattr(server, "user_fields", None)
    if not raw:
        return []
    if isinstance(raw, list):
        return [d for d in (_entry_to_dict(e) for e in raw) if d is not None]
    if isinstance(raw, str):
        import json

        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return []
        if isinstance(parsed, list):
            return [d for d in (_entry_to_dict(e) for e in parsed) if d is not None]
    return []


def server_has_user_fields(server: UserFieldServer) -> bool:
    """True iff the server declares any user fields at all."""
    return bool(coerce_user_fields(server))


def compute_missing_user_fields(
    server: UserFieldServer, stored_values: Optional[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """Return the declared field definitions the user has yet to fill in.

    Only ``required`` fields count as missing — optional fields without a
    stored value are still considered satisfied so the user can save
    partial configurations.
    """
    stored = stored_values or {}
    missing: List[Dict[str, Any]] = []
    for entry in coerce_user_fields(server):
        field_key = entry.get("field_key")
        if not isinstance(field_key, str) or not field_key:
            continue
        if not entry.get("required", True):
            continue
        if not stored.get(field_key):
            missing.append(entry)
    return missing


def build_user_fields_missing_error(
    server: MCPServer,
    missing: List[Dict[str, Any]],
    base_dashboard_url: Optional[str],
) -> Dict[str, Any]:
    """Construct the FastAPI ``detail`` payload for a missing-fields 401.

    Includes a ``config_url`` pointing the end-user (or their agent) at
    the dashboard page where they can fill in the missing fields. The
    URL is built defensively — if no base URL is known we emit a relative
    path so curl-style clients still surface something useful.
    """
    server_id = server.server_id
    display_name = server.alias or server.server_name or server.name or server_id

    config_path = f"/ui?page=mcp-servers&server_id={server_id}"
    if base_dashboard_url:
        # Strip trailing slash to avoid double-slash in the joined URL.
        base = base_dashboard_url.rstrip("/")
        config_url = f"{base}{config_path}"
    else:
        config_url = config_path

    missing_summary = [
        {
            "field_key": f.get("field_key"),
            "display_name": f.get("display_name") or f.get("field_key"),
            "description": f.get("description"),
        }
        for f in missing
    ]
    field_names = ", ".join(
        str(m["display_name"]) for m in missing_summary if m.get("display_name")
    )
    plural = "fields" if len(missing) != 1 else "field"
    message = (
        f"This MCP server ({display_name}) needs your {plural} before it can run: "
        f"{field_names}. Open {config_url} to fill them in, then retry the tool call."
    )
    return {
        "error": "user_fields_missing",
        "server_id": server_id,
        "server_name": server.server_name or server.name,
        "missing_fields": missing_summary,
        "config_url": config_url,
        "message": message,
    }


def resolve_user_field_headers(
    server: MCPServer, stored_values: Dict[str, str]
) -> Dict[str, str]:
    """Build the HTTP header dict to inject for an http/sse MCP server.

    Each declared field with a ``header_name`` contributes one header.
    ``header_value_template`` (default ``"{value}"``) lets admins inject
    well-known prefixes like ``"Bearer {value}"`` without making the user
    re-type them.
    """
    headers: Dict[str, str] = {}
    for entry in coerce_user_fields(server):
        field_key = entry.get("field_key")
        header_name = entry.get("header_name")
        if not field_key or not header_name:
            continue
        value = stored_values.get(field_key)
        if not value:
            continue
        template = entry.get("header_value_template") or "{value}"
        # Use plain substitution rather than ``str.format``: the latter
        # exposes attribute / item access (``{value.__class__}``, ``{value[0]}``)
        # which an admin-supplied template could use — intentionally or by
        # accident — to leak Python object internals into outbound HTTP
        # headers. ``str.replace`` only matches the literal ``{value}`` token.
        try:
            headers[header_name] = template.replace("{value}", value)
        except (TypeError, AttributeError):
            # Non-string template (e.g. corrupt JSONB row) — fall back to
            # raw value rather than crashing the request.
            verbose_logger.warning(
                "MCP user_fields: invalid header_value_template %r for field %r "
                "on server %s; falling back to raw value.",
                template,
                field_key,
                server.server_id,
            )
            headers[header_name] = value
    return headers


def resolve_user_field_env(
    server: MCPServer, stored_values: Dict[str, str]
) -> Dict[str, str]:
    """Build the env-var dict to inject for a stdio MCP server."""
    env: Dict[str, str] = {}
    for entry in coerce_user_fields(server):
        field_key = entry.get("field_key")
        env_var_name = entry.get("env_var_name")
        if not field_key or not env_var_name:
            continue
        value = stored_values.get(field_key)
        if not value:
            continue
        env[env_var_name] = value
    return env
