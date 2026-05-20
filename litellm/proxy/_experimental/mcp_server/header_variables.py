"""Header variable interpolation for MCP servers.

Admins configure variables on an MCP server (e.g. ``DB_PROTOCOL``,
``CORP_USERNAME``) with a scope of either ``global`` or ``per_user``. Global
values are stored on the server record. Per-user values are stored per
``(user_id, server_id)`` pair in ``LiteLLM_MCPUserCredentials`` under a JSON
payload tagged ``type: "header_variables"``. Static headers are interpolated
at call time by replacing ``${VAR_NAME}`` references.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.db import _decode_user_credential
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
from litellm.proxy.utils import PrismaClient
from litellm.types.mcp_server.mcp_server_manager import MCPServer

HEADER_VARIABLE_CREDENTIAL_TYPE = "header_variables"
_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _normalize_header_variables(
    raw: Any,
) -> List[Dict[str, Any]]:
    """Coerce a stored ``header_variables`` value into a clean list of dicts.

    Accepts the JSON list form, a JSON-encoded string, or ``None``. Drops
    entries without a ``name``. Unknown ``scope`` values default to ``global``.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        scope = entry.get("scope")
        if scope not in ("global", "per_user"):
            scope = "global"
        cleaned.append(
            {
                "name": name,
                "value": entry.get("value"),
                "scope": scope,
            }
        )
    return cleaned


def get_per_user_variable_names(server: MCPServer) -> List[str]:
    """Return the names of all per-user header variables defined on ``server``."""
    return [
        v["name"]
        for v in _normalize_header_variables(server.header_variables)
        if v["scope"] == "per_user"
    ]


def get_global_variable_values(server: MCPServer) -> Dict[str, str]:
    """Return a dict of {name: value} for all global header variables."""
    result: Dict[str, str] = {}
    for v in _normalize_header_variables(server.header_variables):
        if v["scope"] != "global":
            continue
        value = v.get("value")
        if value is None:
            continue
        result[v["name"]] = str(value)
    return result


def _decode_header_variables_payload(stored: str) -> Optional[Dict[str, str]]:
    """Return the decoded per-user header-variable values, or ``None``.

    A row is considered a header-variables credential iff its decoded value
    parses as a JSON object with ``"type": "header_variables"``.
    """
    decoded = _decode_user_credential(stored)
    if decoded is None:
        return None
    try:
        parsed = json.loads(decoded)
    except (ValueError, TypeError):
        return None
    if (
        not isinstance(parsed, dict)
        or parsed.get("type") != HEADER_VARIABLE_CREDENTIAL_TYPE
    ):
        return None
    values = parsed.get("values")
    if not isinstance(values, dict):
        return {}
    return {str(k): str(v) for k, v in values.items()}


async def get_user_header_variables(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
) -> Dict[str, str]:
    """Return the user's stored per-user header-variable values for ``server_id``."""
    row = await prisma_client.db.litellm_mcpusercredentials.find_unique(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
    )
    if row is None:
        return {}
    values = _decode_header_variables_payload(row.credential_b64)
    return values or {}


async def store_user_header_variables(
    prisma_client: PrismaClient,
    user_id: str,
    server_id: str,
    values: Dict[str, str],
) -> None:
    """Persist a user's per-user header-variable values for ``server_id``.

    Guards against accidentally overwriting a BYOK / OAuth credential stored
    under the same ``(user_id, server_id)`` pair.
    """
    existing = await prisma_client.db.litellm_mcpusercredentials.find_unique(
        where={"user_id_server_id": {"user_id": user_id, "server_id": server_id}}
    )
    if existing is not None:
        if _decode_header_variables_payload(existing.credential_b64) is None:
            raise ValueError(
                f"Existing credential for user {user_id} and server "
                f"{server_id} is not a header_variables payload. Refusing "
                f"to overwrite."
            )

    payload = {
        "type": HEADER_VARIABLE_CREDENTIAL_TYPE,
        "values": {str(k): str(v) for k, v in values.items()},
    }
    encoded = encrypt_value_helper(json.dumps(payload))
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


def interpolate_string(template: str, context: Dict[str, str]) -> Tuple[str, List[str]]:
    """Substitute ``${VAR}`` placeholders in ``template`` against ``context``.

    Returns ``(result, missing_vars)`` where ``missing_vars`` is the list of
    variable names that were referenced but not present in ``context``.
    """
    missing: List[str] = []

    def _replace(match: "re.Match[str]") -> str:
        var = match.group(1)
        if var in context:
            return context[var]
        missing.append(var)
        return match.group(0)

    result = _VAR_PATTERN.sub(_replace, template)
    return result, missing


def interpolate_static_headers(
    static_headers: Optional[Dict[str, str]],
    context: Dict[str, str],
) -> Tuple[Dict[str, str], List[str]]:
    """Substitute ``${VAR}`` placeholders in the values of ``static_headers``.

    Header *names* are not interpolated; only values. Returns the
    interpolated dict and a deduplicated list of missing variable names.
    """
    if not static_headers:
        return {}, []
    out: Dict[str, str] = {}
    missing_set: List[str] = []
    for key, value in static_headers.items():
        if not isinstance(value, str):
            out[key] = value  # type: ignore[assignment]
            continue
        new_value, missing = interpolate_string(value, context)
        out[key] = new_value
        for m in missing:
            if m not in missing_set:
                missing_set.append(m)
    return out, missing_set


def find_missing_per_user_variables(
    server: MCPServer,
    user_values: Dict[str, str],
) -> List[str]:
    """Return the per-user variable names referenced by ``server.static_headers``
    that the user has not yet provided values for.

    Variables that are declared per-user but never referenced in
    ``static_headers`` are not counted — they would not affect the call.
    """
    per_user_names = set(get_per_user_variable_names(server))
    if not per_user_names:
        return []
    referenced: List[str] = []
    for value in (server.static_headers or {}).values():
        if not isinstance(value, str):
            continue
        for match in _VAR_PATTERN.finditer(value):
            var = match.group(1)
            if var in per_user_names and var not in referenced:
                referenced.append(var)
    return [v for v in referenced if v not in user_values or not user_values[v]]


class MissingPerUserHeaderVariablesError(Exception):
    """Raised when a tool call cannot proceed because per-user header variables
    referenced by ``static_headers`` are not configured for the caller.

    The error message is intentionally user-facing — it surfaces a direct link
    to the dashboard page where the user can fill in the missing values.
    """

    def __init__(
        self,
        *,
        server_name: str,
        server_id: str,
        missing_variables: List[str],
        dashboard_url: str,
    ) -> None:
        self.server_name = server_name
        self.server_id = server_id
        self.missing_variables = missing_variables
        self.dashboard_url = dashboard_url
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        names = ", ".join(self.missing_variables)
        return (
            f"MCP server '{self.server_name}' is missing per-user header "
            f"variable(s): {names}. "
            f"Please set them in the LiteLLM dashboard: {self.dashboard_url}"
        )


def build_dashboard_url(server_id: str, proxy_base_url: Optional[str] = None) -> str:
    """Return a deep link to the dashboard page where a user can fill in
    per-user header variables for ``server_id``.

    ``proxy_base_url`` may be supplied for absolute links; otherwise a path-only
    URL is returned (suitable for same-origin error messages).
    """
    path = f"/ui/?page=mcp-servers&fill_variables_for={server_id}"
    if proxy_base_url:
        return f"{proxy_base_url.rstrip('/')}{path}"
    return path


async def resolve_static_headers_for_user(
    server: MCPServer,
    *,
    user_id: Optional[str],
    prisma_client: Optional[PrismaClient],
    proxy_base_url: Optional[str] = None,
) -> Dict[str, str]:
    """Return ``server.static_headers`` with ``${VAR}`` placeholders resolved.

    Pulls global values from the server record and per-user values from
    ``LiteLLM_MCPUserCredentials``. If a per-user variable is referenced but
    not configured for ``user_id``, raises
    :class:`MissingPerUserHeaderVariablesError` with a dashboard link.

    When the server has no header variables declared, returns
    ``server.static_headers`` unchanged.
    """
    static_headers = server.static_headers or {}
    variables = _normalize_header_variables(server.header_variables)
    if not variables:
        return dict(static_headers)

    context: Dict[str, str] = get_global_variable_values(server)

    per_user_names = [v["name"] for v in variables if v["scope"] == "per_user"]
    user_values: Dict[str, str] = {}
    if per_user_names and prisma_client is not None and user_id:
        try:
            user_values = await get_user_header_variables(
                prisma_client, user_id, server.server_id
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                "Failed to load per-user header variables for user=%s server=%s: %s",
                user_id,
                server.server_id,
                str(e),
            )
            user_values = {}
    context.update({k: v for k, v in user_values.items() if v})

    interpolated, missing = interpolate_static_headers(static_headers, context)

    # Only fail on missing per-user variables — a missing global variable is
    # an admin misconfiguration that we surface in logs but don't block on
    # (the literal ${VAR} string is sent upstream so the failure is visible).
    per_user_set = set(per_user_names)
    missing_per_user = [m for m in missing if m in per_user_set]
    if missing_per_user:
        raise MissingPerUserHeaderVariablesError(
            server_name=server.alias or server.server_name or server.name,
            server_id=server.server_id,
            missing_variables=missing_per_user,
            dashboard_url=build_dashboard_url(server.server_id, proxy_base_url),
        )

    if missing:
        verbose_proxy_logger.warning(
            "MCP server %s references undefined global header variables: %s",
            server.server_id,
            missing,
        )

    return interpolated
