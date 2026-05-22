"""Types, constants, and pure helpers shared by the config export/import endpoints."""

from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from fastapi import HTTPException
from pydantic import BaseModel

from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker

_MASKER = SensitiveDataMasker()

# Sections processed in dependency order:
# budgets and organizations must exist before teams reference them,
# users before keys, etc.
_IMPORT_ORDER: List[str] = [
    "budgets",
    "organizations",
    "teams",
    "users",
    "keys",
    "credentials",
    "models",
    "mcp_servers",
    "agents",
    "guardrails",
    "tags",
    "general_settings",
]

# ---------------------------------------------------------------------------
# General settings that are safe to round-trip across environments.
# Deployment-specific keys (database_url, master_key, host, port, etc.) are
# intentionally excluded.
# ---------------------------------------------------------------------------
SAFE_GENERAL_SETTINGS_KEYS = {
    "alerting",
    "alerting_threshold",
    "alert_types",
    "slack_alerting_settings",
    "smtp_settings",
    "default_team_settings",
    "default_max_internal_user_budget",
    "default_internal_user_budget_duration",
    "disable_adding_master_key_hash_to_db",
    "enforce_user_param",
    "allowed_routes",
    "max_parallel_requests",
    "global_max_parallel_requests",
    "infer_model_from_requests",
    "default_internal_user_params",
    "default_team_model_aliases",
    "upperbound_key_generate_params",
}

# Fields to strip from each entity type — operational/ephemeral state that
# should not be carried across environments.
_STRIP_FIELDS: Dict[str, List[str]] = {
    "budgets": [],
    "organizations": ["spend"],
    "teams": ["spend"],
    "users": ["spend", "user_api_key_hash", "password"],
    "keys": ["token", "spend"],
    "credentials": [],  # credential_values handled separately (redaction)
    "models": [],
    "mcp_servers": [],  # credentials handled separately (redaction)
    "agents": ["spend"],
    "guardrails": [],
    "tags": [],
}

ALL_SECTIONS = list(_STRIP_FIELDS.keys()) + ["general_settings"]

# Fields that must never appear in a keys UPDATE payload.
# token     — @id primary key, written once at creation, immutable thereafter.
# key_alias — used as the WHERE clause; must not also be in the data dict.
# spend     — operational counter managed by the proxy, not a configuration field.
_KEYS_UPDATE_STRIP: Set[str] = {"token", "key_alias", "spend"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LiteLLMExportEnvelope(BaseModel):
    exported_at: str
    source_instance: str
    include_filters: List[str]
    # row_limit is the per-section take value used when producing this snapshot.
    # A section whose length equals row_limit was likely truncated.
    row_limit: int = 1000

    budgets: Optional[List[Dict[str, Any]]] = None
    organizations: Optional[List[Dict[str, Any]]] = None
    teams: Optional[List[Dict[str, Any]]] = None
    users: Optional[List[Dict[str, Any]]] = None
    keys: Optional[List[Dict[str, Any]]] = None
    credentials: Optional[List[Dict[str, Any]]] = None
    models: Optional[List[Dict[str, Any]]] = None
    mcp_servers: Optional[List[Dict[str, Any]]] = None
    agents: Optional[List[Dict[str, Any]]] = None
    guardrails: Optional[List[Dict[str, Any]]] = None
    tags: Optional[List[Dict[str, Any]]] = None
    general_settings: Optional[Dict[str, Any]] = None
    # Sections where len(rows) == row_limit — the export was likely truncated.
    truncated_sections: List[str] = []


class ImportSectionResult(BaseModel):
    created: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    total_processed: int = 0
    warnings: List[str] = []


class ImportResult(BaseModel):
    dry_run: bool
    conflict: str
    sections_attempted: List[str] = []
    budgets: ImportSectionResult = ImportSectionResult()
    organizations: ImportSectionResult = ImportSectionResult()
    teams: ImportSectionResult = ImportSectionResult()
    users: ImportSectionResult = ImportSectionResult()
    keys: ImportSectionResult = ImportSectionResult()
    credentials: ImportSectionResult = ImportSectionResult()
    models: ImportSectionResult = ImportSectionResult()
    mcp_servers: ImportSectionResult = ImportSectionResult()
    agents: ImportSectionResult = ImportSectionResult()
    guardrails: ImportSectionResult = ImportSectionResult()
    tags: ImportSectionResult = ImportSectionResult()
    general_settings: ImportSectionResult = ImportSectionResult()


class ImportRequest(BaseModel):
    data: LiteLLMExportEnvelope
    conflict: Literal["skip", "replace", "merge"] = "skip"
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip(record: Any, fields: List[str]) -> Dict[str, Any]:
    """Return a copy of the record (Prisma model or dict) with fields removed.

    None values are preserved intentionally.  Dropping them would break
    replace-mode import: a field that is NULL in the source would be absent
    from the export payload and therefore absent from the UPDATE statement,
    silently leaving the target's existing non-null value in place instead of
    overwriting it with NULL.
    """
    if hasattr(record, "model_dump"):
        d = record.model_dump()
    elif hasattr(record, "dict"):
        d = record.dict()
    elif isinstance(record, dict):
        d = dict(record)
    else:
        d = dict(record)
    for f in fields:
        d.pop(f, None)
    return d


def _redact_credential_values(record: Dict[str, Any]) -> Dict[str, Any]:
    """Replace credential_values with a redacted placeholder."""
    record = dict(record)
    if "credential_values" in record:
        record["credential_values"] = {"__redacted__": True}
    return record


def _redact_mcp_credentials(record: Dict[str, Any]) -> Dict[str, Any]:
    """Replace MCP server credentials with a redacted placeholder."""
    record = dict(record)
    if "credentials" in record and record["credentials"]:
        record["credentials"] = {"__redacted__": True}
    return record


def _redact_litellm_params(record: Dict[str, Any]) -> Dict[str, Any]:
    """Redact credential sub-fields inside a litellm_params dict.

    Uses SensitiveDataMasker pattern-based detection to replace secret keys
    with the string sentinel ``"__redacted__"`` while leaving non-secret config
    fields (api_base, region_name, etc.) intact so the export remains useful
    for environment promotion.
    """
    params = record.get("litellm_params")
    if not isinstance(params, dict):
        return record
    redacted_params = {
        k: "__redacted__" if _MASKER.is_sensitive_key(k) and v is not None else v
        for k, v in params.items()
    }
    return {**record, "litellm_params": redacted_params}


def _clean_litellm_params_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Strip string-sentinel sub-fields from litellm_params before a DB write.

    _redact_litellm_params marks secret sub-fields with the plain string
    ``"__redacted__"``.  If a redacted snapshot is imported those strings would
    be written into the target DB, making models/agents non-functional and
    silently overwriting working values in replace/merge mode.  This helper
    removes any sub-field whose value is the sentinel so only non-secret config
    fields (api_base, model, region_name, …) reach Prisma.
    """
    params = record.get("litellm_params")
    if not isinstance(params, dict):
        return record
    cleaned = {k: v for k, v in params.items() if v != "__redacted__"}
    return {**record, "litellm_params": cleaned}


def _redact_sensitive_header_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    ``static_headers`` can contain Authorization headers.
    ``env`` can contain environment variable values including secrets.
    ``url`` and ``spec_path`` are connectivity config, not credentials — kept.
    """
    record = dict(record)
    if record.get("static_headers"):
        record["static_headers"] = {"__redacted__": True}
    if record.get("env"):
        record["env"] = {"__redacted__": True}
    return record


def _is_redacted(value: Any) -> bool:
    return isinstance(value, dict) and value.get("__redacted__") is True


# ---------------------------------------------------------------------------
# Envelope validation (runs before any DB writes)
# ---------------------------------------------------------------------------


def _validate_envelope(data: LiteLLMExportEnvelope) -> None:
    """
    Validate the snapshot before touching the DB.

    Checks:
    - required ID fields are present and non-empty
    - no duplicate IDs within a section (would cause undefined upsert order)
    - basic type sanity on a selection of critical fields

    Raises HTTPException(400) with a descriptive message on first error.
    """
    errors: List[str] = []

    def _check_ids(
        section_name: str, records: Optional[List[Dict[str, Any]]], id_field: str
    ) -> None:
        if records is None:
            return
        seen: Set[str] = set()
        for i, rec in enumerate(records):
            val = rec.get(id_field)
            if not val:
                errors.append(
                    f"{section_name}[{i}]: missing required field '{id_field}'"
                )
                continue
            if val in seen:
                errors.append(f"{section_name}: duplicate {id_field}='{val}'")
            seen.add(str(val))

    _check_ids("budgets", data.budgets, "budget_id")
    _check_ids("organizations", data.organizations, "organization_id")
    _check_ids("teams", data.teams, "team_id")
    _check_ids("users", data.users, "user_id")
    _check_ids("credentials", data.credentials, "credential_name")
    _check_ids("models", data.models, "model_id")
    _check_ids("mcp_servers", data.mcp_servers, "server_id")
    _check_ids("agents", data.agents, "agent_name")
    _check_ids("guardrails", data.guardrails, "guardrail_name")

    # keys: key_alias is optional (keyless keys are skipped at import time)
    # — we only warn about them during import, not here

    if errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "Snapshot validation failed", "errors": errors},
        )


# ---------------------------------------------------------------------------
# Cross-entity dependency validation
# ---------------------------------------------------------------------------


def _validate_dependencies(data: LiteLLMExportEnvelope) -> None:
    """
    Verify that cross-entity references within the snapshot are satisfied.

    For example: a team that references an organization_id must have that
    organization present in the snapshot (or we assume it already exists in
    the target — so this only fires when BOTH sections are present).

    Raises HTTPException(400) listing all broken references.
    """
    errors: List[str] = []

    org_ids: Set[str] = {
        r["organization_id"]
        for r in (data.organizations or [])
        if r.get("organization_id")
    }
    team_ids: Set[str] = {r["team_id"] for r in (data.teams or []) if r.get("team_id")}
    user_ids: Set[str] = {r["user_id"] for r in (data.users or []) if r.get("user_id")}
    budget_ids: Set[str] = {
        r["budget_id"] for r in (data.budgets or []) if r.get("budget_id")
    }

    # Teams referencing orgs — only validate when the orgs section is present in
    # the snapshot (None means section was not requested/included).
    if data.teams and data.organizations is not None:
        for team in data.teams:
            oid = team.get("organization_id")
            if oid and oid not in org_ids:
                errors.append(
                    f"teams[team_id={team.get('team_id')}]: "
                    f"references organization_id='{oid}' not found in snapshot"
                )

    # Orgs referencing budgets — only validate when the budgets section is present.
    if data.organizations and data.budgets is not None:
        for org in data.organizations:
            bid = org.get("budget_id")
            if bid and bid not in budget_ids:
                errors.append(
                    f"organizations[organization_id={org.get('organization_id')}]: "
                    f"references budget_id='{bid}' not found in snapshot"
                )

    # Keys referencing teams / users.
    # Only validate when the referenced section was included in the snapshot.
    # Skip keys with no key_alias — they are skipped during import anyway
    # (e.g. internal/system keys like the litellm-dashboard master key).
    if data.keys and data.teams is not None:
        for key in data.keys:
            if key.get("key_alias") is None:
                continue  # will be skipped during import; skip validation too
            tid = key.get("team_id")
            if tid and tid not in team_ids:
                errors.append(
                    f"keys[key_alias={key.get('key_alias')}]: "
                    f"references team_id='{tid}' not found in snapshot"
                )
    if data.keys and data.users is not None:
        for key in data.keys:
            if key.get("key_alias") is None:
                continue  # will be skipped during import
            uid = key.get("user_id")
            if uid and uid not in user_ids:
                errors.append(
                    f"keys[key_alias={key.get('key_alias')}]: "
                    f"references user_id='{uid}' not found in snapshot"
                )

    if errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "Dependency validation failed", "errors": errors},
        )


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Iteratively merge `override` into `base`.  For dict values, descend;
    for everything else, override wins.  Neither input is mutated.

    Implemented with an explicit stack to avoid recursion (the repo's circular-
    import / CPU-spike checker flags recursive functions as unacceptable).
    """
    # Each stack frame is (target_dict, base_dict, override_dict).
    # We build the result top-down and patch nested dicts in-place as we go.
    result: Dict[str, Any] = dict(base)
    stack: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = [
        (result, base, override)
    ]
    while stack:
        target, b, o = stack.pop()
        for key, val in o.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(val, dict)
            ):
                # Create a fresh copy of the nested base dict and schedule it
                # for merging rather than calling _deep_merge again.
                nested: Dict[str, Any] = dict(b.get(key, {}))
                target[key] = nested
                stack.append((nested, b.get(key, {}), val))
            else:
                target[key] = val
    return result


# ---------------------------------------------------------------------------
# Batch DB read helper
# ---------------------------------------------------------------------------


async def _load_existing(
    table: Any,
    id_field: str,
    ids: List[Any],
) -> Dict[str, Any]:
    """
    Fetch all records matching `ids` in a single query and return an
    id → record dict.  Eliminates N+1 `find_unique` calls.
    """
    if not ids:
        return {}
    rows = await table.find_many(where={id_field: {"in": ids}})
    result: Dict[str, Any] = {}
    for row in rows:
        if hasattr(row, "model_dump"):
            d = row.model_dump()
        elif hasattr(row, "dict"):
            d = row.dict()
        else:
            d = dict(row)
        key = d.get(id_field)
        if key is not None:
            result[str(key)] = d
    return result


# ---------------------------------------------------------------------------
# Section list resolver (used by export endpoint)
# ---------------------------------------------------------------------------


def _resolve_sections(include: Optional[str]) -> Set[str]:
    """Parse and validate the ?include= query param; return the set of sections."""
    if not include:
        return set(ALL_SECTIONS)
    requested = {s.strip() for s in include.split(",")}
    unknown = requested - set(ALL_SECTIONS)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sections: {unknown}. Valid sections: {ALL_SECTIONS}",
        )
    return requested
