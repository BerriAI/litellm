"""
GET /config/export  — export all UI/API-managed state as a versioned, re-applicable artifact
POST /config/import — idempotent re-apply of an exported snapshot
"""

import datetime
import json
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()

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
    "users": ["spend", "user_api_key_hash"],
    "keys": ["token", "spend"],
    "credentials": [],  # credential_values handled separately (redaction)
    "models": [],
    "mcp_servers": [],  # credentials handled separately (redaction)
    "agents": ["spend"],
    "guardrails": [],
    "tags": [],
}

ALL_SECTIONS = list(_STRIP_FIELDS.keys()) + ["general_settings"]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LiteLLMExportEnvelope(BaseModel):
    exported_at: str
    source_instance: str
    include_filters: List[str]

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
    """Return a copy of the record (Prisma model or dict) with fields removed."""
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
    # Remove None values to keep the export compact
    return {k: v for k, v in d.items() if v is not None}


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
    Recursively merge `override` into `base`.  For dict values, descend;
    for everything else, override wins.  Neither input is mutated.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Batch DB read helper
# ---------------------------------------------------------------------------


async def _load_existing(
    table: Any,
    id_field: str,
    ids: List[str],
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
# GET /config/export
# ---------------------------------------------------------------------------


@router.get(
    "/config/export",
    tags=["config.yaml"],
    dependencies=[Depends(user_api_key_auth)],
    summary="Export all UI/API-managed proxy state as a versioned artifact",
)
async def export_config(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    include: Optional[str] = Query(
        default=None,
        description="Comma-separated list of sections to include. "
        f"Available: {', '.join(ALL_SECTIONS)}. Defaults to all.",
    ),
    format: Literal["json", "yaml"] = Query(
        default="json",
        description="Output format: json or yaml",
    ),
    redact_secrets: bool = Query(
        default=True,
        description="When true, replaces credential_values and MCP server "
        "credentials with {__redacted__: true}.",
    ),
) -> Response:
    """
    Returns a versioned snapshot of all UI/API-managed proxy state.

    The snapshot is re-applicable via POST /config/import and can be used
    for multi-environment promotion (dev → staging → prod) or disaster recovery.

    **Excluded from export (by design):**
    - Spend logs, audit logs, error logs (operational history)
    - Master key / salt key material
    - End-user records (LiteLLM_EndUserTable)
    - Deployment-specific settings (database_url, host, port, etc.)
    """
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Only proxy admins can export config",
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail="Database not connected. Cannot export config.",
        )

    # Resolve which sections to include
    if include:
        requested = {s.strip() for s in include.split(",")}
        unknown = requested - set(ALL_SECTIONS)
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown sections: {unknown}. Valid sections: {ALL_SECTIONS}",
            )
        sections = requested
    else:
        sections = set(ALL_SECTIONS)

    envelope: Dict[str, Any] = {
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_instance": str(request.base_url).rstrip("/"),
        "include_filters": sorted(sections),
    }

    try:
        if "budgets" in sections:
            rows = await prisma_client.db.litellm_budgettable.find_many()
            envelope["budgets"] = [_strip(r, _STRIP_FIELDS["budgets"]) for r in rows]

        if "organizations" in sections:
            rows = await prisma_client.db.litellm_organizationtable.find_many(
                include={"litellm_budget_table": True}
            )
            envelope["organizations"] = [
                _strip(r, _STRIP_FIELDS["organizations"]) for r in rows
            ]

        if "teams" in sections:
            rows = await prisma_client.db.litellm_teamtable.find_many()
            envelope["teams"] = [_strip(r, _STRIP_FIELDS["teams"]) for r in rows]

        if "users" in sections:
            rows = await prisma_client.db.litellm_usertable.find_many()
            envelope["users"] = [_strip(r, _STRIP_FIELDS["users"]) for r in rows]

        if "keys" in sections:
            rows = await prisma_client.db.litellm_verificationtoken.find_many()
            envelope["keys"] = [_strip(r, _STRIP_FIELDS["keys"]) for r in rows]

        if "credentials" in sections:
            rows = await prisma_client.db.litellm_credentialstable.find_many()
            records = [_strip(r, _STRIP_FIELDS["credentials"]) for r in rows]
            if redact_secrets:
                records = [_redact_credential_values(rec) for rec in records]
            envelope["credentials"] = records

        if "models" in sections:
            rows = await prisma_client.db.litellm_proxymodeltable.find_many()
            envelope["models"] = [_strip(r, _STRIP_FIELDS["models"]) for r in rows]

        if "mcp_servers" in sections:
            rows = await prisma_client.db.litellm_mcpservertable.find_many()
            records = [_strip(r, _STRIP_FIELDS["mcp_servers"]) for r in rows]
            if redact_secrets:
                records = [_redact_mcp_credentials(rec) for rec in records]
            envelope["mcp_servers"] = records

        if "agents" in sections:
            rows = await prisma_client.db.litellm_agentstable.find_many()
            envelope["agents"] = [_strip(r, _STRIP_FIELDS["agents"]) for r in rows]

        if "guardrails" in sections:
            rows = await prisma_client.db.litellm_guardrailstable.find_many()
            envelope["guardrails"] = [
                _strip(r, _STRIP_FIELDS["guardrails"]) for r in rows
            ]

        if "tags" in sections:
            rows = await prisma_client.db.litellm_tagtable.find_many()
            envelope["tags"] = [_strip(r, _STRIP_FIELDS["tags"]) for r in rows]

        if "general_settings" in sections:
            rows = await prisma_client.db.litellm_config.find_many(
                where={"param_name": {"in": list(SAFE_GENERAL_SETTINGS_KEYS)}}
            )
            envelope["general_settings"] = {
                r.param_name: r.param_value for r in rows if r.param_value is not None
            }

    except Exception as e:
        verbose_proxy_logger.error(f"config/export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

    if format == "yaml":
        content = yaml.dump(envelope, allow_unicode=True, sort_keys=False)
        return Response(content=content, media_type="application/yaml")

    import json

    content = json.dumps(envelope, indent=2, default=str)
    return Response(content=content, media_type="application/json")


# ---------------------------------------------------------------------------
# POST /config/import
# ---------------------------------------------------------------------------


@router.post(
    "/config/import",
    tags=["config.yaml"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ImportResult,
    summary="Idempotently re-apply a config snapshot produced by GET /config/export",
)
async def import_config(
    request: Request,
    body: ImportRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> ImportResult:
    """
    Re-applies a versioned config snapshot exported by GET /config/export.

    **Conflict modes:**
    - `skip` (default) — leave existing records unchanged
    - `replace` — overwrite existing records with snapshot values
    - `merge` — deep-merge snapshot values into existing records

    **dry_run=true** — simulate the full import (including DB reads) and
    report exactly what would be created/updated/skipped without writing.

    **Redacted secrets:** credentials and MCP server credentials marked with
    `{__redacted__: true}` are skipped. Bind secrets manually after import.

    **Transaction model:** each section is wrapped in its own DB transaction.
    If a section fails, that section is rolled back; already-committed sections
    are not affected.  The result object reports per-section outcomes.
    """
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Only proxy admins can import config",
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail="Database not connected. Cannot import config.",
        )

    # Structural + dependency validation — fail fast before touching DB
    data = body.data
    _validate_envelope(data)
    _validate_dependencies(data)

    conflict = body.conflict
    dry_run = body.dry_run
    result = ImportResult(dry_run=dry_run, conflict=conflict)
    triggered_by = getattr(user_api_key_dict, "user_id", None) or getattr(
        user_api_key_dict, "api_key", "unknown"
    )

    verbose_proxy_logger.info(
        "config/import started: triggered_by=%s dry_run=%s conflict=%s "
        "source=%s exported_at=%s",
        triggered_by,
        dry_run,
        conflict,
        data.source_instance,
        data.exported_at,
    )

    try:
        # ------------------------------------------------------------------ #
        # Process sections in dependency order.
        # Each section is wrapped in its own transaction so a failure in one
        # section does not roll back already-committed sections.
        # Trade-off vs a single transaction:
        #   + shorter lock window per section
        #   + partial success is recoverable and clearly reported
        #   - not fully atomic across sections (acceptable for config promotion)
        # ------------------------------------------------------------------ #

        # -- budgets --
        if data.budgets is not None:
            result.sections_attempted.append("budgets")
            ids = [r.get("budget_id") for r in data.budgets if r.get("budget_id")]
            existing_map = await _load_existing(
                prisma_client.db.litellm_budgettable, "budget_id", ids
            )
            await _import_section(
                table=prisma_client.db.litellm_budgettable,
                records=data.budgets,
                id_field="budget_id",
                conflict=conflict,
                dry_run=dry_run,
                section_result=result.budgets,
                existing_map=existing_map,
            )

        # -- organizations --
        if data.organizations is not None:
            result.sections_attempted.append("organizations")
            ids = [
                r.get("organization_id")
                for r in data.organizations
                if r.get("organization_id")
            ]
            existing_map = await _load_existing(
                prisma_client.db.litellm_organizationtable, "organization_id", ids
            )
            await _import_section(
                table=prisma_client.db.litellm_organizationtable,
                records=data.organizations,
                id_field="organization_id",
                conflict=conflict,
                dry_run=dry_run,
                section_result=result.organizations,
                existing_map=existing_map,
            )

        # -- teams --
        if data.teams is not None:
            result.sections_attempted.append("teams")
            ids = [r.get("team_id") for r in data.teams if r.get("team_id")]
            existing_map = await _load_existing(
                prisma_client.db.litellm_teamtable, "team_id", ids
            )
            await _import_section(
                table=prisma_client.db.litellm_teamtable,
                records=data.teams,
                id_field="team_id",
                conflict=conflict,
                dry_run=dry_run,
                section_result=result.teams,
                existing_map=existing_map,
            )

        # -- users --
        if data.users is not None:
            result.sections_attempted.append("users")
            ids = [r.get("user_id") for r in data.users if r.get("user_id")]
            existing_map = await _load_existing(
                prisma_client.db.litellm_usertable, "user_id", ids
            )
            await _import_section(
                table=prisma_client.db.litellm_usertable,
                records=data.users,
                id_field="user_id",
                conflict=conflict,
                dry_run=dry_run,
                section_result=result.users,
                existing_map=existing_map,
            )

        # -- keys — upsert by key_alias; skip those without one --
        if data.keys is not None:
            result.sections_attempted.append("keys")
            importable_keys: List[Dict[str, Any]] = []
            for rec in data.keys:
                if not rec.get("key_alias"):
                    result.keys.skipped += 1
                    result.keys.total_processed += 1
                    result.keys.warnings.append(
                        "Skipped key with no key_alias — cannot re-import without a stable identifier"
                    )
                else:
                    importable_keys.append(rec)
            aliases = [r["key_alias"] for r in importable_keys]
            existing_map = await _load_existing(
                prisma_client.db.litellm_verificationtoken, "key_alias", aliases
            )
            await _import_section(
                table=prisma_client.db.litellm_verificationtoken,
                records=importable_keys,
                id_field="key_alias",
                conflict=conflict,
                dry_run=dry_run,
                section_result=result.keys,
                existing_map=existing_map,
            )

        # -- credentials — skip redacted ones --
        if data.credentials is not None:
            result.sections_attempted.append("credentials")
            importable_creds: List[Dict[str, Any]] = []
            for rec in data.credentials:
                if _is_redacted(rec.get("credential_values")):
                    result.credentials.skipped += 1
                    result.credentials.total_processed += 1
                    result.credentials.warnings.append(
                        f"Skipped credential '{rec.get('credential_name')}' — "
                        "credential_values is redacted. Bind secrets manually."
                    )
                else:
                    importable_creds.append(rec)
            names = [
                r["credential_name"]
                for r in importable_creds
                if r.get("credential_name")
            ]
            existing_map = await _load_existing(
                prisma_client.db.litellm_credentialstable, "credential_name", names
            )
            await _import_section(
                table=prisma_client.db.litellm_credentialstable,
                records=importable_creds,
                id_field="credential_name",
                conflict=conflict,
                dry_run=dry_run,
                section_result=result.credentials,
                existing_map=existing_map,
            )

        # -- models --
        if data.models is not None:
            result.sections_attempted.append("models")
            ids = [r.get("model_id") for r in data.models if r.get("model_id")]
            existing_map = await _load_existing(
                prisma_client.db.litellm_proxymodeltable, "model_id", ids
            )
            await _import_section(
                table=prisma_client.db.litellm_proxymodeltable,
                records=data.models,
                id_field="model_id",
                conflict=conflict,
                dry_run=dry_run,
                section_result=result.models,
                existing_map=existing_map,
            )

        # -- mcp_servers — strip redacted credentials before import --
        if data.mcp_servers is not None:
            result.sections_attempted.append("mcp_servers")
            cleaned_servers: List[Dict[str, Any]] = []
            for rec in data.mcp_servers:
                if _is_redacted(rec.get("credentials")):
                    result.mcp_servers.warnings.append(
                        f"MCP server '{rec.get('server_name')}' imported without credentials — "
                        "credentials are redacted. Bind manually."
                    )
                    rec = {k: v for k, v in rec.items() if k != "credentials"}
                cleaned_servers.append(rec)
            ids = [r.get("server_id") for r in cleaned_servers if r.get("server_id")]
            existing_map = await _load_existing(
                prisma_client.db.litellm_mcpservertable, "server_id", ids
            )
            await _import_section(
                table=prisma_client.db.litellm_mcpservertable,
                records=cleaned_servers,
                id_field="server_id",
                conflict=conflict,
                dry_run=dry_run,
                section_result=result.mcp_servers,
                existing_map=existing_map,
            )

        # -- agents --
        if data.agents is not None:
            result.sections_attempted.append("agents")
            ids = [r.get("agent_name") for r in data.agents if r.get("agent_name")]
            existing_map = await _load_existing(
                prisma_client.db.litellm_agentstable, "agent_name", ids
            )
            await _import_section(
                table=prisma_client.db.litellm_agentstable,
                records=data.agents,
                id_field="agent_name",
                conflict=conflict,
                dry_run=dry_run,
                section_result=result.agents,
                existing_map=existing_map,
            )

        # -- guardrails --
        if data.guardrails is not None:
            result.sections_attempted.append("guardrails")
            ids = [
                r.get("guardrail_name")
                for r in data.guardrails
                if r.get("guardrail_name")
            ]
            existing_map = await _load_existing(
                prisma_client.db.litellm_guardrailstable, "guardrail_name", ids
            )
            await _import_section(
                table=prisma_client.db.litellm_guardrailstable,
                records=data.guardrails,
                id_field="guardrail_name",
                conflict=conflict,
                dry_run=dry_run,
                section_result=result.guardrails,
                existing_map=existing_map,
            )

        # -- tags --
        if data.tags is not None:
            result.sections_attempted.append("tags")
            ids = [r.get("tag_name") for r in data.tags if r.get("tag_name")]
            existing_map = await _load_existing(
                prisma_client.db.litellm_tagtable, "tag_name", ids
            )
            await _import_section(
                table=prisma_client.db.litellm_tagtable,
                records=data.tags,
                id_field="tag_name",
                conflict=conflict,
                dry_run=dry_run,
                section_result=result.tags,
                existing_map=existing_map,
            )

        # -- general_settings — handled separately (key/value table, not entity) --
        if data.general_settings is not None:
            result.sections_attempted.append("general_settings")
            await _import_general_settings(
                prisma_client=prisma_client,
                settings=data.general_settings,
                conflict=conflict,
                dry_run=dry_run,
                section_result=result.general_settings,
            )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error("config/import failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

    # Audit log summary
    verbose_proxy_logger.info(
        "config/import complete: triggered_by=%s dry_run=%s sections=%s "
        "totals=created:%d updated:%d skipped:%d errors:%d",
        triggered_by,
        dry_run,
        result.sections_attempted,
        sum(
            getattr(result, s).created
            for s in result.sections_attempted
            if s != "general_settings"
        ),
        sum(
            getattr(result, s).updated
            for s in result.sections_attempted
            if s != "general_settings"
        ),
        sum(
            getattr(result, s).skipped
            for s in result.sections_attempted
            if s != "general_settings"
        ),
        sum(
            getattr(result, s).errors
            for s in result.sections_attempted
            if s != "general_settings"
        ),
    )

    return result


# ---------------------------------------------------------------------------
# #1 — Per-section transaction wrapper
# ---------------------------------------------------------------------------


async def _import_section(
    table: Any,
    records: List[Dict[str, Any]],
    id_field: str,
    conflict: str,
    dry_run: bool,
    section_result: ImportSectionResult,
    existing_map: Dict[str, Any],
    id_query_field: Optional[str] = None,
) -> None:
    """
    Process all records for one section inside a single DB transaction.

    If any record raises an unhandled exception the entire section is rolled
    back and the error is surfaced in section_result.  Per-record errors
    (e.g. constraint violations) are caught by _upsert and counted without
    aborting the section.

    dry_run: skips the transaction context entirely.
    """
    if dry_run:
        for rec in records:
            await _upsert(
                table=table,
                rec=rec,
                id_field=id_field,
                conflict=conflict,
                dry_run=True,
                section_result=section_result,
                existing_map=existing_map,
                id_query_field=id_query_field,
            )
        return

    try:
        async with table._client.tx() as tx:
            tx_table = getattr(tx, table.__class__.__name__.lower(), table)
            for rec in records:
                await _upsert(
                    table=tx_table,
                    rec=rec,
                    id_field=id_field,
                    conflict=conflict,
                    dry_run=False,
                    section_result=section_result,
                    existing_map=existing_map,
                    id_query_field=id_query_field,
                )
    except AttributeError:
        # Prisma client doesn't expose .tx() on individual table objects —
        # fall back to non-transactional writes (maintains backward compat
        # with older prisma-client-py versions).
        verbose_proxy_logger.warning(
            "Prisma transaction not available for this table; "
            "falling back to non-transactional writes."
        )
        for rec in records:
            await _upsert(
                table=table,
                rec=rec,
                id_field=id_field,
                conflict=conflict,
                dry_run=False,
                section_result=section_result,
                existing_map=existing_map,
                id_query_field=id_query_field,
            )
    except Exception as e:
        section_result.errors += len(records)
        section_result.warnings.append(f"Section transaction rolled back: {e}")
        verbose_proxy_logger.error("Section transaction failed: %s", e, exc_info=True)


async def _import_general_settings(
    prisma_client: Any,
    settings: Dict[str, Any],
    conflict: str,
    dry_run: bool,
    section_result: ImportSectionResult,
) -> None:
    """
    Import general_settings key/value rows.
    Handles its own transaction because the table structure differs
    from entity tables (param_name PK, no numeric id field).
    """
    # Batch-read all relevant existing rows in one query
    safe_keys = [k for k in settings if k in SAFE_GENERAL_SETTINGS_KEYS]
    unsafe_keys = [k for k in settings if k not in SAFE_GENERAL_SETTINGS_KEYS]
    for k in unsafe_keys:
        section_result.skipped += 1
        section_result.total_processed += 1
        section_result.warnings.append(
            f"Skipped general_settings key '{k}' — not in safe export allow-list"
        )

    if not safe_keys:
        return

    existing_rows = await prisma_client.db.litellm_config.find_many(
        where={"param_name": {"in": safe_keys}}
    )
    existing_map = {r.param_name: r for r in existing_rows}

    def _apply(param_name: str, param_value: Any) -> Tuple[str, Any]:
        existing = existing_map.get(param_name)
        if existing is None:
            return ("create", param_value)
        if conflict == "skip":
            return ("skip", None)
        if (
            conflict == "merge"
            and isinstance(existing.param_value, dict)
            and isinstance(param_value, dict)
        ):
            return ("update", _deep_merge(existing.param_value, param_value))
        return ("update", param_value)

    if dry_run:
        for k in safe_keys:
            section_result.total_processed += 1
            action, _ = _apply(k, settings[k])
            if action == "create":
                section_result.created += 1
            elif action == "skip":
                section_result.skipped += 1
            else:
                section_result.updated += 1
        return

    for param_name in safe_keys:
        section_result.total_processed += 1
        action, value = _apply(param_name, settings[param_name])
        try:
            if action == "create":
                await prisma_client.db.litellm_config.create(
                    data={"param_name": param_name, "param_value": value}
                )
                section_result.created += 1
            elif action == "skip":
                section_result.skipped += 1
            else:
                await prisma_client.db.litellm_config.update(
                    where={"param_name": param_name},
                    data={"param_value": value},
                )
                section_result.updated += 1
        except Exception as e:
            section_result.errors += 1
            section_result.warnings.append(
                f"Error importing general_settings[{param_name}]: {e}"
            )


# ---------------------------------------------------------------------------
# Generic upsert helper — uses pre-loaded cache (no N+1 queries)
# ---------------------------------------------------------------------------


async def _upsert(
    table: Any,
    rec: Dict[str, Any],
    id_field: str,
    conflict: str,
    dry_run: bool,
    section_result: ImportSectionResult,
    existing_map: Dict[str, Any],
    id_query_field: Optional[str] = None,
) -> None:
    """
    Upsert a single record into `table`.

    id_field       — the dict key in `rec` that holds the stable identifier
    id_query_field — the Prisma `where` field name (defaults to id_field)
    existing_map   — pre-loaded {id: record} map (avoids N+1 find_unique calls)

    In dry_run mode: reads from existing_map but never writes.
    """
    id_query_field = id_query_field or id_field
    record_id = rec.get(id_field)
    section_result.total_processed += 1

    if not record_id:
        section_result.skipped += 1
        section_result.warnings.append(f"Skipped record with missing {id_field}: {rec}")
        return

    existing = existing_map.get(str(record_id))

    # Accurate dry-run: simulate actual outcome without writing
    if dry_run:
        if existing is None:
            section_result.created += 1
        elif conflict == "skip":
            section_result.skipped += 1
        else:
            section_result.updated += 1
        return

    try:
        if existing is None:
            await table.create(data=rec)
            section_result.created += 1
        elif conflict == "skip":
            section_result.skipped += 1
        elif conflict == "replace":
            update_data = {k: v for k, v in rec.items() if k != id_query_field}
            update_data.pop(id_field, None)
            await table.update(where={id_query_field: record_id}, data=update_data)
            section_result.updated += 1
        elif conflict == "merge":
            merged = _deep_merge(existing, rec)
            update_data = {k: v for k, v in merged.items() if k != id_query_field}
            update_data.pop(id_field, None)
            await table.update(where={id_query_field: record_id}, data=update_data)
            section_result.updated += 1
    except Exception as e:
        section_result.errors += 1
        section_result.warnings.append(f"Error upserting {id_field}={record_id}: {e}")
