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


# Fields inside litellm_params that hold provider credentials.
# Non-secret config (api_base, api_version, region_name, vertex_project, etc.)
# is left intact so the export can be re-applied to a target environment.
_LITELLM_PARAMS_SECRET_KEYS: frozenset = frozenset(
    {
        "api_key",
        "aws_access_key_id",
        "aws_secret_access_key",
        "vertex_credentials",
    }
)


def _redact_litellm_params(record: Dict[str, Any]) -> Dict[str, Any]:
    """Redact credential sub-fields inside a litellm_params dict.

    Replaces known secret keys with the string sentinel ``"__redacted__"``
    while leaving non-secret config fields (api_base, region_name, etc.)
    in place so the export remains useful for environment promotion.
    """
    params = record.get("litellm_params")
    if not isinstance(params, dict):
        return record
    redacted_params = {
        k: "__redacted__" if k in _LITELLM_PARAMS_SECRET_KEYS and v is not None else v
        for k, v in params.items()
    }
    return {**record, "litellm_params": redacted_params}


def _redact_mcp_sensitive_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Redact MCP server fields that can carry auth values.

    ``credentials`` is handled by _redact_mcp_credentials.
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
# Helpers for export_config and import_config (extracted to satisfy PLR0915)
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


async def _fetch_export_sections(
    prisma_client: Any,
    sections: Set[str],
    limit: int,
    redact_secrets: bool,
    envelope: Dict[str, Any],
) -> None:
    """Fetch each requested section from the DB and write it into envelope."""

    def _cap(section: str, rows: list) -> list:
        if len(rows) == limit:
            envelope["truncated_sections"].append(section)
        return rows

    if "budgets" in sections:
        rows = _cap(
            "budgets",
            await prisma_client.db.litellm_budgettable.find_many(
                take=limit, order={"created_at": "asc"}
            ),
        )
        envelope["budgets"] = [_strip(r, _STRIP_FIELDS["budgets"]) for r in rows]

    if "organizations" in sections:
        rows = _cap(
            "organizations",
            await prisma_client.db.litellm_organizationtable.find_many(
                take=limit, order={"created_at": "asc"}
            ),
        )
        envelope["organizations"] = [
            _strip(r, _STRIP_FIELDS["organizations"]) for r in rows
        ]

    if "teams" in sections:
        rows = _cap(
            "teams",
            await prisma_client.db.litellm_teamtable.find_many(
                take=limit, order={"created_at": "asc"}
            ),
        )
        envelope["teams"] = [_strip(r, _STRIP_FIELDS["teams"]) for r in rows]

    if "users" in sections:
        rows = _cap(
            "users",
            await prisma_client.db.litellm_usertable.find_many(
                take=limit, order={"created_at": "asc"}
            ),
        )
        envelope["users"] = [_strip(r, _STRIP_FIELDS["users"]) for r in rows]

    if "keys" in sections:
        rows = _cap(
            "keys",
            await prisma_client.db.litellm_verificationtoken.find_many(
                take=limit, order={"created_at": "asc"}
            ),
        )
        envelope["keys"] = [_strip(r, _STRIP_FIELDS["keys"]) for r in rows]

    if "credentials" in sections:
        rows = _cap(
            "credentials",
            await prisma_client.db.litellm_credentialstable.find_many(
                take=limit, order={"created_at": "asc"}
            ),
        )
        envelope["credentials"] = [
            _redact_credential_values(rec) if redact_secrets else rec
            for rec in [_strip(r, _STRIP_FIELDS["credentials"]) for r in rows]
        ]

    if "models" in sections:
        rows = _cap(
            "models",
            await prisma_client.db.litellm_proxymodeltable.find_many(
                take=limit, order={"created_at": "asc"}
            ),
        )
        envelope["models"] = [
            _redact_litellm_params(rec) if redact_secrets else rec
            for rec in [_strip(r, _STRIP_FIELDS["models"]) for r in rows]
        ]

    if "mcp_servers" in sections:
        rows = _cap(
            "mcp_servers",
            await prisma_client.db.litellm_mcpservertable.find_many(
                take=limit, order={"created_at": "asc"}
            ),
        )
        envelope["mcp_servers"] = [
            (
                _redact_mcp_sensitive_fields(_redact_mcp_credentials(rec))
                if redact_secrets
                else rec
            )
            for rec in [_strip(r, _STRIP_FIELDS["mcp_servers"]) for r in rows]
        ]

    if "agents" in sections:
        rows = _cap(
            "agents",
            await prisma_client.db.litellm_agentstable.find_many(
                take=limit, order={"created_at": "asc"}
            ),
        )
        envelope["agents"] = [
            _redact_litellm_params(rec) if redact_secrets else rec
            for rec in [_strip(r, _STRIP_FIELDS["agents"]) for r in rows]
        ]

    if "guardrails" in sections:
        rows = _cap(
            "guardrails",
            await prisma_client.db.litellm_guardrailstable.find_many(
                take=limit, order={"created_at": "asc"}
            ),
        )
        envelope["guardrails"] = [_strip(r, _STRIP_FIELDS["guardrails"]) for r in rows]

    if "tags" in sections:
        rows = _cap(
            "tags",
            await prisma_client.db.litellm_tagtable.find_many(
                take=limit, order={"created_at": "asc"}
            ),
        )
        envelope["tags"] = [_strip(r, _STRIP_FIELDS["tags"]) for r in rows]

    if "general_settings" in sections:
        gs_rows = await prisma_client.db.litellm_config.find_many(
            where={"param_name": {"in": list(SAFE_GENERAL_SETTINGS_KEYS)}}
        )
        envelope["general_settings"] = {
            r.param_name: r.param_value for r in gs_rows if r.param_value is not None
        }


async def _import_keys_section(
    prisma_client: Any,
    records: List[Dict[str, Any]],
    conflict: str,
    dry_run: bool,
    section_result: "ImportSectionResult",
) -> None:
    """
    Keys are exported for metadata visibility only.

    - Create: impossible — token (@id, primary key) is not exported for
      security reasons.  New keys must be created via POST /key/generate.
    - Skip: existing records are left untouched.
    - Replace / Merge: metadata fields are updated via update_many(), which
      accepts a non-unique WHERE clause.  key_alias has @@index (not @unique
      or @id), so table.update(where={"key_alias": ...}) would raise a Prisma
      validation error; update_many() is the correct API for this schema.
      token, key_alias, and spend are always stripped from the UPDATE payload.
    """
    # Phase 1 — separate keyless records (unconditionally skipped) from
    # importable ones.  total_processed is counted here for all records so
    # phase 3 does not double-count.
    importable: List[Dict[str, Any]] = []
    for rec in records:
        section_result.total_processed += 1
        if not rec.get("key_alias"):
            section_result.skipped += 1
            section_result.warnings.append(
                "Skipped key with no key_alias — cannot re-import without a stable identifier"
            )
        else:
            importable.append(rec)

    if not importable:
        return

    # Phase 2 — batch-fetch existing rows (avoids N+1 queries).
    aliases = [r["key_alias"] for r in importable]
    existing_rows = await prisma_client.db.litellm_verificationtoken.find_many(
        where={"key_alias": {"in": aliases}}
    )
    existing_aliases: Set[str] = {
        row.key_alias for row in existing_rows if row.key_alias
    }
    existing_by_alias: Dict[str, Any] = {
        row.key_alias: row for row in existing_rows if row.key_alias
    }

    # Phase 3 — per-record decisions.
    for rec in importable:
        alias = rec["key_alias"]

        if alias not in existing_aliases:
            # Cannot create: token (primary key) was not exported.
            section_result.skipped += 1
            section_result.warnings.append(
                f"Skipped key '{alias}' — new keys cannot be created during import "
                "(token/primary key is not exported). Create via POST /key/generate."
            )
            continue

        if conflict == "skip":
            section_result.skipped += 1
            continue

        if dry_run:
            section_result.updated += 1
            continue

        # Build the UPDATE payload.
        if conflict == "merge":
            existing_obj = existing_by_alias[alias]
            base = (
                existing_obj.model_dump()
                if hasattr(existing_obj, "model_dump")
                else dict(existing_obj)
            )
            update_data = _deep_merge(base, rec)
        else:
            update_data = dict(rec)

        # Strip fields that must not appear in the UPDATE statement.
        for field in _KEYS_UPDATE_STRIP:
            update_data.pop(field, None)

        try:
            # update_many() accepts non-unique WHERE clauses; update() does not.
            await prisma_client.db.litellm_verificationtoken.update_many(
                where={"key_alias": alias},
                data=update_data,
            )
            section_result.updated += 1
        except Exception as e:
            section_result.errors += 1
            section_result.warnings.append(f"Failed to update key '{alias}': {e}")


async def _import_identity_sections(
    prisma_client: Any,
    data: "LiteLLMExportEnvelope",
    conflict: str,
    dry_run: bool,
    result: "ImportResult",
) -> None:
    """Import budgets, organizations, teams, users (dependency order, first half)."""
    if data.budgets is not None:
        result.sections_attempted.append("budgets")
        em = await _load_existing(
            prisma_client.db.litellm_budgettable,
            "budget_id",
            [r.get("budget_id") for r in data.budgets if r.get("budget_id")],
        )
        await _import_section(
            table=prisma_client.db.litellm_budgettable,
            table_name="litellm_budgettable",
            records=data.budgets,
            id_field="budget_id",
            conflict=conflict,
            dry_run=dry_run,
            section_result=result.budgets,
            existing_map=em,
        )

    if data.organizations is not None:
        result.sections_attempted.append("organizations")
        em = await _load_existing(
            prisma_client.db.litellm_organizationtable,
            "organization_id",
            [
                r.get("organization_id")
                for r in data.organizations
                if r.get("organization_id")
            ],
        )
        await _import_section(
            table=prisma_client.db.litellm_organizationtable,
            table_name="litellm_organizationtable",
            records=data.organizations,
            id_field="organization_id",
            conflict=conflict,
            dry_run=dry_run,
            section_result=result.organizations,
            existing_map=em,
        )

    if data.teams is not None:
        result.sections_attempted.append("teams")
        em = await _load_existing(
            prisma_client.db.litellm_teamtable,
            "team_id",
            [r.get("team_id") for r in data.teams if r.get("team_id")],
        )
        await _import_section(
            table=prisma_client.db.litellm_teamtable,
            table_name="litellm_teamtable",
            records=data.teams,
            id_field="team_id",
            conflict=conflict,
            dry_run=dry_run,
            section_result=result.teams,
            existing_map=em,
        )

    if data.users is not None:
        result.sections_attempted.append("users")
        em = await _load_existing(
            prisma_client.db.litellm_usertable,
            "user_id",
            [r.get("user_id") for r in data.users if r.get("user_id")],
        )
        await _import_section(
            table=prisma_client.db.litellm_usertable,
            table_name="litellm_usertable",
            records=data.users,
            id_field="user_id",
            conflict=conflict,
            dry_run=dry_run,
            section_result=result.users,
            existing_map=em,
        )


async def _import_resource_sections(
    prisma_client: Any,
    data: "LiteLLMExportEnvelope",
    conflict: str,
    dry_run: bool,
    result: "ImportResult",
) -> None:
    """Import keys, credentials, models, mcp_servers, agents, guardrails, tags, general_settings."""
    if data.keys is not None:
        result.sections_attempted.append("keys")
        await _import_keys_section(
            prisma_client=prisma_client,
            records=data.keys,
            conflict=conflict,
            dry_run=dry_run,
            section_result=result.keys,
        )

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
        em = await _load_existing(
            prisma_client.db.litellm_credentialstable,
            "credential_name",
            [
                r["credential_name"]
                for r in importable_creds
                if r.get("credential_name")
            ],
        )
        await _import_section(
            table=prisma_client.db.litellm_credentialstable,
            table_name="litellm_credentialstable",
            records=importable_creds,
            id_field="credential_name",
            conflict=conflict,
            dry_run=dry_run,
            section_result=result.credentials,
            existing_map=em,
        )

    if data.models is not None:
        result.sections_attempted.append("models")
        em = await _load_existing(
            prisma_client.db.litellm_proxymodeltable,
            "model_id",
            [r.get("model_id") for r in data.models if r.get("model_id")],
        )
        await _import_section(
            table=prisma_client.db.litellm_proxymodeltable,
            table_name="litellm_proxymodeltable",
            records=data.models,
            id_field="model_id",
            conflict=conflict,
            dry_run=dry_run,
            section_result=result.models,
            existing_map=em,
        )

    await _import_mcp_agents_tail(prisma_client, data, conflict, dry_run, result)


async def _import_mcp_agents_tail(
    prisma_client: Any,
    data: "LiteLLMExportEnvelope",
    conflict: str,
    dry_run: bool,
    result: "ImportResult",
) -> None:
    """Import mcp_servers, agents, guardrails, tags, general_settings."""
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
        em = await _load_existing(
            prisma_client.db.litellm_mcpservertable,
            "server_id",
            [r.get("server_id") for r in cleaned_servers if r.get("server_id")],
        )
        await _import_section(
            table=prisma_client.db.litellm_mcpservertable,
            table_name="litellm_mcpservertable",
            records=cleaned_servers,
            id_field="server_id",
            conflict=conflict,
            dry_run=dry_run,
            section_result=result.mcp_servers,
            existing_map=em,
        )

    if data.agents is not None:
        result.sections_attempted.append("agents")
        em = await _load_existing(
            prisma_client.db.litellm_agentstable,
            "agent_name",
            [r.get("agent_name") for r in data.agents if r.get("agent_name")],
        )
        await _import_section(
            table=prisma_client.db.litellm_agentstable,
            table_name="litellm_agentstable",
            records=data.agents,
            id_field="agent_name",
            conflict=conflict,
            dry_run=dry_run,
            section_result=result.agents,
            existing_map=em,
        )

    if data.guardrails is not None:
        result.sections_attempted.append("guardrails")
        em = await _load_existing(
            prisma_client.db.litellm_guardrailstable,
            "guardrail_name",
            [
                r.get("guardrail_name")
                for r in data.guardrails
                if r.get("guardrail_name")
            ],
        )
        await _import_section(
            table=prisma_client.db.litellm_guardrailstable,
            table_name="litellm_guardrailstable",
            records=data.guardrails,
            id_field="guardrail_name",
            conflict=conflict,
            dry_run=dry_run,
            section_result=result.guardrails,
            existing_map=em,
        )

    if data.tags is not None:
        result.sections_attempted.append("tags")
        em = await _load_existing(
            prisma_client.db.litellm_tagtable,
            "tag_name",
            [r.get("tag_name") for r in data.tags if r.get("tag_name")],
        )
        await _import_section(
            table=prisma_client.db.litellm_tagtable,
            table_name="litellm_tagtable",
            records=data.tags,
            id_field="tag_name",
            conflict=conflict,
            dry_run=dry_run,
            section_result=result.tags,
            existing_map=em,
        )

    if data.general_settings is not None:
        result.sections_attempted.append("general_settings")
        await _import_general_settings(
            prisma_client=prisma_client,
            settings=data.general_settings,
            conflict=conflict,
            dry_run=dry_run,
            section_result=result.general_settings,
        )


async def _import_all_sections(
    prisma_client: Any,
    data: "LiteLLMExportEnvelope",
    conflict: str,
    dry_run: bool,
    result: "ImportResult",
) -> None:
    """
    Process all sections in dependency order.  Each section runs inside its
    own DB transaction so a failure in one section does not roll back
    already-committed sections.
    """
    await _import_identity_sections(prisma_client, data, conflict, dry_run, result)
    await _import_resource_sections(prisma_client, data, conflict, dry_run, result)


# ---------------------------------------------------------------------------
# GET /config/export
# ---------------------------------------------------------------------------


@router.get(
    "/config/export",
    tags=["config.yaml"],
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
    limit: int = Query(
        default=1000,
        ge=1,
        le=5000,
        description="Maximum number of rows to fetch per entity section. "
        "Defaults to 1000; maximum 5000. Use the include parameter to export "
        "individual sections on deployments with very large tables.",
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
    prisma_client = _get_prisma_with_auth(user_api_key_dict, action="export")
    sections = _resolve_sections(include)
    envelope: Dict[str, Any] = {
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_instance": str(request.base_url).rstrip("/"),
        "include_filters": sorted(sections),
        "row_limit": limit,
        "truncated_sections": [],
    }
    try:
        await _fetch_export_sections(
            prisma_client, sections, limit, redact_secrets, envelope
        )
    except Exception as e:
        verbose_proxy_logger.error(f"config/export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

    if format == "yaml":
        content = yaml.dump(envelope, allow_unicode=True, sort_keys=False)
        return Response(content=content, media_type="application/yaml")
    content = json.dumps(envelope, indent=2, default=str)
    return Response(content=content, media_type="application/json")


# ---------------------------------------------------------------------------
# Validates the user as PROXY_ADMIN and returns a Prisma client instance.
# ---------------------------------------------------------------------------


def _get_prisma_with_auth(user_api_key_dict: UserAPIKeyAuth, action: str = "access"):
    from litellm.proxy.proxy_server import prisma_client  # avoid circular import

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail=f"Only proxy admins can {action} config",
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=f"Database not connected. Cannot {action} config.",
        )

    return prisma_client


# ---------------------------------------------------------------------------
# POST /config/import
# ---------------------------------------------------------------------------


@router.post(
    "/config/import",
    tags=["config.yaml"],
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
    prisma_client = _get_prisma_with_auth(user_api_key_dict, action="import")
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
        await _import_all_sections(prisma_client, data, conflict, dry_run, result)
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error("config/import failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

    non_gs = [s for s in result.sections_attempted if s != "general_settings"]
    verbose_proxy_logger.info(
        "config/import complete: triggered_by=%s dry_run=%s sections=%s "
        "totals=created:%d updated:%d skipped:%d errors:%d",
        triggered_by,
        dry_run,
        result.sections_attempted,
        sum(getattr(result, s).created for s in non_gs),
        sum(getattr(result, s).updated for s in non_gs),
        sum(getattr(result, s).skipped for s in non_gs),
        sum(getattr(result, s).errors for s in non_gs),
    )
    return result


# ---------------------------------------------------------------------------
# #1 — Per-section transaction wrapper
# ---------------------------------------------------------------------------


async def _import_section(
    table: Any,
    table_name: str,
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

    table_name must be the exact attribute name used to access this table on
    the Prisma client (e.g. "litellm_teamtable").  It is used to look up the
    corresponding accessor on the transaction object — prisma-client-py exposes
    the same snake_case names on both the client and the transaction.

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

    # Check for transaction support BEFORE entering async with.
    # Catching AttributeError *inside* the block would also swallow
    # AttributeErrors raised by _upsert (e.g. a missing field on the data
    # dict), misclassifying real bugs as "tx not available".
    has_tx = hasattr(table, "_client") and hasattr(table._client, "tx")

    if not has_tx:
        verbose_proxy_logger.warning(
            "Prisma transaction not available for %s; "
            "falling back to non-transactional writes.",
            table_name,
        )
        try:
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
            section_result.warnings.append(f"Section failed: {e}")
            verbose_proxy_logger.error(
                "Section %s failed: %s", table_name, e, exc_info=True
            )
        return

    # Snapshot counts before entering the transaction so that a rollback can
    # restore them to their pre-section state before recording the failure.
    # Without this, _upsert increments created/updated/skipped inside the block
    # and the except branch then adds errors on top, making
    # created + updated + skipped + errors > total_processed.
    _snap = (
        section_result.created,
        section_result.updated,
        section_result.skipped,
        section_result.errors,
        section_result.total_processed,
        list(section_result.warnings),
    )

    try:
        async with table._client.tx() as tx:
            # Use the explicit snake_case table_name to look up the accessor on
            # the transaction object.  table.__class__.__name__.lower() would
            # produce a name without underscores (e.g. "litellmteamtableactions")
            # that never matches, causing getattr to silently return the original
            # non-transactional table handle.
            tx_table = getattr(tx, table_name)
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
    except Exception as e:
        # Restore pre-transaction counts, then record every record as an error.
        (
            section_result.created,
            section_result.updated,
            section_result.skipped,
            section_result.errors,
            section_result.total_processed,
            section_result.warnings,
        ) = _snap
        section_result.errors += len(records)
        section_result.total_processed += len(records)
        section_result.warnings.append(f"Section transaction rolled back: {e}")
        verbose_proxy_logger.error(
            "Section %s transaction failed: %s", table_name, e, exc_info=True
        )


async def _import_general_settings(
    prisma_client: Any,
    settings: Dict[str, Any],
    conflict: str,
    dry_run: bool,
    section_result: ImportSectionResult,
) -> None:
    """
    Import general_settings key/value rows.

    Each key is written individually so that a failure on one setting does not
    prevent the remaining keys from being persisted.  Errors are counted and
    surfaced in section_result.warnings without aborting the section.
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
