"""GET /config/export and POST /config/import endpoints."""

import datetime
import json
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.config_export_types import (
    ALL_SECTIONS,
    SAFE_GENERAL_SETTINGS_KEYS,
    ImportRequest,
    ImportResult,
    LiteLLMExportEnvelope,
    _STRIP_FIELDS,
    _clean_litellm_params_record,
    _is_redacted,
    _load_existing,
    _redact_credential_values,
    _redact_litellm_params,
    _redact_mcp_credentials,
    _redact_sensitive_header_fields,
    _resolve_sections,
    _strip,
    _validate_dependencies,
    _validate_envelope,
)
from litellm.proxy.management_endpoints.config_import_helpers import (
    _import_general_settings,
    _import_keys_section,
    _import_section,
    _post_import_cache_refresh,
)

router = APIRouter()

_IDENTITY_SECTION_TABLES: List[Tuple[str, str, str]] = [
    ("budgets", "litellm_budgettable", "budget_id"),
    ("organizations", "litellm_organizationtable", "organization_id"),
    ("teams", "litellm_teamtable", "team_id"),
    ("users", "litellm_usertable", "user_id"),
]

_RESOURCE_SECTION_TABLES: List[Tuple[str, str, str]] = [
    ("models", "litellm_proxymodeltable", "model_id"),
    ("agents", "litellm_agentstable", "agent_name"),
    ("guardrails", "litellm_guardrailstable", "guardrail_name"),
    ("tags", "litellm_tagtable", "tag_name"),
]


async def _import_standard_section(
    prisma_client: Any,
    data: LiteLLMExportEnvelope,
    section_name: str,
    table_name: str,
    id_field: str,
    conflict: str,
    dry_run: bool,
    result: ImportResult,
) -> None:
    """Import a single standard section using the generic upsert flow."""
    records = getattr(data, section_name, None)
    if records is None:
        return
    records = [_clean_litellm_params_record(r) for r in records]
    result.sections_attempted.append(section_name)
    table = getattr(prisma_client.db, table_name)
    em = await _load_existing(
        table, id_field, [r.get(id_field) for r in records if r.get(id_field)]
    )
    await _import_section(
        table=table,
        table_name=table_name,
        records=records,
        id_field=id_field,
        conflict=conflict,
        dry_run=dry_run,
        section_result=getattr(result, section_name),
        existing_map=em,
    )


async def _import_all_sections(
    prisma_client: Any,
    data: LiteLLMExportEnvelope,
    conflict: str,
    dry_run: bool,
    result: ImportResult,
) -> None:
    """Process all sections in dependency order."""
    for section_name, table_name, id_field in _IDENTITY_SECTION_TABLES:
        await _import_standard_section(
            prisma_client,
            data,
            section_name,
            table_name,
            id_field,
            conflict,
            dry_run,
            result,
        )
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

    for section_name, table_name, id_field in _RESOURCE_SECTION_TABLES:
        await _import_standard_section(
            prisma_client,
            data,
            section_name,
            table_name,
            id_field,
            conflict,
            dry_run,
            result,
        )

    if data.mcp_servers is not None:
        result.sections_attempted.append("mcp_servers")
        _MCP_REDACTED_FIELDS = ("credentials", "static_headers", "env")
        cleaned_servers: List[Dict[str, Any]] = []
        for rec in data.mcp_servers:
            to_strip = [f for f in _MCP_REDACTED_FIELDS if _is_redacted(rec.get(f))]
            for f in to_strip:
                result.mcp_servers.warnings.append(
                    f"MCP server '{rec.get('server_name')}' — '{f}' is redacted; bind manually."
                )
            cleaned_servers.append(
                {k: v for k, v in rec.items() if k not in to_strip} if to_strip else rec
            )
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

    if data.general_settings is not None:
        result.sections_attempted.append("general_settings")
        await _import_general_settings(
            prisma_client=prisma_client,
            settings=data.general_settings,
            conflict=conflict,
            dry_run=dry_run,
            section_result=result.general_settings,
        )


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
                _redact_sensitive_header_fields(_redact_mcp_credentials(rec))
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
            (
                _redact_sensitive_header_fields(_redact_litellm_params(rec))
                if redact_secrets
                else rec
            )
            for rec in [_strip(r, _STRIP_FIELDS["agents"]) for r in rows]
        ]

    if "guardrails" in sections:
        rows = _cap(
            "guardrails",
            await prisma_client.db.litellm_guardrailstable.find_many(
                take=limit, order={"created_at": "asc"}
            ),
        )
        envelope["guardrails"] = [
            (
                _redact_sensitive_header_fields(_redact_litellm_params(rec))
                if redact_secrets
                else rec
            )
            for rec in [_strip(r, _STRIP_FIELDS["guardrails"]) for r in rows]
        ]

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
    Returns a snapshot of all UI/API-managed proxy state.

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
    Re-apply a config snapshot from GET /config/export.

    - Conflict modes: `skip` (default), `replace`, `merge`
    - `dry_run=true`: simulate import without DB writes
    - Redacted secrets (`{__redacted__: true}`) are skipped
    - Each section runs in its own transaction; failures roll back per section
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

    await _post_import_cache_refresh(result, dry_run)

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
