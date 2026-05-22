"""Import helper functions for POST /config/import."""

from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_proxy_logger
from litellm.proxy.management_endpoints.config_export_types import (
    SAFE_GENERAL_SETTINGS_KEYS,
    _KEYS_UPDATE_STRIP,
    ImportResult,
    ImportSectionResult,
    _deep_merge,
)

_ROUTING_SECTIONS = frozenset(
    {"models", "agents", "guardrails", "mcp_servers", "general_settings"}
)
_AUTH_SECTIONS = frozenset({"keys", "users", "teams"})


async def _post_import_cache_refresh(result: ImportResult, dry_run: bool) -> None:
    """Invalidate stale in-memory state after a successful import.

    - Auth sections (keys/users/teams): flush user_api_key_cache so the next
      request re-reads updated access/budget fields from the DB.
    - Routing sections (models/agents/guardrails/mcp_servers/general_settings):
      reload the llm_router deployment list from the DB via clear_cache().
    """
    if dry_run:
        return
    attempted = set(result.sections_attempted)
    if attempted & _AUTH_SECTIONS:
        try:
            from litellm.proxy.proxy_server import user_api_key_cache  # avoid circular

            user_api_key_cache.flush_cache()
        except Exception as e:
            verbose_proxy_logger.warning("post-import auth cache flush failed: %s", e)
    if attempted & _ROUTING_SECTIONS:
        try:
            from litellm.proxy.management_endpoints.model_management_endpoints import (
                clear_cache,
            )

            await clear_cache()
        except Exception as e:
            verbose_proxy_logger.warning(
                "post-import router cache refresh failed: %s", e
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


# ---------------------------------------------------------------------------
# Per-section transaction wrapper
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
        _snap = (
            section_result.created,
            section_result.updated,
            section_result.skipped,
            section_result.errors,
            section_result.total_processed,
            list(section_result.warnings),
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


# ---------------------------------------------------------------------------
# General settings import
# ---------------------------------------------------------------------------


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
# Keys section — special handling (cannot create; update_many for non-unique WHERE)
# ---------------------------------------------------------------------------


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
    existing_aliases: set = {row.key_alias for row in existing_rows if row.key_alias}
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
