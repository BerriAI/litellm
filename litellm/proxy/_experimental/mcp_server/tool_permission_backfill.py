"""
Convert pre-overrides object_permission rows (``mcp_permission_version`` falsy)
to the mcp_tool_overrides storage so the convention evaluator can take over.

``convert_row`` is pure: given the row and the discovered tool inventory per
granted server it produces the fields to persist, or reports which servers
could not supply an inventory. ``run_mcp_tool_permission_backfill`` pages the
table at proxy boot, converts each residual row, and writes with an
update_many compare-and-set so a row edited mid-backfill is skipped rather
than clobbered.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from prisma.fields import Json
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_logger
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.proxy._experimental.mcp_server.tool_classification import classify_tool_op
from litellm.proxy._types import SpecialMCPServerName
from litellm.repositories.object_permission_repository import ObjectPermissionRepository
from litellm.types.mcp import MCPToolOverrideEntry

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
    from litellm.proxy.utils import PrismaClient

    RawRow: TypeAlias = LiteLLM_ObjectPermissionTable | prisma_models.LiteLLM_ObjectPermissionTable

ToolInventory: TypeAlias = Mapping[str, str | None]
Inventories: TypeAlias = Mapping[str, ToolInventory | None]

_ROW_DUMP_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True)
class ConvertedRow:
    mcp_tool_overrides: Mapping[str, MCPToolOverrideEntry]
    mcp_tool_permissions: Mapping[str, Sequence[str]]
    mcp_tool_permissions_archive: Mapping[str, Sequence[str]]
    mcp_permission_version: int = 1


@dataclass(frozen=True)
class Unavailable:
    server_ids: frozenset[str]


class _ConvertedRowRecord(TypedDict, total=False):
    mcp_tool_overrides: ReadOnly[Mapping[str, MCPToolOverrideEntry]]
    mcp_tool_permissions: ReadOnly[Mapping[str, Sequence[str]]]
    mcp_tool_permissions_archive: ReadOnly[Mapping[str, Sequence[str]]]
    mcp_permission_version: ReadOnly[int]


ConversionResult: TypeAlias = ConvertedRow | Unavailable


def _server_override_entry(
    stored: Sequence[str] | None,
    discovered: ToolInventory,
) -> MCPToolOverrideEntry | None:
    allow: Final[frozenset[str]] = (
        frozenset(
            name for name in stored if name not in discovered or classify_tool_op(name, discovered[name]) == "delete"
        )
        if stored
        else frozenset(
            name for name, description in discovered.items() if classify_tool_op(name, description) == "delete"
        )
    )
    deny: Final[frozenset[str]] = (
        frozenset(
            name
            for name, description in discovered.items()
            if name not in stored and classify_tool_op(name, description) != "delete"
        )
        if stored
        else frozenset()
    )
    if not allow and not deny:
        return None
    entry: Final[MCPToolOverrideEntry] = {"allow": sorted(allow), "deny": sorted(deny)}
    return entry


@dataclass(frozen=True)
class BackfillReport:
    converted: frozenset[str]
    cas_missed: frozenset[str]
    unavailable: Mapping[str, frozenset[str]]
    skipped_no_grants: frozenset[str]


def convert_row(
    row: LiteLLM_ObjectPermissionTable,
    inventories: Inventories,
) -> ConversionResult:
    """Compute the converted fields for one row.

    ``inventories`` must carry an entry for every server the row grants
    (expanded mcp_servers, access-group servers, tool-permission keys, the
    all-proxy sentinel already expanded); ``None`` marks a server whose
    catalog could not be discovered and makes the row Unavailable. Servers
    granted only through toolsets are excluded by the caller and stay closed.
    """
    missing: Final[frozenset[str]] = frozenset(
        server_id for server_id, inventory in inventories.items() if inventory is None
    )
    if missing:
        return Unavailable(server_ids=missing)

    legacy: Final = row.mcp_tool_permissions or MappingProxyType({})
    remaining_permissions: Final[Mapping[str, Sequence[str]]] = MappingProxyType(
        {server_id: stored for server_id, stored in legacy.items() if server_id not in inventories or not stored}
    )
    override_entries: Final = MappingProxyType(
        {
            server_id: _server_override_entry(legacy.get(server_id), inventory or MappingProxyType({}))
            for server_id, inventory in inventories.items()
            if server_id not in legacy or legacy[server_id]
        }
    )
    overrides: Final[Mapping[str, MCPToolOverrideEntry]] = MappingProxyType(
        {server_id: entry for server_id, entry in override_entries.items() if entry is not None}
    )
    return ConvertedRow(
        mcp_tool_overrides=overrides,
        mcp_tool_permissions=remaining_permissions,
        mcp_tool_permissions_archive=MappingProxyType(dict(legacy)),
    )


async def resolve_granted_server_ids(
    row: LiteLLM_ObjectPermissionTable,
    manager: "MCPServerManager",
) -> frozenset[str]:
    """Every concrete server_id the row grants, toolset-only servers excluded."""
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    direct: Final = frozenset(
        chain.from_iterable(
            (
                manager.get_registry().keys()
                if identifier == SpecialMCPServerName.all_proxy_servers.value
                else (identifier,)
            )
            for identifier in manager.expand_permission_list(sorted(row.mcp_servers or ()))
        )
    )
    access_group_servers: Final = await MCPRequestHandler._get_mcp_servers_from_access_groups(  # pyright: ignore[reportPrivateUsage]  # shared access-group resolution owned by MCPRequestHandler
        sorted(row.mcp_access_groups or ())
    )
    return frozenset(
        direct
        | frozenset(access_group_servers)
        | frozenset(manager.expand_tool_permissions(row.mcp_tool_permissions).keys())
    )


async def gather_inventories(
    server_ids: frozenset[str],
    manager: "MCPServerManager",
    cache: dict[str, ToolInventory | None] | None = None,  # mutable-ok: caller-owned fetch cache updated in place
) -> Mapping[str, ToolInventory | None]:
    if cache is None:
        return MappingProxyType(
            {server_id: await manager.fetch_unfiltered_inventory(server_id) for server_id in server_ids}
        )
    return MappingProxyType(
        {
            server_id: (
                cache[server_id]
                if server_id in cache
                else cache.setdefault(server_id, await manager.fetch_unfiltered_inventory(server_id))
            )
            for server_id in server_ids
        }
    )


def converted_row_record(conversion: ConvertedRow) -> Mapping[str, object]:
    record: Final[_ConvertedRowRecord] = {
        "mcp_tool_overrides": {**conversion.mcp_tool_overrides},
        "mcp_tool_permissions": {**conversion.mcp_tool_permissions},
        "mcp_tool_permissions_archive": {**conversion.mcp_tool_permissions_archive},
        "mcp_permission_version": 1,
    }
    return record


def _to_model(row: "RawRow") -> LiteLLM_ObjectPermissionTable:
    data: Final[Mapping[str, object]] = _ROW_DUMP_ADAPTER.validate_python(row.model_dump())
    json_fields: Final = (
        "mcp_tool_permissions",
        "mcp_tool_overrides",
        "mcp_tool_permissions_archive",
    )
    normalized: Final = MappingProxyType(
        {
            key: (json.loads(value) if key in json_fields and isinstance(value, str) else value)
            for key, value in data.items()
        }
    )
    return LiteLLM_ObjectPermissionTable.model_validate(normalized)


async def _converted_page(
    rows: Sequence["RawRow"],
    prisma_client: "PrismaClient",
    manager: "MCPServerManager",
    inventory_cache: dict[str, ToolInventory | None],  # mutable-ok: caller-owned fetch cache updated in place
) -> BackfillReport:
    models: Final[Sequence[LiteLLM_ObjectPermissionTable]] = tuple(_to_model(row) for row in rows)
    outcomes: Final = tuple(
        [
            await _convert_one_row(raw, model, prisma_client, manager, inventory_cache)
            for raw, model in zip(rows, models, strict=True)
        ]
    )
    return BackfillReport(
        converted=frozenset(
            model.object_permission_id
            for model, outcome in zip(models, outcomes, strict=True)
            if outcome == "converted"
        ),
        cas_missed=frozenset(
            model.object_permission_id
            for model, outcome in zip(models, outcomes, strict=True)
            if outcome == "cas_missed"
        ),
        unavailable=MappingProxyType(
            {
                model.object_permission_id: outcome
                for model, outcome in zip(models, outcomes, strict=True)
                if isinstance(outcome, frozenset)
            }
        ),
        skipped_no_grants=frozenset(
            model.object_permission_id for model, outcome in zip(models, outcomes, strict=True) if outcome == "skipped"
        ),
    )


async def _convert_one_row(
    raw_row: "RawRow",
    row: LiteLLM_ObjectPermissionTable,
    prisma_client: "PrismaClient",
    manager: "MCPServerManager",
    inventory_cache: dict[str, ToolInventory | None],  # mutable-ok: caller-owned fetch cache updated in place
) -> str | frozenset[str]:
    """Convert one row and CAS-write it. Returns the outcome tag, or the
    unavailable server ids as a frozenset when the row cannot be converted."""
    granted: Final = await resolve_granted_server_ids(row, manager)
    if not granted:
        return "skipped"
    conversion: Final = convert_row(row, await gather_inventories(granted, manager, inventory_cache))
    if isinstance(conversion, Unavailable):
        return conversion.server_ids
    stored_fields: Final = (
        ("mcp_servers", raw_row.mcp_servers, False),
        ("mcp_access_groups", raw_row.mcp_access_groups, False),
        ("mcp_toolsets", raw_row.mcp_toolsets, False),
        ("mcp_tool_permissions", raw_row.mcp_tool_permissions, True),
    )
    equals_filters: Final = {  # mutable-ok: prisma where kwarg requires a JSON-serializable dict
        field: {"equals": Json(value) if is_json else value}  # mutable-ok: prisma where filter shape
        for field, value, is_json in stored_fields
        if value is not None
    }
    where: Final[dict[str, object]] = {  # mutable-ok: prisma where kwarg requires a JSON-serializable dict
        "object_permission_id": row.object_permission_id,
        "mcp_permission_version": 0,
        **equals_filters,
    }
    record: Final = converted_row_record(conversion)
    data: Final[dict[str, object]] = {  # mutable-ok: prisma data kwarg requires a JSON-serializable dict
        "mcp_tool_overrides": json.dumps(record["mcp_tool_overrides"]),
        "mcp_tool_permissions": json.dumps(record["mcp_tool_permissions"]),
        "mcp_tool_permissions_archive": json.dumps(record["mcp_tool_permissions_archive"]),
        "mcp_permission_version": 1,
    }
    updated: Final = await ObjectPermissionRepository(prisma_client).table.update_many(where=where, data=data)
    return "cas_missed" if updated == 0 else "converted"


def _merge_reports(reports: Sequence[BackfillReport]) -> BackfillReport:
    return BackfillReport(
        converted=frozenset(chain.from_iterable(report.converted for report in reports)),
        cas_missed=frozenset(chain.from_iterable(report.cas_missed for report in reports)),
        unavailable=MappingProxyType(dict(chain.from_iterable(report.unavailable.items() for report in reports))),
        skipped_no_grants=frozenset(chain.from_iterable(report.skipped_no_grants for report in reports)),
    )


async def run_mcp_tool_permission_backfill(
    prisma_client: "PrismaClient",
    manager: "MCPServerManager",
    batch_size: int = 200,
) -> BackfillReport:
    """Convert every residual v0 permission row. Idempotent: converted rows
    stop matching the version filter, and CAS-missed or Unavailable rows are
    left for the next boot."""
    table: Final = ObjectPermissionRepository(prisma_client).table
    reports: Final[list[BackfillReport]] = []  # mutable-ok: accumulated per page
    inventory_cache: Final[dict[str, ToolInventory | None]] = {}  # mutable-ok: one fetch per server per run
    cursor: str | None = None  # rebind-ok: cursor pagination
    while True:
        rows = await table.find_many(
            where={"mcp_permission_version": 0},  # mutable-ok: prisma where kwarg
            order={"object_permission_id": "asc"},  # mutable-ok: prisma order kwarg
            take=batch_size,
            cursor={"object_permission_id": cursor} if cursor is not None else None,  # mutable-ok: prisma cursor kwarg
            skip=1 if cursor is not None else None,
        )
        if not rows:
            break
        reports.append(await _converted_page(rows, prisma_client, manager, inventory_cache))
        if len(rows) < batch_size:
            break
        cursor = rows[-1].object_permission_id

    report: Final = _merge_reports(reports)
    verbose_logger.info(
        "MCP tool permission backfill: converted=%s cas_missed=%s skipped_no_grants=%s unavailable=%s",
        sorted(report.converted),
        sorted(report.cas_missed),
        sorted(report.skipped_no_grants),
        MappingProxyType({row_id: tuple(sorted(server_ids)) for row_id, server_ids in report.unavailable.items()}),
    )
    return report
