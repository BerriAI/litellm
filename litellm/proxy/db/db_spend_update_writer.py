"""Capture request accounting and apply stored SpendLogs contributions."""

import asyncio
import dataclasses
import json
import os
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeAlias, cast
from urllib.parse import unquote

from pydantic import ConfigDict, TypeAdapter
from typing_extensions import LiteralString, ReadOnly, TypedDict

import litellm
from litellm._internal_context import service_target, with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching import RedisCache
from litellm.constants import (
    BILLED_RETRY_ATTEMPT_METADATA_KEY,
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
)
from litellm.litellm_core_utils.core_helpers import finite_cost, single_log_line
from litellm.litellm_core_utils.litellm_logging import coerce_model_access_groups
from litellm.proxy._types import (
    DB_RETRY_SAFE_ERROR_TYPES,
    BaseDailySpendTransaction,
    DBSpendUpdateTransactions,
    SpendLogsMetadata,
    SpendLogsPayload,
    ToolDiscoveryQueueItem,
)
from litellm.proxy.common_utils.user_api_key_cache import AUTH_OBJECTS_TARGET, project_cache_key
from litellm.proxy.db.daily_spend_bulk_upsert import (
    DAILY_SPEND_TABLES,
    daily_spend_entity_ids,
)
from litellm.proxy.db.db_span import db_span
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.proxy.db.db_transaction_queue.redis_update_buffer import RedisUpdateBuffer
from litellm.proxy.db.db_transaction_queue.tool_discovery_queue import (
    ToolDiscoveryQueue,
)
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.db.request_error_tracking import request_error_accumulator
from litellm.proxy.db.rollup_lock_timeout import apply_rollup_lock_timeout
from litellm.proxy.route_llm_request import ROUTE_ENDPOINT_MAPPING
from litellm.proxy.spend_tracking.compression_savings import (
    extract_compression_saved_tokens,
)
from litellm.proxy.spend_tracking.savings import (
    compute_savings_spend,
    extract_cache_creation_tokens,
    extract_cache_read_tokens,
    marks_gateway_injection,
)
from litellm.proxy.spend_tracking.spend_log_error_logger import spend_log_error
from litellm.repositories.prisma_protocols import BatchTable
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.accounting.projection import Contribution
    from litellm.proxy.db.autorouter_session_rollup import AutoRouterTurnTransaction
    from litellm.proxy.db.baseline_accounting import DailyBaselineAttribution
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.router import Router
else:
    PrismaClient = Any
    ProxyLogging = Any


RESPONSES_SESSION_CALL_TYPES: Final = frozenset({CallTypes.responses.value, CallTypes.aresponses.value})
_SPEND_METADATA_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_CALL_KWARG_VALUE: Final = TypeAdapter(object)
_JsonPayload: TypeAlias = Mapping[str, object] | Sequence[object] | str | int | float | bool | None
_JSON_VALUE_ADAPTER: Final[TypeAdapter[_JsonPayload]] = TypeAdapter(
    _JsonPayload, config=ConfigDict(strict=True, hide_input_in_errors=True)
)


def _turn_chains_through_the_db(payload: Mapping[str, object], llm_router: "Router | None") -> bool:
    """A Responses turn the chat-completions bridge replays from its spend log on the next `previous_response_id`.

    Native Responses providers hold the conversation themselves, so their rows can wait for the batch write.
    A row with no router deployment behind it is treated as bridged, which keeps today's immediate flush.
    """
    if payload.get("call_type") not in RESPONSES_SESSION_CALL_TYPES:
        return False
    model_id: Final = payload.get("model_id")
    deployment: Final = (
        llm_router.registry().deployment(model_id) if llm_router is not None and isinstance(model_id, str) else None
    )
    if deployment is None:
        return True
    from litellm.responses.main import provider_stores_the_conversation

    litellm_params: Final = deployment.litellm_params
    return not provider_stores_the_conversation(
        litellm_params.model,
        litellm_params.custom_llm_provider,
        {
            "use_chat_completions_api": litellm_params.use_chat_completions_api,
            "model_info": deployment.model_info.model_dump(),
            "api_base": litellm_params.api_base,
        },
    )


def _is_batch_cost_row(payload: SpendLogsPayload) -> bool:
    return payload.get("call_type") == CallTypes.aretrieve_batch.value and payload.get("status") == "success"


def _batch_cost_row_to_write(payload: SpendLogsPayload, disable_spend_logs: bool) -> Mapping[str, object]:
    from litellm.accounting.snapshot import lean_accounting_payload

    return lean_accounting_payload(payload) if disable_spend_logs else payload


class _SpendIncrement(TypedDict):
    increment: ReadOnly[float]


class _MemberSpendRow(TypedDict):
    user_id: ReadOnly[str]
    team_id: ReadOnly[str]
    cost: ReadOnly[float]


class SpendUpsertTable(Protocol):
    def upsert(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> None: ...


class _SpendBatch(Protocol):
    litellm_usertable: BatchTable
    litellm_endusertable: SpendUpsertTable
    litellm_dailymodelusage: SpendUpsertTable

    def execute_raw(self, query: LiteralString, *args: object) -> None: ...

    litellm_verificationtoken: BatchTable
    litellm_teamtable: BatchTable
    litellm_teammembership: BatchTable
    litellm_organizationtable: BatchTable
    litellm_organizationmembership: BatchTable
    litellm_projecttable: BatchTable
    litellm_tagtable: BatchTable
    litellm_agentstable: BatchTable
    litellm_modelaccessgroupbudgettable: BatchTable


_EntitySpendTable: TypeAlias = Literal[
    "litellm_tagtable", "litellm_agentstable", "litellm_modelaccessgroupbudgettable", "litellm_projecttable"
]


_ENTITY_SPEND_TABLES: Final[Mapping[_EntitySpendTable, Callable[[_SpendBatch], BatchTable]]] = MappingProxyType(
    {
        "litellm_tagtable": lambda batcher: batcher.litellm_tagtable,
        "litellm_agentstable": lambda batcher: batcher.litellm_agentstable,
        "litellm_modelaccessgroupbudgettable": lambda batcher: batcher.litellm_modelaccessgroupbudgettable,
        "litellm_projecttable": lambda batcher: batcher.litellm_projecttable,
    }
)


def _entity_spend_table(batcher: _SpendBatch, table_accessor: _EntitySpendTable) -> BatchTable:
    return _ENTITY_SPEND_TABLES[table_accessor](batcher)


class SpendBatchManager(Protocol):
    async def __aenter__(self) -> _SpendBatch: ...

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None: ...


class _SpendTransaction(Protocol):
    def batch_(self) -> SpendBatchManager: ...

    async def execute_raw(self, query: LiteralString, *args: object) -> int: ...


class _SpendTransactionManager(Protocol):
    async def __aenter__(self) -> _SpendTransaction: ...

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None: ...


_SpendTableName = Literal[
    "user_list_transactions",
    "end_user_list_transactions",
    "key_list_transactions",
    "team_list_transactions",
    "team_member_list_transactions",
    "org_list_transactions",
    "org_member_list_transactions",
    "project_list_transactions",
    "tag_list_transactions",
    "model_access_group_list_transactions",
    "agent_list_transactions",
]


def _report_spend_update_step_failure(step_name: str, *, payload: Mapping[str, object], exc: Exception) -> None:
    spend_log_error(
        "Spend tracking - %s failed, this request is missing from that spend rollup. request_id=%s api_key=%s",
        step_name,
        payload.get("request_id"),
        payload.get("api_key"),
        exc=exc,
    )


_EMPTY_SPEND_LOG_METADATA: Final = cast(SpendLogsMetadata, MappingProxyType({}))  # cast-ok: empty read-only metadata


def _parse_spend_log_metadata(payload: Mapping[str, object]) -> SpendLogsMetadata:
    """Malformed metadata costs the request its metadata-derived fields, never its daily rollup row."""
    raw: Final = payload.get("metadata")
    if not isinstance(raw, str):
        return _EMPTY_SPEND_LOG_METADATA
    try:
        parsed: Final[SpendLogsMetadata] = json.loads(raw)
    except json.JSONDecodeError as e:
        spend_log_error(
            "Spend tracking - unreadable spend log metadata, rollups for this request are written without "
            "metadata-derived fields. request_id=%s api_key=%s",
            payload.get("request_id"),
            payload.get("api_key"),
            exc=e,
        )
        return _EMPTY_SPEND_LOG_METADATA
    return parsed


def _is_billed_retry_attempt(metadata: SpendLogsMetadata) -> bool:
    """An earlier billed retry attempt's own spend row, not a request the caller sent."""
    return BILLED_RETRY_ATTEMPT_METADATA_KEY in metadata and metadata[BILLED_RETRY_ATTEMPT_METADATA_KEY] is True


def _timed_request_duration_ms(
    payload: dict | SpendLogsPayload,
    request_status: Literal["success", "failure"],
    is_internal_call: bool,
) -> int | None:
    if is_internal_call or request_status != "success":
        return None
    duration_ms: Final = payload.get("request_duration_ms")
    if not isinstance(duration_ms, int) or duration_ms < 0:
        return None
    return duration_ms


_ENTITY_SPEND_MODELS: Final[Mapping[_EntitySpendTable, str]] = MappingProxyType(
    {
        "litellm_tagtable": "LiteLLM_TagTable",
        "litellm_agentstable": "LiteLLM_AgentsTable",
        "litellm_modelaccessgroupbudgettable": "LiteLLM_ModelAccessGroupBudgetTable",
        "litellm_projecttable": "LiteLLM_ProjectTable",
    }
)


@asynccontextmanager
async def _spend_update_tx(
    prisma_client: PrismaClient,
    table: str,
    call_type: str = "commit_spend_updates",
    outer_transaction: _SpendTransaction | None = None,
) -> AsyncGenerator[_SpendTransaction]:
    if outer_transaction is not None:
        yield outer_transaction
        return
    tx: Final[_SpendTransactionManager] = prisma_client.db.tx(timeout=timedelta(seconds=60))
    async with db_span(call_type, table), tx as transaction:
        await apply_rollup_lock_timeout(transaction)
        yield transaction


# The per-team advisory lock the team endpoints hold while changing a roster (TEAM_ADVISORY_LOCK_SQL),
# so the roster check below cannot interleave with their writes. A row lock would deadlock with the
# access-group endpoints, which lock a team row after an access-group lock.
_TEAM_ADVISORY_LOCK_SQL: Final = "SELECT pg_advisory_xact_lock(hashtext($1)) IS NULL AS locked"

# One statement adds every member's cost to their membership row. A missing row is created only
# while the user is still on the team's roster, so a spend flush landing after a removal never
# recreates the member. The rows travel as one JSON document, not as a numeric array: Prisma
# types a raw array parameter from the first batch a connection sees, so after an all-$0 batch
# (integers) every later fractional batch on that connection failed with "improper binary format".
# total_spend is floored at spend before adding: rows written before total_spend existed (migration
# 20260421135425, default 0) carry their history only in spend, so a plain increment keeps them behind forever.
_TEAM_MEMBER_SPEND_SQL: Final = """
INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, spend, total_spend)
SELECT member.user_id, member.team_id, member.cost, member.cost
FROM jsonb_to_recordset($1::jsonb) AS member(user_id text, team_id text, cost float8)
WHERE EXISTS (
    SELECT 1 FROM "LiteLLM_TeamTable" t
    WHERE t.team_id = member.team_id
      AND t.members_with_roles @> jsonb_build_array(jsonb_build_object('user_id', member.user_id))
)
   OR EXISTS (
    SELECT 1 FROM "LiteLLM_TeamMembership" m
    WHERE m.user_id = member.user_id AND m.team_id = member.team_id
)
ON CONFLICT (user_id, team_id) DO UPDATE
SET spend = "LiteLLM_TeamMembership".spend + EXCLUDED.spend,
    total_spend = GREATEST("LiteLLM_TeamMembership".total_spend, "LiteLLM_TeamMembership".spend) + EXCLUDED.total_spend
"""


async def _write_team_member_spend(transaction: _SpendTransaction, spend_by_member_key: Mapping[str, float]) -> None:
    # key is "team_id::<value>::user_id::<value>"; locks are taken in sorted team_id order like the team endpoints
    rows: Final = sorted((key.split("::")[1], key.split("::")[3], cost) for key, cost in spend_by_member_key.items())
    for team_id in dict.fromkeys(team_id for team_id, _user_id, _cost in rows):
        _ = await transaction.execute_raw(_TEAM_ADVISORY_LOCK_SQL, team_id)
    members: Final = tuple(
        _MemberSpendRow(user_id=user_id, team_id=team_id, cost=cost) for team_id, user_id, cost in rows
    )
    _ = await transaction.execute_raw(_TEAM_MEMBER_SPEND_SQL, json.dumps(members))


def get_llm_router():
    """The proxy's router, or None outside a running proxy.

    Injected rather than imported where it is used, so the savings computation stays
    a pure function of its arguments and the caller owns where the router comes from.
    """
    try:
        from litellm.proxy.proxy_server import llm_router

        return llm_router
    except Exception:  # noqa: BLE001  # no proxy in scope; savings degrade to zero
        return None


class _DeploymentRegistry(Protocol):
    def deployment_dict(self, id: str) -> Mapping[str, object] | None: ...


class _DeploymentLookup(Protocol):
    def registry(self) -> _DeploymentRegistry: ...


def _served_model_access_groups(
    router: _DeploymentLookup | None,
    served_model_id: str | None,
) -> frozenset[str] | None:
    """Access groups declared by the deployment that actually served the request.

    None when the served deployment cannot be identified, in which case the set
    attributed at auth time stands unchanged.
    """
    if router is None or not served_model_id:
        return None
    deployment: Final = router.registry().deployment_dict(served_model_id)
    if deployment is None:
        return None
    model_info: Final = deployment.get("model_info")
    if not isinstance(model_info, Mapping):
        return None
    declared: Final = model_info.get("access_groups")
    if not isinstance(declared, (list, tuple)):
        return frozenset()
    return frozenset(group for group in declared if isinstance(group, str))


def debitable_model_access_groups(
    attributed: Sequence[str] | None,
    served_model_id: str | None,
    router: _DeploymentLookup | None,
) -> tuple[str, ...]:
    """Groups to debit: the set attributed at auth time, narrowed to those the served model belongs to.

    The router may fall back to a model outside the pool auth reserved against, so the
    attributed set is the hard upper bound: a group absent from it is never debited.
    """
    ordered: Final = coerce_model_access_groups(attributed)
    if not ordered:
        return ()
    served: Final = _served_model_access_groups(router=router, served_model_id=served_model_id)
    if served is None:
        return ordered
    return tuple(group for group in ordered if group in served)


class DBSpendUpdateWriter:
    """
    Module responsible for

    1. Writing spend increments to either in memory list of transactions or to redis
    2. Reading increments from redis or in memory list of transactions and committing them to db
    """

    def __init__(
        self,
        redis_cache: RedisCache | None = None,
    ):
        self.redis_cache = redis_cache
        self.redis_update_buffer = RedisUpdateBuffer(redis_cache=self.redis_cache)
        self.pod_lock_manager = PodLockManager()
        self.tool_discovery_queue = ToolDiscoveryQueue()

    async def update_database(
        # LiteLLM management object fields
        self,
        token: str | None,
        user_id: str | None,
        end_user_id: str | None,
        team_id: str | None,
        org_id: str | None,
        # Completion object fields
        kwargs: dict | None,
        completion_response: object,
        start_time: datetime,
        end_time: datetime,
        response_cost: float | None,
        project_id: str | None = None,
    ) -> bool:
        """Record the request's spend, answering whether its cost still needs charging.

        False only for a batch retrieve whose cost row another retrieve already wrote,
        so the caller leaves the key, team, and user counters alone (LIT-7048).
        """
        from litellm.proxy.proxy_server import (
            accounting_window_keys,
            disable_spend_logs,
            general_settings,  # pyright: ignore[reportUnknownVariableType]  # legacy proxy configuration is narrowed below
            litellm_proxy_budget_name,
            prisma_client,
        )
        from litellm.proxy.utils import ProxyUpdateSpend, hash_token

        error_log_settings: Final = cast(  # cast-ok: proxy settings use string keys
            Mapping[str, object], general_settings
        )
        accounting_path: Final = "ledger"
        if kwargs is not None:
            kwargs["accounting_path"] = accounting_path  # rebind-ok: stamp shared callbacks
        try:
            verbose_proxy_logger.debug(
                "Enters prisma db call, response_cost: %s, token: %s; user_id: %s; team_id: %s",
                response_cost,
                token,
                user_id,
                team_id,
            )
            if ProxyUpdateSpend.disable_spend_updates() is True:
                return True
            hashed_token: Final = (
                hash_token(token=token) if isinstance(token, str) and token.startswith("sk-") else token
            )

            ## CREATE SPEND LOG PAYLOAD ##
            from litellm.proxy.spend_tracking.spend_tracking_utils import (
                get_logging_payload,
                get_request_model_access_groups,
            )

            llm_router: Final = get_llm_router()
            payload: Final = get_logging_payload(
                kwargs=kwargs,
                response_obj=completion_response,
                start_time=start_time,
                end_time=end_time,
                llm_router=llm_router,
            )
            billable_cost: Final = finite_cost(response_cost)
            if response_cost is not None and billable_cost is None:
                verbose_proxy_logger.warning(
                    "Spend tracking - recording a non-finite response_cost=%s as 0 for request_id=%s",
                    response_cost,
                    single_log_line(payload["request_id"]),
                )
            payload["spend"] = billable_cost or 0.0
            if isinstance(payload["startTime"], datetime):
                payload["startTime"] = payload["startTime"].isoformat()
            if isinstance(payload["endTime"], datetime):
                payload["endTime"] = payload["endTime"].isoformat()

            if org_id is not None and org_id != "":
                payload["organization_id"] = org_id

            if team_id is not None and team_id != "":
                payload["team_id"] = team_id

            from litellm.accounting.snapshot import accounting_metadata
            from litellm.proxy.hooks.model_budget_shared_windows import captured_model_windows

            request_fields: Final = _SPEND_METADATA_ADAPTER.validate_python(kwargs or {})
            model_windows: Final = captured_model_windows(request_fields)
            payload["accounting_path"] = accounting_path

            async def prepare_ledger() -> None:
                payload["metadata"] = accounting_metadata(
                    payload,
                    entities={
                        "api_key": hashed_token,
                        "user": user_id,
                        "end_user": end_user_id,
                        "team_id": team_id,
                        "organization_id": org_id,
                        "project_id": project_id,
                        "agent_id": payload["agent_id"],
                        "proxy_budget_name": litellm_proxy_budget_name if litellm.max_budget > 0 else None,
                    },
                    model_access_groups=debitable_model_access_groups(
                        get_request_model_access_groups(request_fields), payload["model_id"], llm_router
                    ),
                    windows=(
                        *(await accounting_window_keys(hashed_token, team_id) if billable_cost else ()),
                        *model_windows,
                    ),
                    router=get_llm_router,
                )

            if payload["accounting_path"] == "ledger":
                await prepare_ledger()
            keep_spend_logs: Final = disable_spend_logs is False and not (
                "status" in payload
                and payload["status"] == "failure"
                and error_log_settings.get("disable_error_logs") is True
            )
            if not await self._record_spend_log(
                payload=payload,
                prepare_ledger=prepare_ledger,
                prisma_client=prisma_client,
                disable_spend_logs=not keep_spend_logs,
                llm_router=llm_router,
            ):
                return False

            # The auto-router router-day rollup is an aggregate like the daily spend tables, so it is
            # written whether or not per-request spend logs are kept; per-session rows are not.
            await self._enqueue_autorouter_turn_transaction(
                payload=payload,
                prisma_client=prisma_client,
                spend_logs_kept=keep_spend_logs,
            )
            if keep_spend_logs:
                await self._enqueue_tool_usage_transaction(
                    payload=payload,
                    completion_response=completion_response,
                    prisma_client=prisma_client,
                    kwargs=kwargs,
                )
            else:
                verbose_proxy_logger.debug(
                    "disable_spend_logs=True. Skipping writing spend logs to db. Other spend updates - Key/User/Team table will still occur."
                )

            if kwargs is not None:
                kwargs["accounting_path"] = payload["accounting_path"]  # rebind-ok: forward persisted stamp
            self._record_request_error_safely(payload=payload, prisma_client=prisma_client)
            self._enqueue_tool_registry_upsert(
                kwargs=kwargs,
                completion_response=completion_response,
                hashed_token=hashed_token,
                team_id=team_id,
            )

            verbose_proxy_logger.debug("Runs spend update on all tables")
            return True
        except Exception:
            spend_log_error(
                "Spend tracking - update_database failed. Spend log insertion or daily transaction enqueue "
                "may not have completed for this request. "
                "response_cost=%s, token=%s, user_id=%s, team_id=%s, org_id=%s, end_user_id=%s",
                response_cost,
                token,
                user_id,
                team_id,
                org_id,
                end_user_id,
            )
            return True

    async def _record_spend_log(
        self,
        payload: SpendLogsPayload,
        prisma_client: "PrismaClient | None",
        disable_spend_logs: bool,
        llm_router: "Router | None" = None,
        prepare_ledger: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        if prisma_client is not None and _is_batch_cost_row(payload):
            return await self._claim_batch_cost_spend_log(
                payload=payload,
                prisma_client=prisma_client,
                disable_spend_logs=disable_spend_logs,
                prepare_ledger=prepare_ledger,
            )
        if disable_spend_logs is False or ("accounting_path" in payload and payload["accounting_path"] == "ledger"):
            from litellm.accounting.snapshot import lean_accounting_payload

            stored: Final = lean_accounting_payload(payload) if disable_spend_logs else payload
            await self._insert_spend_log_to_db(payload=stored, prisma_client=prisma_client, llm_router=llm_router)
        return True

    async def _claim_batch_cost_spend_log(
        self,
        payload: SpendLogsPayload,
        prisma_client: "PrismaClient",
        disable_spend_logs: bool,
        prepare_ledger: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        """Write the batch's cost row now, or learn that another retrieve already did.

        Every retrieve of one batch shares this row, so the insert that lands first owns
        the charge and every later one finds the row and charges nothing (LIT-7048). Only
        a row that recorded a charge counts: a failed retrieve, a request whose client
        picked the batch id as its call id, and the $0 row an older proxy left behind
        while the batch was still running all leave the charge to be made.
        """
        from litellm.repositories.table_repositories import SpendLogsRepository

        request_id: Final = payload["request_id"]
        from litellm.proxy.spend_tracking.spend_tracking_utils import (
            configured_spend_logs_metadata_fields,
            spend_log_row_with_retained_metadata,
        )

        row: Final = spend_log_row_with_retained_metadata(
            _batch_cost_row_to_write(payload, disable_spend_logs), configured_spend_logs_metadata_fields()
        )
        spend_logs: Final = SpendLogsRepository(prisma_client).table
        try:
            claimed: Final = await SpendLogsRepository(prisma_client).insert_with_deliveries(
                [prisma_client.jsonify_object(row)]
            )
            if claimed == 1:
                return True
            existing: Final = await spend_logs.find_unique(where={"request_id": request_id})
        except Exception as e:  # noqa: BLE001  # prisma raises its own hierarchy; an unreachable DB queues the row like any other spend log
            verbose_proxy_logger.warning(
                "Could not claim spend row %s for a batch's cost, queueing it: %s", request_id, e
            )
            await self._insert_spend_log_to_db(payload=prisma_client.jsonify_object(row), prisma_client=prisma_client)
            return True
        if existing is None:
            await self._insert_spend_log_to_db(payload=prisma_client.jsonify_object(row), prisma_client=prisma_client)
            return True
        if existing.call_type != CallTypes.aretrieve_batch.value or existing.status != "success":
            from uuid import NAMESPACE_URL, uuid5

            return await self._claim_batch_cost_spend_log(
                payload={**payload, "request_id": str(uuid5(NAMESPACE_URL, "litellm:batch-cost:" + request_id))},
                prisma_client=prisma_client,
                disable_spend_logs=disable_spend_logs,
                prepare_ledger=prepare_ledger,
            )
        if existing.spend > 0:
            verbose_proxy_logger.debug("Cost tracking skipped: spend row %s already charged this batch", request_id)
            return False
        was_ledger: Final = "accounting_path" in payload and payload["accounting_path"] == "ledger"
        payload["accounting_path"] = (  # rebind-ok: adopt persisted stamp on takeover
            "ledger" if getattr(existing, "accounting_path", None) == "ledger" else "legacy"
        )
        if payload["accounting_path"] == "ledger" and not was_ledger and prepare_ledger is not None:
            await prepare_ledger()
        return await self._take_over_uncharged_batch_cost_row(payload=payload, prisma_client=prisma_client, row=row)

    async def _take_over_uncharged_batch_cost_row(
        self, payload: SpendLogsPayload, prisma_client: "PrismaClient", row: Mapping[str, object]
    ) -> bool:
        """Take the batch's cost row over from the poll that left it charging nothing.

        A pre-upgrade proxy wrote that row every time it polled the batch while it was still
        running, so the charge is still to be made and the row still has to end up carrying
        it. The row stops matching the moment it carries a charge, so it is one retrieve that
        takes it over and charges, and every later one reads the charge and charges nothing.
        """
        from litellm.repositories.table_repositories import SpendLogsRepository

        request_id: Final = payload["request_id"]
        if payload["spend"] <= 0:
            verbose_proxy_logger.debug(
                "Cost tracking skipped: this batch costs nothing and spend row %s says so", request_id
            )
            return False
        try:
            takeover: Final = await SpendLogsRepository(prisma_client).take_over_batch(row)
        except Exception as e:  # noqa: BLE001  # prisma raises its own hierarchy; the next retrieve takes the row over
            verbose_proxy_logger.warning(
                "Could not take over spend row %s, leaving this batch's cost to the next retrieve: %s", request_id, e
            )
            return False
        return takeover.charged

    async def _enqueue_tool_usage_transaction(
        self,
        payload: SpendLogsPayload,
        completion_response: object,
        prisma_client: "PrismaClient | None",
        kwargs: "dict | None" = None,
    ) -> None:
        try:
            if prisma_client is None:
                return
            from litellm.proxy.db.spend_log_tool_index import (
                build_tool_usage_transaction,
            )

            transaction: Final = build_tool_usage_transaction(
                request_id=payload["request_id"],
                start_time_iso=str(payload["startTime"]),
                mcp_namespaced_tool_name=payload.get("mcp_namespaced_tool_name"),
                spend=payload["spend"],
                total_tokens=payload["total_tokens"],
                completion_response=completion_response,
                realtime_tool_calls=_CALL_KWARG_VALUE.validate_python((kwargs or {}).get("realtime_tool_calls")),
            )
            if transaction is None:
                return
            async with prisma_client._tool_usage_transactions_lock:
                prisma_client.tool_usage_transactions.append(transaction)
        except Exception as e:
            verbose_proxy_logger.debug("_enqueue_tool_usage_transaction error (non-blocking): %s", e)

    async def _enqueue_autorouter_turn_transaction(
        self,
        payload: SpendLogsPayload,
        prisma_client: "PrismaClient | None",
        spend_logs_kept: bool = True,
    ) -> None:
        try:
            if prisma_client is None:
                return
            metadata_raw: Final = payload.get("metadata")
            if not metadata_raw:
                return
            metadata: Final = _SPEND_METADATA_ADAPTER.validate_json(metadata_raw)
            routing_decision: Final = metadata.get("routing_decision")
            if not isinstance(routing_decision, Mapping) or not routing_decision:
                return
            from litellm.proxy.db.autorouter_session_rollup import (
                build_autorouter_turn_transaction,
            )

            usage_object_raw: Final = metadata.get("usage_object")
            cost_breakdown: Final = metadata.get("cost_breakdown")
            savings_estimate: Final = metadata.get("autorouter_savings_estimate")
            savings_spend: Final = compute_savings_spend(
                model=payload.get("model"),
                custom_llm_provider=payload.get("custom_llm_provider"),
                compression_saved_tokens=0,
                gateway_injected_cache=marks_gateway_injection(metadata, payload.get("model_id")),
                routing_decision=routing_decision,
                usage_object=usage_object_raw if isinstance(usage_object_raw, dict) else None,
                model_id=payload.get("model_id"),
                llm_router=get_llm_router,
                cost_breakdown=cost_breakdown if isinstance(cost_breakdown, Mapping) else None,
                recorded_autorouter_savings=metadata.get("autorouter_savings"),
                recorded_autorouter_savings_estimate=(
                    savings_estimate if isinstance(savings_estimate, Mapping) else None
                ),
                billed_at=payload.get("endTime"),
            )
            transaction: Final = build_autorouter_turn_transaction(
                payload=payload,
                metadata=metadata,
                saved_spend=savings_spend.autorouter,
            )
            try:
                # A baseline observation publishes only once its spend log exists, so without spend logs
                # it could never publish; the plain turn still carries this request's recorded savings.
                if spend_logs_kept and await self._enqueue_baseline_accounting(
                    payload, metadata, transaction, prisma_client
                ):
                    return
            except Exception:  # noqa: BLE001  # optional baseline capture must preserve the original actual-spend rollup
                verbose_proxy_logger.warning("Auto-router baseline observation was unavailable; actual turn retained")
            if transaction is None:
                return
            # Without spend logs only the router-day aggregate is kept: an empty session id makes the
            # session upserts skip the row, so no per-session record is stored.
            kept: Final = transaction if spend_logs_kept else dataclasses.replace(transaction, session_id="")
            async with prisma_client._autorouter_turn_transactions_lock:
                prisma_client.autorouter_turn_transactions.append(kept)
        except Exception as e:  # noqa: BLE001  # a metrics enqueue must never fail the spend write
            verbose_proxy_logger.debug("_enqueue_autorouter_turn_transaction error (non-blocking): %s", e)

    async def _enqueue_baseline_accounting(
        self,
        payload: SpendLogsPayload,
        metadata: Mapping[str, object],
        turn: "AutoRouterTurnTransaction | None",
        prisma_client: "PrismaClient",
    ) -> bool:
        from litellm.proxy.db.baseline_accounting import (
            BaselineAccountingRecord,
        )
        from litellm.proxy.hooks.autorouter_baseline_cache import CapturedBaselineObservation
        from litellm.proxy.spend_tracking.savings import baseline_cost_snapshot

        serialized: Final = metadata.get("autorouter_baseline_observation")
        if not isinstance(serialized, str):
            return False
        captured: Final = CapturedBaselineObservation.model_validate_json(serialized)
        if captured.api_key != payload["api_key"] or captured.session_id != payload["session_id"]:
            return False
        decision: Final = _SPEND_METADATA_ADAPTER.validate_python(
            metadata.get("routing_decision") or MappingProxyType({})
        )
        breakdown: Final = _SPEND_METADATA_ADAPTER.validate_python(
            metadata.get("cost_breakdown") or MappingProxyType({})
        )
        daily: Final = await self._baseline_daily_attribution(payload, prisma_client)
        record: Final = BaselineAccountingRecord(
            scope=captured.scope,
            api_key=captured.api_key,
            session_id=captured.session_id,
            router_name=captured.router_name,
            baseline_model=captured.baseline_model,
            observation=captured.observation.model_copy(update=MappingProxyType({"request_id": payload["request_id"]})),
            pricing=baseline_cost_snapshot(captured.model, captured.prices, payload["spend"], breakdown, decision),
            turn=turn,
            daily=daily,
        )
        async with prisma_client.baseline_accounting_lock:
            if len(prisma_client.baseline_accounting_transactions) >= 10000:
                verbose_proxy_logger.warning("Auto-router baseline observation queue is full")
                return False
            prisma_client.baseline_accounting_transactions.append(record)
        from litellm.proxy.utils import request_spend_log_flush

        request_spend_log_flush(prisma_client)
        return True

    async def _baseline_daily_attribution(
        self,
        payload: SpendLogsPayload,
        prisma_client: "PrismaClient",
    ) -> "DailyBaselineAttribution | None":
        from litellm.proxy.db.baseline_accounting import DailyBaselineAttribution, DailyBaselineTarget

        normalized: Final = cast(SpendLogsPayload, MappingProxyType({**payload, "end_user_id": payload["end_user"]}))
        bases: Final = tuple(
            zip(
                DAILY_SPEND_TABLES,
                await asyncio.gather(
                    *(
                        self._common_add_spend_log_transaction_to_daily_transaction(  # pyright: ignore[reportUnknownMemberType]  # legacy payload union; this caller supplies a validated spend payload
                            normalized,
                            prisma_client,
                            "request_tags" if entity == "tag" else entity,
                        )
                        for entity in DAILY_SPEND_TABLES
                    )
                ),
            )
        )
        base: Final = next((base for _, base in bases if base is not None), None)
        if base is None:
            return None
        return DailyBaselineAttribution(
            date=base["date"],
            api_key=base["api_key"],
            model=base.get("model"),
            custom_llm_provider=base.get("custom_llm_provider"),
            model_group=base.get("model_group"),
            endpoint=base.get("endpoint"),
            mcp_namespaced_tool_name=base.get("mcp_namespaced_tool_name"),
            targets=tuple(
                DailyBaselineTarget(entity=entity, entity_id=identity)
                for entity, values in bases
                if values is not None
                for identity in daily_spend_entity_ids(payload, entity)
            ),
        )

    def _enqueue_tool_registry_upsert(
        self,
        kwargs: dict | None,
        completion_response: object,
        hashed_token: str | None = None,
        team_id: str | None = None,
    ) -> None:
        """
        Extract tool names from the LLM request and response and enqueue them
        for upsert into LiteLLM_ToolTable via ToolDiscoveryQueue.

        Handles four sources:
        - MCP tools: standard_logging_object.mcp_tool_call_metadata.namespaced_tool_name
        - Response tool_calls (OpenAI / Anthropic pass-through converted to OpenAI format):
            completion_response.choices[].message.tool_calls[].function.name
        - Request tools array (OpenAI format): kwargs["tools"][].function.name
        - Request tools array (Anthropic /messages format): kwargs["passthrough_logging_payload"]
            ["request_body"]["tools"][].name
        """
        try:
            if kwargs is None:
                return

            # Extract key_alias from kwargs metadata if available
            key_alias: str | None = None
            _litellm_params: Final = kwargs.get("litellm_params") or {}
            _metadata: Final = _litellm_params.get("metadata") or {}
            key_alias = _metadata.get("user_api_key_alias") or None
            user_agent: Final = _metadata.get("user_agent") or None

            def _enqueue(tool_name: str, origin: str = "user_defined") -> None:
                self.tool_discovery_queue.add_update(
                    ToolDiscoveryQueueItem(
                        tool_name=tool_name,
                        origin=origin,
                        key_hash=hashed_token,
                        team_id=team_id,
                        key_alias=key_alias,
                        user_agent=user_agent,
                    )
                )

            # --- MCP tool calls ---
            sl_object: Final = kwargs.get("standard_logging_object")
            if sl_object is not None:
                mcp_metadata: Final = (sl_object.get("metadata", {}) or {}).get("mcp_tool_call_metadata")
                if mcp_metadata and isinstance(mcp_metadata, dict):
                    tool_name = mcp_metadata.get("namespaced_tool_name") or mcp_metadata.get("name")
                    mcp_server_name: Final = mcp_metadata.get("mcp_server_name")
                    if tool_name:
                        _enqueue(tool_name, origin=mcp_server_name or "user_defined")

            # --- Tools from request body (OpenAI format: tools[].function.name) ---
            request_tools: Final = kwargs.get("tools") or []
            for tool_def in request_tools:
                if not isinstance(tool_def, dict):
                    continue
                fn = tool_def.get("function") or {}
                name = fn.get("name") if isinstance(fn, dict) else None
                if name:
                    _enqueue(name)

            # --- Tools from Anthropic /messages pass-through request body
            #     (Anthropic format: tools[].name, no "function" wrapper) ---
            passthrough_payload: Final = kwargs.get("passthrough_logging_payload") or {}
            request_body: Final = (
                passthrough_payload.get("request_body") if isinstance(passthrough_payload, dict) else None
            ) or {}
            for tool_def in request_body.get("tools") or []:
                if not isinstance(tool_def, dict):
                    continue
                name = tool_def.get("name")
                if name:
                    _enqueue(name)

            # --- Response tool_calls (OpenAI format; Anthropic pass-through converts tool_use here) ---
            from litellm.proxy.db.spend_log_tool_index import response_tool_call_names

            for tool_name in response_tool_call_names(completion_response):
                _enqueue(tool_name)
        except Exception as e:
            verbose_proxy_logger.debug("_enqueue_tool_registry_upsert error (non-blocking): %s", e)

    async def _insert_spend_log_to_db(
        self,
        payload: dict[str, object] | SpendLogsPayload,
        prisma_client: PrismaClient | None = None,
        spend_logs_url: str | None = os.getenv("SPEND_LOGS_URL"),
        llm_router: "Router | None" = None,
    ) -> PrismaClient | None:
        verbose_proxy_logger.debug(
            "Writing spend log to db - request_id: {}, spend: {}".format(
                payload.get("request_id"), payload.get("spend")
            )
        )
        if prisma_client is not None and spend_logs_url is not None or prisma_client is not None:
            from litellm.proxy.utils import enqueue_spend_logs, request_spend_log_flush

            await enqueue_spend_logs(prisma_client, (payload,))
            if _turn_chains_through_the_db(_SPEND_METADATA_ADAPTER.validate_python(payload), llm_router):
                request_spend_log_flush(prisma_client)
        else:
            verbose_proxy_logger.debug("prisma_client is None. Skipping writing spend logs to db.")

        return prisma_client

    async def db_update_spend_transaction_handler(
        self,
        prisma_client: PrismaClient,
        n_retry_times: int,
        proxy_logging_obj: ProxyLogging,
    ):
        from litellm.accounting.worker import flush_accounting_deliveries

        try:
            await flush_accounting_deliveries(self, prisma_client, proxy_logging_obj)
        except Exception:  # noqa: BLE001  # pending deliveries remain available for the next scheduled retry
            verbose_proxy_logger.exception("Accounting delivery flush failed; pending work will retry")

    @staticmethod
    async def handle_spend_update_failure(
        e: Exception,
        attempt: int,
        n_retry_times: int,
        start_time: float,
        proxy_logging_obj: ProxyLogging,
    ) -> None:
        """Retry a failed spend-update transaction on connection errors, deadlocks or a rollup
        ``lock_timeout`` (55P03), else re-raise. All three roll the transaction back before any
        increment applied, so re-sending the same batch cannot double-count."""
        from litellm.proxy.utils import raise_failed_update_spend_exception

        is_retryable = (
            isinstance(e, DB_RETRY_SAFE_ERROR_TYPES)
            or PrismaDBExceptionHandler.is_deadlock_error(e)
            or PrismaDBExceptionHandler.is_lock_timeout_error(e)
        )
        if not is_retryable or attempt >= n_retry_times:
            raise_failed_update_spend_exception(e=e, start_time=start_time, proxy_logging_obj=proxy_logging_obj)
        verbose_proxy_logger.warning(
            "Retrying spend update after retryable DB error (attempt %s/%s): %s",
            attempt + 1,
            n_retry_times,
            e,
        )
        await asyncio.sleep(random.uniform(2.0**attempt, 2.0 ** (attempt + 1)))

    _handle_spend_update_failure = handle_spend_update_failure

    @staticmethod
    async def invalidate_accounting_entities(
        contributions: tuple["Contribution", ...], proxy_logging_obj: ProxyLogging
    ) -> None:
        await DBSpendUpdateWriter._invalidate_project_caches(
            tuple(dict(item.key)["project_id"] for item in contributions if item.table == "LiteLLM_ProjectTable"),
            proxy_logging_obj,
        )
        from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

        details: Final = _SPEND_METADATA_ADAPTER.validate_python(
            proxy_logging_obj.call_details  # pyright: ignore[reportUnknownMemberType]  # legacy logging metadata is validated at this boundary
        )
        user_api_key_cache: Final = details.get("user_api_key_cache")
        if isinstance(user_api_key_cache, UserApiKeyCache):
            with service_target(AUTH_OBJECTS_TARGET):
                for item in contributions:
                    if item.table == "LiteLLM_TeamMembership":
                        await user_api_key_cache.async_delete_cache(
                            key=f"team_membership:{dict(item.key)['user_id']}:{dict(item.key)['team_id']}"
                        )

    async def apply_accounting_entities(
        self,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        transaction: _SpendTransaction,
        updates: DBSpendUpdateTransactions,
    ) -> None:
        await self._commit_spend_updates_to_db(
            prisma_client, 0, proxy_logging_obj, updates, outer_transaction=transaction
        )

    async def _commit_spend_updates_to_db(
        self,
        prisma_client: PrismaClient,
        n_retry_times: int,
        proxy_logging_obj: ProxyLogging,
        db_spend_update_transactions: DBSpendUpdateTransactions,
        on_table_committed: Callable[[_SpendTableName], None] | None = None,
        outer_transaction: _SpendTransaction | None = None,
    ):
        """
        Commits all the spend `UPDATE` transactions to the Database

        """
        from litellm.proxy.utils import ProxyUpdateSpend

        ### UPDATE USER TABLE ###
        user_list_transactions: Final = db_spend_update_transactions["user_list_transactions"]
        verbose_proxy_logger.debug("User Spend transactions: %s", user_list_transactions)
        if user_list_transactions is not None and len(user_list_transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with _spend_update_tx(
                        prisma_client, "LiteLLM_UserTable", outer_transaction=outer_transaction
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            # Sort by ID for consistent lock ordering across pods to prevent deadlocks.
                            # batch_() issues statements sequentially within the tx, so iteration
                            # order = lock acquisition order.
                            for user_id, response_cost in sorted(user_list_transactions.items()):
                                batcher.litellm_usertable.update_many(
                                    where={"user_id": user_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await self.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )
        if on_table_committed is not None:
            on_table_committed("user_list_transactions")

        ### UPDATE END-USER TABLE ###
        end_user_list_transactions: Final = db_spend_update_transactions["end_user_list_transactions"]
        verbose_proxy_logger.debug("End-User Spend transactions: %s", end_user_list_transactions)
        if end_user_list_transactions is not None and len(end_user_list_transactions.keys()) > 0:
            await ProxyUpdateSpend.update_end_user_spend(
                n_retry_times=n_retry_times,
                prisma_client=prisma_client,
                proxy_logging_obj=proxy_logging_obj,
                end_user_list_transactions=end_user_list_transactions,
                outer_transaction=outer_transaction,
            )
        if on_table_committed is not None:
            on_table_committed("end_user_list_transactions")
        ### UPDATE KEY TABLE ###
        key_list_transactions: Final = db_spend_update_transactions["key_list_transactions"]
        verbose_proxy_logger.debug("KEY Spend transactions: %s", key_list_transactions)
        if key_list_transactions is not None and len(key_list_transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with _spend_update_tx(
                        prisma_client, "LiteLLM_VerificationToken", outer_transaction=outer_transaction
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            # Sort by token for consistent lock ordering across pods to prevent deadlocks.
                            for token, response_cost in sorted(key_list_transactions.items()):
                                spend_increment: _SpendIncrement = {"increment": response_cost}
                                batcher.litellm_verificationtoken.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"token": token},
                                    data={
                                        "spend": spend_increment,
                                        "total_spend": spend_increment,
                                        "last_active": datetime.now(timezone.utc),
                                    },
                                )
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await self.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )
        if on_table_committed is not None:
            on_table_committed("key_list_transactions")

        ### UPDATE TEAM TABLE ###
        team_list_transactions: Final = db_spend_update_transactions["team_list_transactions"]
        verbose_proxy_logger.debug("Team Spend transactions: %s", team_list_transactions)
        if team_list_transactions is not None and len(team_list_transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with _spend_update_tx(
                        prisma_client, "LiteLLM_TeamTable", outer_transaction=outer_transaction
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            # Sort by team_id for consistent lock ordering across pods to prevent deadlocks.
                            for team_id, response_cost in sorted(team_list_transactions.items()):
                                verbose_proxy_logger.debug(
                                    "Updating spend for team id=%s by %s", team_id, response_cost
                                )
                                batcher.litellm_teamtable.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"team_id": team_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await self.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )
        if on_table_committed is not None:
            on_table_committed("team_list_transactions")

        ### UPDATE TEAM Membership TABLE with spend ###
        team_member_list_transactions: Final = db_spend_update_transactions["team_member_list_transactions"]
        verbose_proxy_logger.debug("Team Membership Spend transactions: %s", team_member_list_transactions)
        if team_member_list_transactions is not None and len(team_member_list_transactions.keys()) > 0:
            # Track which team memberships will be updated for cache invalidation
            team_memberships_to_invalidate: Final[list[tuple[str, str]]] = []
            for key in team_member_list_transactions:
                # key is "team_id::<value>::user_id::<value>"
                team_id = key.split("::")[1]
                user_id = key.split("::")[3]
                team_memberships_to_invalidate.append((user_id, team_id))

            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with _spend_update_tx(
                        prisma_client, "LiteLLM_TeamMembership", outer_transaction=outer_transaction
                    ) as transaction:
                        await _write_team_member_spend(transaction, team_member_list_transactions)
                    # Transaction succeeded, break out of retry loop
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await self.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )
            if on_table_committed is not None:
                on_table_committed("team_member_list_transactions")

            # Invalidate cache for updated team memberships
            # This ensures budget checks read fresh spend data from the database
            if outer_transaction is None and team_memberships_to_invalidate and proxy_logging_obj is not None:
                user_api_key_cache: Final = proxy_logging_obj.call_details.get("user_api_key_cache")
                if user_api_key_cache is not None:
                    with service_target(AUTH_OBJECTS_TARGET):
                        for user_id, team_id in team_memberships_to_invalidate:
                            cache_key = f"team_membership:{user_id}:{team_id}"
                            await user_api_key_cache.async_delete_cache(key=cache_key)
                            verbose_proxy_logger.debug(
                                "Invalidated team membership cache for user_id=%s, team_id=%s", user_id, team_id
                            )
        elif on_table_committed is not None:
            on_table_committed("team_member_list_transactions")

        ### UPDATE ORG TABLE ###
        org_list_transactions: Final = db_spend_update_transactions["org_list_transactions"]
        verbose_proxy_logger.debug("Org Spend transactions: %s", org_list_transactions)
        if org_list_transactions is not None and len(org_list_transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with _spend_update_tx(
                        prisma_client, "LiteLLM_OrganizationTable", outer_transaction=outer_transaction
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            # Sort by org_id for consistent lock ordering across pods to prevent deadlocks.
                            for org_id, response_cost in sorted(org_list_transactions.items()):
                                batcher.litellm_organizationtable.update_many(  # 'update_many' prevents error from being raised if no row exists
                                    where={"organization_id": org_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await self.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )
        if on_table_committed is not None:
            on_table_committed("org_list_transactions")

        org_member_list_transactions: Final = db_spend_update_transactions.get("org_member_list_transactions")
        verbose_proxy_logger.debug("Org Membership Spend transactions: %s", org_member_list_transactions)
        if org_member_list_transactions is not None and len(org_member_list_transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with (
                        _spend_update_tx(
                            prisma_client, "LiteLLM_OrganizationMembership", outer_transaction=outer_transaction
                        ) as transaction,
                        transaction.batch_() as batcher,
                    ):
                        for key, response_cost in sorted(org_member_list_transactions.items()):
                            _, quoted_org_id, _, quoted_user_id = key.split("::")
                            batcher.litellm_organizationmembership.update_many(
                                where={"organization_id": unquote(quoted_org_id), "user_id": unquote(quoted_user_id)},
                                data={"spend": {"increment": response_cost}},
                            )
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await self.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )
        if on_table_committed is not None:
            on_table_committed("org_member_list_transactions")

        ### UPDATE PROJECT TABLE ###
        project_list_transactions: Final = db_spend_update_transactions.get("project_list_transactions")
        await DBSpendUpdateWriter._update_entity_spend_in_db(
            entity_name="Project",
            transactions=project_list_transactions,
            outer_transaction=outer_transaction,
            table_accessor="litellm_projecttable",
            where_field="project_id",
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
        )
        if on_table_committed is not None:
            on_table_committed("project_list_transactions")
        if outer_transaction is None:
            await DBSpendUpdateWriter._invalidate_project_caches(
                project_ids=tuple(project_list_transactions or ()),
                proxy_logging_obj=proxy_logging_obj,
            )

        ### UPDATE TAG TABLE ###
        tag_list_transactions: Final = db_spend_update_transactions["tag_list_transactions"]
        await DBSpendUpdateWriter._update_entity_spend_in_db(
            entity_name="Tag",
            transactions=tag_list_transactions,
            outer_transaction=outer_transaction,
            table_accessor="litellm_tagtable",
            where_field="tag_name",
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
        )
        if on_table_committed is not None:
            on_table_committed("tag_list_transactions")

        ### UPDATE MODEL ACCESS GROUP TABLE ###
        model_access_group_list_transactions: Final = db_spend_update_transactions.get(
            "model_access_group_list_transactions"
        )
        await DBSpendUpdateWriter._update_entity_spend_in_db(
            entity_name="Model access group",
            transactions=model_access_group_list_transactions,
            outer_transaction=outer_transaction,
            table_accessor="litellm_modelaccessgroupbudgettable",
            where_field="access_group_name",
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
        )
        if on_table_committed is not None:
            on_table_committed("model_access_group_list_transactions")

        ### UPDATE AGENT TABLE ###
        agent_list_transactions: Final = db_spend_update_transactions["agent_list_transactions"]
        await DBSpendUpdateWriter._update_entity_spend_in_db(
            entity_name="Agent",
            transactions=agent_list_transactions,
            outer_transaction=outer_transaction,
            table_accessor="litellm_agentstable",
            where_field="agent_id",
            n_retry_times=n_retry_times,
            prisma_client=prisma_client,
            proxy_logging_obj=proxy_logging_obj,
        )
        if on_table_committed is not None:
            on_table_committed("agent_list_transactions")

    @staticmethod
    @with_service_target(AUTH_OBJECTS_TARGET)
    async def _invalidate_project_caches(project_ids: Sequence[str], proxy_logging_obj: ProxyLogging | None) -> None:
        if not project_ids or proxy_logging_obj is None:
            return
        user_api_key_cache: Final = proxy_logging_obj.call_details.get("user_api_key_cache")
        if user_api_key_cache is None:
            return
        for project_id in project_ids:
            await user_api_key_cache.async_delete_cache(key=project_cache_key(project_id))

    @staticmethod
    async def _update_entity_spend_in_db(
        entity_name: str,
        transactions: dict[str, float] | None,
        table_accessor: _EntitySpendTable,
        where_field: str,
        n_retry_times: int,
        prisma_client: PrismaClient,
        proxy_logging_obj: ProxyLogging,
        outer_transaction: _SpendTransaction | None = None,
    ):
        """
        Helper function to update spend for any entity type (team, org, tag, etc).

        Args:
            entity_name: Name of entity for logging (e.g., "Team", "Org", "Tag")
            transactions: Dictionary of {entity_id: response_cost}
            table_accessor: Prisma table accessor (e.g., prisma_client.db.litellm_teamtable)
            where_field: Field name for where clause (e.g., "team_id", "organization_id", "tag_name")
            n_retry_times: Number of retries on failure
            prisma_client: Prisma client instance
            proxy_logging_obj: Proxy logging object
            outer_transaction: Existing ledger transaction owning the complete delivery update
        """
        verbose_proxy_logger.debug("%s Spend transactions: %s", entity_name, transactions)
        if transactions is not None and len(transactions.keys()) > 0:
            for i in range(n_retry_times + 1):
                start_time = time.time()
                try:
                    async with _spend_update_tx(
                        prisma_client, _ENTITY_SPEND_MODELS[table_accessor], outer_transaction=outer_transaction
                    ) as transaction:
                        async with transaction.batch_() as batcher:
                            # Sort by entity_id for consistent lock ordering across pods to prevent deadlocks.
                            for entity_id, response_cost in sorted(transactions.items()):
                                verbose_proxy_logger.debug(
                                    "Updating spend for %s %s=%s by %s",
                                    entity_name,
                                    where_field,
                                    entity_id,
                                    response_cost,
                                )
                                _entity_spend_table(batcher, table_accessor).update_many(
                                    where={where_field: entity_id},
                                    data={"spend": {"increment": response_cost}},
                                )
                    break
                except Exception as e:
                    if outer_transaction is not None:
                        raise
                    await DBSpendUpdateWriter.handle_spend_update_failure(
                        e=e,
                        attempt=i,
                        n_retry_times=n_retry_times,
                        start_time=start_time,
                        proxy_logging_obj=proxy_logging_obj,
                    )

    def _record_request_error_safely(self, *, payload: SpendLogsPayload, prisma_client: PrismaClient | None) -> None:
        try:
            self._record_request_error(payload=payload, prisma_client=prisma_client)
        except Exception as error:  # noqa: BLE001  # error rollup failure must not skip sibling writes
            _report_spend_update_step_failure("_record_request_error", payload=payload, exc=error)

    @staticmethod
    def _record_request_error(*, payload: SpendLogsPayload, prisma_client: PrismaClient | None) -> None:
        if prisma_client is None:
            return
        start_time: Final = payload["startTime"]
        date: Final = start_time.isoformat() if isinstance(start_time, datetime) else str(start_time or "")
        if not date:
            return
        metadata: Final = _parse_spend_log_metadata(payload)
        internal_origin: Final = bool(metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY))  # pyright: ignore[reportUnknownMemberType]  # TypedDict.get overloads carry Any defaults
        request_error_accumulator.record(
            payload=payload,
            request_status=prisma_client.get_request_status(payload),  # pyright: ignore[reportUnknownMemberType]  # legacy payload union; this caller supplies a typed spend payload
            date=date.split("T")[0],
            is_internal_call=internal_origin or _is_billed_retry_attempt(metadata),
        )

    async def _common_add_spend_log_transaction_to_daily_transaction(
        self,
        payload: dict | SpendLogsPayload,
        prisma_client: PrismaClient,
        type: Literal["user", "team", "org", "request_tags", "end_user", "agent"] = "user",
    ) -> BaseDailySpendTransaction | None:
        entity: Final = "tag" if type == "request_tags" else type
        identity_payload: Final = (
            MappingProxyType({**payload, "end_user": payload.get("end_user_id")}) if type == "end_user" else payload
        )
        if not daily_spend_entity_ids(identity_payload, entity):
            return None
        expected_keys: Final = ("startTime", "api_key")
        if not all(key in payload for key in expected_keys):
            verbose_proxy_logger.debug(
                "Missing expected keys: %s, in payload, skipping from daily_user_spend_transactions", expected_keys
            )
            return None

        any_expected_keys: Final = ["model", "mcp_namespaced_tool_name"]
        if not any(key in payload for key in any_expected_keys):
            verbose_proxy_logger.debug(
                "Missing any expected keys: %s, in payload, skipping from daily_user_spend_transactions",
                any_expected_keys,
            )
            return None
        elif "mcp_namespaced_tool_name" in payload:
            pass
        elif "model" in payload and ("custom_llm_provider" not in payload or "model_group" not in payload):
            verbose_proxy_logger.debug(
                "Missing custom_llm_provider or model_group in payload, skipping from daily_user_spend_transactions"
            )
            return None

        # TODO: remove the successful_requests/failed_requests counters below once the
        # admin UI has fully migrated to LiteLLM_DailyGatewayRequests, which is now the
        # source of truth for SGR. This path derives the counts from spend-log metadata
        # rather than from what the gateway answered, so the two intentionally disagree
        # (see litellm/proxy/middleware/billable_request_metrics_middleware.py). The
        # spend, token and per-entity columns written here stay either way.
        request_status: Final = prisma_client.get_request_status(payload)
        verbose_proxy_logger.debug("Logged request status: %s", request_status)
        _metadata: Final = _parse_spend_log_metadata(payload)
        usage_obj: Final = _metadata.get("usage_object", {}) or {}
        if isinstance(payload["startTime"], datetime):
            start_time: Final = payload["startTime"].isoformat()
            date = start_time.split("T")[0]
        elif isinstance(payload["startTime"], str):
            date = payload["startTime"].split("T")[0]
        else:
            verbose_proxy_logger.debug(
                "Invalid start time: %s, skipping from daily_user_spend_transactions", payload["startTime"]
            )
            return None
        try:
            # Map call_type to endpoint using ROUTE_ENDPOINT_MAPPING
            call_type: Final = payload.get("call_type", None)
            endpoint = None
            if call_type:
                endpoint = ROUTE_ENDPOINT_MAPPING.get(call_type, None)

            is_internal_call: Final = bool(_metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY))
            caller_request: Final = not is_internal_call and not _is_billed_retry_attempt(_metadata)
            cache_read_input_tokens: Final = extract_cache_read_tokens(usage_obj)
            compression_saved_tokens: Final = extract_compression_saved_tokens(_metadata)
            savings_spend: Final = compute_savings_spend(
                model=payload.get("model", None),
                custom_llm_provider=payload.get("custom_llm_provider", None),
                compression_saved_tokens=compression_saved_tokens,
                gateway_injected_cache=marks_gateway_injection(_metadata, payload.get("model_id")),
                routing_decision=_metadata.get("routing_decision"),
                model_id=payload.get("model_id"),
                llm_router=get_llm_router,
                usage_object=usage_obj,
                cost_breakdown=_metadata.get("cost_breakdown"),
                recorded_autorouter_savings=_metadata.get("autorouter_savings"),
                recorded_autorouter_savings_estimate=_metadata.get("autorouter_savings_estimate"),
                billed_at=payload.get("endTime"),
            )
            timed_duration_ms: Final = _timed_request_duration_ms(payload, request_status, is_internal_call)

            daily_transaction: Final = BaseDailySpendTransaction(
                date=date,
                api_key=payload["api_key"],
                model=payload.get("model", None),
                model_group=payload.get("model_group", None),
                mcp_namespaced_tool_name=payload.get("mcp_namespaced_tool_name", None),
                custom_llm_provider=payload.get("custom_llm_provider", None),
                endpoint=endpoint,
                prompt_tokens=payload["prompt_tokens"],
                completion_tokens=payload["completion_tokens"],
                spend=payload["spend"],
                # Internal sub-calls (auto-router classifier, shadow eval's shadow and
                # judge) bill real spend and tokens to the key, but they are not
                # requests the caller made: counting them inflates request-volume
                # readers, and an auto-router savings figure computed on a shadow
                # duplicate credits savings for traffic no user sent.
                api_requests=1 if caller_request else 0,
                successful_requests=1 if caller_request and request_status == "success" else 0,
                failed_requests=1 if caller_request and request_status != "success" else 0,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=extract_cache_creation_tokens(usage_obj),
                compression_saved_tokens=compression_saved_tokens,
                compression_savings_spend=savings_spend.compression,
                prompt_caching_savings_spend=savings_spend.prompt_caching,
                gateway_injected_caching_savings_spend=savings_spend.gateway_injected_caching,
                autorouter_savings_spend=0.0 if is_internal_call else savings_spend.autorouter,
                total_response_time_ms=timed_duration_ms or 0,
                timed_requests=0 if timed_duration_ms is None else 1,
            )
            return daily_transaction
        except Exception as e:
            raise e
