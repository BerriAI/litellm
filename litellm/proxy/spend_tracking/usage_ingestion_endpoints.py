import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    Literal,
    NamedTuple,
    TypeAlias,
    cast,  # noqa: TID251  # untyped tx boundary needs cast for the shim
)

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, ProxyException, SpendLogsPayload, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import hash_token

router: Final = APIRouter()

MAX_RECORDS_PER_REQUEST: Final = 1000


class ExternalUsageRecord(BaseModel):
    api_key: str | None = Field(
        default=None, min_length=1, description="Raw virtual key (sk-...) to attribute usage to. Never logged."
    )
    api_key_hash: str | None = Field(
        default=None,
        min_length=1,
        description="SHA-256 hash of the virtual key. Use instead of api_key to avoid submitting raw keys.",
    )
    model: str = Field(min_length=1)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    start_time: datetime
    end_time: datetime | None = None
    cost: float | None = Field(
        default=None, ge=0, description="Explicit cost in USD. Computed from litellm pricing when omitted."
    )
    idempotency_key: str | None = Field(
        default=None, max_length=255, description="Becomes the spend-log request_id for dedup on retries."
    )
    tags: list[str] | None = None  # mutable-ok: serialized as a JSON array by spend logs
    end_user_id: str | None = None

    @model_validator(mode="after")
    def end_time_not_before_start_time(self) -> "ExternalUsageRecord":
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("end_time must not be before start_time")
        return self

    @model_validator(mode="after")
    def exactly_one_key_identifier(self) -> "ExternalUsageRecord":
        if (self.api_key is None) == (self.api_key_hash is None):
            raise ValueError("exactly one of api_key or api_key_hash is required")
        return self


class UsageIngestRequest(BaseModel):
    records: tuple[ExternalUsageRecord, ...] = Field(min_length=1, max_length=MAX_RECORDS_PER_REQUEST)


class UsageIngestRecordResult(BaseModel):
    request_id: str
    status: Literal["recorded", "duplicate", "error"]
    spend: float | None = None
    error: str | None = None


class UsageIngestResponse(BaseModel):
    results: tuple[UsageIngestRecordResult, ...]


class KeyAttribution(NamedTuple):
    user_id: str | None
    team_id: str | None
    organization_id: str | None


ReservationOutcome: TypeAlias = Literal["reserved", "duplicate", "disabled"]


@dataclass(frozen=True, slots=True)
class UsageIngestionDeps:
    lookup_key: Callable[[str], Awaitable[KeyAttribution | None]]
    reserve_spend_log: Callable[[ExternalUsageRecord, str, str, KeyAttribution, float], Awaitable[ReservationOutcome]]
    record_spend: Callable[..., Awaitable[None]]
    compute_cost: Callable[[litellm.ModelResponse, str], float]
    generate_request_id: Callable[[], str]


def _build_usage_kwargs(record: ExternalUsageRecord, hashed_token: str) -> Mapping[str, object]:
    tags: Final = list(record.tags) if record.tags else []  # mutable-ok: real list required by json serializer
    metadata: Final = MappingProxyType(
        {
            "user_api_key": hashed_token,
            "user_api_key_end_user_id": record.end_user_id,
            "tags": tags,
        }
    )
    return MappingProxyType(
        {
            "model": record.model,
            "call_type": "ingest_external_usage",
            "litellm_params": MappingProxyType({"model": record.model, "metadata": metadata}),
        }
    )


def _build_completion_response(record: ExternalUsageRecord, request_id: str) -> litellm.ModelResponse:
    total_tokens: Final = record.prompt_tokens + record.completion_tokens
    usage: Final = litellm.Usage(
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=total_tokens,
    )
    return litellm.ModelResponse(
        id=request_id,
        model=record.model,
        created=int(record.start_time.timestamp()),
        usage=usage,
    )


def build_spend_log_payload(
    record: ExternalUsageRecord,
    request_id: str,
    hashed_token: str,
    key: KeyAttribution,
    cost: float,
) -> SpendLogsPayload:
    payload: Final[SpendLogsPayload] = get_logging_payload(
        kwargs=_build_usage_kwargs(record, hashed_token),
        response_obj=_build_completion_response(record, request_id),
        start_time=record.start_time,
        end_time=record.end_time or record.start_time,
    )
    payload["spend"] = cost
    if isinstance(payload["startTime"], datetime):
        payload["startTime"] = payload["startTime"].isoformat()
    if isinstance(payload["endTime"], datetime):
        payload["endTime"] = payload["endTime"].isoformat()
    if key.organization_id is not None and key.organization_id != "":
        payload["organization_id"] = key.organization_id
    if key.team_id is not None and key.team_id != "":
        payload["team_id"] = key.team_id
    return payload


def _resolve_cost(deps: UsageIngestionDeps, record: ExternalUsageRecord, response: litellm.ModelResponse) -> float:
    if record.cost is not None:
        return record.cost
    return deps.compute_cost(response, record.model)


async def process_external_usage_record(
    record: ExternalUsageRecord, deps: UsageIngestionDeps
) -> UsageIngestRecordResult:
    request_id: Final = record.idempotency_key or deps.generate_request_id()
    hashed_token: Final = record.api_key_hash if record.api_key_hash is not None else hash_token(record.api_key or "")

    key: Final = await deps.lookup_key(hashed_token)
    if key is None:
        return UsageIngestRecordResult(request_id=request_id, status="error", error="api key not found")

    response: Final = _build_completion_response(record, request_id)

    try:
        cost: Final = _resolve_cost(deps, record, response)
    except Exception as e:  # noqa: BLE001  # pricing lookup raises arbitrary provider-specific errors; any failure means the model is unpriceable and the record must carry an explicit cost
        verbose_proxy_logger.info("ingest usage: cost computation failed for model %s: %s", record.model, e)
        return UsageIngestRecordResult(
            request_id=request_id,
            status="error",
            error="could not compute cost for this model, pass an explicit cost",
        )

    if record.idempotency_key is not None:
        try:
            reservation: Final = await deps.reserve_spend_log(record, request_id, hashed_token, key, cost)
        except Exception as e:  # noqa: BLE001  # booking raises arbitrary persistence errors; an aborted transaction means nothing was booked, so telling the caller to retry is safe
            verbose_proxy_logger.info("ingest usage: transactional booking failed for %s: %s", request_id, e)
            return UsageIngestRecordResult(
                request_id=request_id,
                status="error",
                error="booking failed transactionally, nothing was recorded, safe to retry",
            )
        if reservation == "duplicate":
            return UsageIngestRecordResult(request_id=request_id, status="duplicate")
        if reservation == "disabled":
            return UsageIngestRecordResult(
                request_id=request_id,
                status="error",
                error="spend updates are disabled on this proxy, nothing was recorded",
            )
        return UsageIngestRecordResult(request_id=request_id, status="recorded", spend=cost)

    await deps.record_spend(
        token=hashed_token,
        user_id=key.user_id,
        end_user_id=record.end_user_id,
        team_id=key.team_id,
        kwargs=_build_usage_kwargs(record, hashed_token),
        completion_response=response,
        start_time=record.start_time,
        end_time=record.end_time or record.start_time,
        response_cost=cost,
        org_id=key.organization_id,
    )
    return UsageIngestRecordResult(request_id=request_id, status="recorded", spend=cost)


def _attribution_of(key_row: object) -> KeyAttribution:
    return KeyAttribution(
        user_id=getattr(key_row, "user_id", None),
        team_id=getattr(key_row, "team_id", None),
        organization_id=getattr(key_row, "organization_id", None),
    )


if TYPE_CHECKING:
    from prisma.client import TransactionManager


class _TransactionClientShim:
    def __init__(self, tx: "TransactionManager") -> None:
        self.db: Final = tx


async def reserve_spend_log_atomic(
    record: ExternalUsageRecord,
    request_id: str,
    hashed_token: str,
    key: KeyAttribution,
    cost: float,
) -> ReservationOutcome:
    from litellm.proxy.proxy_server import litellm_proxy_budget_name, prisma_client, proxy_logging_obj
    from litellm.proxy.utils import PrismaClient, ProxyUpdateSpend
    from litellm.repositories.table_repositories import SpendLogsRepository

    if ProxyUpdateSpend.disable_spend_updates() is True:
        return "disabled"

    spend_payload: Final = build_spend_log_payload(record, request_id, hashed_token, key, cost)
    payload: Final = prisma_client.jsonify_object(spend_payload)
    request_tags: Final = spend_payload.get("request_tags")
    from prisma.errors import UniqueViolationError

    writer: Final = proxy_logging_obj.db_spend_update_writer

    try:
        async with prisma_client.tx() as tx:
            shim: Final = cast(PrismaClient, _TransactionClientShim(tx))  # cast-ok: helper uses only .db (untyped)
            await SpendLogsRepository(shim).table.create(data=payload)
            await writer._update_key_db(response_cost=cost, hashed_token=hashed_token, prisma_client=shim)
            await writer._update_user_db(
                response_cost=cost,
                user_id=key.user_id,
                prisma_client=shim,
                litellm_proxy_budget_name=litellm_proxy_budget_name,
                end_user_id=record.end_user_id,
            )
            await writer._update_team_db(
                response_cost=cost, team_id=key.team_id, user_id=key.user_id, prisma_client=shim
            )
            await writer._update_org_db(response_cost=cost, org_id=key.organization_id, prisma_client=shim)
            await writer._update_tag_db(response_cost=cost, request_tags=request_tags, prisma_client=shim)
    except UniqueViolationError:
        return "duplicate"
    return "reserved"


def default_ingestion_deps() -> UsageIngestionDeps:
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    if prisma_client is None:
        raise ProxyException(
            message="Prisma Client is not initialized",
            type="internal_error",
            param="None",
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    async def lookup_key(hashed_token: str) -> KeyAttribution | None:
        from litellm.proxy.auth.auth_checks import get_key_object

        try:
            key_row: Final = await get_key_object(
                hashed_token=hashed_token,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
            )
        except ProxyException:
            return None
        return _attribution_of(key_row)

    return UsageIngestionDeps(
        lookup_key=lookup_key,
        reserve_spend_log=reserve_spend_log_atomic,
        record_spend=proxy_logging_obj.db_spend_update_writer.update_database,
        compute_cost=lambda resp, model: litellm.completion_cost(completion_response=resp, model=model),
        generate_request_id=lambda: str(uuid.uuid4()),
    )


@router.post(
    "/spend/usage",
    tags=["Budget & Spend Tracking"],  # mutable-ok: fastapi decorator contract takes a list
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: fastapi decorator contract takes a list
    response_model=UsageIngestResponse,
)
async def ingest_external_usage(
    request: UsageIngestRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UsageIngestResponse:
    """
    PROXY_ADMIN ONLY: record externally measured usage into the same spend pipeline as proxy-routed traffic.

    For inference traffic that legitimately bypasses the proxy (for example async batch processors
    dispatching directly to model gateways), so budgets and spend stay coherent in litellm as the
    single metering system.

    Attribution (user/team/org) is derived from the given virtual key, submitted either raw
    (api_key) or pre-hashed (api_key_hash) to keep raw keys out of request bodies. Records accept an optional
    idempotency_key, stored as the spend-log request_id: the reservation insert, counter updates and
    dedup are checked atomically at the database primary key, so overlapping retries are safe. The
    reservation row is always written (even when disable_spend_logs is set), because it is both the
    dedup anchor and the audit record for the booked usage. When cost is omitted it is computed from
    litellm pricing; records whose model cannot be priced are rejected with an error instead of
    being booked as zero spend.
    """
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only proxy admins ingest spend records here. Use a key with the proxy_admin role.",
        )

    deps: Final = default_ingestion_deps()
    results: Final = tuple(
        await asyncio.gather(*(process_external_usage_record(record, deps) for record in request.records))
    )
    return UsageIngestResponse(results=results)
