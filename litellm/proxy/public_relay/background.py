from __future__ import annotations

from dataclasses import dataclass
from typing import cast  # noqa: TID251, RUF100  # Prisma JSON values require runtime boundary narrowing.

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.proxy.public_relay.money import UsageQuantity
from litellm.proxy.public_relay.repository import (
    database_handle,
    delete_expired_request_content,
    delete_expired_request_metadata,
    release_request,
    settle_request,
)
from litellm.proxy.utils import PrismaClient


class ExpiredReservationRow(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    request_id: str
    prompt_tokens: int | None
    completion_tokens: int | None
    spend: float | None
    metadata: object
    call_type: str | None


@dataclass(frozen=True, slots=True)
class PublicRelayBackgroundJobs:
    prisma_client: PrismaClient
    pod_lock_manager: PodLockManager

    async def reconcile(self) -> None:
        lock_name = "public-relay-reconcile"
        if await self.pod_lock_manager.acquire_lock(lock_name, ttl=240) is not True:
            return
        try:
            rows = await database_handle(self.prisma_client).query_raw(
                """
                SELECT
                    r."request_id",
                    s."prompt_tokens",
                    s."completion_tokens",
                    s."spend",
                    s."metadata",
                    s."call_type"
                FROM "LiteLLM_PublicRelayReservation" r
                LEFT JOIN "LiteLLM_SpendLogs" s ON s."request_id" = r."request_id"
                WHERE r."status" = 'OPEN' AND r."expires_at" < CURRENT_TIMESTAMP
                ORDER BY r."expires_at"
                LIMIT 500
                """
            )
            reservations = TypeAdapter(list[ExpiredReservationRow]).validate_python(rows)
            for reservation in reservations:
                await self._reconcile_one(reservation)
            await self._check_wallet_invariants()
        finally:
            await self.pod_lock_manager.release_lock(lock_name)

    async def cleanup_retained_data(self) -> None:
        lock_name = "public-relay-retention-cleanup"
        if await self.pod_lock_manager.acquire_lock(lock_name, ttl=240) is not True:
            return
        try:
            await delete_expired_request_content(self.prisma_client)
            from litellm.proxy.public_relay.config import PublicRelaySettings

            value = PublicRelaySettings.from_env()
            await delete_expired_request_metadata(self.prisma_client, value.metadata_retention_days)
        finally:
            await self.pod_lock_manager.release_lock(lock_name)

    async def _reconcile_one(self, reservation: ExpiredReservationRow) -> None:
        if reservation.prompt_tokens is None:
            await release_request(self.prisma_client, reservation.request_id)
            verbose_proxy_logger.warning(
                "Released expired public relay reservation without a SpendLog: %s",
                reservation.request_id,
            )
            return
        await settle_request(
            self.prisma_client,
            reservation.request_id,
            UsageQuantity(
                input_tokens=reservation.prompt_tokens,
                cached_input_tokens=_cached_tokens(reservation.metadata),
                output_tokens=reservation.completion_tokens or 0,
                embedding=reservation.call_type in {"embedding", "aembedding"},
            ),
            round(max(reservation.spend or 0.0, 0.0) * 1_000_000),
        )

    async def _check_wallet_invariants(self) -> None:
        rows = await database_handle(self.prisma_client).query_raw(
            """
            SELECT w."wallet_id"
            FROM "LiteLLM_PublicRelayWallet" w
            LEFT JOIN (
                SELECT "wallet_id", COALESCE(SUM("reserved_micros"), 0) AS "request_reserved"
                FROM "LiteLLM_PublicRelayReservation"
                WHERE "status" = 'OPEN'
                GROUP BY "wallet_id"
            ) r ON r."wallet_id" = w."wallet_id"
            LEFT JOIN (
                SELECT "wallet_id", COALESCE(SUM("amount_micros"), 0) AS "refund_reserved"
                FROM "LiteLLM_PublicRelayRefund"
                WHERE "status" = 'PENDING'
                GROUP BY "wallet_id"
            ) f ON f."wallet_id" = w."wallet_id"
            WHERE w."reserved_micros" != COALESCE(r."request_reserved", 0) + COALESCE(f."refund_reserved", 0)
            LIMIT 20
            """
        )
        if rows:
            verbose_proxy_logger.error("Public relay wallet reservation invariant failed: %s", rows)


def _cached_tokens(metadata: object) -> int:
    if not isinstance(metadata, dict):
        return 0
    typed_metadata = cast(dict[str, object], metadata)  # cast-ok: isinstance validates the JSON object boundary.
    additional = typed_metadata.get("additional_usage_values")
    if not isinstance(additional, dict):
        return 0
    value = cast(dict[str, object], additional).get(  # cast-ok: isinstance validates the JSON object boundary.
        "cache_read_input_tokens"
    )
    return value if isinstance(value, int) else 0
