from __future__ import annotations

import json
import secrets
import uuid
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol, TypeVar, cast  # noqa: TID251, RUF100  # Prisma exposes dynamic JSON types.

from pydantic import BaseModel, TypeAdapter

from litellm.proxy._types import hash_token
from litellm.proxy.auth.auth_utils import abbreviate_api_key
from litellm.proxy.public_relay.api_types import ModelPriceCreateRequest
from litellm.proxy.public_relay.db_types import (
    AccountRow,
    AdminAccountRow,
    AdminPaymentRow,
    KeyRow,
    LedgerRow,
    MarginSummaryRow,
    PaymentRow,
    PriceRow,
    RefundRow,
    RequestLogRow,
    ReservationRow,
    ReservationSettlementRow,
    UsageSummaryRow,
    WalletRow,
)
from litellm.proxy.public_relay.money import (
    PriceQuote,
    UsageQuantity,
    calculate_usage_charge,
    maximum_refund_micros,
)
from litellm.proxy.utils import PrismaClient

PUBLIC_ACCESS_GROUP_NAME = "public-relay-models"
PUBLIC_ALLOWED_ROUTES = (
    "/models",
    "/v1/models",
    "/chat/completions",
    "/v1/chat/completions",
    "/responses",
    "/v1/responses",
    "/embeddings",
    "/v1/embeddings",
)

DatabaseRow = TypeVar("DatabaseRow", bound=BaseModel)


class TransactionProtocol(Protocol):
    async def query_raw(self, query: str, *args: object) -> object: ...

    async def execute_raw(self, query: str, *args: object) -> int: ...


class DatabaseProtocol(TransactionProtocol, Protocol):
    def tx(self) -> AbstractAsyncContextManager[TransactionProtocol]: ...


@dataclass(frozen=True, slots=True)
class CreatedAccount:
    account: AccountRow
    raw_key: str
    key_id: str


@dataclass(frozen=True, slots=True)
class CreatedKey:
    raw_key: str
    key_id: str


@dataclass(frozen=True, slots=True)
class ReservationResult:
    reservation: ReservationRow
    price: PriceRow


@dataclass(frozen=True, slots=True)
class CheckoutOrder:
    payment_id: str
    account_id: str
    wallet_id: str
    amount_micros: int
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class RefundOperation:
    refund: RefundRow
    payment_intent_id: str


async def get_account_by_email(prisma_client: PrismaClient, normalized_email: str) -> AccountRow | None:
    rows = await _database(prisma_client).query_raw(
        """
        SELECT a.*, u."password"
        FROM "LiteLLM_PublicRelayAccount" a
        JOIN "LiteLLM_UserTable" u ON u."user_id" = a."user_id"
        WHERE a."normalized_email" = $1
        LIMIT 1
        """,
        normalized_email,
    )
    return _first(rows, AccountRow)


async def get_account_by_id(prisma_client: PrismaClient, account_id: str) -> AccountRow | None:
    rows = await _database(prisma_client).query_raw(
        """
        SELECT a.*, u."password"
        FROM "LiteLLM_PublicRelayAccount" a
        JOIN "LiteLLM_UserTable" u ON u."user_id" = a."user_id"
        WHERE a."account_id" = $1
        LIMIT 1
        """,
        account_id,
    )
    return _first(rows, AccountRow)


async def create_account(
    prisma_client: PrismaClient,
    normalized_email: str,
    password_hash: str,
) -> CreatedAccount:
    account_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    wallet_id = str(uuid.uuid4())
    raw_key = f"sk-{secrets.token_urlsafe(32)}"
    key_id = hash_token(raw_key)
    access_group_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    metadata = json.dumps(
        {
            "public_relay_account_id": account_id,
            "public_relay": True,
            "public_relay_log_content": True,
        },
        separators=(",", ":"),
    )
    async with _database(prisma_client).tx() as tx:
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_UserTable"
                ("user_id", "user_email", "password", "user_role", "models", "metadata")
            VALUES ($1, $2, $3, 'internal_user', ARRAY['no-default-models']::TEXT[], '{}'::JSONB)
            """,
            user_id,
            normalized_email,
            password_hash,
        )
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayAccount"
                ("account_id", "user_id", "normalized_email", "email_verified_at", "created_at", "updated_at")
            VALUES ($1, $2, $3, $4, $4, $4)
            """,
            account_id,
            user_id,
            normalized_email,
            now,
        )
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayWallet" ("wallet_id", "account_id", "created_at", "updated_at")
            VALUES ($1, $2, $3, $3)
            """,
            wallet_id,
            account_id,
            now,
        )
        access_group_rows = await tx.query_raw(
            """
            INSERT INTO "LiteLLM_AccessGroupTable"
                ("access_group_id", "access_group_name", "description", "created_at", "updated_at")
            VALUES ($1, $2, $3, $4, $4)
            ON CONFLICT ("access_group_name")
            DO UPDATE SET "updated_at" = EXCLUDED."updated_at"
            RETURNING "access_group_id"
            """,
            access_group_id,
            PUBLIC_ACCESS_GROUP_NAME,
            "Models with an active public relay price",
            now,
        )
        resolved_access_group_id = _single_string(access_group_rows, "access_group_id")
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_VerificationToken"
                (
                    "token",
                    "key_name",
                    "key_alias",
                    "models",
                    "user_id",
                    "metadata",
                    "allowed_routes",
                    "access_group_ids",
                    "created_by",
                    "updated_by",
                    "created_at",
                    "updated_at"
                )
            VALUES (
                $1,
                $2,
                'Default',
                ARRAY['no-default-models']::TEXT[],
                $3,
                $4::JSONB,
                $5::TEXT[],
                ARRAY[$6]::TEXT[],
                $3,
                $3,
                $7,
                $7
            )
            """,
            key_id,
            abbreviate_api_key(raw_key),
            user_id,
            metadata,
            list(PUBLIC_ALLOWED_ROUTES),
            resolved_access_group_id,
            now,
        )
    account = AccountRow(
        account_id=account_id,
        user_id=user_id,
        normalized_email=normalized_email,
        status="ACTIVE",
        password=password_hash,
        session_version=0,
        created_at=now,
    )
    return CreatedAccount(account=account, raw_key=raw_key, key_id=key_id)


async def update_password(prisma_client: PrismaClient, account: AccountRow, password_hash: str) -> None:
    async with _database(prisma_client).tx() as tx:
        await tx.execute_raw(
            'UPDATE "LiteLLM_UserTable" SET "password" = $1 WHERE "user_id" = $2',
            password_hash,
            account.user_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayAccount"
            SET "session_version" = "session_version" + 1, "updated_at" = CURRENT_TIMESTAMP
            WHERE "account_id" = $1
            """,
            account.account_id,
        )


async def get_wallet(prisma_client: PrismaClient, account_id: str) -> WalletRow | None:
    rows = await _database(prisma_client).query_raw(
        'SELECT * FROM "LiteLLM_PublicRelayWallet" WHERE "account_id" = $1 LIMIT 1',
        account_id,
    )
    return _first(rows, WalletRow)


async def list_ledger(
    prisma_client: PrismaClient,
    account_id: str,
    cursor: datetime | None,
    limit: int,
) -> tuple[LedgerRow, ...]:
    rows = await _database(prisma_client).query_raw(
        """
        SELECT l.*
        FROM "LiteLLM_PublicRelayLedgerEntry" l
        JOIN "LiteLLM_PublicRelayWallet" w ON w."wallet_id" = l."wallet_id"
        WHERE w."account_id" = $1
          AND ($2::TIMESTAMP IS NULL OR l."created_at" < $2)
        ORDER BY l."created_at" DESC, l."entry_id" DESC
        LIMIT $3
        """,
        account_id,
        cursor,
        limit,
    )
    return tuple(TypeAdapter(list[LedgerRow]).validate_python(rows))


async def publish_price(
    prisma_client: PrismaClient,
    request: ModelPriceCreateRequest,
    created_by: str,
) -> PriceRow:
    price_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    async with _database(prisma_client).tx() as tx:
        version_rows = await tx.query_raw(
            """
            SELECT COALESCE(MAX("version"), 0) + 1 AS "version"
            FROM "LiteLLM_PublicRelayModelPrice"
            WHERE "model_name" = $1
            """,
            request.model_name,
        )
        version = _single_int(version_rows, "version")
        price_rows = await tx.query_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayModelPrice"
                (
                    "price_id",
                    "model_name",
                    "version",
                    "input_micros_per_million",
                    "cached_input_micros_per_million",
                    "output_micros_per_million",
                    "embedding_micros_per_million",
                    "default_max_output_tokens",
                    "max_output_tokens",
                    "enabled",
                    "effective_at",
                    "created_at",
                    "created_by"
                )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $11, $12)
            RETURNING *
            """,
            price_id,
            request.model_name,
            version,
            request.input_micros_per_million,
            request.cached_input_micros_per_million,
            request.output_micros_per_million,
            request.embedding_micros_per_million,
            request.default_max_output_tokens,
            request.max_output_tokens,
            request.enabled,
            now,
            created_by,
        )
        await _sync_public_access_group(tx)
    return TypeAdapter(list[PriceRow]).validate_python(price_rows)[0]


async def list_active_prices(prisma_client: PrismaClient) -> tuple[PriceRow, ...]:
    rows = await _database(prisma_client).query_raw(
        """
        SELECT DISTINCT ON ("model_name") *
        FROM "LiteLLM_PublicRelayModelPrice"
        WHERE "effective_at" <= CURRENT_TIMESTAMP
        ORDER BY "model_name", "version" DESC
        """
    )
    prices = TypeAdapter(list[PriceRow]).validate_python(rows)
    return tuple(price for price in prices if price.enabled)


async def get_active_price(prisma_client: PrismaClient, model_name: str) -> PriceRow | None:
    rows = await _database(prisma_client).query_raw(
        """
        SELECT *
        FROM "LiteLLM_PublicRelayModelPrice"
        WHERE "model_name" = $1 AND "effective_at" <= CURRENT_TIMESTAMP
        ORDER BY "version" DESC
        LIMIT 1
        """,
        model_name,
    )
    price = _first(rows, PriceRow)
    return price if price is not None and price.enabled else None


async def list_api_keys(prisma_client: PrismaClient, account: AccountRow) -> tuple[KeyRow, ...]:
    rows = await _database(prisma_client).query_raw(
        """
        SELECT "token", "key_alias", "metadata", "created_at"
        FROM "LiteLLM_VerificationToken"
        WHERE "user_id" = $1
          AND COALESCE(("metadata"->>'public_relay')::BOOLEAN, false) = true
        ORDER BY "created_at" DESC
        """,
        account.user_id,
    )
    return tuple(TypeAdapter(list[KeyRow]).validate_python(rows))


async def create_api_key(
    prisma_client: PrismaClient,
    account: AccountRow,
    alias: str,
    log_content: bool,
    max_keys: int,
) -> CreatedKey:
    raw_key = f"sk-{secrets.token_urlsafe(32)}"
    key_id = hash_token(raw_key)
    now = datetime.now(timezone.utc)
    metadata = json.dumps(
        {
            "public_relay_account_id": account.account_id,
            "public_relay": True,
            "public_relay_log_content": log_content,
        },
        separators=(",", ":"),
    )
    async with _database(prisma_client).tx() as tx:
        await tx.execute_raw("SELECT pg_advisory_xact_lock(hashtext($1))", f"public-relay-keys:{account.account_id}")
        count_rows = await tx.query_raw(
            """
            SELECT COUNT(*)::INTEGER AS "count"
            FROM "LiteLLM_VerificationToken"
            WHERE "user_id" = $1
              AND COALESCE(("metadata"->>'public_relay')::BOOLEAN, false) = true
            """,
            account.user_id,
        )
        if _single_int(count_rows, "count") >= max_keys:
            raise PermissionError("API key limit reached")
        group_rows = await tx.query_raw(
            'SELECT "access_group_id" FROM "LiteLLM_AccessGroupTable" WHERE "access_group_name" = $1',
            PUBLIC_ACCESS_GROUP_NAME,
        )
        group_id = _single_string(group_rows, "access_group_id")
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_VerificationToken"
                (
                    "token",
                    "key_name",
                    "key_alias",
                    "models",
                    "user_id",
                    "metadata",
                    "allowed_routes",
                    "access_group_ids",
                    "created_by",
                    "updated_by",
                    "created_at",
                    "updated_at"
                )
            VALUES (
                $1,
                $2,
                $3,
                ARRAY['no-default-models']::TEXT[],
                $4,
                $5::JSONB,
                $6::TEXT[],
                ARRAY[$7]::TEXT[],
                $4,
                $4,
                $8,
                $8
            )
            """,
            key_id,
            abbreviate_api_key(raw_key),
            alias,
            account.user_id,
            metadata,
            list(PUBLIC_ALLOWED_ROUTES),
            group_id,
            now,
        )
    return CreatedKey(raw_key=raw_key, key_id=key_id)


async def delete_api_key(prisma_client: PrismaClient, account: AccountRow, key_id: str) -> bool:
    deleted = await _database(prisma_client).execute_raw(
        """
        DELETE FROM "LiteLLM_VerificationToken"
        WHERE "token" = $1
          AND "user_id" = $2
          AND COALESCE(("metadata"->>'public_relay')::BOOLEAN, false) = true
        """,
        key_id,
        account.user_id,
    )
    return int(deleted) == 1


async def reserve_request(
    prisma_client: PrismaClient,
    account_id: str,
    request_id: str,
    model_name: str,
    input_tokens: int,
    max_output_tokens: int,
    embedding: bool,
    reservation_ttl_seconds: int,
    price: PriceRow | None = None,
) -> ReservationResult:
    selected_price = price or await get_active_price(prisma_client, model_name)
    if selected_price is None or selected_price.model_name != model_name:
        raise LookupError("model is not available on the public relay")
    resolved_max_output = 0 if embedding else max_output_tokens
    quote = _price_quote(selected_price)
    reserved_micros = calculate_usage_charge(
        quote,
        UsageQuantity(
            input_tokens=input_tokens,
            output_tokens=resolved_max_output,
            embedding=embedding,
        ),
    )
    reservation_id = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=reservation_ttl_seconds)
    async with _database(prisma_client).tx() as tx:
        existing_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayReservation" WHERE "request_id" = $1',
            request_id,
        )
        existing = _first(existing_rows, ReservationRow)
        if existing is not None:
            return ReservationResult(reservation=existing, price=selected_price)
        wallet_rows = await tx.query_raw(
            """
            SELECT w.*
            FROM "LiteLLM_PublicRelayWallet" w
            JOIN "LiteLLM_PublicRelayAccount" a ON a."account_id" = w."account_id"
            WHERE w."account_id" = $1 AND a."status" = 'ACTIVE'
            FOR UPDATE OF w
            """,
            account_id,
        )
        wallet = _first(wallet_rows, WalletRow)
        if wallet is None:
            raise PermissionError("public relay account is not active")
        if wallet.debt_micros > 0 or wallet.available_micros < reserved_micros:
            raise ArithmeticError("insufficient balance")
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayWallet"
            SET
                "available_micros" = "available_micros" - $1,
                "reserved_micros" = "reserved_micros" + $1,
                "version" = "version" + 1,
                "updated_at" = CURRENT_TIMESTAMP
            WHERE "wallet_id" = $2
            """,
            reserved_micros,
            wallet.wallet_id,
        )
        reservation_rows = await tx.query_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayReservation"
                (
                    "reservation_id",
                    "request_id",
                    "account_id",
                    "wallet_id",
                    "price_id",
                    "reserved_micros",
                    "input_tokens",
                    "max_output_tokens",
                    "expires_at"
                )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING *
            """,
            reservation_id,
            request_id,
            account_id,
            wallet.wallet_id,
            selected_price.price_id,
            reserved_micros,
            input_tokens,
            resolved_max_output,
            expires_at,
        )
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayLedgerEntry"
                (
                    "entry_id",
                    "wallet_id",
                    "entry_type",
                    "amount_micros",
                    "available_after_micros",
                    "reserved_after_micros",
                    "debt_after_micros",
                    "idempotency_key",
                    "request_id",
                    "metadata"
                )
            VALUES ($1, $2, 'RESERVE', 0, $3, $4, $5, $6, $7, $8::JSONB)
            """,
            str(uuid.uuid4()),
            wallet.wallet_id,
            wallet.available_micros - reserved_micros,
            wallet.reserved_micros + reserved_micros,
            wallet.debt_micros,
            f"reserve:{request_id}",
            request_id,
            json.dumps({"reserved_micros": reserved_micros}, separators=(",", ":")),
        )
    reservation = TypeAdapter(list[ReservationRow]).validate_python(reservation_rows)[0]
    return ReservationResult(reservation=reservation, price=selected_price)


async def settle_request(
    prisma_client: PrismaClient,
    request_id: str,
    usage: UsageQuantity,
    upstream_cost_micros: int,
) -> int:
    async with _database(prisma_client).tx() as tx:
        rows = await tx.query_raw(
            """
            SELECT
                r.*,
                p."input_micros_per_million",
                p."cached_input_micros_per_million",
                p."output_micros_per_million",
                p."embedding_micros_per_million"
            FROM "LiteLLM_PublicRelayReservation" r
            JOIN "LiteLLM_PublicRelayModelPrice" p ON p."price_id" = r."price_id"
            WHERE r."request_id" = $1
            FOR UPDATE OF r
            """,
            request_id,
        )
        reservation = _first(rows, ReservationSettlementRow)
        if reservation is None:
            return 0
        if reservation.status != "OPEN":
            existing_rows = await tx.query_raw(
                'SELECT "charged_micros" FROM "LiteLLM_PublicRelayRequestCharge" WHERE "request_id" = $1',
                request_id,
            )
            return _single_int(existing_rows, "charged_micros", default=0)
        calculated = calculate_usage_charge(
            PriceQuote(
                input_micros_per_million=reservation.input_micros_per_million,
                cached_input_micros_per_million=reservation.cached_input_micros_per_million,
                output_micros_per_million=reservation.output_micros_per_million,
                embedding_micros_per_million=reservation.embedding_micros_per_million,
            ),
            usage,
        )
        charged_micros = min(calculated, reservation.reserved_micros)
        wallet_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayWallet" WHERE "wallet_id" = $1 FOR UPDATE',
            reservation.wallet_id,
        )
        wallet = _required_first(wallet_rows, WalletRow)
        released_micros = reservation.reserved_micros - charged_micros
        available_after = wallet.available_micros + released_micros
        reserved_after = wallet.reserved_micros - reservation.reserved_micros
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayWallet"
            SET
                "available_micros" = $1,
                "reserved_micros" = $2,
                "version" = "version" + 1,
                "updated_at" = CURRENT_TIMESTAMP
            WHERE "wallet_id" = $3
            """,
            available_after,
            reserved_after,
            wallet.wallet_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayReservation"
            SET "status" = 'FINALIZED', "updated_at" = CURRENT_TIMESTAMP
            WHERE "reservation_id" = $1
            """,
            reservation.reservation_id,
        )
        if released_micros > 0:
            await tx.execute_raw(
                """
                INSERT INTO "LiteLLM_PublicRelayLedgerEntry"
                    (
                        "entry_id",
                        "wallet_id",
                        "entry_type",
                        "amount_micros",
                        "available_after_micros",
                        "reserved_after_micros",
                        "debt_after_micros",
                        "idempotency_key",
                        "request_id",
                        "metadata"
                    )
                VALUES ($1, $2, 'RELEASE', 0, $3, $4, $5, $6, $7, $8::JSONB)
                """,
                str(uuid.uuid4()),
                wallet.wallet_id,
                available_after,
                reserved_after,
                wallet.debt_micros,
                f"release:{request_id}",
                request_id,
                json.dumps({"released_micros": released_micros}, separators=(",", ":")),
            )
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayRequestCharge"
                (
                    "charge_id",
                    "request_id",
                    "account_id",
                    "price_id",
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "charged_micros",
                    "upstream_cost_micros"
                )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            str(uuid.uuid4()),
            request_id,
            reservation.account_id,
            reservation.price_id,
            usage.input_tokens,
            usage.cached_input_tokens,
            usage.output_tokens,
            charged_micros,
            upstream_cost_micros,
        )
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayLedgerEntry"
                (
                    "entry_id",
                    "wallet_id",
                    "entry_type",
                    "amount_micros",
                    "available_after_micros",
                    "reserved_after_micros",
                    "debt_after_micros",
                    "idempotency_key",
                    "request_id"
                )
            VALUES ($1, $2, 'USAGE', $3, $4, $5, $6, $7, $8)
            """,
            str(uuid.uuid4()),
            wallet.wallet_id,
            -charged_micros,
            available_after,
            reserved_after,
            wallet.debt_micros,
            f"usage:{request_id}",
            request_id,
        )
    return charged_micros


async def release_request(prisma_client: PrismaClient, request_id: str) -> None:
    async with _database(prisma_client).tx() as tx:
        rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayReservation" WHERE "request_id" = $1 FOR UPDATE',
            request_id,
        )
        reservation = _first(rows, ReservationRow)
        if reservation is None or reservation.status != "OPEN":
            return
        wallet_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayWallet" WHERE "wallet_id" = $1 FOR UPDATE',
            reservation.wallet_id,
        )
        wallet = _required_first(wallet_rows, WalletRow)
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayWallet"
            SET
                "available_micros" = "available_micros" + $1,
                "reserved_micros" = "reserved_micros" - $1,
                "version" = "version" + 1,
                "updated_at" = CURRENT_TIMESTAMP
            WHERE "wallet_id" = $2
            """,
            reservation.reserved_micros,
            wallet.wallet_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayReservation"
            SET "status" = 'RELEASED', "updated_at" = CURRENT_TIMESTAMP
            WHERE "reservation_id" = $1
            """,
            reservation.reservation_id,
        )
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayLedgerEntry"
                (
                    "entry_id",
                    "wallet_id",
                    "entry_type",
                    "amount_micros",
                    "available_after_micros",
                    "reserved_after_micros",
                    "debt_after_micros",
                    "idempotency_key",
                    "request_id",
                    "metadata"
                )
            VALUES ($1, $2, 'RELEASE', 0, $3, $4, $5, $6, $7, $8::JSONB)
            """,
            str(uuid.uuid4()),
            wallet.wallet_id,
            wallet.available_micros + reservation.reserved_micros,
            wallet.reserved_micros - reservation.reserved_micros,
            wallet.debt_micros,
            f"release:{request_id}",
            request_id,
            json.dumps({"released_micros": reservation.reserved_micros}, separators=(",", ":")),
        )


async def create_checkout_order(
    prisma_client: PrismaClient,
    account_id: str,
    amount_micros: int,
) -> CheckoutOrder:
    wallet = await get_wallet(prisma_client, account_id)
    if wallet is None:
        raise LookupError("wallet not found")
    payment_id = str(uuid.uuid4())
    idempotency_key = f"checkout:{payment_id}"
    await _database(prisma_client).execute_raw(
        """
        INSERT INTO "LiteLLM_PublicRelayPayment"
            ("payment_id", "account_id", "wallet_id", "amount_micros", "idempotency_key")
        VALUES ($1, $2, $3, $4, $5)
        """,
        payment_id,
        account_id,
        wallet.wallet_id,
        amount_micros,
        idempotency_key,
    )
    return CheckoutOrder(
        payment_id=payment_id,
        account_id=account_id,
        wallet_id=wallet.wallet_id,
        amount_micros=amount_micros,
        idempotency_key=idempotency_key,
    )


async def attach_checkout_session(prisma_client: PrismaClient, payment_id: str, session_id: str) -> None:
    await _database(prisma_client).execute_raw(
        """
        UPDATE "LiteLLM_PublicRelayPayment"
        SET "stripe_checkout_session" = $1, "updated_at" = CURRENT_TIMESTAMP
        WHERE "payment_id" = $2 AND "status" = 'PENDING'
        """,
        session_id,
        payment_id,
    )


async def fail_checkout_creation(prisma_client: PrismaClient, payment_id: str) -> None:
    await _database(prisma_client).execute_raw(
        """
        UPDATE "LiteLLM_PublicRelayPayment"
        SET "status" = 'FAILED', "updated_at" = CURRENT_TIMESTAMP
        WHERE "payment_id" = $1 AND "status" = 'PENDING'
        """,
        payment_id,
    )


async def list_payments(
    prisma_client: PrismaClient,
    account_id: str,
    cursor: datetime | None,
    limit: int,
) -> tuple[PaymentRow, ...]:
    rows = await _database(prisma_client).query_raw(
        """
        SELECT *
        FROM "LiteLLM_PublicRelayPayment"
        WHERE "account_id" = $1
          AND ($2::TIMESTAMP IS NULL OR "created_at" < $2)
        ORDER BY "created_at" DESC, "payment_id" DESC
        LIMIT $3
        """,
        account_id,
        cursor,
        limit,
    )
    return tuple(TypeAdapter(list[PaymentRow]).validate_python(rows))


async def get_usage_summary(prisma_client: PrismaClient, account_id: str) -> UsageSummaryRow:
    rows = await _database(prisma_client).query_raw(
        """
        SELECT
            COUNT(*)::INTEGER AS "request_count",
            COALESCE(SUM("input_tokens"), 0)::BIGINT AS "input_tokens",
            COALESCE(SUM("cached_input_tokens"), 0)::BIGINT AS "cached_input_tokens",
            COALESCE(SUM("output_tokens"), 0)::BIGINT AS "output_tokens",
            COALESCE(SUM("charged_micros"), 0)::BIGINT AS "charged_micros",
            COALESCE(SUM("upstream_cost_micros"), 0)::BIGINT AS "upstream_cost_micros"
        FROM "LiteLLM_PublicRelayRequestCharge"
        WHERE "account_id" = $1
        """,
        account_id,
    )
    return _required_first(rows, UsageSummaryRow)


async def list_admin_accounts(
    prisma_client: PrismaClient,
    cursor: datetime | None,
    limit: int,
) -> tuple[AdminAccountRow, ...]:
    rows = await _database(prisma_client).query_raw(
        """
        SELECT a.*, w."wallet_id", w."available_micros", w."reserved_micros", w."debt_micros"
        FROM "LiteLLM_PublicRelayAccount" a
        JOIN "LiteLLM_PublicRelayWallet" w ON w."account_id" = a."account_id"
        WHERE ($1::TIMESTAMP IS NULL OR a."created_at" < $1)
        ORDER BY a."created_at" DESC, a."account_id" DESC
        LIMIT $2
        """,
        cursor,
        limit,
    )
    return tuple(TypeAdapter(list[AdminAccountRow]).validate_python(rows))


async def list_admin_payments(
    prisma_client: PrismaClient,
    cursor: datetime | None,
    limit: int,
) -> tuple[AdminPaymentRow, ...]:
    rows = await _database(prisma_client).query_raw(
        """
        SELECT p.*, a."normalized_email"
        FROM "LiteLLM_PublicRelayPayment" p
        JOIN "LiteLLM_PublicRelayAccount" a ON a."account_id" = p."account_id"
        WHERE ($1::TIMESTAMP IS NULL OR p."created_at" < $1)
        ORDER BY p."created_at" DESC, p."payment_id" DESC
        LIMIT $2
        """,
        cursor,
        limit,
    )
    return tuple(TypeAdapter(list[AdminPaymentRow]).validate_python(rows))


async def get_margin_summary(prisma_client: PrismaClient) -> MarginSummaryRow:
    rows = await _database(prisma_client).query_raw(
        """
        SELECT
            COALESCE(SUM("charged_micros"), 0)::BIGINT AS "charged_micros",
            COALESCE(SUM("upstream_cost_micros"), 0)::BIGINT AS "upstream_cost_micros"
        FROM "LiteLLM_PublicRelayRequestCharge"
        """
    )
    return _required_first(rows, MarginSummaryRow)


async def list_request_logs(
    prisma_client: PrismaClient,
    account_id: str,
    cursor: datetime | None,
    limit: int,
) -> tuple[RequestLogRow, ...]:
    rows = await _database(prisma_client).query_raw(
        """
        SELECT
            c."request_id",
            p."model_name",
            c."input_tokens",
            c."cached_input_tokens",
            c."output_tokens",
            c."charged_micros",
            c."upstream_cost_micros",
            s."status",
            s."request_duration_ms",
            c."created_at"
        FROM "LiteLLM_PublicRelayRequestCharge" c
        JOIN "LiteLLM_PublicRelayModelPrice" p ON p."price_id" = c."price_id"
        LEFT JOIN "LiteLLM_SpendLogs" s ON s."request_id" = c."request_id"
        WHERE c."account_id" = $1
          AND c."created_at" >= CURRENT_TIMESTAMP - INTERVAL '7 days'
          AND ($2::TIMESTAMP IS NULL OR c."created_at" < $2)
        ORDER BY c."created_at" DESC, c."request_id" DESC
        LIMIT $3
        """,
        account_id,
        cursor,
        limit,
    )
    return tuple(TypeAdapter(list[RequestLogRow]).validate_python(rows))


async def store_request_content(
    prisma_client: PrismaClient,
    request_id: str,
    account_id: str,
    key_version: int,
    nonce_b64: str,
    ciphertext_b64: str,
    retention_days: int,
) -> None:
    expires_at = datetime.now(timezone.utc) + timedelta(days=retention_days)
    await _database(prisma_client).execute_raw(
        """
        INSERT INTO "LiteLLM_PublicRelayRequestContent"
            (
                "content_id",
                "request_id",
                "account_id",
                "key_version",
                "nonce_b64",
                "ciphertext_b64",
                "expires_at"
            )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT ("request_id") DO NOTHING
        """,
        str(uuid.uuid4()),
        request_id,
        account_id,
        key_version,
        nonce_b64,
        ciphertext_b64,
        expires_at,
    )


async def delete_expired_request_content(prisma_client: PrismaClient) -> int:
    deleted = await _database(prisma_client).execute_raw(
        'DELETE FROM "LiteLLM_PublicRelayRequestContent" WHERE "expires_at" < CURRENT_TIMESTAMP'
    )
    return int(deleted)


async def delete_expired_request_metadata(prisma_client: PrismaClient, retention_days: int) -> int:
    async with _database(prisma_client).tx() as tx:
        deleted_charges = await tx.execute_raw(
            """
            DELETE FROM "LiteLLM_PublicRelayRequestCharge"
            WHERE "created_at" < CURRENT_TIMESTAMP - ($1 * INTERVAL '1 day')
            """,
            retention_days,
        )
        deleted_reservations = await tx.execute_raw(
            """
            DELETE FROM "LiteLLM_PublicRelayReservation"
            WHERE "status" != 'OPEN'
              AND "updated_at" < CURRENT_TIMESTAMP - ($1 * INTERVAL '1 day')
            """,
            retention_days,
        )
    return int(deleted_charges) + int(deleted_reservations)


async def credit_checkout(
    prisma_client: PrismaClient,
    event_id: str,
    event_type: str,
    livemode: bool,
    payload: str,
    checkout_session_id: str,
    payment_intent_id: str,
    amount_micros: int,
    account_id: str,
    currency: str,
    payment_id: str,
) -> bool:
    async with _database(prisma_client).tx() as tx:
        inserted = await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayStripeEvent" ("event_id", "event_type", "livemode", "payload")
            VALUES ($1, $2, $3, $4::JSONB)
            ON CONFLICT ("event_id") DO NOTHING
            """,
            event_id,
            event_type,
            livemode,
            payload,
        )
        if int(inserted) == 0:
            return False
        payment_rows = await tx.query_raw(
            """
            SELECT *
            FROM "LiteLLM_PublicRelayPayment"
            WHERE "stripe_checkout_session" = $1
            FOR UPDATE
            """,
            checkout_session_id,
        )
        payment = _required_first(payment_rows, PaymentRow)
        if (
            payment.amount_micros != amount_micros
            or payment.account_id != account_id
            or payment.payment_id != payment_id
            or payment.status != "PENDING"
            or currency.upper() != "USD"
        ):
            raise ValueError("Stripe checkout does not match the pending payment")
        wallet_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayWallet" WHERE "wallet_id" = $1 FOR UPDATE',
            payment.wallet_id,
        )
        wallet = _required_first(wallet_rows, WalletRow)
        available_after = wallet.available_micros + amount_micros
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayWallet"
            SET
                "available_micros" = $1,
                "version" = "version" + 1,
                "updated_at" = CURRENT_TIMESTAMP
            WHERE "wallet_id" = $2
            """,
            available_after,
            wallet.wallet_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayPayment"
            SET
                "status" = 'PAID',
                "stripe_payment_intent" = $1,
                "updated_at" = CURRENT_TIMESTAMP
            WHERE "payment_id" = $2
            """,
            payment_intent_id,
            payment.payment_id,
        )
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayLedgerEntry"
                (
                    "entry_id",
                    "wallet_id",
                    "entry_type",
                    "amount_micros",
                    "available_after_micros",
                    "reserved_after_micros",
                    "debt_after_micros",
                    "idempotency_key",
                    "payment_id"
                )
            VALUES ($1, $2, 'DEPOSIT', $3, $4, $5, $6, $7, $8)
            """,
            str(uuid.uuid4()),
            wallet.wallet_id,
            amount_micros,
            available_after,
            wallet.reserved_micros,
            wallet.debt_micros,
            f"stripe-deposit:{event_id}",
            payment.payment_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayStripeEvent"
            SET "processed_at" = CURRENT_TIMESTAMP
            WHERE "event_id" = $1
            """,
            event_id,
        )
    return True


async def fail_checkout(
    prisma_client: PrismaClient,
    event_id: str,
    event_type: str,
    livemode: bool,
    payload: str,
    checkout_session_id: str,
) -> bool:
    async with _database(prisma_client).tx() as tx:
        inserted = await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayStripeEvent" ("event_id", "event_type", "livemode", "payload")
            VALUES ($1, $2, $3, $4::JSONB)
            ON CONFLICT ("event_id") DO NOTHING
            """,
            event_id,
            event_type,
            livemode,
            payload,
        )
        if int(inserted) == 0:
            return False
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayPayment"
            SET "status" = 'FAILED', "updated_at" = CURRENT_TIMESTAMP
            WHERE "stripe_checkout_session" = $1 AND "status" = 'PENDING'
            """,
            checkout_session_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayStripeEvent"
            SET "processed_at" = CURRENT_TIMESTAMP
            WHERE "event_id" = $1
            """,
            event_id,
        )
    return True


async def apply_dispute(
    prisma_client: PrismaClient,
    event_id: str,
    event_type: str,
    livemode: bool,
    payload: str,
    payment_intent_id: str,
    amount_micros: int,
) -> bool:
    async with _database(prisma_client).tx() as tx:
        inserted = await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayStripeEvent" ("event_id", "event_type", "livemode", "payload")
            VALUES ($1, $2, $3, $4::JSONB)
            ON CONFLICT ("event_id") DO NOTHING
            """,
            event_id,
            event_type,
            livemode,
            payload,
        )
        if int(inserted) == 0:
            return False
        payment_rows = await tx.query_raw(
            """
            SELECT *
            FROM "LiteLLM_PublicRelayPayment"
            WHERE "stripe_payment_intent" = $1
            FOR UPDATE
            """,
            payment_intent_id,
        )
        payment = _required_first(payment_rows, PaymentRow)
        wallet_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayWallet" WHERE "wallet_id" = $1 FOR UPDATE',
            payment.wallet_id,
        )
        wallet = _required_first(wallet_rows, WalletRow)
        recovered = min(wallet.available_micros, amount_micros)
        available_after = wallet.available_micros - recovered
        debt_after = wallet.debt_micros + amount_micros - recovered
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayWallet"
            SET
                "available_micros" = $1,
                "debt_micros" = $2,
                "version" = "version" + 1,
                "updated_at" = CURRENT_TIMESTAMP
            WHERE "wallet_id" = $3
            """,
            available_after,
            debt_after,
            wallet.wallet_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayPayment"
            SET "status" = 'DISPUTED', "updated_at" = CURRENT_TIMESTAMP
            WHERE "payment_id" = $1
            """,
            payment.payment_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayAccount"
            SET "status" = 'FROZEN', "session_version" = "session_version" + 1,
                "updated_at" = CURRENT_TIMESTAMP
            WHERE "account_id" = $1
            """,
            payment.account_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_VerificationToken"
            SET "blocked" = true, "updated_at" = CURRENT_TIMESTAMP
            WHERE "user_id" = (
                SELECT "user_id" FROM "LiteLLM_PublicRelayAccount" WHERE "account_id" = $1
            )
              AND COALESCE(("metadata"->>'public_relay')::BOOLEAN, false) = true
            """,
            payment.account_id,
        )
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayLedgerEntry"
                (
                    "entry_id",
                    "wallet_id",
                    "entry_type",
                    "amount_micros",
                    "available_after_micros",
                    "reserved_after_micros",
                    "debt_after_micros",
                    "idempotency_key",
                    "payment_id"
                )
            VALUES ($1, $2, 'CHARGEBACK', $3, $4, $5, $6, $7, $8)
            """,
            str(uuid.uuid4()),
            wallet.wallet_id,
            -amount_micros,
            available_after,
            wallet.reserved_micros,
            debt_after,
            f"chargeback:{event_id}",
            payment.payment_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayStripeEvent"
            SET "processed_at" = CURRENT_TIMESTAMP
            WHERE "event_id" = $1
            """,
            event_id,
        )
    return True


async def record_stripe_event(
    prisma_client: PrismaClient,
    event_id: str,
    event_type: str,
    livemode: bool,
    payload: str,
) -> bool:
    inserted = await _database(prisma_client).execute_raw(
        """
        INSERT INTO "LiteLLM_PublicRelayStripeEvent"
            ("event_id", "event_type", "livemode", "payload", "processed_at")
        VALUES ($1, $2, $3, $4::JSONB, CURRENT_TIMESTAMP)
        ON CONFLICT ("event_id") DO NOTHING
        """,
        event_id,
        event_type,
        livemode,
        payload,
    )
    return int(inserted) == 1


async def begin_refund(
    prisma_client: PrismaClient,
    payment_id: str,
    amount_micros: int,
    reason: str,
    idempotency_key: str,
) -> RefundOperation:
    async with _database(prisma_client).tx() as tx:
        existing_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayRefund" WHERE "idempotency_key" = $1',
            idempotency_key,
        )
        existing = _first(existing_rows, RefundRow)
        if existing is not None:
            payment_rows = await tx.query_raw(
                'SELECT * FROM "LiteLLM_PublicRelayPayment" WHERE "payment_id" = $1 FOR UPDATE',
                existing.payment_id,
            )
            payment = _required_first(payment_rows, PaymentRow)
            if payment.stripe_payment_intent is None:
                raise RuntimeError("payment intent is missing")
            if existing.status == "FAILED":
                wallet_rows = await tx.query_raw(
                    'SELECT * FROM "LiteLLM_PublicRelayWallet" WHERE "wallet_id" = $1 FOR UPDATE',
                    existing.wallet_id,
                )
                wallet = _required_first(wallet_rows, WalletRow)
                if wallet.available_micros < existing.amount_micros:
                    raise ArithmeticError("refund exceeds the refundable balance")
                await tx.execute_raw(
                    """
                    UPDATE "LiteLLM_PublicRelayWallet"
                    SET
                        "available_micros" = "available_micros" - $1,
                        "reserved_micros" = "reserved_micros" + $1,
                        "version" = "version" + 1,
                        "updated_at" = CURRENT_TIMESTAMP
                    WHERE "wallet_id" = $2
                    """,
                    existing.amount_micros,
                    existing.wallet_id,
                )
                updated_rows = await tx.query_raw(
                    """
                    UPDATE "LiteLLM_PublicRelayRefund"
                    SET "status" = 'PENDING', "error" = NULL, "updated_at" = CURRENT_TIMESTAMP
                    WHERE "refund_id" = $1
                    RETURNING *
                    """,
                    existing.refund_id,
                )
                await tx.execute_raw(
                    """
                    UPDATE "LiteLLM_PublicRelayPayment"
                    SET "status" = 'REFUND_PENDING', "updated_at" = CURRENT_TIMESTAMP
                    WHERE "payment_id" = $1
                    """,
                    existing.payment_id,
                )
                existing = _required_first(updated_rows, RefundRow)
            return RefundOperation(refund=existing, payment_intent_id=payment.stripe_payment_intent)
        payment_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayPayment" WHERE "payment_id" = $1 FOR UPDATE',
            payment_id,
        )
        payment = _required_first(payment_rows, PaymentRow)
        if payment.status not in {"PAID", "PARTIALLY_REFUNDED"}:
            raise ValueError("payment is not refundable")
        if payment.stripe_payment_intent is None:
            raise RuntimeError("payment intent is missing")
        wallet_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayWallet" WHERE "wallet_id" = $1 FOR UPDATE',
            payment.wallet_id,
        )
        wallet = _required_first(wallet_rows, WalletRow)
        maximum_refund = maximum_refund_micros(
            payment.amount_micros,
            payment.refunded_micros,
            wallet.available_micros,
        )
        if amount_micros <= 0 or amount_micros > maximum_refund:
            raise ArithmeticError("refund exceeds the refundable balance")
        refund_id = str(uuid.uuid4())
        refund_rows = await tx.query_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayRefund"
                (
                    "refund_id",
                    "payment_id",
                    "wallet_id",
                    "amount_micros",
                    "idempotency_key",
                    "reason"
                )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
            """,
            refund_id,
            payment.payment_id,
            payment.wallet_id,
            amount_micros,
            idempotency_key,
            reason,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayWallet"
            SET
                "available_micros" = "available_micros" - $1,
                "reserved_micros" = "reserved_micros" + $1,
                "version" = "version" + 1,
                "updated_at" = CURRENT_TIMESTAMP
            WHERE "wallet_id" = $2
            """,
            amount_micros,
            payment.wallet_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayPayment"
            SET "status" = 'REFUND_PENDING', "updated_at" = CURRENT_TIMESTAMP
            WHERE "payment_id" = $1
            """,
            payment.payment_id,
        )
    return RefundOperation(
        refund=_required_first(refund_rows, RefundRow),
        payment_intent_id=payment.stripe_payment_intent,
    )


async def attach_refund_submission(prisma_client: PrismaClient, refund_id: str, stripe_refund_id: str) -> RefundRow:
    rows = await _database(prisma_client).query_raw(
        """
        UPDATE "LiteLLM_PublicRelayRefund"
        SET "stripe_refund_id" = $1, "updated_at" = CURRENT_TIMESTAMP
        WHERE "refund_id" = $2 AND "status" = 'PENDING'
        RETURNING *
        """,
        stripe_refund_id,
        refund_id,
    )
    return _required_first(rows, RefundRow)


async def get_refund_by_stripe_id(prisma_client: PrismaClient, stripe_refund_id: str) -> RefundRow | None:
    rows = await _database(prisma_client).query_raw(
        'SELECT * FROM "LiteLLM_PublicRelayRefund" WHERE "stripe_refund_id" = $1',
        stripe_refund_id,
    )
    return _first(rows, RefundRow)


async def get_refund_by_id(prisma_client: PrismaClient, refund_id: str) -> RefundRow | None:
    rows = await _database(prisma_client).query_raw(
        'SELECT * FROM "LiteLLM_PublicRelayRefund" WHERE "refund_id" = $1',
        refund_id,
    )
    return _first(rows, RefundRow)


async def complete_refund(prisma_client: PrismaClient, refund_id: str, stripe_refund_id: str) -> RefundRow:
    async with _database(prisma_client).tx() as tx:
        refund_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayRefund" WHERE "refund_id" = $1 FOR UPDATE',
            refund_id,
        )
        refund = _required_first(refund_rows, RefundRow)
        if refund.status == "SUCCEEDED":
            return refund
        if refund.status != "PENDING":
            raise ValueError("refund is not pending")
        payment_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayPayment" WHERE "payment_id" = $1 FOR UPDATE',
            refund.payment_id,
        )
        payment = _required_first(payment_rows, PaymentRow)
        wallet_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayWallet" WHERE "wallet_id" = $1 FOR UPDATE',
            refund.wallet_id,
        )
        wallet = _required_first(wallet_rows, WalletRow)
        refunded_after = payment.refunded_micros + refund.amount_micros
        reserved_after = wallet.reserved_micros - refund.amount_micros
        payment_status = "REFUNDED" if refunded_after == payment.amount_micros else "PARTIALLY_REFUNDED"
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayWallet"
            SET
                "reserved_micros" = $1,
                "version" = "version" + 1,
                "updated_at" = CURRENT_TIMESTAMP
            WHERE "wallet_id" = $2
            """,
            reserved_after,
            wallet.wallet_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayPayment"
            SET "refunded_micros" = $1, "status" = $2::"PublicRelayPaymentStatus",
                "updated_at" = CURRENT_TIMESTAMP
            WHERE "payment_id" = $3
            """,
            refunded_after,
            payment_status,
            payment.payment_id,
        )
        updated_rows = await tx.query_raw(
            """
            UPDATE "LiteLLM_PublicRelayRefund"
            SET "status" = 'SUCCEEDED', "stripe_refund_id" = $1, "updated_at" = CURRENT_TIMESTAMP
            WHERE "refund_id" = $2
            RETURNING *
            """,
            stripe_refund_id,
            refund.refund_id,
        )
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayLedgerEntry"
                (
                    "entry_id",
                    "wallet_id",
                    "entry_type",
                    "amount_micros",
                    "available_after_micros",
                    "reserved_after_micros",
                    "debt_after_micros",
                    "idempotency_key",
                    "payment_id",
                    "metadata"
                )
            VALUES ($1, $2, 'REFUND', $3, $4, $5, $6, $7, $8, $9::JSONB)
            """,
            str(uuid.uuid4()),
            wallet.wallet_id,
            -refund.amount_micros,
            wallet.available_micros,
            reserved_after,
            wallet.debt_micros,
            f"refund:{refund.refund_id}",
            payment.payment_id,
            json.dumps({"stripe_refund_id": stripe_refund_id}, separators=(",", ":")),
        )
    return _required_first(updated_rows, RefundRow)


async def fail_refund(prisma_client: PrismaClient, refund_id: str, error: str) -> None:
    async with _database(prisma_client).tx() as tx:
        refund_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayRefund" WHERE "refund_id" = $1 FOR UPDATE',
            refund_id,
        )
        refund = _required_first(refund_rows, RefundRow)
        if refund.status != "PENDING":
            return
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayWallet"
            SET
                "available_micros" = "available_micros" + $1,
                "reserved_micros" = "reserved_micros" - $1,
                "version" = "version" + 1,
                "updated_at" = CURRENT_TIMESTAMP
            WHERE "wallet_id" = $2
            """,
            refund.amount_micros,
            refund.wallet_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayRefund"
            SET "status" = 'FAILED', "error" = $1, "updated_at" = CURRENT_TIMESTAMP
            WHERE "refund_id" = $2
            """,
            error[:1000],
            refund.refund_id,
        )
        payment_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayPayment" WHERE "payment_id" = $1',
            refund.payment_id,
        )
        payment = _required_first(payment_rows, PaymentRow)
        restored_status = "PAID" if payment.refunded_micros == 0 else "PARTIALLY_REFUNDED"
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayPayment"
            SET "status" = $1::"PublicRelayPaymentStatus", "updated_at" = CURRENT_TIMESTAMP
            WHERE "payment_id" = $2
            """,
            restored_status,
            payment.payment_id,
        )


async def adjust_wallet(
    prisma_client: PrismaClient,
    wallet_id: str,
    amount_micros: int,
    reason: str,
    idempotency_key: str,
) -> WalletRow:
    async with _database(prisma_client).tx() as tx:
        existing = await tx.query_raw(
            'SELECT "entry_id" FROM "LiteLLM_PublicRelayLedgerEntry" WHERE "idempotency_key" = $1',
            idempotency_key,
        )
        if existing:
            wallet_rows = await tx.query_raw(
                'SELECT * FROM "LiteLLM_PublicRelayWallet" WHERE "wallet_id" = $1',
                wallet_id,
            )
            return _required_first(wallet_rows, WalletRow)
        wallet_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayWallet" WHERE "wallet_id" = $1 FOR UPDATE',
            wallet_id,
        )
        wallet = _required_first(wallet_rows, WalletRow)
        available_after = wallet.available_micros + amount_micros
        if available_after < 0:
            raise ArithmeticError("adjustment exceeds available balance")
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayWallet"
            SET
                "available_micros" = $1,
                "version" = "version" + 1,
                "updated_at" = CURRENT_TIMESTAMP
            WHERE "wallet_id" = $2
            """,
            available_after,
            wallet.wallet_id,
        )
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayLedgerEntry"
                (
                    "entry_id",
                    "wallet_id",
                    "entry_type",
                    "amount_micros",
                    "available_after_micros",
                    "reserved_after_micros",
                    "debt_after_micros",
                    "idempotency_key",
                    "metadata"
                )
            VALUES ($1, $2, 'ADJUSTMENT', $3, $4, $5, $6, $7, $8::JSONB)
            """,
            str(uuid.uuid4()),
            wallet.wallet_id,
            amount_micros,
            available_after,
            wallet.reserved_micros,
            wallet.debt_micros,
            idempotency_key,
            json.dumps({"reason": reason}, separators=(",", ":")),
        )
    return WalletRow(
        wallet_id=wallet.wallet_id,
        account_id=wallet.account_id,
        available_micros=available_after,
        reserved_micros=wallet.reserved_micros,
        debt_micros=wallet.debt_micros,
    )


async def set_account_status(prisma_client: PrismaClient, account_id: str, status: str) -> bool:
    if status not in {"ACTIVE", "FROZEN", "CLOSED"}:
        raise ValueError("invalid account status")
    async with _database(prisma_client).tx() as tx:
        account_rows = await tx.query_raw(
            'SELECT * FROM "LiteLLM_PublicRelayAccount" WHERE "account_id" = $1 FOR UPDATE',
            account_id,
        )
        account = _first(account_rows, AccountRow)
        if account is None:
            return False
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayAccount"
            SET
                "status" = $1::"PublicRelayAccountStatus",
                "session_version" = "session_version" + 1,
                "updated_at" = CURRENT_TIMESTAMP
            WHERE "account_id" = $2
            """,
            status,
            account_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_VerificationToken"
            SET "blocked" = $1, "updated_at" = CURRENT_TIMESTAMP
            WHERE "user_id" = $2
              AND COALESCE(("metadata"->>'public_relay')::BOOLEAN, false) = true
            """,
            status != "ACTIVE",
            account.user_id,
        )
    return True


def _price_quote(price: PriceRow) -> PriceQuote:
    return PriceQuote(
        input_micros_per_million=price.input_micros_per_million,
        cached_input_micros_per_million=price.cached_input_micros_per_million,
        output_micros_per_million=price.output_micros_per_million,
        embedding_micros_per_million=price.embedding_micros_per_million,
    )


def _database(prisma_client: PrismaClient) -> DatabaseProtocol:
    return cast(DatabaseProtocol, prisma_client.db)  # cast-ok: PrismaClient supplies the documented database API.


def database_handle(prisma_client: PrismaClient) -> DatabaseProtocol:
    return _database(prisma_client)


async def _sync_public_access_group(transaction: TransactionProtocol) -> None:
    group_rows = await transaction.query_raw(
        """
        INSERT INTO "LiteLLM_AccessGroupTable"
            ("access_group_id", "access_group_name", "description")
        VALUES ($1, $2, $3)
        ON CONFLICT ("access_group_name")
        DO UPDATE SET "updated_at" = CURRENT_TIMESTAMP
        RETURNING "access_group_id"
        """,
        str(uuid.uuid4()),
        PUBLIC_ACCESS_GROUP_NAME,
        "Models with an active public relay price",
    )
    group_id = _single_string(group_rows, "access_group_id")
    model_rows = await transaction.query_raw(
        """
        SELECT "model_name"
        FROM (
            SELECT DISTINCT ON ("model_name") "model_name", "enabled"
            FROM "LiteLLM_PublicRelayModelPrice"
            WHERE "effective_at" <= CURRENT_TIMESTAMP
            ORDER BY "model_name", "version" DESC
        ) latest
        WHERE "enabled" = true
        ORDER BY "model_name"
        """
    )
    if not isinstance(model_rows, list):
        raise TypeError("database rows must be a list")
    models = tuple(
        _required_mapping(row)["model_name"]
        for row in cast(list[object], model_rows)  # cast-ok: Prisma query_raw returns a JSON row sequence.
    )
    await transaction.execute_raw(
        """
        UPDATE "LiteLLM_AccessGroupTable"
        SET "access_model_names" = $1::TEXT[], "updated_at" = CURRENT_TIMESTAMP
        WHERE "access_group_id" = $2
        """,
        models,
        group_id,
    )


def _first(rows: object, model: type[DatabaseRow]) -> DatabaseRow | None:
    if not isinstance(rows, list) or not rows:
        return None
    return model.model_validate(
        cast(list[object], rows)[0]  # cast-ok: Prisma query_raw returns a JSON row sequence.
    )


def _required_first(rows: object, model: type[DatabaseRow]) -> DatabaseRow:
    value = _first(rows, model)
    if value is None:
        raise LookupError(f"{model.__name__} not found")
    return value


def _required_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("database row must be a mapping")
    return cast(dict[str, object], value)  # cast-ok: isinstance validates the Prisma JSON mapping boundary.


def _single_string(rows: object, key: str) -> str:
    if not isinstance(rows, list) or not rows:
        raise LookupError(f"{key} not found")
    value = _required_mapping(
        cast(list[object], rows)[0]  # cast-ok: Prisma query_raw returns a JSON row sequence.
    ).get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _single_int(rows: object, key: str, default: int | None = None) -> int:
    if not isinstance(rows, list) or not rows:
        if default is not None:
            return default
        raise LookupError(f"{key} not found")
    value = _required_mapping(
        cast(list[object], rows)[0]  # cast-ok: Prisma query_raw returns a JSON row sequence.
    ).get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value
