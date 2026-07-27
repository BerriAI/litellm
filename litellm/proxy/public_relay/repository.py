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
    AuthTokenRow,
    KeyRow,
    LedgerRow,
    MarginSummaryRow,
    PriceRow,
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
    wallet: WalletRow


@dataclass(frozen=True, slots=True)
class CreatedKey:
    raw_key: str
    key_id: str


@dataclass(frozen=True, slots=True)
class ReservationResult:
    reservation: ReservationRow
    price: PriceRow


@dataclass(frozen=True, slots=True)
class ActivatedAccount:
    account: AccountRow
    raw_key: str
    key_id: str


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


async def create_enterprise(
    prisma_client: PrismaClient,
    normalized_email: str,
    company_name: str,
    notes: str | None,
    initial_credit_micros: int,
    idempotency_key: str,
    activation_token_hash: str,
    activation_expires_at: datetime,
) -> CreatedAccount:
    account_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    wallet_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    async with _database(prisma_client).tx() as tx:
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_UserTable"
                ("user_id", "user_email", "password", "user_role", "models", "metadata")
            VALUES ($1, $2, NULL, 'internal_user', ARRAY['no-default-models']::TEXT[], '{}'::JSONB)
            """,
            user_id,
            normalized_email,
        )
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayAccount"
                ("account_id", "user_id", "normalized_email", "company_name", "notes", "created_at", "updated_at")
            VALUES ($1, $2, $3, $4, $5, $6, $6)
            """,
            account_id,
            user_id,
            normalized_email,
            company_name,
            notes,
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
        if initial_credit_micros > 0:
            await tx.execute_raw(
                """
                UPDATE "LiteLLM_PublicRelayWallet"
                SET "available_micros" = $1, "version" = 1, "updated_at" = $2
                WHERE "wallet_id" = $3
                """,
                initial_credit_micros,
                now,
                wallet_id,
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
                        "idempotency_key",
                        "metadata"
                    )
                VALUES ($1, $2, 'ADJUSTMENT', $3, $3, 0, $4, $5::JSONB)
                """,
                str(uuid.uuid4()),
                wallet_id,
                initial_credit_micros,
                idempotency_key,
                json.dumps({"reason": "Initial enterprise credit"}, separators=(",", ":")),
            )
        await tx.execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayAuthToken"
                ("auth_token_id", "token_hash", "account_id", "purpose", "expires_at")
            VALUES ($1, $2, $3, 'ACTIVATION', $4)
            """,
            str(uuid.uuid4()),
            activation_token_hash,
            account_id,
            activation_expires_at,
        )
    account = AccountRow(
        account_id=account_id,
        user_id=user_id,
        normalized_email=normalized_email,
        company_name=company_name,
        notes=notes,
        status="INVITED",
        password=None,
        session_version=0,
        created_at=now,
    )
    wallet = WalletRow(
        wallet_id=wallet_id,
        account_id=account_id,
        available_micros=initial_credit_micros,
        reserved_micros=0,
    )
    return CreatedAccount(account=account, wallet=wallet)


async def create_auth_token(
    prisma_client: PrismaClient,
    account_id: str,
    token_hash: str,
    purpose: str,
    expires_at: datetime,
) -> AuthTokenRow:
    auth_token_id = str(uuid.uuid4())
    async with _database(prisma_client).tx() as tx:
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayAuthToken"
            SET "consumed_at" = CURRENT_TIMESTAMP
            WHERE "account_id" = $1 AND "purpose" = $2::"PublicRelayAuthTokenPurpose" AND "consumed_at" IS NULL
            """,
            account_id,
            purpose,
        )
        rows = await tx.query_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayAuthToken"
                ("auth_token_id", "token_hash", "account_id", "purpose", "expires_at")
            VALUES ($1, $2, $3, $4::"PublicRelayAuthTokenPurpose", $5)
            RETURNING *
            """,
            auth_token_id,
            token_hash,
            account_id,
            purpose,
            expires_at,
        )
    return _required_first(rows, AuthTokenRow)


async def activate_account(
    prisma_client: PrismaClient,
    token_hash: str,
    password_hash: str,
) -> ActivatedAccount:
    raw_key = f"sk-{secrets.token_urlsafe(32)}"
    key_id = hash_token(raw_key)
    now = datetime.now(timezone.utc)
    async with _database(prisma_client).tx() as tx:
        token_rows = await tx.query_raw(
            """
            SELECT t.*, a."user_id"
            FROM "LiteLLM_PublicRelayAuthToken" t
            JOIN "LiteLLM_PublicRelayAccount" a ON a."account_id" = t."account_id"
            WHERE t."token_hash" = $1
              AND t."purpose" = 'ACTIVATION'
              AND t."consumed_at" IS NULL
              AND t."expires_at" > CURRENT_TIMESTAMP
              AND a."status" = 'INVITED'
            FOR UPDATE OF t, a
            """,
            token_hash,
        )
        if not isinstance(token_rows, list) or not token_rows:
            raise PermissionError("invalid or expired activation token")
        token_mapping = _required_mapping(token_rows[0])
        account_id = _required_string(token_mapping, "account_id")
        user_id = _required_string(token_mapping, "user_id")
        await tx.execute_raw(
            'UPDATE "LiteLLM_UserTable" SET "password" = $1 WHERE "user_id" = $2',
            password_hash,
            user_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayAccount"
            SET "status" = 'ACTIVE', "activated_at" = $1, "updated_at" = $1
            WHERE "account_id" = $2
            """,
            now,
            account_id,
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayAuthToken"
            SET "consumed_at" = $1
            WHERE "auth_token_id" = $2
            """,
            now,
            _required_string(token_mapping, "auth_token_id"),
        )
        await _insert_api_key(tx, account_id, user_id, key_id, raw_key, "Default", False, now)
        account_rows = await tx.query_raw(
            """
            SELECT a.*, u."password"
            FROM "LiteLLM_PublicRelayAccount" a
            JOIN "LiteLLM_UserTable" u ON u."user_id" = a."user_id"
            WHERE a."account_id" = $1
            """,
            account_id,
        )
    return ActivatedAccount(
        account=_required_first(account_rows, AccountRow),
        raw_key=raw_key,
        key_id=key_id,
    )


async def reset_password_with_token(
    prisma_client: PrismaClient,
    token_hash: str,
    password_hash: str,
) -> None:
    async with _database(prisma_client).tx() as tx:
        rows = await tx.query_raw(
            """
            SELECT t."auth_token_id", a."account_id", a."user_id"
            FROM "LiteLLM_PublicRelayAuthToken" t
            JOIN "LiteLLM_PublicRelayAccount" a ON a."account_id" = t."account_id"
            WHERE t."token_hash" = $1
              AND t."purpose" = 'PASSWORD_RESET'
              AND t."consumed_at" IS NULL
              AND t."expires_at" > CURRENT_TIMESTAMP
              AND a."status" = 'ACTIVE'
            FOR UPDATE OF t, a
            """,
            token_hash,
        )
        if not isinstance(rows, list) or not rows:
            raise PermissionError("invalid or expired password reset token")
        value = _required_mapping(rows[0])
        await tx.execute_raw(
            'UPDATE "LiteLLM_UserTable" SET "password" = $1 WHERE "user_id" = $2',
            password_hash,
            _required_string(value, "user_id"),
        )
        await tx.execute_raw(
            """
            UPDATE "LiteLLM_PublicRelayAccount"
            SET "session_version" = "session_version" + 1, "updated_at" = CURRENT_TIMESTAMP
            WHERE "account_id" = $1
            """,
            _required_string(value, "account_id"),
        )
        await tx.execute_raw(
            'UPDATE "LiteLLM_PublicRelayAuthToken" SET "consumed_at" = CURRENT_TIMESTAMP WHERE "auth_token_id" = $1',
            _required_string(value, "auth_token_id"),
        )
        await tx.execute_raw(
            'DELETE FROM "LiteLLM_PublicRelaySession" WHERE "account_id" = $1',
            _required_string(value, "account_id"),
        )


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
        await _insert_api_key(tx, account.account_id, account.user_id, key_id, raw_key, alias, log_content, now)
    return CreatedKey(raw_key=raw_key, key_id=key_id)


async def _insert_api_key(
    tx: TransactionProtocol,
    account_id: str,
    user_id: str,
    key_id: str,
    raw_key: str,
    alias: str,
    log_content: bool,
    now: datetime,
) -> None:
    metadata = json.dumps(
        {
            "public_relay_account_id": account_id,
            "public_relay": True,
            "public_relay_log_content": log_content,
        },
        separators=(",", ":"),
    )
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
        user_id,
        metadata,
        list(PUBLIC_ALLOWED_ROUTES),
        group_id,
        now,
    )


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
        if wallet.available_micros < reserved_micros:
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
                    "idempotency_key",
                    "request_id",
                    "metadata"
                )
            VALUES ($1, $2, 'RESERVE', 0, $3, $4, $5, $6, $7::JSONB)
            """,
            str(uuid.uuid4()),
            wallet.wallet_id,
            wallet.available_micros - reserved_micros,
            wallet.reserved_micros + reserved_micros,
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
                        "idempotency_key",
                        "request_id",
                        "metadata"
                    )
                VALUES ($1, $2, 'RELEASE', 0, $3, $4, $5, $6, $7::JSONB)
                """,
                str(uuid.uuid4()),
                wallet.wallet_id,
                available_after,
                reserved_after,
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
                    "idempotency_key",
                    "request_id"
                )
            VALUES ($1, $2, 'USAGE', $3, $4, $5, $6, $7)
            """,
            str(uuid.uuid4()),
            wallet.wallet_id,
            -charged_micros,
            available_after,
            reserved_after,
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
                    "idempotency_key",
                    "request_id",
                    "metadata"
                )
            VALUES ($1, $2, 'RELEASE', 0, $3, $4, $5, $6, $7::JSONB)
            """,
            str(uuid.uuid4()),
            wallet.wallet_id,
            wallet.available_micros + reservation.reserved_micros,
            wallet.reserved_micros - reservation.reserved_micros,
            f"release:{request_id}",
            request_id,
            json.dumps({"released_micros": reservation.reserved_micros}, separators=(",", ":")),
        )


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
        SELECT a.*, w."wallet_id", w."available_micros", w."reserved_micros"
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
            VALUES ($1, $2, 'ADJUSTMENT', $3, $4, $5, $6, $7::JSONB)
            """,
            str(uuid.uuid4()),
            wallet.wallet_id,
            amount_micros,
            available_after,
            wallet.reserved_micros,
            idempotency_key,
            json.dumps({"reason": reason}, separators=(",", ":")),
        )
    return WalletRow(
        wallet_id=wallet.wallet_id,
        account_id=wallet.account_id,
        available_micros=available_after,
        reserved_micros=wallet.reserved_micros,
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


def _required_string(value: dict[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str):
        raise TypeError(f"{key} must be a string")
    return result


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
