from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, TypeAdapter

from litellm.proxy.public_relay.config import PublicRelaySettings
from litellm.proxy.public_relay.security import hash_rate_limit_key, hash_session_token
from litellm.proxy.utils import PrismaClient


class PortalSession(BaseModel):
    account_id: str
    user_id: str
    email: str
    session_version: int
    csrf_token: str


@dataclass(frozen=True, slots=True)
class RelayStore:
    prisma_client: PrismaClient
    settings: PublicRelaySettings

    async def enforce_limit(self, key: str, limit: int, window_seconds: int) -> None:
        from litellm.proxy.public_relay.repository import database_handle

        now = datetime.now(timezone.utc)
        window_started_at = datetime.fromtimestamp(
            int(now.timestamp()) // window_seconds * window_seconds,
            timezone.utc,
        )
        expires_at = window_started_at + timedelta(seconds=window_seconds)
        rows = await database_handle(self.prisma_client).query_raw(
            """
            INSERT INTO "LiteLLM_PublicRelayRateLimit"
                ("rate_limit_id", "key_hash", "window_started_at", "window_seconds", "count", "expires_at")
            VALUES ($1, $2, $3, $4, 1, $5)
            ON CONFLICT ("key_hash", "window_started_at", "window_seconds")
            DO UPDATE SET "count" = "LiteLLM_PublicRelayRateLimit"."count" + 1
            RETURNING "count"
            """,
            str(uuid.uuid4()),
            hash_rate_limit_key(self.settings.session_secret, key),
            window_started_at,
            window_seconds,
            expires_at,
        )
        count = TypeAdapter(list[dict[str, int]]).validate_python(rows)[0]["count"]
        if count > limit:
            raise PermissionError("rate limit exceeded")

    async def create_session(self, token: str, session: PortalSession) -> None:
        from litellm.proxy.public_relay.repository import database_handle

        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.settings.session_ttl_seconds)
        await database_handle(self.prisma_client).execute_raw(
            """
            INSERT INTO "LiteLLM_PublicRelaySession"
                (
                    "session_id",
                    "token_hash",
                    "account_id",
                    "user_id",
                    "normalized_email",
                    "session_version",
                    "csrf_token",
                    "expires_at"
                )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            str(uuid.uuid4()),
            hash_session_token(self.settings.session_secret, token),
            session.account_id,
            session.user_id,
            session.email,
            session.session_version,
            session.csrf_token,
            expires_at,
        )

    async def get_session(self, token: str) -> PortalSession | None:
        from litellm.proxy.public_relay.repository import database_handle

        rows = await database_handle(self.prisma_client).query_raw(
            """
            SELECT
                "account_id",
                "user_id",
                "normalized_email" AS "email",
                "session_version",
                "csrf_token"
            FROM "LiteLLM_PublicRelaySession"
            WHERE "token_hash" = $1 AND "expires_at" > CURRENT_TIMESTAMP
            LIMIT 1
            """,
            hash_session_token(self.settings.session_secret, token),
        )
        values = TypeAdapter(list[PortalSession]).validate_python(rows)
        return values[0] if values else None

    async def delete_session(self, token: str) -> None:
        from litellm.proxy.public_relay.repository import database_handle

        await database_handle(self.prisma_client).execute_raw(
            'DELETE FROM "LiteLLM_PublicRelaySession" WHERE "token_hash" = $1',
            hash_session_token(self.settings.session_secret, token),
        )
