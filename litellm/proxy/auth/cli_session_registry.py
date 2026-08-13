"""Registry of `lite login` sessions.

The CLI credential is a self-contained encrypted ``UserAPIKeyAuth`` blob rather than a
virtual key, so nothing about it is stored server side and the auth path authenticates
it by decrypting it. This registry is the server-side record that makes a session
listable and revocable: one row per login, keyed by the sha256 of the session token.

A session with no row predates the registry and still authenticates until it expires;
only a row with ``revoked_at`` set is refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Final, Protocol

from litellm.constants import CLI_JWT_EXPIRATION_HOURS, DEFAULT_IN_MEMORY_TTL
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.utils import PrismaClient, hash_token
from litellm.repositories.table_repositories import CLISessionRepository
from litellm.types.cli_session import CLISessionListResponse, CLISessionResponse
from litellm.utils import get_utc_datetime

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache

_REVOCATION_CACHE_KEY_PREFIX: Final = "cli_session_revoked"


class _CLISessionRecord(Protocol):
    def model_dump(self) -> Mapping[str, object]: ...


class _CLISessionTable(Protocol):
    async def find_unique(self, where: Mapping[str, object]) -> _CLISessionRecord | None: ...

    async def find_many(
        self,
        where: Mapping[str, object],
        order: Mapping[str, object],
        skip: int,
        take: int,
    ) -> Sequence[_CLISessionRecord]: ...

    async def count(self, where: Mapping[str, object]) -> int: ...

    async def create(self, data: Mapping[str, object]) -> _CLISessionRecord: ...

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> _CLISessionRecord | None: ...


def _cli_session_table(prisma_client: PrismaClient) -> _CLISessionTable:
    table: Final[_CLISessionTable] = CLISessionRepository(prisma_client).table
    return table


def _revocation_cache_key(session_id: str) -> str:
    return f"{_REVOCATION_CACHE_KEY_PREFIX}:{session_id}"


def cli_session_id(session_token: str) -> str:
    """The registry id for a CLI session token. Never the token itself."""
    return hash_token(session_token)


async def record_cli_session(
    *,
    prisma_client: PrismaClient,
    session_token: str,
    user_id: str,
    team_id: str | None,
) -> CLISessionResponse:
    """Register a freshly minted session. Raises if the row cannot be written, so a
    session that could never be revoked is never handed to the CLI."""
    created: Final = await _cli_session_table(prisma_client).create(
        data={  # mutable-ok: prisma payloads are plain dicts
            "session_id": cli_session_id(session_token),
            "user_id": user_id,
            "team_id": team_id,
            "expires_at": get_utc_datetime() + timedelta(hours=CLI_JWT_EXPIRATION_HOURS),
        }
    )
    return CLISessionResponse.model_validate(created.model_dump())


async def is_cli_session_revoked(
    *,
    session_token: str,
    prisma_client: PrismaClient | None,
    user_api_key_cache: DualCache,
) -> bool:
    """Whether an operator has revoked this session.

    Cached for ``DEFAULT_IN_MEMORY_TTL`` so a session costs one lookup per cache
    interval per replica rather than one per request. That TTL is also the bound on
    how long a revocation takes to reach a replica that did not serve the revoke.

    A lookup that cannot reach the database follows the proxy-wide
    ``allow_requests_on_db_unavailable`` posture rather than inventing its own: an
    operator who opted into serving during an outage keeps serving CLI sessions,
    and one who did not gets the same failure every other DB-backed auth read gives.
    """
    if prisma_client is None:
        return False

    session_id: Final = cli_session_id(session_token)
    cache_key: Final = _revocation_cache_key(session_id)
    cached: Final = await user_api_key_cache.async_get_cache(key=cache_key)
    if cached is not None:
        return bool(cached)

    try:
        session: Final = await _get_cli_session(prisma_client=prisma_client, session_id=session_id)
    except Exception as e:  # noqa: BLE001  # handle_db_exception takes any exception and re-raises what it does not recognise
        PrismaDBExceptionHandler.handle_db_exception(e)
        return False

    revoked: Final = session is not None and session.revoked_at is not None
    await user_api_key_cache.async_set_cache(key=cache_key, value=revoked, ttl=DEFAULT_IN_MEMORY_TTL)
    return revoked


async def _get_cli_session(*, prisma_client: PrismaClient, session_id: str) -> CLISessionResponse | None:
    record: Final = await _cli_session_table(prisma_client).find_unique(
        where={"session_id": session_id}  # mutable-ok: prisma query filters are dict-shaped
    )
    return None if record is None else CLISessionResponse.model_validate(record.model_dump())


async def list_cli_sessions(
    *,
    prisma_client: PrismaClient,
    page: int,
    page_size: int,
) -> CLISessionListResponse:
    """Sessions that have not expired yet, newest first. Expired rows are dead weight:
    the blob's own expiry already refuses them."""
    where: Final[Mapping[str, object]] = {  # mutable-ok: prisma query filters are dict-shaped
        "expires_at": {"gt": get_utc_datetime()}  # mutable-ok: prisma query filters are dict-shaped
    }
    table: Final = _cli_session_table(prisma_client)
    records: Final = await table.find_many(
        where=where,
        order={"created_at": "desc"},  # mutable-ok: prisma order is a plain dict
        skip=(page - 1) * page_size,
        take=page_size,
    )
    return CLISessionListResponse(
        sessions=tuple(CLISessionResponse.model_validate(record.model_dump()) for record in records),
        total_count=await table.count(where=where),
    )


async def revoke_cli_session(
    *,
    prisma_client: PrismaClient,
    user_api_key_cache: DualCache,
    session_id: str,
    revoked_by: str | None,
) -> CLISessionResponse | None:
    """Revoke a session, or return ``None`` if no such session is registered.

    Re-revoking keeps the original ``revoked_at`` so the audit trail records when
    access was actually cut off.
    """
    existing: Final = await _get_cli_session(prisma_client=prisma_client, session_id=session_id)
    if existing is None:
        return None

    revoked: Final = (
        existing
        if existing.revoked_at is not None
        else await _mark_cli_session_revoked(
            prisma_client=prisma_client,
            session_id=session_id,
            revoked_by=revoked_by,
        )
    )
    if revoked is None:
        return None

    await user_api_key_cache.async_set_cache(
        key=_revocation_cache_key(session_id),
        value=True,
        ttl=DEFAULT_IN_MEMORY_TTL,
    )
    return revoked


async def _mark_cli_session_revoked(
    *,
    prisma_client: PrismaClient,
    session_id: str,
    revoked_by: str | None,
) -> CLISessionResponse | None:
    record: Final = await _cli_session_table(prisma_client).update(
        where={"session_id": session_id},  # mutable-ok: prisma query filters are dict-shaped
        data={  # mutable-ok: prisma payloads are plain dicts
            "revoked_at": get_utc_datetime(),
            "revoked_by": revoked_by,
        },
    )
    return None if record is None else CLISessionResponse.model_validate(record.model_dump())
