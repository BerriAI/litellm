"""SQLAlchemy async engine + session factory for the LiteLLM proxy.

Replaces the Prisma engine subprocess that ``PrismaWrapper`` previously
managed. A handful of features the Prisma layer carried over remain the
responsibility of the layers above this module:

* RDS IAM token rotation (``IAM_TOKEN_DB_AUTH=True``) -- handled here by
  recreating the engine on token refresh.
* Read-replica routing (``DATABASE_URL_READ_REPLICA``) -- handled by
  exposing a separate read-only engine alongside the writer.
* Postgres ``DATETIME`` / ``JSONB`` semantics -- the SQLModel classes in
  :mod:`models` already pin Postgres-specific types via
  ``sqlalchemy.dialects.postgresql``.

This module is intentionally narrow: engine + sessionmaker + lifecycle.
The Prisma-compatible query API lives in :mod:`compat`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    """Return an ``asyncpg``-compatible Postgres URL.

    LiteLLM accepts ``postgres://`` and ``postgresql://`` URLs (Prisma
    accepted both). SQLAlchemy needs ``postgresql+asyncpg://`` for the
    async driver. We rewrite the scheme conservatively and leave the rest
    of the URL (host, port, query string) untouched so that connection
    parameters such as ``sslmode`` survive.
    """
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://") :]
    return url


class LiteLLMDB:
    """Owns the writer (and optional reader) async engines for the proxy.

    The class is deliberately simple. We do not attempt to replicate the
    subprocess-watchdog / engine-PID-reaping behaviour that ``PrismaWrapper``
    inherited from prisma-client-py: SQLAlchemy uses an in-process
    connection pool and exposes the same recovery surface via
    ``engine.dispose()``.
    """

    def __init__(
        self,
        database_url: str,
        *,
        read_replica_url: Optional[str] = None,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_recycle: int = 3600,
    ) -> None:
        if not database_url:
            raise ValueError("LiteLLMDB requires a non-empty DATABASE_URL")
        self._writer_url = database_url
        self._reader_url = read_replica_url
        self._echo = echo
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_recycle = pool_recycle

        self._writer: AsyncEngine = self._build_engine(database_url)
        self._reader: Optional[AsyncEngine] = (
            self._build_engine(read_replica_url) if read_replica_url else None
        )
        self._writer_sm: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._writer, expire_on_commit=False, class_=AsyncSession
        )
        self._reader_sm: Optional[async_sessionmaker[AsyncSession]] = (
            async_sessionmaker(
                self._reader, expire_on_commit=False, class_=AsyncSession
            )
            if self._reader is not None
            else None
        )

        self._iam_refresh_task: Optional[asyncio.Task[None]] = None
        self._iam_refresh_interval_seconds: Optional[int] = None
        self._iam_token_provider = None  # callable[[], str]
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Engine plumbing
    # ------------------------------------------------------------------

    def _build_engine(self, url: str) -> AsyncEngine:
        return create_async_engine(
            _normalize_url(url),
            echo=self._echo,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
            pool_recycle=self._pool_recycle,
            pool_pre_ping=True,
            future=True,
        )

    @property
    def writer(self) -> AsyncEngine:
        return self._writer

    @property
    def reader(self) -> AsyncEngine:
        return self._reader if self._reader is not None else self._writer

    # ------------------------------------------------------------------
    # Session factories
    # ------------------------------------------------------------------

    def session(self) -> AsyncSession:
        """A new writer session (caller is responsible for ``await session.close()``)."""
        return self._writer_sm()

    def reader_session(self) -> AsyncSession:
        """A new reader session, or a writer session if no replica is configured."""
        if self._reader_sm is not None:
            return self._reader_sm()
        return self._writer_sm()

    @asynccontextmanager
    async def session_ctx(self) -> AsyncIterator[AsyncSession]:
        """Context manager that opens, yields, and closes a writer session."""
        session = self._writer_sm()
        try:
            yield session
        finally:
            await session.close()

    @asynccontextmanager
    async def reader_session_ctx(self) -> AsyncIterator[AsyncSession]:
        if self._reader_sm is None:
            async with self.session_ctx() as session:
                yield session
            return
        session = self._reader_sm()
        try:
            yield session
        finally:
            await session.close()

    # ------------------------------------------------------------------
    # Lifecycle (Prisma-compatible surface)
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Eagerly establish a single connection on each engine.

        SQLAlchemy lazily connects on first query, so this is mostly a
        smoke test that catches misconfigured URLs / unreachable hosts at
        startup time -- which is what callers expect from the previous
        ``await prisma_client.db.connect()`` semantics.
        """
        try:
            async with self._writer.connect() as conn:
                await conn.execute(_select_one())
            if self._reader is not None:
                async with self._reader.connect() as conn:
                    await conn.execute(_select_one())
        except SQLAlchemyError as exc:
            logger.error("LiteLLMDB.connect() failed: %s", exc)
            raise
        self._connected = True

    async def disconnect(self) -> None:
        await self.stop_token_refresh_task()
        try:
            await self._writer.dispose()
        finally:
            if self._reader is not None:
                await self._reader.dispose()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # IAM token rotation
    # ------------------------------------------------------------------

    def configure_iam_token_refresh(
        self, *, token_provider, interval_seconds: int
    ) -> None:
        """Wire up a periodic engine recreate using the given token provider.

        ``token_provider`` is a callable returning a fresh password; we
        rebuild the URL and dispose+recreate the engines on each tick.
        """
        if not callable(token_provider):
            raise TypeError("token_provider must be callable")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._iam_token_provider = token_provider
        self._iam_refresh_interval_seconds = interval_seconds

    async def start_token_refresh_task(self) -> None:
        if self._iam_refresh_task is not None and not self._iam_refresh_task.done():
            return
        if (
            self._iam_token_provider is None
            or self._iam_refresh_interval_seconds is None
        ):
            return  # IAM auth not configured
        loop = asyncio.get_event_loop()
        self._iam_refresh_task = loop.create_task(self._iam_refresh_loop())

    async def stop_token_refresh_task(self) -> None:
        task = self._iam_refresh_task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        self._iam_refresh_task = None

    async def _iam_refresh_loop(self) -> None:
        assert self._iam_refresh_interval_seconds is not None
        assert self._iam_token_provider is not None
        while True:
            try:
                await asyncio.sleep(self._iam_refresh_interval_seconds)
                await self._rotate_iam_token()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("IAM token rotation failed; will retry")

    async def _rotate_iam_token(self) -> None:
        """Rebuild engines with a freshly minted IAM token."""
        assert self._iam_token_provider is not None
        new_password = self._iam_token_provider()
        new_writer_url = _replace_password(self._writer_url, new_password)
        old_writer = self._writer
        self._writer = self._build_engine(new_writer_url)
        self._writer_sm = async_sessionmaker(
            self._writer, expire_on_commit=False, class_=AsyncSession
        )
        await old_writer.dispose()

        if self._reader is not None and self._reader_url is not None:
            new_reader_url = _replace_password(self._reader_url, new_password)
            old_reader = self._reader
            self._reader = self._build_engine(new_reader_url)
            self._reader_sm = async_sessionmaker(
                self._reader, expire_on_commit=False, class_=AsyncSession
            )
            await old_reader.dispose()


def _select_one():
    """Lazy import so the module loads without sqlalchemy installed."""
    from sqlalchemy import text

    return text("SELECT 1")


def _replace_password(url: str, new_password: str) -> str:
    """Return ``url`` with its password component replaced.

    Avoids importing ``urllib`` at module import time because some
    deployments swap in a different URL parser.
    """
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url)
    if parsed.username is None:
        return url
    netloc_user = parsed.username
    netloc_host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    new_netloc = f"{netloc_user}:{new_password}@{netloc_host}{port}"
    return urlunparse(parsed._replace(netloc=new_netloc))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DB: Optional[LiteLLMDB] = None


def configure(database_url: str, **kwargs) -> LiteLLMDB:
    """Initialise (or replace) the module-level :class:`LiteLLMDB` singleton."""
    global _DB
    if _DB is not None:
        # Best-effort: caller is reconfiguring (e.g. after credential rotation).
        # We do not auto-dispose the previous engine because the caller may
        # still hold sessions; PrismaClient.connect handles lifecycle there.
        pass
    _DB = LiteLLMDB(database_url, **kwargs)
    return _DB


def get_db() -> LiteLLMDB:
    if _DB is None:
        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "LiteLLMDB has not been configured -- call engine.configure(...) "
                "or set DATABASE_URL before any DB query."
            )
        return configure(url, read_replica_url=os.getenv("DATABASE_URL_READ_REPLICA"))
    return _DB
