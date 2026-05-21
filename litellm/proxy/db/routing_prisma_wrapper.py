"""
RoutingPrismaWrapper: routes Prisma reads to a read-replica client and writes
to a writer client. Used when DATABASE_URL_READ_REPLICA is configured;
otherwise PrismaClient uses the writer-only PrismaWrapper directly.
"""

import os
from typing import Any, Callable, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.prisma_client import PrismaWrapper

# Per-model action methods that read from the database. These are routed to
# the read replica when one is configured.
_MODEL_READ_METHODS = frozenset(
    {
        "find_first",
        "find_first_or_raise",
        "find_many",
        "find_unique",
        "find_unique_or_raise",
        "count",
        "group_by",
        "query_first",
        "query_raw",
    }
)

# Top-level Prisma client methods that read from the database.
_TOP_LEVEL_READ_METHODS = frozenset({"query_first", "query_raw"})


class _RoutedActions:
    """Per-model accessor that sends reads to the reader and writes to the writer.

    `should_use_reader` is consulted on every read dispatch so a mid-call flip
    of the routing wrapper's reader-availability flag (e.g. after the reader
    fails a recreate) is observed without re-fetching the actions accessor.
    """

    __slots__ = ("_writer_actions", "_reader_actions", "_should_use_reader")

    def __init__(
        self,
        writer_actions: Any,
        reader_actions: Any,
        should_use_reader: Callable[[], bool],
    ):
        self._writer_actions = writer_actions
        self._reader_actions = reader_actions
        self._should_use_reader = should_use_reader

    def __getattr__(self, name: str) -> Any:
        if name in _MODEL_READ_METHODS and self._should_use_reader():
            return getattr(self._reader_actions, name)
        return getattr(self._writer_actions, name)


class RoutingPrismaWrapper:
    """
    Routes Prisma operations between a writer and a reader Prisma client.

    Reads (find_*, count, group_by, query_raw, query_first) go to the reader;
    everything else (writes, transactions, raw execute) goes to the writer.
    Lifecycle methods (connect, disconnect, IAM token refresh) act on both
    clients so callers do not need to know about the split. When
    IAM_TOKEN_DB_AUTH is enabled, both writer and reader refresh their tokens
    independently on their own ~12-minute cadence.

    Reader degradation: a reader-side failure (failed connect, failed
    recreate) is non-fatal — the wrapper sets `_reader_unavailable=True`, logs
    a warning, and routes subsequent reads to the writer. The next successful
    `connect()` or `recreate_prisma_client()` clears the flag. This keeps the
    proxy serving traffic during transient reader outages instead of failing
    startup or returning errors for read-heavy endpoints.
    """

    def __init__(self, writer: PrismaWrapper, reader: PrismaWrapper):
        self._writer = writer
        self._reader = reader
        # When True, reads fall back to the writer. Flipped on by reader
        # connect/recreate failures and flipped off on the next reader recovery.
        self._reader_unavailable: bool = False

    @property
    def writer(self) -> PrismaWrapper:
        return self._writer

    @property
    def reader(self) -> PrismaWrapper:
        return self._reader

    @property
    def reader_unavailable(self) -> bool:
        return self._reader_unavailable

    def _should_use_reader(self) -> bool:
        return not self._reader_unavailable

    async def connect(self, *args: Any, **kwargs: Any) -> None:
        await self._writer.connect(*args, **kwargs)
        verbose_proxy_logger.info("[writer] DB connected")
        try:
            await self._reader.connect(*args, **kwargs)
            self._reader_unavailable = False
            verbose_proxy_logger.info("[reader] DB connected")
        except Exception as e:
            # Degrade gracefully: the proxy keeps serving traffic with reads
            # routed to the writer until the reader endpoint is reachable.
            # Aborting startup here would tie proxy availability to an
            # opt-in, best-effort reader endpoint.
            self._reader_unavailable = True
            verbose_proxy_logger.warning(
                "Failed to connect to read replica DB: %s. "
                "Falling back to the writer for reads until the reader is reachable.",
                e,
            )

    async def disconnect(self, *args: Any, **kwargs: Any) -> None:
        first_error: Optional[BaseException] = None
        for client in (self._writer, self._reader):
            try:
                await client.disconnect(*args, **kwargs)
            except Exception as e:
                if first_error is None:
                    first_error = e
                verbose_proxy_logger.warning("Error disconnecting Prisma client: %s", e)
        if first_error is not None:
            raise first_error

    def is_connected(self) -> bool:
        # Reflects writer health only. The reader is best-effort; its
        # availability is tracked via `_reader_unavailable` and a degraded
        # reader must NOT cause a writer reconnect (would loop indefinitely
        # since recreate_prisma_client only fixes writer-side problems).
        return bool(self._writer.is_connected())

    async def start_token_refresh_task(self) -> None:
        await self._writer.start_token_refresh_task()
        await self._reader.start_token_refresh_task()

    async def stop_token_refresh_task(self) -> None:
        await self._writer.stop_token_refresh_task()
        await self._reader.stop_token_refresh_task()

    async def recreate_prisma_client(
        self, new_db_url: str, http_client: Optional[Any] = None
    ) -> None:
        """Recreate both writer and reader Prisma clients.

        The writer reconnect path in PrismaClient calls
        `self.db.recreate_prisma_client(...)`. Without this method, a DB-wide
        connectivity event would only re-create the writer; the reader engine
        would stay broken and every routed read would fail. We always recreate
        the writer first (its URL is the one passed in), then best-effort
        recreate the reader. A reader failure flips `_reader_unavailable=True`
        so reads transparently fall through to the writer.
        """
        await self._writer.recreate_prisma_client(new_db_url, http_client=http_client)
        try:
            await self._recreate_reader(http_client=http_client)
            self._reader_unavailable = False
        except Exception as e:
            self._reader_unavailable = True
            verbose_proxy_logger.warning(
                "Failed to recreate reader Prisma client: %s. "
                "Reads will fall back to the writer until the reader recovers.",
                e,
            )

    async def _recreate_reader(self, http_client: Optional[Any] = None) -> None:
        """Resolve the reader URL and recreate its Prisma client.

        IAM-enabled readers regenerate their token (host/port/user came from
        the parsed reader URL at construction time). Non-IAM readers reuse
        the URL stored in `DATABASE_URL_READ_REPLICA`.
        """
        if self._reader.iam_token_db_auth:
            new_reader_url = self._reader.get_rds_iam_token()
            if not new_reader_url:
                raise RuntimeError(
                    "Failed to generate fresh IAM token for read replica"
                )
            await self._reader.recreate_prisma_client(
                new_reader_url, http_client=http_client
            )
            return
        reader_url = os.getenv("DATABASE_URL_READ_REPLICA", "")
        if not reader_url:
            raise RuntimeError(
                "DATABASE_URL_READ_REPLICA not set; cannot recreate read replica client"
            )
        await self._reader.recreate_prisma_client(reader_url, http_client=http_client)

    def __getattr__(self, name: str) -> Any:
        if name in _TOP_LEVEL_READ_METHODS:
            target = self._writer if self._reader_unavailable else self._reader
            return getattr(target, name)
        writer_attr = getattr(self._writer, name)
        # Per-model action accessors are non-callable instances that expose
        # both `find_many` and `create`. Methods like execute_raw / batch_ /
        # tx are callables and stay on the writer untouched.
        if (
            not callable(writer_attr)
            and hasattr(writer_attr, "find_many")
            and hasattr(writer_attr, "create")
        ):
            try:
                reader_attr = getattr(self._reader, name)
            except AttributeError:
                return writer_attr
            return _RoutedActions(writer_attr, reader_attr, self._should_use_reader)
        return writer_attr
