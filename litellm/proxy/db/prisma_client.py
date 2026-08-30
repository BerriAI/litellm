"""
This file contains the PrismaWrapper class, which wraps the Prisma client and keeps the
database token (AWS RDS IAM or Microsoft Entra ID) fresh.
"""

import asyncio
import os
import random
import signal
import subprocess
import time
import urllib
import urllib.parse
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, Final, Protocol

from litellm._logging import verbose_proxy_logger
from litellm.proxy.db.token_auth import (
    DEFAULT_POSTGRES_PORT,
    DatabaseTokenAuth,
    IAMEndpoint,
    RdsIamTokenAuth,
    mint_database_token,
    parse_database_token_expiration,
    parse_iam_endpoint_from_url,
)
from litellm.secret_managers.main import str_to_bool

__all__ = (
    "IAMEndpoint",
    "PrismaManager",
    "PrismaWrapper",
    "parse_iam_endpoint_from_url",
)


class _PrismaProcess(Protocol):
    pid: object


class _PrismaEngine(Protocol):
    @property
    def process(self) -> _PrismaProcess: ...

    async def query(self, content: str, *, tx_id: str | None) -> object: ...

    async def start_transaction(self, *, content: str) -> str: ...

    async def commit_transaction(self, tx_id: str) -> None: ...

    async def rollback_transaction(self, tx_id: str) -> None: ...


class _PrismaClient(Protocol):
    _Prisma__engine: _PrismaEngine

    @property
    def _engine(self) -> _PrismaEngine: ...


class _PrismaDrainTracker:
    def __init__(self) -> None:
        self._active_operations = 0
        self._transactions: frozenset[str] = frozenset()
        self._drained = asyncio.Event()
        self._drained.set()

    def begin_operation(self) -> None:
        if self._active_operations == 0:
            self._drained.clear()
        self._active_operations += 1

    def end_operation(self) -> None:
        self._active_operations -= 1
        if self._active_operations == 0:
            self._drained.set()

    def transaction_started(self, transaction_id: str) -> None:
        self._transactions = self._transactions.union((transaction_id,))

    def transaction_finished(self, transaction_id: str) -> None:
        if transaction_id not in self._transactions:
            return
        self._transactions = self._transactions.difference((transaction_id,))
        self.end_operation()

    async def wait_until_drained(self) -> None:
        await self._drained.wait()


class _TrackedPrismaEngine:
    def __init__(self, engine: _PrismaEngine, tracker: _PrismaDrainTracker) -> None:
        self._engine = engine
        self.tracker = tracker

    @property
    def process(self) -> _PrismaProcess:
        return self._engine.process

    def __getattr__(self, name: str) -> object:
        return getattr(self._engine, name)

    async def query(self, content: str, *, tx_id: str | None) -> object:
        self.tracker.begin_operation()
        try:
            return await self._engine.query(content, tx_id=tx_id)
        finally:
            self.tracker.end_operation()

    async def start_transaction(self, *, content: str) -> str:
        self.tracker.begin_operation()
        try:
            transaction_id: Final = await self._engine.start_transaction(content=content)
        except (Exception, asyncio.CancelledError):
            self.tracker.end_operation()
            raise
        self.tracker.transaction_started(transaction_id)
        return transaction_id

    async def commit_transaction(self, tx_id: str) -> None:
        self.tracker.begin_operation()
        try:
            await self._engine.commit_transaction(tx_id)
        finally:
            self.tracker.end_operation()
            self.tracker.transaction_finished(tx_id)

    async def rollback_transaction(self, tx_id: str) -> None:
        self.tracker.begin_operation()
        try:
            await self._engine.rollback_transaction(tx_id)
        finally:
            self.tracker.end_operation()
            self.tracker.transaction_finished(tx_id)


class PrismaWrapper:
    """
    Wrapper around Prisma client that handles token-based database authentication.

    When a token strategy is active (AWS RDS IAM or Microsoft Entra ID), this wrapper:
    1. Proactively refreshes the token before it expires (background task)
    2. Falls back to synchronous refresh if a token is found expired
    3. Uses proper locking to prevent race conditions during reconnection

    RDS IAM tokens are valid for 15 minutes and Entra tokens for about an hour. This
    wrapper refreshes 3 minutes before whatever expiry the live token carries.
    """

    # Buffer time in seconds before token expiration to trigger refresh
    # Refresh 3 minutes (180 seconds) before the token expires
    TOKEN_REFRESH_BUFFER_SECONDS = 180

    # Fallback refresh interval if token parsing fails (10 minutes)
    FALLBACK_REFRESH_INTERVAL_SECONDS = 600

    # Floor on the proactive loop's sleep, so a token whose expiry does not advance
    # (azure-identity hands back its cached token when a renewal attempt fails) costs
    # one retry every 30 seconds instead of spinning the loop with no sleep at all.
    TOKEN_REFRESH_MIN_SLEEP_SECONDS = 30

    ENGINE_RETIREMENT_DRAIN_TIMEOUT_SECONDS = 90

    def __init__(
        self,
        original_prisma: Any,
        iam_token_db_auth: bool = False,
        *,
        token_auth: DatabaseTokenAuth | None = None,
        db_url_env_var: str = "DATABASE_URL",
        iam_endpoint: IAMEndpoint | None = None,
        recreate_uses_datasource: bool = False,
        log_prefix: str = "",
    ):
        # Set before `_original_prisma` so the `iam_token_db_auth` property below can
        # never send `__getattr__` looking for a half-built strategy on the raw client.
        self._token_auth = token_auth if token_auth is not None else (RdsIamTokenAuth() if iam_token_db_auth else None)
        self._original_prisma = original_prisma

        # Per-connection knobs so the same wrapper can be used for the writer
        # (defaults: DATABASE_URL env, IAM endpoint from DATABASE_HOST/etc.,
        # recreate via env reload) or for a reader (DATABASE_URL_READ_REPLICA
        # env, IAM endpoint parsed from that URL, recreate via datasource
        # override since Prisma only auto-reads DATABASE_URL).
        self._db_url_env_var = db_url_env_var
        self._iam_endpoint = iam_endpoint
        self._recreate_uses_datasource = recreate_uses_datasource
        # Tag every log line emitted by this wrapper instance so writer and
        # reader can be told apart in interleaved output (e.g. "[writer] RDS
        # IAM token refresh scheduled in 720 seconds"). Empty string (default)
        # keeps backward-compatible logs for the single-DB case.
        self._log_prefix = f"{log_prefix} " if log_prefix else ""

        # Background token refresh task management
        self._token_refresh_task: asyncio.Task | None = None
        self._reconnection_lock = asyncio.Lock()
        self._last_refresh_time: datetime | None = None
        self._active_drain_tracker = self._instrument_prisma_client(original_prisma)
        self._retirement_tasks: frozenset[asyncio.Task[None]] = frozenset()

        # Coordination for planned engine restarts (issue #29176). Every
        # `recreate_prisma_client` SIGTERMs the running query-engine on
        # purpose. The engine-death watcher (in `PrismaClient`) must be able
        # to tell that planned kill apart from a real crash, otherwise it
        # triggers its own reconnect and kills the freshly-spawned engine.
        #   - `_expected_engine_deaths`: PIDs we intentionally killed; the
        #     watcher consumes these instead of reconnecting.
        #   - `_engine_generation`: monotonic counter bumped on every
        #     successful recreate, used by callers as an optimistic-lock token
        #     so racing/cascading recreates collapse into a single restart.
        #   - `on_engine_replaced`: optional callback fired after a recreate so
        #     the owner (PrismaClient) can re-arm its watcher on the new PID.
        self._expected_engine_deaths: set[int] = set()
        self._engine_generation: int = 0
        self.on_engine_replaced: Callable[[], None] | None = None

    @property
    def token_auth(self) -> DatabaseTokenAuth | None:
        """The active database token strategy, or None for password auth."""
        return self._token_auth

    @property
    def token_label(self) -> str:
        """Human name of the active token kind, for log lines."""
        return self._token_auth.label if self._token_auth is not None else "database token"

    @property
    def iam_token_db_auth(self) -> bool:
        """Whether any token strategy is active.

        Read-only: the kind of token is chosen once, by injection, so there is no way
        to flip this back on and silently get AWS RDS on an Azure deployment.
        """
        return self._token_auth is not None

    @staticmethod
    def _read_engine(prisma_client: _PrismaClient) -> _PrismaEngine:
        return prisma_client._engine

    @staticmethod
    def _write_engine(prisma_client: _PrismaClient, engine: _PrismaEngine) -> None:
        prisma_client._Prisma__engine = engine

    def _instrument_prisma_client(self, prisma_client: _PrismaClient) -> _PrismaDrainTracker | None:
        from prisma.errors import ClientNotConnectedError

        try:
            engine: Final = self._read_engine(prisma_client)
        except (AttributeError, ClientNotConnectedError):
            return None
        if isinstance(engine, _TrackedPrismaEngine):
            return engine.tracker
        tracker: Final = _PrismaDrainTracker()
        self._write_engine(prisma_client, _TrackedPrismaEngine(engine, tracker))
        return tracker

    def _get_engine_pid(self, prisma_client: _PrismaClient | None = None) -> int:
        """Get the PID of the current Prisma engine subprocess, or 0 if unavailable.

        Must never raise: it runs inside the reconnect path, where the client
        may be in any broken state. Prisma's ``_engine`` is a property that
        raises ``ClientNotConnectedError`` on a disconnected client; if that
        escaped here, ``recreate_prisma_client`` would fail before it could
        build a replacement client and the reconnect loop could never recover.
        """
        from prisma.errors import ClientNotConnectedError

        try:
            target_prisma: Final = self._original_prisma if prisma_client is None else prisma_client
            pid: Final = self._read_engine(target_prisma).process.pid
            if isinstance(pid, int):
                return pid
        except (AttributeError, ClientNotConnectedError, TypeError):
            pass
        return 0

    async def _retire_engine_when_drained(self, pid: int, tracker: _PrismaDrainTracker | None) -> None:
        if tracker is not None:
            try:
                await asyncio.wait_for(
                    tracker.wait_until_drained(),
                    timeout=self.ENGINE_RETIREMENT_DRAIN_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                verbose_proxy_logger.warning(
                    "%sReplaced prisma engine PID %s did not drain within %ss; killing it with work still in flight.",
                    self._log_prefix,
                    pid,
                    self.ENGINE_RETIREMENT_DRAIN_TIMEOUT_SECONDS,
                )
        await self._kill_engine_process(pid)

    def _schedule_engine_retirement(self, pid: int, tracker: _PrismaDrainTracker | None) -> None:
        if pid <= 0:
            return
        retirement_task: Final = asyncio.create_task(self._retire_engine_when_drained(pid, tracker))
        self._retirement_tasks = self._retirement_tasks.union((retirement_task,))
        retirement_task.add_done_callback(self._retirement_finished)

    def _retirement_finished(self, retirement_task: asyncio.Task[None]) -> None:
        self._retirement_tasks = self._retirement_tasks.difference((retirement_task,))

    async def connect(self, timeout: int | timedelta | None = None) -> None:
        if timeout is None:
            await self._original_prisma.connect()
        else:
            await self._original_prisma.connect(timeout)
        self._active_drain_tracker = self._instrument_prisma_client(self._original_prisma)

    @staticmethod
    async def _kill_engine_process(pid: int) -> None:
        """Force-kill the engine subprocess to prevent DB connection pool leaks.

        Called on every reconnect (in `recreate_prisma_client`) to retire the
        old query-engine subprocess without invoking prisma-client-py's
        synchronous `disconnect()` — which blocks the asyncio event loop on
        `subprocess.Popen.wait()` for 30-120+ seconds when the engine is
        stuck on TCP close.

        Sends SIGTERM for graceful shutdown, waits briefly, then SIGKILL as
        a backstop.
        """
        if pid <= 0:
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            return  # Already dead or inaccessible
        verbose_proxy_logger.warning(
            "Sent SIGTERM to prisma-query-engine PID %s during reconnect.",
            pid,
        )
        # Brief wait for graceful shutdown, then force-kill
        await asyncio.sleep(0.5)
        try:
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            verbose_proxy_logger.warning(
                "Sent SIGKILL to prisma-query-engine PID %s (did not exit after SIGTERM).",
                pid,
            )
        except (ProcessLookupError, PermissionError, OSError):
            pass  # Exited after SIGTERM — expected

    def _extract_token_from_db_url(self, db_url: str | None) -> str | None:
        """
        Extract the token (password) from the DATABASE_URL.

        The token contains the AWS signature with X-Amz-Date and X-Amz-Expires parameters.

        Important: We must parse the URL while it's still encoded to preserve structure,
        then decode the password portion. Otherwise the '?' in the token breaks URL parsing.
        """
        if db_url is None:
            return None
        try:
            # Parse URL while still encoded to preserve structure
            parsed: Final = urllib.parse.urlparse(db_url)
            if parsed.password:
                # Now decode just the password/token
                return urllib.parse.unquote(parsed.password)
            return None
        except Exception:
            return None

    def _parse_token_expiration(self, token: str | None) -> datetime | None:
        """
        Parse the token to extract its expiration time.

        Returns the datetime when the token expires, or None if parsing fails.
        """
        if token is None or self._token_auth is None:
            return None
        return parse_database_token_expiration(self._token_auth, token)

    def _calculate_seconds_until_refresh(self) -> float:
        """
        Calculate exactly how many seconds until we need to refresh the token.

        Uses precise timing: sleeps until (token_expiration - buffer_seconds).
        For a 15-minute (900s) token with 180s buffer, this returns ~720s (12 min).

        Returns:
            Number of seconds to sleep before the next refresh, never less than
            TOKEN_REFRESH_MIN_SLEEP_SECONDS so a token whose expiry never advances
            cannot spin the loop.
            Returns FALLBACK_REFRESH_INTERVAL_SECONDS if parsing fails.
        """
        db_url: Final = os.getenv(self._db_url_env_var)
        token: Final = self._extract_token_from_db_url(db_url)
        expiration_time: Final = self._parse_token_expiration(token)

        if expiration_time is None:
            # If we can't parse the token, use fallback interval
            verbose_proxy_logger.debug(
                "Could not parse token expiration, using fallback interval of %ss",
                self.FALLBACK_REFRESH_INTERVAL_SECONDS,
            )
            return self.FALLBACK_REFRESH_INTERVAL_SECONDS

        # Calculate when we should refresh (expiration - buffer)
        refresh_at: Final = expiration_time - timedelta(seconds=self.TOKEN_REFRESH_BUFFER_SECONDS)

        # How long until refresh time?
        now: Final = datetime.utcnow()
        seconds_until_refresh: Final = (refresh_at - now).total_seconds()

        # Past refresh time means refresh as soon as the floor allows, not instantly:
        # a provider that keeps handing back the same token would otherwise leave the
        # loop re-minting and recreating the query engine with no sleep between passes.
        return max(self.TOKEN_REFRESH_MIN_SLEEP_SECONDS, seconds_until_refresh)

    def is_token_expired(self, token_url: str | None) -> bool:
        """Check if the token in the given URL is expired."""
        if token_url is None:
            return True

        token: Final = self._extract_token_from_db_url(token_url)
        expiration_time: Final = self._parse_token_expiration(token)

        if expiration_time is None:
            # If we can't parse the token, assume it's expired to trigger refresh
            verbose_proxy_logger.debug("Could not parse token expiration, treating as expired")
            return True

        return datetime.utcnow() > expiration_time

    def get_rds_iam_token(self) -> str | None:
        """Mint a fresh database token and update the configured DB URL env var.

        When the wrapper was constructed with an explicit `iam_endpoint`
        (typical for a reader wrapper whose host/port/user came from a parsed
        URL), use that. Otherwise fall back to the DATABASE_HOST/PORT/USER/
        NAME/SCHEMA env vars (writer behavior).
        """
        auth: Final = self._token_auth
        if auth is None:
            return None

        endpoint: Final = self._iam_endpoint if self._iam_endpoint is not None else self._endpoint_from_env()
        db_url: Final = endpoint.build_url(mint_database_token(auth, endpoint))
        os.environ[self._db_url_env_var] = db_url
        return db_url

    @staticmethod
    def _endpoint_from_env() -> IAMEndpoint:
        host: Final = os.getenv("DATABASE_HOST")
        user: Final = os.getenv("DATABASE_USER")
        name: Final = os.getenv("DATABASE_NAME")
        if not host or not user or not name:
            missing: Final = tuple(
                env
                for env, value in (("DATABASE_HOST", host), ("DATABASE_USER", user), ("DATABASE_NAME", name))
                if not value
            )
            raise RuntimeError(
                f"Cannot mint a database token: {', '.join(missing)} unset. Set them so the "
                "connection URL can be reassembled around a freshly minted token."
            )
        return IAMEndpoint(
            host=host,
            # Default to the Postgres standard port; passing None to
            # `generate_iam_auth_token` makes botocore embed the literal
            # string "None" in the presigned URL, which then fails to parse.
            port=os.getenv("DATABASE_PORT", DEFAULT_POSTGRES_PORT),
            user=user,
            name=name,
            schema=os.getenv("DATABASE_SCHEMA"),
        )

    @property
    def engine_generation(self) -> int:
        """How many query-engine replacements have completed on this wrapper.

        Bumped under `_reconnection_lock` only after a replacement engine has
        connected, so a change across an await proves a *successful* planned
        replacement happened in between — a replacement that failed (a real
        outage) leaves it untouched.
        """
        return self._engine_generation

    async def _reconnection_settled(self) -> None:
        async with self._reconnection_lock:
            pass

    async def wait_for_planned_engine_replacement(self, timeout_seconds: float) -> None:
        """Wait, bounded, for an in-flight planned engine replacement to finish.

        Both replacement paths (`recreate_prisma_client` and
        `_safe_refresh_token`) hold `_reconnection_lock` across their whole
        kill/connect window, so re-acquiring it means the replacement has
        settled one way or the other. Gives up silently on timeout: a caller
        that stopped waiting must treat the replacement as not completed and
        consult `engine_generation` rather than assume success.
        """
        if timeout_seconds <= 0 or not self._reconnection_lock.locked():
            return
        try:
            await asyncio.wait_for(self._reconnection_settled(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return

    async def recreate_prisma_client(
        self,
        new_db_url: str,
        http_client: Any | None = None,
        *,
        expected_generation: int | None = None,
    ) -> bool:
        """Disconnect and reconnect the Prisma client with a new database URL.

        Kills the old engine subprocess directly (SIGTERM → SIGKILL) rather than
        calling `disconnect()`. prisma-client-py's `disconnect()` calls a
        synchronous `subprocess.Popen.wait()` that can freeze the asyncio event
        loop for 30-120+ seconds when the engine is stuck on TCP close,
        breaking `/health/liveliness` and causing Kubernetes pod restarts.

        The writer wrapper relies on Prisma re-reading `DATABASE_URL` from env;
        the reader wrapper opts into `recreate_uses_datasource=True` so the
        new URL is passed explicitly via `datasource={"url": ...}` (Prisma
        does not auto-read alternate env vars like DATABASE_URL_READ_REPLICA).

        Serializes all recreations through `self._reconnection_lock` so the
        IAM-refresh path and the engine-death/transport-error reconnect paths
        cannot recreate concurrently (issue #29176). `expected_generation`, if
        given, is an optimistic-lock token: when it no longer matches
        `self._engine_generation` once the lock is held, another path already
        replaced the engine, so this call is a no-op and returns ``False``.

        Returns:
            bool: ``True`` if the client was actually recreated, ``False`` if
            the recreate was skipped because the engine generation moved on.
        """
        async with self._reconnection_lock:
            return await self._recreate_prisma_client_locked(
                new_db_url,
                http_client=http_client,
                expected_generation=expected_generation,
            )

    async def _recreate_prisma_client_locked(
        self,
        new_db_url: str,
        http_client: Any | None = None,
        *,
        expected_generation: int | None = None,
    ) -> bool:
        """Core recreate logic. Caller MUST hold `self._reconnection_lock`.

        Split out so callers that already hold the lock (e.g.
        `_safe_refresh_token`, which double-checks token freshness under the
        lock) don't re-acquire it — `asyncio.Lock` is not reentrant.
        """
        from prisma import Prisma

        if expected_generation is not None and expected_generation != self._engine_generation:
            verbose_proxy_logger.info(
                "%sSkipping Prisma client recreate: engine already replaced (generation %s != expected %s).",
                self._log_prefix,
                self._engine_generation,
                expected_generation,
            )
            return False

        old_engine_pid: Final = self._get_engine_pid()
        if old_engine_pid > 0:
            # Record BEFORE the kill so the engine-death watcher, which may
            # fire the instant the process dies, recognizes this as a planned
            # restart and does not launch its own reconnect.
            #
            # A stale entry can linger when the watcher re-arms on the new PID
            # before the old PID's death callback runs (the callback then
            # early-returns on PID mismatch without consuming it). Such entries
            # are harmless but would accumulate on a long-running proxy (~one
            # per IAM refresh), so cap the set — those old PIDs are long dead.
            if len(self._expected_engine_deaths) >= 64:
                self._expected_engine_deaths.clear()
            self._expected_engine_deaths.add(old_engine_pid)
            await self._kill_engine_process(old_engine_pid)

        kwargs: Final[dict[str, Any]] = {}
        if http_client is not None:
            kwargs["http"] = http_client
        if self._recreate_uses_datasource:
            kwargs["datasource"] = {"url": new_db_url}
        self._original_prisma = Prisma(**kwargs)

        await self._original_prisma.connect()
        self._active_drain_tracker = self._instrument_prisma_client(self._original_prisma)
        self._engine_generation += 1

        # Let the owner (PrismaClient) re-arm its engine-death watcher on the
        # newly-spawned engine PID. Scheduled, never awaited, so a slow watcher
        # can't stall the refresh while we hold the reconnection lock.
        if self.on_engine_replaced is not None:
            self.on_engine_replaced()

        return True

    async def _replace_prisma_client_for_token_refresh_locked(
        self,
        new_db_url: str,
    ) -> None:
        from prisma import Prisma
        from prisma.types import DatasourceOverride

        if self._recreate_uses_datasource:
            datasource: Final = DatasourceOverride(url=new_db_url)
            replacement_prisma = Prisma(datasource=datasource)
        else:
            replacement_prisma = Prisma()

        old_prisma: Final = self._original_prisma
        old_engine_pid: Final = self._get_engine_pid(old_prisma)
        old_drain_tracker = self._active_drain_tracker
        if old_drain_tracker is None:
            old_drain_tracker = self._instrument_prisma_client(old_prisma)
        try:
            await replacement_prisma.connect()
        except (Exception, asyncio.CancelledError):
            self._schedule_engine_retirement(self._get_engine_pid(replacement_prisma), None)
            raise
        replacement_drain_tracker: Final = self._instrument_prisma_client(replacement_prisma)

        if old_engine_pid > 0:
            if len(self._expected_engine_deaths) >= 64:
                self._expected_engine_deaths.clear()
            self._expected_engine_deaths.add(old_engine_pid)

        self._original_prisma = replacement_prisma
        self._active_drain_tracker = replacement_drain_tracker
        self._engine_generation += 1

        if self.on_engine_replaced is not None:
            self.on_engine_replaced()

        self._schedule_engine_retirement(old_engine_pid, old_drain_tracker)

    async def start_token_refresh_task(self) -> None:
        """
        Start the background token refresh task.

        This task proactively refreshes the database token before it expires,
        preventing connection failures. Should be called after the initial
        Prisma client connection is established.
        """
        if not self.iam_token_db_auth:
            verbose_proxy_logger.debug("Database token auth not enabled, skipping token refresh task")
            return

        if self._token_refresh_task is not None:
            verbose_proxy_logger.debug("Token refresh task already running")
            return

        self._token_refresh_task = asyncio.create_task(self._token_refresh_loop())
        verbose_proxy_logger.info(
            "%sStarted %s proactive refresh background task",
            self._log_prefix,
            self.token_label,
        )

    async def stop_token_refresh_task(self) -> None:
        """
        Stop the background token refresh task gracefully.

        Should be called during application shutdown to clean up resources.
        """
        if self._token_refresh_task is None:
            return

        self._token_refresh_task.cancel()
        try:
            await self._token_refresh_task
        except asyncio.CancelledError:
            pass
        self._token_refresh_task = None
        verbose_proxy_logger.info(
            "%sStopped %s refresh background task",
            self._log_prefix,
            self.token_label,
        )

    async def _token_refresh_loop(self) -> None:
        """
        Background loop that proactively refreshes database tokens before expiration.

        Uses precise timing: calculates the exact sleep duration until the token
        needs to be refreshed (expiration - 3 minute buffer), then refreshes.
        This is more efficient than polling, requiring only 1 wake-up per token cycle.
        """
        verbose_proxy_logger.info(
            "%s%s refresh loop started. Tokens will be refreshed %ss before expiration.",
            self._log_prefix,
            self.token_label,
            self.TOKEN_REFRESH_BUFFER_SECONDS,
        )

        while True:
            try:
                # Calculate exactly how long to sleep until next refresh
                sleep_seconds = self._calculate_seconds_until_refresh()

                if sleep_seconds > 0:
                    verbose_proxy_logger.info(
                        f"{self._log_prefix}{self.token_label} refresh scheduled in "
                        f"{sleep_seconds:.0f} seconds ({sleep_seconds / 60:.1f} minutes)"
                    )
                    await asyncio.sleep(sleep_seconds)

                # Refresh the token
                verbose_proxy_logger.info(
                    "%sProactively refreshing %s...",
                    self._log_prefix,
                    self.token_label,
                )
                await self._safe_refresh_token()

            except asyncio.CancelledError:
                verbose_proxy_logger.info(
                    "%s%s refresh loop cancelled",
                    self._log_prefix,
                    self.token_label,
                )
                break
            except Exception as e:
                verbose_proxy_logger.error(
                    "%sError in %s refresh loop: %s. Retrying in %ss...",
                    self._log_prefix,
                    self.token_label,
                    e,
                    self.FALLBACK_REFRESH_INTERVAL_SECONDS,
                )
                # On error, wait before retrying to avoid tight error loops
                try:
                    await asyncio.sleep(self.FALLBACK_REFRESH_INTERVAL_SECONDS)
                except asyncio.CancelledError:
                    break

    async def _safe_refresh_token(self) -> None:
        """
        Refresh the database token with proper locking to prevent race conditions.

        Uses an asyncio lock to ensure only one refresh operation happens at a time,
        preventing multiple concurrent reconnection attempts.
        """
        async with self._reconnection_lock:
            # Double-checked under the lock: another trigger (e.g. the
            # proactive loop racing a __getattr__ fallback) may have already
            # refreshed while we waited. Recreating again would needlessly kill
            # the engine that refresh just spawned (issue #29176), so coalesce
            # by skipping when the current token still has comfortable runway.
            if self._token_refresh_not_needed(os.getenv(self._db_url_env_var)):
                verbose_proxy_logger.debug(
                    "%s%s still fresh; skipping redundant refresh.",
                    self._log_prefix,
                    self.token_label,
                )
                return

            previous_db_url: Final = os.getenv(self._db_url_env_var)
            new_db_url: Final = self.get_rds_iam_token()
            if new_db_url:
                try:
                    await self._replace_prisma_client_for_token_refresh_locked(new_db_url)
                except (Exception, asyncio.CancelledError):
                    if previous_db_url is None:
                        os.environ.pop(self._db_url_env_var, None)
                    else:
                        os.environ[self._db_url_env_var] = previous_db_url
                    raise
                self._last_refresh_time = datetime.utcnow()
                verbose_proxy_logger.info(
                    "%s%s refreshed successfully.",
                    self._log_prefix,
                    self.token_label,
                )
            else:
                verbose_proxy_logger.error(
                    "%sFailed to generate new %s during proactive refresh",
                    self._log_prefix,
                    self.token_label,
                )

    def _token_refresh_not_needed(self, token_url: str | None) -> bool:
        """True iff the token in ``token_url`` has more than the refresh buffer
        of runway left, so a refresh would be redundant.

        Used to coalesce stacked refresh triggers. Deliberately mirrors the
        proactive loop's schedule (refresh at ``expiration - buffer``): a token
        with exactly ``buffer`` seconds left is NOT considered fresh, so the
        legitimate proactive refresh still fires. Unparseable tokens return
        ``False`` (refresh) — skipping them would mean never refreshing.
        """
        token: Final = self._extract_token_from_db_url(token_url)
        expiration_time: Final = self._parse_token_expiration(token)
        if expiration_time is None:
            return False
        seconds_left: Final = (expiration_time - datetime.utcnow()).total_seconds()
        return seconds_left > self.TOKEN_REFRESH_BUFFER_SECONDS

    def __getattr__(self, name: str):
        """
        Proxy attribute access to the underlying Prisma client.

        If IAM token auth is enabled and the token is found expired here, the
        proactive refresh task has missed its window. Behavior depends on
        whether we're called from inside a running event loop:

        - Inside the loop (typical: from a coroutine): schedule a refresh as a
          background task and return the (stale) attribute. The caller's await
          will likely fail with a connection error and be retried by upper
          layers (`call_with_db_reconnect_retry`); by that time the refresh
          has either completed or escalated to the proactive loop's error
          path. We CANNOT block here — `run_coroutine_threadsafe(...)` +
          `future.result()` from inside the same loop deadlocks the loop
          (loop thread is blocked, scheduled coroutine never runs, 30s timeout).

        - No running loop (sync caller, mostly tests): run the refresh in a
          fresh loop and re-fetch the attribute.
        """
        original_attr = getattr(self._original_prisma, name)

        if self.iam_token_db_auth:
            db_url: Final = os.getenv(self._db_url_env_var)

            # Check if token is expired (should be rare if background task is running)
            if self.is_token_expired(db_url):
                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None

                if running_loop is not None:
                    verbose_proxy_logger.warning(
                        "%s%s expired in __getattr__ - proactive refresh "
                        "may have failed. Scheduling async refresh; the current "
                        "request may fail and be retried with the fresh token.",
                        self._log_prefix,
                        self.token_label,
                    )
                    # Non-blocking: schedule the locked refresh on the
                    # running loop. The reconnection lock inside
                    # `_safe_refresh_token` coalesces concurrent triggers.
                    running_loop.create_task(self._safe_refresh_token())
                else:
                    verbose_proxy_logger.warning(
                        "%s%s expired in __getattr__ - proactive refresh "
                        "may have failed. Triggering synchronous fallback refresh...",
                        self._log_prefix,
                        self.token_label,
                    )
                    new_db_url: Final = self.get_rds_iam_token()
                    if new_db_url:
                        asyncio.run(self.recreate_prisma_client(new_db_url))
                        # Re-fetch attribute against the recreated Prisma instance.
                        original_attr = getattr(self._original_prisma, name)
                        verbose_proxy_logger.info(
                            "%sSynchronous token refresh completed successfully",
                            self._log_prefix,
                        )
                    else:
                        raise ValueError(f"Failed to get {self.token_label}")

        return original_attr


class PrismaManager:
    @staticmethod
    def _get_prisma_dir() -> str:
        """Get the path to the migrations directory"""
        abspath: Final = os.path.abspath(__file__)
        dname: Final = os.path.dirname(os.path.dirname(abspath))
        return dname

    @staticmethod
    def _apply_replica_identity_full_if_requested() -> None:
        """
        `prisma db push` bypasses litellm-proxy-extras, so the opt-in
        REPLICA IDENTITY FULL step has to be driven from here too.

        litellm-proxy-extras is an optional install, so this is a no-op when it
        is absent.
        """
        try:
            from litellm_proxy_extras.utils import ProxyExtrasDBManager
        except ImportError:
            return
        ProxyExtrasDBManager.apply_replica_identity_full_if_requested()

    @staticmethod
    def _raise_if_partitioned_spend_logs() -> None:
        """`prisma db push` rewrites a doc-partitioned LiteLLM_SpendLogs
        primary key back to ("request_id"), which Postgres rejects. Fail fast
        with guidance instead of retrying into that raw error. No-op when
        litellm-proxy-extras is absent."""
        try:
            from litellm_proxy_extras.utils import (
                PARTITIONED_SPEND_LOGS_PUSH_ERROR,
                ProxyExtrasDBManager,
            )
        except ImportError:
            return
        if ProxyExtrasDBManager.spend_logs_is_partitioned():
            raise RuntimeError(PARTITIONED_SPEND_LOGS_PUSH_ERROR)

    @staticmethod
    def setup_database(use_migrate: bool = False, use_v2_resolver: bool = False) -> bool:
        """
        Set up the database using either prisma migrate or prisma db push

        Args:
            use_migrate: Use `prisma migrate deploy` instead of `db push`.
            use_v2_resolver: Opt into the v2 migration resolver that avoids
                the diff-and-force recovery behavior (which caused schema
                thrashing during rolling deploys). Defaults to False.

        Returns:
            bool: True if setup was successful, False otherwise
        """

        for attempt in range(4):
            original_dir = os.getcwd()
            prisma_dir = PrismaManager._get_prisma_dir()
            os.chdir(prisma_dir)
            try:
                if use_migrate:
                    try:
                        from litellm_proxy_extras.utils import ProxyExtrasDBManager
                    except ImportError as e:
                        verbose_proxy_logger.error("\x1b[1;31mLiteLLM: Failed to import proxy extras. Got %s\x1b[0m", e)
                        return False

                    prisma_dir = PrismaManager._get_prisma_dir()

                    return ProxyExtrasDBManager.setup_database(
                        use_migrate=use_migrate,
                        use_v2_resolver=use_v2_resolver,
                    )
                else:
                    PrismaManager._raise_if_partitioned_spend_logs()
                    # Use prisma db push with increased timeout
                    subprocess.run(
                        [
                            "prisma",
                            "db",
                            "push",
                            "--accept-data-loss",
                            "--skip-generate",
                        ],
                        timeout=60,
                        check=True,
                    )
                    PrismaManager._apply_replica_identity_full_if_requested()
                    return True
            except subprocess.TimeoutExpired:
                verbose_proxy_logger.warning("Attempt %s timed out", attempt + 1)
                time.sleep(random.randrange(5, 15))
            except subprocess.CalledProcessError as e:
                attempts_left = 3 - attempt
                retry_msg = f" Retrying... ({attempts_left} attempts left)" if attempts_left > 0 else ""
                verbose_proxy_logger.warning("The process failed to execute. Details: %s.%s", e, retry_msg)
                time.sleep(random.randrange(5, 15))
            finally:
                os.chdir(original_dir)
        return False


def should_update_prisma_schema(
    disable_updates: bool | str | None = None,
) -> bool:
    """
    Determines if Prisma Schema updates should be applied during startup.

    Args:
        disable_updates: Controls whether schema updates are disabled.
            Accepts boolean or string ('true'/'false'). Defaults to checking DISABLE_SCHEMA_UPDATE env var.

    Returns:
        bool: True if schema updates should be applied, False if updates are disabled.

    Examples:
        >>> should_update_prisma_schema()  # Checks DISABLE_SCHEMA_UPDATE env var
        >>> should_update_prisma_schema(True)  # Explicitly disable updates
        >>> should_update_prisma_schema("false")  # Enable updates using string
    """
    if disable_updates is None:
        disable_updates = os.getenv("DISABLE_SCHEMA_UPDATE", "false")

    if isinstance(disable_updates, str):
        disable_updates = str_to_bool(disable_updates)

    return not bool(disable_updates)
