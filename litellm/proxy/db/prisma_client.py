"""
This file contains the PrismaWrapper class, which is used to wrap the Prisma client and handle the RDS IAM token.
"""

import asyncio
import os
import random
import signal
import subprocess
import time
import urllib
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.secret_managers.main import str_to_bool


@dataclass(frozen=True)
class IAMEndpoint:
    """Static parts of an RDS IAM-authenticated Postgres connection.

    The IAM token rotates every ~15 minutes; everything else (host, port, user,
    database name, schema) stays fixed. We capture the static fields once so
    refresh just regenerates the token and reassembles the URL.
    """

    host: str
    port: str
    user: str
    name: str
    schema: Optional[str] = None

    def build_url(self, token: str) -> str:
        url = f"postgresql://{self.user}:{token}@{self.host}:{self.port}/{self.name}"
        if self.schema:
            url += f"?schema={self.schema}"
        return url


def parse_iam_endpoint_from_url(url: str) -> IAMEndpoint:
    """Parse an IAMEndpoint from a Postgres URL.

    Used so a reader URL can drive its own IAM refresh without requiring
    callers to set parallel DATABASE_HOST_READ_REPLICA / etc. env vars.
    """
    parsed = urllib.parse.urlparse(url)
    if not parsed.hostname or not parsed.username:
        raise ValueError("Cannot parse IAM endpoint from URL: missing host or username")
    name = (parsed.path or "/").lstrip("/")
    if not name:
        raise ValueError("Cannot parse IAM endpoint from URL: missing database name")
    port = str(parsed.port) if parsed.port else "5432"
    schema: Optional[str] = None
    if parsed.query:
        qs = urllib.parse.parse_qs(parsed.query)
        schema_vals = qs.get("schema")
        if schema_vals:
            schema = schema_vals[0]
    return IAMEndpoint(
        host=parsed.hostname,
        port=port,
        user=parsed.username,
        name=name,
        schema=schema,
    )


class PrismaWrapper:
    """
    Wrapper around Prisma client that handles RDS IAM token authentication.

    When iam_token_db_auth is enabled, this wrapper:
    1. Proactively refreshes IAM tokens before they expire (background task)
    2. Falls back to synchronous refresh if a token is found expired
    3. Uses proper locking to prevent race conditions during reconnection

    RDS IAM tokens are valid for 15 minutes. This wrapper refreshes them
    3 minutes before expiration to ensure uninterrupted database connectivity.
    """

    # Buffer time in seconds before token expiration to trigger refresh
    # Refresh 3 minutes (180 seconds) before the token expires
    TOKEN_REFRESH_BUFFER_SECONDS = 180

    # Fallback refresh interval if token parsing fails (10 minutes)
    FALLBACK_REFRESH_INTERVAL_SECONDS = 600

    def __init__(
        self,
        original_prisma: Any,
        iam_token_db_auth: bool,
        *,
        db_url_env_var: str = "DATABASE_URL",
        iam_endpoint: Optional[IAMEndpoint] = None,
        recreate_uses_datasource: bool = False,
        log_prefix: str = "",
    ):
        self._original_prisma = original_prisma
        self.iam_token_db_auth = iam_token_db_auth

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
        self._token_refresh_task: Optional[asyncio.Task] = None
        self._reconnection_lock = asyncio.Lock()
        self._last_refresh_time: Optional[datetime] = None

    def _get_engine_pid(self) -> int:
        """Get the PID of the current Prisma engine subprocess, or 0 if unavailable."""
        try:
            engine = self._original_prisma._engine
            process = getattr(engine, "process", None) if engine is not None else None
            if process is not None:
                pid = process.pid
                if isinstance(pid, int):
                    return pid
        except (AttributeError, TypeError):
            pass
        return 0

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

    def _extract_token_from_db_url(self, db_url: Optional[str]) -> Optional[str]:
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
            parsed = urllib.parse.urlparse(db_url)
            if parsed.password:
                # Now decode just the password/token
                return urllib.parse.unquote(parsed.password)
            return None
        except Exception:
            return None

    def _parse_token_expiration(self, token: Optional[str]) -> Optional[datetime]:
        """
        Parse the token to extract its expiration time.

        Returns the datetime when the token expires, or None if parsing fails.
        """
        if token is None:
            return None

        try:
            # Token format: ...?X-Amz-Date=YYYYMMDDTHHMMSSZ&X-Amz-Expires=900&...
            if "?" not in token:
                return None

            query_string = token.split("?", 1)[1]
            params = urllib.parse.parse_qs(query_string)

            expires_str = params.get("X-Amz-Expires", [None])[0]
            date_str = params.get("X-Amz-Date", [None])[0]

            if not expires_str or not date_str:
                return None

            token_created = datetime.strptime(date_str, "%Y%m%dT%H%M%SZ")
            expires_in = int(expires_str)

            return token_created + timedelta(seconds=expires_in)
        except Exception as e:
            verbose_proxy_logger.debug(f"Failed to parse token expiration: {e}")
            return None

    def _calculate_seconds_until_refresh(self) -> float:
        """
        Calculate exactly how many seconds until we need to refresh the token.

        Uses precise timing: sleeps until (token_expiration - buffer_seconds).
        For a 15-minute (900s) token with 180s buffer, this returns ~720s (12 min).

        Returns:
            Number of seconds to sleep before the next refresh.
            Returns 0 if token should be refreshed immediately.
            Returns FALLBACK_REFRESH_INTERVAL_SECONDS if parsing fails.
        """
        db_url = os.getenv(self._db_url_env_var)
        token = self._extract_token_from_db_url(db_url)
        expiration_time = self._parse_token_expiration(token)

        if expiration_time is None:
            # If we can't parse the token, use fallback interval
            verbose_proxy_logger.debug(
                f"Could not parse token expiration, using fallback interval of "
                f"{self.FALLBACK_REFRESH_INTERVAL_SECONDS}s"
            )
            return self.FALLBACK_REFRESH_INTERVAL_SECONDS

        # Calculate when we should refresh (expiration - buffer)
        refresh_at = expiration_time - timedelta(
            seconds=self.TOKEN_REFRESH_BUFFER_SECONDS
        )

        # How long until refresh time?
        now = datetime.utcnow()
        seconds_until_refresh = (refresh_at - now).total_seconds()

        # If already past refresh time, return 0 (refresh immediately)
        return max(0, seconds_until_refresh)

    def is_token_expired(self, token_url: Optional[str]) -> bool:
        """Check if the token in the given URL is expired."""
        if token_url is None:
            return True

        token = self._extract_token_from_db_url(token_url)
        expiration_time = self._parse_token_expiration(token)

        if expiration_time is None:
            # If we can't parse the token, assume it's expired to trigger refresh
            verbose_proxy_logger.debug(
                "Could not parse token expiration, treating as expired"
            )
            return True

        return datetime.utcnow() > expiration_time

    def get_rds_iam_token(self) -> Optional[str]:
        """Generate a new RDS IAM token and update the configured DB URL env var.

        When the wrapper was constructed with an explicit `iam_endpoint`
        (typical for a reader wrapper whose host/port/user came from a parsed
        URL), use that. Otherwise fall back to the legacy DATABASE_HOST/PORT/
        USER/NAME/SCHEMA env vars (writer behavior).
        """
        if not self.iam_token_db_auth:
            return None

        from litellm.proxy.auth.rds_iam_token import generate_iam_auth_token

        if self._iam_endpoint is not None:
            endpoint = self._iam_endpoint
            token = generate_iam_auth_token(
                db_host=endpoint.host, db_port=endpoint.port, db_user=endpoint.user
            )
            _db_url = endpoint.build_url(token)
        else:
            db_host = os.getenv("DATABASE_HOST")
            # Default to the Postgres standard port; passing None to
            # `generate_iam_auth_token` makes botocore embed the literal
            # string "None" in the presigned URL, which then fails to parse.
            db_port = os.getenv("DATABASE_PORT", "5432")
            db_user = os.getenv("DATABASE_USER")
            db_name = os.getenv("DATABASE_NAME")
            db_schema = os.getenv("DATABASE_SCHEMA")

            token = generate_iam_auth_token(
                db_host=db_host, db_port=db_port, db_user=db_user
            )

            _db_url = f"postgresql://{db_user}:{token}@{db_host}:{db_port}/{db_name}"
            if db_schema:
                _db_url += f"?schema={db_schema}"

        os.environ[self._db_url_env_var] = _db_url
        return _db_url

    async def recreate_prisma_client(
        self, new_db_url: str, http_client: Optional[Any] = None
    ):
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
        """
        from prisma import Prisma  # type: ignore

        old_engine_pid = self._get_engine_pid()
        if old_engine_pid > 0:
            await self._kill_engine_process(old_engine_pid)

        kwargs: Dict[str, Any] = {}
        if http_client is not None:
            kwargs["http"] = http_client
        if self._recreate_uses_datasource:
            kwargs["datasource"] = {"url": new_db_url}
        self._original_prisma = Prisma(**kwargs)

        await self._original_prisma.connect()

    async def start_token_refresh_task(self) -> None:
        """
        Start the background token refresh task.

        This task proactively refreshes RDS IAM tokens before they expire,
        preventing connection failures. Should be called after the initial
        Prisma client connection is established.
        """
        if not self.iam_token_db_auth:
            verbose_proxy_logger.debug(
                "IAM token auth not enabled, skipping token refresh task"
            )
            return

        if self._token_refresh_task is not None:
            verbose_proxy_logger.debug("Token refresh task already running")
            return

        self._token_refresh_task = asyncio.create_task(self._token_refresh_loop())
        verbose_proxy_logger.info(
            "%sStarted RDS IAM token proactive refresh background task",
            self._log_prefix,
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
            "%sStopped RDS IAM token refresh background task", self._log_prefix
        )

    async def _token_refresh_loop(self) -> None:
        """
        Background loop that proactively refreshes RDS IAM tokens before expiration.

        Uses precise timing: calculates the exact sleep duration until the token
        needs to be refreshed (expiration - 3 minute buffer), then refreshes.
        This is more efficient than polling, requiring only 1 wake-up per token cycle.
        """
        verbose_proxy_logger.info(
            f"{self._log_prefix}RDS IAM token refresh loop started. "
            f"Tokens will be refreshed {self.TOKEN_REFRESH_BUFFER_SECONDS}s before expiration."
        )

        while True:
            try:
                # Calculate exactly how long to sleep until next refresh
                sleep_seconds = self._calculate_seconds_until_refresh()

                if sleep_seconds > 0:
                    verbose_proxy_logger.info(
                        f"{self._log_prefix}RDS IAM token refresh scheduled in "
                        f"{sleep_seconds:.0f} seconds ({sleep_seconds / 60:.1f} minutes)"
                    )
                    await asyncio.sleep(sleep_seconds)

                # Refresh the token
                verbose_proxy_logger.info(
                    "%sProactively refreshing RDS IAM token...", self._log_prefix
                )
                await self._safe_refresh_token()

            except asyncio.CancelledError:
                verbose_proxy_logger.info(
                    "%sRDS IAM token refresh loop cancelled", self._log_prefix
                )
                break
            except Exception as e:
                verbose_proxy_logger.error(
                    f"{self._log_prefix}Error in RDS IAM token refresh loop: {e}. "
                    f"Retrying in {self.FALLBACK_REFRESH_INTERVAL_SECONDS}s..."
                )
                # On error, wait before retrying to avoid tight error loops
                try:
                    await asyncio.sleep(self.FALLBACK_REFRESH_INTERVAL_SECONDS)
                except asyncio.CancelledError:
                    break

    async def _safe_refresh_token(self) -> None:
        """
        Refresh the RDS IAM token with proper locking to prevent race conditions.

        Uses an asyncio lock to ensure only one refresh operation happens at a time,
        preventing multiple concurrent reconnection attempts.
        """
        async with self._reconnection_lock:
            new_db_url = self.get_rds_iam_token()
            if new_db_url:
                await self.recreate_prisma_client(new_db_url)
                self._last_refresh_time = datetime.utcnow()
                verbose_proxy_logger.info(
                    "%sRDS IAM token refreshed successfully. New token valid for ~15 minutes.",
                    self._log_prefix,
                )
            else:
                verbose_proxy_logger.error(
                    "%sFailed to generate new RDS IAM token during proactive refresh",
                    self._log_prefix,
                )

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
            db_url = os.getenv(self._db_url_env_var)

            # Check if token is expired (should be rare if background task is running)
            if self.is_token_expired(db_url):
                try:
                    running_loop = asyncio.get_running_loop()
                except RuntimeError:
                    running_loop = None

                if running_loop is not None:
                    verbose_proxy_logger.warning(
                        "%sRDS IAM token expired in __getattr__ — proactive refresh "
                        "may have failed. Scheduling async refresh; the current "
                        "request may fail and be retried with the fresh token.",
                        self._log_prefix,
                    )
                    # Non-blocking: schedule the locked refresh on the
                    # running loop. The reconnection lock inside
                    # `_safe_refresh_token` coalesces concurrent triggers.
                    running_loop.create_task(self._safe_refresh_token())
                else:
                    verbose_proxy_logger.warning(
                        "%sRDS IAM token expired in __getattr__ — proactive refresh "
                        "may have failed. Triggering synchronous fallback refresh...",
                        self._log_prefix,
                    )
                    new_db_url = self.get_rds_iam_token()
                    if new_db_url:
                        asyncio.run(self.recreate_prisma_client(new_db_url))
                        # Re-fetch attribute against the recreated Prisma instance.
                        original_attr = getattr(self._original_prisma, name)
                        verbose_proxy_logger.info(
                            "%sSynchronous token refresh completed successfully",
                            self._log_prefix,
                        )
                    else:
                        raise ValueError("Failed to get RDS IAM token")

        return original_attr


class PrismaManager:
    @staticmethod
    def _get_prisma_dir() -> str:
        """Get the path to the migrations directory"""
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(os.path.dirname(abspath))
        return dname

    @staticmethod
    def setup_database(
        use_migrate: bool = False, use_v2_resolver: bool = False
    ) -> bool:
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
                        verbose_proxy_logger.error(
                            f"\033[1;31mLiteLLM: Failed to import proxy extras. Got {e}\033[0m"
                        )
                        return False

                    prisma_dir = PrismaManager._get_prisma_dir()

                    return ProxyExtrasDBManager.setup_database(
                        use_migrate=use_migrate,
                        use_v2_resolver=use_v2_resolver,
                    )
                else:
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
                    return True
            except subprocess.TimeoutExpired:
                verbose_proxy_logger.warning(f"Attempt {attempt + 1} timed out")
                time.sleep(random.randrange(5, 15))
            except subprocess.CalledProcessError as e:
                attempts_left = 3 - attempt
                retry_msg = (
                    f" Retrying... ({attempts_left} attempts left)"
                    if attempts_left > 0
                    else ""
                )
                verbose_proxy_logger.warning(
                    f"The process failed to execute. Details: {e}.{retry_msg}"
                )
                time.sleep(random.randrange(5, 15))
            finally:
                os.chdir(original_dir)
        return False


def should_update_prisma_schema(
    disable_updates: Optional[Union[bool, str]] = None,
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
