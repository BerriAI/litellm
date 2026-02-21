"""
This file contains the PrismaWrapper class, which is used to wrap the Prisma client and handle the RDS IAM token.
"""

import asyncio
import os
import random
import subprocess
import time
import urllib
import urllib.parse
from datetime import datetime, timedelta
from typing import Any, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.secret_managers.main import str_to_bool


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

    def __init__(self, original_prisma: Any, iam_token_db_auth: bool):
        self._original_prisma = original_prisma
        self.iam_token_db_auth = iam_token_db_auth

        # Background token refresh task management
        self._token_refresh_task: Optional[asyncio.Task] = None
        self._reconnection_lock = asyncio.Lock()
        self._last_refresh_time: Optional[datetime] = None

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
        db_url = os.getenv("DATABASE_URL")
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
        """Generate a new RDS IAM token and update DATABASE_URL."""
        if self.iam_token_db_auth:
            from litellm.proxy.auth.rds_iam_token import generate_iam_auth_token

            db_host = os.getenv("DATABASE_HOST")
            db_port = os.getenv("DATABASE_PORT")
            db_user = os.getenv("DATABASE_USER")
            db_name = os.getenv("DATABASE_NAME")
            db_schema = os.getenv("DATABASE_SCHEMA")

            token = generate_iam_auth_token(
                db_host=db_host, db_port=db_port, db_user=db_user
            )

            _db_url = f"postgresql://{db_user}:{token}@{db_host}:{db_port}/{db_name}"
            if db_schema:
                _db_url += f"?schema={db_schema}"

            os.environ["DATABASE_URL"] = _db_url
            return _db_url
        return None

    async def recreate_prisma_client(
        self, new_db_url: str, http_client: Optional[Any] = None
    ):
        """Disconnect and reconnect the Prisma client with a new database URL."""
        from prisma import Prisma  # type: ignore

        try:
            await self._original_prisma.disconnect()
        except Exception as e:
            verbose_proxy_logger.warning(f"Failed to disconnect Prisma client: {e}")

        if http_client is not None:
            self._original_prisma = Prisma(http=http_client)
        else:
            self._original_prisma = Prisma()

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
            "Started RDS IAM token proactive refresh background task"
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
        verbose_proxy_logger.info("Stopped RDS IAM token refresh background task")

    async def _token_refresh_loop(self) -> None:
        """
        Background loop that proactively refreshes RDS IAM tokens before expiration.

        Uses precise timing: calculates the exact sleep duration until the token
        needs to be refreshed (expiration - 3 minute buffer), then refreshes.
        This is more efficient than polling, requiring only 1 wake-up per token cycle.
        """
        verbose_proxy_logger.info(
            f"RDS IAM token refresh loop started. "
            f"Tokens will be refreshed {self.TOKEN_REFRESH_BUFFER_SECONDS}s before expiration."
        )

        while True:
            try:
                # Calculate exactly how long to sleep until next refresh
                sleep_seconds = self._calculate_seconds_until_refresh()

                if sleep_seconds > 0:
                    verbose_proxy_logger.info(
                        f"RDS IAM token refresh scheduled in {sleep_seconds:.0f} seconds "
                        f"({sleep_seconds / 60:.1f} minutes)"
                    )
                    await asyncio.sleep(sleep_seconds)

                # Refresh the token
                verbose_proxy_logger.info("Proactively refreshing RDS IAM token...")
                await self._safe_refresh_token()

            except asyncio.CancelledError:
                verbose_proxy_logger.info("RDS IAM token refresh loop cancelled")
                break
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Error in RDS IAM token refresh loop: {e}. "
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
                    "RDS IAM token refreshed successfully. New token valid for ~15 minutes."
                )
            else:
                verbose_proxy_logger.error(
                    "Failed to generate new RDS IAM token during proactive refresh"
                )

    def __getattr__(self, name: str):
        """
        Proxy attribute access to the underlying Prisma client.

        If IAM token auth is enabled and the token is expired, this method
        provides a synchronous fallback to refresh the token. However, this
        should rarely be needed since the background task proactively refreshes
        tokens before they expire.

        FIXED: Now properly waits for reconnection to complete before returning,
        instead of the previous fire-and-forget pattern that caused the bug.
        """
        original_attr = getattr(self._original_prisma, name)

        if self.iam_token_db_auth:
            db_url = os.getenv("DATABASE_URL")

            # Check if token is expired (should be rare if background task is running)
            if self.is_token_expired(db_url):
                verbose_proxy_logger.warning(
                    "RDS IAM token expired in __getattr__ - proactive refresh may have failed. "
                    "Triggering synchronous fallback refresh..."
                )

                new_db_url = self.get_rds_iam_token()
                if new_db_url:
                    loop = asyncio.get_event_loop()

                    if loop.is_running():
                        # FIXED: Actually wait for the reconnection to complete!
                        # The previous code used fire-and-forget which caused the bug.
                        future = asyncio.run_coroutine_threadsafe(
                            self.recreate_prisma_client(new_db_url), loop
                        )
                        try:
                            # Wait up to 30 seconds for reconnection
                            future.result(timeout=30)
                            verbose_proxy_logger.info(
                                "Synchronous token refresh completed successfully"
                            )
                        except Exception as e:
                            verbose_proxy_logger.error(
                                f"Failed to refresh token synchronously: {e}"
                            )
                            raise
                    else:
                        asyncio.run(self.recreate_prisma_client(new_db_url))

                    # Get the NEW attribute after reconnection
                    original_attr = getattr(self._original_prisma, name)
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
    def setup_database(use_migrate: bool = False) -> bool:
        """
        Set up the database using either prisma migrate or prisma db push

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

                    return ProxyExtrasDBManager.setup_database(use_migrate=use_migrate)
                else:
                    # Use prisma db push with increased timeout
                    subprocess.run(
                        ["prisma", "db", "push", "--accept-data-loss"],
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
