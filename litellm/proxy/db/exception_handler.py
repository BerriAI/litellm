from typing import Any, Awaitable, Callable, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    DB_CONNECTION_ERROR_TYPES,
    ProxyErrorTypes,
    ProxyException,
)
from litellm.secret_managers.main import str_to_bool


class PrismaDBExceptionHandler:
    """
    Class to handle DB Exceptions or Connection Errors
    """

    @staticmethod
    def should_allow_request_on_db_unavailable() -> bool:
        """
        Returns True if the request should be allowed to proceed despite the DB connection error
        """
        from litellm.proxy.proxy_server import general_settings

        _allow_requests_on_db_unavailable: Union[bool, str] = general_settings.get(
            "allow_requests_on_db_unavailable", False
        )
        if isinstance(_allow_requests_on_db_unavailable, bool):
            return _allow_requests_on_db_unavailable
        if str_to_bool(_allow_requests_on_db_unavailable) is True:
            return True
        return False

    @staticmethod
    def is_database_connection_error(e: Exception) -> bool:
        """True iff the exception is (or could be) a DB-connectivity
        failure, i.e. something that justifies the
        ``allow_requests_on_db_unavailable`` HA fallback.

        Known data-layer PrismaError subclasses (``UniqueViolationError``,
        ``RecordNotFoundError``, etc.) are explicitly excluded — the DB IS
        reachable, so a fallback that grants an anonymous token would be
        an auth bypass. Unknown / bare ``PrismaError`` instances default
        to True so genuine outages that don't match a specific subclass
        still trigger the fallback.
        """
        import prisma

        # Explicit data-layer exclusion: DB IS reachable, fallback must
        # NOT fire.
        data_layer_errors = (
            prisma.errors.DataError,
            prisma.errors.UniqueViolationError,
            prisma.errors.ForeignKeyViolationError,
            prisma.errors.MissingRequiredValueError,
            prisma.errors.RawQueryError,
            prisma.errors.TableNotFoundError,
            prisma.errors.RecordNotFoundError,
        )
        if isinstance(e, data_layer_errors):
            return False
        if isinstance(e, DB_CONNECTION_ERROR_TYPES):
            return True
        if isinstance(e, prisma.errors.PrismaError):
            return True
        if isinstance(e, ProxyException) and e.type == ProxyErrorTypes.no_db_connection:
            return True
        return False

    @staticmethod
    def is_prisma_data_error(e: Exception) -> bool:
        """True iff ``e`` is a base prisma ``DataError``: the database processed
        the statement and refused the data itself (e.g. ``invalid byte sequence
        for encoding "UTF8": 0x00``), as opposed to a connectivity failure.

        Matched by exact type, not ``isinstance``: the specific data-layer
        subclasses (``UniqueViolationError``, ``TableNotFoundError``,
        ``MissingRequiredValueError`` ...) all derive from ``DataError`` but
        carry their own semantics, and a systemic one like a missing table must
        not be mistaken for a single poison row and bisected away. A raw
        Postgres execution error with no prisma P-code surfaces as the base
        ``DataError``.

        prisma also wraps the P1001 "can't reach database server" outage as a
        base ``DataError``, so a caller that must not treat an outage as a
        per-row data rejection has to additionally consult
        ``is_database_service_unavailable_error`` before acting on a True here.
        """
        import prisma

        return type(e) is prisma.errors.DataError

    @staticmethod
    def is_database_transport_error(e: Exception) -> bool:
        """
        Returns True only for transport/connectivity failures where a reconnect
        attempt makes sense (e.g. DB is unreachable, connection dropped).

        Use this for reconnect logic — data-layer errors like UniqueViolationError
        mean the DB IS reachable, so reconnecting would be pointless.
        """
        import prisma

        if isinstance(e, DB_CONNECTION_ERROR_TYPES):
            return True
        if isinstance(
            e,
            (
                prisma.errors.ClientNotConnectedError,
                prisma.errors.HTTPClientClosedError,
            ),
        ):
            return True
        if isinstance(e, prisma.errors.PrismaError):
            error_message = str(e).lower()
            connection_keywords = (
                "can't reach database server",
                "cannot reach database server",
                "can't connect",
                "cannot connect",
                "connection error",
                "connection closed",
                "timed out",
                "timeout",
                "connection refused",
                "network is unreachable",
                "no route to host",
                "broken pipe",
            )
            if any(keyword in error_message for keyword in connection_keywords):
                return True
        if isinstance(e, ProxyException) and e.type == ProxyErrorTypes.no_db_connection:
            return True
        return False

    @staticmethod
    def is_prisma_engine_internal_error(e: Exception) -> bool:
        """True iff ``e`` is a non-``PrismaError`` exception raised from inside
        prisma-client-py's query-engine layer.

        During the instant a DB connection is torn down, the query engine can
        return a malformed error payload (``user_facing_error.meta`` is
        ``null``). prisma-client-py's ``handle_response_errors`` then crashes
        with ``AttributeError: 'NoneType' object has no attribute 'get'``
        before it can raise the proper P1001 "can't reach database server"
        error. That AttributeError carries no connection keyword, so it can't
        be matched by message; identify it by its ``prisma.engine`` origin
        instead.

        Recognized ``PrismaError`` subclasses are excluded: connectivity ones
        are already classified by type/keyword above, and data-layer ones
        (the DB IS reachable) must stay 401.
        """
        import prisma

        if isinstance(e, prisma.errors.PrismaError):
            return False
        tb = getattr(e, "__traceback__", None)
        while tb is not None:
            if tb.tb_frame.f_globals.get("__name__", "").startswith("prisma.engine"):
                return True
            tb = tb.tb_next
        return False

    @staticmethod
    def is_database_service_unavailable_error(e: Exception) -> bool:
        """True iff the exception means the database could not answer at the
        infrastructure level (connection refused, socket/interface failure,
        timeout) rather than a genuine auth failure (key not found) or a
        data-layer error (the DB IS reachable and rejected the data).

        Auth must answer 401 only for a key the DB confirms is invalid. When
        the DB itself is unreachable, the request has to surface as 503 so
        callers retry instead of treating valid keys as invalid during an
        outage.

        Note: prisma-client-py mislabels the P1001 "can't reach database
        server" connectivity failure as a ``DataError`` (a data-layer type),
        so a type-only check misses real outages. ``is_database_transport_error``
        keyword-matches the connection message and catches that masquerade,
        while genuine data errors (no connection keyword) correctly stay 401.

        The Postgres "cached plan must not change result type" error is matched
        here, not in ``is_database_transport_error``: it is a transient stale-DB-
        state condition (not an invalid key), but the connection is healthy so it
        must not trigger a reconnect.

        A non-``PrismaError`` raised from inside the prisma query engine (e.g.
        the ``AttributeError`` from ``handle_response_errors`` when the engine
        returns a malformed error payload mid-tear-down) is also treated as
        unavailable; see ``is_prisma_engine_internal_error``.
        """
        import asyncio

        if PrismaDBExceptionHandler.is_database_connection_error(e):
            return True
        if PrismaDBExceptionHandler.is_database_transport_error(e):
            return True
        if PrismaDBExceptionHandler.is_prisma_engine_internal_error(e):
            return True
        if "cached plan must not change result type" in str(e).lower():
            return True

        # OSError already covers ConnectionError and (Py3.3+) TimeoutError.
        # asyncio.TimeoutError is a distinct class before Py3.11.
        if isinstance(e, (OSError, asyncio.TimeoutError)):
            return True

        try:
            import asyncpg
        except ImportError:
            return False

        return isinstance(
            e,
            (
                asyncpg.exceptions.PostgresConnectionError,
                asyncpg.exceptions.InterfaceError,
            ),
        )

    @staticmethod
    def handle_db_exception(e: Exception):
        """
        Primary handler for `allow_requests_on_db_unavailable` flag. Decides whether to raise a DB Exception or not based on the flag.

        - If exception is a DB Connection Error, and `allow_requests_on_db_unavailable` is True,
            - Do not raise an exception, return None
        - Else, raise the exception
        """
        if (
            PrismaDBExceptionHandler.is_database_connection_error(e)
            and PrismaDBExceptionHandler.should_allow_request_on_db_unavailable()
        ):
            return None
        raise e


# Default fallback timeouts when neither the caller nor the prisma_client
# expose `_db_auth_reconnect_timeout_seconds` / `_db_auth_reconnect_lock_timeout_seconds`.
# Match the auth path's existing defaults so behavior is uniform across read paths.
_DEFAULT_RECONNECT_TIMEOUT_SECONDS = 2.0
_DEFAULT_RECONNECT_LOCK_TIMEOUT_SECONDS = 0.1


def _coerce_timeout(value: Any, fallback: float) -> float:
    """Return `value` if it is a real int/float, else `fallback`. Guards
    against tests that mock `prisma_client` and leave the timeout slots as
    MagicMock instances."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return fallback


async def call_with_db_reconnect_retry(
    prisma_client: Any,
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    reason: str,
    timeout_seconds: Optional[float] = None,
    lock_timeout_seconds: Optional[float] = None,
) -> Any:
    """Run a Prisma read coroutine with one transport-reconnect-and-retry.

    The canonical "self-heal a transient DB transport blip" wrapper used by
    `PrismaClient.get_generic_data` and other read paths. Mirrors the inline
    pattern in `auth_checks._fetch_key_object_from_db_with_reconnect` so we
    have a single implementation rather than three drifting copies.

    Behavior:
      1. Await `coro_factory()`. On success, return its value.
      2. On exception, if it is NOT a transport error (per
         `is_database_transport_error`), re-raise — data-layer errors like
         `UniqueViolationError` mean the DB is reachable, reconnect would be
         pointless.
      3. If `prisma_client` does not expose `attempt_db_reconnect`, re-raise.
         This guards against partial stand-ins / older clients in tests.
      4. Call `prisma_client.attempt_db_reconnect(reason=...)`. If it returns
         False (cooldown / lock contention / reconnect failure), re-raise.
      5. Otherwise await `coro_factory()` a second time and return / propagate
         its result. At-most-one retry by construction — no infinite loop.

    `coro_factory` MUST be a zero-arg callable that returns a fresh awaitable
    on each call. Passing an already-awaited coroutine would fail on retry
    with `RuntimeError: cannot reuse already awaited coroutine`.

    `reason` should follow `<subsystem>_<operation>_<table>_failure` so
    telemetry distinguishes between fan-out callers (e.g.
    `_update_config_from_db` issues four concurrent reads).

    Args:
        prisma_client: The `PrismaClient` (or stand-in) that owns
            `attempt_db_reconnect` and the `_db_auth_reconnect_*` defaults.
        coro_factory: Zero-arg callable returning the read awaitable.
        reason: Telemetry tag forwarded to `attempt_db_reconnect`.
        timeout_seconds: Optional override for the reconnect cycle timeout.
            Defaults to `prisma_client._db_auth_reconnect_timeout_seconds`,
            then to 2.0s.
        lock_timeout_seconds: Optional override for how long the helper will
            wait to acquire the reconnect lock. Defaults to
            `prisma_client._db_auth_reconnect_lock_timeout_seconds`, then to
            0.1s.

    Returns:
        Whatever `coro_factory()` returns (on first or second attempt).

    Raises:
        Whatever `coro_factory()` raises if the failure is not a transport
        error, or if the reconnect attempt does not succeed, or if the retry
        also fails.
    """
    try:
        return await coro_factory()
    except Exception as first_exc:
        if not PrismaDBExceptionHandler.is_database_transport_error(first_exc):
            raise
        if not hasattr(prisma_client, "attempt_db_reconnect"):
            raise

        resolved_timeout = _coerce_timeout(
            (
                timeout_seconds
                if timeout_seconds is not None
                else getattr(prisma_client, "_db_auth_reconnect_timeout_seconds", None)
            ),
            _DEFAULT_RECONNECT_TIMEOUT_SECONDS,
        )
        resolved_lock_timeout = _coerce_timeout(
            (
                lock_timeout_seconds
                if lock_timeout_seconds is not None
                else getattr(prisma_client, "_db_auth_reconnect_lock_timeout_seconds", None)
            ),
            _DEFAULT_RECONNECT_LOCK_TIMEOUT_SECONDS,
        )

        verbose_proxy_logger.warning(
            "DB transport error on read; attempting reconnect-and-retry. reason=%s error=%s",
            reason,
            first_exc,
        )

        # Preserve the original transport error in telemetry. If
        # `attempt_db_reconnect` itself raises (e.g. lock cancellation, timer
        # error, unexpected internal failure), surfacing that exception
        # instead of `first_exc` would mask the actual DB transport problem
        # in `failure_handler` / `db_exceptions` alerts. Chain the reconnect
        # error as the cause for debuggability without losing the original.
        try:
            did_reconnect = await prisma_client.attempt_db_reconnect(
                reason=reason,
                timeout_seconds=resolved_timeout,
                lock_timeout_seconds=resolved_lock_timeout,
            )
        except Exception as reconnect_exc:
            verbose_proxy_logger.warning(
                "DB reconnect attempt raised; preserving original transport error. reason=%s reconnect_error=%s",
                reason,
                reconnect_exc,
            )
            raise first_exc from reconnect_exc
        if not did_reconnect:
            raise

        # At most one retry. If the retry also raises a transport error, we
        # propagate — repeated reconnect-loops are the watchdog's job, not
        # this helper's.
        return await coro_factory()
