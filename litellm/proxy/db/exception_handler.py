from typing import Union

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
        """
        Returns True if the exception is from a database outage / connection error
        """
        import prisma

        if isinstance(e, DB_CONNECTION_ERROR_TYPES):
            return True
        if isinstance(e, prisma.errors.PrismaError):
            return True
        if isinstance(e, ProxyException) and e.type == ProxyErrorTypes.no_db_connection:
            return True
        return False

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


def is_deadlock_error(e: Exception) -> bool:
    """
    Best-effort detector for Postgres deadlocks (SQLSTATE 40P01) coming through Prisma.

    We can't rely on a strongly-typed error from the Prisma Python client for this,
    so we match on the SQLSTATE code or message text.
    """
    message = str(e)
    if "40P01" in message:
        return True
    if "deadlock detected" in message.lower():
        return True
    try:
        import prisma  # type: ignore

        if isinstance(e, prisma.errors.PrismaError) and (
            "40P01" in message or "deadlock" in message.lower()
        ):
            return True
    except Exception:
        # If prisma isn't available or anything goes wrong, fall back to string checks
        pass
    return False
