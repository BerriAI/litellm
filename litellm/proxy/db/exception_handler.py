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
        Returns True if the exception is from a database outage / connection error.
        Any PrismaError qualifies — the DB failed to serve the request.
        Used by allow_requests_on_db_unavailable logic and endpoint 503 responses.
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
            e, (prisma.errors.ClientNotConnectedError, prisma.errors.HTTPClientClosedError)
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
