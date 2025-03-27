from litellm.proxy._types import (
    DB_CONNECTION_ERROR_TYPES,
    ProxyErrorTypes,
    ProxyException,
)


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

        if general_settings.get("allow_requests_on_db_unavailable", False) is True:
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
