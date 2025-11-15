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
    def is_database_retriable_exception(e: Exception) -> bool:
        """
        Returns True if the execption is from a condition (e.g. deadlock, broken connection, etc.) that should be retried.
        """
        import re

        if isinstance(e, DB_CONNECTION_ERROR_TYPES): # TODO: is this actually needed?
            return True
        
        # Deadlocks should normally be retried.
        # Postgres right now, on deadlock, triggers an exception similar to:
        #   Error occurred during query execution: ConnectorError(ConnectorError { user_facing_error: None, 
        #   kind: QueryError(PostgresError { code: "40P01", message: "deadlock detected", severity: "ERROR", 
        #   detail: Some("Process 3753505 waits for ShareLock on transaction 5729447; blocked by process 3755128.\n
        #   Process 3755128 waits for ShareLock on transaction 5729448; blocked by process 3753505."), column: None, 
        #   hint: Some("See server log for query details.") }), transient: false })
        # Unfortunately there does not seem to be a easy way to properly parse that or otherwise detect the specific
        # issue, so just match using a regular expression. This is definitely not ideal, but not much we can do about
        # it.
        if re.search(r'\bConnectorError\b.*?\bQueryError\b.*?\bPostgresError\b.*?"40P01"', str(e), re.DOTALL):
            return True

        # TODO: add additional specific cases (be careful to not add exceptions that should not be retried!)
        # If many more additional regular expressions are added, it may make sense to combine them into a single one,
        # or use something like hyperscan.

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
