from sentry_sdk.consts import SPANSTATUS, SPANDATA
from sentry_sdk.integrations import _check_minimum_version, Integration, DidNotEnable
from sentry_sdk.tracing_utils import add_query_source, record_sql_queries
from sentry_sdk.utils import (
    capture_internal_exceptions,
    ensure_integration_enabled,
    parse_version,
)

try:
    from sqlalchemy.engine import Engine  # type: ignore
    from sqlalchemy.event import listen  # type: ignore
    from sqlalchemy import __version__ as SQLALCHEMY_VERSION  # type: ignore
except ImportError:
    raise DidNotEnable("SQLAlchemy not installed.")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import ContextManager
    from typing import Optional

    from sentry_sdk.tracing import Span


class SqlalchemyIntegration(Integration):
    identifier = "sqlalchemy"
    origin = f"auto.db.{identifier}"

    @staticmethod
    def setup_once():
        # type: () -> None
        version = parse_version(SQLALCHEMY_VERSION)
        _check_minimum_version(SqlalchemyIntegration, version)

        listen(Engine, "before_cursor_execute", _before_cursor_execute)
        listen(Engine, "after_cursor_execute", _after_cursor_execute)
        listen(Engine, "handle_error", _handle_error)


@ensure_integration_enabled(SqlalchemyIntegration)
def _before_cursor_execute(
    conn, cursor, statement, parameters, context, executemany, *args
):
    # type: (Any, Any, Any, Any, Any, bool, *Any) -> None
    ctx_mgr = record_sql_queries(
        cursor,
        statement,
        parameters,
        paramstyle=context and context.dialect and context.dialect.paramstyle or None,
        executemany=executemany,
        span_origin=SqlalchemyIntegration.origin,
    )
    context._sentry_sql_span_manager = ctx_mgr

    span = ctx_mgr.__enter__()

    if span is not None:
        _set_db_data(span, conn)
        context._sentry_sql_span = span


@ensure_integration_enabled(SqlalchemyIntegration)
def _after_cursor_execute(conn, cursor, statement, parameters, context, *args):
    # type: (Any, Any, Any, Any, Any, *Any) -> None
    ctx_mgr = getattr(
        context, "_sentry_sql_span_manager", None
    )  # type: Optional[ContextManager[Any]]

    if ctx_mgr is not None:
        context._sentry_sql_span_manager = None
        ctx_mgr.__exit__(None, None, None)

    span = getattr(context, "_sentry_sql_span", None)  # type: Optional[Span]
    if span is not None:
        with capture_internal_exceptions():
            add_query_source(span)


def _handle_error(context, *args):
    # type: (Any, *Any) -> None
    execution_context = context.execution_context
    if execution_context is None:
        return

    span = getattr(execution_context, "_sentry_sql_span", None)  # type: Optional[Span]

    if span is not None:
        span.set_status(SPANSTATUS.INTERNAL_ERROR)

    # _after_cursor_execute does not get called for crashing SQL stmts. Judging
    # from SQLAlchemy codebase it does seem like any error coming into this
    # handler is going to be fatal.
    ctx_mgr = getattr(
        execution_context, "_sentry_sql_span_manager", None
    )  # type: Optional[ContextManager[Any]]

    if ctx_mgr is not None:
        execution_context._sentry_sql_span_manager = None
        ctx_mgr.__exit__(None, None, None)


# See: https://docs.sqlalchemy.org/en/20/dialects/index.html
def _get_db_system(name):
    # type: (str) -> Optional[str]
    name = str(name)

    if "sqlite" in name:
        return "sqlite"

    if "postgres" in name:
        return "postgresql"

    if "mariadb" in name:
        return "mariadb"

    if "mysql" in name:
        return "mysql"

    if "oracle" in name:
        return "oracle"

    return None


def _set_db_data(span, conn):
    # type: (Span, Any) -> None
    db_system = _get_db_system(conn.engine.name)
    if db_system is not None:
        span.set_data(SPANDATA.DB_SYSTEM, db_system)

    if conn.engine.url is None:
        return

    db_name = conn.engine.url.database
    if db_name is not None:
        span.set_data(SPANDATA.DB_NAME, db_name)

    server_address = conn.engine.url.host
    if server_address is not None:
        span.set_data(SPANDATA.SERVER_ADDRESS, server_address)

    server_port = conn.engine.url.port
    if server_port is not None:
        span.set_data(SPANDATA.SERVER_PORT, server_port)
