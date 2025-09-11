import sentry_sdk
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.integrations import _check_minimum_version, Integration, DidNotEnable
from sentry_sdk.tracing import Span
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.utils import capture_internal_exceptions, ensure_integration_enabled

from typing import TYPE_CHECKING, TypeVar

# Hack to get new Python features working in older versions
# without introducing a hard dependency on `typing_extensions`
# from: https://stackoverflow.com/a/71944042/300572
if TYPE_CHECKING:
    from typing import ParamSpec, Callable
else:
    # Fake ParamSpec
    class ParamSpec:
        def __init__(self, _):
            self.args = None
            self.kwargs = None

    # Callable[anything] will return None
    class _Callable:
        def __getitem__(self, _):
            return None

    # Make instances
    Callable = _Callable()


try:
    import clickhouse_driver  # type: ignore[import-not-found]

except ImportError:
    raise DidNotEnable("clickhouse-driver not installed.")


class ClickhouseDriverIntegration(Integration):
    identifier = "clickhouse_driver"
    origin = f"auto.db.{identifier}"

    @staticmethod
    def setup_once() -> None:
        _check_minimum_version(ClickhouseDriverIntegration, clickhouse_driver.VERSION)

        # Every query is done using the Connection's `send_query` function
        clickhouse_driver.connection.Connection.send_query = _wrap_start(
            clickhouse_driver.connection.Connection.send_query
        )

        # If the query contains parameters then the send_data function is used to send those parameters to clickhouse
        clickhouse_driver.client.Client.send_data = _wrap_send_data(
            clickhouse_driver.client.Client.send_data
        )

        # Every query ends either with the Client's `receive_end_of_query` (no result expected)
        # or its `receive_result` (result expected)
        clickhouse_driver.client.Client.receive_end_of_query = _wrap_end(
            clickhouse_driver.client.Client.receive_end_of_query
        )
        if hasattr(clickhouse_driver.client.Client, "receive_end_of_insert_query"):
            # In 0.2.7, insert queries are handled separately via `receive_end_of_insert_query`
            clickhouse_driver.client.Client.receive_end_of_insert_query = _wrap_end(
                clickhouse_driver.client.Client.receive_end_of_insert_query
            )
        clickhouse_driver.client.Client.receive_result = _wrap_end(
            clickhouse_driver.client.Client.receive_result
        )


P = ParamSpec("P")
T = TypeVar("T")


def _wrap_start(f: Callable[P, T]) -> Callable[P, T]:
    @ensure_integration_enabled(ClickhouseDriverIntegration, f)
    def _inner(*args: P.args, **kwargs: P.kwargs) -> T:
        connection = args[0]
        query = args[1]
        query_id = args[2] if len(args) > 2 else kwargs.get("query_id")
        params = args[3] if len(args) > 3 else kwargs.get("params")

        span = sentry_sdk.start_span(
            op=OP.DB,
            name=query,
            origin=ClickhouseDriverIntegration.origin,
        )

        connection._sentry_span = span  # type: ignore[attr-defined]

        _set_db_data(span, connection)

        span.set_data("query", query)

        if query_id:
            span.set_data("db.query_id", query_id)

        if params and should_send_default_pii():
            span.set_data("db.params", params)

        # run the original code
        ret = f(*args, **kwargs)

        return ret

    return _inner


def _wrap_end(f: Callable[P, T]) -> Callable[P, T]:
    def _inner_end(*args: P.args, **kwargs: P.kwargs) -> T:
        res = f(*args, **kwargs)
        instance = args[0]
        span = getattr(instance.connection, "_sentry_span", None)  # type: ignore[attr-defined]

        if span is not None:
            if res is not None and should_send_default_pii():
                span.set_data("db.result", res)

            with capture_internal_exceptions():
                span.scope.add_breadcrumb(
                    message=span._data.pop("query"), category="query", data=span._data
                )

            span.finish()

        return res

    return _inner_end


def _wrap_send_data(f: Callable[P, T]) -> Callable[P, T]:
    def _inner_send_data(*args: P.args, **kwargs: P.kwargs) -> T:
        instance = args[0]  # type: clickhouse_driver.client.Client
        data = args[2]
        span = getattr(instance.connection, "_sentry_span", None)

        if span is not None:
            _set_db_data(span, instance.connection)

            if should_send_default_pii():
                db_params = span._data.get("db.params", [])
                db_params.extend(data)
                span.set_data("db.params", db_params)

        return f(*args, **kwargs)

    return _inner_send_data


def _set_db_data(
    span: Span, connection: clickhouse_driver.connection.Connection
) -> None:
    span.set_data(SPANDATA.DB_SYSTEM, "clickhouse")
    span.set_data(SPANDATA.SERVER_ADDRESS, connection.host)
    span.set_data(SPANDATA.SERVER_PORT, connection.port)
    span.set_data(SPANDATA.DB_NAME, connection.database)
    span.set_data(SPANDATA.DB_USER, connection.user)
