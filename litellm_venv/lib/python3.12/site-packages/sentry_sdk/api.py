import inspect
import warnings
from contextlib import contextmanager

from sentry_sdk import tracing_utils, Client
from sentry_sdk._init_implementation import init
from sentry_sdk.consts import INSTRUMENTER
from sentry_sdk.scope import Scope, _ScopeManager, new_scope, isolation_scope
from sentry_sdk.tracing import NoOpSpan, Transaction, trace
from sentry_sdk.crons import monitor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing import Any
    from typing import Dict
    from typing import Generator
    from typing import Optional
    from typing import overload
    from typing import Callable
    from typing import TypeVar
    from typing import ContextManager
    from typing import Union

    from typing_extensions import Unpack

    from sentry_sdk.client import BaseClient
    from sentry_sdk._types import (
        Event,
        Hint,
        Breadcrumb,
        BreadcrumbHint,
        ExcInfo,
        MeasurementUnit,
        LogLevelStr,
        SamplingContext,
    )
    from sentry_sdk.tracing import Span, TransactionKwargs

    T = TypeVar("T")
    F = TypeVar("F", bound=Callable[..., Any])
else:

    def overload(x):
        # type: (T) -> T
        return x


# When changing this, update __all__ in __init__.py too
__all__ = [
    "init",
    "add_breadcrumb",
    "capture_event",
    "capture_exception",
    "capture_message",
    "configure_scope",
    "continue_trace",
    "flush",
    "get_baggage",
    "get_client",
    "get_global_scope",
    "get_isolation_scope",
    "get_current_scope",
    "get_current_span",
    "get_traceparent",
    "is_initialized",
    "isolation_scope",
    "last_event_id",
    "new_scope",
    "push_scope",
    "set_context",
    "set_extra",
    "set_level",
    "set_measurement",
    "set_tag",
    "set_tags",
    "set_user",
    "start_span",
    "start_transaction",
    "trace",
    "monitor",
]


def scopemethod(f):
    # type: (F) -> F
    f.__doc__ = "%s\n\n%s" % (
        "Alias for :py:meth:`sentry_sdk.Scope.%s`" % f.__name__,
        inspect.getdoc(getattr(Scope, f.__name__)),
    )
    return f


def clientmethod(f):
    # type: (F) -> F
    f.__doc__ = "%s\n\n%s" % (
        "Alias for :py:meth:`sentry_sdk.Client.%s`" % f.__name__,
        inspect.getdoc(getattr(Client, f.__name__)),
    )
    return f


@scopemethod
def get_client():
    # type: () -> BaseClient
    return Scope.get_client()


def is_initialized():
    # type: () -> bool
    """
    .. versionadded:: 2.0.0

    Returns whether Sentry has been initialized or not.

    If a client is available and the client is active
    (meaning it is configured to send data) then
    Sentry is initialized.
    """
    return get_client().is_active()


@scopemethod
def get_global_scope():
    # type: () -> Scope
    return Scope.get_global_scope()


@scopemethod
def get_isolation_scope():
    # type: () -> Scope
    return Scope.get_isolation_scope()


@scopemethod
def get_current_scope():
    # type: () -> Scope
    return Scope.get_current_scope()


@scopemethod
def last_event_id():
    # type: () -> Optional[str]
    """
    See :py:meth:`sentry_sdk.Scope.last_event_id` documentation regarding
    this method's limitations.
    """
    return Scope.last_event_id()


@scopemethod
def capture_event(
    event,  # type: Event
    hint=None,  # type: Optional[Hint]
    scope=None,  # type: Optional[Any]
    **scope_kwargs,  # type: Any
):
    # type: (...) -> Optional[str]
    return get_current_scope().capture_event(event, hint, scope=scope, **scope_kwargs)


@scopemethod
def capture_message(
    message,  # type: str
    level=None,  # type: Optional[LogLevelStr]
    scope=None,  # type: Optional[Any]
    **scope_kwargs,  # type: Any
):
    # type: (...) -> Optional[str]
    return get_current_scope().capture_message(
        message, level, scope=scope, **scope_kwargs
    )


@scopemethod
def capture_exception(
    error=None,  # type: Optional[Union[BaseException, ExcInfo]]
    scope=None,  # type: Optional[Any]
    **scope_kwargs,  # type: Any
):
    # type: (...) -> Optional[str]
    return get_current_scope().capture_exception(error, scope=scope, **scope_kwargs)


@scopemethod
def add_breadcrumb(
    crumb=None,  # type: Optional[Breadcrumb]
    hint=None,  # type: Optional[BreadcrumbHint]
    **kwargs,  # type: Any
):
    # type: (...) -> None
    return get_isolation_scope().add_breadcrumb(crumb, hint, **kwargs)


@overload
def configure_scope():
    # type: () -> ContextManager[Scope]
    pass


@overload
def configure_scope(  # noqa: F811
    callback,  # type: Callable[[Scope], None]
):
    # type: (...) -> None
    pass


def configure_scope(  # noqa: F811
    callback=None,  # type: Optional[Callable[[Scope], None]]
):
    # type: (...) -> Optional[ContextManager[Scope]]
    """
    Reconfigures the scope.

    :param callback: If provided, call the callback with the current scope.

    :returns: If no callback is provided, returns a context manager that returns the scope.
    """
    warnings.warn(
        "sentry_sdk.configure_scope is deprecated and will be removed in the next major version. "
        "Please consult our migration guide to learn how to migrate to the new API: "
        "https://docs.sentry.io/platforms/python/migration/1.x-to-2.x#scope-configuring",
        DeprecationWarning,
        stacklevel=2,
    )

    scope = get_isolation_scope()
    scope.generate_propagation_context()

    if callback is not None:
        # TODO: used to return None when client is None. Check if this changes behavior.
        callback(scope)

        return None

    @contextmanager
    def inner():
        # type: () -> Generator[Scope, None, None]
        yield scope

    return inner()


@overload
def push_scope():
    # type: () -> ContextManager[Scope]
    pass


@overload
def push_scope(  # noqa: F811
    callback,  # type: Callable[[Scope], None]
):
    # type: (...) -> None
    pass


def push_scope(  # noqa: F811
    callback=None,  # type: Optional[Callable[[Scope], None]]
):
    # type: (...) -> Optional[ContextManager[Scope]]
    """
    Pushes a new layer on the scope stack.

    :param callback: If provided, this method pushes a scope, calls
        `callback`, and pops the scope again.

    :returns: If no `callback` is provided, a context manager that should
        be used to pop the scope again.
    """
    warnings.warn(
        "sentry_sdk.push_scope is deprecated and will be removed in the next major version. "
        "Please consult our migration guide to learn how to migrate to the new API: "
        "https://docs.sentry.io/platforms/python/migration/1.x-to-2.x#scope-pushing",
        DeprecationWarning,
        stacklevel=2,
    )

    if callback is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with push_scope() as scope:
                callback(scope)
        return None

    return _ScopeManager()


@scopemethod
def set_tag(key, value):
    # type: (str, Any) -> None
    return get_isolation_scope().set_tag(key, value)


@scopemethod
def set_tags(tags):
    # type: (Mapping[str, object]) -> None
    return get_isolation_scope().set_tags(tags)


@scopemethod
def set_context(key, value):
    # type: (str, Dict[str, Any]) -> None
    return get_isolation_scope().set_context(key, value)


@scopemethod
def set_extra(key, value):
    # type: (str, Any) -> None
    return get_isolation_scope().set_extra(key, value)


@scopemethod
def set_user(value):
    # type: (Optional[Dict[str, Any]]) -> None
    return get_isolation_scope().set_user(value)


@scopemethod
def set_level(value):
    # type: (LogLevelStr) -> None
    return get_isolation_scope().set_level(value)


@clientmethod
def flush(
    timeout=None,  # type: Optional[float]
    callback=None,  # type: Optional[Callable[[int, float], None]]
):
    # type: (...) -> None
    return get_client().flush(timeout=timeout, callback=callback)


@scopemethod
def start_span(
    **kwargs,  # type: Any
):
    # type: (...) -> Span
    return get_current_scope().start_span(**kwargs)


@scopemethod
def start_transaction(
    transaction=None,  # type: Optional[Transaction]
    instrumenter=INSTRUMENTER.SENTRY,  # type: str
    custom_sampling_context=None,  # type: Optional[SamplingContext]
    **kwargs,  # type: Unpack[TransactionKwargs]
):
    # type: (...) -> Union[Transaction, NoOpSpan]
    """
    Start and return a transaction on the current scope.

    Start an existing transaction if given, otherwise create and start a new
    transaction with kwargs.

    This is the entry point to manual tracing instrumentation.

    A tree structure can be built by adding child spans to the transaction,
    and child spans to other spans. To start a new child span within the
    transaction or any span, call the respective `.start_child()` method.

    Every child span must be finished before the transaction is finished,
    otherwise the unfinished spans are discarded.

    When used as context managers, spans and transactions are automatically
    finished at the end of the `with` block. If not using context managers,
    call the `.finish()` method.

    When the transaction is finished, it will be sent to Sentry with all its
    finished child spans.

    :param transaction: The transaction to start. If omitted, we create and
        start a new transaction.
    :param instrumenter: This parameter is meant for internal use only. It
        will be removed in the next major version.
    :param custom_sampling_context: The transaction's custom sampling context.
    :param kwargs: Optional keyword arguments to be passed to the Transaction
        constructor. See :py:class:`sentry_sdk.tracing.Transaction` for
        available arguments.
    """
    return get_current_scope().start_transaction(
        transaction, instrumenter, custom_sampling_context, **kwargs
    )


def set_measurement(name, value, unit=""):
    # type: (str, float, MeasurementUnit) -> None
    transaction = get_current_scope().transaction
    if transaction is not None:
        transaction.set_measurement(name, value, unit)


def get_current_span(scope=None):
    # type: (Optional[Scope]) -> Optional[Span]
    """
    Returns the currently active span if there is one running, otherwise `None`
    """
    return tracing_utils.get_current_span(scope)


def get_traceparent():
    # type: () -> Optional[str]
    """
    Returns the traceparent either from the active span or from the scope.
    """
    return get_current_scope().get_traceparent()


def get_baggage():
    # type: () -> Optional[str]
    """
    Returns Baggage either from the active span or from the scope.
    """
    baggage = get_current_scope().get_baggage()
    if baggage is not None:
        return baggage.serialize()

    return None


def continue_trace(
    environ_or_headers, op=None, name=None, source=None, origin="manual"
):
    # type: (Dict[str, Any], Optional[str], Optional[str], Optional[str], str) -> Transaction
    """
    Sets the propagation context from environment or headers and returns a transaction.
    """
    return get_isolation_scope().continue_trace(
        environ_or_headers, op, name, source, origin
    )
