import warnings
from contextlib import contextmanager

from sentry_sdk import (
    get_client,
    get_global_scope,
    get_isolation_scope,
    get_current_scope,
)
from sentry_sdk._compat import with_metaclass
from sentry_sdk.consts import INSTRUMENTER
from sentry_sdk.scope import _ScopeManager
from sentry_sdk.client import Client
from sentry_sdk.tracing import (
    NoOpSpan,
    Span,
    Transaction,
)

from sentry_sdk.utils import (
    logger,
    ContextVar,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import ContextManager
    from typing import Dict
    from typing import Generator
    from typing import List
    from typing import Optional
    from typing import overload
    from typing import Tuple
    from typing import Type
    from typing import TypeVar
    from typing import Union

    from typing_extensions import Unpack

    from sentry_sdk.scope import Scope
    from sentry_sdk.client import BaseClient
    from sentry_sdk.integrations import Integration
    from sentry_sdk._types import (
        Event,
        Hint,
        Breadcrumb,
        BreadcrumbHint,
        ExcInfo,
        LogLevelStr,
        SamplingContext,
    )
    from sentry_sdk.tracing import TransactionKwargs

    T = TypeVar("T")

else:

    def overload(x):
        # type: (T) -> T
        return x


class SentryHubDeprecationWarning(DeprecationWarning):
    """
    A custom deprecation warning to inform users that the Hub is deprecated.
    """

    _MESSAGE = (
        "`sentry_sdk.Hub` is deprecated and will be removed in a future major release. "
        "Please consult our 1.x to 2.x migration guide for details on how to migrate "
        "`Hub` usage to the new API: "
        "https://docs.sentry.io/platforms/python/migration/1.x-to-2.x"
    )

    def __init__(self, *_):
        # type: (*object) -> None
        super().__init__(self._MESSAGE)


@contextmanager
def _suppress_hub_deprecation_warning():
    # type: () -> Generator[None, None, None]
    """Utility function to suppress deprecation warnings for the Hub."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SentryHubDeprecationWarning)
        yield


_local = ContextVar("sentry_current_hub")


class HubMeta(type):
    @property
    def current(cls):
        # type: () -> Hub
        """Returns the current instance of the hub."""
        warnings.warn(SentryHubDeprecationWarning(), stacklevel=2)
        rv = _local.get(None)
        if rv is None:
            with _suppress_hub_deprecation_warning():
                # This will raise a deprecation warning; suppress it since we already warned above.
                rv = Hub(GLOBAL_HUB)
            _local.set(rv)
        return rv

    @property
    def main(cls):
        # type: () -> Hub
        """Returns the main instance of the hub."""
        warnings.warn(SentryHubDeprecationWarning(), stacklevel=2)
        return GLOBAL_HUB


class Hub(with_metaclass(HubMeta)):  # type: ignore
    """
    .. deprecated:: 2.0.0
        The Hub is deprecated. Its functionality will be merged into :py:class:`sentry_sdk.scope.Scope`.

    The hub wraps the concurrency management of the SDK.  Each thread has
    its own hub but the hub might transfer with the flow of execution if
    context vars are available.

    If the hub is used with a with statement it's temporarily activated.
    """

    _stack = None  # type: List[Tuple[Optional[Client], Scope]]
    _scope = None  # type: Optional[Scope]

    # Mypy doesn't pick up on the metaclass.

    if TYPE_CHECKING:
        current = None  # type: Hub
        main = None  # type: Hub

    def __init__(
        self,
        client_or_hub=None,  # type: Optional[Union[Hub, Client]]
        scope=None,  # type: Optional[Any]
    ):
        # type: (...) -> None
        warnings.warn(SentryHubDeprecationWarning(), stacklevel=2)

        current_scope = None

        if isinstance(client_or_hub, Hub):
            client = get_client()
            if scope is None:
                # hub cloning is going on, we use a fork of the current/isolation scope for context manager
                scope = get_isolation_scope().fork()
                current_scope = get_current_scope().fork()
        else:
            client = client_or_hub  # type: ignore
            get_global_scope().set_client(client)

        if scope is None:  # so there is no Hub cloning going on
            # just the current isolation scope is used for context manager
            scope = get_isolation_scope()
            current_scope = get_current_scope()

        if current_scope is None:
            # just the current current scope is used for context manager
            current_scope = get_current_scope()

        self._stack = [(client, scope)]  # type: ignore
        self._last_event_id = None  # type: Optional[str]
        self._old_hubs = []  # type: List[Hub]

        self._old_current_scopes = []  # type: List[Scope]
        self._old_isolation_scopes = []  # type: List[Scope]
        self._current_scope = current_scope  # type: Scope
        self._scope = scope  # type: Scope

    def __enter__(self):
        # type: () -> Hub
        self._old_hubs.append(Hub.current)
        _local.set(self)

        current_scope = get_current_scope()
        self._old_current_scopes.append(current_scope)
        scope._current_scope.set(self._current_scope)

        isolation_scope = get_isolation_scope()
        self._old_isolation_scopes.append(isolation_scope)
        scope._isolation_scope.set(self._scope)

        return self

    def __exit__(
        self,
        exc_type,  # type: Optional[type]
        exc_value,  # type: Optional[BaseException]
        tb,  # type: Optional[Any]
    ):
        # type: (...) -> None
        old = self._old_hubs.pop()
        _local.set(old)

        old_current_scope = self._old_current_scopes.pop()
        scope._current_scope.set(old_current_scope)

        old_isolation_scope = self._old_isolation_scopes.pop()
        scope._isolation_scope.set(old_isolation_scope)

    def run(
        self, callback  # type: Callable[[], T]
    ):
        # type: (...) -> T
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.

        Runs a callback in the context of the hub.  Alternatively the
        with statement can be used on the hub directly.
        """
        with self:
            return callback()

    def get_integration(
        self, name_or_class  # type: Union[str, Type[Integration]]
    ):
        # type: (...) -> Any
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.client._Client.get_integration` instead.

        Returns the integration for this hub by name or class.  If there
        is no client bound or the client does not have that integration
        then `None` is returned.

        If the return value is not `None` the hub is guaranteed to have a
        client attached.
        """
        return get_client().get_integration(name_or_class)

    @property
    def client(self):
        # type: () -> Optional[BaseClient]
        """
        .. deprecated:: 2.0.0
            This property is deprecated and will be removed in a future release.
            Please use :py:func:`sentry_sdk.api.get_client` instead.

        Returns the current client on the hub.
        """
        client = get_client()

        if not client.is_active():
            return None

        return client

    @property
    def scope(self):
        # type: () -> Scope
        """
        .. deprecated:: 2.0.0
            This property is deprecated and will be removed in a future release.
            Returns the current scope on the hub.
        """
        return get_isolation_scope()

    def last_event_id(self):
        # type: () -> Optional[str]
        """
        Returns the last event ID.

        .. deprecated:: 1.40.5
            This function is deprecated and will be removed in a future release. The functions `capture_event`, `capture_message`, and `capture_exception` return the event ID directly.
        """
        logger.warning(
            "Deprecated: last_event_id is deprecated. This will be removed in the future. The functions `capture_event`, `capture_message`, and `capture_exception` return the event ID directly."
        )
        return self._last_event_id

    def bind_client(
        self, new  # type: Optional[BaseClient]
    ):
        # type: (...) -> None
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.set_client` instead.

        Binds a new client to the hub.
        """
        get_global_scope().set_client(new)

    def capture_event(self, event, hint=None, scope=None, **scope_kwargs):
        # type: (Event, Optional[Hint], Optional[Scope], Any) -> Optional[str]
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.capture_event` instead.

        Captures an event.

        Alias of :py:meth:`sentry_sdk.Scope.capture_event`.

        :param event: A ready-made event that can be directly sent to Sentry.

        :param hint: Contains metadata about the event that can be read from `before_send`, such as the original exception object or a HTTP request object.

        :param scope: An optional :py:class:`sentry_sdk.Scope` to apply to events.
            The `scope` and `scope_kwargs` parameters are mutually exclusive.

        :param scope_kwargs: Optional data to apply to event.
            For supported `**scope_kwargs` see :py:meth:`sentry_sdk.Scope.update_from_kwargs`.
            The `scope` and `scope_kwargs` parameters are mutually exclusive.
        """
        last_event_id = get_current_scope().capture_event(
            event, hint, scope=scope, **scope_kwargs
        )

        is_transaction = event.get("type") == "transaction"
        if last_event_id is not None and not is_transaction:
            self._last_event_id = last_event_id

        return last_event_id

    def capture_message(self, message, level=None, scope=None, **scope_kwargs):
        # type: (str, Optional[LogLevelStr], Optional[Scope], Any) -> Optional[str]
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.capture_message` instead.

        Captures a message.

        Alias of :py:meth:`sentry_sdk.Scope.capture_message`.

        :param message: The string to send as the message to Sentry.

        :param level: If no level is provided, the default level is `info`.

        :param scope: An optional :py:class:`sentry_sdk.Scope` to apply to events.
            The `scope` and `scope_kwargs` parameters are mutually exclusive.

        :param scope_kwargs: Optional data to apply to event.
            For supported `**scope_kwargs` see :py:meth:`sentry_sdk.Scope.update_from_kwargs`.
            The `scope` and `scope_kwargs` parameters are mutually exclusive.

        :returns: An `event_id` if the SDK decided to send the event (see :py:meth:`sentry_sdk.client._Client.capture_event`).
        """
        last_event_id = get_current_scope().capture_message(
            message, level=level, scope=scope, **scope_kwargs
        )

        if last_event_id is not None:
            self._last_event_id = last_event_id

        return last_event_id

    def capture_exception(self, error=None, scope=None, **scope_kwargs):
        # type: (Optional[Union[BaseException, ExcInfo]], Optional[Scope], Any) -> Optional[str]
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.capture_exception` instead.

        Captures an exception.

        Alias of :py:meth:`sentry_sdk.Scope.capture_exception`.

        :param error: An exception to capture. If `None`, `sys.exc_info()` will be used.

        :param scope: An optional :py:class:`sentry_sdk.Scope` to apply to events.
            The `scope` and `scope_kwargs` parameters are mutually exclusive.

        :param scope_kwargs: Optional data to apply to event.
            For supported `**scope_kwargs` see :py:meth:`sentry_sdk.Scope.update_from_kwargs`.
            The `scope` and `scope_kwargs` parameters are mutually exclusive.

        :returns: An `event_id` if the SDK decided to send the event (see :py:meth:`sentry_sdk.client._Client.capture_event`).
        """
        last_event_id = get_current_scope().capture_exception(
            error, scope=scope, **scope_kwargs
        )

        if last_event_id is not None:
            self._last_event_id = last_event_id

        return last_event_id

    def add_breadcrumb(self, crumb=None, hint=None, **kwargs):
        # type: (Optional[Breadcrumb], Optional[BreadcrumbHint], Any) -> None
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.add_breadcrumb` instead.

        Adds a breadcrumb.

        :param crumb: Dictionary with the data as the sentry v7/v8 protocol expects.

        :param hint: An optional value that can be used by `before_breadcrumb`
            to customize the breadcrumbs that are emitted.
        """
        get_isolation_scope().add_breadcrumb(crumb, hint, **kwargs)

    def start_span(self, instrumenter=INSTRUMENTER.SENTRY, **kwargs):
        # type: (str, Any) -> Span
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.start_span` instead.

        Start a span whose parent is the currently active span or transaction, if any.

        The return value is a :py:class:`sentry_sdk.tracing.Span` instance,
        typically used as a context manager to start and stop timing in a `with`
        block.

        Only spans contained in a transaction are sent to Sentry. Most
        integrations start a transaction at the appropriate time, for example
        for every incoming HTTP request. Use
        :py:meth:`sentry_sdk.start_transaction` to start a new transaction when
        one is not already in progress.

        For supported `**kwargs` see :py:class:`sentry_sdk.tracing.Span`.
        """
        scope = get_current_scope()
        return scope.start_span(instrumenter=instrumenter, **kwargs)

    def start_transaction(
        self,
        transaction=None,
        instrumenter=INSTRUMENTER.SENTRY,
        custom_sampling_context=None,
        **kwargs
    ):
        # type: (Optional[Transaction], str, Optional[SamplingContext], Unpack[TransactionKwargs]) -> Union[Transaction, NoOpSpan]
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.start_transaction` instead.

        Start and return a transaction.

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

        For supported `**kwargs` see :py:class:`sentry_sdk.tracing.Transaction`.
        """
        scope = get_current_scope()

        # For backwards compatibility, we allow passing the scope as the hub.
        # We need a major release to make this nice. (if someone searches the code: deprecated)
        # Type checking disabled for this line because deprecated keys are not allowed in the type signature.
        kwargs["hub"] = scope  # type: ignore

        return scope.start_transaction(
            transaction, instrumenter, custom_sampling_context, **kwargs
        )

    def continue_trace(self, environ_or_headers, op=None, name=None, source=None):
        # type: (Dict[str, Any], Optional[str], Optional[str], Optional[str]) -> Transaction
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.continue_trace` instead.

        Sets the propagation context from environment or headers and returns a transaction.
        """
        return get_isolation_scope().continue_trace(
            environ_or_headers=environ_or_headers, op=op, name=name, source=source
        )

    @overload
    def push_scope(
        self, callback=None  # type: Optional[None]
    ):
        # type: (...) -> ContextManager[Scope]
        pass

    @overload
    def push_scope(  # noqa: F811
        self, callback  # type: Callable[[Scope], None]
    ):
        # type: (...) -> None
        pass

    def push_scope(  # noqa
        self,
        callback=None,  # type: Optional[Callable[[Scope], None]]
        continue_trace=True,  # type: bool
    ):
        # type: (...) -> Optional[ContextManager[Scope]]
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.

        Pushes a new layer on the scope stack.

        :param callback: If provided, this method pushes a scope, calls
            `callback`, and pops the scope again.

        :returns: If no `callback` is provided, a context manager that should
            be used to pop the scope again.
        """
        if callback is not None:
            with self.push_scope() as scope:
                callback(scope)
            return None

        return _ScopeManager(self)

    def pop_scope_unsafe(self):
        # type: () -> Tuple[Optional[Client], Scope]
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.

        Pops a scope layer from the stack.

        Try to use the context manager :py:meth:`push_scope` instead.
        """
        rv = self._stack.pop()
        assert self._stack, "stack must have at least one layer"
        return rv

    @overload
    def configure_scope(
        self, callback=None  # type: Optional[None]
    ):
        # type: (...) -> ContextManager[Scope]
        pass

    @overload
    def configure_scope(  # noqa: F811
        self, callback  # type: Callable[[Scope], None]
    ):
        # type: (...) -> None
        pass

    def configure_scope(  # noqa
        self,
        callback=None,  # type: Optional[Callable[[Scope], None]]
        continue_trace=True,  # type: bool
    ):
        # type: (...) -> Optional[ContextManager[Scope]]
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.

        Reconfigures the scope.

        :param callback: If provided, call the callback with the current scope.

        :returns: If no callback is provided, returns a context manager that returns the scope.
        """
        scope = get_isolation_scope()

        if continue_trace:
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

    def start_session(
        self, session_mode="application"  # type: str
    ):
        # type: (...) -> None
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.start_session` instead.

        Starts a new session.
        """
        get_isolation_scope().start_session(
            session_mode=session_mode,
        )

    def end_session(self):
        # type: (...) -> None
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.end_session` instead.

        Ends the current session if there is one.
        """
        get_isolation_scope().end_session()

    def stop_auto_session_tracking(self):
        # type: (...) -> None
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.stop_auto_session_tracking` instead.

        Stops automatic session tracking.

        This temporarily session tracking for the current scope when called.
        To resume session tracking call `resume_auto_session_tracking`.
        """
        get_isolation_scope().stop_auto_session_tracking()

    def resume_auto_session_tracking(self):
        # type: (...) -> None
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.resume_auto_session_tracking` instead.

        Resumes automatic session tracking for the current scope if
        disabled earlier.  This requires that generally automatic session
        tracking is enabled.
        """
        get_isolation_scope().resume_auto_session_tracking()

    def flush(
        self,
        timeout=None,  # type: Optional[float]
        callback=None,  # type: Optional[Callable[[int, float], None]]
    ):
        # type: (...) -> None
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.client._Client.flush` instead.

        Alias for :py:meth:`sentry_sdk.client._Client.flush`
        """
        return get_client().flush(timeout=timeout, callback=callback)

    def get_traceparent(self):
        # type: () -> Optional[str]
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.get_traceparent` instead.

        Returns the traceparent either from the active span or from the scope.
        """
        current_scope = get_current_scope()
        traceparent = current_scope.get_traceparent()

        if traceparent is None:
            isolation_scope = get_isolation_scope()
            traceparent = isolation_scope.get_traceparent()

        return traceparent

    def get_baggage(self):
        # type: () -> Optional[str]
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.get_baggage` instead.

        Returns Baggage either from the active span or from the scope.
        """
        current_scope = get_current_scope()
        baggage = current_scope.get_baggage()

        if baggage is None:
            isolation_scope = get_isolation_scope()
            baggage = isolation_scope.get_baggage()

        if baggage is not None:
            return baggage.serialize()

        return None

    def iter_trace_propagation_headers(self, span=None):
        # type: (Optional[Span]) -> Generator[Tuple[str, str], None, None]
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.iter_trace_propagation_headers` instead.

        Return HTTP headers which allow propagation of trace data. Data taken
        from the span representing the request, if available, or the current
        span on the scope if not.
        """
        return get_current_scope().iter_trace_propagation_headers(
            span=span,
        )

    def trace_propagation_meta(self, span=None):
        # type: (Optional[Span]) -> str
        """
        .. deprecated:: 2.0.0
            This function is deprecated and will be removed in a future release.
            Please use :py:meth:`sentry_sdk.Scope.trace_propagation_meta` instead.

        Return meta tags which should be injected into HTML templates
        to allow propagation of trace information.
        """
        if span is not None:
            logger.warning(
                "The parameter `span` in trace_propagation_meta() is deprecated and will be removed in the future."
            )

        return get_current_scope().trace_propagation_meta(
            span=span,
        )


with _suppress_hub_deprecation_warning():
    # Suppress deprecation warning for the Hub here, since we still always
    # import this module.
    GLOBAL_HUB = Hub()
_local.set(GLOBAL_HUB)


# Circular imports
from sentry_sdk import scope
