import os
import sys
import warnings
from copy import copy, deepcopy
from collections import deque
from contextlib import contextmanager
from enum import Enum
from datetime import datetime, timezone
from functools import wraps
from itertools import chain

from sentry_sdk.attachments import Attachment
from sentry_sdk.consts import DEFAULT_MAX_BREADCRUMBS, FALSE_VALUES, INSTRUMENTER
from sentry_sdk.feature_flags import FlagBuffer, DEFAULT_FLAG_CAPACITY
from sentry_sdk.profiler.continuous_profiler import try_autostart_continuous_profiler
from sentry_sdk.profiler.transaction_profiler import Profile
from sentry_sdk.session import Session
from sentry_sdk.tracing_utils import (
    Baggage,
    has_tracing_enabled,
    normalize_incoming_data,
    PropagationContext,
)
from sentry_sdk.tracing import (
    BAGGAGE_HEADER_NAME,
    SENTRY_TRACE_HEADER_NAME,
    NoOpSpan,
    Span,
    Transaction,
)
from sentry_sdk.utils import (
    capture_internal_exception,
    capture_internal_exceptions,
    ContextVar,
    datetime_from_isoformat,
    disable_capture_event,
    event_from_exception,
    exc_info_from_error,
    logger,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

    from typing import Any
    from typing import Callable
    from typing import Deque
    from typing import Dict
    from typing import Generator
    from typing import Iterator
    from typing import List
    from typing import Optional
    from typing import ParamSpec
    from typing import Tuple
    from typing import TypeVar
    from typing import Union

    from typing_extensions import Unpack

    from sentry_sdk._types import (
        Breadcrumb,
        BreadcrumbHint,
        ErrorProcessor,
        Event,
        EventProcessor,
        ExcInfo,
        Hint,
        LogLevelStr,
        SamplingContext,
        Type,
    )

    from sentry_sdk.tracing import TransactionKwargs

    import sentry_sdk

    P = ParamSpec("P")
    R = TypeVar("R")

    F = TypeVar("F", bound=Callable[..., Any])
    T = TypeVar("T")


# Holds data that will be added to **all** events sent by this process.
# In case this is a http server (think web framework) with multiple users
# the data will be added to events of all users.
# Typically this is used for process wide data such as the release.
_global_scope = None  # type: Optional[Scope]

# Holds data for the active request.
# This is used to isolate data for different requests or users.
# The isolation scope is usually created by integrations, but may also
# be created manually
_isolation_scope = ContextVar("isolation_scope", default=None)

# Holds data for the active span.
# This can be used to manually add additional data to a span.
_current_scope = ContextVar("current_scope", default=None)

global_event_processors = []  # type: List[EventProcessor]


class ScopeType(Enum):
    CURRENT = "current"
    ISOLATION = "isolation"
    GLOBAL = "global"
    MERGED = "merged"


class _ScopeManager:
    def __init__(self, hub=None):
        # type: (Optional[Any]) -> None
        self._old_scopes = []  # type: List[Scope]

    def __enter__(self):
        # type: () -> Scope
        isolation_scope = Scope.get_isolation_scope()

        self._old_scopes.append(isolation_scope)

        forked_scope = isolation_scope.fork()
        _isolation_scope.set(forked_scope)

        return forked_scope

    def __exit__(self, exc_type, exc_value, tb):
        # type: (Any, Any, Any) -> None
        old_scope = self._old_scopes.pop()
        _isolation_scope.set(old_scope)


def add_global_event_processor(processor):
    # type: (EventProcessor) -> None
    global_event_processors.append(processor)


def _attr_setter(fn):
    # type: (Any) -> Any
    return property(fset=fn, doc=fn.__doc__)


def _disable_capture(fn):
    # type: (F) -> F
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        # type: (Any, *Dict[str, Any], **Any) -> Any
        if not self._should_capture:
            return
        try:
            self._should_capture = False
            return fn(self, *args, **kwargs)
        finally:
            self._should_capture = True

    return wrapper  # type: ignore


class Scope:
    """The scope holds extra information that should be sent with all
    events that belong to it.
    """

    # NOTE: Even though it should not happen, the scope needs to not crash when
    # accessed by multiple threads. It's fine if it's full of races, but those
    # races should never make the user application crash.
    #
    # The same needs to hold for any accesses of the scope the SDK makes.

    __slots__ = (
        "_level",
        "_name",
        "_fingerprint",
        # note that for legacy reasons, _transaction is the transaction *name*,
        # not a Transaction object (the object is stored in _span)
        "_transaction",
        "_transaction_info",
        "_user",
        "_tags",
        "_contexts",
        "_extras",
        "_breadcrumbs",
        "_event_processors",
        "_error_processors",
        "_should_capture",
        "_span",
        "_session",
        "_attachments",
        "_force_auto_session_tracking",
        "_profile",
        "_propagation_context",
        "client",
        "_type",
        "_last_event_id",
        "_flags",
    )

    def __init__(self, ty=None, client=None):
        # type: (Optional[ScopeType], Optional[sentry_sdk.Client]) -> None
        self._type = ty

        self._event_processors = []  # type: List[EventProcessor]
        self._error_processors = []  # type: List[ErrorProcessor]

        self._name = None  # type: Optional[str]
        self._propagation_context = None  # type: Optional[PropagationContext]

        self.client = NonRecordingClient()  # type: sentry_sdk.client.BaseClient

        if client is not None:
            self.set_client(client)

        self.clear()

        incoming_trace_information = self._load_trace_data_from_env()
        self.generate_propagation_context(incoming_data=incoming_trace_information)

    def __copy__(self):
        # type: () -> Scope
        """
        Returns a copy of this scope.
        This also creates a copy of all referenced data structures.
        """
        rv = object.__new__(self.__class__)  # type: Scope

        rv._type = self._type
        rv.client = self.client
        rv._level = self._level
        rv._name = self._name
        rv._fingerprint = self._fingerprint
        rv._transaction = self._transaction
        rv._transaction_info = dict(self._transaction_info)
        rv._user = self._user

        rv._tags = dict(self._tags)
        rv._contexts = dict(self._contexts)
        rv._extras = dict(self._extras)

        rv._breadcrumbs = copy(self._breadcrumbs)
        rv._event_processors = list(self._event_processors)
        rv._error_processors = list(self._error_processors)
        rv._propagation_context = self._propagation_context

        rv._should_capture = self._should_capture
        rv._span = self._span
        rv._session = self._session
        rv._force_auto_session_tracking = self._force_auto_session_tracking
        rv._attachments = list(self._attachments)

        rv._profile = self._profile

        rv._last_event_id = self._last_event_id

        rv._flags = deepcopy(self._flags)

        return rv

    @classmethod
    def get_current_scope(cls):
        # type: () -> Scope
        """
        .. versionadded:: 2.0.0

        Returns the current scope.
        """
        current_scope = _current_scope.get()
        if current_scope is None:
            current_scope = Scope(ty=ScopeType.CURRENT)
            _current_scope.set(current_scope)

        return current_scope

    @classmethod
    def set_current_scope(cls, new_current_scope):
        # type: (Scope) -> None
        """
        .. versionadded:: 2.0.0

        Sets the given scope as the new current scope overwriting the existing current scope.
        :param new_current_scope: The scope to set as the new current scope.
        """
        _current_scope.set(new_current_scope)

    @classmethod
    def get_isolation_scope(cls):
        # type: () -> Scope
        """
        .. versionadded:: 2.0.0

        Returns the isolation scope.
        """
        isolation_scope = _isolation_scope.get()
        if isolation_scope is None:
            isolation_scope = Scope(ty=ScopeType.ISOLATION)
            _isolation_scope.set(isolation_scope)

        return isolation_scope

    @classmethod
    def set_isolation_scope(cls, new_isolation_scope):
        # type: (Scope) -> None
        """
        .. versionadded:: 2.0.0

        Sets the given scope as the new isolation scope overwriting the existing isolation scope.
        :param new_isolation_scope: The scope to set as the new isolation scope.
        """
        _isolation_scope.set(new_isolation_scope)

    @classmethod
    def get_global_scope(cls):
        # type: () -> Scope
        """
        .. versionadded:: 2.0.0

        Returns the global scope.
        """
        global _global_scope
        if _global_scope is None:
            _global_scope = Scope(ty=ScopeType.GLOBAL)

        return _global_scope

    @classmethod
    def last_event_id(cls):
        # type: () -> Optional[str]
        """
        .. versionadded:: 2.2.0

        Returns event ID of the event most recently captured by the isolation scope, or None if no event
        has been captured. We do not consider events that are dropped, e.g. by a before_send hook.
        Transactions also are not considered events in this context.

        The event corresponding to the returned event ID is NOT guaranteed to actually be sent to Sentry;
        whether the event is sent depends on the transport. The event could be sent later or not at all.
        Even a sent event could fail to arrive in Sentry due to network issues, exhausted quotas, or
        various other reasons.
        """
        return cls.get_isolation_scope()._last_event_id

    def _merge_scopes(self, additional_scope=None, additional_scope_kwargs=None):
        # type: (Optional[Scope], Optional[Dict[str, Any]]) -> Scope
        """
        Merges global, isolation and current scope into a new scope and
        adds the given additional scope or additional scope kwargs to it.
        """
        if additional_scope and additional_scope_kwargs:
            raise TypeError("cannot provide scope and kwargs")

        final_scope = copy(_global_scope) if _global_scope is not None else Scope()
        final_scope._type = ScopeType.MERGED

        isolation_scope = _isolation_scope.get()
        if isolation_scope is not None:
            final_scope.update_from_scope(isolation_scope)

        current_scope = _current_scope.get()
        if current_scope is not None:
            final_scope.update_from_scope(current_scope)

        if self != current_scope and self != isolation_scope:
            final_scope.update_from_scope(self)

        if additional_scope is not None:
            if callable(additional_scope):
                additional_scope(final_scope)
            else:
                final_scope.update_from_scope(additional_scope)

        elif additional_scope_kwargs:
            final_scope.update_from_kwargs(**additional_scope_kwargs)

        return final_scope

    @classmethod
    def get_client(cls):
        # type: () -> sentry_sdk.client.BaseClient
        """
        .. versionadded:: 2.0.0

        Returns the currently used :py:class:`sentry_sdk.Client`.
        This checks the current scope, the isolation scope and the global scope for a client.
        If no client is available a :py:class:`sentry_sdk.client.NonRecordingClient` is returned.
        """
        current_scope = _current_scope.get()
        try:
            client = current_scope.client
        except AttributeError:
            client = None

        if client is not None and client.is_active():
            return client

        isolation_scope = _isolation_scope.get()
        try:
            client = isolation_scope.client
        except AttributeError:
            client = None

        if client is not None and client.is_active():
            return client

        try:
            client = _global_scope.client  # type: ignore
        except AttributeError:
            client = None

        if client is not None and client.is_active():
            return client

        return NonRecordingClient()

    def set_client(self, client=None):
        # type: (Optional[sentry_sdk.client.BaseClient]) -> None
        """
        .. versionadded:: 2.0.0

        Sets the client for this scope.

        :param client: The client to use in this scope.
            If `None` the client of the scope will be replaced by a :py:class:`sentry_sdk.NonRecordingClient`.

        """
        self.client = client if client is not None else NonRecordingClient()

    def fork(self):
        # type: () -> Scope
        """
        .. versionadded:: 2.0.0

        Returns a fork of this scope.
        """
        forked_scope = copy(self)
        return forked_scope

    def _load_trace_data_from_env(self):
        # type: () -> Optional[Dict[str, str]]
        """
        Load Sentry trace id and baggage from environment variables.
        Can be disabled by setting SENTRY_USE_ENVIRONMENT to "false".
        """
        incoming_trace_information = None

        sentry_use_environment = (
            os.environ.get("SENTRY_USE_ENVIRONMENT") or ""
        ).lower()
        use_environment = sentry_use_environment not in FALSE_VALUES
        if use_environment:
            incoming_trace_information = {}

            if os.environ.get("SENTRY_TRACE"):
                incoming_trace_information[SENTRY_TRACE_HEADER_NAME] = (
                    os.environ.get("SENTRY_TRACE") or ""
                )

            if os.environ.get("SENTRY_BAGGAGE"):
                incoming_trace_information[BAGGAGE_HEADER_NAME] = (
                    os.environ.get("SENTRY_BAGGAGE") or ""
                )

        return incoming_trace_information or None

    def set_new_propagation_context(self):
        # type: () -> None
        """
        Creates a new propagation context and sets it as `_propagation_context`. Overwriting existing one.
        """
        self._propagation_context = PropagationContext()

    def generate_propagation_context(self, incoming_data=None):
        # type: (Optional[Dict[str, str]]) -> None
        """
        Makes sure the propagation context is set on the scope.
        If there is `incoming_data` overwrite existing propagation context.
        If there is no `incoming_data` create new propagation context, but do NOT overwrite if already existing.
        """
        if incoming_data:
            propagation_context = PropagationContext.from_incoming_data(incoming_data)
            if propagation_context is not None:
                self._propagation_context = propagation_context

        if self._type != ScopeType.CURRENT:
            if self._propagation_context is None:
                self.set_new_propagation_context()

    def get_dynamic_sampling_context(self):
        # type: () -> Optional[Dict[str, str]]
        """
        Returns the Dynamic Sampling Context from the Propagation Context.
        If not existing, creates a new one.
        """
        if self._propagation_context is None:
            return None

        baggage = self.get_baggage()
        if baggage is not None:
            self._propagation_context.dynamic_sampling_context = (
                baggage.dynamic_sampling_context()
            )

        return self._propagation_context.dynamic_sampling_context

    def get_traceparent(self, *args, **kwargs):
        # type: (Any, Any) -> Optional[str]
        """
        Returns the Sentry "sentry-trace" header (aka the traceparent) from the
        currently active span or the scopes Propagation Context.
        """
        client = self.get_client()

        # If we have an active span, return traceparent from there
        if has_tracing_enabled(client.options) and self.span is not None:
            return self.span.to_traceparent()

        # If this scope has a propagation context, return traceparent from there
        if self._propagation_context is not None:
            traceparent = "%s-%s" % (
                self._propagation_context.trace_id,
                self._propagation_context.span_id,
            )
            return traceparent

        # Fall back to isolation scope's traceparent. It always has one
        return self.get_isolation_scope().get_traceparent()

    def get_baggage(self, *args, **kwargs):
        # type: (Any, Any) -> Optional[Baggage]
        """
        Returns the Sentry "baggage" header containing trace information from the
        currently active span or the scopes Propagation Context.
        """
        client = self.get_client()

        # If we have an active span, return baggage from there
        if has_tracing_enabled(client.options) and self.span is not None:
            return self.span.to_baggage()

        # If this scope has a propagation context, return baggage from there
        if self._propagation_context is not None:
            dynamic_sampling_context = (
                self._propagation_context.dynamic_sampling_context
            )
            if dynamic_sampling_context is None:
                return Baggage.from_options(self)
            else:
                return Baggage(dynamic_sampling_context)

        # Fall back to isolation scope's baggage. It always has one
        return self.get_isolation_scope().get_baggage()

    def get_trace_context(self):
        # type: () -> Any
        """
        Returns the Sentry "trace" context from the Propagation Context.
        """
        if self._propagation_context is None:
            return None

        trace_context = {
            "trace_id": self._propagation_context.trace_id,
            "span_id": self._propagation_context.span_id,
            "parent_span_id": self._propagation_context.parent_span_id,
            "dynamic_sampling_context": self.get_dynamic_sampling_context(),
        }  # type: Dict[str, Any]

        return trace_context

    def trace_propagation_meta(self, *args, **kwargs):
        # type: (*Any, **Any) -> str
        """
        Return meta tags which should be injected into HTML templates
        to allow propagation of trace information.
        """
        span = kwargs.pop("span", None)
        if span is not None:
            logger.warning(
                "The parameter `span` in trace_propagation_meta() is deprecated and will be removed in the future."
            )

        meta = ""

        sentry_trace = self.get_traceparent()
        if sentry_trace is not None:
            meta += '<meta name="%s" content="%s">' % (
                SENTRY_TRACE_HEADER_NAME,
                sentry_trace,
            )

        baggage = self.get_baggage()
        if baggage is not None:
            meta += '<meta name="%s" content="%s">' % (
                BAGGAGE_HEADER_NAME,
                baggage.serialize(),
            )

        return meta

    def iter_headers(self):
        # type: () -> Iterator[Tuple[str, str]]
        """
        Creates a generator which returns the `sentry-trace` and `baggage` headers from the Propagation Context.
        """
        if self._propagation_context is not None:
            traceparent = self.get_traceparent()
            if traceparent is not None:
                yield SENTRY_TRACE_HEADER_NAME, traceparent

            dsc = self.get_dynamic_sampling_context()
            if dsc is not None:
                baggage = Baggage(dsc).serialize()
                yield BAGGAGE_HEADER_NAME, baggage

    def iter_trace_propagation_headers(self, *args, **kwargs):
        # type: (Any, Any) -> Generator[Tuple[str, str], None, None]
        """
        Return HTTP headers which allow propagation of trace data.

        If a span is given, the trace data will taken from the span.
        If no span is given, the trace data is taken from the scope.
        """
        client = self.get_client()
        if not client.options.get("propagate_traces"):
            warnings.warn(
                "The `propagate_traces` parameter is deprecated. Please use `trace_propagation_targets` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return

        span = kwargs.pop("span", None)
        span = span or self.span

        if has_tracing_enabled(client.options) and span is not None:
            for header in span.iter_headers():
                yield header
        else:
            # If this scope has a propagation context, return headers from there
            # (it could be that self is not the current scope nor the isolation scope)
            if self._propagation_context is not None:
                for header in self.iter_headers():
                    yield header
            else:
                # otherwise try headers from current scope
                current_scope = self.get_current_scope()
                if current_scope._propagation_context is not None:
                    for header in current_scope.iter_headers():
                        yield header
                else:
                    # otherwise fall back to headers from isolation scope
                    isolation_scope = self.get_isolation_scope()
                    if isolation_scope._propagation_context is not None:
                        for header in isolation_scope.iter_headers():
                            yield header

    def get_active_propagation_context(self):
        # type: () -> Optional[PropagationContext]
        if self._propagation_context is not None:
            return self._propagation_context

        current_scope = self.get_current_scope()
        if current_scope._propagation_context is not None:
            return current_scope._propagation_context

        isolation_scope = self.get_isolation_scope()
        if isolation_scope._propagation_context is not None:
            return isolation_scope._propagation_context

        return None

    def clear(self):
        # type: () -> None
        """Clears the entire scope."""
        self._level = None  # type: Optional[LogLevelStr]
        self._fingerprint = None  # type: Optional[List[str]]
        self._transaction = None  # type: Optional[str]
        self._transaction_info = {}  # type: MutableMapping[str, str]
        self._user = None  # type: Optional[Dict[str, Any]]

        self._tags = {}  # type: Dict[str, Any]
        self._contexts = {}  # type: Dict[str, Dict[str, Any]]
        self._extras = {}  # type: MutableMapping[str, Any]
        self._attachments = []  # type: List[Attachment]

        self.clear_breadcrumbs()
        self._should_capture = True  # type: bool

        self._span = None  # type: Optional[Span]
        self._session = None  # type: Optional[Session]
        self._force_auto_session_tracking = None  # type: Optional[bool]

        self._profile = None  # type: Optional[Profile]

        self._propagation_context = None

        # self._last_event_id is only applicable to isolation scopes
        self._last_event_id = None  # type: Optional[str]
        self._flags = None  # type: Optional[FlagBuffer]

    @_attr_setter
    def level(self, value):
        # type: (LogLevelStr) -> None
        """
        When set this overrides the level.

        .. deprecated:: 1.0.0
            Use :func:`set_level` instead.

        :param value: The level to set.
        """
        logger.warning(
            "Deprecated: use .set_level() instead. This will be removed in the future."
        )

        self._level = value

    def set_level(self, value):
        # type: (LogLevelStr) -> None
        """
        Sets the level for the scope.

        :param value: The level to set.
        """
        self._level = value

    @_attr_setter
    def fingerprint(self, value):
        # type: (Optional[List[str]]) -> None
        """When set this overrides the default fingerprint."""
        self._fingerprint = value

    @property
    def transaction(self):
        # type: () -> Any
        # would be type: () -> Optional[Transaction], see https://github.com/python/mypy/issues/3004
        """Return the transaction (root span) in the scope, if any."""

        # there is no span/transaction on the scope
        if self._span is None:
            return None

        # there is an orphan span on the scope
        if self._span.containing_transaction is None:
            return None

        # there is either a transaction (which is its own containing
        # transaction) or a non-orphan span on the scope
        return self._span.containing_transaction

    @transaction.setter
    def transaction(self, value):
        # type: (Any) -> None
        # would be type: (Optional[str]) -> None, see https://github.com/python/mypy/issues/3004
        """When set this forces a specific transaction name to be set.

        Deprecated: use set_transaction_name instead."""

        # XXX: the docstring above is misleading. The implementation of
        # apply_to_event prefers an existing value of event.transaction over
        # anything set in the scope.
        # XXX: note that with the introduction of the Scope.transaction getter,
        # there is a semantic and type mismatch between getter and setter. The
        # getter returns a Transaction, the setter sets a transaction name.
        # Without breaking version compatibility, we could make the setter set a
        # transaction name or transaction (self._span) depending on the type of
        # the value argument.

        logger.warning(
            "Assigning to scope.transaction directly is deprecated: use scope.set_transaction_name() instead."
        )
        self._transaction = value
        if self._span and self._span.containing_transaction:
            self._span.containing_transaction.name = value

    def set_transaction_name(self, name, source=None):
        # type: (str, Optional[str]) -> None
        """Set the transaction name and optionally the transaction source."""
        self._transaction = name

        if self._span and self._span.containing_transaction:
            self._span.containing_transaction.name = name
            if source:
                self._span.containing_transaction.source = source

        if source:
            self._transaction_info["source"] = source

    @_attr_setter
    def user(self, value):
        # type: (Optional[Dict[str, Any]]) -> None
        """When set a specific user is bound to the scope. Deprecated in favor of set_user."""
        self.set_user(value)

    def set_user(self, value):
        # type: (Optional[Dict[str, Any]]) -> None
        """Sets a user for the scope."""
        self._user = value
        session = self.get_isolation_scope()._session
        if session is not None:
            session.update(user=value)

    @property
    def span(self):
        # type: () -> Optional[Span]
        """Get/set current tracing span or transaction."""
        return self._span

    @span.setter
    def span(self, span):
        # type: (Optional[Span]) -> None
        self._span = span
        # XXX: this differs from the implementation in JS, there Scope.setSpan
        # does not set Scope._transactionName.
        if isinstance(span, Transaction):
            transaction = span
            if transaction.name:
                self._transaction = transaction.name
                if transaction.source:
                    self._transaction_info["source"] = transaction.source

    @property
    def profile(self):
        # type: () -> Optional[Profile]
        return self._profile

    @profile.setter
    def profile(self, profile):
        # type: (Optional[Profile]) -> None

        self._profile = profile

    def set_tag(self, key, value):
        # type: (str, Any) -> None
        """
        Sets a tag for a key to a specific value.

        :param key: Key of the tag to set.

        :param value: Value of the tag to set.
        """
        self._tags[key] = value

    def set_tags(self, tags):
        # type: (Mapping[str, object]) -> None
        """Sets multiple tags at once.

        This method updates multiple tags at once. The tags are passed as a dictionary
        or other mapping type.

        Calling this method is equivalent to calling `set_tag` on each key-value pair
        in the mapping. If a tag key already exists in the scope, its value will be
        updated. If the tag key does not exist in the scope, the key-value pair will
        be added to the scope.

        This method only modifies tag keys in the `tags` mapping passed to the method.
        `scope.set_tags({})` is, therefore, a no-op.

        :param tags: A mapping of tag keys to tag values to set.
        """
        self._tags.update(tags)

    def remove_tag(self, key):
        # type: (str) -> None
        """
        Removes a specific tag.

        :param key: Key of the tag to remove.
        """
        self._tags.pop(key, None)

    def set_context(
        self,
        key,  # type: str
        value,  # type: Dict[str, Any]
    ):
        # type: (...) -> None
        """
        Binds a context at a certain key to a specific value.
        """
        self._contexts[key] = value

    def remove_context(
        self, key  # type: str
    ):
        # type: (...) -> None
        """Removes a context."""
        self._contexts.pop(key, None)

    def set_extra(
        self,
        key,  # type: str
        value,  # type: Any
    ):
        # type: (...) -> None
        """Sets an extra key to a specific value."""
        self._extras[key] = value

    def remove_extra(
        self, key  # type: str
    ):
        # type: (...) -> None
        """Removes a specific extra key."""
        self._extras.pop(key, None)

    def clear_breadcrumbs(self):
        # type: () -> None
        """Clears breadcrumb buffer."""
        self._breadcrumbs = deque()  # type: Deque[Breadcrumb]

    def add_attachment(
        self,
        bytes=None,  # type: Union[None, bytes, Callable[[], bytes]]
        filename=None,  # type: Optional[str]
        path=None,  # type: Optional[str]
        content_type=None,  # type: Optional[str]
        add_to_transactions=False,  # type: bool
    ):
        # type: (...) -> None
        """Adds an attachment to future events sent from this scope.

        The parameters are the same as for the :py:class:`sentry_sdk.attachments.Attachment` constructor.
        """
        self._attachments.append(
            Attachment(
                bytes=bytes,
                path=path,
                filename=filename,
                content_type=content_type,
                add_to_transactions=add_to_transactions,
            )
        )

    def add_breadcrumb(self, crumb=None, hint=None, **kwargs):
        # type: (Optional[Breadcrumb], Optional[BreadcrumbHint], Any) -> None
        """
        Adds a breadcrumb.

        :param crumb: Dictionary with the data as the sentry v7/v8 protocol expects.

        :param hint: An optional value that can be used by `before_breadcrumb`
            to customize the breadcrumbs that are emitted.
        """
        client = self.get_client()

        if not client.is_active():
            logger.info("Dropped breadcrumb because no client bound")
            return

        before_breadcrumb = client.options.get("before_breadcrumb")
        max_breadcrumbs = client.options.get("max_breadcrumbs", DEFAULT_MAX_BREADCRUMBS)

        crumb = dict(crumb or ())  # type: Breadcrumb
        crumb.update(kwargs)
        if not crumb:
            return

        hint = dict(hint or ())  # type: Hint

        if crumb.get("timestamp") is None:
            crumb["timestamp"] = datetime.now(timezone.utc)
        if crumb.get("type") is None:
            crumb["type"] = "default"

        if before_breadcrumb is not None:
            new_crumb = before_breadcrumb(crumb, hint)
        else:
            new_crumb = crumb

        if new_crumb is not None:
            self._breadcrumbs.append(new_crumb)
        else:
            logger.info("before breadcrumb dropped breadcrumb (%s)", crumb)

        while len(self._breadcrumbs) > max_breadcrumbs:
            self._breadcrumbs.popleft()

    def start_transaction(
        self,
        transaction=None,
        instrumenter=INSTRUMENTER.SENTRY,
        custom_sampling_context=None,
        **kwargs,
    ):
        # type: (Optional[Transaction], str, Optional[SamplingContext], Unpack[TransactionKwargs]) -> Union[Transaction, NoOpSpan]
        """
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

        :param transaction: The transaction to start. If omitted, we create and
            start a new transaction.
        :param instrumenter: This parameter is meant for internal use only. It
            will be removed in the next major version.
        :param custom_sampling_context: The transaction's custom sampling context.
        :param kwargs: Optional keyword arguments to be passed to the Transaction
            constructor. See :py:class:`sentry_sdk.tracing.Transaction` for
            available arguments.
        """
        kwargs.setdefault("scope", self)

        client = self.get_client()

        configuration_instrumenter = client.options["instrumenter"]

        if instrumenter != configuration_instrumenter:
            return NoOpSpan()

        try_autostart_continuous_profiler()

        custom_sampling_context = custom_sampling_context or {}

        # kwargs at this point has type TransactionKwargs, since we have removed
        # the client and custom_sampling_context from it.
        transaction_kwargs = kwargs  # type: TransactionKwargs

        # if we haven't been given a transaction, make one
        if transaction is None:
            transaction = Transaction(**transaction_kwargs)

        # use traces_sample_rate, traces_sampler, and/or inheritance to make a
        # sampling decision
        sampling_context = {
            "transaction_context": transaction.to_json(),
            "parent_sampled": transaction.parent_sampled,
        }
        sampling_context.update(custom_sampling_context)
        transaction._set_initial_sampling_decision(sampling_context=sampling_context)

        if transaction.sampled:
            profile = Profile(
                transaction.sampled, transaction._start_timestamp_monotonic_ns
            )
            profile._set_initial_sampling_decision(sampling_context=sampling_context)

            transaction._profile = profile

            # we don't bother to keep spans if we already know we're not going to
            # send the transaction
            max_spans = (client.options["_experiments"].get("max_spans")) or 1000
            transaction.init_span_recorder(maxlen=max_spans)

        return transaction

    def start_span(self, instrumenter=INSTRUMENTER.SENTRY, **kwargs):
        # type: (str, Any) -> Span
        """
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

        The instrumenter parameter is deprecated for user code, and it will
        be removed in the next major version. Going forward, it should only
        be used by the SDK itself.
        """
        if kwargs.get("description") is not None:
            warnings.warn(
                "The `description` parameter is deprecated. Please use `name` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        with new_scope():
            kwargs.setdefault("scope", self)

            client = self.get_client()

            configuration_instrumenter = client.options["instrumenter"]

            if instrumenter != configuration_instrumenter:
                return NoOpSpan()

            # get current span or transaction
            span = self.span or self.get_isolation_scope().span

            if span is None:
                # New spans get the `trace_id` from the scope
                if "trace_id" not in kwargs:
                    propagation_context = self.get_active_propagation_context()
                    if propagation_context is not None:
                        kwargs["trace_id"] = propagation_context.trace_id

                span = Span(**kwargs)
            else:
                # Children take `trace_id`` from the parent span.
                span = span.start_child(**kwargs)

            return span

    def continue_trace(
        self, environ_or_headers, op=None, name=None, source=None, origin="manual"
    ):
        # type: (Dict[str, Any], Optional[str], Optional[str], Optional[str], str) -> Transaction
        """
        Sets the propagation context from environment or headers and returns a transaction.
        """
        self.generate_propagation_context(environ_or_headers)

        transaction = Transaction.continue_from_headers(
            normalize_incoming_data(environ_or_headers),
            op=op,
            origin=origin,
            name=name,
            source=source,
        )

        return transaction

    def capture_event(self, event, hint=None, scope=None, **scope_kwargs):
        # type: (Event, Optional[Hint], Optional[Scope], Any) -> Optional[str]
        """
        Captures an event.

        Merges given scope data and calls :py:meth:`sentry_sdk.client._Client.capture_event`.

        :param event: A ready-made event that can be directly sent to Sentry.

        :param hint: Contains metadata about the event that can be read from `before_send`, such as the original exception object or a HTTP request object.

        :param scope: An optional :py:class:`sentry_sdk.Scope` to apply to events.
            The `scope` and `scope_kwargs` parameters are mutually exclusive.

        :param scope_kwargs: Optional data to apply to event.
            For supported `**scope_kwargs` see :py:meth:`sentry_sdk.Scope.update_from_kwargs`.
            The `scope` and `scope_kwargs` parameters are mutually exclusive.

        :returns: An `event_id` if the SDK decided to send the event (see :py:meth:`sentry_sdk.client._Client.capture_event`).
        """
        if disable_capture_event.get(False):
            return None

        scope = self._merge_scopes(scope, scope_kwargs)

        event_id = self.get_client().capture_event(event=event, hint=hint, scope=scope)

        if event_id is not None and event.get("type") != "transaction":
            self.get_isolation_scope()._last_event_id = event_id

        return event_id

    def capture_message(self, message, level=None, scope=None, **scope_kwargs):
        # type: (str, Optional[LogLevelStr], Optional[Scope], Any) -> Optional[str]
        """
        Captures a message.

        :param message: The string to send as the message.

        :param level: If no level is provided, the default level is `info`.

        :param scope: An optional :py:class:`sentry_sdk.Scope` to apply to events.
            The `scope` and `scope_kwargs` parameters are mutually exclusive.

        :param scope_kwargs: Optional data to apply to event.
            For supported `**scope_kwargs` see :py:meth:`sentry_sdk.Scope.update_from_kwargs`.
            The `scope` and `scope_kwargs` parameters are mutually exclusive.

        :returns: An `event_id` if the SDK decided to send the event (see :py:meth:`sentry_sdk.client._Client.capture_event`).
        """
        if disable_capture_event.get(False):
            return None

        if level is None:
            level = "info"

        event = {
            "message": message,
            "level": level,
        }  # type: Event

        return self.capture_event(event, scope=scope, **scope_kwargs)

    def capture_exception(self, error=None, scope=None, **scope_kwargs):
        # type: (Optional[Union[BaseException, ExcInfo]], Optional[Scope], Any) -> Optional[str]
        """Captures an exception.

        :param error: An exception to capture. If `None`, `sys.exc_info()` will be used.

        :param scope: An optional :py:class:`sentry_sdk.Scope` to apply to events.
            The `scope` and `scope_kwargs` parameters are mutually exclusive.

        :param scope_kwargs: Optional data to apply to event.
            For supported `**scope_kwargs` see :py:meth:`sentry_sdk.Scope.update_from_kwargs`.
            The `scope` and `scope_kwargs` parameters are mutually exclusive.

        :returns: An `event_id` if the SDK decided to send the event (see :py:meth:`sentry_sdk.client._Client.capture_event`).
        """
        if disable_capture_event.get(False):
            return None

        if error is not None:
            exc_info = exc_info_from_error(error)
        else:
            exc_info = sys.exc_info()

        event, hint = event_from_exception(
            exc_info, client_options=self.get_client().options
        )

        try:
            return self.capture_event(event, hint=hint, scope=scope, **scope_kwargs)
        except Exception:
            capture_internal_exception(sys.exc_info())

        return None

    def start_session(self, *args, **kwargs):
        # type: (*Any, **Any) -> None
        """Starts a new session."""
        session_mode = kwargs.pop("session_mode", "application")

        self.end_session()

        client = self.get_client()
        self._session = Session(
            release=client.options.get("release"),
            environment=client.options.get("environment"),
            user=self._user,
            session_mode=session_mode,
        )

    def end_session(self, *args, **kwargs):
        # type: (*Any, **Any) -> None
        """Ends the current session if there is one."""
        session = self._session
        self._session = None

        if session is not None:
            session.close()
            self.get_client().capture_session(session)

    def stop_auto_session_tracking(self, *args, **kwargs):
        # type: (*Any, **Any) -> None
        """Stops automatic session tracking.

        This temporarily session tracking for the current scope when called.
        To resume session tracking call `resume_auto_session_tracking`.
        """
        self.end_session()
        self._force_auto_session_tracking = False

    def resume_auto_session_tracking(self):
        # type: (...) -> None
        """Resumes automatic session tracking for the current scope if
        disabled earlier.  This requires that generally automatic session
        tracking is enabled.
        """
        self._force_auto_session_tracking = None

    def add_event_processor(
        self, func  # type: EventProcessor
    ):
        # type: (...) -> None
        """Register a scope local event processor on the scope.

        :param func: This function behaves like `before_send.`
        """
        if len(self._event_processors) > 20:
            logger.warning(
                "Too many event processors on scope! Clearing list to free up some memory: %r",
                self._event_processors,
            )
            del self._event_processors[:]

        self._event_processors.append(func)

    def add_error_processor(
        self,
        func,  # type: ErrorProcessor
        cls=None,  # type: Optional[Type[BaseException]]
    ):
        # type: (...) -> None
        """Register a scope local error processor on the scope.

        :param func: A callback that works similar to an event processor but is invoked with the original exception info triple as second argument.

        :param cls: Optionally, only process exceptions of this type.
        """
        if cls is not None:
            cls_ = cls  # For mypy.
            real_func = func

            def func(event, exc_info):
                # type: (Event, ExcInfo) -> Optional[Event]
                try:
                    is_inst = isinstance(exc_info[1], cls_)
                except Exception:
                    is_inst = False
                if is_inst:
                    return real_func(event, exc_info)
                return event

        self._error_processors.append(func)

    def _apply_level_to_event(self, event, hint, options):
        # type: (Event, Hint, Optional[Dict[str, Any]]) -> None
        if self._level is not None:
            event["level"] = self._level

    def _apply_breadcrumbs_to_event(self, event, hint, options):
        # type: (Event, Hint, Optional[Dict[str, Any]]) -> None
        event.setdefault("breadcrumbs", {}).setdefault("values", []).extend(
            self._breadcrumbs
        )

        # Attempt to sort timestamps
        try:
            for crumb in event["breadcrumbs"]["values"]:
                if isinstance(crumb["timestamp"], str):
                    crumb["timestamp"] = datetime_from_isoformat(crumb["timestamp"])

            event["breadcrumbs"]["values"].sort(key=lambda crumb: crumb["timestamp"])
        except Exception as err:
            logger.debug("Error when sorting breadcrumbs", exc_info=err)
            pass

    def _apply_user_to_event(self, event, hint, options):
        # type: (Event, Hint, Optional[Dict[str, Any]]) -> None
        if event.get("user") is None and self._user is not None:
            event["user"] = self._user

    def _apply_transaction_name_to_event(self, event, hint, options):
        # type: (Event, Hint, Optional[Dict[str, Any]]) -> None
        if event.get("transaction") is None and self._transaction is not None:
            event["transaction"] = self._transaction

    def _apply_transaction_info_to_event(self, event, hint, options):
        # type: (Event, Hint, Optional[Dict[str, Any]]) -> None
        if event.get("transaction_info") is None and self._transaction_info is not None:
            event["transaction_info"] = self._transaction_info

    def _apply_fingerprint_to_event(self, event, hint, options):
        # type: (Event, Hint, Optional[Dict[str, Any]]) -> None
        if event.get("fingerprint") is None and self._fingerprint is not None:
            event["fingerprint"] = self._fingerprint

    def _apply_extra_to_event(self, event, hint, options):
        # type: (Event, Hint, Optional[Dict[str, Any]]) -> None
        if self._extras:
            event.setdefault("extra", {}).update(self._extras)

    def _apply_tags_to_event(self, event, hint, options):
        # type: (Event, Hint, Optional[Dict[str, Any]]) -> None
        if self._tags:
            event.setdefault("tags", {}).update(self._tags)

    def _apply_contexts_to_event(self, event, hint, options):
        # type: (Event, Hint, Optional[Dict[str, Any]]) -> None
        if self._contexts:
            event.setdefault("contexts", {}).update(self._contexts)

        contexts = event.setdefault("contexts", {})

        # Add "trace" context
        if contexts.get("trace") is None:
            if has_tracing_enabled(options) and self._span is not None:
                contexts["trace"] = self._span.get_trace_context()
            else:
                contexts["trace"] = self.get_trace_context()

    def _apply_flags_to_event(self, event, hint, options):
        # type: (Event, Hint, Optional[Dict[str, Any]]) -> None
        flags = self.flags.get()
        if len(flags) > 0:
            event.setdefault("contexts", {}).setdefault("flags", {}).update(
                {"values": flags}
            )

    def _drop(self, cause, ty):
        # type: (Any, str) -> Optional[Any]
        logger.info("%s (%s) dropped event", ty, cause)
        return None

    def run_error_processors(self, event, hint):
        # type: (Event, Hint) -> Optional[Event]
        """
        Runs the error processors on the event and returns the modified event.
        """
        exc_info = hint.get("exc_info")
        if exc_info is not None:
            error_processors = chain(
                self.get_global_scope()._error_processors,
                self.get_isolation_scope()._error_processors,
                self.get_current_scope()._error_processors,
            )

            for error_processor in error_processors:
                new_event = error_processor(event, exc_info)
                if new_event is None:
                    return self._drop(error_processor, "error processor")

                event = new_event

        return event

    def run_event_processors(self, event, hint):
        # type: (Event, Hint) -> Optional[Event]
        """
        Runs the event processors on the event and returns the modified event.
        """
        ty = event.get("type")
        is_check_in = ty == "check_in"

        if not is_check_in:
            # Get scopes without creating them to prevent infinite recursion
            isolation_scope = _isolation_scope.get()
            current_scope = _current_scope.get()

            event_processors = chain(
                global_event_processors,
                _global_scope and _global_scope._event_processors or [],
                isolation_scope and isolation_scope._event_processors or [],
                current_scope and current_scope._event_processors or [],
            )

            for event_processor in event_processors:
                new_event = event
                with capture_internal_exceptions():
                    new_event = event_processor(event, hint)
                if new_event is None:
                    return self._drop(event_processor, "event processor")
                event = new_event

        return event

    @_disable_capture
    def apply_to_event(
        self,
        event,  # type: Event
        hint,  # type: Hint
        options=None,  # type: Optional[Dict[str, Any]]
    ):
        # type: (...) -> Optional[Event]
        """Applies the information contained on the scope to the given event."""
        ty = event.get("type")
        is_transaction = ty == "transaction"
        is_check_in = ty == "check_in"

        # put all attachments into the hint. This lets callbacks play around
        # with attachments. We also later pull this out of the hint when we
        # create the envelope.
        attachments_to_send = hint.get("attachments") or []
        for attachment in self._attachments:
            if not is_transaction or attachment.add_to_transactions:
                attachments_to_send.append(attachment)
        hint["attachments"] = attachments_to_send

        self._apply_contexts_to_event(event, hint, options)

        if is_check_in:
            # Check-ins only support the trace context, strip all others
            event["contexts"] = {
                "trace": event.setdefault("contexts", {}).get("trace", {})
            }

        if not is_check_in:
            self._apply_level_to_event(event, hint, options)
            self._apply_fingerprint_to_event(event, hint, options)
            self._apply_user_to_event(event, hint, options)
            self._apply_transaction_name_to_event(event, hint, options)
            self._apply_transaction_info_to_event(event, hint, options)
            self._apply_tags_to_event(event, hint, options)
            self._apply_extra_to_event(event, hint, options)

        if not is_transaction and not is_check_in:
            self._apply_breadcrumbs_to_event(event, hint, options)
            self._apply_flags_to_event(event, hint, options)

        event = self.run_error_processors(event, hint)
        if event is None:
            return None

        event = self.run_event_processors(event, hint)
        if event is None:
            return None

        return event

    def update_from_scope(self, scope):
        # type: (Scope) -> None
        """Update the scope with another scope's data."""
        if scope._level is not None:
            self._level = scope._level
        if scope._fingerprint is not None:
            self._fingerprint = scope._fingerprint
        if scope._transaction is not None:
            self._transaction = scope._transaction
        if scope._transaction_info is not None:
            self._transaction_info.update(scope._transaction_info)
        if scope._user is not None:
            self._user = scope._user
        if scope._tags:
            self._tags.update(scope._tags)
        if scope._contexts:
            self._contexts.update(scope._contexts)
        if scope._extras:
            self._extras.update(scope._extras)
        if scope._breadcrumbs:
            self._breadcrumbs.extend(scope._breadcrumbs)
        if scope._span:
            self._span = scope._span
        if scope._attachments:
            self._attachments.extend(scope._attachments)
        if scope._profile:
            self._profile = scope._profile
        if scope._propagation_context:
            self._propagation_context = scope._propagation_context
        if scope._session:
            self._session = scope._session
        if scope._flags:
            if not self._flags:
                self._flags = deepcopy(scope._flags)
            else:
                for flag in scope._flags.get():
                    self._flags.set(flag["flag"], flag["result"])

    def update_from_kwargs(
        self,
        user=None,  # type: Optional[Any]
        level=None,  # type: Optional[LogLevelStr]
        extras=None,  # type: Optional[Dict[str, Any]]
        contexts=None,  # type: Optional[Dict[str, Any]]
        tags=None,  # type: Optional[Dict[str, str]]
        fingerprint=None,  # type: Optional[List[str]]
    ):
        # type: (...) -> None
        """Update the scope's attributes."""
        if level is not None:
            self._level = level
        if user is not None:
            self._user = user
        if extras is not None:
            self._extras.update(extras)
        if contexts is not None:
            self._contexts.update(contexts)
        if tags is not None:
            self._tags.update(tags)
        if fingerprint is not None:
            self._fingerprint = fingerprint

    def __repr__(self):
        # type: () -> str
        return "<%s id=%s name=%s type=%s>" % (
            self.__class__.__name__,
            hex(id(self)),
            self._name,
            self._type,
        )

    @property
    def flags(self):
        # type: () -> FlagBuffer
        if self._flags is None:
            max_flags = (
                self.get_client().options["_experiments"].get("max_flags")
                or DEFAULT_FLAG_CAPACITY
            )
            self._flags = FlagBuffer(capacity=max_flags)
        return self._flags


@contextmanager
def new_scope():
    # type: () -> Generator[Scope, None, None]
    """
    .. versionadded:: 2.0.0

    Context manager that forks the current scope and runs the wrapped code in it.
    After the wrapped code is executed, the original scope is restored.

    Example Usage:

    .. code-block:: python

        import sentry_sdk

        with sentry_sdk.new_scope() as scope:
            scope.set_tag("color", "green")
            sentry_sdk.capture_message("hello") # will include `color` tag.

        sentry_sdk.capture_message("hello, again") # will NOT include `color` tag.

    """
    # fork current scope
    current_scope = Scope.get_current_scope()
    new_scope = current_scope.fork()
    token = _current_scope.set(new_scope)

    try:
        yield new_scope

    finally:
        # restore original scope
        _current_scope.reset(token)


@contextmanager
def use_scope(scope):
    # type: (Scope) -> Generator[Scope, None, None]
    """
    .. versionadded:: 2.0.0

    Context manager that uses the given `scope` and runs the wrapped code in it.
    After the wrapped code is executed, the original scope is restored.

    Example Usage:
    Suppose the variable `scope` contains a `Scope` object, which is not currently
    the active scope.

    .. code-block:: python

        import sentry_sdk

        with sentry_sdk.use_scope(scope):
            scope.set_tag("color", "green")
            sentry_sdk.capture_message("hello") # will include `color` tag.

        sentry_sdk.capture_message("hello, again") # will NOT include `color` tag.

    """
    # set given scope as current scope
    token = _current_scope.set(scope)

    try:
        yield scope

    finally:
        # restore original scope
        _current_scope.reset(token)


@contextmanager
def isolation_scope():
    # type: () -> Generator[Scope, None, None]
    """
    .. versionadded:: 2.0.0

    Context manager that forks the current isolation scope and runs the wrapped code in it.
    The current scope is also forked to not bleed data into the existing current scope.
    After the wrapped code is executed, the original scopes are restored.

    Example Usage:

    .. code-block:: python

        import sentry_sdk

        with sentry_sdk.isolation_scope() as scope:
            scope.set_tag("color", "green")
            sentry_sdk.capture_message("hello") # will include `color` tag.

        sentry_sdk.capture_message("hello, again") # will NOT include `color` tag.

    """
    # fork current scope
    current_scope = Scope.get_current_scope()
    forked_current_scope = current_scope.fork()
    current_token = _current_scope.set(forked_current_scope)

    # fork isolation scope
    isolation_scope = Scope.get_isolation_scope()
    new_isolation_scope = isolation_scope.fork()
    isolation_token = _isolation_scope.set(new_isolation_scope)

    try:
        yield new_isolation_scope

    finally:
        # restore original scopes
        _current_scope.reset(current_token)
        _isolation_scope.reset(isolation_token)


@contextmanager
def use_isolation_scope(isolation_scope):
    # type: (Scope) -> Generator[Scope, None, None]
    """
    .. versionadded:: 2.0.0

    Context manager that uses the given `isolation_scope` and runs the wrapped code in it.
    The current scope is also forked to not bleed data into the existing current scope.
    After the wrapped code is executed, the original scopes are restored.

    Example Usage:

    .. code-block:: python

        import sentry_sdk

        with sentry_sdk.isolation_scope() as scope:
            scope.set_tag("color", "green")
            sentry_sdk.capture_message("hello") # will include `color` tag.

        sentry_sdk.capture_message("hello, again") # will NOT include `color` tag.

    """
    # fork current scope
    current_scope = Scope.get_current_scope()
    forked_current_scope = current_scope.fork()
    current_token = _current_scope.set(forked_current_scope)

    # set given scope as isolation scope
    isolation_token = _isolation_scope.set(isolation_scope)

    try:
        yield isolation_scope

    finally:
        # restore original scopes
        _current_scope.reset(current_token)
        _isolation_scope.reset(isolation_token)


def should_send_default_pii():
    # type: () -> bool
    """Shortcut for `Scope.get_client().should_send_default_pii()`."""
    return Scope.get_client().should_send_default_pii()


# Circular imports
from sentry_sdk.client import NonRecordingClient

if TYPE_CHECKING:
    import sentry_sdk.client
