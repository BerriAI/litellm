import os
import time
import warnings
from threading import Thread, Lock
from contextlib import contextmanager

import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.session import Session
from sentry_sdk.utils import format_timestamp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import Dict
    from typing import Generator
    from typing import List
    from typing import Optional
    from typing import Union


def is_auto_session_tracking_enabled(hub=None):
    # type: (Optional[sentry_sdk.Hub]) -> Union[Any, bool, None]
    """DEPRECATED: Utility function to find out if session tracking is enabled."""

    # Internal callers should use private _is_auto_session_tracking_enabled, instead.
    warnings.warn(
        "This function is deprecated and will be removed in the next major release. "
        "There is no public API replacement.",
        DeprecationWarning,
        stacklevel=2,
    )

    if hub is None:
        hub = sentry_sdk.Hub.current

    should_track = hub.scope._force_auto_session_tracking

    if should_track is None:
        client_options = hub.client.options if hub.client else {}
        should_track = client_options.get("auto_session_tracking", False)

    return should_track


@contextmanager
def auto_session_tracking(hub=None, session_mode="application"):
    # type: (Optional[sentry_sdk.Hub], str) -> Generator[None, None, None]
    """DEPRECATED: Use track_session instead
    Starts and stops a session automatically around a block.
    """
    warnings.warn(
        "This function is deprecated and will be removed in the next major release. "
        "Use track_session instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if hub is None:
        hub = sentry_sdk.Hub.current
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        should_track = is_auto_session_tracking_enabled(hub)
    if should_track:
        hub.start_session(session_mode=session_mode)
    try:
        yield
    finally:
        if should_track:
            hub.end_session()


def is_auto_session_tracking_enabled_scope(scope):
    # type: (sentry_sdk.Scope) -> bool
    """
    DEPRECATED: Utility function to find out if session tracking is enabled.
    """

    warnings.warn(
        "This function is deprecated and will be removed in the next major release. "
        "There is no public API replacement.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Internal callers should use private _is_auto_session_tracking_enabled, instead.
    return _is_auto_session_tracking_enabled(scope)


def _is_auto_session_tracking_enabled(scope):
    # type: (sentry_sdk.Scope) -> bool
    """
    Utility function to find out if session tracking is enabled.
    """

    should_track = scope._force_auto_session_tracking
    if should_track is None:
        client_options = sentry_sdk.get_client().options
        should_track = client_options.get("auto_session_tracking", False)

    return should_track


@contextmanager
def auto_session_tracking_scope(scope, session_mode="application"):
    # type: (sentry_sdk.Scope, str) -> Generator[None, None, None]
    """DEPRECATED: This function is a deprecated alias for track_session.
    Starts and stops a session automatically around a block.
    """

    warnings.warn(
        "This function is a deprecated alias for track_session and will be removed in the next major release.",
        DeprecationWarning,
        stacklevel=2,
    )

    with track_session(scope, session_mode=session_mode):
        yield


@contextmanager
def track_session(scope, session_mode="application"):
    # type: (sentry_sdk.Scope, str) -> Generator[None, None, None]
    """
    Start a new session in the provided scope, assuming session tracking is enabled.
    This is a no-op context manager if session tracking is not enabled.
    """

    should_track = _is_auto_session_tracking_enabled(scope)
    if should_track:
        scope.start_session(session_mode=session_mode)
    try:
        yield
    finally:
        if should_track:
            scope.end_session()


TERMINAL_SESSION_STATES = ("exited", "abnormal", "crashed")
MAX_ENVELOPE_ITEMS = 100


def make_aggregate_envelope(aggregate_states, attrs):
    # type: (Any, Any) -> Any
    return {"attrs": dict(attrs), "aggregates": list(aggregate_states.values())}


class SessionFlusher:
    def __init__(
        self,
        capture_func,  # type: Callable[[Envelope], None]
        flush_interval=60,  # type: int
    ):
        # type: (...) -> None
        self.capture_func = capture_func
        self.flush_interval = flush_interval
        self.pending_sessions = []  # type: List[Any]
        self.pending_aggregates = {}  # type: Dict[Any, Any]
        self._thread = None  # type: Optional[Thread]
        self._thread_lock = Lock()
        self._aggregate_lock = Lock()
        self._thread_for_pid = None  # type: Optional[int]
        self._running = True

    def flush(self):
        # type: (...) -> None
        pending_sessions = self.pending_sessions
        self.pending_sessions = []

        with self._aggregate_lock:
            pending_aggregates = self.pending_aggregates
            self.pending_aggregates = {}

        envelope = Envelope()
        for session in pending_sessions:
            if len(envelope.items) == MAX_ENVELOPE_ITEMS:
                self.capture_func(envelope)
                envelope = Envelope()

            envelope.add_session(session)

        for attrs, states in pending_aggregates.items():
            if len(envelope.items) == MAX_ENVELOPE_ITEMS:
                self.capture_func(envelope)
                envelope = Envelope()

            envelope.add_sessions(make_aggregate_envelope(states, attrs))

        if len(envelope.items) > 0:
            self.capture_func(envelope)

    def _ensure_running(self):
        # type: (...) -> None
        """
        Check that we have an active thread to run in, or create one if not.

        Note that this might fail (e.g. in Python 3.12 it's not possible to
        spawn new threads at interpreter shutdown). In that case self._running
        will be False after running this function.
        """
        if self._thread_for_pid == os.getpid() and self._thread is not None:
            return None
        with self._thread_lock:
            if self._thread_for_pid == os.getpid() and self._thread is not None:
                return None

            def _thread():
                # type: (...) -> None
                while self._running:
                    time.sleep(self.flush_interval)
                    if self._running:
                        self.flush()

            thread = Thread(target=_thread)
            thread.daemon = True
            try:
                thread.start()
            except RuntimeError:
                # Unfortunately at this point the interpreter is in a state that no
                # longer allows us to spawn a thread and we have to bail.
                self._running = False
                return None

            self._thread = thread
            self._thread_for_pid = os.getpid()

        return None

    def add_aggregate_session(
        self, session  # type: Session
    ):
        # type: (...) -> None
        # NOTE on `session.did`:
        # the protocol can deal with buckets that have a distinct-id, however
        # in practice we expect the python SDK to have an extremely high cardinality
        # here, effectively making aggregation useless, therefore we do not
        # aggregate per-did.

        # For this part we can get away with using the global interpreter lock
        with self._aggregate_lock:
            attrs = session.get_json_attrs(with_user_info=False)
            primary_key = tuple(sorted(attrs.items()))
            secondary_key = session.truncated_started  # (, session.did)
            states = self.pending_aggregates.setdefault(primary_key, {})
            state = states.setdefault(secondary_key, {})

            if "started" not in state:
                state["started"] = format_timestamp(session.truncated_started)
            # if session.did is not None:
            #     state["did"] = session.did
            if session.status == "crashed":
                state["crashed"] = state.get("crashed", 0) + 1
            elif session.status == "abnormal":
                state["abnormal"] = state.get("abnormal", 0) + 1
            elif session.errors > 0:
                state["errored"] = state.get("errored", 0) + 1
            else:
                state["exited"] = state.get("exited", 0) + 1

    def add_session(
        self, session  # type: Session
    ):
        # type: (...) -> None
        if session.session_mode == "request":
            self.add_aggregate_session(session)
        else:
            self.pending_sessions.append(session.to_json())
        self._ensure_running()

    def kill(self):
        # type: (...) -> None
        self._running = False

    def __del__(self):
        # type: (...) -> None
        self.kill()
