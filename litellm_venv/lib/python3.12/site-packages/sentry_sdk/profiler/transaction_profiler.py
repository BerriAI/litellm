"""
This file is originally based on code from https://github.com/nylas/nylas-perftools,
which is published under the following license:

The MIT License (MIT)

Copyright (c) 2014 Nylas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import atexit
import os
import platform
import random
import sys
import threading
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import deque

import sentry_sdk
from sentry_sdk._lru_cache import LRUCache
from sentry_sdk.profiler.utils import (
    DEFAULT_SAMPLING_FREQUENCY,
    extract_stack,
)
from sentry_sdk.utils import (
    capture_internal_exception,
    get_current_thread_meta,
    is_gevent,
    is_valid_sample_rate,
    logger,
    nanosecond_time,
    set_in_app_in_frames,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import Deque
    from typing import Dict
    from typing import List
    from typing import Optional
    from typing import Set
    from typing import Type
    from typing_extensions import TypedDict

    from sentry_sdk.profiler.utils import (
        ProcessedStack,
        ProcessedFrame,
        ProcessedThreadMetadata,
        FrameId,
        StackId,
        ThreadId,
        ExtractedSample,
    )
    from sentry_sdk._types import Event, SamplingContext, ProfilerMode

    ProcessedSample = TypedDict(
        "ProcessedSample",
        {
            "elapsed_since_start_ns": str,
            "thread_id": ThreadId,
            "stack_id": int,
        },
    )

    ProcessedProfile = TypedDict(
        "ProcessedProfile",
        {
            "frames": List[ProcessedFrame],
            "stacks": List[ProcessedStack],
            "samples": List[ProcessedSample],
            "thread_metadata": Dict[ThreadId, ProcessedThreadMetadata],
        },
    )


try:
    from gevent.monkey import get_original
    from gevent.threadpool import ThreadPool as _ThreadPool

    ThreadPool = _ThreadPool  # type: Optional[Type[_ThreadPool]]
    thread_sleep = get_original("time", "sleep")
except ImportError:
    thread_sleep = time.sleep

    ThreadPool = None


_scheduler = None  # type: Optional[Scheduler]


# The minimum number of unique samples that must exist in a profile to be
# considered valid.
PROFILE_MINIMUM_SAMPLES = 2


def has_profiling_enabled(options):
    # type: (Dict[str, Any]) -> bool
    profiles_sampler = options["profiles_sampler"]
    if profiles_sampler is not None:
        return True

    profiles_sample_rate = options["profiles_sample_rate"]
    if profiles_sample_rate is not None and profiles_sample_rate > 0:
        return True

    profiles_sample_rate = options["_experiments"].get("profiles_sample_rate")
    if profiles_sample_rate is not None:
        logger.warning(
            "_experiments['profiles_sample_rate'] is deprecated. "
            "Please use the non-experimental profiles_sample_rate option "
            "directly."
        )
        if profiles_sample_rate > 0:
            return True

    return False


def setup_profiler(options):
    # type: (Dict[str, Any]) -> bool
    global _scheduler

    if _scheduler is not None:
        logger.debug("[Profiling] Profiler is already setup")
        return False

    frequency = DEFAULT_SAMPLING_FREQUENCY

    if is_gevent():
        # If gevent has patched the threading modules then we cannot rely on
        # them to spawn a native thread for sampling.
        # Instead we default to the GeventScheduler which is capable of
        # spawning native threads within gevent.
        default_profiler_mode = GeventScheduler.mode
    else:
        default_profiler_mode = ThreadScheduler.mode

    if options.get("profiler_mode") is not None:
        profiler_mode = options["profiler_mode"]
    else:
        profiler_mode = options.get("_experiments", {}).get("profiler_mode")
        if profiler_mode is not None:
            logger.warning(
                "_experiments['profiler_mode'] is deprecated. Please use the "
                "non-experimental profiler_mode option directly."
            )
        profiler_mode = profiler_mode or default_profiler_mode

    if (
        profiler_mode == ThreadScheduler.mode
        # for legacy reasons, we'll keep supporting sleep mode for this scheduler
        or profiler_mode == "sleep"
    ):
        _scheduler = ThreadScheduler(frequency=frequency)
    elif profiler_mode == GeventScheduler.mode:
        _scheduler = GeventScheduler(frequency=frequency)
    else:
        raise ValueError("Unknown profiler mode: {}".format(profiler_mode))

    logger.debug(
        "[Profiling] Setting up profiler in {mode} mode".format(mode=_scheduler.mode)
    )
    _scheduler.setup()

    atexit.register(teardown_profiler)

    return True


def teardown_profiler():
    # type: () -> None

    global _scheduler

    if _scheduler is not None:
        _scheduler.teardown()

    _scheduler = None


MAX_PROFILE_DURATION_NS = int(3e10)  # 30 seconds


class Profile:
    def __init__(
        self,
        sampled,  # type: Optional[bool]
        start_ns,  # type: int
        hub=None,  # type: Optional[sentry_sdk.Hub]
        scheduler=None,  # type: Optional[Scheduler]
    ):
        # type: (...) -> None
        self.scheduler = _scheduler if scheduler is None else scheduler

        self.event_id = uuid.uuid4().hex  # type: str

        self.sampled = sampled  # type: Optional[bool]

        # Various framework integrations are capable of overwriting the active thread id.
        # If it is set to `None` at the end of the profile, we fall back to the default.
        self._default_active_thread_id = get_current_thread_meta()[0] or 0  # type: int
        self.active_thread_id = None  # type: Optional[int]

        try:
            self.start_ns = start_ns  # type: int
        except AttributeError:
            self.start_ns = 0

        self.stop_ns = 0  # type: int
        self.active = False  # type: bool

        self.indexed_frames = {}  # type: Dict[FrameId, int]
        self.indexed_stacks = {}  # type: Dict[StackId, int]
        self.frames = []  # type: List[ProcessedFrame]
        self.stacks = []  # type: List[ProcessedStack]
        self.samples = []  # type: List[ProcessedSample]

        self.unique_samples = 0

        # Backwards compatibility with the old hub property
        self._hub = None  # type: Optional[sentry_sdk.Hub]
        if hub is not None:
            self._hub = hub
            warnings.warn(
                "The `hub` parameter is deprecated. Please do not use it.",
                DeprecationWarning,
                stacklevel=2,
            )

    def update_active_thread_id(self):
        # type: () -> None
        self.active_thread_id = get_current_thread_meta()[0]
        logger.debug(
            "[Profiling] updating active thread id to {tid}".format(
                tid=self.active_thread_id
            )
        )

    def _set_initial_sampling_decision(self, sampling_context):
        # type: (SamplingContext) -> None
        """
        Sets the profile's sampling decision according to the following
        precedence rules:

        1. If the transaction to be profiled is not sampled, that decision
        will be used, regardless of anything else.

        2. Use `profiles_sample_rate` to decide.
        """

        # The corresponding transaction was not sampled,
        # so don't generate a profile for it.
        if not self.sampled:
            logger.debug(
                "[Profiling] Discarding profile because transaction is discarded."
            )
            self.sampled = False
            return

        # The profiler hasn't been properly initialized.
        if self.scheduler is None:
            logger.debug(
                "[Profiling] Discarding profile because profiler was not started."
            )
            self.sampled = False
            return

        client = sentry_sdk.get_client()
        if not client.is_active():
            self.sampled = False
            return

        options = client.options

        if callable(options.get("profiles_sampler")):
            sample_rate = options["profiles_sampler"](sampling_context)
        elif options["profiles_sample_rate"] is not None:
            sample_rate = options["profiles_sample_rate"]
        else:
            sample_rate = options["_experiments"].get("profiles_sample_rate")

        # The profiles_sample_rate option was not set, so profiling
        # was never enabled.
        if sample_rate is None:
            logger.debug(
                "[Profiling] Discarding profile because profiling was not enabled."
            )
            self.sampled = False
            return

        if not is_valid_sample_rate(sample_rate, source="Profiling"):
            logger.warning(
                "[Profiling] Discarding profile because of invalid sample rate."
            )
            self.sampled = False
            return

        # Now we roll the dice. random.random is inclusive of 0, but not of 1,
        # so strict < is safe here. In case sample_rate is a boolean, cast it
        # to a float (True becomes 1.0 and False becomes 0.0)
        self.sampled = random.random() < float(sample_rate)

        if self.sampled:
            logger.debug("[Profiling] Initializing profile")
        else:
            logger.debug(
                "[Profiling] Discarding profile because it's not included in the random sample (sample rate = {sample_rate})".format(
                    sample_rate=float(sample_rate)
                )
            )

    def start(self):
        # type: () -> None
        if not self.sampled or self.active:
            return

        assert self.scheduler, "No scheduler specified"
        logger.debug("[Profiling] Starting profile")
        self.active = True
        if not self.start_ns:
            self.start_ns = nanosecond_time()
        self.scheduler.start_profiling(self)

    def stop(self):
        # type: () -> None
        if not self.sampled or not self.active:
            return

        assert self.scheduler, "No scheduler specified"
        logger.debug("[Profiling] Stopping profile")
        self.active = False
        self.stop_ns = nanosecond_time()

    def __enter__(self):
        # type: () -> Profile
        scope = sentry_sdk.get_isolation_scope()
        old_profile = scope.profile
        scope.profile = self

        self._context_manager_state = (scope, old_profile)

        self.start()

        return self

    def __exit__(self, ty, value, tb):
        # type: (Optional[Any], Optional[Any], Optional[Any]) -> None
        self.stop()

        scope, old_profile = self._context_manager_state
        del self._context_manager_state

        scope.profile = old_profile

    def write(self, ts, sample):
        # type: (int, ExtractedSample) -> None
        if not self.active:
            return

        if ts < self.start_ns:
            return

        offset = ts - self.start_ns
        if offset > MAX_PROFILE_DURATION_NS:
            self.stop()
            return

        self.unique_samples += 1

        elapsed_since_start_ns = str(offset)

        for tid, (stack_id, frame_ids, frames) in sample:
            try:
                # Check if the stack is indexed first, this lets us skip
                # indexing frames if it's not necessary
                if stack_id not in self.indexed_stacks:
                    for i, frame_id in enumerate(frame_ids):
                        if frame_id not in self.indexed_frames:
                            self.indexed_frames[frame_id] = len(self.indexed_frames)
                            self.frames.append(frames[i])

                    self.indexed_stacks[stack_id] = len(self.indexed_stacks)
                    self.stacks.append(
                        [self.indexed_frames[frame_id] for frame_id in frame_ids]
                    )

                self.samples.append(
                    {
                        "elapsed_since_start_ns": elapsed_since_start_ns,
                        "thread_id": tid,
                        "stack_id": self.indexed_stacks[stack_id],
                    }
                )
            except AttributeError:
                # For some reason, the frame we get doesn't have certain attributes.
                # When this happens, we abandon the current sample as it's bad.
                capture_internal_exception(sys.exc_info())

    def process(self):
        # type: () -> ProcessedProfile

        # This collects the thread metadata at the end of a profile. Doing it
        # this way means that any threads that terminate before the profile ends
        # will not have any metadata associated with it.
        thread_metadata = {
            str(thread.ident): {
                "name": str(thread.name),
            }
            for thread in threading.enumerate()
        }  # type: Dict[str, ProcessedThreadMetadata]

        return {
            "frames": self.frames,
            "stacks": self.stacks,
            "samples": self.samples,
            "thread_metadata": thread_metadata,
        }

    def to_json(self, event_opt, options):
        # type: (Event, Dict[str, Any]) -> Dict[str, Any]
        profile = self.process()

        set_in_app_in_frames(
            profile["frames"],
            options["in_app_exclude"],
            options["in_app_include"],
            options["project_root"],
        )

        return {
            "environment": event_opt.get("environment"),
            "event_id": self.event_id,
            "platform": "python",
            "profile": profile,
            "release": event_opt.get("release", ""),
            "timestamp": event_opt["start_timestamp"],
            "version": "1",
            "device": {
                "architecture": platform.machine(),
            },
            "os": {
                "name": platform.system(),
                "version": platform.release(),
            },
            "runtime": {
                "name": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "transactions": [
                {
                    "id": event_opt["event_id"],
                    "name": event_opt["transaction"],
                    # we start the transaction before the profile and this is
                    # the transaction start time relative to the profile, so we
                    # hardcode it to 0 until we can start the profile before
                    "relative_start_ns": "0",
                    # use the duration of the profile instead of the transaction
                    # because we end the transaction after the profile
                    "relative_end_ns": str(self.stop_ns - self.start_ns),
                    "trace_id": event_opt["contexts"]["trace"]["trace_id"],
                    "active_thread_id": str(
                        self._default_active_thread_id
                        if self.active_thread_id is None
                        else self.active_thread_id
                    ),
                }
            ],
        }

    def valid(self):
        # type: () -> bool
        client = sentry_sdk.get_client()
        if not client.is_active():
            return False

        if not has_profiling_enabled(client.options):
            return False

        if self.sampled is None or not self.sampled:
            if client.transport:
                client.transport.record_lost_event(
                    "sample_rate", data_category="profile"
                )
            return False

        if self.unique_samples < PROFILE_MINIMUM_SAMPLES:
            if client.transport:
                client.transport.record_lost_event(
                    "insufficient_data", data_category="profile"
                )
            logger.debug("[Profiling] Discarding profile because insufficient samples.")
            return False

        return True

    @property
    def hub(self):
        # type: () -> Optional[sentry_sdk.Hub]
        warnings.warn(
            "The `hub` attribute is deprecated. Please do not access it.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._hub

    @hub.setter
    def hub(self, value):
        # type: (Optional[sentry_sdk.Hub]) -> None
        warnings.warn(
            "The `hub` attribute is deprecated. Please do not set it.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._hub = value


class Scheduler(ABC):
    mode = "unknown"  # type: ProfilerMode

    def __init__(self, frequency):
        # type: (int) -> None
        self.interval = 1.0 / frequency

        self.sampler = self.make_sampler()

        # cap the number of new profiles at any time so it does not grow infinitely
        self.new_profiles = deque(maxlen=128)  # type: Deque[Profile]
        self.active_profiles = set()  # type: Set[Profile]

    def __enter__(self):
        # type: () -> Scheduler
        self.setup()
        return self

    def __exit__(self, ty, value, tb):
        # type: (Optional[Any], Optional[Any], Optional[Any]) -> None
        self.teardown()

    @abstractmethod
    def setup(self):
        # type: () -> None
        pass

    @abstractmethod
    def teardown(self):
        # type: () -> None
        pass

    def ensure_running(self):
        # type: () -> None
        """
        Ensure the scheduler is running. By default, this method is a no-op.
        The method should be overridden by any implementation for which it is
        relevant.
        """
        return None

    def start_profiling(self, profile):
        # type: (Profile) -> None
        self.ensure_running()
        self.new_profiles.append(profile)

    def make_sampler(self):
        # type: () -> Callable[..., None]
        cwd = os.getcwd()

        cache = LRUCache(max_size=256)

        def _sample_stack(*args, **kwargs):
            # type: (*Any, **Any) -> None
            """
            Take a sample of the stack on all the threads in the process.
            This should be called at a regular interval to collect samples.
            """
            # no profiles taking place, so we can stop early
            if not self.new_profiles and not self.active_profiles:
                # make sure to clear the cache if we're not profiling so we dont
                # keep a reference to the last stack of frames around
                return

            # This is the number of profiles we want to pop off.
            # It's possible another thread adds a new profile to
            # the list and we spend longer than we want inside
            # the loop below.
            #
            # Also make sure to set this value before extracting
            # frames so we do not write to any new profiles that
            # were started after this point.
            new_profiles = len(self.new_profiles)

            now = nanosecond_time()

            try:
                sample = [
                    (str(tid), extract_stack(frame, cache, cwd))
                    for tid, frame in sys._current_frames().items()
                ]
            except AttributeError:
                # For some reason, the frame we get doesn't have certain attributes.
                # When this happens, we abandon the current sample as it's bad.
                capture_internal_exception(sys.exc_info())
                return

            # Move the new profiles into the active_profiles set.
            #
            # We cannot directly add the to active_profiles set
            # in `start_profiling` because it is called from other
            # threads which can cause a RuntimeError when it the
            # set sizes changes during iteration without a lock.
            #
            # We also want to avoid using a lock here so threads
            # that are starting profiles are not blocked until it
            # can acquire the lock.
            for _ in range(new_profiles):
                self.active_profiles.add(self.new_profiles.popleft())

            inactive_profiles = []

            for profile in self.active_profiles:
                if profile.active:
                    profile.write(now, sample)
                else:
                    # If a thread is marked inactive, we buffer it
                    # to `inactive_profiles` so it can be removed.
                    # We cannot remove it here as it would result
                    # in a RuntimeError.
                    inactive_profiles.append(profile)

            for profile in inactive_profiles:
                self.active_profiles.remove(profile)

        return _sample_stack


class ThreadScheduler(Scheduler):
    """
    This scheduler is based on running a daemon thread that will call
    the sampler at a regular interval.
    """

    mode = "thread"  # type: ProfilerMode
    name = "sentry.profiler.ThreadScheduler"

    def __init__(self, frequency):
        # type: (int) -> None
        super().__init__(frequency=frequency)

        # used to signal to the thread that it should stop
        self.running = False
        self.thread = None  # type: Optional[threading.Thread]
        self.pid = None  # type: Optional[int]
        self.lock = threading.Lock()

    def setup(self):
        # type: () -> None
        pass

    def teardown(self):
        # type: () -> None
        if self.running:
            self.running = False
            if self.thread is not None:
                self.thread.join()

    def ensure_running(self):
        # type: () -> None
        """
        Check that the profiler has an active thread to run in, and start one if
        that's not the case.

        Note that this might fail (e.g. in Python 3.12 it's not possible to
        spawn new threads at interpreter shutdown). In that case self.running
        will be False after running this function.
        """
        pid = os.getpid()

        # is running on the right process
        if self.running and self.pid == pid:
            return

        with self.lock:
            # another thread may have tried to acquire the lock
            # at the same time so it may start another thread
            # make sure to check again before proceeding
            if self.running and self.pid == pid:
                return

            self.pid = pid
            self.running = True

            # make sure the thread is a daemon here otherwise this
            # can keep the application running after other threads
            # have exited
            self.thread = threading.Thread(name=self.name, target=self.run, daemon=True)
            try:
                self.thread.start()
            except RuntimeError:
                # Unfortunately at this point the interpreter is in a state that no
                # longer allows us to spawn a thread and we have to bail.
                self.running = False
                self.thread = None
                return

    def run(self):
        # type: () -> None
        last = time.perf_counter()

        while self.running:
            self.sampler()

            # some time may have elapsed since the last time
            # we sampled, so we need to account for that and
            # not sleep for too long
            elapsed = time.perf_counter() - last
            if elapsed < self.interval:
                thread_sleep(self.interval - elapsed)

            # after sleeping, make sure to take the current
            # timestamp so we can use it next iteration
            last = time.perf_counter()


class GeventScheduler(Scheduler):
    """
    This scheduler is based on the thread scheduler but adapted to work with
    gevent. When using gevent, it may monkey patch the threading modules
    (`threading` and `_thread`). This results in the use of greenlets instead
    of native threads.

    This is an issue because the sampler CANNOT run in a greenlet because
    1. Other greenlets doing sync work will prevent the sampler from running
    2. The greenlet runs in the same thread as other greenlets so when taking
       a sample, other greenlets will have been evicted from the thread. This
       results in a sample containing only the sampler's code.
    """

    mode = "gevent"  # type: ProfilerMode
    name = "sentry.profiler.GeventScheduler"

    def __init__(self, frequency):
        # type: (int) -> None

        if ThreadPool is None:
            raise ValueError("Profiler mode: {} is not available".format(self.mode))

        super().__init__(frequency=frequency)

        # used to signal to the thread that it should stop
        self.running = False
        self.thread = None  # type: Optional[_ThreadPool]
        self.pid = None  # type: Optional[int]

        # This intentionally uses the gevent patched threading.Lock.
        # The lock will be required when first trying to start profiles
        # as we need to spawn the profiler thread from the greenlets.
        self.lock = threading.Lock()

    def setup(self):
        # type: () -> None
        pass

    def teardown(self):
        # type: () -> None
        if self.running:
            self.running = False
            if self.thread is not None:
                self.thread.join()

    def ensure_running(self):
        # type: () -> None
        pid = os.getpid()

        # is running on the right process
        if self.running and self.pid == pid:
            return

        with self.lock:
            # another thread may have tried to acquire the lock
            # at the same time so it may start another thread
            # make sure to check again before proceeding
            if self.running and self.pid == pid:
                return

            self.pid = pid
            self.running = True

            self.thread = ThreadPool(1)  # type: ignore[misc]
            try:
                self.thread.spawn(self.run)
            except RuntimeError:
                # Unfortunately at this point the interpreter is in a state that no
                # longer allows us to spawn a thread and we have to bail.
                self.running = False
                self.thread = None
                return

    def run(self):
        # type: () -> None
        last = time.perf_counter()

        while self.running:
            self.sampler()

            # some time may have elapsed since the last time
            # we sampled, so we need to account for that and
            # not sleep for too long
            elapsed = time.perf_counter() - last
            if elapsed < self.interval:
                thread_sleep(self.interval - elapsed)

            # after sleeping, make sure to take the current
            # timestamp so we can use it next iteration
            last = time.perf_counter()
