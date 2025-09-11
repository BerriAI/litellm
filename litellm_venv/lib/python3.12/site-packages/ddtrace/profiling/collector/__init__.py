# -*- encoding: utf-8 -*-
import typing  # noqa:F401

from ddtrace.internal import periodic
from ddtrace.internal import service
from ddtrace.settings.profiling import config

from .. import event  # noqa:F401
from ..recorder import Recorder


class CollectorError(Exception):
    pass


class CollectorUnavailable(CollectorError):
    pass


class Collector(service.Service):
    """A profile collector."""

    def __init__(self, recorder: Recorder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder

    @staticmethod
    def snapshot():
        """Take a snapshot of collected data.

        :return: A list of sample list to push in the recorder.
        """


class PeriodicCollector(Collector, periodic.PeriodicService):
    """A collector that needs to run periodically."""

    __slots__ = ()

    def periodic(self):
        # type: (...) -> None
        """Collect events and push them into the recorder."""
        for events in self.collect():
            if self.recorder:
                self.recorder.push_events(events)

    def collect(self):
        # type: (...) -> typing.Iterable[typing.Iterable[event.Event]]
        """Collect the actual data.

        :return: A list of event list to push in the recorder.
        """
        raise NotImplementedError


class CaptureSampler(object):
    """Determine the events that should be captured based on a sampling percentage."""

    def __init__(self, capture_pct: float = 100.0):
        if capture_pct < 0 or capture_pct > 100:
            raise ValueError("Capture percentage should be between 0 and 100 included")
        self.capture_pct: float = capture_pct
        self._counter: int = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        attrs_str = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
        return f"{class_name}({attrs_str})"

    def capture(self):
        self._counter += self.capture_pct
        if self._counter >= 100:
            self._counter -= 100
            return True
        return False


class CaptureSamplerCollector(Collector):
    def __init__(self, recorder, capture_pct=config.capture_pct, *args, **kwargs):
        super().__init__(recorder, *args, **kwargs)
        self.capture_pct = capture_pct
        self._capture_sampler = CaptureSampler(self.capture_pct)
