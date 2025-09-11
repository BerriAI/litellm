import typing


if typing.TYPE_CHECKING:  # pragma: no cover
    from .. import recorder  # noqa:F401


class ExportError(Exception):
    pass


class Exporter(object):
    """Exporter base class."""

    def export(
        self,
        events,  # type: recorder.EventsType
        start_time_ns,  # type: int
        end_time_ns,  # type: int
    ):
        # type: (...) -> typing.Any
        """Export events.

        :param events: List of events to export.
        :param start_time_ns: The start time of recording.
        :param end_time_ns: The end time of recording.
        """
        raise NotImplementedError


class NullExporter(Exporter):
    """Exporter that does nothing."""

    def export(
        self,
        events,  # type: recorder.EventsType
        start_time_ns,  # type: int
        end_time_ns,  # type: int
    ):
        # type: (...) -> None
        """Discard events."""
        pass
