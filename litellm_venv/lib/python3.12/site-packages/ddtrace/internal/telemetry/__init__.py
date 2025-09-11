"""
Instrumentation Telemetry API.
This is normally started automatically when ``ddtrace`` is imported. It can be disabled by setting
``DD_INSTRUMENTATION_TELEMETRY_ENABLED`` variable to ``False``.
"""
from .writer import TelemetryWriter


telemetry_writer = TelemetryWriter()  # type: TelemetryWriter

__all__ = ["telemetry_writer"]
