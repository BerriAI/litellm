from typing import TYPE_CHECKING

import opentracing

import ddtrace


if TYPE_CHECKING:  # pragma: no cover
    from ddtrace.opentracer import Tracer  # noqa:F401


"""
Helper routines for Datadog OpenTracing.
"""


def set_global_tracer(tracer):
    # type: (Tracer) -> None
    """Sets the global tracers to the given tracer."""

    # overwrite the opentracer reference
    opentracing.tracer = tracer

    # overwrite the Datadog tracer reference
    ddtrace.tracer = tracer._dd_tracer
