"""
OpenTelemetry API
=================

The dd-trace-py library provides an implementation of the
`OpenTelemetry API <https://opentelemetry-python.readthedocs.io/en/latest/api/index.html>`_.
When ddtrace OpenTelemetry support is configured, all operations defined in the
OpenTelemetry trace api can be used to create, configure, and propagate a distributed trace.
All operations defined the opentelemetry trace api are configured to use the ddtrace global tracer (``ddtrace.tracer``)
and generate datadog compatible traces. By default all opentelemetry traces are submitted to a Datadog agent.


Configuration
-------------

When using ``ddtrace-run``, OpenTelemetry support can be enabled by setting
the ``DD_TRACE_OTEL_ENABLED`` environment variable to True (the default value is ``False``).

OpenTelemetry support can be enabled programmatically by setting ``DD_TRACE_OTEL_ENABLED=True``
and setting the ``ddtrace.opentelemetry.TracerProvider``. These configurations
must be set before any OpenTelemetry Tracers are initialized::

    import os
    # Must be set before ddtrace is imported!
    os.environ["DD_TRACE_OTEL_ENABLED"] = "true"

    from opentelemetry.trace import set_tracer_provider
    from ddtrace.opentelemetry import TracerProvider

    set_tracer_provider(TracerProvider())

    ...


Usage
-----

Datadog and OpenTelemetry APIs can be used interchangeably::

    # Sample Usage
    from opentelemetry import trace
    import ddtrace

    oteltracer = trace.get_tracer(__name__)

    with oteltracer.start_as_current_span("otel-span") as parent_span:
        parent_span.set_attribute("otel_key", "otel_val")
        with ddtrace.tracer.trace("ddtrace-span") as child_span:
            child_span.set_tag("dd_key", "dd_val")

    @oteltracer.start_as_current_span("span_name")
    def some_function():
        pass


Mapping
-------

The OpenTelemetry API support implementation maps OpenTelemetry spans to Datadog spans. This mapping is described by the following table, using the protocol buffer field names used in `OpenTelemetry <https://github.com/open-telemetry/opentelemetry-proto/blob/724e427879e3d2bae2edc0218fff06e37b9eb46e/opentelemetry/proto/trace/v1/trace.proto#L80>`_ and `Datadog <https://github.com/DataDog/datadog-agent/blob/dc4958d9bf9f0e286a0854569012a3bd3e33e968/pkg/proto/datadog/trace/span.proto#L7>`_.


.. list-table::
    :header-rows: 1
    :widths: 30, 30, 40

    * - OpenTelemetry
      - Datadog
      - Description
    * - ``trace_id``
      - ``traceID``
      -
    * - ``span_id``
      - ``spanID``
      -
    * - ``trace_state``
      - ``meta["tracestate"]``
      - Datadog vendor-specific data is set in trace state using the ``dd=`` prefix
    * - ``parent_span_id``
      - ``parentID``
      -
    * - ``name``
      - ``resource``
      -
    * - ``kind``
      - ``meta["span.kind"]``
      -
    * - ``start_time_unix_nano``
      - ``start``
      -
    * - ``end_time_unix_nano``
      - ``duration``
      - Derived from start and end time
    * - ``attributes[<key>]``
      - ``meta[<key>]``
      - Datadog tags (``meta``) are set for each OpenTelemetry attribute
    * - ``links[]``
      - ``meta["_dd.span_links"]``
      -
    * - ``status``
      - ``error``
      - Derived from status
    * - ``events[]``
      - N/A
      - Span events not supported on the Datadog platform


"""  # noqa: E501

from ddtrace.internal.opentelemetry.trace import TracerProvider


__all__ = [
    "TracerProvider",
]
