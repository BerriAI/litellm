"""
This integration instruments the ``confluent-kafka<https://github.com/confluentinc/confluent-kafka-python>``
library to trace event streaming.

Enabling
~~~~~~~~

The kafka integration is enabled automatically when using
:ref:`ddtrace-run<ddtracerun>` or :ref:`import ddtrace.auto<ddtraceauto>`.

Or use :func:`patch() <ddtrace.patch>` to manually enable the integration::

    from ddtrace import patch
    patch(kafka=True)
    import confluent_kafka
    ...

Global Configuration
~~~~~~~~~~~~~~~~~~~~

.. py:data:: ddtrace.config.kafka["service"]

   The service name reported by default for your kafka spans.

   This option can also be set with the ``DD_KAFKA_SERVICE`` environment
   variable.

   Default: ``"kafka"``


To configure the kafka integration using the
``Pin`` API::

    from ddtrace import Pin
    from ddtrace import patch

    # Make sure to patch before importing confluent_kafka
    patch(kafka=True)

    import confluent_kafka

    Pin.override(confluent_kafka, service="custom-service-name")
"""
from ddtrace.internal.utils.importlib import require_modules


required_modules = ["confluent_kafka"]

with require_modules(required_modules) as missing_modules:
    if not missing_modules:
        # Required to allow users to import from `ddtrace.contrib.kafka.patch` directly
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", DeprecationWarning)
            from . import patch as _  # noqa: F401, I001

        # Expose public methods
        from ddtrace.contrib.internal.kafka.patch import get_version
        from ddtrace.contrib.internal.kafka.patch import patch
        from ddtrace.contrib.internal.kafka.patch import unpatch

        __all__ = ["patch", "unpatch", "get_version"]
