# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
from typing import Optional, Union, Mapping, TYPE_CHECKING
from functools import lru_cache

if TYPE_CHECKING:
    from .tracing.opentelemetry import OpenTelemetryTracer


def _get_tracer_impl():
    # Check if OpenTelemetry is available/installed.
    try:
        from .tracing.opentelemetry import OpenTelemetryTracer

        return OpenTelemetryTracer
    except ImportError:
        return None


@lru_cache
def _get_tracer_cached(
    library_name: Optional[str],
    library_version: Optional[str],
    schema_url: Optional[str],
    attributes_key: Optional[frozenset],
) -> Optional["OpenTelemetryTracer"]:
    tracer_impl = _get_tracer_impl()
    if tracer_impl:
        # Convert attributes_key back to dict if needed
        attributes = dict(attributes_key) if attributes_key else None
        return tracer_impl(
            library_name=library_name,
            library_version=library_version,
            schema_url=schema_url,
            attributes=attributes,
        )
    return None


def get_tracer(
    *,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    schema_url: Optional[str] = None,
    attributes: Optional[Mapping[str, Union[str, bool, int, float]]] = None,
) -> Optional["OpenTelemetryTracer"]:
    """Get the OpenTelemetry tracer instance if available.

    If OpenTelemetry is not available, this method will return None. This method caches
    the tracer instance for each unique set of parameters.

    :keyword library_name: The name of the library to use in the tracer.
    :paramtype library_name: str
    :keyword library_version: The version of the library to use in the tracer.
    :paramtype library_version: str
    :keyword schema_url: Specifies the Schema URL of the emitted spans. Defaults to
        "https://opentelemetry.io/schemas/1.23.1".
    :paramtype schema_url: str
    :keyword attributes: Attributes to add to the emitted spans.
    :paramtype attributes: Mapping[str, Union[str, bool, int, float]]
    :return: The OpenTelemetry tracer instance if available.
    :rtype: Optional[~azure.core.tracing.opentelemetry.OpenTelemetryTracer]
    """
    attributes_key = frozenset(attributes.items()) if attributes else None
    return _get_tracer_cached(library_name, library_version, schema_url, attributes_key)
