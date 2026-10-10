import inspect
from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from opentelemetry.metrics import Histogram, Meter

_ADVISORY_PARAM: Final = "explicit_bucket_boundaries_advisory"


@cache
def _warn_bucket_advisory_unsupported() -> None:
    verbose_logger.warning(
        "LITELLM_OTEL_SEMCONV_HISTOGRAM_BUCKETS needs opentelemetry-api >= 1.30.0; "
        "histograms keep the SDK default bucket boundaries"
    )


def create_histogram(
    meter: "Meter",
    *,
    name: str,
    unit: str,
    description: str,
    boundaries: Sequence[float] | None = None,
) -> "Histogram":
    if boundaries is None:
        return meter.create_histogram(name=name, unit=unit, description=description)
    if _ADVISORY_PARAM not in inspect.signature(meter.create_histogram).parameters:
        _warn_bucket_advisory_unsupported()
        return meter.create_histogram(name=name, unit=unit, description=description)
    return meter.create_histogram(
        name=name,
        unit=unit,
        description=description,
        explicit_bucket_boundaries_advisory=boundaries,
    )
