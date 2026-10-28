from typing import Final

from litellm.accounting.types import CostEvent, InboundUsageEvent, Measurement, OutboundUsageEvent


def measurements_for(event: InboundUsageEvent | OutboundUsageEvent | CostEvent) -> tuple[Measurement, ...]:
    """Return the complete contribution set for one event, ready for one apply call."""
    metrics: Final = event.charges if isinstance(event, CostEvent) else event.metrics
    occurred_at: Final = event.occurred_at if isinstance(event, CostEvent) else event.started_at
    return tuple(
        Measurement(source_event_id=event.event_id, occurred_at=occurred_at, dimensions=event.dimensions, metric=metric)
        for metric in metrics or ()
    )
