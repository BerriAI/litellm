"""GenAI client metrics (token usage + operation duration histograms)."""

from dataclasses import dataclass

from opentelemetry.metrics import Histogram, Meter

from litellm.integrations.otel.model.semconv import Metric


@dataclass(frozen=True)
class GenAIMetrics:
    token_usage: Histogram
    operation_duration: Histogram


def create_genai_metrics(meter: Meter) -> GenAIMetrics:
    return GenAIMetrics(
        token_usage=meter.create_histogram(
            name=Metric.TOKEN_USAGE,
            unit="{token}",
            description="Number of tokens used per GenAI request.",
        ),
        operation_duration=meter.create_histogram(
            name=Metric.OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration.",
        ),
    )
