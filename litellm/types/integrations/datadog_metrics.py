from typing_extensions import TypedDict


class DatadogMetricPoint(TypedDict):
    timestamp: int  # Unix epoch seconds
    value: float  # The metric value


class DatadogMetricSeries(TypedDict, total=False):
    metric: str
    type: int  # 0=unspecified, 1=count, 2=rate, 3=gauge
    points: list[DatadogMetricPoint]
    tags: list[str]
    interval: int | None  # Required for count (type=1) and rate (type=2) metrics


class DatadogMetricsPayload(TypedDict):
    series: list[DatadogMetricSeries]
