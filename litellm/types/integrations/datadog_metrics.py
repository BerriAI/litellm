from typing import Literal

from typing_extensions import ReadOnly, TypedDict


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


DatadogDistributionPoint = tuple[int, tuple[float, ...]]


class DatadogDistributionSeries(TypedDict):
    metric: ReadOnly[str]
    type: ReadOnly[Literal["distribution"]]
    points: ReadOnly[tuple[DatadogDistributionPoint, ...]]
    tags: ReadOnly[tuple[str, ...]]


class DatadogDistributionPayload(TypedDict):
    series: ReadOnly[tuple[DatadogDistributionSeries, ...]]
