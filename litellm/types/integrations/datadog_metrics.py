from collections.abc import Sequence
from typing import Literal

from typing_extensions import NotRequired, ReadOnly, TypedDict


class DatadogMetricPoint(TypedDict):
    timestamp: int  # Unix epoch seconds
    value: float  # The metric value


class DatadogMetricSeries(TypedDict):
    metric: ReadOnly[str]
    type: ReadOnly[Literal[0, 1, 2, 3]]  # 0=unspecified, 1=count, 2=rate, 3=gauge
    points: ReadOnly[Sequence[DatadogMetricPoint]]
    tags: ReadOnly[Sequence[str]]
    interval: ReadOnly[NotRequired[int]]  # Required for count (type=1) and rate (type=2) metrics


class DatadogMetricsPayload(TypedDict):
    series: ReadOnly[Sequence[DatadogMetricSeries]]


DatadogDistributionPoint = tuple[int, tuple[float, ...]]


class DatadogDistributionSeries(TypedDict):
    metric: ReadOnly[str]
    type: ReadOnly[Literal["distribution"]]
    points: ReadOnly[tuple[DatadogDistributionPoint, ...]]
    tags: ReadOnly[tuple[str, ...]]


class DatadogDistributionPayload(TypedDict):
    series: ReadOnly[tuple[DatadogDistributionSeries, ...]]
