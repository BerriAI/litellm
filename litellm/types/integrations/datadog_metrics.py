from typing import List

from typing_extensions import TypedDict


class DatadogMetricPoint(TypedDict):
    timestamp: int  # Unix epoch seconds
    value: float  # The metric value


class DatadogMetricSeries(TypedDict):
    metric: str
    type: int  # 0=unspecified, 1=count, 2=rate, 3=gauge
    points: List[DatadogMetricPoint]
    tags: List[str]


class DatadogMetricsPayload(TypedDict):
    series: List[DatadogMetricSeries]
