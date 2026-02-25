from typing import List

from typing_extensions import TypedDict


class DatadogMetricPoint(TypedDict):
    timestamp: int  # Unix epoch seconds
    value: float  # The metric value


class DatadogMetricSeries(TypedDict):
    metric: str
    type: int  # 1=count, 2=rate, 3=gauge, distribution is submitted as type=3, but distributions use a different endpoint /api/v1/distribution_points, wait actually according to DD /api/v2/series: 0=unspecified, 1=count, 2=rate, 3=gauge. For histogram/distribution we use type 3 or 1.
    points: List[DatadogMetricPoint]
    tags: List[str]


class DatadogMetricsPayload(TypedDict):
    series: List[DatadogMetricSeries]
