from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from typing_extensions import ReadOnly, TypedDict

from litellm.types.integrations.custom_logger import StandardCustomLoggerInitParams


class NewRelicInitParams(StandardCustomLoggerInitParams):
    """
    Params for initializing a New Relic logger on litellm
    """


#: Region -> Metric API endpoint. A fixed table by design: team config picks a
#: region enum rather than a free-form endpoint, so callback vars can never
#: redirect metrics to an arbitrary host.
NEWRELIC_METRIC_ENDPOINT_BY_REGION: Final[Mapping[str, str]] = MappingProxyType(
    {
        "us": "https://metric-api.newrelic.com/metric/v1",
        "eu": "https://metric-api.eu.newrelic.com/metric/v1",
    }
)

NEWRELIC_DEFAULT_REGION: Final = "us"

#: Metric API caps a payload at 2000 data points / 1MB compressed; each queued
#: record expands to at most 6 metrics, so cap the per-flush record count well
#: below that.
NEWRELIC_METRICS_MAX_BATCH_SIZE: Final = 250

#: Hard cap on records retained across failed flushes (5xx/network requeue).
#: Beyond this the oldest records are dropped.
NEWRELIC_METRICS_MAX_RETRY_QUEUE_SIZE: Final = 10_000
# Outer passes over a stopped logger's queue: each pass retries the whole
# queue, so records that arrive mid-drain still get attempts before the bounded
# terminal drop. Serialized by a per-logger drain lock, so this bounds work.
NEWRELIC_METRICS_MAX_DRAIN_PASSES: Final = 3
# Metric API caps attribute values; 255 keeps caller-controlled model strings
# from inflating the shared batch payload into a 413
NEWRELIC_METRIC_ATTRIBUTE_MAX_LEN: Final = 255

NEWRELIC_METRIC_REQUESTS: Final = "litellm.requests"
NEWRELIC_METRIC_COST_USD: Final = "litellm.cost.usd"
NEWRELIC_METRIC_PROMPT_TOKENS: Final = "litellm.tokens.prompt"
NEWRELIC_METRIC_COMPLETION_TOKENS: Final = "litellm.tokens.completion"
NEWRELIC_METRIC_TOTAL_TOKENS: Final = "litellm.tokens.total"
NEWRELIC_METRIC_REQUEST_DURATION_MS: Final = "litellm.request.duration_ms"


class NewRelicSummaryValue(TypedDict):
    """Value shape of a Metric API ``summary`` data point."""

    count: ReadOnly[int]
    sum: ReadOnly[float]
    min: ReadOnly[float]
    max: ReadOnly[float]


class NewRelicCountMetric(TypedDict):
    name: ReadOnly[str]
    type: ReadOnly[Literal["count"]]
    value: ReadOnly[float]
    attributes: ReadOnly[Mapping[str, str]]


class NewRelicSummaryMetric(TypedDict):
    name: ReadOnly[str]
    type: ReadOnly[Literal["summary"]]
    value: ReadOnly[NewRelicSummaryValue]
    attributes: ReadOnly[Mapping[str, str]]


NewRelicMetric = NewRelicCountMetric | NewRelicSummaryMetric


#: ``interval.ms`` has a dot in it, so the functional TypedDict form is required.
NewRelicMetricCommon = TypedDict(
    "NewRelicMetricCommon",
    {  # mutable-ok: functional TypedDict requires a dict-literal fields argument ("interval.ms" key)
        "timestamp": ReadOnly[int],
        "interval.ms": ReadOnly[int],
    },
)


class NewRelicMetricEnvelope(TypedDict):
    """One element of the Metric API request body (``[{common, metrics}]``)."""

    common: ReadOnly[NewRelicMetricCommon]
    metrics: ReadOnly[Sequence[NewRelicMetric]]


@dataclass(frozen=True, slots=True)
class NewRelicMetricRecord:
    """One request's contribution to the per-flush aggregation."""

    team_id: str
    team_alias: str
    model_group: str
    model: str
    custom_llm_provider: str
    status: str
    response_cost: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: float

    @property
    def bucket_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.team_id,
            self.team_alias,
            self.model_group,
            self.model,
            self.custom_llm_provider,
            self.status,
        )
