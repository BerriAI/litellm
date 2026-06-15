"""GenAI client metrics: the six ``gen_ai.client.*`` histograms plus the
recorder that builds attributes, applies the shared cardinality filter, and
records a request's metrics in the success path.

The instrument names/units/descriptions and the recording + timing math mirror
the v1 :mod:`litellm.integrations.opentelemetry` integration so both engines emit
identical metrics. The attribute cardinality filter is reused from v1 by import
(no duplication of the valid-name set or its validation).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, FrozenSet, Mapping, Optional

from opentelemetry.metrics import Histogram, Meter

import litellm
from litellm.integrations.opentelemetry import (
    METRIC_METADATA_KEYS,
    TOKEN_TYPE_ATTRIBUTE,
    _build_metric_attribute_filter,
    _resolve_metric_attribute_filter,
)
from litellm.integrations.otel.model.semconv import Metric, resolve_operation
from litellm.integrations.otel.model.utils import to_seconds
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps


@dataclass(frozen=True)
class GenAIMetrics:
    operation_duration: Histogram
    token_usage: Histogram
    token_cost: Histogram
    time_to_first_token: Histogram
    time_per_output_token: Histogram
    response_duration: Histogram


def create_genai_metrics(meter: Meter) -> GenAIMetrics:
    return GenAIMetrics(
        operation_duration=meter.create_histogram(
            name=Metric.OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration",
        ),
        token_usage=meter.create_histogram(
            name=Metric.TOKEN_USAGE,
            unit="{token}",
            description="GenAI token usage",
        ),
        token_cost=meter.create_histogram(
            name=Metric.TOKEN_COST,
            unit="USD",
            description="GenAI request cost",
        ),
        time_to_first_token=meter.create_histogram(
            name=Metric.TIME_TO_FIRST_TOKEN,
            unit="s",
            description="Time to first token for streaming requests",
        ),
        time_per_output_token=meter.create_histogram(
            name=Metric.TIME_PER_OUTPUT_TOKEN,
            unit="s",
            description="Average time per output token (generation time / completion tokens)",
        ),
        response_duration=meter.create_histogram(
            name=Metric.RESPONSE_DURATION,
            unit="s",
            description="Total LLM API generation time (excludes LiteLLM overhead)",
        ),
    )


class GenAIMetricRecorder:
    """Records the six GenAI histograms for one successful LLM call.

    The cardinality filter is resolved lazily on the first record: the proxy
    populates ``callback_settings.otel.attributes`` after the logger is built, so
    reading it at construction time would miss it. ``gen_ai.token.type`` is added
    to the token-usage attributes after filtering so the input/output split always
    survives.
    """

    def __init__(
        self, metrics: GenAIMetrics, callback_name: Optional[str] = None
    ) -> None:
        self._metrics = metrics
        self._callback_name = callback_name
        self._include: Optional[FrozenSet[str]] = None
        self._exclude: Optional[FrozenSet[str]] = None
        self._filter_resolved = False

    def record(
        self,
        kwargs: Mapping[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        common_attrs = self._filter_attributes(self._common_attributes(kwargs))
        duration_s = (end_time - start_time).total_seconds()

        self._metrics.operation_duration.record(duration_s, attributes=common_attrs)
        self._record_token_usage(response_obj, common_attrs)

        cost = kwargs.get("response_cost")
        if cost:
            self._metrics.token_cost.record(cost, attributes=common_attrs)

        self._record_time_to_first_token(kwargs, common_attrs)
        self._record_time_per_output_token(
            kwargs, response_obj, end_time, duration_s, common_attrs
        )
        self._record_response_duration(kwargs, end_time, common_attrs)

    # ------------------------------------------------------------------ #
    #  Attribute building + cardinality filter
    # ------------------------------------------------------------------ #

    def _common_attributes(self, kwargs: Mapping[str, Any]) -> dict:
        params = kwargs.get("litellm_params") or {}
        provider = params.get("custom_llm_provider", "Unknown")
        common_attrs: dict = {
            "gen_ai.operation.name": resolve_operation(kwargs.get("call_type")).value,
            "gen_ai.system": provider,
            "gen_ai.request.model": kwargs.get("model"),
            "gen_ai.framework": "litellm",
        }

        std_log = kwargs.get("standard_logging_object")
        md = getattr(std_log, "metadata", None) or (std_log or {}).get("metadata", {})
        for key in METRIC_METADATA_KEYS:
            value = md.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                common_attrs[f"metadata.{key}"] = safe_dumps(value)
            else:
                common_attrs[f"metadata.{key}"] = str(value)

        hidden_params = getattr(std_log, "hidden_params", None) or (std_log or {}).get(
            "hidden_params", {}
        )
        if hidden_params:
            common_attrs["hidden_params"] = safe_dumps(hidden_params)

        return common_attrs

    def _ensure_filter(self) -> None:
        if self._filter_resolved:
            return
        attributes = None
        if self._callback_name in (None, "otel"):
            otel_settings = (litellm.callback_settings or {}).get("otel") or {}
            raw = (
                otel_settings.get("attributes")
                if isinstance(otel_settings, dict)
                else None
            )
            if raw is not None:
                attributes = _build_metric_attribute_filter(raw)
        # A bad filter (include_list + exclude_list both set, an unfilterable name)
        # raises here; the caller (logger._record_metrics) surfaces it once at ERROR
        # so the operator-fixable config error is visible. Not cached on the raise
        # path -- _filter_resolved stays False -- so a corrected config takes effect
        # without reconstructing the recorder.
        self._include, self._exclude = _resolve_metric_attribute_filter(attributes)
        self._filter_resolved = True

    def _filter_attributes(self, attrs: dict) -> dict:
        self._ensure_filter()
        if self._include is not None:
            return {k: v for k, v in attrs.items() if k in self._include}
        if self._exclude is not None:
            return {k: v for k, v in attrs.items() if k not in self._exclude}
        return attrs

    # ------------------------------------------------------------------ #
    #  Per-metric recording
    # ------------------------------------------------------------------ #

    def _record_token_usage(self, response_obj: Any, common_attrs: dict) -> None:
        if not response_obj:
            return
        usage = response_obj.get("usage")
        if not usage:
            return
        in_attrs = {**common_attrs, TOKEN_TYPE_ATTRIBUTE: "input"}
        out_attrs = {**common_attrs, TOKEN_TYPE_ATTRIBUTE: "output"}
        self._metrics.token_usage.record(
            usage.get("prompt_tokens", 0), attributes=in_attrs
        )
        self._metrics.token_usage.record(
            usage.get("completion_tokens", 0), attributes=out_attrs
        )

    def _record_time_to_first_token(
        self, kwargs: Mapping[str, Any], common_attrs: dict
    ) -> None:
        if not kwargs.get("optional_params", {}).get("stream", False):
            return
        api_call_start = to_seconds(kwargs.get("api_call_start_time"))
        completion_start = to_seconds(kwargs.get("completion_start_time"))
        if api_call_start is None or completion_start is None:
            return
        self._metrics.time_to_first_token.record(
            completion_start - api_call_start, attributes=common_attrs
        )

    def _record_time_per_output_token(
        self,
        kwargs: Mapping[str, Any],
        response_obj: Any,
        end_time: datetime,
        duration_s: float,
        common_attrs: dict,
    ) -> None:
        completion_tokens = None
        if response_obj and (usage := response_obj.get("usage")):
            completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None or completion_tokens <= 0:
            return

        end_ts = to_seconds(end_time)
        if end_ts is None:
            generation_time = duration_s
        else:
            completion_start_time = kwargs.get("completion_start_time")
            api_call_start_time = kwargs.get("api_call_start_time")
            if completion_start_time is not None:
                completion_start = to_seconds(completion_start_time)
                generation_time = (
                    duration_s
                    if completion_start is None
                    else end_ts - completion_start
                )
            elif api_call_start_time is not None:
                api_call_start = to_seconds(api_call_start_time)
                generation_time = (
                    duration_s if api_call_start is None else end_ts - api_call_start
                )
            else:
                generation_time = duration_s

        if generation_time > 0:
            self._metrics.time_per_output_token.record(
                generation_time / completion_tokens, attributes=common_attrs
            )

    def _record_response_duration(
        self, kwargs: Mapping[str, Any], end_time: datetime, common_attrs: dict
    ) -> None:
        api_call_start_time = kwargs.get("api_call_start_time")
        if api_call_start_time is None:
            return
        _end_time = kwargs.get("end_time") or end_time
        if _end_time is None:
            _end_time = datetime.now()
        api_call_start = to_seconds(api_call_start_time)
        end_ts = to_seconds(_end_time)
        if api_call_start is None or end_ts is None:
            return
        duration = end_ts - api_call_start
        if duration > 0:
            self._metrics.response_duration.record(duration, attributes=common_attrs)
