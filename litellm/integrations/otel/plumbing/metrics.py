"""GenAI client metrics: the six ``gen_ai.client.*`` histograms plus the
recorder that builds attributes, applies the shared cardinality filter, and
records a request's metrics on both the success and the failure path.

The instrument names/units/descriptions and the recording + timing math mirror
the v1 :mod:`litellm.integrations.opentelemetry` integration so both engines emit
identical metrics. The attribute cardinality filter is reused from v1 by import
(no duplication of the valid-name set or its validation).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, TypeAlias

from opentelemetry.metrics import Histogram, Meter

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.opentelemetry import (
    METRIC_METADATA_KEYS,
    TOKEN_TYPE_ATTRIBUTE,
    _build_metric_attribute_filter,
    _resolve_metric_attribute_filter,
)
from litellm.integrations.otel.model.metadata import time_to_first_chunk_seconds
from litellm.integrations.otel.model.semconv import (
    Error,
    GenAI,
    Metric,
    resolve_operation,
    resolve_provider,
)
from litellm.integrations.otel.model.utils import to_seconds
from litellm.litellm_core_utils.internal_call_metadata import is_unbilled_non_inference_call_from_params
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps


def _provider_attributes(custom_llm_provider: object) -> Mapping[str, str]:
    """The provider labels for one call's metrics.

    ``gen_ai.provider.name`` carries the semconv-mapped value; the deprecated
    ``gen_ai.system`` spelling is dual-emitted with the raw litellm provider
    string it has always carried, so a dashboard already querying it keeps
    matching. A call with no provider gets neither label: a placeholder value
    would mint a permanent series that no operator can act on.
    """
    if not isinstance(custom_llm_provider, str) or not custom_llm_provider:
        return {}
    return {
        GenAI.PROVIDER_NAME: resolve_provider(custom_llm_provider),
        GenAI.SYSTEM: custom_llm_provider,
    }


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


# A metric datapoint's attributes. Values are the strings the recorder builds, except
# the request model, which is whatever the caller passed and may be absent.
MetricAttributes: TypeAlias = Mapping[str, "str | None"]

ERROR_TYPE_FALLBACK: Final = "_OTHER"

# Every attribute a metric datapoint may carry, on either path. A label value that
# is unique per request is a new time series that will never be written to again, so
# this set is what keeps the series count bounded by the deployment's own
# key/team/user/deployment count rather than by its traffic. Each entry is a fixed
# enum or an operator-provisioned identifier.
#
# Deliberately excluded is everything the *client* supplies or that moves per
# request: ``metadata.requester_metadata`` and ``metadata.spend_logs_metadata`` (both
# free-form from the request body), ``metadata.user_api_key_end_user_id`` (the body's
# ``user`` field), and ``metadata.requester_ip_address``. Those stay on the span,
# where cardinality is free and where they already are.
# ``metadata.user_api_key_user_email`` is left out too: it is bounded, but it is PII
# duplicating the user id already here.
#
# This is a CEILING, applied before the operator's own include/exclude filter, so an
# operator can narrow it but never widen it back to an unbounded attribute.
METRIC_ATTRIBUTE_CEILING: Final[frozenset[str]] = frozenset(
    (
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.system",
        "gen_ai.request.model",
        "gen_ai.framework",
        "metadata.user_api_key_hash",
        "metadata.user_api_key_alias",
        "metadata.user_api_key_team_id",
        "metadata.user_api_key_team_alias",
        "metadata.user_api_key_org_id",
        "metadata.user_api_key_user_id",
        "hidden_params",
    )
)

# The only ``hidden_params`` field that becomes part of the ``hidden_params`` label.
# The object as a whole is per-request by construction -- ``response_cost``,
# ``litellm_overhead_time_ms``, ``cache_key``, ``usage_object`` and the provider's
# ``additional_headers`` rate-limit counters all move on every call -- so dumping it
# whole made one series per request out of every instrument.
#
# ``model_id`` is the router's own deployment id, so it is bounded by the deployment
# list and is what a per-deployment panel joins on. ``api_base`` is deliberately NOT
# here even though it names the same thing: it is a documented per-call parameter, so
# in SDK use it is chosen by the caller rather than provisioned by the operator, and a
# caller varying it would put the per-request cardinality straight back.
BOUNDED_HIDDEN_PARAM_KEYS: Final[tuple[str, ...]] = ("model_id",)


def resolve_error_type(kwargs: Mapping[str, Any]) -> str:
    """The ``error.type`` value for a failed request.

    Bounded by construction: the mapped provider exception's class name (the same
    ``error_information.error_class`` the failure span stamps), else the provider
    status code, else the raw exception's class name, else ``_OTHER`` — the value
    the convention reserves for a failure the instrumentation cannot classify. The
    exception *message* is unbounded and never becomes a label; it stays on the
    span and its exception event, where high cardinality is free.
    """
    std_log: Final = kwargs.get("standard_logging_object")
    info: Final = getattr(std_log, "error_information", None) or (std_log or {}).get("error_information") or {}
    error_class: Final = info.get("error_class") or info.get("error_code")
    if error_class:
        return str(error_class)
    exception: Final = kwargs.get("exception")
    if exception is not None:
        return type(exception).__name__
    return ERROR_TYPE_FALLBACK


class GenAIMetricRecorder:
    """Records the six GenAI histograms for one successful LLM call, and the
    duration histogram alone for one failed LLM call (see :meth:`record_failure`).

    The cardinality filter is resolved lazily on the first record: the proxy
    populates ``callback_settings.otel.attributes`` after the logger is built, so
    reading it at construction time would miss it. ``gen_ai.token.type`` is added
    to the token-usage attributes after filtering so the input/output split always
    survives.
    """

    def __init__(self, metrics: GenAIMetrics, callback_name: str | None = None) -> None:
        self._metrics = metrics
        self._callback_name = callback_name
        self._include: frozenset[str] | None = None
        self._exclude: frozenset[str] | None = None
        self._filter_resolved = False

    def record(
        self,
        kwargs: Mapping[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        common_attrs: Final = self._filter_attributes(self._bounded_attributes(kwargs))
        duration_s: Final = (end_time - start_time).total_seconds()
        usage_is_replayed: Final = is_unbilled_non_inference_call_from_params(
            kwargs.get("call_type"), kwargs.get("litellm_params"), response_obj
        )

        self._metrics.operation_duration.record(duration_s, attributes=common_attrs)
        if not usage_is_replayed:
            self._record_token_usage(response_obj, common_attrs)

        cost: Final = kwargs.get("response_cost")
        if cost:
            self._metrics.token_cost.record(cost, attributes=common_attrs)

        self._record_time_to_first_token(kwargs, common_attrs)
        if not usage_is_replayed:
            self._record_time_per_output_token(kwargs, response_obj, end_time, duration_s, common_attrs)
        self._record_response_duration(kwargs, end_time, common_attrs)

    def record_failure(
        self,
        kwargs: Mapping[str, Any],
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Record the one metric a failed request can honestly report: the
        operation's duration, tagged with ``error.type``.

        The other five instruments all describe a completed generation and have
        nothing to measure here. litellm hands the failure callback no
        ``response_obj`` at all, so there is no usage to split into input/output
        tokens and no completion-token count to divide generation time by; it also
        zeroes ``response_cost`` on failure. Recording them anyway would put a
        fabricated zero into series that dashboards average.

        The attribute set is :data:`METRIC_ATTRIBUTE_CEILING`, the same cap the
        success path uses. A failure needs no provider spend, so a caller who can put
        a unique value into a client-supplied attribute could mint one histogram
        series per request for free; the cap is what makes that impossible on either
        path.

        ``error.type`` is stamped after both filters, exactly like
        ``gen_ai.token.type``, so an operator's include/exclude list cannot strip
        the discriminator and silently merge failures back into the success series.
        """
        attributes: Final = {
            **self._filter_attributes(self._bounded_attributes(kwargs)),
            Error.TYPE: resolve_error_type(kwargs),
        }
        self._metrics.operation_duration.record((end_time - start_time).total_seconds(), attributes=attributes)

    # ------------------------------------------------------------------ #
    #  Attribute building + cardinality filter
    # ------------------------------------------------------------------ #

    def _common_attributes(self, kwargs: Mapping[str, Any]) -> dict:
        params: Final = kwargs.get("litellm_params") or {}
        common_attrs: Final[dict] = {
            GenAI.OPERATION_NAME: resolve_operation(kwargs.get("call_type")).value,
            **_provider_attributes(params.get("custom_llm_provider")),
            GenAI.REQUEST_MODEL: kwargs.get("model"),
            "gen_ai.framework": "litellm",
        }

        std_log: Final = kwargs.get("standard_logging_object")
        md: Final = getattr(std_log, "metadata", None) or (std_log or {}).get("metadata", {})
        for key in METRIC_METADATA_KEYS:
            value = md.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                common_attrs[f"metadata.{key}"] = safe_dumps(value)
            else:
                common_attrs[f"metadata.{key}"] = str(value)

        hidden_params: Final = getattr(std_log, "hidden_params", None) or (std_log or {}).get("hidden_params", {})
        bounded_hidden_params: Final = {
            key: hidden_params[key]
            for key in BOUNDED_HIDDEN_PARAM_KEYS
            if isinstance(hidden_params, Mapping) and hidden_params.get(key) is not None
        }
        if bounded_hidden_params:
            common_attrs["hidden_params"] = safe_dumps(bounded_hidden_params)

        return common_attrs

    def _bounded_attributes(self, kwargs: Mapping[str, Any]) -> MetricAttributes:
        """The datapoint attributes, capped at :data:`METRIC_ATTRIBUTE_CEILING`.

        The cap runs BEFORE the operator's include/exclude filter so the filter can
        only narrow it. An operator who names an excluded attribute in an include
        list gets nothing for it rather than reintroducing an unbounded label.
        """
        return {k: v for k, v in self._common_attributes(kwargs).items() if k in METRIC_ATTRIBUTE_CEILING}

    def _ensure_filter(self) -> None:
        if self._filter_resolved:
            return
        attributes = None
        if self._callback_name in (None, "otel"):
            otel_settings: Final = (litellm.callback_settings or {}).get("otel") or {}
            raw: Final = otel_settings.get("attributes") if isinstance(otel_settings, dict) else None
            if raw is not None:
                attributes = _build_metric_attribute_filter(raw)
        # A bad filter (include_list + exclude_list both set, an unfilterable name)
        # raises here; the caller (logger._record_metrics) surfaces it once at ERROR
        # so the operator-fixable config error is visible. Not cached on the raise
        # path -- _filter_resolved stays False -- so a corrected config takes effect
        # without reconstructing the recorder.
        self._include, self._exclude = _resolve_metric_attribute_filter(attributes)
        self._filter_resolved = True
        self._warn_about_metric_ineligible_names()

    def _warn_about_metric_ineligible_names(self) -> None:
        """Say so when the operator's filter names an attribute the ceiling removes.

        The shared validator accepts every span attribute name, so a name that is
        legal on a span but metric-ineligible would otherwise be a silent no-op: an
        ``include_list`` naming it emits nothing for it and an ``exclude_list`` naming
        it looks like it worked. Logged once, when the filter resolves, rather than
        per request.
        """
        named: Final = (self._include or frozenset()) | (self._exclude or frozenset())
        ineligible: Final = sorted(named - METRIC_ATTRIBUTE_CEILING - {TOKEN_TYPE_ATTRIBUTE})
        if ineligible:
            verbose_logger.warning(
                "OTel metrics: %s cannot be a metric attribute and is being ignored; it varies "
                "per request or is client-supplied, so it would make one time series per request. "
                "It is still on the span. Metric attributes are limited to: %s",
                ", ".join(ineligible),
                ", ".join(sorted(METRIC_ATTRIBUTE_CEILING)),
            )

    def _filter_attributes(self, attrs: MetricAttributes) -> MetricAttributes:
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
        usage: Final = response_obj.get("usage")
        if not usage:
            return
        in_attrs: Final = {**common_attrs, TOKEN_TYPE_ATTRIBUTE: "input"}
        out_attrs: Final = {**common_attrs, TOKEN_TYPE_ATTRIBUTE: "output"}
        self._metrics.token_usage.record(usage.get("prompt_tokens", 0), attributes=in_attrs)
        self._metrics.token_usage.record(usage.get("completion_tokens", 0), attributes=out_attrs)

    def _record_time_to_first_token(self, kwargs: Mapping[str, Any], common_attrs: dict) -> None:
        time_to_first_chunk: Final = time_to_first_chunk_seconds(kwargs)
        if time_to_first_chunk is None:
            return
        self._metrics.time_to_first_token.record(time_to_first_chunk, attributes=common_attrs)

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

        end_ts: Final = to_seconds(end_time)
        if end_ts is None:
            generation_time = duration_s
        else:
            completion_start_time: Final = kwargs.get("completion_start_time")
            api_call_start_time: Final = kwargs.get("api_call_start_time")
            if completion_start_time is not None:
                completion_start: Final = to_seconds(completion_start_time)
                generation_time = duration_s if completion_start is None else end_ts - completion_start
            elif api_call_start_time is not None:
                api_call_start: Final = to_seconds(api_call_start_time)
                generation_time = duration_s if api_call_start is None else end_ts - api_call_start
            else:
                generation_time = duration_s

        if generation_time > 0:
            self._metrics.time_per_output_token.record(generation_time / completion_tokens, attributes=common_attrs)

    def _record_response_duration(self, kwargs: Mapping[str, Any], end_time: datetime, common_attrs: dict) -> None:
        api_call_start_time: Final = kwargs.get("api_call_start_time")
        if api_call_start_time is None:
            return
        _end_time = kwargs.get("end_time") or end_time
        if _end_time is None:
            _end_time = datetime.now()
        api_call_start: Final = to_seconds(api_call_start_time)
        end_ts: Final = to_seconds(_end_time)
        if api_call_start is None or end_ts is None:
            return
        duration: Final = end_ts - api_call_start
        if duration > 0:
            self._metrics.response_duration.record(duration, attributes=common_attrs)
