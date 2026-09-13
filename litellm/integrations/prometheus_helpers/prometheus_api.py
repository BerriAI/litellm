"""
Helper functions to query prometheus API
"""

import json
import time
from datetime import datetime, timedelta
from typing import Final

from pydantic import BaseModel, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm import get_secret
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

PROMETHEUS_URL: Final[str | None] = get_secret("PROMETHEUS_URL")
PROMETHEUS_SELECTED_INSTANCE: Final[str | None] = get_secret("PROMETHEUS_SELECTED_INSTANCE")
async_http_handler: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)

_RAW_JSON_PAYLOAD: Final = TypeAdapter(object)


class PrometheusRangeSample(BaseModel):
    """One ``matrix`` series of the Prometheus HTTP query API."""

    metric: dict[str, object]
    values: list[tuple[float, str]]


class PrometheusQueryData(BaseModel):
    result: list[PrometheusRangeSample]


class PrometheusQueryResponse(BaseModel):
    data: PrometheusQueryData


class PrometheusDailySpend(TypedDict):
    date: ReadOnly[str]
    spend: ReadOnly[float]


async def get_metric_from_prometheus(
    metric_name: str,
) -> list[PrometheusRangeSample]:
    # Get the start of the current day in Unix timestamp
    if PROMETHEUS_URL is None:
        raise ValueError("PROMETHEUS_URL not set please set 'PROMETHEUS_URL=<>' in .env")

    query: Final = f"{metric_name}[24h]"
    now: Final = int(time.time())
    response: Final = await async_http_handler.get(
        f"{PROMETHEUS_URL}/api/v1/query", params={"query": query, "time": now}
    )  # End of the day
    _json_response: Final = _RAW_JSON_PAYLOAD.validate_python(response.json())
    verbose_logger.debug("json response from prometheus /query api %s", _json_response)
    results: Final = PrometheusQueryResponse.model_validate(_json_response).data.result
    return results


async def get_fallback_metric_from_prometheus() -> str:
    """
    Gets fallback metrics from prometheus for the last 24 hours
    """
    response_message = ""
    relevant_metrics: Final = [
        "litellm_deployment_successful_fallbacks_total",
        "litellm_deployment_failed_fallbacks_total",
    ]
    for metric in relevant_metrics:
        response_json = await get_metric_from_prometheus(
            metric_name=metric,
        )

        if response_json:
            verbose_logger.debug("response json %s", response_json)
            for result in response_json:
                verbose_logger.debug("result= %s", result)
                metric_labels = result.metric
                metric_values = result.values
                most_recent_value = metric_values[0]

                if PROMETHEUS_SELECTED_INSTANCE is not None:
                    if metric_labels.get("instance") != PROMETHEUS_SELECTED_INSTANCE:
                        continue

                value = int(float(most_recent_value[1]))  # Convert value to integer
                primary_model = metric_labels.get("primary_model", "Unknown")
                fallback_model = metric_labels.get("fallback_model", "Unknown")
                response_message += f"`{value} successful fallback requests` with primary model=`{primary_model}` -> fallback model=`{fallback_model}`"
                response_message += "\n"
        verbose_logger.debug("response message %s", response_message)
    return response_message


def is_prometheus_connected() -> bool:
    if PROMETHEUS_URL is not None:
        return True
    return False


def _quote_promql_string_literal(value: str) -> str:
    """Render ``value`` as a PromQL double-quoted string literal.

    PromQL string literals follow Go's escape rules
    (https://prometheus.io/docs/prometheus/latest/querying/basics/): a
    backslash begins an escape sequence and a bare ``"`` ends the literal.
    Without escaping, callers that accept arbitrary user-supplied values
    (like the ``api_key`` filter on ``/global/spend/logs``) can inject extra
    label matchers or selectors and read cross-tenant metrics.

    JSON's quoting rules are a strict subset of Go's, so ``json.dumps`` of
    a Python string produces a literal Prometheus accepts: ``\\``, ``\\"``,
    and the standard ``\\n`` / ``\\t`` / ``\\uNNNN`` control-character
    escapes. The returned value already includes the surrounding quotes.
    """
    return json.dumps(value, ensure_ascii=False)


async def get_daily_spend_from_prometheus(api_key: str | None) -> list[PrometheusDailySpend]:
    """
    Expected Response Format:
    [
    {
        "date": "2024-08-18T00:00:00+00:00",
        "spend": 1.001818099998933
    },
    ...]
    """
    if PROMETHEUS_URL is None:
        raise ValueError("PROMETHEUS_URL not set please set 'PROMETHEUS_URL=<>' in .env")

    # Calculate the start and end dates for the last 30 days
    end_date: Final = datetime.utcnow()
    start_date: Final = end_date - timedelta(days=30)

    # Format dates as ISO 8601 strings with UTC offset
    start_str: Final = start_date.isoformat() + "+00:00"
    end_str: Final = end_date.isoformat() + "+00:00"

    url: Final = f"{PROMETHEUS_URL}/api/v1/query_range"

    if api_key is None:
        query = "sum(delta(litellm_spend_metric_total[1d]))"
    else:
        quoted_api_key: Final = _quote_promql_string_literal(api_key)
        query = f"sum(delta(litellm_spend_metric_total{{hashed_api_key={quoted_api_key}}}[1d]))"

    params: Final = {
        "query": query,
        "start": start_str,
        "end": end_str,
        "step": "86400",  # Step size of 1 day in seconds
    }

    response: Final = await async_http_handler.get(url, params=params)
    _json_response: Final = _RAW_JSON_PAYLOAD.validate_python(response.json())
    verbose_logger.debug("json response from prometheus /query api %s", _json_response)
    results: Final = PrometheusQueryResponse.model_validate(_json_response).data.result
    formatted_results: Final[list[PrometheusDailySpend]] = [
        {
            "date": datetime.fromtimestamp(float(timestamp)).isoformat() + "+00:00",
            "spend": float(value),
        }
        for result in results
        for timestamp, value in result.values
    ]

    return formatted_results
