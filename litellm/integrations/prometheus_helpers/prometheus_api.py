"""
Helper functions to query prometheus API
"""

import asyncio
import os
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict

import litellm
from litellm import Router, get_secret
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.prometheus import ModelMetricsData

PROMETHEUS_URL: Optional[str] = get_secret("PROMETHEUS_URL")  # type: ignore

PROMETHEUS_SELECTED_INSTANCE: Optional[str] = get_secret("PROMETHEUS_SELECTED_INSTANCE")  # type: ignore
async_http_handler = get_async_httpx_client(
    llm_provider=httpxSpecialProvider.LoggingCallback
)


async def get_metric_from_prometheus(
    metric_name: str,
):
    # Get the start of the current day in Unix timestamp
    if PROMETHEUS_URL is None:
        raise ValueError(
            "PROMETHEUS_URL not set please set 'PROMETHEUS_URL=<>' in .env"
        )

    query = f"{metric_name}[24h]"
    now = int(time.time())
    response = await async_http_handler.get(
        f"{PROMETHEUS_URL}/api/v1/query", params={"query": query, "time": now}
    )  # End of the day
    _json_response = response.json()
    verbose_logger.debug("json response from prometheus /query api %s", _json_response)
    results = response.json()["data"]["result"]
    return results


async def get_fallback_metric_from_prometheus():
    """
    Gets fallback metrics from prometheus for the last 24 hours
    """
    response_message = ""
    relevant_metrics = [
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
                metric = result["metric"]
                metric_values = result["values"]
                most_recent_value = metric_values[0]

                if PROMETHEUS_SELECTED_INSTANCE is not None:
                    if metric.get("instance") != PROMETHEUS_SELECTED_INSTANCE:
                        continue

                value = int(float(most_recent_value[1]))  # Convert value to integer
                primary_model = metric.get("primary_model", "Unknown")
                fallback_model = metric.get("fallback_model", "Unknown")
                response_message += f"`{value} successful fallback requests` with primary model=`{primary_model}` -> fallback model=`{fallback_model}`"
                response_message += "\n"
        verbose_logger.debug("response message %s", response_message)
    return response_message


def is_prometheus_connected() -> bool:
    if PROMETHEUS_URL is not None:
        return True
    return False


async def get_daily_spend_from_prometheus(api_key: Optional[str]):
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
        raise ValueError(
            "PROMETHEUS_URL not set please set 'PROMETHEUS_URL=<>' in .env"
        )

    # Calculate the start and end dates for the last 30 days
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)

    # Format dates as ISO 8601 strings with UTC offset
    start_str = start_date.isoformat() + "+00:00"
    end_str = end_date.isoformat() + "+00:00"

    url = f"{PROMETHEUS_URL}/api/v1/query_range"

    if api_key is None:
        query = "sum(delta(litellm_spend_metric_total[1d]))"
    else:
        query = (
            f'sum(delta(litellm_spend_metric_total{{hashed_api_key="{api_key}"}}[1d]))'
        )

    params = {
        "query": query,
        "start": start_str,
        "end": end_str,
        "step": "86400",  # Step size of 1 day in seconds
    }

    response = await async_http_handler.get(url, params=params)
    _json_response = response.json()
    verbose_logger.debug("json response from prometheus /query api %s", _json_response)
    results = response.json()["data"]["result"]
    formatted_results = []

    for result in results:
        metric_data = result["values"]
        for timestamp, value in metric_data:
            # Convert timestamp to ISO 8601 string with UTC offset
            date = datetime.fromtimestamp(float(timestamp)).isoformat() + "+00:00"
            spend = float(value)
            formatted_results.append({"date": date, "spend": spend})

    return formatted_results


async def get_model_metrics_from_prometheus(
    router: Router,
    model_group: str,
    start_time: datetime,
    end_time: datetime,
    step: str = "1h",
) -> List[ModelMetricsData]:
    if PROMETHEUS_URL is None:
        raise ValueError(
            "PROMETHEUS_URL not set please set 'PROMETHEUS_URL=<>' in .env"
        )

    model_ids = router.get_model_ids(model_group)
    model_ids_str = "|".join(model_ids)
    query = """histogram_quantile(0.95, 
    avg by (le, model_id, api_base) (
        litellm_deployment_latency_per_output_token_bucket{{
        model_id=~"{model_ids}"
        }}
    )
    )""".format(
        model_ids=model_ids_str
    )

    # Parameters for the query
    params: Dict[str, Any] = {"query": query}

    if start_time and end_time:
        # Range query
        url = f"{PROMETHEUS_URL}/query_range"
        params.update(
            {"start": start_time.timestamp(), "end": end_time.timestamp(), "step": step}
        )
    else:
        # Instant query
        url = f"{PROMETHEUS_URL}/query"
    response = await async_http_handler.get(url, params=params)
    response_json = response.json()

    if response_json.get("status") != "success":
        raise ValueError(f"Error querying prometheus: {response_json}")

    results = response_json["data"]["result"]
    formatted_results: List[ModelMetricsData] = []
    for result in results:
        metric_data = result["values"]
        model_id = result["metric"]["model_id"]
        api_base = result["metric"]["api_base"]
        model_info = router.get_model_info(model_id)
        for model_metric in metric_data:
            day: Optional[date] = None
            avg_latency_per_token: Optional[float] = None
            timestamp, avg_latency_per_token_str = model_metric
            day = datetime.fromtimestamp(timestamp).date()
            avg_latency_per_token = float(avg_latency_per_token_str)
            if day is None:
                raise ValueError(f"Day not found for model_id: {model_id}")

            if avg_latency_per_token is None:
                raise ValueError(
                    f"Avg latency per token not found for model_id: {model_id}"
                )

            if model_info is None:
                raise ValueError(f"Model info not found for model_id: {model_id}")
            model_metrics_data = ModelMetricsData(
                api_base=api_base,
                model_group=model_info["model_name"],
                model=model_info["litellm_params"].get("model") or "",
                day=day,
                avg_latency_per_token=avg_latency_per_token,
            )
            formatted_results.append(model_metrics_data)
    return formatted_results
