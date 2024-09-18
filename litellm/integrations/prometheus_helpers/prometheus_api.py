"""
Helper functions to query prometheus API
"""

import asyncio
import os
import time

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

PROMETHEUS_URL = litellm.get_secret("PROMETHEUS_URL")
PROMETHEUS_SELECTED_INSTANCE = litellm.get_secret("PROMETHEUS_SELECTED_INSTANCE")
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
