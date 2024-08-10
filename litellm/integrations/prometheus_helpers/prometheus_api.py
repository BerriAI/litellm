"""
Helper functions to query prometheus API
"""

import asyncio
import os
import time

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

PROMETHEUS_URL = litellm.get_secret("PROMETHEUS_URL")
PROMETHEUS_SELECTED_INSTANCE = litellm.get_secret("PROMETHEUS_SELECTED_INSTANCE")
async_http_handler = AsyncHTTPHandler()


async def get_metric_from_prometheus(
    metric_name: str,
):
    # Get the start of the current day in Unix timestamp
    if PROMETHEUS_URL is None:
        raise ValueError(
            "PROMETHEUS_URL not set please set 'PROMETHEUS_URL=<>' in .env"
        )

    start_of_day = int(time.time()) - (int(time.time()) % 86400)
    query = metric_name
    response = await async_http_handler.get(
        f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}
    )  # End of the day
    _json_response = response.json()
    results = response.json()["data"]["result"]
    return results


async def get_fallback_metric_from_prometheus():
    response_json = await get_metric_from_prometheus(
        metric_name="llm_deployment_successful_fallbacks_total"
    )

    if not response_json:
        return "No fallback data available."

    response_message = ""
    for result in response_json:
        result = response_json
        metric = result["metric"]
        value = int(float(result["value"][1]))  # Convert value to integer

        primary_model = metric.get("primary_model", "Unknown")
        fallback_model = metric.get("fallback_model", "Unknown")
        response_message += f"`{value} successful fallback requests` with `primary model={primary_model} -> fallback model={fallback_model}`"
    return response_message
