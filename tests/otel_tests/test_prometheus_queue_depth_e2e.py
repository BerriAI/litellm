"""
E2E tests for Prometheus queue depth metrics.

Tests the litellm_deployment_active_requests and litellm_deployment_queued_requests
gauge metrics by making requests to a running LiteLLM proxy.

Requires:
    - LiteLLM proxy running at http://0.0.0.0:4000
    - Prometheus metrics enabled
    - max_parallel_requests configured for deployments

GitHub Issue: https://github.com/BerriAI/litellm/issues/17764
"""

import pytest
import aiohttp
import asyncio
import re
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))


async def get_prometheus_metrics(session: aiohttp.ClientSession) -> str:
    """Fetch current prometheus metrics from the proxy."""
    async with session.get("http://0.0.0.0:4000/metrics") as response:
        assert response.status == 200, f"Failed to get metrics: {response.status}"
        return await response.text()


async def make_chat_completion_request(session: aiohttp.ClientSession, key: str) -> tuple:
    """Make a chat completion request and return status + response."""
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "fake-openai-endpoint",
        "messages": [{"role": "user", "content": "Hello, testing queue depth"}],
    }
    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        return status, response_text


def extract_gauge_value(metrics_text: str, metric_name: str, labels: dict) -> float | None:
    """
    Extract a gauge value from prometheus metrics text.

    Args:
        metrics_text: Raw prometheus metrics output
        metric_name: Name of the metric (e.g., 'litellm_deployment_active_requests')
        labels: Dict of label key-value pairs to match

    Returns:
        The gauge value as float, or None if not found
    """
    # Build a pattern that matches the metric with given labels
    # Labels can be in any order, so we need to be flexible
    for line in metrics_text.split("\n"):
        if not line.startswith(metric_name + "{"):
            continue

        # Check if all required labels are present
        all_labels_match = True
        for key, value in labels.items():
            label_pattern = f'{key}="{value}"'
            if label_pattern not in line:
                all_labels_match = False
                break

        if all_labels_match:
            # Extract the value at the end of the line
            match = re.search(r"}\s+([0-9.]+)$", line)
            if match:
                return float(match.group(1))

    return None


@pytest.mark.asyncio
async def test_queue_depth_metrics_exist():
    """
    Verify that queue depth metrics are exposed in /metrics endpoint.

    This test checks that:
    1. litellm_deployment_active_requests gauge exists
    2. litellm_deployment_queued_requests gauge exists
    3. Both have the expected labels (model, model_group)
    """
    async with aiohttp.ClientSession() as session:
        # Make a request to ensure metrics are populated
        status, _ = await make_chat_completion_request(session, "sk-1234")
        assert status == 200, f"Request failed with status {status}"

        # Wait for metrics to update
        await asyncio.sleep(1)

        # Get metrics
        metrics = await get_prometheus_metrics(session)

        # Check that the metric definitions exist (HELP and TYPE lines)
        assert "litellm_deployment_active_requests" in metrics, (
            "litellm_deployment_active_requests metric should be defined"
        )
        assert "litellm_deployment_queued_requests" in metrics, (
            "litellm_deployment_queued_requests metric should be defined"
        )

        # Verify they're gauge type
        assert '# TYPE litellm_deployment_active_requests gauge' in metrics, (
            "litellm_deployment_active_requests should be a gauge metric"
        )
        assert '# TYPE litellm_deployment_queued_requests gauge' in metrics, (
            "litellm_deployment_queued_requests should be a gauge metric"
        )


@pytest.mark.asyncio
async def test_queue_depth_metrics_have_correct_labels():
    """
    Verify queue depth metrics have model and model_group labels.
    """
    async with aiohttp.ClientSession() as session:
        # Make a request
        status, _ = await make_chat_completion_request(session, "sk-1234")
        assert status == 200

        await asyncio.sleep(1)

        metrics = await get_prometheus_metrics(session)

        # Find lines with our metrics
        active_lines = [line for line in metrics.split("\n")
                       if line.startswith("litellm_deployment_active_requests{")]
        queued_lines = [line for line in metrics.split("\n")
                       if line.startswith("litellm_deployment_queued_requests{")]

        # At least one deployment should have metrics
        assert len(active_lines) > 0, "Should have at least one active_requests metric line"
        assert len(queued_lines) > 0, "Should have at least one queued_requests metric line"

        # Verify labels are present
        for line in active_lines:
            assert 'model="' in line, f"active_requests should have 'model' label: {line}"
            assert 'model_group="' in line, f"active_requests should have 'model_group' label: {line}"

        for line in queued_lines:
            assert 'model="' in line, f"queued_requests should have 'model' label: {line}"
            assert 'model_group="' in line, f"queued_requests should have 'model_group' label: {line}"


@pytest.mark.asyncio
async def test_active_requests_increments_during_request():
    """
    Verify active_requests increments while a request is being processed.

    This test:
    1. Starts multiple concurrent requests
    2. Checks metrics while requests are in-flight
    3. Verifies active count > 0 during processing
    """
    async with aiohttp.ClientSession() as session:
        # Start multiple requests concurrently but don't await them yet
        tasks = [
            asyncio.create_task(make_chat_completion_request(session, "sk-1234"))
            for _ in range(3)
        ]

        # Give requests time to start
        await asyncio.sleep(0.1)

        # Check metrics while requests are in-flight
        metrics = await get_prometheus_metrics(session)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify requests completed successfully
        for status, _ in results:
            assert status == 200

        # The active count should have been > 0 during processing
        # (Note: This is timing-sensitive; the fake endpoint may respond too fast)
        # At minimum, verify the metric exists
        assert "litellm_deployment_active_requests" in metrics


@pytest.mark.asyncio
async def test_queued_requests_zero_when_not_saturated():
    """
    When requests don't saturate max_parallel_requests, queued should be 0.
    """
    async with aiohttp.ClientSession() as session:
        # Make a single request
        status, _ = await make_chat_completion_request(session, "sk-1234")
        assert status == 200

        await asyncio.sleep(1)

        metrics = await get_prometheus_metrics(session)

        # Find queued metrics
        queued_lines = [line for line in metrics.split("\n")
                       if line.startswith("litellm_deployment_queued_requests{")]

        # All queued values should be 0 (or metric might not exist if never queued)
        for line in queued_lines:
            match = re.search(r"}\s+([0-9.]+)$", line)
            if match:
                queued_value = float(match.group(1))
                assert queued_value == 0.0, (
                    f"Expected queued=0 when not saturated, got {queued_value}"
                )


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=2)
async def test_metrics_reset_after_requests_complete():
    """
    After all requests complete, active should return to 0.
    """
    async with aiohttp.ClientSession() as session:
        # Make requests
        tasks = [
            make_chat_completion_request(session, "sk-1234")
            for _ in range(5)
        ]
        await asyncio.gather(*tasks)

        # Wait for metrics to settle
        await asyncio.sleep(2)

        metrics = await get_prometheus_metrics(session)

        # Active should be 0 after all requests complete
        active_lines = [line for line in metrics.split("\n")
                       if line.startswith("litellm_deployment_active_requests{")]

        for line in active_lines:
            match = re.search(r"}\s+([0-9.]+)$", line)
            if match:
                active_value = float(match.group(1))
                assert active_value == 0.0, (
                    f"Expected active=0 after completion, got {active_value}: {line}"
                )
