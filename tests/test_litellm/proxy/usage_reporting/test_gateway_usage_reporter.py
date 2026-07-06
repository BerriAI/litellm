"""
Tests for the gateway usage reporter background loop.

Verifies counter draining, payload construction, HTTP posting, and the
background loop's retry-on-failure behaviour.
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from litellm.proxy.usage_reporting.gateway_usage_reporter import (
    UsageReportPayload,
    _build_payload,
    _Counters,
    _counters,
    _drain_counters,
    _post_usage,
    gateway_usage_reporter_loop,
    record_request,
)


@pytest.fixture(autouse=True)
def reset_counters():
    _counters.total_requests = 0
    _counters.successful_requests = 0
    yield
    _counters.total_requests = 0
    _counters.successful_requests = 0


class TestRecordRequest:
    def test_success_increments_both_counters(self):
        record_request(succeeded=True)
        assert _counters.total_requests == 1
        assert _counters.successful_requests == 1

    def test_failure_increments_only_total(self):
        record_request(succeeded=False)
        assert _counters.total_requests == 1
        assert _counters.successful_requests == 0

    def test_mixed_requests(self):
        record_request(succeeded=True)
        record_request(succeeded=True)
        record_request(succeeded=False)
        assert _counters.total_requests == 3
        assert _counters.successful_requests == 2


class TestDrainCounters:
    @pytest.mark.asyncio
    async def test_drain_returns_snapshot_and_resets(self):
        record_request(succeeded=True)
        record_request(succeeded=False)

        snapshot = await _drain_counters()
        assert snapshot.total_requests == 2
        assert snapshot.successful_requests == 1

        assert _counters.total_requests == 0
        assert _counters.successful_requests == 0

    @pytest.mark.asyncio
    async def test_drain_empty_counters(self):
        snapshot = await _drain_counters()
        assert snapshot.total_requests == 0
        assert snapshot.successful_requests == 0


class TestBuildPayload:
    def test_payload_fields(self):
        snapshot = _Counters(total_requests=10, successful_requests=7)
        payload = _build_payload(snapshot)
        assert payload.total_requests == 10
        assert payload.successful_requests == 7
        assert payload.failed_requests == 3
        assert isinstance(payload.hostname, str)
        assert isinstance(payload.pod_id, str)

    def test_pod_id_from_env(self):
        snapshot = _Counters(total_requests=1, successful_requests=1)
        with patch.dict(os.environ, {"POD_NAME": "pod-abc-123"}):
            payload = _build_payload(snapshot)
        assert payload.pod_id == "pod-abc-123"


class TestPostUsage:
    @pytest.mark.asyncio
    async def test_posts_json_payload(self):
        payload = UsageReportPayload(
            total_requests=5,
            successful_requests=4,
            failed_requests=1,
            hostname="test-host",
            pod_id="test-pod",
        )

        mock_response = AsyncMock()
        mock_response.raise_for_status = lambda: None

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)

        await _post_usage(mock_client, "https://billing.example.com/usage", payload)

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs.args[0] == "https://billing.example.com/usage"
        posted_json = call_kwargs.kwargs["json"]
        assert posted_json["total_requests"] == 5
        assert posted_json["successful_requests"] == 4
        assert posted_json["failed_requests"] == 1
        assert posted_json["hostname"] == "test-host"
        assert posted_json["pod_id"] == "test-pod"


class TestGatewayUsageReporterLoop:
    @pytest.mark.asyncio
    async def test_loop_exits_immediately_without_env_var(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LITELLM_USAGE_ENDPOINT", None)
            await gateway_usage_reporter_loop()

    @pytest.mark.asyncio
    async def test_loop_posts_accumulated_counts(self):
        mock_response = AsyncMock()
        mock_response.raise_for_status = lambda: None

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        record_request(succeeded=True)
        record_request(succeeded=True)
        record_request(succeeded=False)

        with patch.dict(
            os.environ,
            {
                "LITELLM_USAGE_ENDPOINT": "https://billing.example.com/usage",
                "LITELLM_USAGE_REPORT_INTERVAL_SECONDS": "0",
            },
        ):
            with patch(
                "litellm.proxy.usage_reporting.gateway_usage_reporter.httpx.AsyncClient", return_value=mock_client
            ):

                async def cancel_after_one_iteration():
                    await asyncio.sleep(0.1)
                    raise asyncio.CancelledError()

                with pytest.raises(asyncio.CancelledError):
                    await asyncio.gather(
                        gateway_usage_reporter_loop(),
                        cancel_after_one_iteration(),
                    )

        mock_client.post.assert_called_once()
        posted_json = mock_client.post.call_args.kwargs["json"]
        assert posted_json["total_requests"] == 3
        assert posted_json["successful_requests"] == 2
        assert posted_json["failed_requests"] == 1

    @pytest.mark.asyncio
    async def test_loop_skips_post_when_no_requests(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.dict(
            os.environ,
            {
                "LITELLM_USAGE_ENDPOINT": "https://billing.example.com/usage",
                "LITELLM_USAGE_REPORT_INTERVAL_SECONDS": "0",
            },
        ):
            with patch(
                "litellm.proxy.usage_reporting.gateway_usage_reporter.httpx.AsyncClient", return_value=mock_client
            ):

                async def cancel_after_one_iteration():
                    await asyncio.sleep(0.1)
                    raise asyncio.CancelledError()

                with pytest.raises(asyncio.CancelledError):
                    await asyncio.gather(
                        gateway_usage_reporter_loop(),
                        cancel_after_one_iteration(),
                    )

        mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_loop_survives_http_error(self):
        def _raise_503():
            raise httpx.HTTPStatusError("503", request=httpx.Request("POST", "http://x"), response=httpx.Response(503))

        mock_response = AsyncMock()
        mock_response.raise_for_status = _raise_503

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        record_request(succeeded=True)

        with patch.dict(
            os.environ,
            {
                "LITELLM_USAGE_ENDPOINT": "https://billing.example.com/usage",
                "LITELLM_USAGE_REPORT_INTERVAL_SECONDS": "0",
            },
        ):
            with patch(
                "litellm.proxy.usage_reporting.gateway_usage_reporter.httpx.AsyncClient", return_value=mock_client
            ):

                async def cancel_after_one_iteration():
                    await asyncio.sleep(0.1)
                    raise asyncio.CancelledError()

                with pytest.raises(asyncio.CancelledError):
                    await asyncio.gather(
                        gateway_usage_reporter_loop(),
                        cancel_after_one_iteration(),
                    )
