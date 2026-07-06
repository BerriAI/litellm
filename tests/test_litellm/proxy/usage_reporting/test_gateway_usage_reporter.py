"""
Tests for the gateway usage reporter background loop.

Verifies counter operations, payload construction, HTTP posting with auth,
counter preservation on POST failure, shutdown flush, and concurrent
record + drain safety.
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
    _flush_once,
    _post_usage,
    _snapshot_counters,
    _subtract_counters,
    flush_on_shutdown,
    gateway_usage_reporter_loop,
    record_request,
)


@pytest.fixture(autouse=True)
def _reset_counters():
    _counters.total_requests = 0
    _counters.successful_requests = 0
    yield
    _counters.total_requests = 0
    _counters.successful_requests = 0


class TestRecordRequest:
    @pytest.mark.asyncio
    async def test_success_increments_both_counters(self):
        await record_request(succeeded=True)
        assert _counters.total_requests == 1
        assert _counters.successful_requests == 1

    @pytest.mark.asyncio
    async def test_failure_increments_only_total(self):
        await record_request(succeeded=False)
        assert _counters.total_requests == 1
        assert _counters.successful_requests == 0

    @pytest.mark.asyncio
    async def test_mixed_requests(self):
        await record_request(succeeded=True)
        await record_request(succeeded=True)
        await record_request(succeeded=False)
        assert _counters.total_requests == 3
        assert _counters.successful_requests == 2


class TestSnapshotAndSubtract:
    @pytest.mark.asyncio
    async def test_snapshot_returns_copy_without_resetting(self):
        await record_request(succeeded=True)
        await record_request(succeeded=False)

        snapshot = await _snapshot_counters()
        assert snapshot.total_requests == 2
        assert snapshot.successful_requests == 1
        assert _counters.total_requests == 2
        assert _counters.successful_requests == 1

    @pytest.mark.asyncio
    async def test_subtract_removes_snapshot_values(self):
        await record_request(succeeded=True)
        await record_request(succeeded=True)
        await record_request(succeeded=False)

        snapshot = _Counters(total_requests=2, successful_requests=1)
        await _subtract_counters(snapshot)
        assert _counters.total_requests == 1
        assert _counters.successful_requests == 1

    @pytest.mark.asyncio
    async def test_concurrent_record_and_snapshot(self):
        tasks = [record_request(succeeded=True) for _ in range(100)]
        tasks += [record_request(succeeded=False) for _ in range(50)]
        await asyncio.gather(*tasks)

        snapshot = await _snapshot_counters()
        assert snapshot.total_requests == 150
        assert snapshot.successful_requests == 100


class TestBuildPayload:
    def test_payload_fields(self):
        snapshot = _Counters(total_requests=10, successful_requests=7)
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        payload = _build_payload(snapshot, period_start=now, period_end=now)
        assert payload.total_requests == 10
        assert payload.successful_requests == 7
        assert payload.failed_requests == 3
        assert isinstance(payload.hostname, str)
        assert isinstance(payload.pod_id, str)
        assert payload.period_start == now.isoformat()
        assert payload.period_end == now.isoformat()
        assert len(payload.report_id) == 36

    def test_pod_id_from_env(self):
        snapshot = _Counters(total_requests=1, successful_requests=1)
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        with patch.dict(os.environ, {"POD_NAME": "pod-abc-123"}):
            payload = _build_payload(snapshot, period_start=now, period_end=now)
        assert payload.pod_id == "pod-abc-123"

    def test_deployment_id_from_env(self):
        snapshot = _Counters(total_requests=1, successful_requests=1)
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        with patch.dict(os.environ, {"LITELLM_DEPLOYMENT_ID": "deploy-xyz"}):
            payload = _build_payload(snapshot, period_start=now, period_end=now)
        assert payload.deployment_id == "deploy-xyz"

    def test_deployment_id_defaults_empty(self):
        snapshot = _Counters(total_requests=1, successful_requests=1)
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LITELLM_DEPLOYMENT_ID", None)
            payload = _build_payload(snapshot, period_start=now, period_end=now)
        assert payload.deployment_id == ""


class TestPostUsage:
    @pytest.mark.asyncio
    async def test_posts_json_payload_with_auth(self):
        payload = UsageReportPayload(
            total_requests=5,
            successful_requests=4,
            failed_requests=1,
            hostname="test-host",
            pod_id="test-pod",
            deployment_id="deploy-1",
            period_start="2026-01-01T00:00:00+00:00",
            period_end="2026-01-01T00:05:00+00:00",
            report_id="abc-123",
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=None)

        await _post_usage(mock_client, "https://billing.example.com/usage", payload, "my-secret-token")

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["url"] == "https://billing.example.com/usage"
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer my-secret-token"
        posted_json = call_kwargs.kwargs["json"]
        assert posted_json["total_requests"] == 5
        assert posted_json["successful_requests"] == 4
        assert posted_json["failed_requests"] == 1
        assert posted_json["deployment_id"] == "deploy-1"
        assert posted_json["report_id"] == "abc-123"

    @pytest.mark.asyncio
    async def test_posts_without_auth_header_when_no_token(self):
        payload = UsageReportPayload(
            total_requests=1,
            successful_requests=1,
            failed_requests=0,
            hostname="h",
            pod_id="p",
            deployment_id="",
            period_start="",
            period_end="",
            report_id="r",
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=None)

        await _post_usage(mock_client, "https://billing.example.com/usage", payload, "")
        call_kwargs = mock_client.post.call_args
        assert call_kwargs.kwargs["headers"] is None


class TestFlushOnce:
    @pytest.mark.asyncio
    async def test_counters_preserved_on_post_failure(self):
        await record_request(succeeded=True)
        await record_request(succeeded=True)
        await record_request(succeeded=False)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "503",
                request=httpx.Request("POST", "http://x"),
                response=httpx.Response(503),
            )
        )

        with pytest.raises(httpx.HTTPStatusError):
            await _flush_once(mock_client, "https://billing.example.com/usage", "")

        assert _counters.total_requests == 3
        assert _counters.successful_requests == 2

    @pytest.mark.asyncio
    async def test_counters_subtracted_on_success(self):
        await record_request(succeeded=True)
        await record_request(succeeded=False)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=None)

        await _flush_once(mock_client, "https://billing.example.com/usage", "")

        assert _counters.total_requests == 0
        assert _counters.successful_requests == 0

    @pytest.mark.asyncio
    async def test_skips_post_when_no_requests(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=None)

        await _flush_once(mock_client, "https://billing.example.com/usage", "")

        mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_requests_during_flush_not_lost(self):
        await record_request(succeeded=True)
        await record_request(succeeded=True)

        original_post = AsyncMock(return_value=None)

        async def post_with_concurrent_request(**kwargs):
            await record_request(succeeded=True)
            return await original_post(**kwargs)

        mock_client = AsyncMock()
        mock_client.post = post_with_concurrent_request

        await _flush_once(mock_client, "https://billing.example.com/usage", "")

        assert _counters.total_requests == 1
        assert _counters.successful_requests == 1


class TestGatewayUsageReporterLoop:
    @pytest.mark.asyncio
    async def test_loop_exits_immediately_without_env_var(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LITELLM_USAGE_ENDPOINT", None)
            await gateway_usage_reporter_loop()

    @pytest.mark.asyncio
    async def test_loop_posts_accumulated_counts(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=None)

        await record_request(succeeded=True)
        await record_request(succeeded=True)
        await record_request(succeeded=False)

        with patch.dict(
            os.environ,
            {
                "LITELLM_USAGE_ENDPOINT": "https://billing.example.com/usage",
                "LITELLM_USAGE_REPORT_INTERVAL_SECONDS": "0",
            },
        ):
            with patch(
                "litellm.proxy.usage_reporting.gateway_usage_reporter.get_async_httpx_client",
                return_value=mock_client,
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
    async def test_loop_survives_http_error_and_preserves_counters(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "503",
                request=httpx.Request("POST", "http://x"),
                response=httpx.Response(503),
            )
        )

        await record_request(succeeded=True)

        with patch.dict(
            os.environ,
            {
                "LITELLM_USAGE_ENDPOINT": "https://billing.example.com/usage",
                "LITELLM_USAGE_REPORT_INTERVAL_SECONDS": "0",
            },
        ):
            with patch(
                "litellm.proxy.usage_reporting.gateway_usage_reporter.get_async_httpx_client",
                return_value=mock_client,
            ):

                async def cancel_after_one_iteration():
                    await asyncio.sleep(0.1)
                    raise asyncio.CancelledError()

                with pytest.raises(asyncio.CancelledError):
                    await asyncio.gather(
                        gateway_usage_reporter_loop(),
                        cancel_after_one_iteration(),
                    )

        assert _counters.total_requests == 1
        assert _counters.successful_requests == 1


class TestFlushOnShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_flushes_pending_counts(self):
        await record_request(succeeded=True)
        await record_request(succeeded=False)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=None)

        with patch.dict(
            os.environ,
            {"LITELLM_USAGE_ENDPOINT": "https://billing.example.com/usage"},
        ):
            with patch(
                "litellm.proxy.usage_reporting.gateway_usage_reporter.get_async_httpx_client",
                return_value=mock_client,
            ):
                await flush_on_shutdown()

        mock_client.post.assert_called_once()
        posted_json = mock_client.post.call_args.kwargs["json"]
        assert posted_json["total_requests"] == 2
        assert posted_json["successful_requests"] == 1
        assert _counters.total_requests == 0

    @pytest.mark.asyncio
    async def test_shutdown_noop_without_env_var(self):
        await record_request(succeeded=True)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LITELLM_USAGE_ENDPOINT", None)
            await flush_on_shutdown()
        assert _counters.total_requests == 1
