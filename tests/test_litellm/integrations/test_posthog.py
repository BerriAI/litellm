"""
Regression tests for the PostHog batch logger.

Background (issue #38904): PostHogLogger.async_send_batch caught every
exception, logged it, and returned normally. CustomBatchLogger.flush_queue
then saw "no exception" and cleared log_queue unconditionally -- so any
batch that failed to send (network error, bad credentials, PostHog outage)
was silently discarded with no retry.

Follow-up (PR #38905 review): naively re-raising on any failure, without
isolating per-credential-group failures, meant a single malformed
credential override (e.g. metadata={"posthog_api_key": {"bad": "key"}})
could crash the whole batch during credential grouping (an unhashable dict
can't be a dict key) *and*, once re-raising preserved the entire queue,
that one bad entry would block delivery for every other tenant's events on
every subsequent flush, forever (modulo CustomBatchLogger's queue-overflow
eviction).

These tests cover: (1) failures are no longer silently swallowed, (2) a
malformed per-request credential override is sanitized before it ever
reaches the queue, and (3) one credential group's failure doesn't block
another credential group's events from being sent and removed from the
queue.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.posthog import PostHogLogger


@pytest.fixture
def posthog_logger(monkeypatch):
    monkeypatch.setenv("POSTHOG_API_KEY", "test-posthog-key")
    monkeypatch.delenv("POSTHOG_API_URL", raising=False)
    logger = PostHogLogger()
    logger.async_client = MagicMock()
    logger.async_client.post = AsyncMock()
    return logger


def _ok_response():
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    return response


def _make_event(tag: str) -> dict:
    return {"event": "$ai_generation", "properties": {"tag": tag}, "distinct_id": tag}


class TestPreserveEventsAddedDuringFlush:
    def test_posthog_logger_opts_into_retry_preservation(self, posthog_logger):
        """CustomBatchLogger only retries failed events for subclasses that
        opt in via this flag -- PostHogLogger must opt in, or
        flush_queue clears log_queue even when async_send_batch raises."""
        assert posthog_logger.preserve_events_added_during_flush is True


class TestGetCredentialsForRequestSanitizesOverrides:
    def test_non_string_api_key_override_falls_back_to_default(self, posthog_logger):
        """#38905 review: metadata={"posthog_api_key": {"bad": "key"}} must
        never produce a non-string api_key -- a dict can't be hashed as
        part of the (api_key, api_url) grouping key in async_send_batch."""
        kwargs = {
            "standard_callback_dynamic_params": {
                "posthog_api_key": {"bad": "key"},
                "posthog_api_url": None,
            }
        }

        api_key, api_url = posthog_logger._get_credentials_for_request(kwargs)

        assert api_key == "test-posthog-key"
        assert isinstance(api_key, str)
        assert isinstance(api_url, str)

    def test_non_string_api_url_override_falls_back_to_default(self, posthog_logger):
        kwargs = {
            "standard_callback_dynamic_params": {
                "posthog_api_url": ["not", "a", "url"],
            }
        }

        _, api_url = posthog_logger._get_credentials_for_request(kwargs)

        assert api_url == posthog_logger.posthog_host
        assert isinstance(api_url, str)

    def test_valid_string_override_is_still_honored(self, posthog_logger):
        """The fix must not break legitimate per-tenant credential overrides."""
        kwargs = {
            "standard_callback_dynamic_params": {
                "posthog_api_key": "tenant-specific-key",
                "posthog_api_url": "https://tenant.posthog.example.com",
            }
        }

        api_key, api_url = posthog_logger._get_credentials_for_request(kwargs)

        assert api_key == "tenant-specific-key"
        assert api_url == "https://tenant.posthog.example.com"

    def test_no_dynamic_params_uses_instance_defaults(self, posthog_logger):
        api_key, api_url = posthog_logger._get_credentials_for_request({})

        assert api_key == "test-posthog-key"
        assert api_url == posthog_logger.posthog_host


class TestAsyncSendBatchDoesNotSilentlyDropEvents:
    @pytest.mark.asyncio
    async def test_async_send_batch_raises_on_http_failure(self, posthog_logger):
        """The original bug: a failed send was caught and only logged,
        never raised, so CustomBatchLogger.flush_queue couldn't tell the
        send had failed and cleared log_queue anyway."""
        posthog_logger.async_client.post.side_effect = Exception("connection refused")
        posthog_logger.log_queue = [
            {"event": _make_event("a"), "api_key": "k1", "api_url": "https://posthog.example.com"}
        ]

        with pytest.raises(Exception):
            await posthog_logger.async_send_batch()

    @pytest.mark.asyncio
    async def test_failed_group_events_remain_queued_for_retry(self, posthog_logger):
        posthog_logger.async_client.post.side_effect = Exception("connection refused")
        item = {"event": _make_event("a"), "api_key": "k1", "api_url": "https://posthog.example.com"}
        posthog_logger.log_queue = [item]

        with pytest.raises(Exception):
            await posthog_logger.async_send_batch()

        assert posthog_logger.log_queue == [item]

    @pytest.mark.asyncio
    async def test_async_send_batch_noop_on_empty_queue(self, posthog_logger):
        posthog_logger.log_queue = []

        await posthog_logger.async_send_batch()  # must not raise

        posthog_logger.async_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_send_batch_succeeds_and_clears_queue(self, posthog_logger):
        posthog_logger.async_client.post.return_value = _ok_response()
        posthog_logger.log_queue = [
            {"event": _make_event("a"), "api_key": "k1", "api_url": "https://posthog.example.com"}
        ]

        await posthog_logger.async_send_batch()

        assert posthog_logger.log_queue == []


class TestAsyncSendBatchIsolatesCredentialGroupFailures:
    """#38905 review: 'isolate failed credential groups so one tenant
    cannot block unrelated tenants' telemetry'."""

    @pytest.mark.asyncio
    async def test_one_failing_tenant_does_not_block_another_tenants_events(self, posthog_logger):
        good_item = {"event": _make_event("good"), "api_key": "good-key", "api_url": "https://good.example.com"}
        bad_item = {"event": _make_event("bad"), "api_key": "bad-key", "api_url": "https://bad.example.com"}
        posthog_logger.log_queue = [good_item, bad_item]

        async def post_side_effect(url, content, headers):
            if "good" in url:
                return _ok_response()
            raise Exception("bad tenant's PostHog project is unreachable")

        posthog_logger.async_client.post.side_effect = post_side_effect

        with pytest.raises(Exception):
            await posthog_logger.async_send_batch()

        # The good tenant's event was sent and must not be retried again.
        assert good_item not in posthog_logger.log_queue
        # The bad tenant's event is preserved for retry.
        assert posthog_logger.log_queue == [bad_item]
        # The good tenant's batch was actually sent.
        assert posthog_logger.async_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_events_appended_during_flush_are_not_dropped(self, posthog_logger):
        """Simulates another coroutine appending a new event to log_queue
        while this flush's network call is in flight (a real await point).
        That new event must survive -- it belongs to neither the succeeded
        nor the failed group snapshotted at the start of this flush."""
        existing_item = {"event": _make_event("existing"), "api_key": "k1", "api_url": "https://posthog.example.com"}
        concurrently_added_item = {
            "event": _make_event("concurrent"),
            "api_key": "k1",
            "api_url": "https://posthog.example.com",
        }
        posthog_logger.log_queue = [existing_item]

        async def post_side_effect(url, content, headers):
            # Simulate a concurrent _log_async_event call appending a new
            # event mid-flush, before this group's send resolves.
            posthog_logger.log_queue.append(concurrently_added_item)
            return _ok_response()

        posthog_logger.async_client.post.side_effect = post_side_effect

        await posthog_logger.async_send_batch()

        assert existing_item not in posthog_logger.log_queue
        assert concurrently_added_item in posthog_logger.log_queue


class TestMalformedCredentialsDoNotPoisonTheSharedQueue:
    """End-to-end regression for the exact scenario Veria AI flagged:
    a request with metadata={"posthog_api_key": {"bad": "key"}} must not
    crash batch sending for every other tenant."""

    @pytest.mark.asyncio
    async def test_malformed_credentials_do_not_crash_grouping_or_block_other_tenants(self, posthog_logger):
        malformed_kwargs = {
            "standard_callback_dynamic_params": {"posthog_api_key": {"bad": "key"}},
            "standard_logging_object": {},
        }
        # What _log_async_event would have stored, now sanitized by
        # _get_credentials_for_request instead of storing the raw dict.
        api_key, api_url = posthog_logger._get_credentials_for_request(malformed_kwargs)
        malformed_item = {"event": _make_event("malformed-tenant"), "api_key": api_key, "api_url": api_url}
        good_item = {"event": _make_event("good-tenant"), "api_key": "good-key", "api_url": "https://good.example.com"}

        posthog_logger.log_queue = [malformed_item, good_item]
        posthog_logger.async_client.post.return_value = _ok_response()

        # Must not raise TypeError (unhashable dict as a grouping key) --
        # and since credentials were sanitized to the default string
        # credentials, this "malformed" tenant is just grouped with the
        # instance default and both groups send fine.
        await posthog_logger.async_send_batch()

        assert posthog_logger.log_queue == []
