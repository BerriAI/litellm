import asyncio
from unittest.mock import patch

import pytest

from litellm.proxy.pass_through_endpoints.llm_provider_handlers.batch_attribution import (
    is_collection_route,
    log_batch_registration_result,
    optional_str,
    request_tags_from_metadata,
)


@pytest.mark.parametrize(
    "value, expected",
    [("a", "a"), ("", ""), (None, None), (7, None), (["a"], None)],
)
def test_optional_str(value, expected):
    assert optional_str(value) == expected


class TestRequestTagsFromMetadata:
    """Tags for the batch-cost spend row. These feed LiteLLM_ManagedObjectTable.request_tags,
    which is the only record of the creating request's tags by the time CheckBatchCost bills
    the batch hours later."""

    @pytest.mark.parametrize(
        "metadata, expected",
        [
            # a request that sent its own tags (x-litellm-tags header or body metadata)
            ({"tags": ["req:a", "req:b"]}, ("req:a", "req:b")),
            # request tags win over the key's own tags
            (
                {"tags": ["req:a"], "user_api_key_auth_metadata": {"tags": ["key:b"]}},
                ("req:a",),
            ),
            # no request tags: fall back to the tags the key itself carries, because a
            # tagged key does not put its tags in the top-level metadata on this path
            ({"user_api_key_auth_metadata": {"tags": ["key:b"]}}, ("key:b",)),
            # an empty request tag list is not a selection, so the key's tags still apply
            (
                {"tags": [], "user_api_key_auth_metadata": {"tags": ["key:b"]}},
                ("key:b",),
            ),
            # neither: no tags on the spend row
            ({}, None),
            # order is preserved, so the spend row is reproducible
            ({"tags": ["z", "a", "m"]}, ("z", "a", "m")),
        ],
    )
    def test_precedence(self, metadata, expected):
        assert request_tags_from_metadata(metadata) == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            # non-string entries are dropped rather than crashing the create
            (["env:prod", 7, None, "team:ml"], ("env:prod", "team:ml")),
            # nothing usable survives, so this is treated as no request tags at all
            ([7, None], None),
            # a non-list is not a tag list
            ("env:prod", None),
            ({"env": "prod"}, None),
            (None, None),
        ],
    )
    def test_malformed_tags_are_dropped(self, raw, expected):
        assert request_tags_from_metadata({"tags": raw}) == expected

    def test_malformed_key_auth_metadata_is_ignored(self):
        assert request_tags_from_metadata({"user_api_key_auth_metadata": "nope"}) is None


@pytest.mark.parametrize(
    "url_route, suffix, expected",
    [
        ("https://api.anthropic.com/v1/messages/batches", "/v1/messages/batches", True),
        ("https://api.anthropic.com/v1/messages/batches/", "/v1/messages/batches", True),
        ("https://api.anthropic.com/v1/messages/batches?limit=20", "/v1/messages/batches", True),
        ("https://api.anthropic.com/v1/messages/batches/msgbatch_1", "/v1/messages/batches", False),
        # a proxied base with a path prefix still resolves, because this is a suffix match
        ("https://gateway.internal/anthropic/v1/messages/batches", "/v1/messages/batches", True),
        ("https://aiplatform.googleapis.com/v1/projects/p/locations/l/batchPredictionJobs", "batchPredictionJobs", True),
        ("https://aiplatform.googleapis.com/v1/projects/p/locations/l/batchPredictionJobs/9", "batchPredictionJobs", False),
    ],
)
def test_is_collection_route(url_route, suffix, expected):
    assert is_collection_route(url_route, suffix) is expected


class TestLogBatchRegistrationResult:
    """The managed object write is fire and forget, so its outcome only ever reaches an
    operator through this log line."""

    @staticmethod
    async def _finished_task(coro):
        task = asyncio.ensure_future(coro)
        await asyncio.gather(task, return_exceptions=True)
        return task

    @pytest.mark.asyncio
    async def test_success_names_the_provider(self):
        async def ok():
            return None

        task = await self._finished_task(ok())
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.batch_attribution.verbose_proxy_logger"
        ) as logger:
            log_batch_registration_result(task, "Anthropic", "uoi", "b1", is_batch_create=True)

        logger.error.assert_not_called()
        logger.info.assert_called_once()
        assert "Anthropic" in logger.info.call_args[0]

    @pytest.mark.asyncio
    async def test_a_failed_create_says_the_cost_is_lost(self):
        async def boom():
            raise RuntimeError("db down")

        task = await self._finished_task(boom())
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.batch_attribution.verbose_proxy_logger"
        ) as logger:
            log_batch_registration_result(task, "Vertex AI", "uoi", "b1", is_batch_create=True)

        logger.info.assert_not_called()
        assert "its cost will not be tracked" in logger.error.call_args[0]

    @pytest.mark.asyncio
    async def test_a_failed_refresh_says_the_row_is_stale(self):
        async def boom():
            raise RuntimeError("db down")

        task = await self._finished_task(boom())
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.batch_attribution.verbose_proxy_logger"
        ) as logger:
            log_batch_registration_result(task, "Vertex AI", "uoi", "b1", is_batch_create=False)

        logger.info.assert_not_called()
        assert "its status and output file may be stale" in logger.error.call_args[0]

    @pytest.mark.asyncio
    async def test_a_cancelled_write_is_reported_not_reraised(self):
        async def slow():
            await asyncio.sleep(60)

        task = asyncio.ensure_future(slow())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.batch_attribution.verbose_proxy_logger"
        ) as logger:
            log_batch_registration_result(task, "Anthropic", "uoi", "b1", is_batch_create=True)

        logger.info.assert_not_called()
        logger.error.assert_called_once()
