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
        "raw, expected",
        [
            (["bad\x00tag"], ("badtag",)),
            (["\x00leading"], ("leading",)),
            (["trailing\x00"], ("trailing",)),
            (["a\x00b\x00c"], ("abc",)),
            (["\x00"], ("",)),
            # every element, not just the first
            (["clean", "bad\x00tag"], ("clean", "badtag")),
            (["one\x00", "two\x00", "three\x00"], ("one", "two", "three")),
        ],
    )
    def test_nul_bytes_are_stripped_from_request_tags(self, raw, expected):
        """Regression for PostgreSQL 22P05: an unstripped NUL aborts the managed object row
        insert, so the batch is never cost tracked."""
        assert request_tags_from_metadata({"tags": raw}) == expected

    def test_nul_bytes_are_stripped_from_key_tags_fallback(self):
        """Regression for PostgreSQL 22P05: the key-tags fallback shares the same helper."""
        assert request_tags_from_metadata({"user_api_key_auth_metadata": {"tags": ["key\x00tag"]}}) == ("keytag",)


@pytest.mark.parametrize(
    "url_route, suffix, expected",
    [
        ("https://api.anthropic.com/v1/messages/batches", "/v1/messages/batches", True),
        ("https://api.anthropic.com/v1/messages/batches/", "/v1/messages/batches", True),
        ("https://api.anthropic.com/v1/messages/batches?limit=20", "/v1/messages/batches", True),
        ("https://api.anthropic.com/v1/messages/batches/msgbatch_1", "/v1/messages/batches", False),
        # a proxied base with a path prefix still resolves, because this is a suffix match
        ("https://gateway.internal/anthropic/v1/messages/batches", "/v1/messages/batches", True),
        (
            "https://aiplatform.googleapis.com/v1/projects/p/locations/l/batchPredictionJobs",
            "batchPredictionJobs",
            True,
        ),
        (
            "https://aiplatform.googleapis.com/v1/projects/p/locations/l/batchPredictionJobs/9",
            "batchPredictionJobs",
            False,
        ),
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


def _pending_batch(batch_id: str):
    from litellm.types.utils import LiteLLMBatch

    return LiteLLMBatch(
        id=batch_id,
        object="batch",
        status="validating",
        completion_window="24h",
        created_at=1762905329,
        endpoint="/v1/chat/completions",
        input_file_id="files/wireinput123",
    )


def test_unified_batch_object_id_decodes_to_the_deployment_and_provider_batch_id():
    import base64

    from litellm.proxy.openai_files_endpoints.common_utils import (
        get_batch_id_from_unified_batch_id,
        get_model_id_from_unified_batch_id,
    )
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.batch_attribution import unified_batch_object_id

    unified = unified_batch_object_id("gemini/gemini-3.8-flash", "wire7h3x9batch")
    padded = unified + "=" * (-len(unified) % 4)
    decoded = base64.urlsafe_b64decode(padded).decode()
    assert get_model_id_from_unified_batch_id(decoded) == "gemini/gemini-3.8-flash"
    assert get_batch_id_from_unified_batch_id(decoded) == "wire7h3x9batch"


def test_register_passthrough_batch_stores_the_calling_key_team_and_tags():
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.batch_attribution import (
        register_passthrough_batch,
    )

    hook = MagicMock()
    hook.store_unified_object_id = AsyncMock()
    batch_object = _pending_batch("b1")
    metadata = {
        "user_api_key": "hashed-key",
        "user_api_key_user_id": "user-7",
        "user_api_key_team_id": "team-3",
        "tags": ["pipeline:nightly"],
    }

    async def run():
        with patch("litellm.proxy.proxy_server.proxy_logging_obj") as proxy_logging:
            proxy_logging.get_proxy_hook.return_value = hook
            register_passthrough_batch("Gemini", "uoi", batch_object, "b1", metadata)
            await asyncio.sleep(0)
            await asyncio.sleep(0)

    asyncio.run(run())
    hook.store_unified_object_id.assert_awaited_once()
    stored = hook.store_unified_object_id.await_args.kwargs
    assert stored["unified_object_id"] == "uoi"
    assert stored["model_object_id"] == "b1"
    assert stored["file_object"] is batch_object
    assert stored["file_purpose"] == "batch"
    assert stored["persist_attribution"] is True
    assert stored["request_tags"] == ("pipeline:nightly",)
    assert stored["user_api_key_dict"].api_key == "hashed-key"
    assert stored["user_api_key_dict"].user_id == "user-7"
    assert stored["user_api_key_dict"].team_id == "team-3"


def test_register_passthrough_batch_leaves_the_user_unset_for_a_key_without_one():
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.batch_attribution import (
        register_passthrough_batch,
    )

    hook = MagicMock()
    hook.store_unified_object_id = AsyncMock()

    async def run():
        with patch("litellm.proxy.proxy_server.proxy_logging_obj") as proxy_logging:
            proxy_logging.get_proxy_hook.return_value = hook
            register_passthrough_batch("Gemini", "uoi", _pending_batch("b1"), "b1", {"user_api_key": "hashed-key"})
            await asyncio.sleep(0)
            await asyncio.sleep(0)

    asyncio.run(run())
    stored = hook.store_unified_object_id.await_args.kwargs
    assert stored["user_api_key_dict"].user_id is None
    assert stored["user_api_key_dict"].api_key == "hashed-key"


def test_register_passthrough_batch_without_a_managed_files_hook_stores_nothing():
    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.batch_attribution import (
        register_passthrough_batch,
    )

    async def run():
        with (
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as proxy_logging,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.batch_attribution.verbose_proxy_logger"
            ) as logger,
        ):
            proxy_logging.get_proxy_hook.return_value = None
            register_passthrough_batch("Gemini", "uoi", _pending_batch("b1"), "b1", {})
            await asyncio.sleep(0)
            return logger.warning.call_args

    warned = asyncio.run(run())
    assert warned is not None and "Gemini" in warned.args and "b1" in warned.args, warned
