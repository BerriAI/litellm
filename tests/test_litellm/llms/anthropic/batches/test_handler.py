"""
Unit tests for litellm/llms/anthropic/batches/handler.py

AnthropicBatchesHandler is the HTTP/auth glue for retrieving Anthropic Message
Batches. It resolves credentials, builds the retrieve URL + auth headers via the
provider config, fires a single GET against the async httpx client, and hands the
response to the config's transform. These tests mock ONLY the genuine I/O seams -
the async httpx client (network) and credential resolution (secret managers /
env) - and assert exactly which seam fired, with what URL/headers, and that the
parsed result is the LiteLLMBatch the transform produced.

The sync ``retrieve_batch`` dispatch (``_is_async`` true -> coroutine, false ->
asyncio.run) is exercised directly, mirroring the dispatch-contract discipline in
tests/test_litellm/batches/test_main.py.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.batches.handler import AnthropicBatchesHandler
from litellm.types.utils import LiteLLMBatch


def _ok_batch_response():
    """A real httpx.Response shaped like an Anthropic MessageBatch retrieval."""
    return httpx.Response(
        status_code=200,
        json={
            "id": "msgbatch_abc",
            "processing_status": "ended",
            "created_at": "2024-09-24T10:00:00Z",
            "ended_at": "2024-09-24T11:00:00Z",
            "request_counts": {"succeeded": 2, "errored": 0},
        },
        request=httpx.Request(
            "GET", "https://api.anthropic.com/v1/messages/batches/msgbatch_abc"
        ),
    )


@pytest.fixture
def handler():
    return AnthropicBatchesHandler()


@pytest.fixture
def patched_client():
    """Patch the async httpx client seam; yield the (fake_client, factory)."""
    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=_ok_batch_response())
    with patch(
        "litellm.llms.anthropic.batches.handler.get_async_httpx_client",
        return_value=fake_client,
    ) as factory:
        yield fake_client, factory


@pytest.mark.asyncio
async def test_aretrieve_batch_fires_get_with_correct_url_and_headers(
    handler, patched_client
):
    fake_client, factory = patched_client

    batch = await handler.aretrieve_batch(
        batch_id="msgbatch_abc",
        api_base="https://api.anthropic.com",
        api_key="sk-ant-test",
        timeout=60.0,
        max_retries=0,
    )

    # The single network seam fired exactly once.
    fake_client.get.assert_awaited_once()
    _, call_kwargs = fake_client.get.call_args
    # Exact URL built by get_retrieve_batch_url.
    assert call_kwargs["url"] == (
        "https://api.anthropic.com/v1/messages/batches/msgbatch_abc"
    )
    # Auth + version + beta headers built by validate_environment.
    headers = call_kwargs["headers"]
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["anthropic-beta"] == "message-batches-2024-09-24"

    # Response parsed through the config transform.
    assert isinstance(batch, LiteLLMBatch)
    assert batch.id == "msgbatch_abc"
    assert batch.status == "completed"
    assert batch.request_counts.completed == 2


@pytest.mark.asyncio
async def test_aretrieve_batch_uses_anthropic_provider_for_client(
    handler, patched_client
):
    from litellm.types.utils import LlmProviders

    _, factory = patched_client
    await handler.aretrieve_batch(
        batch_id="msgbatch_abc",
        api_base="https://api.anthropic.com",
        api_key="sk-ant-test",
        timeout=60.0,
        max_retries=0,
    )
    _, kwargs = factory.call_args
    assert kwargs["llm_provider"] == LlmProviders.ANTHROPIC


@pytest.mark.asyncio
async def test_aretrieve_batch_resolves_api_key_from_model_info(
    handler, patched_client
):
    fake_client, _ = patched_client
    # api_key=None -> handler falls back to AnthropicModelInfo.get_api_key().
    with patch.object(
        handler.anthropic_model_info, "get_api_key", return_value="sk-from-env"
    ):
        await handler.aretrieve_batch(
            batch_id="msgbatch_abc",
            api_base="https://api.anthropic.com",
            api_key=None,
            timeout=60.0,
            max_retries=0,
        )
    _, call_kwargs = fake_client.get.call_args
    assert call_kwargs["headers"]["x-api-key"] == "sk-from-env"


@pytest.mark.asyncio
async def test_aretrieve_batch_missing_api_key_raises(handler, patched_client):
    fake_client, _ = patched_client
    # No api_key and resolver yields None -> hard error before any network call.
    with patch.object(
        handler.anthropic_model_info, "get_api_key", return_value=None
    ):
        with pytest.raises(ValueError, match="Missing Anthropic API Key"):
            await handler.aretrieve_batch(
                batch_id="msgbatch_abc",
                api_base="https://api.anthropic.com",
                api_key=None,
                timeout=60.0,
                max_retries=0,
            )
    fake_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_aretrieve_batch_resolves_default_api_base(handler, patched_client):
    fake_client, _ = patched_client
    # api_base=None -> resolved via get_api_base() default before URL build.
    with patch.object(
        handler.anthropic_model_info,
        "get_api_base",
        return_value="https://api.anthropic.com",
    ):
        await handler.aretrieve_batch(
            batch_id="msgbatch_abc",
            api_base=None,
            api_key="sk-ant-test",
            timeout=60.0,
            max_retries=0,
        )
    _, call_kwargs = fake_client.get.call_args
    assert call_kwargs["url"] == (
        "https://api.anthropic.com/v1/messages/batches/msgbatch_abc"
    )


@pytest.mark.asyncio
async def test_aretrieve_batch_raises_for_status(handler):
    # A non-2xx response must surface via raise_for_status (no silent parse).
    error_response = httpx.Response(
        status_code=404,
        json={"error": "not found"},
        request=httpx.Request(
            "GET", "https://api.anthropic.com/v1/messages/batches/missing"
        ),
    )
    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=error_response)
    with patch(
        "litellm.llms.anthropic.batches.handler.get_async_httpx_client",
        return_value=fake_client,
    ):
        with pytest.raises(httpx.HTTPStatusError):
            await handler.aretrieve_batch(
                batch_id="missing",
                api_base="https://api.anthropic.com",
                api_key="sk-ant-test",
                timeout=60.0,
                max_retries=0,
            )


@pytest.mark.asyncio
async def test_aretrieve_batch_invokes_pre_call_logging(handler, patched_client):
    fake_client, _ = patched_client
    logging_obj = MagicMock()
    await handler.aretrieve_batch(
        batch_id="msgbatch_abc",
        api_base="https://api.anthropic.com",
        api_key="sk-ant-test",
        timeout=60.0,
        max_retries=0,
        logging_obj=logging_obj,
    )
    logging_obj.pre_call.assert_called_once()
    pre_kwargs = logging_obj.pre_call.call_args.kwargs
    assert pre_kwargs["input"] == "msgbatch_abc"
    assert pre_kwargs["api_key"] == "sk-ant-test"
    # The logged api_base is the full retrieve URL, not the bare base.
    assert pre_kwargs["additional_args"]["api_base"] == (
        "https://api.anthropic.com/v1/messages/batches/msgbatch_abc"
    )


@pytest.mark.asyncio
async def test_aretrieve_batch_builds_default_logging_obj_when_absent(
    handler, patched_client
):
    # logging_obj=None -> handler constructs a real Logging object; the call
    # must still complete (no AttributeError on a missing logger).
    _, _ = patched_client
    with patch(
        "litellm.litellm_core_utils.litellm_logging.Logging"
    ) as logging_cls:
        logging_cls.return_value = MagicMock()
        batch = await handler.aretrieve_batch(
            batch_id="msgbatch_abc",
            api_base="https://api.anthropic.com",
            api_key="sk-ant-test",
            timeout=60.0,
            max_retries=0,
            logging_obj=None,
        )
    logging_cls.assert_called_once()
    # call_type wires through to the constructed logging object.
    assert logging_cls.call_args.kwargs["call_type"] == "batch_retrieve"
    assert batch.id == "msgbatch_abc"


# =========================================================================== #
# retrieve_batch dispatch (sync wrapper)
# =========================================================================== #


async def test_retrieve_batch_async_returns_coroutine(handler, patched_client):
    # _is_async=True -> returns the un-awaited coroutine (caller awaits it).
    import asyncio

    coro = handler.retrieve_batch(
        _is_async=True,
        batch_id="msgbatch_abc",
        api_base="https://api.anthropic.com",
        api_key="sk-ant-test",
        timeout=60.0,
        max_retries=0,
    )
    assert asyncio.iscoroutine(coro)
    # Await directly - robust under asyncio_mode=auto's session-scoped loop
    # (manually driving get_event_loop().run_until_complete() breaks when prior
    # async tests in the suite have already used/closed that loop).
    batch = await coro
    assert batch.id == "msgbatch_abc"


def test_retrieve_batch_sync_runs_to_result(handler, patched_client):
    # _is_async=False -> asyncio.run(...) returns the resolved LiteLLMBatch.
    batch = handler.retrieve_batch(
        _is_async=False,
        batch_id="msgbatch_abc",
        api_base="https://api.anthropic.com",
        api_key="sk-ant-test",
        timeout=60.0,
        max_retries=0,
    )
    assert isinstance(batch, LiteLLMBatch)
    assert batch.id == "msgbatch_abc"
    assert batch.status == "completed"
