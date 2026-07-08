"""Unit tests for ``AzureBatchesAPI`` (litellm/llms/azure/batches/handler.py).

The Azure batches handler is HTTP/auth glue: each public method
(create/retrieve/cancel/list) resolves an Azure OpenAI client via the inherited
``get_azure_openai_client`` seam, branches on ``_is_async`` (returning the
``a*`` coroutine in the async case, calling the sync client otherwise), validates
the client type, and parses the SDK response into ``LiteLLMBatch``.

We mock only true boundaries:
  * ``get_azure_openai_client`` - the credential/client-construction seam. We
    assert the EXACT auth args (api_key / api_base / api_version / client /
    _is_async / litellm_params) forwarded to it.
  * the returned Azure OpenAI client's ``batches.*`` methods - the network call.
    We assert the request data forwarded and that the SDK response is parsed
    into ``LiteLLMBatch`` (sibling SDK methods asserted NOT called).

Pure logic (the _is_async branch, the isinstance guards, the model_dump parse)
runs for real.
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from openai import AsyncOpenAI, OpenAI  # noqa: E402

from litellm.llms.azure.azure import AsyncAzureOpenAI, AzureOpenAI  # noqa: E402
from litellm.llms.azure.batches.handler import AzureBatchesAPI  # noqa: E402
from litellm.types.utils import LiteLLMBatch  # noqa: E402

GET_CLIENT = "litellm.llms.azure.batches.handler.AzureBatchesAPI.get_azure_openai_client"

AUTH_KW = dict(
    api_key="sk-azure-test",
    api_base="https://my-azure.openai.azure.com",
    api_version="2024-12-01",
    timeout=600.0,
    max_retries=3,
)

CREATE_DATA = {
    "completion_window": "24h",
    "endpoint": "/v1/chat/completions",
    "input_file_id": "file-abc",
}
RETRIEVE_DATA = {"batch_id": "batch-123"}
CANCEL_DATA = {"batch_id": "batch-123"}


def _batch_dict(batch_id: str = "batch-123", status: str = "completed") -> dict:
    """A minimal-but-valid dict for ``LiteLLMBatch(**response.model_dump())``."""
    return {
        "id": batch_id,
        "completion_window": "24h",
        "created_at": 1700000000,
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-abc",
        "object": "batch",
        "status": status,
        "output_file_id": "file-out-xyz",
    }


def _sdk_response(batch_dict: dict) -> MagicMock:
    """An object that mimics the OpenAI SDK Batch: only ``.model_dump()`` is used."""
    resp = MagicMock()
    resp.model_dump.return_value = batch_dict
    return resp


def _sync_client() -> MagicMock:
    """A sync Azure client (passes ``isinstance(.., AzureOpenAI)``)."""
    return MagicMock(spec=AzureOpenAI)


def _async_client() -> MagicMock:
    """An async Azure client (passes ``isinstance(.., AsyncAzureOpenAI)``).

    The ``batches.*`` SDK methods are awaited by the handler, so they must be
    AsyncMocks.
    """
    client = MagicMock(spec=AsyncAzureOpenAI)
    client.batches.create = AsyncMock()
    client.batches.retrieve = AsyncMock()
    client.batches.cancel = AsyncMock()
    client.batches.list = AsyncMock()
    return client


@pytest.fixture
def handler() -> AzureBatchesAPI:
    return AzureBatchesAPI()


# =========================================================================== #
# create_batch - sync path
# =========================================================================== #


def test_create_sync_forwards_auth_to_client_seam(handler):
    client = _sync_client()
    client.batches.create.return_value = _sdk_response(_batch_dict())

    with patch(GET_CLIENT, return_value=client) as get_client:
        result = handler.create_batch(
            _is_async=False, create_batch_data=CREATE_DATA, **AUTH_KW
        )

    # EXACT auth args forwarded to the client-construction seam.
    assert get_client.call_count == 1
    kw = get_client.call_args.kwargs
    assert kw["api_key"] == "sk-azure-test"
    assert kw["api_base"] == "https://my-azure.openai.azure.com"
    assert kw["api_version"] == "2024-12-01"
    assert kw["_is_async"] is False
    assert kw["client"] is None
    # litellm_params defaults to {} (not None) when not supplied.
    assert kw["litellm_params"] == {}

    # PAYLOAD: request data forwarded verbatim to the SDK as kwargs.
    client.batches.create.assert_called_once_with(**CREATE_DATA)
    # sibling SDK seams untouched.
    client.batches.retrieve.assert_not_called()
    client.batches.cancel.assert_not_called()

    # RESULT: parsed into LiteLLMBatch from the SDK response's model_dump.
    assert isinstance(result, LiteLLMBatch)
    assert result.id == "batch-123"
    assert result.status == "completed"
    assert result.output_file_id == "file-out-xyz"


def test_create_sync_passes_litellm_params_through(handler):
    client = _sync_client()
    client.batches.create.return_value = _sdk_response(_batch_dict())
    lp = {"azure_ad_token": "tok", "tenant_id": "t1"}

    with patch(GET_CLIENT, return_value=client) as get_client:
        handler.create_batch(
            _is_async=False,
            create_batch_data=CREATE_DATA,
            litellm_params=lp,
            **AUTH_KW,
        )

    assert get_client.call_args.kwargs["litellm_params"] == lp


def test_create_sync_explicit_client_forwarded_to_seam(handler):
    sentinel_client = _sync_client()
    sentinel_client.batches.create.return_value = _sdk_response(_batch_dict())

    with patch(GET_CLIENT, return_value=sentinel_client) as get_client:
        handler.create_batch(
            _is_async=False,
            create_batch_data=CREATE_DATA,
            client=sentinel_client,
            **AUTH_KW,
        )

    assert get_client.call_args.kwargs["client"] is sentinel_client


def test_create_raises_when_client_is_none(handler):
    with patch(GET_CLIENT, return_value=None):
        with pytest.raises(ValueError, match="client is not initialized"):
            handler.create_batch(
                _is_async=False, create_batch_data=CREATE_DATA, **AUTH_KW
            )


# =========================================================================== #
# create_batch - async path
# =========================================================================== #


@pytest.mark.asyncio
async def test_create_async_returns_coroutine_and_awaits_async_client(handler):
    client = _async_client()
    client.batches.create.return_value = _sdk_response(_batch_dict())

    with patch(GET_CLIENT, return_value=client) as get_client:
        coro = handler.create_batch(
            _is_async=True, create_batch_data=CREATE_DATA, **AUTH_KW
        )
        assert asyncio.iscoroutine(coro)
        result = await coro

    assert get_client.call_args.kwargs["_is_async"] is True
    client.batches.create.assert_awaited_once_with(**CREATE_DATA)
    assert isinstance(result, LiteLLMBatch)
    assert result.id == "batch-123"


@pytest.mark.asyncio
async def test_create_async_rejects_sync_client(handler):
    """_is_async=True but seam returns a sync client -> ValueError, no network."""
    sync_client = _sync_client()

    with patch(GET_CLIENT, return_value=sync_client):
        with pytest.raises(ValueError, match="not an instance of AsyncOpenAI"):
            handler.create_batch(
                _is_async=True, create_batch_data=CREATE_DATA, **AUTH_KW
            )

    sync_client.batches.create.assert_not_called()


@pytest.mark.asyncio
async def test_acreate_batch_parses_response(handler):
    client = _async_client()
    client.batches.create.return_value = _sdk_response(_batch_dict(status="validating"))

    result = await handler.acreate_batch(
        create_batch_data=CREATE_DATA, azure_client=client
    )

    client.batches.create.assert_awaited_once_with(**CREATE_DATA)
    assert isinstance(result, LiteLLMBatch)
    assert result.status == "validating"


# =========================================================================== #
# retrieve_batch
# =========================================================================== #


def test_retrieve_sync_dispatch_payload_and_result(handler):
    client = _sync_client()
    client.batches.retrieve.return_value = _sdk_response(_batch_dict())

    with patch(GET_CLIENT, return_value=client) as get_client:
        result = handler.retrieve_batch(
            _is_async=False, retrieve_batch_data=RETRIEVE_DATA, **AUTH_KW
        )

    assert get_client.call_args.kwargs["_is_async"] is False
    client.batches.retrieve.assert_called_once_with(**RETRIEVE_DATA)
    client.batches.create.assert_not_called()
    client.batches.cancel.assert_not_called()
    assert isinstance(result, LiteLLMBatch)
    assert result.id == "batch-123"


def test_retrieve_raises_when_client_is_none(handler):
    with patch(GET_CLIENT, return_value=None):
        with pytest.raises(ValueError, match="client is not initialized"):
            handler.retrieve_batch(
                _is_async=False, retrieve_batch_data=RETRIEVE_DATA, **AUTH_KW
            )


@pytest.mark.asyncio
async def test_retrieve_async_returns_coroutine_and_awaits(handler):
    client = _async_client()
    client.batches.retrieve.return_value = _sdk_response(_batch_dict())

    with patch(GET_CLIENT, return_value=client) as get_client:
        coro = handler.retrieve_batch(
            _is_async=True, retrieve_batch_data=RETRIEVE_DATA, **AUTH_KW
        )
        assert asyncio.iscoroutine(coro)
        result = await coro

    assert get_client.call_args.kwargs["_is_async"] is True
    client.batches.retrieve.assert_awaited_once_with(**RETRIEVE_DATA)
    assert isinstance(result, LiteLLMBatch)


@pytest.mark.asyncio
async def test_retrieve_async_rejects_sync_client(handler):
    sync_client = _sync_client()
    with patch(GET_CLIENT, return_value=sync_client):
        with pytest.raises(ValueError, match="not an instance of AsyncOpenAI"):
            handler.retrieve_batch(
                _is_async=True, retrieve_batch_data=RETRIEVE_DATA, **AUTH_KW
            )
    sync_client.batches.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_aretrieve_batch_parses_response(handler):
    client = _async_client()
    client.batches.retrieve.return_value = _sdk_response(_batch_dict())

    result = await handler.aretrieve_batch(
        retrieve_batch_data=RETRIEVE_DATA, client=client
    )

    client.batches.retrieve.assert_awaited_once_with(**RETRIEVE_DATA)
    assert isinstance(result, LiteLLMBatch)


# =========================================================================== #
# cancel_batch  (has an EXTRA sync-side isinstance guard the others lack)
# =========================================================================== #


def test_cancel_sync_dispatch_payload_and_result(handler):
    client = _sync_client()
    client.batches.cancel.return_value = _sdk_response(_batch_dict(status="cancelled"))

    with patch(GET_CLIENT, return_value=client) as get_client:
        result = handler.cancel_batch(
            _is_async=False, cancel_batch_data=CANCEL_DATA, **AUTH_KW
        )

    assert get_client.call_args.kwargs["_is_async"] is False
    client.batches.cancel.assert_called_once_with(**CANCEL_DATA)
    client.batches.create.assert_not_called()
    client.batches.retrieve.assert_not_called()
    assert isinstance(result, LiteLLMBatch)
    assert result.status == "cancelled"


def test_cancel_raises_when_client_is_none(handler):
    with patch(GET_CLIENT, return_value=None):
        with pytest.raises(ValueError, match="client is not initialized"):
            handler.cancel_batch(
                _is_async=False, cancel_batch_data=CANCEL_DATA, **AUTH_KW
            )


def test_cancel_sync_rejects_non_sync_client(handler):
    """cancel_batch has a unique sync-side guard: if _is_async is False but the
    resolved client is async (neither AzureOpenAI nor OpenAI), it must raise
    rather than call .cancel()."""
    async_client = _async_client()

    with patch(GET_CLIENT, return_value=async_client):
        with pytest.raises(ValueError, match="sync client"):
            handler.cancel_batch(
                _is_async=False, cancel_batch_data=CANCEL_DATA, **AUTH_KW
            )

    async_client.batches.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_async_returns_coroutine_and_awaits(handler):
    client = _async_client()
    client.batches.cancel.return_value = _sdk_response(_batch_dict(status="cancelled"))

    with patch(GET_CLIENT, return_value=client) as get_client:
        coro = handler.cancel_batch(
            _is_async=True, cancel_batch_data=CANCEL_DATA, **AUTH_KW
        )
        assert asyncio.iscoroutine(coro)
        result = await coro

    assert get_client.call_args.kwargs["_is_async"] is True
    client.batches.cancel.assert_awaited_once_with(**CANCEL_DATA)
    assert isinstance(result, LiteLLMBatch)
    assert result.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_async_rejects_sync_client(handler):
    sync_client = _sync_client()
    with patch(GET_CLIENT, return_value=sync_client):
        with pytest.raises(ValueError, match="async client"):
            handler.cancel_batch(
                _is_async=True, cancel_batch_data=CANCEL_DATA, **AUTH_KW
            )
    sync_client.batches.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_acancel_batch_parses_response(handler):
    client = _async_client()
    client.batches.cancel.return_value = _sdk_response(_batch_dict(status="cancelled"))

    result = await handler.acancel_batch(cancel_batch_data=CANCEL_DATA, client=client)

    client.batches.cancel.assert_awaited_once_with(**CANCEL_DATA)
    assert isinstance(result, LiteLLMBatch)
    assert result.status == "cancelled"


# =========================================================================== #
# list_batches  (returns the raw SDK response, NOT a LiteLLMBatch)
# =========================================================================== #


def test_list_sync_forwards_after_limit_and_returns_raw_response(handler):
    client = _sync_client()
    raw = MagicMock(name="raw_list_response")
    client.batches.list.return_value = raw

    with patch(GET_CLIENT, return_value=client) as get_client:
        result = handler.list_batches(
            _is_async=False, after="cur-1", limit=20, **AUTH_KW
        )

    assert get_client.call_args.kwargs["_is_async"] is False
    client.batches.list.assert_called_once_with(after="cur-1", limit=20)
    # list returns the SDK response untouched (no LiteLLMBatch parsing).
    assert result is raw


def test_list_sync_defaults_after_and_limit_to_none(handler):
    client = _sync_client()
    client.batches.list.return_value = MagicMock()

    with patch(GET_CLIENT, return_value=client):
        handler.list_batches(_is_async=False, **AUTH_KW)

    client.batches.list.assert_called_once_with(after=None, limit=None)


def test_list_raises_when_client_is_none(handler):
    with patch(GET_CLIENT, return_value=None):
        with pytest.raises(ValueError, match="client is not initialized"):
            handler.list_batches(_is_async=False, **AUTH_KW)


@pytest.mark.asyncio
async def test_list_async_returns_coroutine_and_awaits(handler):
    client = _async_client()
    raw = MagicMock(name="raw_async_list_response")
    client.batches.list.return_value = raw

    with patch(GET_CLIENT, return_value=client) as get_client:
        coro = handler.list_batches(
            _is_async=True, after="cur-2", limit=7, **AUTH_KW
        )
        assert asyncio.iscoroutine(coro)
        result = await coro

    assert get_client.call_args.kwargs["_is_async"] is True
    client.batches.list.assert_awaited_once_with(after="cur-2", limit=7)
    assert result is raw


@pytest.mark.asyncio
async def test_list_async_rejects_sync_client(handler):
    sync_client = _sync_client()
    with patch(GET_CLIENT, return_value=sync_client):
        with pytest.raises(ValueError, match="not an instance of AsyncOpenAI"):
            handler.list_batches(_is_async=True, **AUTH_KW)
    sync_client.batches.list.assert_not_called()


@pytest.mark.asyncio
async def test_alist_batches_returns_raw_response(handler):
    client = _async_client()
    raw = MagicMock(name="raw")
    client.batches.list.return_value = raw

    result = await handler.alist_batches(client=client, after="a", limit=2)

    client.batches.list.assert_awaited_once_with(after="a", limit=2)
    assert result is raw


# =========================================================================== #
# Cross-cutting: an OpenAI (non-Azure) client also satisfies the type guards,
# since the Union allows OpenAI / AsyncOpenAI (Azure-v1 path returns these).
# =========================================================================== #


def test_create_sync_accepts_plain_openai_client(handler):
    client = MagicMock(spec=OpenAI)
    client.batches.create.return_value = _sdk_response(_batch_dict())

    with patch(GET_CLIENT, return_value=client):
        result = handler.create_batch(
            _is_async=False, create_batch_data=CREATE_DATA, **AUTH_KW
        )

    assert isinstance(result, LiteLLMBatch)


@pytest.mark.asyncio
async def test_create_async_accepts_plain_async_openai_client(handler):
    client = MagicMock(spec=AsyncOpenAI)
    client.batches.create = AsyncMock(return_value=_sdk_response(_batch_dict()))

    with patch(GET_CLIENT, return_value=client):
        result = await handler.create_batch(
            _is_async=True, create_batch_data=CREATE_DATA, **AUTH_KW
        )

    assert isinstance(result, LiteLLMBatch)
