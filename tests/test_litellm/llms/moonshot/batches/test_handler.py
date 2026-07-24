"""Unit tests for MoonshotBatchesAPI (litellm/llms/moonshot/batches/handler.py).

The handler is a thin credential-resolution + OpenAI-SDK-delegation layer.
We mock the two seams:
  * ``MoonshotBatchesAPI._get_client`` – credential / client-construction.
  * the returned client's ``batches.*`` methods – the network call.

Pure logic (the _is_async branch, isinstance guards, model_dump parse) runs for
real.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from openai import AsyncOpenAI, OpenAI  # noqa: E402

from litellm.llms.moonshot.batches.handler import MoonshotBatchesAPI  # noqa: E402
from litellm.types.utils import LiteLLMBatch  # noqa: E402

GET_CLIENT = "litellm.llms.moonshot.batches.handler.MoonshotBatchesAPI._get_client"

AUTH_KW = dict(
    api_key="sk-moonshot-test",
    api_base="https://api.moonshot.ai/v1",
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
    return {
        "id": batch_id,
        "completion_window": "24h",
        "created_at": 1700000000,
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-abc",
        "object": "batch",
        "status": status,
    }


def _sdk_response(batch_dict: dict) -> MagicMock:
    resp = MagicMock()
    resp.model_dump.return_value = batch_dict
    return resp


def _sync_client() -> MagicMock:
    return MagicMock(spec=OpenAI)


def _async_client() -> MagicMock:
    client = MagicMock(spec=AsyncOpenAI)
    client.batches.create = AsyncMock()
    client.batches.retrieve = AsyncMock()
    client.batches.cancel = AsyncMock()
    client.batches.list = AsyncMock()
    return client


# ====================================================================== create


class TestCreateBatch:
    def test_sync_create_batch(self):
        handler = MoonshotBatchesAPI()
        sdk = _sync_client()
        sdk.batches.create.return_value = _sdk_response(_batch_dict())

        with patch(GET_CLIENT, return_value=sdk) as mock_get:
            result = handler.create_batch(
                _is_async=False,
                create_batch_data=CREATE_DATA,
                **AUTH_KW,
            )

        mock_get.assert_called_once_with(
            api_key=AUTH_KW["api_key"],
            api_base=AUTH_KW["api_base"],
            timeout=AUTH_KW["timeout"],
            max_retries=AUTH_KW["max_retries"],
            _is_async=False,
            client=None,
        )
        sdk.batches.create.assert_called_once_with(**CREATE_DATA)
        assert isinstance(result, LiteLLMBatch)
        assert result.id == "batch-123"

    @pytest.mark.asyncio
    async def test_async_create_batch(self):
        handler = MoonshotBatchesAPI()
        sdk = _async_client()
        sdk.batches.create.return_value = _sdk_response(_batch_dict())

        with patch(GET_CLIENT, return_value=sdk):
            coro = handler.create_batch(
                _is_async=True,
                create_batch_data=CREATE_DATA,
                **AUTH_KW,
            )
        result = await coro

        sdk.batches.create.assert_called_once_with(**CREATE_DATA)
        assert isinstance(result, LiteLLMBatch)


# ==================================================================== retrieve


class TestRetrieveBatch:
    def test_sync_retrieve_batch(self):
        handler = MoonshotBatchesAPI()
        sdk = _sync_client()
        sdk.batches.retrieve.return_value = _sdk_response(_batch_dict())

        with patch(GET_CLIENT, return_value=sdk):
            result = handler.retrieve_batch(
                _is_async=False,
                retrieve_batch_data=RETRIEVE_DATA,
                **AUTH_KW,
            )

        sdk.batches.retrieve.assert_called_once_with(**RETRIEVE_DATA)
        assert isinstance(result, LiteLLMBatch)

    @pytest.mark.asyncio
    async def test_async_retrieve_batch(self):
        handler = MoonshotBatchesAPI()
        sdk = _async_client()
        sdk.batches.retrieve.return_value = _sdk_response(_batch_dict())

        with patch(GET_CLIENT, return_value=sdk):
            coro = handler.retrieve_batch(
                _is_async=True,
                retrieve_batch_data=RETRIEVE_DATA,
                **AUTH_KW,
            )
        result = await coro

        sdk.batches.retrieve.assert_called_once_with(**RETRIEVE_DATA)
        assert isinstance(result, LiteLLMBatch)


# ====================================================================== cancel


class TestCancelBatch:
    def test_sync_cancel_batch(self):
        handler = MoonshotBatchesAPI()
        sdk = _sync_client()
        sdk.batches.cancel.return_value = _sdk_response(_batch_dict(status="cancelling"))

        with patch(GET_CLIENT, return_value=sdk):
            result = handler.cancel_batch(
                _is_async=False,
                cancel_batch_data=CANCEL_DATA,
                **AUTH_KW,
            )

        sdk.batches.cancel.assert_called_once_with(**CANCEL_DATA)
        assert isinstance(result, LiteLLMBatch)

    @pytest.mark.asyncio
    async def test_async_cancel_batch(self):
        handler = MoonshotBatchesAPI()
        sdk = _async_client()
        sdk.batches.cancel.return_value = _sdk_response(_batch_dict(status="cancelling"))

        with patch(GET_CLIENT, return_value=sdk):
            coro = handler.cancel_batch(
                _is_async=True,
                cancel_batch_data=CANCEL_DATA,
                **AUTH_KW,
            )
        result = await coro

        sdk.batches.cancel.assert_called_once_with(**CANCEL_DATA)
        assert isinstance(result, LiteLLMBatch)


# ======================================================================== list


class TestListBatches:
    def test_sync_list_batches(self):
        handler = MoonshotBatchesAPI()
        sdk = _sync_client()
        list_resp = MagicMock()
        sdk.batches.list.return_value = list_resp

        with patch(GET_CLIENT, return_value=sdk):
            result = handler.list_batches(
                _is_async=False,
                after=None,
                limit=10,
                **AUTH_KW,
            )

        sdk.batches.list.assert_called_once_with(after=None, limit=10)
        assert result is list_resp

    @pytest.mark.asyncio
    async def test_async_list_batches(self):
        handler = MoonshotBatchesAPI()
        sdk = _async_client()
        list_resp = MagicMock()
        sdk.batches.list.return_value = list_resp

        with patch(GET_CLIENT, return_value=sdk):
            coro = handler.list_batches(
                _is_async=True,
                after="cursor-abc",
                limit=5,
                **AUTH_KW,
            )
        result = await coro

        sdk.batches.list.assert_called_once_with(after="cursor-abc", limit=5)
        assert result is list_resp


# ============================================================= _get_client unit


class TestGetClient:
    def test_uses_env_key_when_no_api_key(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_API_KEY", "env-key")
        handler = MoonshotBatchesAPI()
        with patch("litellm.llms.moonshot.batches.handler.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock(spec=OpenAI)
            handler._get_client(
                api_key=None,
                api_base="https://api.moonshot.ai/v1",
                timeout=30.0,
                max_retries=2,
                _is_async=False,
            )
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["api_key"] == "env-key"

    def test_defaults_to_moonshot_base_url(self):
        handler = MoonshotBatchesAPI()
        with patch("litellm.llms.moonshot.batches.handler.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock(spec=OpenAI)
            handler._get_client(
                api_key="sk-test",
                api_base=None,
                timeout=30.0,
                max_retries=None,
                _is_async=False,
            )
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["base_url"] == "https://api.moonshot.ai/v1"

    def test_returns_async_client_when_is_async(self):
        handler = MoonshotBatchesAPI()
        with patch("litellm.llms.moonshot.batches.handler.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock(spec=AsyncOpenAI)
            handler._get_client(
                api_key="sk-test",
                api_base="https://api.moonshot.ai/v1",
                timeout=30.0,
                max_retries=None,
                _is_async=True,
            )
        mock_cls.assert_called_once()

    def test_returns_provided_client_unchanged(self):
        handler = MoonshotBatchesAPI()
        existing = MagicMock(spec=OpenAI)
        result = handler._get_client(
            api_key="sk-test",
            api_base="https://api.moonshot.ai/v1",
            timeout=30.0,
            max_retries=None,
            _is_async=False,
            client=existing,
        )
        assert result is existing

    def test_raises_when_no_key_available(self, monkeypatch):
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        handler = MoonshotBatchesAPI()
        with pytest.raises(ValueError, match="No Moonshot API key found"):
            handler._get_client(
                api_key=None,
                api_base="https://api.moonshot.ai/v1",
                timeout=30.0,
                max_retries=None,
                _is_async=False,
            )

    def test_explicit_key_passed_to_openai_client(self):
        handler = MoonshotBatchesAPI()
        with patch("litellm.llms.moonshot.batches.handler.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock(spec=OpenAI)
            handler._get_client(
                api_key="sk-explicit",
                api_base="https://api.moonshot.ai/v1",
                timeout=30.0,
                max_retries=None,
                _is_async=False,
            )
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-explicit"
