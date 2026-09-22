"""Tests for litellm/llms/azure/azure.py AzureChatCompletion handler behaviour."""

import asyncio
import time
from typing import Final

import pytest
from openai import AsyncAzureOpenAI, AzureOpenAI

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.azure.azure import AzureChatCompletion


class _FakeRawResponse:
    headers: Final = {"x-ms-is-spilled-over": "true"}

    def parse(self):
        return iter(())


class _FakeRawCompletions:
    def create(self, **kwargs):
        return _FakeRawResponse()


def test_sync_streaming_stamps_response_headers_on_the_logging_obj() -> None:
    """Sync streaming must mirror async_streaming and record the provider response
    headers on model_call_details, or downstream consumers (spillover-aware cost
    calculation) cannot see them."""
    client = AzureOpenAI(api_key="fake", api_version="2024-02-01", azure_endpoint="https://fake.openai.azure.com")
    client.chat.completions.with_raw_response = _FakeRawCompletions()

    logging_obj = LiteLLMLoggingObj(
        model="azure/gpt-4o-spill-test",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="spill-sync-1",
        function_id="f",
    )

    AzureChatCompletion().streaming(
        logging_obj=logging_obj,
        api_base="https://fake.openai.azure.com",
        api_key="fake",
        api_version="2024-02-01",
        dynamic_params=False,
        data={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
        model="gpt-4o-spill-test",
        timeout=30.0,
        max_retries=0,
        client=client,
    )

    assert logging_obj.model_call_details["response_headers"] == {"x-ms-is-spilled-over": "true"}


class _CancelledRawCompletions:
    async def create(self, **kwargs):
        raise asyncio.CancelledError()


@pytest.mark.asyncio
async def test_acompletion_propagates_cancelled_error() -> None:
    client = AsyncAzureOpenAI(
        api_key="fake-key",
        api_version="2024-02-01",
        azure_endpoint="https://fake-resource.openai.azure.com",
    )
    client.chat.completions.with_raw_response = _CancelledRawCompletions()

    with pytest.raises(asyncio.CancelledError):
        await litellm.acompletion(
            model="azure/fake-deployment",
            messages=[{"role": "user", "content": "hi"}],
            client=client,
        )
