"""Tests for litellm/llms/azure/azure.py AzureChatCompletion handler behaviour."""

import asyncio
import time
from typing import Final

import pytest
from openai import AsyncAzureOpenAI, AzureOpenAI

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.azure.azure import AzureChatCompletion
from litellm.utils import ModelResponse


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


class _CancellingRawCompletions:
    """The upstream SDK call is cancelled mid-flight, as it is when the caller hangs up."""

    async def create(self, **kwargs):
        raise asyncio.CancelledError()


def _cancelling_async_client() -> AsyncAzureOpenAI:
    client: Final = AsyncAzureOpenAI(
        api_key="fake", api_version="2024-02-01", azure_endpoint="https://fake.openai.azure.com"
    )
    client.chat.completions.with_raw_response = _CancellingRawCompletions()
    return client


async def _acompletion_on_a_cancelled_call() -> None:
    await AzureChatCompletion().acompletion(
        api_key="fake",
        api_version="2024-02-01",
        model="gpt-4o",
        api_base="https://fake.openai.azure.com",
        data={"messages": [{"role": "user", "content": "Hi"}]},
        timeout=30.0,
        dynamic_params=False,
        model_response=ModelResponse(),
        logging_obj=LiteLLMLoggingObj(
            model="azure/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            call_type="acompletion",
            start_time=time.time(),
            litellm_call_id="cancel-1",
            function_id="f",
        ),
        max_retries=0,
        client=_cancelling_async_client(),
    )


@pytest.mark.asyncio
async def test_a_cancelled_azure_call_propagates_the_cancellation():
    """https://github.com/BerriAI/litellm/issues/42222

    The handler used to relabel a cancellation as AzureOpenAIError(status_code=500).
    The router cools down on 5xx, so every caller that hung up counted against a
    healthy deployment's health.
    """
    with pytest.raises(asyncio.CancelledError):
        await _acompletion_on_a_cancelled_call()


@pytest.mark.asyncio
async def test_a_cancelled_azure_call_is_not_caught_as_a_deployment_failure():
    """Deployment-failure accounting catches Exception. A cancellation must sail past it,
    which it only does while it stays a BaseException rather than a 5xx provider error."""
    caught_as_failure = None
    try:
        await _acompletion_on_a_cancelled_call()
    except asyncio.CancelledError:
        pass
    except Exception as e:  # noqa: BLE001  # stands in for the router's failure path
        caught_as_failure = e

    assert caught_as_failure is None, f"cancellation reached deployment-failure handling as {caught_as_failure!r}"
