import asyncio
import os
import sys

import pytest
from openai import AsyncAzureOpenAI

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm


@pytest.mark.asyncio
async def test_acompletion_propagates_cancelled_error():
    client = AsyncAzureOpenAI(
        api_key="fake-key",
        api_version="2024-02-01",
        azure_endpoint="https://fake-resource.openai.azure.com",
    )

    async def cancelled_create(**kwargs):
        raise asyncio.CancelledError()

    client.chat.completions.with_raw_response.create = cancelled_create

    with pytest.raises(asyncio.CancelledError):
        await litellm.acompletion(
            model="azure/fake-deployment",
            messages=[{"role": "user", "content": "hi"}],
            client=client,
        )
