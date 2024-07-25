# What is this?
## Unit tests for the CustomLLM class


import asyncio
import os
import sys
import time
import traceback

import openai
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from dotenv import load_dotenv

import litellm
from litellm import CustomLLM, acompletion, completion, get_llm_provider


class MyCustomLLM(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore


class MyCustomAsyncLLM(CustomLLM):
    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore


def test_get_llm_provider():
    from litellm.utils import custom_llm_setup

    my_custom_llm = MyCustomLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_llm", "custom_handler": my_custom_llm}
    ]

    custom_llm_setup()

    model, provider, _, _ = get_llm_provider(model="custom_llm/my-fake-model")

    assert provider == "custom_llm"


def test_simple_completion():
    my_custom_llm = MyCustomLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_llm", "custom_handler": my_custom_llm}
    ]
    resp = completion(
        model="custom_llm/my-fake-model",
        messages=[{"role": "user", "content": "Hello world!"}],
    )

    assert resp.choices[0].message.content == "Hi!"


@pytest.mark.asyncio
async def test_simple_acompletion():
    my_custom_llm = MyCustomAsyncLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_llm", "custom_handler": my_custom_llm}
    ]
    resp = await acompletion(
        model="custom_llm/my-fake-model",
        messages=[{"role": "user", "content": "Hello world!"}],
    )

    assert resp.choices[0].message.content == "Hi!"
