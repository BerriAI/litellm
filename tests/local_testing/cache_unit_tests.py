from abc import ABC, abstractmethod
from litellm.caching import LiteLLMCacheType
import os
import sys
import time
import traceback
from litellm._uuid import uuid

from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import hashlib
import random

import pytest

import litellm
from litellm.caching import Cache
from litellm import completion, embedding


class LLMCachingUnitTests(ABC):

    @abstractmethod
    def get_cache_type(self) -> LiteLLMCacheType:
        pass

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_cache_completion(self, sync_mode):
        litellm._turn_on_debug()

        random_number = random.randint(
            1, 100000
        )  # add a random number to ensure it's always adding / reading from cache
        messages = [
            {
                "role": "user",
                "content": f"write a one sentence poem about: {random_number}",
            }
        ]

        cache_type = self.get_cache_type()
        litellm.cache = Cache(
            type=cache_type,
        )

        if sync_mode:
            response1 = completion(
                "gpt-3.5-turbo",
                messages=messages,
                caching=True,
                max_tokens=20,
                mock_response="This number is so great!",
            )
        else:
            response1 = await litellm.acompletion(
                "gpt-3.5-turbo",
                messages=messages,
                caching=True,
                max_tokens=20,
                mock_response="This number is so great!",
            )
        # response2 is mocked to a different response from response1,
        # but the completion from the cache should be used instead of the mock
        # response since the input is the same as response1
        await asyncio.sleep(0.5)
        if sync_mode:
            response2 = completion(
                "gpt-3.5-turbo",
                messages=messages,
                caching=True,
                max_tokens=20,
                mock_response="This number is great!",
            )
        else:
            response2 = await litellm.acompletion(
                "gpt-3.5-turbo",
                messages=messages,
                caching=True,
                max_tokens=20,
                mock_response="This number is great!",
            )
        if (
            response1["choices"][0]["message"]["content"]
            != response2["choices"][0]["message"]["content"]
        ):  # 1 and 2 should be the same
            # 1&2 have the exact same input params. This MUST Be a CACHE HIT
            print(f"response1: {response1}")
            print(f"response2: {response2}")
            pytest.fail(
                f"Error occurred: response1 - {response1['choices'][0]['message']['content']} != response2 - {response2['choices'][0]['message']['content']}"
            )
        # Since the parameters are not the same as response1, response3 should actually
        # be the mock response
        if sync_mode:
            response3 = completion(
                "gpt-3.5-turbo",
                messages=messages,
                caching=True,
                temperature=0.5,
                mock_response="This number is awful!",
            )
        else:
            response3 = await litellm.acompletion(
                "gpt-3.5-turbo",
                messages=messages,
                caching=True,
                temperature=0.5,
                mock_response="This number is awful!",
            )

        print("\nresponse 1", response1)
        print("\nresponse 2", response2)
        print("\nresponse 3", response3)
        # print("\nresponse 4", response4)
        litellm.cache = None
        litellm.success_callback = []
        litellm._async_success_callback = []

        # 1 & 2 should be exactly the same
        # 1 & 3 should be different, since input params are diff

        if (
            response1["choices"][0]["message"]["content"]
            == response3["choices"][0]["message"]["content"]
        ):
            # if input params like max_tokens, temperature are diff it should NOT be a cache hit
            print(f"response1: {response1}")
            print(f"response3: {response3}")
            pytest.fail(
                f"Response 1 == response 3. Same model, diff params shoudl not cache Error"
                f" occurred:"
            )

        assert response1.id == response2.id
        assert response1.created == response2.created
        assert (
            response1.choices[0].message.content == response2.choices[0].message.content
        )

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_disk_cache_embedding(self, sync_mode):
        litellm._turn_on_debug()

        random_number = random.randint(
            1, 100000
        )  # add a random number to ensure it's always adding / reading from cache
        input = [f"hello {random_number}"]
        litellm.cache = Cache(
            type="disk",
        )

        if sync_mode:
            response1 = embedding(
                "openai/text-embedding-ada-002",
                input=input,
                caching=True,
            )
        else:
            response1 = await litellm.aembedding(
                "openai/text-embedding-ada-002",
                input=input,
                caching=True,
            )
        # response2 is mocked to a different response from response1,
        # but the completion from the cache should be used instead of the mock
        # response since the input is the same as response1
        await asyncio.sleep(0.5)
        if sync_mode:
            response2 = embedding(
                "openai/text-embedding-ada-002",
                input=input,
                caching=True,
            )
        else:
            response2 = await litellm.aembedding(
                "openai/text-embedding-ada-002",
                input=input,
                caching=True,
            )

        if response2._hidden_params["cache_hit"] is not True:
            pytest.fail("Cache hit should be True")

        # Since the parameters are not the same as response1, response3 should actually
        # be the mock response
        if sync_mode:
            response3 = embedding(
                "openai/text-embedding-ada-002",
                input=input,
                user="charlie",
                caching=True,
            )
        else:
            response3 = await litellm.aembedding(
                "openai/text-embedding-ada-002",
                input=input,
                caching=True,
                user="charlie",
            )

        print("\nresponse 1", response1)
        print("\nresponse 2", response2)
        print("\nresponse 3", response3)
        # print("\nresponse 4", response4)
        litellm.cache = None
        litellm.success_callback = []
        litellm._async_success_callback = []

        # 1 & 2 should be exactly the same
        # 1 & 3 should be different, since input params are diff

        if response3._hidden_params.get("cache_hit") is True:
            pytest.fail("Cache hit should not be True")
