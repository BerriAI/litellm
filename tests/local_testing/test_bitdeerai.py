import os
import sys
import traceback
from dotenv import load_dotenv

load_dotenv()
import pytest

from litellm import completion, acompletion, embedding, aembedding, EmbeddingResponse

@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_chat_completion_bitdeerai(sync_mode):
    try:
        set_verbose = True
        messages = [
            {
                "role": "user",
                "content": "Write me a poem about the blue sky",
            },
        ]

        if sync_mode:
            response = completion(
                model="bitdeerai/deepseek-ai/DeepSeek-V3",
                messages=messages,
            )
            print(response)
            assert response is not None
        else:
            response = await acompletion(
                model="bitdeerai/deepseek-ai/DeepSeek-V3",
                messages=messages,
            )
            print(response)
            assert response is not None
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_chat_completion_bitdeerai_stream(sync_mode):
    try:
        set_verbose = True
        messages = [
            {
                "role": "user",
                "content": "Write me a poem about the blue sky",
            },
        ]

        if sync_mode is False:
            response = await acompletion(
                model="bitdeerai/deepseek-ai/DeepSeek-V3",
                messages=messages,
                max_tokens=100,
                stream=True,
            )

            async for chunk in response:
                print(chunk)
        else:
            response = completion(
                model="bitdeerai/deepseek-ai/DeepSeek-V3",
                messages=messages,
                max_tokens=100,
                stream=True,
            )
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [False, True])
@pytest.mark.asyncio
async def test_embedding_bitdeerai(sync_mode):
    set_verbose = True
    model_name = "bitdeerai/BAAI/bge-m3"
    try:
        user_message = "The cat danced gracefully under the moonlight, its shadow twirling like a silent partner."
        if sync_mode:
            response = embedding(
                model=model_name,
                input=[user_message],

            )
        else:
            response = await aembedding(
                model=model_name,
                input=[user_message],
            )
        assert isinstance(response, EmbeddingResponse)
        assert len(response.data[0]['embedding']) == 1024
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")