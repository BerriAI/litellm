import os
import sys
import traceback
from dotenv import load_dotenv

load_dotenv()
import pytest

from litellm import completion, acompletion, set_verbose

@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_chat_completion_snowflake(sync_mode):
    try:
        messages = [
            {
                "role": "user",
                "content": "Write me a poem about the blue sky",
            },
        ]

        if sync_mode:
            response = completion(
                model="snowflake/mistral-7b",
                messages=messages,
            )
            print(response)
            assert response is not None
        else:
            response = await acompletion(
                model="snowflake/mistral-7b",
                messages=messages,
            )
            print(response)
            assert response is not None
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_chat_completion_snowflake_stream(sync_mode):
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
                model="snowflake/mistral-7b",
                messages=messages,
                max_tokens=100,
                stream=True,
            )
            
            chunk_count = 0
            async for chunk in response:
                print(chunk)
                chunk_count += 1
            assert chunk_count > 0
        else:
            response = completion(
                model="snowflake/mistral-7b",
                messages=messages,
                max_tokens=100,
                stream=True,
            )
                
            chunk_count = 0
            for chunk in response:
                print(chunk)
                chunk_count += 1
            assert chunk_count > 0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
