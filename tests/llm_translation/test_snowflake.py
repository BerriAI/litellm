import os
import sys
import traceback
from dotenv import load_dotenv

load_dotenv()
import pytest

from litellm import completion, acompletion
from litellm.exceptions import APIConnectionError

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
                api_base = "https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions"
            )
            print(response)
            assert response is not None
        else:
            response = await acompletion(
                model="snowflake/mistral-7b",
                messages=messages,
                api_base = "https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions"
            )
            print(response)
            assert response is not None
    except APIConnectionError as e:
        # Skip test if Snowflake API is unavailable (502 error)
        if "Application failed to respond" in str(e) or "502" in str(e):
            pytest.skip(f"Snowflake API unavailable: {e}")
        else:
            raise  # Re-raise if it's a different APIConnectionError
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
                api_base = "https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions"
            )
            
            async for chunk in response:
                print(chunk)
        else:
            response = completion(
                model="snowflake/mistral-7b",
                messages=messages,
                max_tokens=100,
                stream=True,
                api_base = "https://exampleopenaiendpoint-production.up.railway.app/v1/chat/completions"
            )

            for chunk in response:
                print(chunk)
    except APIConnectionError as e:
        # Skip test if Snowflake API is unavailable (502 error)
        if "Application failed to respond" in str(e) or "502" in str(e):
            pytest.skip(f"Snowflake API unavailable: {e}")
        else:
            raise  # Re-raise if it's a different APIConnectionError
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
