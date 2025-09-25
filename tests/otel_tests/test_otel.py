# What this tests ?
## Tests /chat/completions by generating a key and then making a chat completions request
import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union
from litellm._uuid import uuid


async def generate_key(
    session,
    models=[
        "gpt-4",
        "text-embedding-ada-002",
        "dall-e-2",
        "fake-openai-endpoint",
        "mistral-embed",
    ],
):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {
        "models": models,
        "duration": None,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        return await response.json()


async def chat_completion(session, key, model: Union[str, List] = "gpt-4"):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"Hello! {str(uuid.uuid4())}"},
        ],
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        return await response.json()


async def get_otel_spans(session, key):
    url = "http://0.0.0.0:4000/otel-spans"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        return await response.json()


@pytest.mark.asyncio
async def test_chat_completion_check_otel_spans():
    """
    - Create key
    Make chat completion call
    - Create user
    make chat completion call
    """
    async with aiohttp.ClientSession() as session:
        key_gen = await generate_key(session=session)
        key = key_gen["key"]
        await chat_completion(session=session, key=key, model="fake-openai-endpoint")

        await asyncio.sleep(3)

        otel_spans = await get_otel_spans(session=session, key=key)
        print("otel_spans: ", otel_spans)

        all_otel_spans = otel_spans["otel_spans"]
        most_recent_parent = str(otel_spans["most_recent_parent"])
        print("Most recent OTEL parent: ", most_recent_parent)
        print("\n spans grouped by parent: ", otel_spans["spans_grouped_by_parent"])
        parent_trace_spans = otel_spans["spans_grouped_by_parent"][most_recent_parent]

        print("Parent trace spans: ", parent_trace_spans)

        # either 5 or 6 traces depending on how many redis calls were made
        assert len(parent_trace_spans) >= 5

        # 'postgres', 'redis', 'raw_gen_ai_request', 'litellm_request', 'Received Proxy Server Request' in the span
        assert "postgres" in parent_trace_spans
        assert "redis" in parent_trace_spans
        assert "raw_gen_ai_request" in parent_trace_spans
        assert "litellm_request" in parent_trace_spans
        assert "batch_write_to_db" in parent_trace_spans
