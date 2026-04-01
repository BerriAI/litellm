import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union

import aiohttp
import asyncio
import json
import os
import dotenv


dotenv.load_dotenv()


async def cohere_rerank(session):
    url = "http://localhost:4000/v1/rerank"
    headers = {
        "Authorization": f"Bearer {os.getenv('COHERE_API_KEY')}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data = {
        "model": "rerank-english-v3.0",
        "query": "What is the capital of the United States?",
        "top_n": 3,
        "documents": [
            "Carson City is the capital city of the American state of Nevada.",
            "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
            "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
            "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
            "Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
        ],
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()
        print(f"Status: {status}")
        print(f"Response:\n{response_text}")
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        return await response.json()


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="new test just added by @ishaan-jaff, still figuring out how to run this in ci/cd"
)
async def test_basic_passthrough():
    """
    - Make request to pass through endpoint

    - This SHOULD not go through LiteLLM user_api_key_auth
    - This should forward headers from request to pass through endpoint
    """
    async with aiohttp.ClientSession() as session:
        response = await cohere_rerank(session)
        print("response from cohere rerank", response)

        assert response["id"] is not None
        assert response["results"] is not None
