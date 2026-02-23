import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union
from litellm._uuid import uuid


async def make_rerank_curl_request(
    session,
    key,
    query,
    documents,
    model="rerank-english-v3.0",
    top_n=3,
):
    url = "http://0.0.0.0:4000/rerank"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        if status != 200:
            raise Exception(response_text)

        return await response.json()


@pytest.mark.asyncio
async def test_basic_rerank_on_proxy():
    """
    Test litellm.rerank() on proxy

    This SHOULD NOT call the pass through endpoints :)
    """
    async with aiohttp.ClientSession() as session:
        docs = [
            "Carson City is the capital city of the American state of Nevada.",
            "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
            "Washington, D.C. is the capital of the United States.",
            "Capital punishment has existed in the United States since before it was a country.",
        ]

        try:
            response = await make_rerank_curl_request(
                session,
                "sk-1234",
                query="What is the capital of the United States?",
                documents=docs,
            )
            print("response=", response)
        except Exception as e:
            print(e)
            pytest.fail("Rerank request failed")
