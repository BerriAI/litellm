import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union
import uuid


async def chat_completion(
    session,
    key,
    messages,
    model: Union[str, List] = "gpt-4",
    guardrails: Optional[List] = None,
):
    url = "http://0.0.0.0:4000/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": messages,
        "guardrails": [
            "aporia-post-guard",
            "aporia-pre-guard",
        ],  # default guardrails for all tests
    }

    if guardrails is not None:
        data["guardrails"] = guardrails

    print("data=", data)

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            return response_text

        # response headers
        response_headers = response.headers
        print("response headers=", response_headers)

        return await response.json(), response_headers


@pytest.mark.asyncio
async def test_llm_guard_triggered_safe_request():
    """
    - Tests a request where no content mod is triggered
    - Assert that the guardrails applied are returned in the response headers
    """
    async with aiohttp.ClientSession() as session:
        response, headers = await chat_completion(
            session,
            "sk-1234",
            model="fake-openai-endpoint",
            messages=[{"role": "user", "content": f"Hello what's the weather"}],
        )
        await asyncio.sleep(3)

        print("response=", response, "response headers", headers)

        assert "x-litellm-applied-guardrails" in headers

        assert (
            headers["x-litellm-applied-guardrails"]
            == "aporia-pre-guard,aporia-post-guard"
        )


@pytest.mark.asyncio
async def test_llm_guard_triggered():
    """
    - Tests a request where no content mod is triggered
    - Assert that the guardrails applied are returned in the response headers
    """
    async with aiohttp.ClientSession() as session:
        try:
            response, headers = await chat_completion(
                session,
                "sk-1234",
                model="fake-openai-endpoint",
                messages=[
                    {"role": "user", "content": f"Hello my name is ishaan@berri.ai"}
                ],
            )
            pytest.fail("Should have thrown an exception")
        except Exception as e:
            print(e)
            assert "Aporia detected and blocked PII" in str(e)


@pytest.mark.asyncio
async def test_no_llm_guard_triggered():
    """
    - Tests a request where no content mod is triggered
    - Assert that the guardrails applied are returned in the response headers
    """
    async with aiohttp.ClientSession() as session:
        response, headers = await chat_completion(
            session,
            "sk-1234",
            model="fake-openai-endpoint",
            messages=[{"role": "user", "content": f"Hello what's the weather"}],
            guardrails=[],
        )
        await asyncio.sleep(3)

        print("response=", response, "response headers", headers)

        assert "x-litellm-applied-guardrails" not in headers
