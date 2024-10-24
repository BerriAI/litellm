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
            raise Exception(response_text)

        # response headers
        response_headers = response.headers
        print("response headers=", response_headers)

        return await response.json(), response_headers


async def generate_key(session, guardrails):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    if guardrails:
        data = {
            "guardrails": guardrails,
        }
    else:
        data = {}

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        return await response.json()


@pytest.mark.asyncio
@pytest.mark.skip(reason="Aporia account disabled")
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
            guardrails=[
                "aporia-post-guard",
                "aporia-pre-guard",
            ],
        )
        await asyncio.sleep(3)

        print("response=", response, "response headers", headers)

        assert "x-litellm-applied-guardrails" in headers

        assert (
            headers["x-litellm-applied-guardrails"]
            == "aporia-pre-guard,aporia-post-guard"
        )


@pytest.mark.asyncio
@pytest.mark.skip(reason="Aporia account disabled")
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
                guardrails=[
                    "aporia-post-guard",
                    "aporia-pre-guard",
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


@pytest.mark.asyncio
@pytest.mark.skip(reason="Aporia account disabled")
async def test_guardrails_with_api_key_controls():
    """
    - Make two API Keys
        - Key 1 with no guardrails
        - Key 2 with guardrails
    - Request to Key 1 -> should be success with no guardrails
    - Request to Key 2 -> should be error since guardrails are triggered
    """
    async with aiohttp.ClientSession() as session:
        key_with_guardrails = await generate_key(
            session=session,
            guardrails=[
                "aporia-post-guard",
                "aporia-pre-guard",
            ],
        )

        key_with_guardrails = key_with_guardrails["key"]

        key_without_guardrails = await generate_key(session=session, guardrails=None)

        key_without_guardrails = key_without_guardrails["key"]

        # test no guardrails triggered for key without guardrails
        response, headers = await chat_completion(
            session,
            key_without_guardrails,
            model="fake-openai-endpoint",
            messages=[{"role": "user", "content": f"Hello what's the weather"}],
        )
        await asyncio.sleep(3)

        print("response=", response, "response headers", headers)
        assert "x-litellm-applied-guardrails" not in headers

        # test guardrails triggered for key with guardrails
        try:
            response, headers = await chat_completion(
                session,
                key_with_guardrails,
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
async def test_bedrock_guardrail_triggered():
    """
    - Tests a request where our bedrock guardrail should be triggered
    - Assert that the guardrails applied are returned in the response headers
    """
    async with aiohttp.ClientSession() as session:
        try:
            response, headers = await chat_completion(
                session,
                "sk-1234",
                model="fake-openai-endpoint",
                messages=[{"role": "user", "content": f"Hello do you like coffee?"}],
                guardrails=["bedrock-pre-guard"],
            )
            pytest.fail("Should have thrown an exception")
        except Exception as e:
            print(e)
            assert "GUARDRAIL_INTERVENED" in str(e)
            assert "Violated guardrail policy" in str(e)


@pytest.mark.asyncio
async def test_custom_guardrail_during_call_triggered():
    """
    - Tests a request where our bedrock guardrail should be triggered
    - Assert that the guardrails applied are returned in the response headers
    """
    async with aiohttp.ClientSession() as session:
        try:
            response, headers = await chat_completion(
                session,
                "sk-1234",
                model="fake-openai-endpoint",
                messages=[{"role": "user", "content": f"Hello do you like litellm?"}],
                guardrails=["custom-during-guard"],
            )
            pytest.fail("Should have thrown an exception")
        except Exception as e:
            print(e)
            assert "Guardrail failed words - `litellm` detected" in str(e)
