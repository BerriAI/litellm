import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union
import json
from litellm._uuid import uuid


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
        response_headers = dict(response.headers)
        print("response headers=", response_headers)

        return await response.json(), response_headers


async def generate_key(
    session, guardrails: Optional[List] = None, team_id: Optional[str] = None
):
    url = "http://0.0.0.0:4000/key/generate"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {}
    if guardrails:
        data["guardrails"] = guardrails
    if team_id:
        data["team_id"] = team_id

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
                "bedrock-pre-guard",
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
        response, headers = await chat_completion(
            session,
            key_with_guardrails,
            model="fake-openai-endpoint",
            messages=[{"role": "user", "content": f"Hello my name is ishaan@berri.ai"}],
        )

        assert "x-litellm-applied-guardrails" in headers
        assert headers["x-litellm-applied-guardrails"] == "bedrock-pre-guard"


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
                messages=[{"role": "user", "content": "Hello do you like coffee?"}],
                guardrails=["bedrock-pre-guard"],
            )
            pytest.fail("Should have thrown an exception")
        except Exception as e:
            print(e)
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


async def create_team(session, guardrails: Optional[List] = None):
    url = "http://0.0.0.0:4000/team/new"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    data = {"guardrails": guardrails}

    print("request data=", data)

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        return await response.json()


@pytest.mark.asyncio
async def test_guardrails_with_team_controls():
    """
    - Create a team with guardrails
    - Make two API Keys
        - Key 1 not associated with team
        - Key 2 associated with team (inherits team guardrails)
    - Request with Key 1 -> should be success with no guardrails
    - Request with Key 2 -> should error since team guardrails are triggered
    """
    async with aiohttp.ClientSession() as session:

        # Create team with guardrails
        team = await create_team(
            session=session,
            guardrails=[
                "bedrock-pre-guard",
            ],
        )

        print("team=", team)

        team_id = team["team_id"]

        # Create key with team association
        key_with_team = await generate_key(session=session, team_id=team_id)
        key_with_team = key_with_team["key"]

        # Create key without team
        key_without_team = await generate_key(
            session=session,
        )
        key_without_team = key_without_team["key"]

        # Test no guardrails triggered for key without a team
        response, headers = await chat_completion(
            session,
            key_without_team,
            model="fake-openai-endpoint",
            messages=[{"role": "user", "content": "Hello my name is ishaan@berri.ai"}],
        )
        await asyncio.sleep(3)

        print("response=", response, "response headers", headers)
        assert "x-litellm-applied-guardrails" not in headers

        response, headers = await chat_completion(
            session,
            key_with_team,
            model="fake-openai-endpoint",
            messages=[{"role": "user", "content": "Hello my name is ishaan@berri.ai"}],
        )

        print("response headers=", json.dumps(headers, indent=4))

        assert "x-litellm-applied-guardrails" in headers
        assert headers["x-litellm-applied-guardrails"] == "bedrock-pre-guard"
