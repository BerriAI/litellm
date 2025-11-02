import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union
from litellm._uuid import uuid


async def make_moderations_curl_request(
    session,
    key,
    request_data: dict,
):
    url = "http://0.0.0.0:4000/moderations"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    async with session.post(url, headers=headers, json=request_data) as response:
        status = response.status
        response_text = await response.text()

        if status != 200:
            raise Exception(response_text)

        return await response.json()


@pytest.mark.asyncio
async def test_basic_moderations_on_proxy_no_model():
    """
    Test moderations endpoint on proxy when no `model` is specified in the request
    """
    async with aiohttp.ClientSession() as session:
        test_text = "I want to harm someone"  # Test text that should trigger moderation
        request_data = {
            "input": test_text,
        }
        try:
            response = await make_moderations_curl_request(
                session,
                "sk-1234",
                request_data,
            )
            print("response=", response)
        except Exception as e:
            print(e)
            pytest.fail("Moderations request failed")


@pytest.mark.asyncio
async def test_basic_moderations_on_proxy_with_model():
    """
    Test moderations endpoint on proxy when `model` is specified in the request
    """
    async with aiohttp.ClientSession() as session:
        test_text = "I want to harm someone"  # Test text that should trigger moderation
        request_data = {
            "input": test_text,
            "model": "omni-moderation-latest",
        }
        try:
            response = await make_moderations_curl_request(
                session,
                "sk-1234",
                request_data,
            )
            print("response=", response)
        except Exception as e:
            pytest.fail("Moderations request failed")
