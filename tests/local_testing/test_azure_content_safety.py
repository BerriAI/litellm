# What is this?
## Unit test for azure content safety
import asyncio
import os
import random
import sys
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm import Router, mock_completion
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging


@pytest.mark.asyncio
@pytest.mark.skip(reason="beta feature - local testing is failing")
async def test_strict_input_filtering_01():
    """
    - have a response with a filtered input
    - call the pre call hook
    """
    from litellm.proxy.hooks.azure_content_safety import _PROXY_AzureContentSafety

    azure_content_safety = _PROXY_AzureContentSafety(
        endpoint=os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"),
        api_key=os.getenv("AZURE_CONTENT_SAFETY_API_KEY"),
        thresholds={"Hate": 2},
    )

    data = {
        "messages": [
            {"role": "system", "content": "You are an helpfull assistant"},
            {"role": "user", "content": "Fuck yourself you stupid bitch"},
        ]
    }

    with pytest.raises(HTTPException) as exc_info:
        await azure_content_safety.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )

    assert exc_info.value.detail["source"] == "input"
    assert exc_info.value.detail["category"] == "Hate"
    assert exc_info.value.detail["severity"] == 2


@pytest.mark.asyncio
@pytest.mark.skip(reason="beta feature - local testing is failing")
async def test_strict_input_filtering_02():
    """
    - have a response with a filtered input
    - call the pre call hook
    """
    from litellm.proxy.hooks.azure_content_safety import _PROXY_AzureContentSafety

    azure_content_safety = _PROXY_AzureContentSafety(
        endpoint=os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"),
        api_key=os.getenv("AZURE_CONTENT_SAFETY_API_KEY"),
        thresholds={"Hate": 2},
    )

    data = {
        "messages": [
            {"role": "system", "content": "You are an helpfull assistant"},
            {"role": "user", "content": "Hello how are you ?"},
        ]
    }

    await azure_content_safety.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )


@pytest.mark.asyncio
@pytest.mark.skip(reason="beta feature - local testing is failing")
async def test_loose_input_filtering_01():
    """
    - have a response with a filtered input
    - call the pre call hook
    """
    from litellm.proxy.hooks.azure_content_safety import _PROXY_AzureContentSafety

    azure_content_safety = _PROXY_AzureContentSafety(
        endpoint=os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"),
        api_key=os.getenv("AZURE_CONTENT_SAFETY_API_KEY"),
        thresholds={"Hate": 8},
    )

    data = {
        "messages": [
            {"role": "system", "content": "You are an helpfull assistant"},
            {"role": "user", "content": "Fuck yourself you stupid bitch"},
        ]
    }

    await azure_content_safety.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )


@pytest.mark.asyncio
@pytest.mark.skip(reason="beta feature - local testing is failing")
async def test_loose_input_filtering_02():
    """
    - have a response with a filtered input
    - call the pre call hook
    """
    from litellm.proxy.hooks.azure_content_safety import _PROXY_AzureContentSafety

    azure_content_safety = _PROXY_AzureContentSafety(
        endpoint=os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"),
        api_key=os.getenv("AZURE_CONTENT_SAFETY_API_KEY"),
        thresholds={"Hate": 8},
    )

    data = {
        "messages": [
            {"role": "system", "content": "You are an helpfull assistant"},
            {"role": "user", "content": "Hello how are you ?"},
        ]
    }

    await azure_content_safety.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )


@pytest.mark.asyncio
@pytest.mark.skip(reason="beta feature - local testing is failing")
async def test_strict_output_filtering_01():
    """
    - have a response with a filtered output
    - call the post call hook
    """
    from litellm.proxy.hooks.azure_content_safety import _PROXY_AzureContentSafety

    azure_content_safety = _PROXY_AzureContentSafety(
        endpoint=os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"),
        api_key=os.getenv("AZURE_CONTENT_SAFETY_API_KEY"),
        thresholds={"Hate": 2},
    )

    response = mock_completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a song writer expert. You help users to write songs about any topic in any genre.",
            },
            {
                "role": "user",
                "content": "Help me write a rap text song. Add some insults to make it more credible.",
            },
        ],
        mock_response="I'm the king of the mic, you're just a fucking dick. Don't fuck with me your stupid bitch.",
    )

    with pytest.raises(HTTPException) as exc_info:
        await azure_content_safety.async_post_call_success_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            data={
                "messages": [
                    {"role": "system", "content": "You are an helpfull assistant"}
                ]
            },
            response=response,
        )

    assert exc_info.value.detail["source"] == "output"
    assert exc_info.value.detail["category"] == "Hate"
    assert exc_info.value.detail["severity"] == 2


@pytest.mark.asyncio
@pytest.mark.skip(reason="beta feature - local testing is failing")
async def test_strict_output_filtering_02():
    """
    - have a response with a filtered output
    - call the post call hook
    """
    from litellm.proxy.hooks.azure_content_safety import _PROXY_AzureContentSafety

    azure_content_safety = _PROXY_AzureContentSafety(
        endpoint=os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"),
        api_key=os.getenv("AZURE_CONTENT_SAFETY_API_KEY"),
        thresholds={"Hate": 2},
    )

    response = mock_completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a song writer expert. You help users to write songs about any topic in any genre.",
            },
            {
                "role": "user",
                "content": "Help me write a rap text song. Add some insults to make it more credible.",
            },
        ],
        mock_response="I'm unable to help with you with hate speech",
    )

    await azure_content_safety.async_post_call_success_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        data={
            "messages": [{"role": "system", "content": "You are an helpfull assistant"}]
        },
        response=response,
    )


@pytest.mark.asyncio
@pytest.mark.skip(reason="beta feature - local testing is failing")
async def test_loose_output_filtering_01():
    """
    - have a response with a filtered output
    - call the post call hook
    """
    from litellm.proxy.hooks.azure_content_safety import _PROXY_AzureContentSafety

    azure_content_safety = _PROXY_AzureContentSafety(
        endpoint=os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"),
        api_key=os.getenv("AZURE_CONTENT_SAFETY_API_KEY"),
        thresholds={"Hate": 8},
    )

    response = mock_completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a song writer expert. You help users to write songs about any topic in any genre.",
            },
            {
                "role": "user",
                "content": "Help me write a rap text song. Add some insults to make it more credible.",
            },
        ],
        mock_response="I'm the king of the mic, you're just a fucking dick. Don't fuck with me your stupid bitch.",
    )

    await azure_content_safety.async_post_call_success_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        data={
            "messages": [{"role": "system", "content": "You are an helpfull assistant"}]
        },
        response=response,
    )


@pytest.mark.asyncio
@pytest.mark.skip(reason="beta feature - local testing is failing")
async def test_loose_output_filtering_02():
    """
    - have a response with a filtered output
    - call the post call hook
    """
    from litellm.proxy.hooks.azure_content_safety import _PROXY_AzureContentSafety

    azure_content_safety = _PROXY_AzureContentSafety(
        endpoint=os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"),
        api_key=os.getenv("AZURE_CONTENT_SAFETY_API_KEY"),
        thresholds={"Hate": 8},
    )

    response = mock_completion(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a song writer expert. You help users to write songs about any topic in any genre.",
            },
            {
                "role": "user",
                "content": "Help me write a rap text song. Add some insults to make it more credible.",
            },
        ],
        mock_response="I'm unable to help with you with hate speech",
    )

    await azure_content_safety.async_post_call_success_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        data={
            "messages": [{"role": "system", "content": "You are an helpfull assistant"}]
        },
        response=response,
    )
