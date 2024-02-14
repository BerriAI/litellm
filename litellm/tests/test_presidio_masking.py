# What is this?
## Unit test for presidio pii masking
import sys, os, asyncio, time, random
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm.proxy.hooks.presidio_pii_masking import _OPTIONAL_PresidioPIIMasking
from litellm import Router, mock_completion
from litellm.proxy.utils import ProxyLogging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache


@pytest.mark.asyncio
async def test_output_parsing():
    """
    - have presidio pii masking - mask an input message
    - make llm completion call
    - have presidio pii masking - output parse message
    - assert that no masked tokens are in the input message
    """
    litellm.output_parse_pii = True
    pii_masking = _OPTIONAL_PresidioPIIMasking(mock_testing=True)

    initial_message = [
        {
            "role": "user",
            "content": "hello world, my name is Jane Doe. My number is: 034453334",
        }
    ]

    filtered_message = [
        {
            "role": "user",
            "content": "hello world, my name is <PERSON>. My number is: <PHONE_NUMBER>",
        }
    ]

    pii_masking.pii_tokens = {"<PERSON>": "Jane Doe", "<PHONE_NUMBER>": "034453334"}

    response = mock_completion(
        model="gpt-3.5-turbo",
        messages=filtered_message,
        mock_response="Hello <PERSON>! How can I assist you today?",
    )
    new_response = await pii_masking.async_post_call_success_hook(
        user_api_key_dict=UserAPIKeyAuth(), response=response
    )

    assert (
        new_response.choices[0].message.content
        == "Hello Jane Doe! How can I assist you today?"
    )


# asyncio.run(test_output_parsing())
