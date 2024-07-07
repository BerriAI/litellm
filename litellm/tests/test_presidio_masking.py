# What is this?
## Unit test for presidio pii masking
import asyncio
import os
import random
import sys
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv

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
from litellm.proxy.hooks.presidio_pii_masking import _OPTIONAL_PresidioPIIMasking
from litellm.proxy.utils import ProxyLogging


@pytest.mark.parametrize(
    "base_url",
    [
        "presidio-analyzer-s3pa:10000",
        "https://presidio-analyzer-s3pa:10000",
        "http://presidio-analyzer-s3pa:10000",
    ],
)
def test_validate_environment_missing_http(base_url):
    pii_masking = _OPTIONAL_PresidioPIIMasking(mock_testing=True)

    os.environ["PRESIDIO_ANALYZER_API_BASE"] = f"{base_url}/analyze"
    os.environ["PRESIDIO_ANONYMIZER_API_BASE"] = f"{base_url}/anonymize"
    pii_masking.validate_environment()

    expected_url = base_url
    if not (base_url.startswith("https://") or base_url.startswith("http://")):
        expected_url = "http://" + base_url

    assert (
        pii_masking.presidio_anonymizer_api_base == f"{expected_url}/anonymize/"
    ), "Got={}, Expected={}".format(
        pii_masking.presidio_anonymizer_api_base, f"{expected_url}/anonymize/"
    )
    assert pii_masking.presidio_analyzer_api_base == f"{expected_url}/analyze/"


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


### UNIT TESTS FOR PRESIDIO PII MASKING ###

input_a_anonymizer_results = {
    "text": "hello world, my name is <PERSON>. My number is: <PHONE_NUMBER>",
    "items": [
        {
            "start": 48,
            "end": 62,
            "entity_type": "PHONE_NUMBER",
            "text": "<PHONE_NUMBER>",
            "operator": "replace",
        },
        {
            "start": 24,
            "end": 32,
            "entity_type": "PERSON",
            "text": "<PERSON>",
            "operator": "replace",
        },
    ],
}

input_b_anonymizer_results = {
    "text": "My name is <PERSON>, who are you? Say my name in your response",
    "items": [
        {
            "start": 11,
            "end": 19,
            "entity_type": "PERSON",
            "text": "<PERSON>",
            "operator": "replace",
        }
    ],
}


#   Test if PII masking works with input A
@pytest.mark.asyncio
async def test_presidio_pii_masking_input_a():
    """
    Tests to see if correct parts of sentence anonymized
    """
    pii_masking = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True, mock_redacted_text=input_a_anonymizer_results
    )

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    new_data = await pii_masking.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={
            "messages": [
                {
                    "role": "user",
                    "content": "hello world, my name is Jane Doe. My number is: 23r323r23r2wwkl",
                }
            ]
        },
        call_type="completion",
    )

    assert "<PERSON>" in new_data["messages"][0]["content"]
    assert "<PHONE_NUMBER>" in new_data["messages"][0]["content"]


#   Test if PII masking works with input B (also test if the response != A's response)
@pytest.mark.asyncio
async def test_presidio_pii_masking_input_b():
    """
    Tests to see if correct parts of sentence anonymized
    """
    pii_masking = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True, mock_redacted_text=input_b_anonymizer_results
    )

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    new_data = await pii_masking.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={
            "messages": [
                {
                    "role": "user",
                    "content": "My name is Jane Doe, who are you? Say my name in your response",
                }
            ]
        },
        call_type="completion",
    )

    assert "<PERSON>" in new_data["messages"][0]["content"]
    assert "<PHONE_NUMBER>" not in new_data["messages"][0]["content"]
