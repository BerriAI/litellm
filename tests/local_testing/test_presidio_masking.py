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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import litellm
from litellm import Router, mock_completion
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.presidio import (
    _OPTIONAL_PresidioPIIMasking,
)
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
    litellm.set_verbose = True
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
        user_api_key_dict=UserAPIKeyAuth(),
        data={
            "messages": [{"role": "system", "content": "You are an helpfull assistant"}]
        },
        response=response,
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


@pytest.mark.asyncio
async def test_presidio_pii_masking_logging_output_only_no_pre_api_hook():
    from litellm.types.guardrails import GuardrailEventHooks

    pii_masking = _OPTIONAL_PresidioPIIMasking(
        logging_only=True,
        mock_testing=True,
        mock_redacted_text=input_b_anonymizer_results,
    )

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    test_messages = [
        {
            "role": "user",
            "content": "My name is Jane Doe, who are you? Say my name in your response",
        }
    ]

    assert (
        pii_masking.should_run_guardrail(
            data={"messages": test_messages},
            event_type=GuardrailEventHooks.pre_call,
        )
        is False
    )


@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)
@pytest.mark.asyncio
async def test_presidio_pii_masking_logging_output_only_logged_response(sync_mode):
    import litellm

    litellm.set_verbose = True
    pii_masking = _OPTIONAL_PresidioPIIMasking(
        logging_only=True,
        mock_testing=True,
        mock_redacted_text=input_b_anonymizer_results,
        guardrail_name="presidio",
    )

    test_messages = [
        {
            "role": "user",
            "content": "My name is Jane Doe, who are you? Say my name in your response",
        }
    ]

    if sync_mode:
        target_function = "log_success_event"
        mock_call = MagicMock()
    else:
        target_function = "async_log_success_event"
        mock_call = AsyncMock()

    with patch.object(pii_masking, target_function, new=mock_call) as mock_call:
        litellm.callbacks = [pii_masking]
        if sync_mode:
            response = litellm.completion(
                model="gpt-3.5-turbo", messages=test_messages, mock_response="Hi Peter!"
            )
            time.sleep(3)
        else:
            response = await litellm.acompletion(
                model="gpt-3.5-turbo", messages=test_messages, mock_response="Hi Peter!"
            )
            await asyncio.sleep(3)

        assert response.choices[0].message.content == "Hi Peter!"  # type: ignore

        mock_call.assert_called_once()

        print(mock_call.call_args.kwargs["kwargs"]["messages"][0]["content"])

        assert (
            mock_call.call_args.kwargs["kwargs"]["messages"][0]["content"]
            == "My name is <PERSON>, who are you? Say my name in your response"
        )


@pytest.mark.asyncio
async def test_presidio_pii_masking_logging_output_only_logged_response_guardrails_config():
    from typing import Dict, List, Optional

    import litellm
    from litellm.proxy.guardrails.init_guardrails import initialize_guardrails
    from litellm.types.guardrails import (
        GuardrailItem,
        GuardrailItemSpec,
        GuardrailEventHooks,
    )

    litellm.set_verbose = True
    os.environ["PRESIDIO_ANALYZER_API_BASE"] = "http://localhost:5002"
    os.environ["PRESIDIO_ANONYMIZER_API_BASE"] = "http://localhost:5001"

    guardrails_config: List[Dict[str, GuardrailItemSpec]] = [
        {
            "pii_masking": {
                "callbacks": ["presidio"],
                "default_on": True,
                "logging_only": True,
            }
        }
    ]
    litellm_settings = {"guardrails": guardrails_config}

    assert len(litellm.guardrail_name_config_map) == 0
    initialize_guardrails(
        guardrails_config=guardrails_config,
        premium_user=True,
        config_file_path="",
        litellm_settings=litellm_settings,
    )

    assert len(litellm.guardrail_name_config_map) == 1

    pii_masking_obj: Optional[_OPTIONAL_PresidioPIIMasking] = None
    for callback in litellm.callbacks:
        print(f"CALLBACK: {callback}")
        if isinstance(callback, _OPTIONAL_PresidioPIIMasking):
            pii_masking_obj = callback

    assert pii_masking_obj is not None

    assert hasattr(pii_masking_obj, "logging_only")
    assert pii_masking_obj.event_hook == GuardrailEventHooks.logging_only

    assert pii_masking_obj.should_run_guardrail(
        data={}, event_type=GuardrailEventHooks.logging_only
    )
