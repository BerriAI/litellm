from typing import Dict

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.presidio import (
    _OPTIONAL_PresidioPIIMasking,
)
from litellm.types.utils import Choices, Message, ModelResponse


input_anonymizer_results_a = {
    "text": "hello world, my name is <PERSON>. My number is: <PHONE_NUMBER>",
    "items": [
        {
            "start": 48,
            "end": 63,
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


@pytest.mark.asyncio
async def test_token_map_populated_and_success_cleanup():
    guard = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        mock_redacted_text=input_anonymizer_results_a,
        output_parse_pii=True,
    )
    cache = DualCache()
    user_key = UserAPIKeyAuth(api_key="sk-test")

    call_id = "call-1"
    request_data = {
        "litellm_call_id": call_id,
        "messages": [
            {
                "role": "user",
                "content": "hello world, my name is Jane Doe. My number is: 23r323r23r2wwkl",
            }
        ],
        "metadata": {"standard_logging_guardrail_information": {}},
    }

    # pre_call should analyze/anonymize and populate token map
    new_data = await guard.async_pre_call_hook(
        user_api_key_dict=user_key,
        cache=cache,
        data=request_data,
        call_type="completion",
    )

    assert new_data["messages"][0]["content"] == "hello world, my name is <PERSON>. My number is: <PHONE_NUMBER>"

    # tokens are recorded under the instance map keyed by call id
    tokens: Dict[str, str] = guard._pii_token_maps.get(call_id) or {}
    assert tokens.get("<PERSON>") == "Jane Doe"
    assert tokens.get("<PHONE_NUMBER>") == "23r323r23r2wwkl"

    # success hook should unmask in response and cleanup the map
    response = ModelResponse(
        model="test-model",
        choices=[
            Choices(message=Message(role="assistant", content="Hi <PERSON> (<PHONE_NUMBER>)"))
        ],
    )

    updated = await guard.async_post_call_success_hook(
        data={"litellm_call_id": call_id},
        user_api_key_dict=user_key,
        response=response,
    )
    assert isinstance(updated, ModelResponse)
    assert (
        updated.choices[0].message.content
        == "Hi Jane Doe (23r323r23r2wwkl)"
    )
    # map is cleaned
    assert guard._pii_token_maps.get(call_id) is None


@pytest.mark.asyncio
async def test_failure_hook_cleans_token_map():
    guard = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        mock_redacted_text=input_anonymizer_results_a,
        output_parse_pii=True,
    )
    cache = DualCache()
    user_key = UserAPIKeyAuth(api_key="sk-test")

    call_id = "call-failed"
    request_data = {
        "litellm_call_id": call_id,
        "messages": [
            {
                "role": "user",
                "content": "hello world, my name is Jane Doe. My number is: 23r323r23r2wwkl",
            }
        ],
        "metadata": {"standard_logging_guardrail_information": {}},
    }

    _ = await guard.async_pre_call_hook(
        user_api_key_dict=user_key,
        cache=cache,
        data=request_data,
        call_type="completion",
    )
    assert guard._pii_token_maps.get(call_id)

    # simulate a failure after pre_call
    await guard.async_post_call_failure_hook(
        request_data={"litellm_call_id": call_id},
        original_exception=Exception("boom"),
        user_api_key_dict=user_key,
    )
    assert guard._pii_token_maps.get(call_id) is None


@pytest.mark.asyncio
async def test_isolation_between_call_ids():
    guard = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        mock_redacted_text=input_anonymizer_results_a,
        output_parse_pii=True,
    )
    cache = DualCache()
    user_key = UserAPIKeyAuth(api_key="sk-test")

    # call A
    call_id_a = "call-a"
    req_a = {
        "litellm_call_id": call_id_a,
        "messages": [
            {
                "role": "user",
                "content": "hello world, my name is Jane Doe. My number is: 1111",
            }
        ],
        "metadata": {"standard_logging_guardrail_information": {}},
    }
    # call B
    call_id_b = "call-b"
    req_b = {
        "litellm_call_id": call_id_b,
        "messages": [
            {
                "role": "user",
                "content": "hello world, my name is Alice. My number is: 2222",
            }
        ],
        "metadata": {"standard_logging_guardrail_information": {}},
    }

    # Run pre_call for both
    await guard.async_pre_call_hook(user_api_key_dict=user_key, cache=cache, data=req_a, call_type="completion")
    await guard.async_pre_call_hook(user_api_key_dict=user_key, cache=cache, data=req_b, call_type="completion")

    # Both maps exist separately
    assert guard._pii_token_maps.get(call_id_a) is not None
    assert guard._pii_token_maps.get(call_id_b) is not None
    assert guard._pii_token_maps.get(call_id_a) is not guard._pii_token_maps.get(call_id_b)

    # Cleanup only A
    await guard.async_post_call_failure_hook(
        request_data={"litellm_call_id": call_id_a},
        original_exception=Exception("fail a"),
        user_api_key_dict=user_key,
    )
    assert guard._pii_token_maps.get(call_id_a) is None
    assert guard._pii_token_maps.get(call_id_b) is not None

