#### What this tests ####
#    This tests setting rules before / after making llm api calls
import asyncio
import re
import time
import traceback

import pytest

import litellm
from litellm import acompletion, completion


def my_pre_call_rule(input: str):
    print(f"input: {input}")
    print(f"INSIDE MY PRE CALL RULE, len(input) - {len(input)}")
    if len(input) > 10:
        return False
    return True


## Test 1: Pre-call rule
def test_pre_call_rule():
    try:
        litellm.pre_call_rules = [my_pre_call_rule]
        ### completion
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "say something inappropriate"}],
        )
        pytest.fail(f"Completion call should have been failed. ")
    except Exception:
        pass

    ### async completion
    async def test_async_response():
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            response = await acompletion(model="gpt-3.5-turbo", messages=messages)
            pytest.fail(f"acompletion call should have been failed. ")
        except Exception as e:
            pass

    asyncio.run(test_async_response())
    litellm.pre_call_rules = []


def my_post_call_rule(input: str):
    input = input.lower()
    print(f"input: {input}")
    print(f"INSIDE MY POST CALL RULE, len(input) - {len(input)}")
    if len(input) < 200:
        return {
            "decision": False,
            "message": "This violates LiteLLM Proxy Rules. Response too short",
        }
    return {"decision": True}


def my_post_call_rule_2(input: str):
    input = input.lower()
    print(f"input: {input}")
    print(f"INSIDE MY POST CALL RULE, len(input) - {len(input)}")
    if len(input) < 200 and len(input) > 0:
        return {
            "decision": False,
            "message": "This violates LiteLLM Proxy Rules. Response too short",
        }
    return {"decision": True}


# test_pre_call_rule()
# Test 2: Post-call rule
# commenting out of ci/cd since llm's have variable output which was causing our pipeline to fail erratically.
def test_post_call_rule():
    litellm.pre_call_rules = []
    litellm.post_call_rules = [my_post_call_rule]

    ### completion
    with pytest.raises(Exception, match=re.escape("This violates LiteLLM Proxy Rules. Response too short")) as exc_info:
        completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "say sorry"}],
            max_tokens=2,
        )
    assert exc_info.value.message == "This violates LiteLLM Proxy Rules. Response too short"
    # print(f"MAKING ACOMPLETION CALL")
    # litellm.set_verbose = True
    ### async completion
    # async def test_async_response():
    #     messages=[{"role": "user", "content": "say sorry"}]
    #     try:
    #         response = await acompletion(model="gpt-3.5-turbo", messages=messages)
    #         pytest.fail(f"acompletion call should have been failed.")
    #     except Exception as e:
    #         pass
    # asyncio.run(test_async_response())
    litellm.pre_call_rules = []
    litellm.post_call_rules = []


# test_post_call_rule()


def test_post_call_rule_streaming():
    litellm.pre_call_rules = []
    litellm.post_call_rules = [my_post_call_rule_2]
    ### completion
    response = completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "say sorry"}],
        max_tokens=2,
        stream=True,
    )

    with pytest.raises(Exception, match=re.escape("This violates LiteLLM Proxy Rules. Response too short")) as exc_info:
        list(response)
    assert "This violates LiteLLM Proxy Rules. Response too short" in exc_info.value.message


@pytest.mark.asyncio
async def test_post_call_processing_error_async_response():
    try:
        response = await acompletion(
            model="command-nightly",  # Just used as an example
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            api_base="https://openai-proxy.berriai.repl.co",  # Just used as an example
            custom_llm_provider="openai",
        )
        pytest.fail("This call should have failed")
    except Exception as e:
        pass
