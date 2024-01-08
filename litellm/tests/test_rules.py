#### What this tests ####
#    This tests setting rules before / after making llm api calls
import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion, acompletion


def my_pre_call_rule(input: str):
    print(f"input: {input}")
    print(f"INSIDE MY PRE CALL RULE, len(input) - {len(input)}")
    if len(input) > 10:
        return False
    return True


def my_post_call_rule(input: str):
    input = input.lower()
    print(f"input: {input}")
    print(f"INSIDE MY POST CALL RULE, len(input) - {len(input)}")
    if "sorry" in input:
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
    except:
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


# test_pre_call_rule()
## Test 2: Post-call rule
# commenting out of ci/cd since llm's have variable output which was causing our pipeline to fail erratically.
# def test_post_call_rule():
#     try:
#         litellm.pre_call_rules = []
#         litellm.post_call_rules = [my_post_call_rule]
#         ### completion
#         response = completion(model="gpt-3.5-turbo",
#                       messages=[{"role": "user", "content": "say sorry"}],
#                       fallbacks=["deepinfra/Gryphe/MythoMax-L2-13b"])
#         pytest.fail(f"Completion call should have been failed. ")
#     except:
#         pass
#     print(f"MAKING ACOMPLETION CALL")
#     # litellm.set_verbose = True
#     ### async completion
#     async def test_async_response():
#         messages=[{"role": "user", "content": "say sorry"}]
#         try:
#             response = await acompletion(model="gpt-3.5-turbo", messages=messages)
#             pytest.fail(f"acompletion call should have been failed.")
#         except Exception as e:
#             pass
#     asyncio.run(test_async_response())
#     litellm.pre_call_rules = []
#     litellm.post_call_rules = []

# test_post_call_rule()
