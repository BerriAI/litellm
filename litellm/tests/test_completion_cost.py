import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
import litellm
from litellm import get_max_tokens, model_cost, open_ai_chat_completion_models
import pytest


def test_get_gpt3_tokens():
    max_tokens = get_max_tokens("gpt-3.5-turbo")
    print(max_tokens)
    assert max_tokens == 4097
    # print(results)


# test_get_gpt3_tokens()


def test_get_palm_tokens():
    # # 🦄🦄🦄🦄🦄🦄🦄🦄
    max_tokens = get_max_tokens("palm/chat-bison")
    assert max_tokens == 4096
    print(max_tokens)


# test_get_palm_tokens()


def test_zephyr_hf_tokens():
    max_tokens = get_max_tokens("huggingface/HuggingFaceH4/zephyr-7b-beta")
    print(max_tokens)
    assert max_tokens == 32768


# test_zephyr_hf_tokens()


def test_cost_ft_gpt_35():
    try:
        # this tests if litellm.completion_cost can calculate cost for ft:gpt-3.5-turbo:my-org:custom_suffix:id
        # it needs to lookup  ft:gpt-3.5-turbo in the litellm model_cost map to get the correct cost
        from litellm import ModelResponse, Choices, Message
        from litellm.utils import Usage

        resp = ModelResponse(
            id="chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac",
            choices=[
                Choices(
                    finish_reason=None,
                    index=0,
                    message=Message(
                        content=" Sure! Here is a short poem about the sky:\n\nA canvas of blue, a",
                        role="assistant",
                    ),
                )
            ],
            created=1700775391,
            model="ft:gpt-3.5-turbo:my-org:custom_suffix:id",
            object="chat.completion",
            system_fingerprint=None,
            usage=Usage(prompt_tokens=21, completion_tokens=17, total_tokens=38),
        )

        cost = litellm.completion_cost(completion_response=resp)
        print("\n Calculated Cost for ft:gpt-3.5", cost)
        input_cost = model_cost["ft:gpt-3.5-turbo"]["input_cost_per_token"]
        output_cost = model_cost["ft:gpt-3.5-turbo"]["output_cost_per_token"]
        print(input_cost, output_cost)
        expected_cost = (input_cost * resp.usage.prompt_tokens) + (
            output_cost * resp.usage.completion_tokens
        )
        print("\n Excpected cost", expected_cost)
        assert cost == expected_cost
    except Exception as e:
        pytest.fail(
            f"Cost Calc failed for ft:gpt-3.5. Expected {expected_cost}, Calculated cost {cost}"
        )


# test_cost_ft_gpt_35()


def test_cost_azure_gpt_35():
    try:
        # this tests if litellm.completion_cost can calculate cost for azure/chatgpt-deployment-2 which maps to azure/gpt-3.5-turbo
        # for this test we check if passing `model` to completion_cost overrides the completion cost
        from litellm import ModelResponse, Choices, Message
        from litellm.utils import Usage

        resp = ModelResponse(
            id="chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac",
            choices=[
                Choices(
                    finish_reason=None,
                    index=0,
                    message=Message(
                        content=" Sure! Here is a short poem about the sky:\n\nA canvas of blue, a",
                        role="assistant",
                    ),
                )
            ],
            model="azure/gpt-35-turbo",  # azure always has model written like this
            usage=Usage(prompt_tokens=21, completion_tokens=17, total_tokens=38),
        )

        cost = litellm.completion_cost(
            completion_response=resp, model="azure/gpt-35-turbo"
        )
        print("\n Calculated Cost for azure/gpt-3.5-turbo", cost)
        input_cost = model_cost["azure/gpt-35-turbo"]["input_cost_per_token"]
        output_cost = model_cost["azure/gpt-35-turbo"]["output_cost_per_token"]
        expected_cost = (input_cost * resp.usage.prompt_tokens) + (
            output_cost * resp.usage.completion_tokens
        )
        print("\n Excpected cost", expected_cost)
        assert cost == expected_cost
    except Exception as e:
        pytest.fail(
            f"Cost Calc failed for azure/gpt-3.5-turbo. Expected {expected_cost}, Calculated cost {cost}"
        )


test_cost_azure_gpt_35()


def test_cost_azure_embedding():
    try:
        import asyncio

        litellm.set_verbose = True

        async def _test():
            response = await litellm.aembedding(
                model="azure/azure-embedding-model",
                input=["good morning from litellm", "gm"],
            )

            print(response)

            return response

        response = asyncio.run(_test())

        cost = litellm.completion_cost(completion_response=response)

        print("Cost", cost)
        expected_cost = float("7e-07")
        assert cost == expected_cost

    except Exception as e:
        pytest.fail(
            f"Cost Calc failed for azure/gpt-3.5-turbo. Expected {expected_cost}, Calculated cost {cost}"
        )


# test_cost_azure_embedding()
