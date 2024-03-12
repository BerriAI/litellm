import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
import litellm
from litellm import (
    get_max_tokens,
    model_cost,
    open_ai_chat_completion_models,
    TranscriptionResponse,
)
import pytest


def test_get_gpt3_tokens():
    max_tokens = get_max_tokens("gpt-3.5-turbo")
    print(max_tokens)
    assert max_tokens == 4096
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
            model="gpt-35-turbo",  # azure always has model written like this
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


# test_cost_azure_gpt_35()


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


def test_cost_openai_image_gen():
    cost = litellm.completion_cost(
        model="dall-e-2",
        size="1024-x-1024",
        quality="standard",
        n=1,
        call_type="image_generation",
    )
    assert cost == 0.019922944


def test_cost_bedrock_pricing():
    """
    - get pricing specific to region for a model
    """
    from litellm import ModelResponse, Choices, Message
    from litellm.utils import Usage

    litellm.set_verbose = True
    input_tokens = litellm.token_counter(
        model="bedrock/anthropic.claude-instant-v1",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )
    print(f"input_tokens: {input_tokens}")
    output_tokens = litellm.token_counter(
        model="bedrock/anthropic.claude-instant-v1",
        text="It's all going well",
        count_response_tokens=True,
    )
    print(f"output_tokens: {output_tokens}")
    resp = ModelResponse(
        id="chatcmpl-e41836bb-bb8b-4df2-8e70-8f3e160155ac",
        choices=[
            Choices(
                finish_reason=None,
                index=0,
                message=Message(
                    content="It's all going well",
                    role="assistant",
                ),
            )
        ],
        created=1700775391,
        model="anthropic.claude-instant-v1",
        object="chat.completion",
        system_fingerprint=None,
        usage=Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )
    resp._hidden_params = {
        "custom_llm_provider": "bedrock",
        "region_name": "ap-northeast-1",
    }

    cost = litellm.completion_cost(
        model="anthropic.claude-instant-v1",
        completion_response=resp,
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )
    predicted_cost = input_tokens * 0.00000223 + 0.00000755 * output_tokens
    assert cost == predicted_cost


@pytest.mark.skip(reason="AWS disabled our access")
def test_cost_bedrock_pricing_actual_calls():
    litellm.set_verbose = True
    model = "anthropic.claude-instant-v1"
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    response = litellm.completion(model=model, messages=messages)
    assert response._hidden_params["region_name"] is not None
    cost = litellm.completion_cost(
        completion_response=response,
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )
    assert cost > 0


def test_whisper_openai():
    litellm.set_verbose = True
    transcription = TranscriptionResponse(
        text="Four score and seven years ago, our fathers brought forth on this continent a new nation, conceived in liberty and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure."
    )
    transcription._hidden_params = {
        "model": "whisper-1",
        "custom_llm_provider": "openai",
        "optional_params": {},
        "model_id": None,
    }
    _total_time_in_seconds = 3

    transcription._response_ms = _total_time_in_seconds * 1000
    cost = litellm.completion_cost(model="whisper-1", completion_response=transcription)

    print(f"cost: {cost}")
    print(f"whisper dict: {litellm.model_cost['whisper-1']}")
    expected_cost = round(
        litellm.model_cost["whisper-1"]["output_cost_per_second"]
        * _total_time_in_seconds,
        5,
    )
    assert cost == expected_cost


def test_whisper_azure():
    litellm.set_verbose = True
    transcription = TranscriptionResponse(
        text="Four score and seven years ago, our fathers brought forth on this continent a new nation, conceived in liberty and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure."
    )
    transcription._hidden_params = {
        "model": "whisper-1",
        "custom_llm_provider": "azure",
        "optional_params": {},
        "model_id": None,
    }
    _total_time_in_seconds = 3

    transcription._response_ms = _total_time_in_seconds * 1000
    cost = litellm.completion_cost(
        model="azure/azure-whisper", completion_response=transcription
    )

    print(f"cost: {cost}")
    print(f"whisper dict: {litellm.model_cost['whisper-1']}")
    expected_cost = round(
        litellm.model_cost["whisper-1"]["output_cost_per_second"]
        * _total_time_in_seconds,
        5,
    )
    assert cost == expected_cost
