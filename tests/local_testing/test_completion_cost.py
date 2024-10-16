import os
import sys
import traceback

import litellm.cost_calculator

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import os
import time
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm import (
    TranscriptionResponse,
    completion_cost,
    cost_per_token,
    get_max_tokens,
    model_cost,
    open_ai_chat_completion_models,
)
from litellm.types.utils import PromptTokensDetails
from litellm.litellm_core_utils.litellm_logging import CustomLogger


class CustomLoggingHandler(CustomLogger):
    response_cost: Optional[float] = None

    def __init__(self):
        super().__init__()

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.response_cost = kwargs["response_cost"]

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"kwargs - {kwargs}")
        print(f"kwargs response cost - {kwargs.get('response_cost')}")
        self.response_cost = kwargs["response_cost"]

        print(f"response_cost: {self.response_cost} ")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print("Reaches log failure event!")
        self.response_cost = kwargs["response_cost"]

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print("Reaches async log failure event!")
        self.response_cost = kwargs["response_cost"]


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_custom_pricing(sync_mode):
    new_handler = CustomLoggingHandler()
    litellm.callbacks = [new_handler]
    if sync_mode:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey!"}],
            mock_response="What do you want?",
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
        )
        time.sleep(5)
    else:
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey!"}],
            mock_response="What do you want?",
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
        )

        await asyncio.sleep(5)

    print(f"new_handler.response_cost: {new_handler.response_cost}")
    assert new_handler.response_cost is not None

    assert new_handler.response_cost == 0


@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)
@pytest.mark.asyncio
async def test_failure_completion_cost(sync_mode):
    new_handler = CustomLoggingHandler()
    litellm.callbacks = [new_handler]
    if sync_mode:
        try:
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hey!"}],
                mock_response=Exception("this should trigger an error"),
            )
        except Exception:
            pass
        time.sleep(5)
    else:
        try:
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hey!"}],
                mock_response=Exception("this should trigger an error"),
            )
        except Exception:
            pass
        await asyncio.sleep(5)

    print(f"new_handler.response_cost: {new_handler.response_cost}")
    assert new_handler.response_cost is not None

    assert new_handler.response_cost == 0


def test_custom_pricing_as_completion_cost_param():
    from litellm import Choices, Message, ModelResponse
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

    cost = litellm.completion_cost(
        completion_response=resp,
        custom_cost_per_token={
            "input_cost_per_token": 1000,
            "output_cost_per_token": 20,
        },
    )

    expected_cost = 1000 * 21 + 17 * 20

    assert round(cost, 5) == round(expected_cost, 5)


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
        from litellm import Choices, Message, ModelResponse
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

        cost = litellm.completion_cost(
            completion_response=resp, custom_llm_provider="openai"
        )
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
        from litellm import Choices, Message, ModelResponse
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
        pytest.fail(f"Cost Calc failed for azure/gpt-3.5-turbo. {str(e)}")


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
    from litellm import Choices, Message, ModelResponse
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


def test_cost_bedrock_pricing_actual_calls():
    litellm.set_verbose = True
    model = "anthropic.claude-instant-v1"
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    response = litellm.completion(
        model=model, messages=messages, mock_response="hello cool one"
    )

    print("response", response)
    cost = litellm.completion_cost(
        model="bedrock/anthropic.claude-instant-v1",
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


def test_dalle_3_azure_cost_tracking():
    litellm.set_verbose = True
    # model = "azure/dall-e-3-test"
    # response = litellm.image_generation(
    #     model=model,
    #     prompt="A cute baby sea otter",
    #     api_version="2023-12-01-preview",
    #     api_base=os.getenv("AZURE_SWEDEN_API_BASE"),
    #     api_key=os.getenv("AZURE_SWEDEN_API_KEY"),
    #     base_model="dall-e-3",
    # )
    # print(f"response: {response}")
    response = litellm.ImageResponse(
        created=1710265780,
        data=[
            {
                "b64_json": None,
                "revised_prompt": "A close-up image of an adorable baby sea otter. Its fur is thick and fluffy to provide buoyancy and insulation against the cold water. Its eyes are round, curious and full of life. It's lying on its back, floating effortlessly on the calm sea surface under the warm sun. Surrounding the otter are patches of colorful kelp drifting along the gentle waves, giving the scene a touch of vibrancy. The sea otter has its small paws folded on its chest, and it seems to be taking a break from its play.",
                "url": "https://dalleprodsec.blob.core.windows.net/private/images/3e5d00f3-700e-4b75-869d-2de73c3c975d/generated_00.png?se=2024-03-13T17%3A49%3A51Z&sig=R9RJD5oOSe0Vp9Eg7ze%2FZ8QR7ldRyGH6XhMxiau16Jc%3D&ske=2024-03-19T11%3A08%3A03Z&skoid=e52d5ed7-0657-4f62-bc12-7e5dbb260a96&sks=b&skt=2024-03-12T11%3A08%3A03Z&sktid=33e01921-4d64-4f8c-a055-5bdaffd5e33d&skv=2020-10-02&sp=r&spr=https&sr=b&sv=2020-10-02",
            }
        ],
    )
    response.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    response._hidden_params = {"model": "dall-e-3", "model_id": None}
    print(f"response hidden params: {response._hidden_params}")
    cost = litellm.completion_cost(
        completion_response=response, call_type="image_generation"
    )
    assert cost > 0


def test_replicate_llama3_cost_tracking():
    litellm.set_verbose = True
    model = "replicate/meta/meta-llama-3-8b-instruct"
    litellm.register_model(
        {
            "replicate/meta/meta-llama-3-8b-instruct": {
                "input_cost_per_token": 0.00000005,
                "output_cost_per_token": 0.00000025,
                "litellm_provider": "replicate",
            }
        }
    )
    response = litellm.ModelResponse(
        id="chatcmpl-cad7282f-7f68-41e7-a5ab-9eb33ae301dc",
        choices=[
            litellm.utils.Choices(
                finish_reason="stop",
                index=0,
                message=litellm.utils.Message(
                    content="I'm doing well, thanks for asking! I'm here to help you with any questions or tasks you may have. How can I assist you today?",
                    role="assistant",
                ),
            )
        ],
        created=1714401369,
        model="replicate/meta/meta-llama-3-8b-instruct",
        object="chat.completion",
        system_fingerprint=None,
        usage=litellm.utils.Usage(
            prompt_tokens=48, completion_tokens=31, total_tokens=79
        ),
    )
    cost = litellm.completion_cost(
        completion_response=response,
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )

    print(f"cost: {cost}")
    cost = round(cost, 5)
    expected_cost = round(
        litellm.model_cost["replicate/meta/meta-llama-3-8b-instruct"][
            "input_cost_per_token"
        ]
        * 48
        + litellm.model_cost["replicate/meta/meta-llama-3-8b-instruct"][
            "output_cost_per_token"
        ]
        * 31,
        5,
    )
    assert cost == expected_cost


@pytest.mark.parametrize("is_streaming", [True, False])  #
def test_groq_response_cost_tracking(is_streaming):
    from litellm.utils import (
        CallTypes,
        Choices,
        Delta,
        Message,
        ModelResponse,
        StreamingChoices,
        Usage,
    )

    response = ModelResponse(
        id="chatcmpl-876cce24-e520-4cf8-8649-562a9be11c02",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Hi! I'm an AI, so I don't have emotions or feelings like humans do, but I'm functioning properly and ready to help with any questions or topics you'd like to discuss! How can I assist you today?",
                    role="assistant",
                ),
            )
        ],
        created=1717519830,
        model="llama3-70b-8192",
        object="chat.completion",
        system_fingerprint="fp_c1a4bcec29",
        usage=Usage(completion_tokens=46, prompt_tokens=17, total_tokens=63),
    )
    response._hidden_params["custom_llm_provider"] = "groq"
    print(response)

    response_cost = litellm.response_cost_calculator(
        response_object=response,
        model="groq/llama3-70b-8192",
        custom_llm_provider="groq",
        call_type=CallTypes.acompletion.value,
        optional_params={},
    )

    assert isinstance(response_cost, float)
    assert response_cost > 0.0

    print(f"response_cost: {response_cost}")


from litellm.types.utils import CallTypes


def test_together_ai_qwen_completion_cost():
    input_kwargs = {
        "completion_response": litellm.ModelResponse(
            **{
                "id": "890db0c33c4ef94b-SJC",
                "choices": [
                    {
                        "finish_reason": "eos",
                        "index": 0,
                        "message": {
                            "content": "I am Qwen, a large language model created by Alibaba Cloud.",
                            "role": "assistant",
                        },
                    }
                ],
                "created": 1717900130,
                "model": "together_ai/qwen/Qwen2-72B-Instruct",
                "object": "chat.completion",
                "system_fingerprint": None,
                "usage": {
                    "completion_tokens": 15,
                    "prompt_tokens": 23,
                    "total_tokens": 38,
                },
            }
        ),
        "model": "qwen/Qwen2-72B-Instruct",
        "prompt": "",
        "messages": [],
        "completion": "",
        "total_time": 0.0,
        "call_type": "completion",
        "custom_llm_provider": "together_ai",
        "region_name": None,
        "size": None,
        "quality": None,
        "n": None,
        "custom_cost_per_token": None,
        "custom_cost_per_second": None,
    }

    response = litellm.cost_calculator.get_model_params_and_category(
        model_name="qwen/Qwen2-72B-Instruct", call_type=CallTypes.completion
    )

    assert response == "together-ai-41.1b-80b"


@pytest.mark.parametrize("above_128k", [False, True])
@pytest.mark.parametrize("provider", ["gemini"])
def test_gemini_completion_cost(above_128k, provider):
    """
    Check if cost correctly calculated for gemini models based on context window
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    if provider == "gemini":
        model_name = "gemini-1.5-flash-latest"
    else:
        model_name = "gemini-1.5-flash-preview-0514"
    if above_128k:
        prompt_tokens = 128001.0
        output_tokens = 228001.0
    else:
        prompt_tokens = 128.0
        output_tokens = 228.0
    ## GET MODEL FROM LITELLM.MODEL_INFO
    model_info = litellm.get_model_info(model=model_name, custom_llm_provider=provider)

    ## EXPECTED COST
    if above_128k:
        assert (
            model_info["input_cost_per_token_above_128k_tokens"] is not None
        ), "model info for model={} does not have pricing for > 128k tokens\nmodel_info={}".format(
            model_name, model_info
        )
        assert (
            model_info["output_cost_per_token_above_128k_tokens"] is not None
        ), "model info for model={} does not have pricing for > 128k tokens\nmodel_info={}".format(
            model_name, model_info
        )
        input_cost = (
            prompt_tokens * model_info["input_cost_per_token_above_128k_tokens"]
        )
        output_cost = (
            output_tokens * model_info["output_cost_per_token_above_128k_tokens"]
        )
    else:
        input_cost = prompt_tokens * model_info["input_cost_per_token"]
        output_cost = output_tokens * model_info["output_cost_per_token"]

    ## CALCULATED COST
    calculated_input_cost, calculated_output_cost = cost_per_token(
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        custom_llm_provider=provider,
    )

    assert calculated_input_cost == input_cost
    assert calculated_output_cost == output_cost


def _count_characters(text):
    # Remove white spaces and count characters
    filtered_text = "".join(char for char in text if not char.isspace())
    return len(filtered_text)


def test_vertex_ai_completion_cost():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    text = "The quick brown fox jumps over the lazy dog."
    characters = _count_characters(text=text)

    model_info = litellm.get_model_info(model="gemini-1.5-flash")

    print("\nExpected model info:\n{}\n\n".format(model_info))

    expected_input_cost = characters * model_info["input_cost_per_character"]

    ## CALCULATED COST
    calculated_input_cost, calculated_output_cost = cost_per_token(
        model="gemini-1.5-flash",
        custom_llm_provider="vertex_ai",
        prompt_characters=characters,
        completion_characters=0,
    )

    assert round(expected_input_cost, 6) == round(calculated_input_cost, 6)
    print("expected_input_cost: {}".format(expected_input_cost))
    print("calculated_input_cost: {}".format(calculated_input_cost))


@pytest.mark.skip(reason="new test - WIP, working on fixing this")
def test_vertex_ai_medlm_completion_cost():
    """Test for medlm completion cost ."""

    with pytest.raises(Exception) as e:
        model = "vertex_ai/medlm-medium"
        messages = [{"role": "user", "content": "Test MedLM completion cost."}]
        predictive_cost = completion_cost(
            model=model, messages=messages, custom_llm_provider="vertex_ai"
        )

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model = "vertex_ai/medlm-medium"
    messages = [{"role": "user", "content": "Test MedLM completion cost."}]
    predictive_cost = completion_cost(
        model=model, messages=messages, custom_llm_provider="vertex_ai"
    )
    assert predictive_cost > 0

    model = "vertex_ai/medlm-large"
    messages = [{"role": "user", "content": "Test MedLM completion cost."}]
    predictive_cost = completion_cost(model=model, messages=messages)
    assert predictive_cost > 0


def test_vertex_ai_claude_completion_cost():
    from litellm import Choices, Message, ModelResponse
    from litellm.utils import Usage

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    litellm.set_verbose = True
    input_tokens = litellm.token_counter(
        model="vertex_ai/claude-3-sonnet@20240229",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )
    print(f"input_tokens: {input_tokens}")
    output_tokens = litellm.token_counter(
        model="vertex_ai/claude-3-sonnet@20240229",
        text="It's all going well",
        count_response_tokens=True,
    )
    print(f"output_tokens: {output_tokens}")
    response = ModelResponse(
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
        model="vertex_ai/claude-3-sonnet@20240229",
        object="chat.completion",
        system_fingerprint=None,
        usage=Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )
    cost = litellm.completion_cost(
        model="vertex_ai/claude-3-sonnet@20240229",
        completion_response=response,
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )
    predicted_cost = input_tokens * 0.000003 + 0.000015 * output_tokens
    assert cost == predicted_cost


def test_vertex_ai_embedding_completion_cost(caplog):
    """
    Relevant issue - https://github.com/BerriAI/litellm/issues/4630
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    text = "The quick brown fox jumps over the lazy dog."
    input_tokens = litellm.token_counter(
        model="vertex_ai/textembedding-gecko", text=text
    )

    model_info = litellm.get_model_info(model="vertex_ai/textembedding-gecko")

    print("\nExpected model info:\n{}\n\n".format(model_info))

    expected_input_cost = input_tokens * model_info["input_cost_per_token"]

    ## CALCULATED COST
    calculated_input_cost, calculated_output_cost = cost_per_token(
        model="textembedding-gecko",
        custom_llm_provider="vertex_ai",
        prompt_tokens=input_tokens,
        call_type="aembedding",
    )

    assert round(expected_input_cost, 6) == round(calculated_input_cost, 6)
    print("expected_input_cost: {}".format(expected_input_cost))
    print("calculated_input_cost: {}".format(calculated_input_cost))

    captured_logs = [rec.message for rec in caplog.records]
    for item in captured_logs:
        print("\nitem:{}\n".format(item))
        if (
            "litellm.litellm_core_utils.llm_cost_calc.google.cost_per_character(): Exception occured "
            in item
        ):
            raise Exception("Error log raised for calculating embedding cost")


# def test_vertex_ai_embedding_completion_cost_e2e():
#     """
#     Relevant issue - https://github.com/BerriAI/litellm/issues/4630
#     """
#     from test_amazing_vertex_completion import load_vertex_ai_credentials

#     load_vertex_ai_credentials()
#     os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
#     litellm.model_cost = litellm.get_model_cost_map(url="")

#     text = "The quick brown fox jumps over the lazy dog."
#     input_tokens = litellm.token_counter(
#         model="vertex_ai/textembedding-gecko", text=text
#     )

#     model_info = litellm.get_model_info(model="vertex_ai/textembedding-gecko")

#     print("\nExpected model info:\n{}\n\n".format(model_info))

#     expected_input_cost = input_tokens * model_info["input_cost_per_token"]

#     ## CALCULATED COST
#     resp = litellm.embedding(model="textembedding-gecko", input=[text])

#     calculated_input_cost = resp._hidden_params["response_cost"]

#     assert round(expected_input_cost, 6) == round(calculated_input_cost, 6)
#     print("expected_input_cost: {}".format(expected_input_cost))
#     print("calculated_input_cost: {}".format(calculated_input_cost))

#     assert False


def test_completion_azure_ai():
    try:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm.set_verbose = True
        response = litellm.completion(
            model="azure_ai/Mistral-large-nmefg",
            messages=[{"content": "what llm are you", "role": "user"}],
            max_tokens=15,
            num_retries=3,
            api_base=os.getenv("AZURE_AI_MISTRAL_API_BASE"),
            api_key=os.getenv("AZURE_AI_MISTRAL_API_KEY"),
        )
        print(response)

        assert "response_cost" in response._hidden_params
        assert isinstance(response._hidden_params["response_cost"], float)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_completion_cost_hidden_params(sync_mode):
    litellm.return_response_headers = True
    if sync_mode:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            mock_response="Hello world",
        )
    else:
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            mock_response="Hello world",
        )

    assert "response_cost" in response._hidden_params
    assert isinstance(response._hidden_params["response_cost"], float)


def test_vertex_ai_gemini_predict_cost():
    model = "gemini-1.5-flash"
    messages = [{"role": "user", "content": "Hey, hows it going???"}]
    predictive_cost = completion_cost(model=model, messages=messages)

    assert predictive_cost > 0


def test_vertex_ai_llama_predict_cost():
    model = "meta/llama3-405b-instruct-maas"
    messages = [{"role": "user", "content": "Hey, hows it going???"}]
    custom_llm_provider = "vertex_ai"
    predictive_cost = completion_cost(
        model=model, messages=messages, custom_llm_provider=custom_llm_provider
    )

    assert predictive_cost == 0


@pytest.mark.parametrize("usage", ["litellm_usage", "openai_usage"])
def test_vertex_ai_mistral_predict_cost(usage):
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    if usage == "litellm_usage":
        response_usage = Usage(prompt_tokens=32, completion_tokens=55, total_tokens=87)
    else:
        from openai.types.completion_usage import CompletionUsage

        response_usage = CompletionUsage(
            prompt_tokens=32, completion_tokens=55, total_tokens=87
        )
    response_object = ModelResponse(
        id="26c0ef045020429d9c5c9b078c01e564",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Hello! I'm Litellm Bot, your helpful assistant. While I can't provide real-time weather updates, I can help you find a reliable weather service or guide you on how to check the weather on your device. Would you like assistance with that?",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
        created=1722124652,
        model="vertex_ai/mistral-large",
        object="chat.completion",
        system_fingerprint=None,
        usage=response_usage,
    )
    model = "mistral-large@2407"
    messages = [{"role": "user", "content": "Hey, hows it going???"}]
    custom_llm_provider = "vertex_ai"
    predictive_cost = completion_cost(
        completion_response=response_object,
        model=model,
        messages=messages,
        custom_llm_provider=custom_llm_provider,
    )

    assert predictive_cost > 0


@pytest.mark.parametrize("model", ["openai/tts-1", "azure/tts-1"])
def test_completion_cost_tts(model):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    cost = completion_cost(
        model=model,
        prompt="the quick brown fox jumped over the lazy dogs",
        call_type="speech",
    )

    assert cost > 0


def test_completion_cost_anthropic():
    """
    model_name: claude-3-haiku-20240307
    litellm_params:
      model: anthropic/claude-3-haiku-20240307
      max_tokens: 4096
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "claude-3-haiku-20240307",
                "litellm_params": {
                    "model": "anthropic/claude-3-haiku-20240307",
                    "max_tokens": 4096,
                },
            }
        ]
    )
    data = {
        "model": "claude-3-haiku-20240307",
        "prompt_tokens": 21,
        "completion_tokens": 20,
        "response_time_ms": 871.7040000000001,
        "custom_llm_provider": "anthropic",
        "region_name": None,
        "prompt_characters": 0,
        "completion_characters": 0,
        "custom_cost_per_token": None,
        "custom_cost_per_second": None,
        "call_type": "acompletion",
    }

    input_cost, output_cost = cost_per_token(**data)

    assert input_cost > 0
    assert output_cost > 0

    print(input_cost)
    print(output_cost)


def test_completion_cost_deepseek():
    litellm.set_verbose = True
    model_name = "deepseek/deepseek-chat"
    messages_1 = [
        {
            "role": "system",
            "content": "You are a history expert. The user will provide a series of questions, and your answers should be concise and start with `Answer:`",
        },
        {
            "role": "user",
            "content": "In what year did Qin Shi Huang unify the six states?",
        },
        {"role": "assistant", "content": "Answer: 221 BC"},
        {"role": "user", "content": "Who was the founder of the Han Dynasty?"},
        {"role": "assistant", "content": "Answer: Liu Bang"},
        {"role": "user", "content": "Who was the last emperor of the Tang Dynasty?"},
        {"role": "assistant", "content": "Answer: Li Zhu"},
        {
            "role": "user",
            "content": "Who was the founding emperor of the Ming Dynasty?",
        },
        {"role": "assistant", "content": "Answer: Zhu Yuanzhang"},
        {
            "role": "user",
            "content": "Who was the founding emperor of the Qing Dynasty?",
        },
    ]

    message_2 = [
        {
            "role": "system",
            "content": "You are a history expert. The user will provide a series of questions, and your answers should be concise and start with `Answer:`",
        },
        {
            "role": "user",
            "content": "In what year did Qin Shi Huang unify the six states?",
        },
        {"role": "assistant", "content": "Answer: 221 BC"},
        {"role": "user", "content": "Who was the founder of the Han Dynasty?"},
        {"role": "assistant", "content": "Answer: Liu Bang"},
        {"role": "user", "content": "Who was the last emperor of the Tang Dynasty?"},
        {"role": "assistant", "content": "Answer: Li Zhu"},
        {
            "role": "user",
            "content": "Who was the founding emperor of the Ming Dynasty?",
        },
        {"role": "assistant", "content": "Answer: Zhu Yuanzhang"},
        {"role": "user", "content": "When did the Shang Dynasty fall?"},
    ]
    try:
        response_1 = litellm.completion(model=model_name, messages=messages_1)
        response_2 = litellm.completion(model=model_name, messages=message_2)
        # Add any assertions here to check the response
        print(response_2)
        assert response_2.usage.prompt_cache_hit_tokens is not None
        assert response_2.usage.prompt_cache_miss_tokens is not None
        assert (
            response_2.usage.prompt_tokens
            == response_2.usage.prompt_cache_miss_tokens
            + response_2.usage.prompt_cache_hit_tokens
        )
        assert (
            response_2.usage._cache_read_input_tokens
            == response_2.usage.prompt_cache_hit_tokens
        )
    except litellm.APIError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_cost_azure_common_deployment_name():
    from litellm.utils import (
        CallTypes,
        Choices,
        Delta,
        Message,
        ModelResponse,
        StreamingChoices,
        Usage,
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "azure/gpt-4-0314",
                    "max_tokens": 4096,
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "model_info": {"base_model": "azure/gpt-4"},
            }
        ]
    )

    response = ModelResponse(
        id="chatcmpl-876cce24-e520-4cf8-8649-562a9be11c02",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Hi! I'm an AI, so I don't have emotions or feelings like humans do, but I'm functioning properly and ready to help with any questions or topics you'd like to discuss! How can I assist you today?",
                    role="assistant",
                ),
            )
        ],
        created=1717519830,
        model="gpt-4",
        object="chat.completion",
        system_fingerprint="fp_c1a4bcec29",
        usage=Usage(completion_tokens=46, prompt_tokens=17, total_tokens=63),
    )
    response._hidden_params["custom_llm_provider"] = "azure"
    print(response)

    with patch.object(
        litellm.cost_calculator, "completion_cost", new=MagicMock()
    ) as mock_client:
        _ = litellm.response_cost_calculator(
            response_object=response,
            model="gpt-4-0314",
            custom_llm_provider="azure",
            call_type=CallTypes.acompletion.value,
            optional_params={},
            base_model="azure/gpt-4",
        )

        mock_client.assert_called()

        print(f"mock_client.call_args: {mock_client.call_args.kwargs}")
        assert "azure/gpt-4" == mock_client.call_args.kwargs["model"]


def test_completion_cost_anthropic_prompt_caching():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    from litellm.utils import Choices, Message, ModelResponse, Usage

    model = "anthropic/claude-3-5-sonnet-20240620"

    ## WRITE TO CACHE ## (MORE EXPENSIVE)
    response_1 = ModelResponse(
        id="chatcmpl-3f427194-0840-4d08-b571-56bfe38a5424",
        choices=[
            Choices(
                finish_reason="length",
                index=0,
                message=Message(
                    content="Hello! I'm doing well, thank you for",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
        created=1725036547,
        model="claude-3-5-sonnet-20240620",
        object="chat.completion",
        system_fingerprint=None,
        usage=Usage(
            completion_tokens=10,
            prompt_tokens=114,
            total_tokens=124,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
            cache_creation_input_tokens=100,
            cache_read_input_tokens=0,
        ),
    )

    cost_1 = completion_cost(model=model, completion_response=response_1)

    _model_info = litellm.get_model_info(
        model="claude-3-5-sonnet-20240620", custom_llm_provider="anthropic"
    )
    expected_cost = (
        (
            response_1.usage.prompt_tokens
            - response_1.usage.prompt_tokens_details.cached_tokens
        )
        * _model_info["input_cost_per_token"]
        + response_1.usage.prompt_tokens_details.cached_tokens
        * _model_info["cache_read_input_token_cost"]
        + response_1.usage.cache_creation_input_tokens
        * _model_info["cache_creation_input_token_cost"]
        + response_1.usage.completion_tokens * _model_info["output_cost_per_token"]
    )  # Cost of processing (non-cache hit + cache hit) + Cost of cache-writing (cache writing)

    assert round(expected_cost, 5) == round(cost_1, 5)

    print(f"expected_cost: {expected_cost}, cost_1: {cost_1}")

    ## READ FROM CACHE ## (LESS EXPENSIVE)
    response_2 = ModelResponse(
        id="chatcmpl-3f427194-0840-4d08-b571-56bfe38a5424",
        choices=[
            Choices(
                finish_reason="length",
                index=0,
                message=Message(
                    content="Hello! I'm doing well, thank you for",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
        created=1725036547,
        model="claude-3-5-sonnet-20240620",
        object="chat.completion",
        system_fingerprint=None,
        usage=Usage(
            completion_tokens=10,
            prompt_tokens=114,
            total_tokens=134,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=100),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=100,
        ),
    )

    cost_2 = completion_cost(model=model, completion_response=response_2)

    assert cost_1 > cost_2


@pytest.mark.parametrize(
    "model",
    [
        "databricks/databricks-meta-llama-3-1-70b-instruct",
        "databricks/databricks-meta-llama-3-70b-instruct",
        "databricks/databricks-dbrx-instruct",
        "databricks/databricks-mixtral-8x7b-instruct",
    ],
)
def test_completion_cost_databricks(model):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model, messages = model, [{"role": "user", "content": "What is 2+2?"}]

    resp = litellm.completion(model=model, messages=messages)  # works fine

    print(resp)
    cost = completion_cost(completion_response=resp)


@pytest.mark.parametrize(
    "model",
    [
        "databricks/databricks-bge-large-en",
        "databricks/databricks-gte-large-en",
    ],
)
def test_completion_cost_databricks_embedding(model):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    resp = litellm.embedding(model=model, input=["hey, how's it going?"])  # works fine

    print(resp)
    cost = completion_cost(completion_response=resp)


from litellm.llms.fireworks_ai.cost_calculator import get_base_model_for_pricing


@pytest.mark.parametrize(
    "model, base_model",
    [
        ("fireworks_ai/llama-v3p1-405b-instruct", "fireworks-ai-default"),
        ("fireworks_ai/mixtral-8x7b-instruct", "fireworks-ai-moe-up-to-56b"),
    ],
)
def test_get_model_params_fireworks_ai(model, base_model):
    pricing_model = get_base_model_for_pricing(model_name=model)
    assert base_model == pricing_model


@pytest.mark.parametrize(
    "model",
    ["fireworks_ai/llama-v3p1-405b-instruct", "fireworks_ai/mixtral-8x7b-instruct"],
)
def test_completion_cost_fireworks_ai(model):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    resp = litellm.completion(model=model, messages=messages)  # works fine

    print(resp)
    cost = completion_cost(completion_response=resp)


def test_cost_azure_openai_prompt_caching():
    from litellm.utils import Choices, Message, ModelResponse, Usage
    from litellm.types.utils import PromptTokensDetails, CompletionTokensDetails
    from litellm import get_model_info

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model = "azure/o1-mini"

    ## LLM API CALL ## (MORE EXPENSIVE)
    response_1 = ModelResponse(
        id="chatcmpl-3f427194-0840-4d08-b571-56bfe38a5424",
        choices=[
            Choices(
                finish_reason="length",
                index=0,
                message=Message(
                    content="Hello! I'm doing well, thank you for",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
        created=1725036547,
        model=model,
        object="chat.completion",
        system_fingerprint=None,
        usage=Usage(
            completion_tokens=10,
            prompt_tokens=14,
            total_tokens=24,
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=2),
        ),
    )

    ## PROMPT CACHE HIT ## (LESS EXPENSIVE)
    response_2 = ModelResponse(
        id="chatcmpl-3f427194-0840-4d08-b571-56bfe38a5424",
        choices=[
            Choices(
                finish_reason="length",
                index=0,
                message=Message(
                    content="Hello! I'm doing well, thank you for",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
        created=1725036547,
        model=model,
        object="chat.completion",
        system_fingerprint=None,
        usage=Usage(
            completion_tokens=10,
            prompt_tokens=0,
            total_tokens=10,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=14,
            ),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=2),
        ),
    )

    cost_1 = completion_cost(model=model, completion_response=response_1)
    cost_2 = completion_cost(model=model, completion_response=response_2)
    assert cost_1 > cost_2

    model_info = get_model_info(model=model, custom_llm_provider="azure")
    usage = response_2.usage

    _expected_cost2 = (
        (usage.prompt_tokens - usage.prompt_tokens_details.cached_tokens)
        * model_info["input_cost_per_token"]
        + (usage.completion_tokens * model_info["output_cost_per_token"])
        + (
            usage.prompt_tokens_details.cached_tokens
            * model_info["cache_read_input_token_cost"]
        )
    )

    print("_expected_cost2", _expected_cost2)
    print("cost_2", cost_2)

    assert cost_2 == _expected_cost2


def test_completion_cost_vertex_llama3():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    from litellm.utils import Choices, Message, ModelResponse, Usage

    response = ModelResponse(
        id="2024-09-19|14:52:01.823070-07|3.10.13.64|-333502972",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="My name is Litellm Bot, and I'm here to help you with any questions or tasks you may have. As for the weather, I'd be happy to provide you with the current conditions and forecast for your location. However, I'm a large language model, I don't have real-time access to your location, so I'll need you to tell me where you are or provide me with a specific location you're interested in knowing the weather for.\\n\\nOnce you provide me with that information, I can give you the current weather conditions, including temperature, humidity, wind speed, and more, as well as a forecast for the next few days. Just let me know how I can assist you!",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
        created=1726782721,
        model="vertex_ai/meta/llama3-405b-instruct-maas",
        object="chat.completion",
        system_fingerprint="",
        usage=Usage(
            completion_tokens=152,
            prompt_tokens=27,
            total_tokens=179,
            completion_tokens_details=None,
        ),
    )

    model = "vertex_ai/meta/llama3-8b-instruct-maas"
    cost = completion_cost(model=model, completion_response=response)

    assert cost == 0


def test_cost_openai_prompt_caching():
    from litellm.utils import Choices, Message, ModelResponse, Usage
    from litellm import get_model_info

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    model = "gpt-4o-mini-2024-07-18"

    ## LLM API CALL ## (MORE EXPENSIVE)
    response_1 = ModelResponse(
        id="chatcmpl-3f427194-0840-4d08-b571-56bfe38a5424",
        choices=[
            Choices(
                finish_reason="length",
                index=0,
                message=Message(
                    content="Hello! I'm doing well, thank you for",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
        created=1725036547,
        model=model,
        object="chat.completion",
        system_fingerprint=None,
        usage=Usage(
            completion_tokens=10,
            prompt_tokens=14,
            total_tokens=24,
        ),
    )

    ## PROMPT CACHE HIT ## (LESS EXPENSIVE)
    response_2 = ModelResponse(
        id="chatcmpl-3f427194-0840-4d08-b571-56bfe38a5424",
        choices=[
            Choices(
                finish_reason="length",
                index=0,
                message=Message(
                    content="Hello! I'm doing well, thank you for",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
        created=1725036547,
        model=model,
        object="chat.completion",
        system_fingerprint=None,
        usage=Usage(
            completion_tokens=10,
            prompt_tokens=0,
            total_tokens=10,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=14,
            ),
        ),
    )

    cost_1 = completion_cost(model=model, completion_response=response_1)
    cost_2 = completion_cost(model=model, completion_response=response_2)
    assert cost_1 > cost_2

    model_info = get_model_info(model=model, custom_llm_provider="openai")
    usage = response_2.usage

    _expected_cost2 = (
        (usage.prompt_tokens - usage.prompt_tokens_details.cached_tokens)
        * model_info["input_cost_per_token"]
        + usage.completion_tokens * model_info["output_cost_per_token"]
        + usage.prompt_tokens_details.cached_tokens
        * model_info["cache_read_input_token_cost"]
    )

    print("_expected_cost2", _expected_cost2)
    print("cost_2", cost_2)

    assert cost_2 == _expected_cost2


@pytest.mark.parametrize(
    "model",
    [
        "cohere/rerank-english-v3.0",
        "azure_ai/cohere-rerank-v3-english",
    ],
)
def test_completion_cost_azure_ai_rerank(model):
    from litellm import RerankResponse, rerank

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    response = RerankResponse(
        id="b01dbf2e-63c8-4981-9e69-32241da559ed",
        results=[
            {
                "document": {
                    "id": "1",
                    "text": "Paris is the capital of France.",
                },
                "index": 0,
                "relevance_score": 0.990732,
            },
        ],
        meta={},
    )
    print("response", response)
    model = model
    cost = completion_cost(
        model=model, completion_response=response, call_type="arerank"
    )
    assert cost > 0


def test_together_ai_embedding_completion_cost():
    from litellm.utils import Choices, EmbeddingResponse, Message, ModelResponse, Usage

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    response = EmbeddingResponse(
        model="togethercomputer/m2-bert-80M-8k-retrieval",
        data=[
            {
                "embedding": [
                    -0.18039076,
                    0.11614138,
                    0.37174946,
                    0.27238843,
                    -0.21933095,
                    -0.15207036,
                    0.17764972,
                    -0.08700938,
                    -0.23863377,
                    -0.24203257,
                    0.20441775,
                    0.04630023,
                    -0.07832973,
                    -0.193581,
                    0.2009999,
                    -0.30106494,
                    0.21179546,
                    -0.23836501,
                    -0.14919636,
                    -0.045276586,
                    0.08645845,
                    -0.027714893,
                    -0.009854938,
                    0.25298217,
                    -0.1081501,
                    -0.2383125,
                    0.23080236,
                    0.011114239,
                    0.06954927,
                    -0.21081704,
                    0.06937218,
                    -0.16756944,
                    -0.2030545,
                    -0.19809915,
                    -0.031914014,
                    -0.15959585,
                    0.17361341,
                    0.30239972,
                    -0.09923253,
                    0.12680714,
                    -0.13018028,
                    0.1302273,
                    0.19179879,
                    0.17068875,
                    0.065124996,
                    -0.15515316,
                    0.08250379,
                    0.07309733,
                    -0.07283606,
                    0.21411736,
                    0.15457751,
                    -0.08725933,
                    0.07227311,
                    0.056812778,
                    -0.077683985,
                    0.06833304,
                    0.0328722,
                    0.2719641,
                    -0.06989647,
                    0.22805125,
                    0.14953858,
                    0.0792393,
                    0.07793462,
                    0.16176109,
                    -0.15616545,
                    -0.25149494,
                    -0.065352336,
                    -0.38410214,
                    -0.27288514,
                    0.13946335,
                    -0.21873806,
                    0.1365704,
                    0.11738016,
                    -0.1141173,
                    0.022973377,
                    -0.16935326,
                    0.026940947,
                    -0.09990286,
                    -0.05157219,
                    0.21006724,
                    0.15897459,
                    0.011987913,
                    0.02576497,
                    -0.11819022,
                    -0.09184997,
                    -0.31881434,
                    -0.17055357,
                    -0.09523704,
                    0.008458802,
                    -0.015483258,
                    0.038404867,
                    0.014673892,
                    -0.041162584,
                    0.002691519,
                    0.04601874,
                    0.059108324,
                    0.007177156,
                    0.066804245,
                    0.038554087,
                    -0.038720075,
                    -0.2145991,
                    -0.15713418,
                    -0.03712905,
                    -0.066650696,
                    0.04227769,
                    0.018708894,
                    -0.26332214,
                    0.0012769096,
                    -0.13878848,
                    -0.33141217,
                    0.118736655,
                    0.03026654,
                    0.1017467,
                    -0.08000539,
                    0.00092649367,
                    0.13062756,
                    -0.03785864,
                    -0.2038575,
                    0.07655428,
                    -0.24818295,
                    -0.0600955,
                    0.114760056,
                    0.027571939,
                    -0.047068622,
                    -0.19806816,
                    0.0774084,
                    -0.05213658,
                    -0.042000014,
                    0.051924672,
                    -0.14131106,
                    -0.2309609,
                    0.20305444,
                    0.0700591,
                    0.13863273,
                    -0.06145084,
                    -0.039423797,
                    -0.055951696,
                    0.04732105,
                    0.078736484,
                    0.2566198,
                    0.054494765,
                    0.017602794,
                    -0.107575715,
                    -0.017887019,
                    -0.26046592,
                    -0.077659994,
                    -0.08430523,
                    0.18806657,
                    -0.12292346,
                    0.06288608,
                    -0.106739804,
                    -0.06600645,
                    -0.14719339,
                    -0.05070389,
                    0.23234129,
                    -0.034023043,
                    0.056019265,
                    -0.03627352,
                    0.11740493,
                    0.060294818,
                    -0.21726903,
                    -0.09775424,
                    0.27007395,
                    0.28328258,
                    0.022495652,
                    0.13218465,
                    0.07199022,
                    -0.15933248,
                    0.02381037,
                    -0.08288268,
                    0.020621575,
                    0.17395815,
                    0.06978612,
                    0.18418784,
                    -0.12663148,
                    -0.21287888,
                    0.21239495,
                    0.10222956,
                    0.03952703,
                    -0.066957936,
                    -0.035802357,
                    0.03683884,
                    0.22524163,
                    -0.029355489,
                    -0.11534147,
                    -0.041979663,
                    -0.012147716,
                    -0.07279564,
                    0.17417553,
                    0.05546745,
                    -0.1773277,
                    -0.26984993,
                    0.31703642,
                    0.05958132,
                    -0.14933203,
                    -0.084655434,
                    0.074604444,
                    -0.077568695,
                    0.25167143,
                    -0.17753932,
                    -0.006415411,
                    0.068613894,
                    -0.0031754146,
                    -0.0039771493,
                    0.015294107,
                    0.11839045,
                    -0.04570732,
                    0.103238374,
                    -0.09678329,
                    -0.21713412,
                    0.047976546,
                    -0.14346297,
                    0.17429878,
                    -0.31257913,
                    0.15445377,
                    -0.10576352,
                    -0.16792995,
                    -0.17988597,
                    -0.14238739,
                    -0.088244036,
                    0.2760547,
                    0.088823885,
                    -0.08074319,
                    -0.028918687,
                    0.107819095,
                    0.12004892,
                    0.13343112,
                    -0.1332874,
                    -0.0946055,
                    -0.20433402,
                    0.17760132,
                    0.11774745,
                    0.16756779,
                    -0.0937686,
                    0.23887308,
                    0.27315456,
                    0.08657822,
                    0.027402503,
                    -0.06605757,
                    0.29859266,
                    -0.21552202,
                    0.026192812,
                    0.1328459,
                    0.13072926,
                    0.19236198,
                    0.01760772,
                    -0.042355467,
                    0.08815041,
                    -0.013158761,
                    -0.23350924,
                    -0.043668386,
                    -0.15479062,
                    -0.024266671,
                    0.08113482,
                    0.14451654,
                    -0.29152337,
                    -0.028919466,
                    0.15022752,
                    -0.26923147,
                    0.23846954,
                    0.03292609,
                    -0.23572414,
                    -0.14883325,
                    -0.12743121,
                    -0.052229587,
                    -0.14230779,
                    0.284658,
                    0.36885592,
                    -0.13176951,
                    -0.16442224,
                    -0.20283924,
                    0.048434418,
                    -0.16231743,
                    -0.0010730615,
                    0.1408047,
                    0.09481033,
                    0.018139571,
                    -0.030843062,
                    0.13304341,
                    -0.1516288,
                    -0.051779557,
                    0.46940327,
                    -0.07969027,
                    -0.051570967,
                    -0.038892798,
                    0.11187677,
                    0.1703113,
                    -0.39926252,
                    0.06859773,
                    0.08364686,
                    0.14696898,
                    0.026642298,
                    0.13225247,
                    0.05730332,
                    0.35534015,
                    0.11189959,
                    0.039673142,
                    -0.056019083,
                    0.15707816,
                    -0.11053284,
                    0.12823457,
                    0.20075114,
                    0.040237684,
                    -0.19367051,
                    0.13039409,
                    -0.26038498,
                    -0.05770229,
                    -0.009781617,
                    0.15812513,
                    -0.10420735,
                    -0.020158196,
                    0.13160926,
                    -0.20823349,
                    -0.045596864,
                    -0.2074525,
                    0.1546387,
                    0.30158705,
                    0.13175933,
                    0.11967154,
                    -0.09094463,
                    0.0019428955,
                    -0.06745872,
                    0.02998099,
                    -0.18385777,
                    0.014330351,
                    0.07141392,
                    -0.17461702,
                    0.099743806,
                    -0.016181415,
                    0.1661396,
                    0.070834026,
                    0.110713825,
                    0.14590909,
                    0.15404254,
                    -0.21658006,
                    0.00715122,
                    -0.10229453,
                    -0.09980027,
                    -0.09406554,
                    -0.014849227,
                    -0.26285952,
                    0.069972225,
                    0.05732395,
                    -0.10685719,
                    0.037572138,
                    -0.18863359,
                    -0.00083297276,
                    -0.16088934,
                    -0.117982,
                    -0.16381365,
                    -0.008932539,
                    -0.06549256,
                    -0.08928683,
                    0.29934987,
                    0.16532114,
                    -0.27117223,
                    -0.12302226,
                    -0.28685933,
                    -0.14041144,
                    -0.0062569617,
                    -0.20768198,
                    -0.15385273,
                    0.20506454,
                    -0.21685128,
                    0.1081962,
                    -0.13133131,
                    0.18937315,
                    0.14751591,
                    0.2786974,
                    -0.060183275,
                    0.10365405,
                    0.109799005,
                    -0.044105034,
                    -0.04260162,
                    0.025758557,
                    0.07590695,
                    0.0726137,
                    -0.09882405,
                    0.26437432,
                    0.15884234,
                    0.115702584,
                    0.0015900572,
                    0.11673009,
                    -0.18648374,
                    0.3080215,
                    -0.26407364,
                    -0.15610488,
                    0.12658228,
                    -0.05672454,
                    0.016239772,
                    -0.092462406,
                    -0.36205122,
                    -0.2925843,
                    -0.104364775,
                    -0.2598659,
                    -0.14073578,
                    0.10225995,
                    -0.2612335,
                    -0.17479639,
                    0.17488293,
                    -0.2437756,
                    0.114384405,
                    -0.13196659,
                    -0.067482576,
                    0.024756929,
                    0.11779123,
                    0.2751749,
                    -0.13306957,
                    -0.034118645,
                    -0.14177705,
                    0.27164033,
                    0.06266008,
                    0.11199439,
                    -0.09814594,
                    0.13231735,
                    0.019105865,
                    -0.2652429,
                    -0.12924416,
                    0.0840029,
                    0.098754935,
                    0.025883028,
                    -0.33059177,
                    -0.10544467,
                    -0.14131607,
                    -0.09680401,
                    -0.047318626,
                    -0.08157771,
                    -0.11271855,
                    0.12637804,
                    0.11703408,
                    0.014556337,
                    0.22788583,
                    -0.05599293,
                    0.25811172,
                    0.22956331,
                    0.13004553,
                    0.15419081,
                    -0.07971162,
                    0.11692607,
                    -0.2859737,
                    0.059627946,
                    -0.02716421,
                    0.117603,
                    -0.061154094,
                    -0.13555732,
                    0.17092334,
                    -0.16639015,
                    0.2919375,
                    -0.020189757,
                    0.18548165,
                    -0.32514027,
                    0.19324942,
                    -0.117969565,
                    0.23577307,
                    -0.18052326,
                    -0.10520473,
                    -0.2647645,
                    -0.29393113,
                    0.052641366,
                    -0.07733946,
                    -0.10684275,
                    -0.15046178,
                    0.065737076,
                    -0.0022297644,
                    -0.010802031,
                    -0.115943395,
                    -0.11602136,
                    0.24265991,
                    -0.12240144,
                    0.11817584,
                    0.026270682,
                    -0.25762397,
                    -0.14545679,
                    0.014168602,
                    0.106698096,
                    0.12905516,
                    -0.12560321,
                    0.15034604,
                    0.071529925,
                    0.123048246,
                    -0.058863316,
                    -0.12251829,
                    0.20463347,
                    0.06841168,
                    0.13706751,
                    0.05893755,
                    -0.12269708,
                    0.096701816,
                    -0.3237337,
                    -0.2213742,
                    -0.073655166,
                    -0.12979327,
                    0.14173084,
                    0.19167605,
                    -0.14523135,
                    0.06963011,
                    -0.019228822,
                    -0.14134938,
                    0.22017507,
                    0.007933044,
                    -0.0065696104,
                    0.074060634,
                    -0.13231485,
                    0.1387053,
                    -0.14480218,
                    -0.007837481,
                    0.29880494,
                    0.101618655,
                    0.14514285,
                    -0.066113696,
                    -0.041709363,
                    0.21512671,
                    -0.090142876,
                    -0.010337287,
                    0.13212202,
                    0.08307805,
                    0.10144794,
                    -0.024808172,
                    0.21877879,
                    -0.071282186,
                    -8.786433e-05,
                    -0.014574037,
                    -0.11954953,
                    -0.096931055,
                    -0.2557228,
                    0.1090451,
                    0.15424186,
                    -0.029206438,
                    -0.2898023,
                    0.22510754,
                    -0.019507697,
                    0.1566895,
                    -0.24820097,
                    -0.012163554,
                    0.12401036,
                    0.024711533,
                    0.24737844,
                    -0.06311193,
                    0.0652544,
                    -0.067403205,
                    0.15362221,
                    -0.12093675,
                    0.096014425,
                    0.17337392,
                    -0.017509578,
                    0.015355054,
                    0.055885684,
                    -0.08358914,
                    -0.018012024,
                    0.069017515,
                    0.32854614,
                    0.0063175815,
                    -0.09058244,
                    0.000681382,
                    -0.10825181,
                    0.13190223,
                    0.009358909,
                    -0.12205342,
                    0.08268384,
                    -0.260608,
                    -0.11042252,
                    -0.022601532,
                    -0.080661446,
                    -0.035559367,
                    0.14736788,
                    0.061933476,
                    -0.07815901,
                    0.110823035,
                    -0.00875032,
                    -0.064237975,
                    -0.04546554,
                    -0.05909862,
                    0.23463917,
                    -0.20451859,
                    -0.16576467,
                    0.10957323,
                    -0.08632836,
                    -0.27395645,
                    0.0002913844,
                    0.13701706,
                    -0.058854006,
                    0.30768716,
                    -0.037643027,
                    -0.1365738,
                    0.095908396,
                    -0.05029932,
                    0.14793666,
                    0.30881998,
                    -0.018806668,
                    -0.15902956,
                    0.07953607,
                    -0.07259314,
                    0.17318867,
                    0.123503335,
                    -0.11327983,
                    -0.24497227,
                    -0.092871994,
                    0.31053993,
                    0.09460377,
                    -0.21152224,
                    -0.03127119,
                    -0.018713845,
                    -0.014523326,
                    -0.18656968,
                    0.2255386,
                    -0.1902719,
                    0.18821372,
                    -0.16890709,
                    -0.04607359,
                    0.13054903,
                    -0.05379203,
                    -0.051014878,
                    0.054293603,
                    -0.07299424,
                    -0.06728367,
                    -0.052388195,
                    -0.29960096,
                    -0.22351485,
                    -0.06481434,
                    -0.1619141,
                    0.24709718,
                    -0.1203425,
                    0.029514981,
                    -0.01951599,
                    -0.072677284,
                    -0.25097945,
                    0.03758907,
                    0.14380245,
                    -0.037721623,
                    -0.19958745,
                    0.2408246,
                    -0.13995907,
                    -0.028115002,
                    -0.14780775,
                    0.17445801,
                    0.11311988,
                    0.05306163,
                    0.0018454103,
                    0.00088805315,
                    -0.27949628,
                    -0.23556526,
                    -0.18175222,
                    -0.28372183,
                    -0.43095905,
                    0.22644317,
                    0.06072053,
                    0.02278773,
                    0.021752749,
                    0.053462002,
                    -0.30636713,
                    0.15607472,
                    -0.16657323,
                    -0.07240017,
                    0.1410017,
                    -0.026987495,
                    0.15029654,
                    0.03340291,
                    -0.2056912,
                    0.055395555,
                    0.11999902,
                    0.06368412,
                    -0.025476053,
                    -0.1702383,
                    -0.23432998,
                    0.14855467,
                    -0.07505147,
                    -0.030296376,
                    -0.07001051,
                    0.10510949,
                    0.10420236,
                    0.09809715,
                    0.17195594,
                    0.19430229,
                    -0.16121922,
                    -0.081139356,
                    0.15032287,
                    0.10385191,
                    -0.18741366,
                    0.008690719,
                    -0.12941097,
                    -0.027797364,
                    -0.2148853,
                    0.037788823,
                    0.16691138,
                    0.099181786,
                    -0.0955518,
                    -0.0074798446,
                    -0.17511943,
                    0.14543307,
                    -0.029364567,
                    -0.21223477,
                    -0.05881982,
                    0.11064195,
                    -0.2877007,
                    -0.023934823,
                    -0.15569815,
                    0.015789302,
                    -0.035767324,
                    -0.15110208,
                    0.07125638,
                    0.05703369,
                    -0.08454703,
                    -0.07080854,
                    0.025179204,
                    -0.10522502,
                    -0.03670824,
                    -0.11075579,
                    0.0681693,
                    -0.28287485,
                    0.2769406,
                    0.026260372,
                    0.07289979,
                    0.04669447,
                    -0.16541554,
                    0.040775143,
                    0.035916835,
                    0.03648039,
                    0.11299418,
                    0.14765884,
                    0.031163761,
                    0.0011800596,
                    -0.10715472,
                    0.02665826,
                    -0.06237457,
                    0.15672882,
                    0.09038829,
                    0.0061029866,
                    -0.2592228,
                    -0.21008603,
                    0.019810716,
                    -0.08721265,
                    0.107840165,
                    0.28438854,
                    -0.16649202,
                    0.19627784,
                    0.040611178,
                    0.16516201,
                    0.24990341,
                    -0.16222852,
                    -0.009037945,
                    0.053751092,
                    0.1647804,
                    -0.16184275,
                    -0.29710436,
                    0.043035872,
                    0.04667557,
                    0.14761224,
                    -0.09030331,
                    -0.024515491,
                    0.10857025,
                    0.19865094,
                    -0.07794062,
                    0.17942934,
                    0.13322048,
                    -0.16857187,
                    0.055713065,
                    0.18661156,
                    -0.07864222,
                    0.23296827,
                    0.10348465,
                    -0.11750994,
                    -0.065938555,
                    -0.04377608,
                    0.14903909,
                    0.019000417,
                    0.21033548,
                    0.12162547,
                    0.1273347,
                ],
                "index": 0,
                "object": "embedding",
            }
        ],
        object="list",
        usage=Usage(
            completion_tokens=0,
            prompt_tokens=0,
            total_tokens=0,
            completion_tokens_details=None,
        ),
    )

    cost = completion_cost(
        completion_response=response,
        custom_llm_provider="together_ai",
        call_type="embedding",
    )


def test_completion_cost_params():
    """
    Relevant Issue: https://github.com/BerriAI/litellm/issues/6133
    """
    litellm.set_verbose = True
    resp1_prompt_cost, resp1_completion_cost = cost_per_token(
        model="gemini-1.5-pro-002",
        prompt_tokens=1000,
        completion_tokens=1000,
        custom_llm_provider="vertex_ai_beta",
    )

    resp2_prompt_cost, resp2_completion_cost = cost_per_token(
        model="gemini-1.5-pro-002", prompt_tokens=1000, completion_tokens=1000
    )

    assert resp2_prompt_cost > 0

    assert resp1_prompt_cost == resp2_prompt_cost
    assert resp1_completion_cost == resp2_completion_cost

    resp3_prompt_cost, resp3_completion_cost = cost_per_token(
        model="vertex_ai/gemini-1.5-pro-002", prompt_tokens=1000, completion_tokens=1000
    )

    assert resp3_prompt_cost > 0

    assert resp3_prompt_cost == resp1_prompt_cost
    assert resp3_completion_cost == resp1_completion_cost


def test_completion_cost_params_2():
    """
    Relevant Issue: https://github.com/BerriAI/litellm/issues/6133
    """
    litellm.set_verbose = True

    prompt_characters = 1000
    completion_characters = 1000
    resp1_prompt_cost, resp1_completion_cost = cost_per_token(
        model="gemini-1.5-pro-002",
        prompt_characters=prompt_characters,
        completion_characters=completion_characters,
        prompt_tokens=1000,
        completion_tokens=1000,
    )

    print(resp1_prompt_cost, resp1_completion_cost)

    model_info = litellm.get_model_info("gemini-1.5-pro-002")
    input_cost_per_character = model_info["input_cost_per_character"]
    output_cost_per_character = model_info["output_cost_per_character"]

    assert resp1_prompt_cost == input_cost_per_character * prompt_characters
    assert resp1_completion_cost == output_cost_per_character * completion_characters


def test_completion_cost_params_gemini_3():
    from litellm.utils import Choices, Message, ModelResponse, Usage

    from litellm.litellm_core_utils.llm_cost_calc.google import cost_per_character

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    response = ModelResponse(
        id="chatcmpl-61043504-4439-48be-9996-e29bdee24dc3",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Sí. \n",
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
        created=1728529259,
        model="gemini-1.5-flash",
        object="chat.completion",
        system_fingerprint=None,
        usage=Usage(
            completion_tokens=2,
            prompt_tokens=3771,
            total_tokens=3773,
            completion_tokens_details=None,
            prompt_tokens_details=None,
        ),
        vertex_ai_grounding_metadata=[],
        vertex_ai_safety_results=[
            [
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "probability": "NEGLIGIBLE",
                },
                {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"},
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "probability": "NEGLIGIBLE",
                },
            ]
        ],
        vertex_ai_citation_metadata=[],
    )

    pc, cc = cost_per_character(
        **{
            "model": "gemini-1.5-flash",
            "custom_llm_provider": "vertex_ai",
            "prompt_tokens": 3771,
            "completion_tokens": 2,
            "prompt_characters": None,
            "completion_characters": 3,
        }
    )

    model_info = litellm.get_model_info("gemini-1.5-flash")

    assert round(pc, 10) == round(3771 * model_info["input_cost_per_token"], 10)
    assert round(cc, 10) == round(
        3 * model_info["output_cost_per_character"],
        10,
    )
