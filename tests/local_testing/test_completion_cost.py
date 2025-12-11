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
import base64
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
from litellm.llms.custom_httpx.http_handler import HTTPHandler
import json
import httpx
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


def test_get_gemini_tokens():
    # # 🦄🦄🦄🦄🦄🦄🦄🦄
    max_tokens = get_max_tokens("gemini/gemini-1.5-flash")
    assert max_tokens == 8192
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

        litellm.set_verbose = True

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
        print(f"Error: {e}")
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
            model="azure/gpt-35-turbo",  # azure always has model written like this
            usage=Usage(prompt_tokens=21, completion_tokens=17, total_tokens=38),
        )

        cost = litellm.completion_cost(
            completion_response=resp, model="azure/chatgpt-deployment-2"
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
                model="azure/text-embedding-ada-002",
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


def test_cost_bedrock_pricing_actual_calls():
    litellm.set_verbose = True
    model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    response = litellm.completion(
        model=model, messages=messages, mock_response="hello cool one"
    )

    print("response", response)
    cost = litellm.completion_cost(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        completion_response=response,
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
    )
    assert cost > 0


def test_whisper_openai():
    litellm.set_verbose = True
    transcription = TranscriptionResponse(
        text="Four score and seven years ago, our fathers brought forth on this continent a new nation, conceived in liberty and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure."
    )

    setattr(transcription, "duration", 3)
    transcription._hidden_params = {
        "model": "whisper-1",
        "custom_llm_provider": "openai",
        "optional_params": {},
        "model_id": None,
    }
    _total_time_in_seconds = 3

    cost = litellm.completion_cost(model="whisper-1", completion_response=transcription)

    print(f"cost: {cost}")
    print(f"whisper dict: {litellm.model_cost['whisper-1']}")
    expected_cost = round(
        litellm.model_cost["whisper-1"]["output_cost_per_second"]
        * _total_time_in_seconds,
        5,
    )
    assert round(cost, 5) == round(expected_cost, 5)


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
    setattr(transcription, "duration", _total_time_in_seconds)

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
    assert round(cost, 5) == round(expected_cost, 5)


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
        model="groq/llama-3.3-70b-versatile",
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
        model="claude-3-sonnet",
        object="chat.completion",
        system_fingerprint=None,
        usage=Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )
    cost = litellm.completion_cost(
        model="vertex_ai/claude-3-sonnet",
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


@pytest.mark.parametrize(
    "model", ["openai/tts-1", "azure/tts-1", "openai/gpt-4o-mini-tts"]
)
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
        assert "azure/gpt-4" == mock_client.call_args.kwargs["base_model"]


@pytest.mark.parametrize(
    "model, custom_llm_provider",
    [
        ("claude-3-5-sonnet-20240620", "anthropic"),
        ("gemini/gemini-1.5-flash-001", "gemini"),
    ],
)
def test_completion_cost_prompt_caching(model, custom_llm_provider):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    from litellm.utils import Choices, Message, ModelResponse, Usage

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
        model=model,
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
        model=model, custom_llm_provider=custom_llm_provider
    )
    expected_cost = (
        (
            response_1.usage.prompt_tokens
            - response_1.usage.prompt_tokens_details.cached_tokens
            - response_1.usage.prompt_tokens_details.cache_creation_tokens
        )
        * _model_info["input_cost_per_token"]
        + (response_1.usage.prompt_tokens_details.cached_tokens or 0)
        * _model_info["cache_read_input_token_cost"]
        + (response_1.usage.cache_creation_input_tokens or 0)
        * _model_info["cache_creation_input_token_cost"]
        + (response_1.usage.completion_tokens or 0)
        * _model_info["output_cost_per_token"]
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
        model=model,
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


@pytest.mark.flaky(retries=6, delay=2)
@pytest.mark.parametrize(
    "model",
    [
        "databricks/databricks-meta-llama-3.2-3b-instruct",
        "databricks/databricks-meta-llama-3-70b-instruct",
        "databricks/databricks-dbrx-instruct",
        # "databricks/databricks-mixtral-8x7b-instruct",
    ],
)
@pytest.mark.skip(reason="databricks is having an active outage")
def test_completion_cost_databricks(model):
    litellm._turn_on_debug()
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    model, messages = model, [{"role": "user", "content": "What is 2+2?"}]

    resp = litellm.completion(model=model, messages=messages)  # works fine

    print(resp)
    print(f"hidden_params: {resp._hidden_params}")
    assert resp._hidden_params["response_cost"] > 0


@pytest.mark.parametrize(
    "model",
    [
        "databricks/databricks-bge-large-en",
        "databricks/databricks-gte-large-en",
    ],
)
def test_completion_cost_databricks_embedding(model, monkeypatch):
    """
    Test completion cost calculation for Databricks embedding models using mocked HTTP responses.
    """
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    mock_response_data = {
        "object": "list",
        "model": model.split("/")[1],
        "data": [
            {
                "index": 0,
                "object": "embedding",
                "embedding": [
                    0.06768798828125,
                    -0.01291656494140625,
                    -0.0501708984375,
                    0.0245361328125,
                    -0.030364990234375,
                ],
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8,
            "completion_tokens": 0,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    }

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data

    sync_handler = HTTPHandler()

    with patch.object(HTTPHandler, "post", return_value=mock_response):
        resp = litellm.embedding(
            model=model, input=["hey, how's it going?"], client=sync_handler
        )

        print(resp)
        cost = completion_cost(completion_response=resp)


from litellm.llms.fireworks_ai.cost_calculator import get_base_model_for_pricing


@pytest.mark.parametrize(
    "model, base_model",
    [
        ("fireworks_ai/llama-v3p3-70b-instruct", "fireworks-ai-above-16b"),
    ],
)
def test_get_model_params_fireworks_ai(model, base_model):
    pricing_model = get_base_model_for_pricing(model_name=model)
    assert base_model == pricing_model


@pytest.mark.parametrize(
    "model",
    [
        "fireworks_ai/llama-v3p3-70b-instruct",
    ],
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
    from litellm.types.utils import (
        PromptTokensDetailsWrapper,
        CompletionTokensDetailsWrapper,
    )
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
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=2
            ),
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
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=14,
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                reasoning_tokens=2
            ),
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

    assert (
        abs(cost_2 - _expected_cost2) < 1e-5
    )  # Allow for small floating-point differences


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
            prompt_tokens=14,
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
        meta={
            "billed_units": {
                "search_units": 1,
            }
        },
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

    from litellm.llms.vertex_ai.cost_calculator import cost_per_character

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    usage = Usage(
        completion_tokens=2,
        prompt_tokens=3771,
        total_tokens=3773,
        completion_tokens_details=None,
        prompt_tokens_details=None,
    )

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
        usage=usage,
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
            "prompt_characters": None,
            "completion_characters": 3,
            "usage": usage,
        }
    )

    model_info = litellm.get_model_info("gemini-1.5-flash")

    assert round(pc, 10) == round(3771 * model_info["input_cost_per_token"], 10)
    assert round(cc, 10) == round(
        3 * model_info["output_cost_per_character"],
        10,
    )


@pytest.mark.asyncio
# @pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.parametrize("stream", [False])  # True,
async def test_test_completion_cost_gpt4o_audio_output_from_model(stream):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    from litellm.types.utils import (
        Choices,
        Message,
        ModelResponse,
        Usage,
        ChatCompletionAudioResponse,
        PromptTokensDetails,
        CompletionTokensDetailsWrapper,
        PromptTokensDetailsWrapper,
    )

    usage_object = Usage(
        completion_tokens=34,
        prompt_tokens=16,
        total_tokens=50,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=28, reasoning_tokens=0, text_tokens=6
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=0, cached_tokens=0, text_tokens=16, image_tokens=0
        ),
    )
    completion = ModelResponse(
        id="chatcmpl-AJnhcglpTV5u84s1cTxWFeIkGKAo7",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content=None,
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                    audio=ChatCompletionAudioResponse(
                        id="audio_6712c25ce73c819080b41362648bc6cb",
                        data="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MQApACMAIgAwACMAIAArACYAGgAUACAADgAIABEAHwAXAAkAHQAWAA8ABQATABAABwAKAAYACAD8/wUAAAADAAYACwAMAAoAEwAXAB0AEQASABMAFgAKAAsAGAAUABUACwAPAA4AEAASABMAFQAUABoAGAASAA8AFAANAAYABgAGAAQABAADAAYAAgAKAAsACwARABQAGAARABEAEAANAAgABAADAAQAAwAFAA4AEQATABUAEQAQAA4ADwAMAAkACwAIAAMA///8//z/AAAEAAMAAwACAAkABgAIAAsABgAFAAkABgAHAAQABgACAAUAAgABAAUAAgAHAAYACQAGAAkABQACAAAABAABAP//BAAAAAAA//8DAAAAAgAGAAgABwABAAQABQAEAAEA/v8BAP3//f/9//z//f/8//z/+f/5//r//v/9//z/AQD///////8CAAEA/v/9/wAA/v/8//7//f/7//z/+f/6//r/+//4//n/+f/8//v/+P/5//r/+//8//v//f/7//z//P/6//z//P/7//r/+v/7//7/AQAAAP//AAD//////v8BAP3///////3//P/6//3////+//n//P/4//n/9f/3//j/9v/1//P/9f/1//T/+P/2//f/9f/3//v/+//8//r/+//7//3//P/6//v//P/6//r//v8AAP///v/8//3//v/9//z//P/6//r//v////z/+//+/wAA//8AAP3/AQD///7///8DAP7/BgABAAEACQAHAAMABgAFAAUABwAHAAcAAQABAAUABAAEAAYABQAHAAQABAAEAAMABQAGAAYA//8AAAEAAwAAAAEABQAFAAEAAQACAAMABgAIAAYABAABAAMABQAGAAQAAwACAAMABgAHAAMACAAHAAQAAwABAAIABAAEAAQAAwAHAAEA//8AAAIABQAGAAAAAwD9////AgAEAAAAAgAFAAAAAgABAAEAAgAEAAUAAQADAAQABQADAAUAAwAHAAQABwAIAAQABAADAAMABwAFAAYAAwAFAAcABgAJAAYACAAHAAsABQAHAAkACQAKAAYACAAHAAkACgAOAAwADAAMAAsAEAAMAA0ADAAOAA0ADwAOAA4ADQAQABIADQATABEAFgARABMAEQAUAA4AFQATABIAFgAVABQADwAQABEAEwASABIADgAPABEAEgASABEAEgARABAAEAARAA0AEgAUABMAEQATABEAFQASABQAFQAVAA4AEQARABEAFAAVABAAEQARAA8AEAAPABIAFAAWABIAFAAQABUAEwAXABgAEQAUABQAFAASABQAFAAUABEAEAARABAAFAATABMAEwARABAADwAQABEAEQASABIAEAAPABAAEQAUABQADwASAA8AEAANABAAEAASAAwAEgARAA8AEAAPAA8ADgANAAwADAANABAADgANAAwADQAOAAsACwAPAA8ADQAMABAADAAQAA8ABgAKABAADQAOAAwADQAHAAwACwAMAA0ACQAKAA0ACQAGAAcABwAJAAcACQAKAAsACAAKAAcACwAKAAgAAgAFAAoACQALAAsADQAJAAwABgAGAAcACQAKAAoABQAEAAMABAAGAAUACAAFAAgADAAKAAIAAwABAAIAAQAEAAMAAwABAPz/AwAFAAEAAwACAAMABgAGAAUAAQABAAQAAgAFAAcABAADAAMABgADAAMAAQAGAAQAAAACAAIABgAGAAUAAQACAAEABAAAAAAABgAGAP7/AAAAAP7/AwADAAEAAAADAAYAAQAFAAIABAABAAIAAwADAAEAAgAFAAQABAAEAAQABQABAAIABQAFAAMAAwAFAAQAAQAAAAMAAAAIAAUA/////wYAAAAEAAQAAgD//wEAAgAAAP7/BQAEAP//AQD+/wEAAAAAAP//AQADAAEA/v/+/wAA/v8AAAIA/v/7//7/AAAAAP/////9//z/AAABAP///v/6//7/+//8//r/+P/4//n//P/9//3//v/9//z//f/7//7//P8BAPn////4//n//f/9//X//P/7//j/+//8//f/+P/6//n//P/6//v/+f/5//v/+f/3//f/+v/7//r//P/4//f/9v/2//j/+/////f/+f/2//r/+v/6//n/+v/6//b/+f/3//j/+f/3//j/9//z//T/9P/6//b/9//0//n/9P/0//r/9v/9//f/+P/x//b//f/6//X//f/7//b/+//2//b/+P/6//r/+//8//f/+P/1//n/+P/6//f/+v/4//r/+f/7//X//f/4//X/+v/1//X/9//7//n/+f/4//z/+f/9//z/9v/4//z/+P/8//3/+f/1//j/9//6//n/+v/7//z//v/7//n/+P/0//f//P/9//n/9v/1//n/+f8AAP//9//4//3/+v/8/wAA/f/7//r/+//9//n//f/1//n/+P/5//j/+v/8//r/+P/2//r/+//6//j/+P/5//r/+v/+//j/+f/4//n//v/6//j/+//8//b/+P/3//r//P/+/wAA+P/8//n/+//5//r////7//f/+v////3//v/6//3//v/8//z/+v/9//3//f/7//z//f/9//n/+//1//j//f/9//j//P/7//n/+P/6//f/+f/6//v/+v/7//n/+v/7//7//v////v/AAD8//z//P/5//b//P/6//f/+//6//n/+v/7//7/+//+//f////4//n/+v/2//r//f8GAPv/+P/y/wEA+f/9/wAA+f/7////+f/8//3//v/7//z/+//6//z/9//8//r//v/5//v/+f/1//X/+P/+//n/+P/0//b/+v/6//v/+P/4//n//P/7//v/+f/4//7//f////7/+f/5//v/+//7//r/+P/5//r/+f/5//v////7//v/+v/3//j/+f/3//v/9//7//n/+P/5//3//f/5//T/+v/6//X/+f/6/wEA+/8CAP3/AQD8//7//v/8//v/AQADAPr/+f/1//v/+/8GAAQA/P/7//7/AAD8/wAA/P8BAAAA/v/+//3/AAD6//7/+v8BAP3/+//9/wAA/P/8//7/9v/4//3//v8DAP3//P/3//7/+v/+//7//v///wAA/v/+//7//P8AAP//+v/4//v/AAD///n/AAD///v/AAD8//7/+v8AAP7/+/8AAP///v/6//r/BAD9//r/+P/6//z//f8EAAAA/P/1////AAAGAP3/AAD///3//v8AAPn////9///////5//z//v8AAP7/AgABAAMA+v/+//z//P/5//7//P///wEA+//+//v/AAD7//v///////n/+//6//j/AQAAAPn/+P/3//n///8CAPj//P/5//z/+f/9//b/+v/3//v//f8CAPn//f/4//X/AAD5//f/9v////z/9f/6//7////3//r////8//j/9v////z/+v/8/wEAAAD7//3/9//7//b/AAD9//r/+f/2//r//P//////+//9//7/+//8//r/+f/5//z/+//8//7/+f/6//3/+//8////AQAAAPz//P/5//v/+/8EAPn/+//8/////v/9////AgACAP7/AAD/////AAAFAP///v/6/wEAAAACAP7/+f/+/wYA//8BAAYABgAFAAQA//8CAAAABQAFAP//AwACAAMAAwADAAIAAwABAAEAAQAAAAAA/v8CAP7/+//8/wEA//8AAAAAAAAAAAIA//8CAAMAAAD9/wIA/P/+////AgAGAAEA/f/6//3///8DAP7/AwD9//3/AwAEAPv//f/5//n/AQACAP7//f8BAP//AwAEAAAAAgD8/wIA/f8EAP//AQAAAAAABgD5//7/+/8IAAAA+//8/wgAAQADAAQA/v8CAAMAAQABAAQABwAEAAIABAADAAEA//8CAAIAAAABAAIAAgADAP//AwAAAAMABQAIAAIAAwAFAAAAAwACAAQABAAGAAQA/f8BAAQA//8DAAMA/v/7/wIA/P8BAP//AwD//wEA/f8AAAYAAwAHAAQABAD+//7/AAD//wIAAwAKAAYABgAEAAcAAwACAAQAAgAGAAgAAwD+//7/CgAFAAQAAQAFAAcAAgAGAAQABQAAAAQAAAAGAP3/BQAEAAQACgAHAAIAAgADAAEAAwADAAIAAQACAAMAAwAKAAYABgACAAcABwAMAAAABwAAAAIAAwAGAAAAAgAAAP3/AgABAP7//v8CAAEABQADAAUAAwABAAAA//8BAAUAAwACAAUACQAIAAQA/v8AAAQABQAHAAMABwACAAQABQADAAEABAAHAAMABAADAAIAAQABAAIA//8CAAIABAAAAAYABAALAPz/AwD//wIABgADAP//BQAIAAUA/P8AAAEABAAFAAEAAQD5/wAA//8DAAEA/v/9//////8AAAIAAgABAAUAAgADAAUAAgAEAAIAAQD+//z/AwABAP///P8CAAIA/v/9/wEAAAD9/wAAAgADAP7/AgD//wAAAAD///3//f8EAAEA///7//3//////wIA/v8DAP//AwD9////AwAEAPr//f/8//r/AwADAAAA/v/7//7///8DAP3/AQD7//r/AQD///j/AAABAAEA/P////3/AAD+//z////8////+/8FAAMA/P/7/wAAAQD+/wAA/f/9//3/+v/6//z//v/8//7//P/+//7//v///wAA+//6//v/9//5//r//v/9//v//f/8//v//v8BAAIA/v8DAAAA///6//z////9//r//P8CAP///v/8//3//P////7/+f/4//3//v8AAPz/+P/9//r//f/8//z//f/z//j/9//9//r/+v/7/wEA+P/7//z/+P/6//7//f/7//z/+v/8//v//P/+//3///8AAAAA/v/7//v/+v/5//z/+v8AAPv/+//2//7//P8AAP3/9f/6///////9//3/AAD+//3/+v/8//n/+v/8//3/+//6//3/+P/+//f//f/6//v/AQD9//v/9////wAA//8AAP7////5//r//P/+//n//f////z//f/5//3/+/8AAAIAAQAEAAEA+v/3//3//v/+//z/AAABAPr//f/7//7///8CAAIA/v///wAA/v///////P/7//3/+/8BAAIAAQD9/wAA+//+//z//f/+/wAA/f/7//3///8EAP3/AAD8//7/BwAIAPv/AQD7//3//v8CAPz//////wEAAwACAP7///8AAPz/AQD+/wEA/P8CAAQA//8CAAIABAACAAAAAwABAAAAAgD/////AAACAP7/AgABAAQABQAFAAEAAAAAAP7//f8CAAAAAgADAAMAAQD8/wAAAAAFAAEA/////wIA/v8AAAMAAwABAAcABQACAAIAAQAGAAAA/v8AAAUACAAEAAMABgAEAAUACAAGAAMAAwAEAAUAAgAFAAMABwADAAEA//8BAPz/AQABAAEABQD+/wEA/v8GAAIA/f8AAAYAAAD//wAA///9/wEAAQADAAEA///+/wUAAwAGAAMA///9/wQAAQAJAAcAAAD//wcABAAIAAcAAgAAAAYABAAFAAQABgAGAAMAAQD//wQAAwAJAAMABAD9//7/AgAFAP7/AgAKAAYACAADAAQABAAFAAYAAAAGAAgABQAEAAcACgAHAAMABgAGAAUACQAKAAUABQADAAIA/f8AAAMACQALAAgAAwD//wMABAAHAAUAAwADAAYAAAAGAAcAAwD//wcAAwAFAAUAAQAFAAUABAD+/wEABQAAAAAAAAAEAAEAAAD+/wMA/v8CAAQA/v8AAAMAAwACAAIABwAEAAAA//////7/AgADAAAAAgD///////8CAP7//P/9/wIAAQACAAAAAwACAAEAAAD///7///8EAP7/AAD4//z//f8AAAAA/v8CAP///P/7////AgACAP////8AAPz//P/+//z//P/8//3//P/8//3//v/4//j//P/7//3/+////////////wAA/P/9/wEA/v////v//f/5//n/AQAAAPn/+//4//r//v////n/+//6//3//v8AAPv////7//j//f/7//n/+//9//v/+v/4//z//P/6//z/+P/5//n/+f/7//r////3//v/+//8//r/9f/3//v/+P/7///////+/wEAAQD9//7//P/8//f/9v/3//f/+f/5//X/9//2//j/+f/6//n//P/8//f/+v/3//r//f/9//z/+v/7//n/+f/6//v/+f/7//j//P/9/wAA/P////r//v/9//z/BQD8//n/+/////7/+/8CAAAA///5//7/+//9//n//P/8//z/AQD///n/+v/5//z/+/8AAPz//f/7//b/+//4//b//P8AAAEA/v8CAP3/+//5//3/+//8//7//v8DAPv/AwD3//3//P8BAP7/9v/4/wUA/v8BAAIA/v/+///////9/wEA/f/+//3//f/8//7/AgADAP7/AAABAP//AQABAP3/AwD//wAA/P/8/////f8DAAAA/v/+/wEA/v/7//7//P/+//7/AAACAAMAAgABAAMAAQADAAMA//8BAP//AQADAAAABQD+/wEAAAAEAAMA//8BAAIA/v///wEA/v/+//7/AAACAAAA///+/wAA/P/9//7//f/9//////8AAP7/AAD9//////8EAAIA/P/6/wEAAgACAAEA+//6/wQAAgAFAAIA///6/wIA/f8CAAEA/v/+/////v/7/wEA/v8BAAAA///9//3/BQD+//r//P/+//3///8BAP/////6//7//f8BAP3/AQD9/wAA/v8EAAAAAQAAAAEABQACAP///v8BAP3/AQABAAMA//8BAAAA+//6//7/AQAEAAQAAwACAAAAAAD+///////8//7//v8EAAUAAgD//wEA+f8AAPz//P/5/wIA/v//////+v/+//7////+/wEAAQAAAP7/AAD+/////f/8////+//+//3//f/9//7//f8AAAEA/v/8///////+//r//P/7//r/+//+////AQD+////+//6//3/+f8BAPr//v/6//z/AgD+//v//P/8//j/+P/8//z/+v/8/wAA/v/8//v//P/8//v/+//8//3/+v/8//7//v/7//z//P8AAP3//v/+//v//v/2//f//v////f//v/5//n/AAAEAPr/+//6//v/+v/5//v/+//8//v/+//6//r//P8AAAAA/P/4//3//f/5//v/+/8AAPr/+//6//////////3//v/8//7/+v/9//z/AAD9//v////9//v//f8AAPz//f/4//3/+v/+//3/9v/9//3//v/8//7/AAD8//v//v/9//v//f////3//P/6//v///////v//P/9////AwAFAP7////6//7//f8BAP7/AwAAAP7/AgD+//z//P8FAAMAAAD+/////v/7//v//f8CAAIA/v////////////7//f/+////AQABAP//AgACAP/////9//r//f/8//z//v8AAP7///8BAAQAAgAEAAEAAwAEAAIA/f8AAP//AgD///3/AQACAP////8BAP7////9/////P/+//7/+v/6/wIA/v8DAAEA+v/9/wAA/f/+/wEA/f/9//3/AQADAAMABgADAAIA/v8EAAQABAACAAUAAwAEAP////////z/AQACAAIA//8BAAAAAAD+/wAA/P8BAAEA/f8AAAEAAwAAAAAA/v8CAAAABAD//wQAAQAEAAIA+P/7/wQAAQADAAUABAAEAAQAAAABAAMABgAHAAAA/P///wAAAQABAP//AwAAAP3/AQD///////8DAAIAAwADAAQAAwD9////AQAEAP7//P8BAAEAAwD+/wIAAwAGAAAA9//+/wUABAAFAAQABgAAAAIAAQAEAAEA/f/9/wEA+//6//7///8CAAAA/v/+/wIAAgADAPz////9//v//f8BAP3/AwACAAQAAwADAP//AQAAAP//AwACAAEA//8CAAAA//8AAAEA/f/9/wIACAAEAP///f///////P8DAP7////6//3/AQABAP3/AQD///7/BwABAAAA+v/+//////8AAAUA//8AAAEA+//9/wEAAwABAAEAAAD///v//P8CAP///v8AAP//AAD8/wAA/v8EAP3/AgAAAPv//f/7//3//f8EAAIA/f/8/wQA//8AAAAA/v8BAAEAAQADAAEAAwD+//7//v/+//3/AQADAAIA/v/9/wAAAgACAAIAAAD///z//P/+//z//f/9/wMAAgD9//z//v8BAPz/AAD//////v/+//7/+/8BAAIAAQABAAMAAQABAAAABAAAAAAAAQABAAEA/f/8////BAAGAPz////9//3/AQD9//7//P/+//3/9//8//7/AgD9//3//v////3//f8BAAIA/f/8//z//f8AAP7////9//3//f////3//f///wEA/v/+////AAACAP7//f/9////AQD+//3/AwADAAIA/f8DAP7//v/9////BAD//wAAAAAEAAAAAwABAAMA/v/+/wAA/v///wEAAQD9//3/+//4//n/+/8DAP7//P/6/wEA/v/7/wEAAgAGAP/////7////AQAFAPv/AgD8//v/AAAAAPj/AQAAAP3/AAADAAEAAAACAAMAAwABAAEAAgD+/////v8BAP3/AQD+////AQD///7//v8AAP7/AAD9/wEAAAACAAAA/v///wMA/P/+/wIA/v/+//3///8AAP7/AwAAAAMAAgAFAAEA+v/8/wEABAADAAAA/f/9/////v8FAAMA/v/7/wIAAAABAAMA/v8DAP3//v/+//3/AwD9//////8BAP7//v8BAAAA/P/8//3////9/////P8AAAAA///+/wAA/v/8//7/AgABAAAA/f////3/AgABAP7////7////AQAGAP7/AQD9//z/AgAAAP3//f8BAAIAAwAEAAEAAgAAAP//AQD+/wAA/v8CAAMA/f8CAAMAAwD8/wEABAACAP7/AgACAP3/AAD9//7//v///////////////f8AAAIA/v8FAAAABAD///7/AgAAAPr/AwD///z/AQAEAP///v///wIAAAAFAPz/AwD8//3////8//z/AgAIAAEA/v/6/wQA/v8DAAIA+v/5/wIA//8DAAIA/////wAA///+/wAA/f///wAAAQD/////AAD+//3//f8BAP7////9//7//v/9/wEAAwAHAAIAAwD+//////8AAP7/AgAEAAEAAAAAAP//AAAAAAIAAAACAP/////////////9/wAA//8AAP3//v/7/wEA+v8BAAEA+f/7/wAA/f/5//n//f8AAPz////9/wAAAgAGAP7/AgAAAAAA////////AAAFAP///f/7/wIA//8HAAUA/P/8/wQABwAGAAQAAAABAAQA//8DAAIABAD9/wIA/f8CAAAA/v8BAAEA/f/9////+//9////AwADAAAAAAD7/wEA+/8BAP/////+/wAA/P/8//3//f8BAAAA/f/6////BAAJAPz/AgD9//v/AwABAP3///8CAP3//v////7//f/8//v/AwD///7/+//6//7//v8HAAEA///6////AAABAPz/AQAEAP///P/8//z//v/+/wQAAAD//wAAAgD+//3/BQADAAEA/f8AAPz/+//5//7/+////wIA/P////7/AAD7//z/AQABAPz//v/+//v/BAABAPz//v////7///8DAP3//f/4//3//P/9//r/+v/6//7/AgAGAP7/AAD6//j/AAD9//r/+//8//7//f8DAAAAAgD7//3/BgADAP7/+f8AAP7/+////wMAAwD9//////8BAPb/AgD9//7//v/9//r//P8AAP///P/7//v/+/////z/+P/2//7//P/8//3/+P/7/wAA/f8CAAAA/v/9/////P/5//v/+P8AAPv/+v/9/////v/8//////8AAP//AAD//wEA/P8BAPz//P/2//v/+/8AAP3/+f/9/wEA/P/7/wIAAwAKAAEAAAD9//7/BgABAPz///8FAP7/AQABAAEA/v/8/wAA/P8CAP3///8BAP/////+//3//P/+//z/AgABAAMAAAABAAEA/P8AAAEA/P/7//v/AQADAP3/+//8/////f8CAP3/BQD//wEAAwAFAP7/AgD9//z/AQACAP///P8DAAIABAABAAEAAQABAAIA+/8DAAIABQAAAP//BQD8//////8IAAQA/P/8/wkAAAAEAAUA/f/9/wMA/v8DAAUACAACAAEAAQAAAAAAAAAEAAEA///6//3/AwADAP7//v/+/wAAAQAIAPz//f/8//r/AQD+////AgACAP///P8CAAIA//8DAAIAAAD8/wEA/v////3////+////+v/8/wIAAAAEAAMAAgD+//7//P/8////AQAGAP/////+/wIA/P8BAAIA/v8BAAYAAAD8////BgAAAAIA+/8BAAEAAgACAAEAAAD5////AAAIAPz/AwD+/wAABQAFAP7/AAABAP7/BgAEAP//AAD+/wIA//8IAAIAAQD+/wIAAQAFAP7/BAACAAUABQAFAAAAAwD+//v/AAAAAP7//P/+//3/BQACAAMAAQADAAAA/P/9/wQAAwACAAQABwAEAAIA/f/+/wEABAAEAAIAAwD+/wIABAAGAP3/BAADAAEABAAFAAIABAAFAAQAAAD+//7/AAADAAYABQALAAEAAgD5/wAAAwACAP7/AwAEAAIAAQACAAEAAwAFAAEA/P/5//7/AAACAAEA/f/6//v//P8DAAEA///4/wQA///+/////P8BAP/////8//3/AQD9//r/+///////+f/5//r//f/6////BQADAP7/AAD+/wAAAwAAAP///f8AAP7//P/+/wAAAQAAAAEA/P/+//v//v/4//v/AAADAPr////7//v/AAAAAPz//v/9//3//v////z//f/9//3///////v/AwD//wEA/v/+//z/AQABAP3//v/6////+v8DAAQAAAD+///////7/wAAAAAGAP//AQD8//7/BAABAP3/+//+//7/AAABAP7////9//z/+f/4//z///8DAAAA/f/9////AAAAAAIAAQAFAAAA///7//7/AwD+//v/AAAJAAMA/v/7/wEAAAADAAAA+v/4/wIAAQAEAP7/+P/9//3//////wAAAAD8//3//f/9//3///8CAAIA/v/7//7//f8AAAEAAQADAAAA/f/9//7/AAADAAAAAQABAAIAAAAAAP7/+f/3//7//P8DAAAA/v/7/wIA/v8AAP7/+v8BAAAAAwD/////AQAAAPr//f8AAP///////wAA/f8DAAIA+//7//7/AQD///3//f/6/wAA+v8AAAEAAAAAAAIAAgD7//z/AQAAAAAA/f8DAAIA/f/7/wAA/P/+/wAAAAAEAP//+//5////AgACAP3/AQAAAPn////+/wAA//8DAAQA//8BAAQAAwABAP///v/8//7//v8EAAUAAwD//wIA/P/+////AAADAAEAAAD8////AwAHAP//AQD8//z/BQAFAPz/AQD///3///8BAP7///8BAAIABgADAAAA///8//v//v8AAAAA/f/8/wAA//8CAP//AgACAP7/AQAAAAEAAQD7//z//P8CAP3/AQD+/wEABAADAAAA//8CAP3//f///wEAAgAFAAIAAAD9/wEA/v///wEAAAADAAEA//8AAP//AQD6//3/AQAAAAEA/f8EAAAA/f/7/wAAAwABAP3/AAABAAEABAACAP7/AgADAAEA/f8DAP//BQD///3///////z///8BAP//AQD9/////v8DAAMA/P/+/wQAAAAAAP//AAD9////+//8//r/+P/4/wAAAAACAP///P/+/wMA/v8DAAQA///9//////8DAP7//v/8/wAAAQACAAEAAgAEAAEA///5//3/AAAFAAAAAQD7//z/AgADAP3//v8CAAEAAQABAP3//f/9/wEAAQADAAEAAAACAAMAAwAFAAEABQADAP7/BgACAAAAAAADAAEA+P/6/wIAAgAEAAUA/f/7//7///8CAAEAAAACAAMA/v8BAAAAAQD7/wEA//8DAAAA/f8BAAIA///5//3/AAACAP7//v8AAAAAAQD8//7/+/8CAAAA+//7/wMAAAAAAAIABQAGAAIAAgD9//z/AgD///7/AQD///3///8EAAEA+//7/////v8CAPz/AQD9////AgABAPz///8DAP7/AwD8//z/AQAAAP//AQAHAAAA/f/7////AQABAAEA///+//3/AQAFAP7////5//v///////z////9//3///8AAP7//P/9////AQACAAEAAAADAAEA/f/9/wAAAQD//wAAAgACAP///f/6//z/BQABAP3/+//+/wIAAwACAP7/AQD9//j/+//9//z///8BAAAA/P/9//7//v/9//7/+v/8//v//f//////BAD8/wAA/P8CAAAA+f/8/wIA/f/9/wAA//8BAAEAAwD///////8AAPz//f/9//3//f/6//v/+//+//7//v8AAAIAAgADAP3//v/7////AQACAAEAAAD//////f///wMAAQACAP7//f8AAAEA/v8AAPz////7//z/BAACAPz////9//7/BAAGAAEAAAD9//z/+/8CAPz////7////BAAAAPr///8BAAEAAgADAAAA/f/9//v/+/////r/AwACAAMAAwACAP7////9/wAAAAAEAAIAAwADAP7/AwD9/wEAAQAHAAMA+P/3/wIA/P8DAAQA/P/7/wEA/f/9/wEA+/8DAAAAAwD+//7/BgAEAP3/AwADAAAA/v8BAP//AwACAAUAAgAAAAAA/P8BAAAAAAD//wMAAgADAAAA+//7/wEAAwABAAEAAAD9/wEA/v8DAAAA//8BAAQAAAACAAIAAQD//wIA/f8AAAMAAQADAAEA///8/wAAAQACAP3/AAACAPv/AQD9/wAAAAADAAIA//8DAAYAAQD9//////8BAPz//P8CAAMAAQD8/wIAAgAGAAIAAAD8/wQABAAHAAUAAAD9/wQA/v8CAAEAAgAAAAMA+////wQAAQAGAAIABQD8/wMABQAEAP3/AQABAPv/AwD+//3//f8AAAIA/f8AAAEAAQD9//z/+//+//7///8EAAQABQAAAP7/+//9//7//f8CAAAA/v8CAAEAAgD5//7/AQAEAP3///////7//P/4/wAAAAADAP3/AQABAAYABQAIAP//+P/8//v/AAD8/wAAAQADAAEA+/8CAP3/BQD9/wIAAwADAPv/AgABAP7/+f/+//z/+v8AAAIA/P/6/wIA/v8FAAAA/v///wIA/f/9////AAD7//z/+/8AAAEAAgABAAIA/v/9//z//P8CAP7/AQABAAAAAwD8/wAA//8EAAAA+/8BAAMAAAD9/wQAAwACAP////////7/AgD///3/+v///wEABAAFAAIA///8//3///8BAPn/AAD+////BAACAP7/BAD8//3///8GAP////8BAAEABQD9/wAA/f8CAAAA+//+/////f/9/wAA/v8AAAEAAwD//////f8AAPv/+v/+/wIAAQD///7/+//7//7//v8CAPz//P/3//v/AwAGAP7/AgABAP3/AAD9//z/AAD+//3/+//8//z//f/9//7///////3/AQABAAEA////////AAD///z/AAD+//7//v8BAAQA/////////P/5//v//f////3/AgAGAAUAAgABAAAA//8CAAEA/f/8////AAADAAMA/v8CAAQAAQACAAAAAAD6//7//f8BAP3//v/6/wEA/f/8//z///8BAP3//P/8/wAAAgAHAP7////8/wEABAAGAPv/AAD9//z/AgADAPz/AwD+//v///8CAP7//f/8/wEAAwAEAP7//v/7//3///8CAAAAAAAAAAIA/v/7//3/AQAFAAMAAAD//wEA//8DAAEA///+//////8BAAQAAAAAAAMAAQACAP7//v/7//7//f8BAAEABAD+/wIAAgACAAEA/P8CAAEA////////AgD//wAA//8AAAAAAQAEAAEAAgABAAAAAAACAAAAAgD//////f/7//z/AAAHAAYAAgD//wQA/v8BAP/////+/wAAAgAAAAEAAAAAAAAAAgABAAAAAAD//////f/+//7/AAD+//7//f8DAAIA/v/9/wMA//8BAAMA/v8CAAQAAAABAP//BAD6//7//f8CAAEA/f8BAAEAAAD8//3///8BAP3/AwAFAAQAAAACAAEA/v//////+P/7//////8DAAMABQD8/wMA/v8DAAIABAADAAQA/v///////P/8//7/AgD//wEAAwAEAP7/+//8//3/BAACAAAA/v/7//v//v8BAP///v8AAAAAAAABAP//AAD9/wEA/v8EAP/////+//7/BgD//wEA/v8HAAIA/v/+/wQAAwADAAMA/P/9//////////3/AQADAP7//v/8//7/AgACAP3//f/9/wAAAAAFAP3////7//3///8BAPv//P/9/wAABQAGAAIAAgD9////AAAAAP////8DAAEA/f/6//7//f8FAAIA+//8/wIA/f///wEAAgAAAP7/AQD9//7/BAAFAP/////9////AgABAP///////wEA/P8BAP7/AAD//wAA///7//3///8DAP//AAD9/wEAAAAFAAMA///+/wMAAgD+//3/+P/8//n//v//////AQD8/wEA/f8EAAMA/P8CAAEAAQD+//7/AAAAAPv////+////AgACAPv//f/9//7/BAACAAEAAAD9/////P/+//7/AAACAAAAAwD//////P/6////AwAJAAAAAQD6/wIAAwAFAP3/AAD///3/AwADAP7///8AAP/////+//////8AAP//AQD9/wEA/P/+/wEA/v8FAAEAAgD8/wAABAABAP7/BgAGAP7/AgD6//////8DAAQAAQAFAAQA///7/wAAAAADAP3/AAD+/wAAAgAEAPr/AQD6//v/AwABAP7/AQACAAMAAgADAAEAAgAAAP3/+v/6//3//f8DAAEA/f/6/wAA/v///wAAAAAAAAIAAQAAAP7//v/9////BAAFAP///P/6//7/AAAHAAYA/f///wEA/v///////v/4//3//f8GAAEAAgD9/wMA+//5//z/+//+//3//f/5//z/AgACAP3//v/9//3/AAACAP3////+////AAD///7/AAADAP///v/8////AQAEAAUA/P/+/wAAAAAAAP//BAADAPz//P8AAP7/AAD9////AgD+//7/+v8AAAAA///+/wAABAAFAP///v/5//z/AgABAP7//f/+//7//P//////AAD//wEAAAADAP//AwAEAAUABgADAP7/AQACAAAAAAD+//7/AQD+//3//P/8////AgADAAMA/v8AAP7/AQAAAP//AAD8/wAA//8GAP7/+//5/wQA//8BAAMA/v///wUAAAABAAAAAgD+/wEAAAAAAAEA+/8BAAIABAAAAAMAAAD8//3//v8DAP///f/8//3///8BAAIA///+////AQAAAAIA//8AAAMABAACAAMAAQAAAP7/AAD///3///8AAAAA//8CAAQABAACAAIAAQD+//7/AAD9/wAA/v///////f8AAAEAAgD+//r/AAD///r//////wQAAAAHAAMABAAAAAAAAgAAAAAAAgAIAAEA/f/3/wEA//8HAAUA+//+/wMAAgD+/wAA/f8BAP//AAAAAP3/AQD9/////f8DAP//+//7/wAA+/8AAAAA+//9/wIAAgAFAP///P/6/wIA/f///////v/+/wAA//8AAAEAAAADAAEA/f/8//z/AQACAPz///8BAAAABgABAAAA/f8AAP///v8DAAEAAgD7//z/BAD///n/+v/8//7/AAAEAAEA/P/6/wEA/v8DAP//AAABAAAAAAAAAPz/AQABAAIAAgD+/wAAAgADAAMABAACAAYAAAADAP///P/6/wEA/P8AAAMA/v8BAP//AwAAAAEABAADAP3//v/8//z/BAAEAPv//v/9////AwAGAP3/AAD6//7//v8AAPr//////wMABQAGAAAAAQD///v/AgD5//z/+v8BAAAA9//7/wMAAwD9/wEAAgABAAAA/f8BAAEAAAABAAMAAwAAAP///P/8//n/AwADAP/////7//7///8CAAMA/f///wAA/P/7//3////+////AgACAAIA/P/+//3//f/+/wAAAwADAAAAAAD9//////8DAPv//P/9/wAAAAD//wEABQADAP7////+//3/AAADAP/////6/wEAAQADAAAA+f/+/wQA/P///wQABgACAAMA/f8CAP////8BAPz/AgD8////AgABAAAAAgABAP3/AAD+//3/+/8AAP7/+f/5/wEA//8AAP7//P/+/wMAAAACAAAA///6/wIA+v8BAAEA//8CAAAA+v/5//3//v8AAP//AgD+//7/BwAGAPz/+//2//n/AwAFAP7///8AAP7/AgACAP7////6/////P8EAP//AAD+/wAABAD4//n/+f8DAPr/+//4/wQA/f///wEA/P8BAAAA///+/wQABgADAAAAAgACAP7//P///wEAAAAAAAAA/f////3/AgAAAAEABAADAP7/AAABAPv///8AAAAAAAACAAAA+//9/////P8BAP//+P/3////+//+//7/AQD8/wAA+v/9/wEA//8FAAMA///5//v/AAD+//7/AAAEAAAAAQABAAQAAQAAAAIA//8CAAUA///8//n/BAD9//v/+v///wMAAQAGAAIAAgD9/wAA//8GAPz/BgADAAIABwACAP3//v/+//7/AgADAAEA/v/9//3//f8EAAIAAQD9/wEAAwAKAP7/BAD7/wAABAAGAPz/AAD9//r///////v//P////7/AwABAAIAAAD8//v//P/8/wIA///9/wEABQAHAAMA+//9/wEABAADAP//AwD//wAAAAAAAPz/AgADAAAABAACAAEA//8AAAEA/P8AAAEAAQD8/wIAAgALAPv/AAD8////BAACAPz/BAAFAAEA+f/7//3/AAAFAP7//v/0/////f8FAAEA+//7/wEA/P/7//7///8AAAUAAAACAAMAAgACAAAA///8//r/AwABAAEA/f8BAAAA/P/+/wAA/f/7/wEAAwAHAP7/AQD9//7/AgAAAP7//v8HAAUAAgD+/wAA///9/////f8CAP//BAD+/wAABQAEAPv//f////z/AgADAAUAAAD+/wEA//8GAP3/AwD+//7/BQAAAPn/AgAEAAIAAAABAP7/AAD+//z/AAD//wIA+/8BAAIA/P/9/wAAAgAAAAMAAgABAP7//P/8//3////+/wAA/v8BAAEAAgACAAEA/f/6//3/+v/9////AgABAP//AAD///7///8CAAQAAgAHAAIAAAD6//3/AgABAPz///8DAAAAAQACAP//AQABAP7//P/7//3///8CAP3/+f////7/AQABAAEAAAD0//v/+v8CAP////8BAAYA+//+/////P/9/wIAAQD+/wAA//////7///8CAAIABAADAAEA/f/8//z/+v/7//7//f8CAP///v/7/wEAAQADAP//+P/8/wEAAAD+//7//v/9//7//P////z/+//8/wAA/f/+/wAA+/////r//v/6//3/AgAAAP7/+v///wAAAgADAAAA///6//v///8AAPv//v8AAP3//f/6//7/+/8BAAQAAgAHAAQA/v/6//7////+//z/AAABAPr//P/5//3//v8AAAEA/f8AAAEA/v/+/wAA///+//3//P8BAAMABAD//wMA/f8BAP7///8BAAEA///8//3/AAAFAAAAAwD9//7/BgAHAPz/AQD9////AAAEAP3/AAABAAEAAgABAP3//v////v//f/9/wEA/P8AAAMA/v8AAAEAAQD///v/AQD+//7////7//v//f8BAP3/AQAAAAMABAAFAAEAAQD//////P8BAP//AgACAAEA///6////AAADAAAA/v/+/wEA/v8BAAMAAwD//wMAAQAAAP///P8CAP///f///wMABQABAP//AgABAAEABQADAAAAAAACAAIAAAADAAEABAD///3/+//+//r/////////AQD7//////8EAAEA/f///wMA/f/9//7//f/8/wIAAAABAP///P/9/wEAAAADAAAA/f/8/wMA//8IAAUA/P/6/wIAAAAEAAIA/P/9/wEAAAABAAAAAgACAP/////7/////v8EAP7/AAD6//v///8BAPn//f8FAAAAAgD9/wAAAAAAAAIA/P8CAAMA/////wEABwAFAP//AgABAAAABAAIAAIAAwAAAP//+P/6//3/AgAHAAMA/f/4/////v8CAAEA/v8AAAQAAAADAAIA/v/5/wAA/v8AAAEA/v8DAAMAAQD8//7/AgAAAP///f8CAP///f/7/wEA/P8BAAMA/v///wIAAQAAAAEABwAFAP/////9//v/AwADAAAAAgAAAP//AAAEAAEA/v/+/wQAAQADAP//AwAAAAAAAQAAAP7/AQAEAP7/AQD6////AgACAAIAAAAFAAAA/P/6////AgADAAAAAAACAP//AgABAP7//v/8//z/+//+/wAAAAD8//3/AAD///7//P//////AgADAAIA//8CAAQA//8CAAAAAAD8//z/AgAAAPv//P/6//3/AgACAPv//P/7////AwAFAP//BQD///z/AwABAP3//v////3//f/+/wIAAAD+////+v/+//z//v///wEABwD//wEAAAACAP3/+P/7/wAA/P///wIAAgABAAMABAAAAAAA/f/+//v/+//8//3////9//n/+//4//7///8AAP7/AQAAAPr//v/7//7/AgACAAAA//////3//P/+/wEA/v8AAPz///8BAAMA//8CAP3/AAD+//7/CAABAP3/AAACAAEAAAAGAAIAAgD8//3/+v/+//r//v8AAAIABwAFAP//AAD//wAA/v8CAP3//v/8//f//v/7//n/AQAEAAQAAQAFAP///v/6//3//v//////AAAFAP3/BAD6/wAA//8CAP//9P/2/wQA/P8DAAQAAAD//wAAAAD//wQA//8AAP//AAD+//3/AwAEAP7/AAACAAAAAQACAP//AwAAAAIA/v/9//7//f8DAAAAAAAAAAMAAgD+/wAA/v/+////AgACAAEAAQD//wIAAAACAAMA//8CAAAAAAACAAAAAwD9/wAA/v8DAAMA//8DAAQAAAD+/wEA/////wAAAgAEAAAAAAD9/////v8BAAAA/f/+/wEA//////3////9//3//f8DAAEA/P/7/wEAAgADAAEA/P/8/wQAAwAFAAEA///4/wEA+/8DAAIA/v8BAAIA/f/8/wIA//8DAAAAAQD9//3/BQD///v//P/+//3/AgACAAEA///5//3///8DAP7/AQD8/////f8EAAAAAgACAAIABQABAP3//f////3/AQADAAEA//8AAAAA/f/6////AgAEAAMAAwADAP///v/9//7//v/7//3//f8FAAcABAACAAMA+v/+//r//v/8/wMAAQABAAEA+/8AAP3/AAD+/wEAAwABAP3/AQACAAIA/v8AAAEA/P8BAAAAAAD8/////v8EAAMA///9/wIAAQAAAP7/AQD///7//f8AAAEABQABAAEA/P/8////+/8EAP3/AAD8/wAABAABAP7/AQAAAP3//v8CAAAAAAAAAAMAAQAAAP3///////7///8AAP///P/+/wIAAwD//wAA/v8CAAAAAgADAP7/AgD7//z/AwAEAPn/AwD8//z/AgAIAP3////7//z//v/7//v//P//////AQABAP///v8AAAEA///8////AAD+/wAA//8DAP3//v/8/wEAAQADAP7/AAD+////+v/9//3/AQABAP7/AgAAAP////8FAAAAAAD5/wAA/f8AAAAA+f8AAP7/AQD9//7/AwD+//n////+//z//v8AAP7//v/8//3///8CAP////8AAAIABQAGAP//AgD9/wAA/v8DAP7/BAAAAP//AwD+//3//v8HAAMAAQD+/wAA/v/8//3///8EAAMA///+/wAAAAD+//z//v8AAP7/AQAAAP//AQAAAAAAAAD+//3////9//v//f////3/AAACAAUAAQACAP//AQABAAAA+/8BAP//AgD///7/AgAAAAAA//8CAP7////+/wEA/f8AAAIA+//6/wIA/f8CAAEA+v/9//7//v8AAAMAAAD/////AwADAAEAAwD//////P8CAAMAAgAAAAUABAAFAAAAAQD///z/AQABAAAA/f8AAAAAAQD+/wAA+////////f8AAAAAAQD+/wEA/v8DAAEAAwD+/wQAAgAEAAEA+f/6/wIAAQADAAMAAgADAAMA/v8AAAMABAAHAP7/+//9/wAAAgADAP//AwABAP7/AQD9//z//v8CAAEAAgAFAAYABAD9/wEAAgADAP3/+v///wAAAwD+/wEAAQACAP7/9////wMABQAFAAQABgAAAAEAAgADAP///f/6/wAA+v/7//3//P8BAP///f/9/wMAAgAEAPz/AQD9//v//v////r/AAACAAUAAgAEAP//AgD/////BAADAAAA//8DAAAA/v/+/wEA/P/8/wEABgADAAAA/f/+/////v8EAP7//v/5//7/AgACAPz/AAD+//3/BgAAAP//+f/+///////+/wMA/f/+////+v/9/wAAAgD//wEAAAAAAPv//P8BAP7///8AAP7/AQD6//7//P8DAPz/AgD///v//f/8//v//P8DAAEA/v/+/wIAAgABAAAA////////AAACAAAAAQD8//3//P/+//7/AAACAAIA/v8AAAIAAQD//wIAAAD///z//f////v/+//9/wIABAD///7//v8AAPr/AAD//////f/7//z/+f8AAAEAAAACAAQAAQAAAP3/AgD+////AQADAAEA/v/9////BwAHAAAAAQD+//3/AAD9//3//f/+//3/+f/9//7/AQD///7//f///wAA/v8AAAIA/P/9//r///8BAP/////8//z//P8AAP////8AAAIAAAD9//7/AAABAP7//v/+/wAAAgAAAP7/AgABAAMAAAAFAP//AAD9////AwABAP//AQAFAAAABgABAAMA//8AAAIAAQACAAMAAQD8//7//f/7//v//f8DAP///v/8/wIA///9/wAAAAAFAP7/AAD7/wEAAwAGAPr/AwD6//n/AwABAPf/AAAAAP3///8CAAEA//8AAAMAAAAAAAEAAgD//wAAAAAEAP3/AgD+////BAABAP////8AAP7/AAD8/wMAAQAFAAEA+//9/wMA+//+/wMA/v/+//z//f/9//3/AwAAAAMAAwAEAP///P/7/wAABAAEAP///v/8//////8HAAUA/v/8/wIAAAD//wIA/v8EAP3//v/+//3/BAD8//7/AAD///3///8EAAEAAAD+//7/AAD7/////f8DAAEA/v/+/wAA/f/7////AwABAP///f/8//z/AQAAAPz//v/7////AwAGAP7////7//z/AAD///7//v8BAAEAAgADAAAAAgD///7/AgD9/wAA/v8EAAMA/P8DAAMAAgD6/wAABAACAP3/AwACAP7/AgD+//7///8AAP///f////7//f8BAAIA//8FAAAABAD+//7/AQD///n/AwD9//z/AgAEAAAA//8AAAIAAQAEAPz/AgD9//3/AAD9//3/AgAIAAEA///6/wcAAgADAAIA+//7/wIA//8DAAIAAQAAAP//AAD+/////f///wAAAQAAAAAAAAD9//3//P8DAAAA///+/wAA///9/wEAAwAGAAEABAAAAAAAAQABAP//AwAEAAIA//8AAP//AQABAAIAAgADAAEAAQD+/////v/7//7//f8AAP//AQD+/wMA/f8BAAIA+v/8/////f/5//n//f8AAP3/AQD+/wEAAgAGAP//AwABAAEAAAD//wAAAwAHAAAA/P/6/wMA//8IAAcA/v/7/wIABgAFAAMA/v8AAAMA//8CAAEAAwD8/wIA/f8DAP///P///wIA/f/+/wEA/P/+/wAAAwAEAAAA///8/wIA/f8DAP//AAD9/wAA/f/+/////v8CAAAA/v/6//7/BAAKAP3/AwD9//7/BwAFAP3//v8BAPz///8AAAAA///+//z/AwD///7//f/7/wAAAAAKAAMAAAD6/wAAAgABAPv///8FAAEA/v/+/wAAAgABAAUAAAAAAP//AgAAAAAACAAFAAMA//8CAP3//P/7/wAA/f8AAAMA/v8CAAAAAgD9//3/BAAFAP//AQD///3/BQACAPz///8AAAAAAgAFAP7//v/5//7//f/9//v//P/9/wEABAAHAAAAAgD9//n/AAD+//3//f8AAAEA//8DAAIABAD8//3/BQADAAAA/P8EAAIA/P///wUABQACAAMAAQABAPr/BAABAAEAAAD8//v//f8AAAAA/f////7//v8BAP7//P/5/////v////3/+P/6/////P8BAAMAAAAAAAIA///7//3//P8BAPz/+v/+////AAD//wEAAQABAAAAAQD+/////P8BAP3//f/3//7//v8CAP//+v/9/wEA/P/7/wIAAgAJAAEAAAD+//7/BQD9//v//v8FAP7/AQD/////+//5//7/+/8AAPz//v////7//v/9//3//f/9//v/AgD//wEA//8AAAEA/P///wAA/f/8//z///8CAPz/+f/7//7//P8AAPz/BQAAAAAAAwAFAP7/AQD8//z/AQABAP7//P8CAAEAAwABAAEAAAD//wEA/P8CAAAABAABAP//BQD6//7//v8FAAIA+v/6/wYA//8DAAMA/P/7/wAA/P8CAAMABwACAAEAAAD+//7///8CAAEA/v/8//7/AwADAPz//P/7//7/AQAHAPz//f/8//v/BAD/////AwAEAAEA/f8DAAMA//8CAAEAAAD8/wAA/f////7////9/wAA+//9/wIAAQAFAAMAAQD9//v//P/8//3///8FAP/////+/wMA/f8AAAEA/P/+/wUA///+////BgAAAAEA+v///wAAAAABAAEA///4//7/AAAHAPv/AwD+/wAABgAHAP7/AAD///3/BQADAP3////7/////v8GAAIAAQD//wMAAwAGAP7/AQABAAIAAgABAP3/AAD7//r/AAAAAPz//f/9//v/AwD//wAAAAACAP7/+v/7/wEA///+/wIABgAEAAIA/P/8////AgADAP//AgD9/wAAAgAEAPv/AgABAP3/AgABAP7/AQACAAIA/v/9//3///8BAAQAAwAKAP//AAD4//3/AQAAAPz/AQABAAAA/v////7/AAAFAAIA/P/3//z//v8AAAAA/P/6//z//P8CAP///v/5/wQA/v/9/////P////3//v/7//z/AgD+//z//v8BAAAA+v/7//z//f/7////BAAEAAAAAQD+/wEAAwABAP///v8BAAEA/v/+/wEAAgACAAEA/f////v//v/4//v/AQAFAP3/AwAAAAAABAACAP//AAD///7///8AAP3//v//////AQAAAPz/BAAAAAIAAAAAAP7/AgACAP7/AAD8/wEA/P8FAAUAAgAAAAMAAwD//wMAAQAGAAEAAgD9////AwACAP///v///wAAAgADAAAAAQD+//3//P/5//z///8CAAAA/f///wIABAADAAUAAQAHAAEAAQD9////BQD///z/AAAJAAIA/v/9/wMAAgAEAAEA+//4/wIAAgAGAAAA+v8AAP//AQAAAAIAAgD9//7///8AAP7/AAACAAIA/f/8/wAA//8BAAQAAwAEAAEA///9//7//v8CAAEAAgABAAIAAAD9//z/+v/4/////f8FAAEA///9/wMAAAABAP7/+/8BAP//AwAAAP//AAD///r//P8AAP///v8AAAAA/v8CAAIA+v/7//z/AQD+//3//v/5////+f8AAAAAAAABAAIAAgD7//z/AAD//////f8EAAMA/f/6/wAA/f/+/wEAAgAIAAAA/P/4//7/AgABAP7/AQACAPr////+/////v8AAAMA/v8CAAQABAAEAAMAAAD8//7//f8DAAQAAgD+/wMA+//9//3/AAAEAAEAAAD6/wAABAAIAAAAAgD8//v/BAAEAPz/AQABAP////8DAP////8AAAIABQADAAEA///+//7//////wEA/P/9/wAA/v8AAAAAAwACAP3/AQD+///////7//7//v8FAP//AgD//wEABAACAAAA//8CAP///v8AAAAAAgAFAAIAAAD9/wEA//8BAAAAAAACAAEA//8AAP//AAD7/////////wEA/v8FAAEA/f/8/wEAAwAAAP3/AgABAAEABAADAP7/AgACAAIA/v8EAP//BQAAAP3//v/+//v//v8BAP//AQD9/wAA//8DAAEA+//+/wQA/v/////////8/////P/9//v/+v/6/wIAAgAEAAAA/P/+/wMA//8DAAMA///8/wAAAAAEAAAA/f/8/wAAAQABAAAABAAEAAEA/v/4//z/AAAFAP//AAD6//r///8DAPv//v8DAAEAAgAAAP7//f/+/wAAAAABAAAAAAABAAEAAgAEAAIABQAEAP//BgADAAEA//8CAAAA9v/5/wEAAQADAAQA/P/7/wAAAAAEAAMAAgABAAMA/P//////AAD6/wEA/v8CAAAA/P8CAAQAAQD7////AgAEAAAA//8AAP//AAD8////+/8DAAEA/P/7/wMA//8AAAMABQAGAAIAAgD9//3/AwACAAAAAwAAAP3///8EAAIA/P/8/wEAAAAFAP3/AQD+/wAAAgABAPv/AAADAPz/AgD7//3/AQABAAAAAAAGAAIA/f/6/wAAAwADAAEAAQD///7/BAAGAAAA///7//z//v8AAP7/AAD9//7///////z//P/+/wAAAgADAAEAAAABAAEA/P/9////AgAAAAAAAwACAP///P/6//3/BAAAAP3//P/9/wIAAwADAP3/AwD+//n//v/+//3///8BAP//+//9/wAAAAD9/////f/+//z//f////7/BAD7/wAA/P8DAAEA+P/8/wQA/v///wIA//8AAAEAAwD/////AQACAP3////+//z////7//z/+//+/wAA/v8AAAEAAwACAPz/AQD9////AQACAAAA/v/+//7//v8BAAQAAQADAP7//v8BAAIA/f/+//r//v/6//z/BgADAP3/AAD+//7/AwAEAAEA///9//z/+f8BAPz////8////BAAAAPn/AAAAAAEAAQADAAAA/P/7//r//P/+//z/BAAEAAQAAwADAAAAAAD//wAAAQADAAMAAwACAPz/AQD8/wAAAAAHAAMA9//2/wIA/P8EAAUA/P/8/wIAAAD//wMA+/8CAAAAAwD+//3/BQAEAP3/BAAEAAEA//8CAAEAAwABAAMAAgAAAAAA/f8EAP///P/8/wMAAQACAAAA/P/9/wMABAACAP///v/7/wAA+/8CAAEA/v///wIA/P/9///////+/wAA/P8AAAIAAQADAAEA///8/wAAAwADAPz///////r/AwD//wEAAAACAP///v8CAAYAAgD+/wAAAQADAP3//P8AAAAA/v/6/wAAAgAFAAIA///7/wIAAgAGAAMA/v/6/wMA/v8DAAIAAAAAAAUA+v/+/wQA/v8EAP7/AQD5/wEABgAEAP3/AQADAP7/BQABAAAA/v8AAAIA/P8DAAAAAwD7//v/+//+//z/AAADAAEABQAAAP3//P/+/////v8DAP///f8CAAAA///y//7/AwAIAP7//f///////P/3//////8EAP//AwABAAYABwAJAP3/9f/6//f////7/wEAAwADAAMA+f8CAP3/AwD9/wIABAACAPv/AwABAP7/+v/+//3/+f///wMA/v/8/wQA/f8EAP///////wIA/v/9/wAAAQD7//z//f///wEABAACAAEA/P/9//3/+/8DAP3/AAABAAEABQD8//7/AQAGAAAA/f8EAAIAAAD6/wIAAQACAP7///8BAAAABgADAAAA+//+/wEAAgADAAIA///+////AQADAPr/AAD9////BgAFAP//BwD9//3/AAAJAP3////+//7/AwD7//7//f8DAAAA/f8CAAIAAAD+/wAA/P/+/wAAAQAAAAAA/f8BAP3/+v/+/wQAAgAAAP7/+v/8//7//v/+//z//f/6//3/AgAFAP7/AQABAP7/AAD9//7////+//3/+v/6//r//v/+////AAAAAP7/AgAAAAAA//8AAP/////9//r//v8AAP////8BAAQAAQABAAAA///7//7/AAD///v/AwAGAAUAAQAAAAEAAQAGAAQA/v/8//////8CAAMAAAAEAAUAAgABAP7//v/5//z//P8AAP//AQD+/wEA///+//3///8CAPz/+//8////AAADAP7////+/wQABAAIAP3/AgD9//n/AgD///z/AwAAAPz///8EAAEA///7/wIAAwAEAP///f/+//z//v///wEA/v8AAAIA/P/8//7/AQACAAMAAgABAAAAAAABAAAAAQD//wAAAQABAAMA/////wIAAwAFAP/////8//z//v//////AgD8/wEAAgAAAP///f8DAAIA///9/wAAAgAAAP////8AAAAAAgACAP//AgAAAP7//v8BAAEAAwADAAMA/f/8////AQAIAAUAAQAAAAQA/f///wEA/v8AAAEABAABAAEAAgD8//3/AAAEAAAA//8AAAQA///+/wEAAQACAP7//f8CAAAAAgD//wMA//8BAAQA/f8CAAMAAgAAAP7/BQD7//7//v8DAAIA/P8BAAIAAQD7//z//v8CAP//AwAFAAUAAgADAAAAAAAAAP7/+v/9////AAACAAQABAD+/wMA/f8CAAAAAgADAAIA+////////////wAAAAD9////BQAIAAAA/P/8//3/AgACAP7////5//v/AgAEAAAA////////AAADAAAAAAD9/wAA//8EAAIAAgABAP3/BAD9////+/8EAAIA/P/+/wIAAwABAAEA/f/+//3/AwABAPz/AAACAPz//f/7//v///8BAP3/+//9/wAAAQAEAP7/AQD8//z//v8AAPr/+//7//3/AgAFAAEAAAD9/wAAAAD//////v8CAP///P/7/////P8EAAEA+//7/wAA+v/+/wEAAgAAAAAAAQD7//v/BAADAP3//f/8//3/AQACAAAAAQABAAEA/v8DAP7/AAD+/wEA///7////AQADAP//BAABAAIAAAABAAEAAAAAAAIAAAD///7/+v/9//r//v8BAAEAAgAAAAMA/v8AAAMA/v8DAP7/AQD8//z/AgADAPz/AQD+/wAABAADAP3///////v/AgABAAAAAQD9/wEAAAABAP//AQACAAIAAwACAAAA/f/5//3/AgAHAAAAAQD9/wEABAAHAP//AgAAAPz/AQAAAP3/AAAAAAAA/v/8//////8FAAAA/f/7/wMA/P/+/wIA/f8FAAEAAQD7////BAADAPz/BgADAPv/AwD+//v///8AAP3/AAAFAAIA/v/6/wEA/P8AAPz//v/8//7/AQAEAPr/AAD8//z/BwACAP3/AAADAAIAAQACAAIAAQABAP///P/+//////8FAAEA/f/5//7//v8BAP7//v///wQAAgACAP///P/7/wAAAwADAP3/+f/4//3/AAAIAAYA/P/9/wAA/v8BAAEA///+//////8DAP7/BAD7/wAA/P/7//3//v8DAAEA///7//3/AAAAAP///v/9//3/AAADAAAAAAD9////AAAAAPz///////z////+//z/AAADAAQA//8BAP///v/8//v/AwACAP3///8CAAEAAQD/////AQD7//r/+v/+/wEAAAAAAAEAAgAEAAEA///4//7/AwABAP7/AAAAAPz/+v/9//7/AAAAAAEA//8CAAAAAQADAAUABgAEAAEAAgACAAEAAAD8//v/AQD8//r//P/9//3/AAACAAQA//8CAP3/BAAAAAAAAQD//wEA//8HAP7/+//5/wUAAAABAAMA/v///wUAAAACAAEABAABAAEA///+////+v8AAP7/AQD+/wEA///7//v///8FAAAA///9////AgACAAAA/P/+////AAD+/wIA//8CAAEAAgADAAEAAQD///7///////3//v8AAAIAAAAAAAEAAQD+/wIAAAD////////+/wEA/P/+/wAA/P/+/wAAAQD9//r///////r//v/+/wMA/v8GAAEAAwAAAP//AgAAAP//AwAFAP7/+//2//7//P8GAAYA/f/+/wMABAD/////+/8AAP3//f/9//3/AQD8/wEA/v8EAAAA/P/+/wIA/P8AAAMA+v/9/wMABQAGAP///f/6/wMAAAACAP//AAD+/wMA/f/+/wAA/v8DAAQA/P/6//3/AwAEAP3/AQAAAP7/BQD///3/+/////3//f8DAAEAAQD7//r/BQD+//r/+v/7//3/AAAJAAUA/v/4/wAAAgAIAAAAAAABAAEAAQAAAPv/AQD+/wEA///8//7/AAABAAAABQAEAAUAAQAFAAAA/P/7/wMA/P8CAAUA/v8BAP7/AgD//wEABAADAP//AQD+//z/BAAEAP3//v/9//3/AgAFAP3/AAD6/////v8BAPv/AAD//wMABAAJAAEABAD///r/AgD4//z/+/8DAAIA/f8DAAYABAD9/wEAAwD///3/+/8DAP///v8CAAUABQAAAP//+//7//j/BAACAP///v/8//7//////wEA/P////7/+//7//v//v/7//7//v8AAP//+v/9/////P/8////AgACAP/////8/////v8EAP3//f/9/wAAAQAAAAAAAwABAP3/AQD///3///8DAP/////8/wEAAQADAP//+P/8/wQA/v8AAAYABgAGAAMA/v8BAP7/AgABAPz/AQD8//3///8AAAAAAQD///7///8AAP///f8BAP7/+//5/wEA/f////7//f///wMA//8AAAAA/v/9/wMA/P8DAAEAAQACAAAA+P/6//z//P//////AwD+//7/BwAJAP3//v/4//j/AwADAP3//P8AAP7/AwAEAP/////6/////P8EAP7/AwABAAEABwD6//z/+/8HAP3/+v/4/wYA//8BAAEA+//+/wAA//8AAAQABgADAP//AgAAAP3/+v/9/wEA//8BAAAA//8BAP//AwD//wIABAAEAP7//f////r/AAD//wIABAAFAAUA/v8BAAEA/f8BAAAA/v/5////+v/9//3////8/wIA+//+/wEA//8FAAIAAgD8//z////7//z//v8FAAAAAAAAAAMAAAD9/////P///wIA/P/9//r/BgD//wEA+//+/wMAAQAFAAEAAAD6/////v8GAPn/AwAAAAEABQADAP3//v/+//v/AgABAP///f/8//7//v8HAAQAAAD6////AwAIAPz/AgD9/wIABAAHAP3////5//j///////f/+v/9//r/AQD//wAA//8AAP7/+v/9/wIA/v/7//7/AwAEAAEA+f/5/wAAAwADAP3/AgD6//3//v8AAPr/AAADAP//BAACAAEA////////+///////AQD8/wAAAQAMAPr/AQD6////AQD+//n/AgADAAEA9//7//z/AAAFAP//AAD2/wAA/P8CAP//+v/8/wAA/f/9//7//P/8/wIA//8AAAMAAwAFAP///f/4//f/AwD9//7//f8CAAIA/P8AAAEAAAD8/wAAAQAGAP//AwD+/wAABAAAAP7/+/8FAAIA/v/8/wAAAQD//wAA/f8DAP7/AwD9/wAABwAFAPr//v/+//3/AAAFAAUAAwD/////AAADAP3/AgAAAP7/AQD///r/BAABAAIA/v8DAP//BAAAAPz/BAD//wIA+P8EAAMA/v8BAAMABQD+/wMAAwAFAP7////7//3/AgADAAIAAQABAAAAAwAEAAEA/P/6//z/+f/5//z/AAADAAAA/f8AAAAAAwADAAQAAgAGAAAAAgD8/wEAAwD///z///8EAP3///8CAAAAAAACAAAA/v/8/wAABAADAP7/+f8BAPz/AgACAAIAAADz//z/+/8GAAEA/f8AAAcA+//9/wEA+//+/wUAAwABAAAAAAD9//7///8CAAEAAwAFAAIAAQD8//3/+v/6//3//v8EAP7//f/6/wIAAAADAAAA9//7/wEAAAD////////+/////v8BAP7/+//9/wAA/v/9/wAA/P8BAPv/AQD9/wAAAwD///7/+f/9//7/AwAGAAEAAQD6//v/AQD+//j//v8CAAAA/P/5//7//P8CAAUAAwAJAAUA/f/5/wAAAQAAAP7/AQABAPr//P/3//z///8AAAIAAQAFAAMAAAAAAAIAAAD9//v/+f8AAAEAAQD//wQA/P8BAP7//v8AAAIAAQD6//3/AwAIAP7/AQD7//z/CAAHAPv/AgD9/////v8FAP3//v8AAAEABAABAP7/AAABAPz//f/9/wAA/P8BAAQA/P8AAAIABAADAPz/AQD9/wAA///4//z//P8CAP7/AgD//wMABgAGAAAA//////3//f8EAAIABgAGAAQA///6//3///8DAP3//f/9/wAA/P8BAAMAAQD+/wIA///8//3//v8GAP///f8BAAUABQD///3/AQAAAAAABAAFAAAAAQACAAIAAAADAP//BgD///v//P////r/AAAAAAAAAgD9/wAA//8GAAAA/P/7/wQA+//8/////v8AAAQAAAABAP///P/7/wAAAAAEAP///P/9/wIA/v8HAAUA/P/5/wEA//8DAAIA/P/8/wEAAgACAP//AgAAAP7//f/6/wAA/v8FAP3/AAD5//r//v8AAPf/+/8DAAAABAD//wAA//8AAAIA/f8BAAEA///9/wAABgAGAAAABQADAAAABgAIAP//AAAAAPz/9//6//3/AwAIAAQA/v/7/wEAAAADAAEA/v///wUA/f8CAAIA/P/4/wEA/v8BAP///P8EAAYABQD9//7/AwABAP///f8BAPz/+v/8/wIA+f8AAAIA/f///wMAAAD+/wAABwAGAP///v/7//r/BAAEAAAABAABAAAA//8GAAEA/f/9/wIAAgADAAAABAAFAAAAAAD+//7///8DAP7/AQD6////AwAEAAMAAQAIAAIA/P/4//3/AwADAAAAAgADAP7/AQACAAAA+//7//z//P8AAAIAAAD7//3/AQD///z/+//+//7/AQACAAEA/v8BAAIA/v8DAP//AQD7//z/BQAGAPz//P/4//r/AgACAPv//P/6/wAABAAHAP3/BAD+//v/BAAAAPr/AAAAAPz/+//9/wIA///9/wEA/f8CAP7/AAACAAEABgD5/wAA/v8FAAEA+f/7/wMA/P/9/wAA/////wMABQAAAAEA//8AAPr//P/9//z/AAD+//v/+v/5//////8AAP//AwABAPr////7////AgACAAEAAAABAP///v/+/wEA/f8BAPz///8BAAMA//8EAP3/AgD+//z/CAABAPr///8AAP////8EAAIA///7//z/+/8AAPv/AAAAAAIABwAEAP3//v/8//7///8DAP7//v/9//n////8//r/AgADAAMAAAAEAP///v/8//7/AAABAAIAAgAGAP7/AgD4////AAAGAAEA9P/1/wYA/v8DAAYA/////wIA/////wQA/v////3/AAD+//z/AgADAP7/AwAEAAIAAgADAAEAAwAAAAAA+//6/////P8FAAAA/f/8/wEA///9/////P/+/wAAAgACAAAAAAD//wEA/v8BAAIA/v8AAP////8AAP3/AgD9/////f8AAAEA//8DAAMA///+/wEA/v/9//3//v8BAP7/AgD//wAA/v////7//f///wEA///+//3//v/9//z//P8BAAAA/f/6/wEAAgADAAEA/P/7/wIAAgAEAP///f/2////+/8CAAEA/f///wQA/P/8/wEA/v8CAAAAAAD8//7/CQACAPz//f/+//7/AQADAAEAAAD7//3//f8DAP7/AwD9//3//f8CAAAAAAACAAAABAAAAP7/+//+//7/AQACAAIA//8BAP//+//2//7/AQAFAAIA//8AAP7//f/6//3///8AAP3//f8BAAYABgAFAAAA+f////v//v/6/wEA////////+P8BAP//AgAAAAQABQACAP3/AgABAAAA/v8BAAIA+////////P/7/////P8EAAIAAAD+/wMAAQD///3/AAD9//3//P8BAAIABQACAAMA+//8//7/+f8CAP7/AQD+/wEABwABAP7/AQADAP////8EAAEAAAD+/wQAAAABAP7/AAAAAAAAAgACAP7//P8AAAIAAgD///////8BAAAAAQADAP3/BAD8////BQAGAP7/BgD///7/AwAJAP3//f/8//z/AAD7//7//v8CAAAA/f8AAAAA/////wAA/f/9/wEAAQAAAAEA/v8CAP///P/8/wEAAgABAP3/+//8/////v////v/AAD+//7/AgADAP//AQADAP/////7//////8AAP7/+v8AAAAAAgD+////AAD8//r/AAD///z/+//+//7//f/7//v///8BAAAAAAAAAAMABQAFAP/////7//3//f8AAP7/BwAIAAUAAAD9//7///8HAAQA///+/wEA///+/wAAAAAFAAQAAAAAAP3/AAD9//3//f//////AgABAAEAAAD+//3///8BAPr/+//6//z/AAADAP7////+/wIAAgAFAPv/AQD8//r/AAABAP7/BAAAAPz/AQADAAAA/f/6//7/AgADAP///f////3//f/7/////P////7/+f///wIABAACAAUAAgAAAAAAAgAEAAAAAgD9/wEAAAACAAQA/v/+/////v8AAPz////9//z/AAABAAEAAQD9/wEAAwACAAAA/P8DAAMA///+////AwD+/wAA/f8AAAEAAQABAAMAAwAEAAEA/P/+/wIAAQACAAEA+//+/wQAAwAGAAcABgAHAAUA///9//3/AAACAP7/BQACAAAAAQD7//7//v8EAP7//f/+/wEA/f/8/wAA/v8FAP////8CAP//AgD5/////P8FAAMA9//+/wkABAADAAEAAwD8//3//f8AAAAA/f///wEAAAD9//v//f8BAP7///8CAAUAAQAFAAAAAAD+//z//P////7////9/wEAAwD//wEA/P////3/AwAFAAEA+v/8//7/AAABAAIAAgD//wMACAAHAAAA/P//////AQADAAQAAQD9//3/AQABAAAA/v/+//z/AAAFAAQAAAAAAAIAAgADAAIAAAAAAPz/AgD+//7//P8CAAMA//8CAAEAAwAAAAEA//8AAP3/AgD///v/AgADAPz//P/5//b/+/////v/+f/7////AAADAP7/AAD9//7/AAABAPz//f/8//3/BAACAP3//v/7//z//f8AAAAA//8AAAAA////////+/8DAP7/+f/8/wMA+////wIAAwAAAP//AgD6//n/BAAEAPz//v/5//z/AgAFAP//AQD//wAAAAAHAP7/AgD9/wAA///9//7/AQAEAAEABgAAAAMAAQADAAEA/P/+/wMA//8AAP//+P/7//r//v/+/wAAAwAAAAIA//8AAAAA/P8BAP7/AQD8//3/BAAGAP7/AQD//wAAAgAAAP3//P/+//z/AAD//wQAAgD+/wEAAAACAAEABAAEAAEAAwAAAP7/+//5//3/AQAGAAEAAQD9/wEAAQAFAAEABAABAPv/AQD///z//v/8//7//f/9/////P8BAP7//f/8/wIA+v/8/wEA/P8FAAEAAQD6//7/BgACAP7/BAAFAP3/BAD+//////////7//v8IAAUA///5/wEAAAAGAP3/AAD7//7/AwAGAPj/AAD8//3/BQAAAP//AQAEAAIAAwAFAAMA///+/////P///////v8DAP///f/4//v/+//9//3///8CAAYAAwABAP7//P/6//7/AQAGAPz/+f/6/wAAAAAHAAcA+v/+/wEA/v///wAAAAD9/////v8CAP7/AgD6/wAA+v/6//j/+/8BAAAA/v/6//7///8BAP///v/6//3/AQAEAP3//v/7//z/AQAAAPv////8//j/+//8//z/AAACAAQA//8BAP7////6//r/BgAGAPz//v8BAP//AgD+//3/AgD8//z//P8BAAIAAAD+//7/AQADAP7//v/1//v/AgD///r/AQAAAPv/+P/8//v//////wAA/P8DAAIAAwAEAAYABwAEAP//AQACAAAAAgAAAP7/AgD9//3//P/+////AAADAAQA/v/9//3/BAAAAAEAAAABAAMAAgAGAPz//P/4/wUA/v8AAAUAAAADAAkAAQD+/wAABgADAP///P/8//3//P8EAP7/BwD+/wEA//8BAPj///8DAP7//f/8//7/AAAFAAIA+//5/wMA/f8BAAQA//8AAAcABQAHAAIA/v/7////AAAAAPz/+/8AAAYAAQABAAIACAADAAIA/v/7//////8AAAEA/P/9//z/+//7//7/AQD+//n/AQABAPv//P/6/wIA/v8JAP//AwD6//r/AAACAPz/BAAFAP7////7//z//f8CAAQA/P/+/wMABAABAAEA/f8DAAQAAQD///7/AQD9/wQAAAACAP3/+/8BAAIA/////wIA+//5/wIABAAIAP7////+/wgABAD+//z//v///wEA/v8BAP///f8FAAUA/f/5//7/AwAFAP3/AAD9/wAACQD///3//f8AAPn/+/8BAAAAAgD9//v/AgD9//3/+//+/wIAAAAKAAYAAAD1//3/AQAJAP3//v/+/wEABAD///v/AQD//wEA+//5//3//f8DAAEABgADAAQAAAACAAEA/v/8/wIA+/8BAAQAAgAEAP//AgD9/wAAAQAFAPv////6//3/CQAGAP3///////3/AgAEAPv/AAD7/wIA//8DAPr/AgD//wAAAwAKAP//BQD9//n/AQD3//z/+/8HAAIA+P/9/wUA///6/wQAAgADAP7//v8CAP3///8CAAUABwACAP7/+//2//r/AwADAP7/+//5/wEA/f8BAAMA+////wUA/v/+//3////9//7//f/7/wIA/v8EAAAA///6//r/BQADAP3////9//3///8CAP///P/+/wQA/P8AAAMAAgD///7/BAAAAP3/AAACAP///v/+/wEAAAAAAP7/+f/+/wIA//8DAAcABQAEAAIA/f8DAAEAAwABAP3/AAD8//3//f///wAAAQD+//v/AAAAAAEA+P8BAP7/+v/7/wYA//////z/+////wUAAAD8//3/AAD9/wIA+v8CAAAAAgACAAAA9//3//3//f8EAAEABQD///v/BAALAP3/AAD4//n/BAACAP3/+P////z/AwAEAAAA///4/////v8IAAIAAwD+/wAAAgD9//r//v8DAPv//v/7/wQA//8DAAIA/v8BAAIA///+/wcABAADAP//BAACAP3/+v/9/////P8AAAEA/f/6//z/AAACAAEABAAFAAEA//////v//v/9//7/AgABAAEA/P8AAAAA+/////z/+v/3/wAA/f///wAAAgD+/wIA+v8AAAQAAQAFAAEAAAD3//v/AQABAPz/AQAEAAEAAgAEAAQAAAAAAAAA9v/4/wIA/v8DAAAACAD+/wIA+f/9/wEA//8CAAIA///7/wAAAQAHAPr/AAD6//3/CQAEAP7/AAD7//n/AQAGAP7//P/5//r//P8EAAAAAgD8////BAAJAP//AgD9/wQABQAGAPz/AQD8//r/AwABAPr//P8AAP3/AgD///3/AgD///v//P8BAAUAAQD+/wEABQAHAAIA+//8/wAAAgADAP7/BAD9//7//f/9//r//f8DAPz/AQAAAAEA//8BAAUA//8BAAAAAAD+/wAA//8KAP3/AQD9////AgD9//r/AgADAAAA+//9/wAAAAAGAAIA/f/x//7//P8FAAEA+f8BAAAA+//8/wEA//8AAAEAAgD//wIABQAEAP///f/9//r/AwD7////+/8DAAAA/P///wAAAQD7//3//P8EAP7/BAD//wIAAgD//wEA//8FAAIA/v/9/wEA/v8AAAAA///8/wAAAAD//wAA/v8AAP3/////////AwAFAAIAAQD//wAAAQAHAPn/AAD+/wAABQABAPr/AwACAAAAAAD+/wAAAQADAAAAAwADAAUA//8AAAIAAQACAAEAAQAAAAEAAgAEAAAAAQD9//r//P/+/wIAAQABAAEAAwADAAEAAAD+//3//P/4//3/+f/+//3/+/8CAAEAAQD+/wIABgALAP//BgD7//7/AwABAPf//v8AAAEA//8HAAEABAAAAP7/AQD8//z/AQAFAP7/+/8AAAAAAQABAAQAAwD8/wAA/v8DAAAA/f8CAAIA+v/5//7/AAABAAMAAwACAP//BAD+/wAA/f8EAAIAAQD//wEA///9//3/9v/8/wIABAAFAAMAAgD///z//v8AAPv//f8BAAAABAD//////f/+//n//f8BAPr//P/9/wAA/v8CAAMA+/////7/AgD8////AgD+/////f////7/AgABAP///f/6//n/AQD+//7//P8BAAEA/v/8/wEAAAAAAAIABAALAAIA/P/5////BAD7//r///8AAPn/AQD///7//P/6////+v8EAAQABAADAAMAAgD8//r//f8BAAEAAQABAAYA/v/9//7/AgAEAP3/AAD4//v/AgACAPz//v/9//v////+//v/BQADAP//+/8BAP7/AAD//wEA//////7//v8CAAAA///9/wEA/P///wIA/v8BAAMAAwAAAPr////+/wAAAQD7//7///8GAP//BAABAAUABAAHAAIAAQABAAAA/v////7/AgAFAAAAAgD8/wIA/f8BAP3/AQADAAMAAAD+////AAD+//3//f/9/////v8GAP///v/9/wIABQADAPz/AwD+////BAAGAP3/AwABAAEABAAFAPr/BAD///3/+//+//3/AAAEAAQAAAD6/wIAAQAEAAAA+v/+/wEA+//8/wEAAwAAAAIA//////j//P/5/wEAAgAFAP7/+/8AAAUA//8AAAUA/f/9//7//v8CAAAA/v/9/wEAAQAAAP//AwABAAEAAQD///z//v8AAP3/AAD+//////8EAP////8AAAEAAQAAAP3//P/9//3/AgD+//3//v/8//v///8HAAAABAACAP7/BQD///3//P8DAP7/9P/4////AQD9/wIA/P/8////AAAEAAEAAgD//wAA+//8//z/AAD9/wAA/P8AAP7/+////wAAAQD7//7/BAAFAPz///8CAP7////+/wIA+v8CAAIA+v/9/wMAAQAAAAQACAAEAAAAAAD9//r/AwABAP//AQAAAP////8EAAAA/P/7/////v8HAP7/BwD+////AwAAAPf/AAD///v/AgD///7/////////AAAFAP7//v/4////AgABAAAABAADAAAABAAEAAEAAAD8//v//P/9//3/AAABAP///f/9//3//f/9//7/AAACAAMAAgACAP///v/9//7///////7//v8BAAEA/f/5////BAAFAPz////6////BAAEAPr/BAAAAPr/AQAAAP3/AQAAAP//+///////AwD//wAA//8AAP////8BAP//AgD5////+P8AAAEA+P/8/wMA/f/7//3//f8BAAEABwAAAP7/AwAEAPz/AQD///z/AAAAAAAA/f///////v8AAAAAAwACAP///v/7////AwAEAAAA///9/////v8CAAMA//8CAAIAAAACAAMA/f8BAP3//v/7//v/AQAAAPv/AAD9////AwAAAAIA/v8CAPz//v////3/+//8/wEAAAACAP3/AAD+/wAAAQAHAP7//f/4//r/AgABAPv/BAAHAAIABAD/////AAD+/wAA/f8BAAMABAD///3/AAACAAAAAwAEAAEA/f/4/wAA/f8HAAQA+P/6/////f/9/wQA/f///wAABQACAP7/BgAGAAAABAACAP7/+//8//7/AQABAAMAAgACAAMAAQADAP//AQABAAEAAAD+//////8AAAAAAwADAAEABQADAP///P/9//z////+//7/+v/9/wEAAQAHAAIAAwD+//7/AwACAPz////9/wAABAAEAP7/AAD9//r/AQD7/wAA/P8FAAMA+v///wgACAD//wIAAgAFAPr/AQABAP//AgD8//3/AgACAAIAAAD//wEAAgAGAP//+//5/wIA//8DAAIA/////wQA/f8BAAQA/v8BAAEA///7/wAAAQAEAP///////wEABAACAAEAAAAAAAAAAAD//wAA//8AAPz//v/8//z//v8DAAEA//8AAAAA/f/8/wAA/v8FAP7///8AAAAABAD2//3/AQAKAP7/AAAAAP///P/5/wEA//8GAP//BAACAAQABAAGAP7/+f/9//f////6//7//v8BAAMA+v8EAP//AQD8/wIABAAEAPr//P/+/////f/+//v/AAD//////P/+/wAAAQAAAP7/AAAAAAAA+///////AQD8//r/+f/+////AAABAAAAAAD8//v///8AAPz/AAD///////8AAAMAAQADAP7//P/+/wAA/f/9/wMAAQACAP//AAD///7/AwADAAMA/f/+/wAAAwACAAAA/f/9//7/AgAFAP3/AQD//wAACAADAP7/BgD///z//v8FAAAA//////7/AQD8//7//P8AAP3//f///////f/8/wEA//8EAAMABAD+//3//v////3//v8DAAAA/v/+/wAA+f/+/wAA+v/8/wIA//8AAP//AgAAAAEA/f8BAP7//v/8/wIA/v/8//////8BAPv/AQD6//7/BgAJAP3/BAD+//r/BQAGAPz/AgD6//r///8IAAIAAQD+/wMABQAEAP///v8AAAAAAQD///z/AQD+//7/AgABAP3///////3/AgD///7///8BAP///f8AAAIAAAD//wEAAwAEAAAA///9//7/AgADAP7/AAD8//3/AgACAPn//f////3/AQADAP7/AQABAAQAAAABAPz///8BAP//AAAHAP7/AwD7//7/AAD///3//////wEAAAABAP///f8CAAAA/f/5//3//f/+/wAA/f/+//3//v8BAP//AQD9/wMAAQAAAAAA/v8BAAAA///8/wAAAgAAAP7/AwAHAAIAAAD9//z///8AAP7/AwABAAEAAwD//wEAAQADAAMAAAAAAAIAAQAAAAEAAAAEAAIAAQD////////8//z//f8CAP//BQADAAMAAgACAAAAAwAGAAMAAgACAAAA/f8CAAEAAQAAAP//AwD+/wAAAwD//wAABAAFAAAAAAAAAAMAAAACAAIAAwABAAAA/v8CAP////8BAAIAAgD+/wAA/f8DAAEAAQD///7/BQAAAP//AQACAAAA/v/7//7//P/+//3/+v/+/wAAAgACAAQAAgAHAAAABAD///3//v/+//r/AAAFAAUAAQAAAAMAAgACAAEA/v/7/wEA/f8DAP3//P/+/wAA/v/7/wEAAwAEAAEA///+//7/AQAEAP3//f/5//z/AQAEAAEAAwADAP//AgD//wAA/v////////8AAAEAAAAAAP3/+//7/wEAAAACAAEAAAAAAP//AQD///3//f8AAP7/AwD//////f8BAPj//v////z//v/8//z//P8EAAMA/P/+/wAAAgD9//7/AAD+//z//P/+//z/AQABAAAA///9//7/AgD//wEA/f8DAAQA/P/9/wIAAAD///7///8FAAAA/v/8/wAABAD+//3////+//j/AAD//////f/8/wAA/f8DAAQABAAEAAMA/v/9//z/AQAAAP////8BAAMA///9//3/BAAFAP7/AAD8//7/AwACAAEAAAD+//3//P////z/AQD///7///8AAP////8AAAIAAwAFAAAAAAD+//////////3//f/+/wAAAAABAP//AgABAP3/AgD//wAA///7/////v8BAP7/BAACAAQAAwAFAAAAAAD8//3/AgADAP//AQAEAP//BQABAAIA/v/+/////v8DAAEAAQD+/////v/7//v//v8DAP7//v/+/wEA/f/8/wAA//8EAAAAAgD8/wEABAAHAPv/BAD+//v/AgABAPj/AwAAAPv//v8BAAAA//8AAAIAAAD//wIAAQAAAAAA/f8BAP3//v/7////BQADAP//AAAAAPz////8/wEAAQADAP//+////wMA/v8AAAMA/v/7//3//v8BAP3/AQD9/wIAAQACAAAA/P/9/wAAAwAAAPz//P/9//3/AAACAAIA/v/+/wAAAAAAAAAA/f8CAPz//f////7/BQD+////AAD///3//v8FAAEAAQABAP7/AgD7/wEA/f8EAAMA/f///wIA///8/wEAAAAAAP7///////7/AwABAP7//f/8////AgACAP/////8//r//P////3//v/+/wEAAwAFAP//AgABAP3//v/5////+/8DAAIA+/8AAAMAAgD8/wEAAwACAP3/AQABAP7/AQD9//3/AAD///7//f8EAAEA//8AAAAAAAACAP3/AgD+//3/AQACAPv/BQD9//7/AQACAP3/AQAAAP//AQACAP7///8AAAIAAAD//wMAAwACAAAAAQADAAUAAAD9////AAD///////8BAP//AgABAP3////7//z/+/8AAP//AwAAAP7///////z///8EAAAA///8/wAA/v/9//////8BAAEAAgACAAMAAAD/////AgACAP////8AAAEAAQD+//7//f8BAP//AgD9//3/AQD9//3//f////7////9/wEA+////wIA+f/8////AAD///z//P/8//7/AQD+//////8DAP//AgAAAP////8AAP7//v8AAP3//////wAAAgADAAIA/f/7/wAABAAFAAAA/v/9/wEA/v8CAAAA/v/+/wAA//8BAAAA/P8CAP///v/9/wAAAAD///3/AQD///3/AQD//wEA/v8BAP3/AAD//////f8AAAEA/f8BAP7////8////AwAIAP///f/6//7/BAAEAPv/AAAAAPv/AQD+////AAAAAP///////wEAAAD9/wEA//8IAAIABAABAP//AAD8//7//P8HAAEA/P/8/wEAAQD9/wYAAQAEAAIABAACAAAABgABAP3///8BAPz/+v/7/wAA/P///wQAAQAEAAEAAgD+//3/AQAEAP7/AQD9//7/BAACAP7/AwACAP//AgABAP3//f/9/wAA/f/9/////P/+/wAAAwAHAAEAAwD9//z/AQD///3//v///wEAAgACAP///P/7//3/AwAEAAEA/P////7///8BAAIACQAEAAIAAAD+//v/BAAEAAIAAQD8//7//v8BAAEA/f///wAA/f8DAAIA///+/wEAAgD9//7/+v8AAP7/AAABAAAABAD//////v///wAA/f8CAAEA/P/7/wIAAAAAAAAAAgACAP7/AQD//wAA/v8FAP///f/7/wEAAgAEAAQA/f///wEA/P8AAAQAAAACAP//AAAAAAEABAD8//v///8FAP7/BAABAP///f/5/////f8CAP//AwADAAMA/v8AAP7//P/+//r/AgD9////AQAAAAIA/f8CAAAA/v///wIABAAEAP7/+f/9//3//f////3/AQD//wAAAAD///3/AgD///z/AQAAAP//+v8AAAIAAgD9//7/+////wAA/////wEAAAD+//7/AAD///3///8AAAAA///+/wQA/v8BAP///P/+/wEA/////wMAAQAAAPz///8BAAAAAwADAAEA/f/+/wEA/f8AAP3//v/9//7/AQABAPr/AQABAP7/AwACAAAABgAAAP///v8AAP7//f8AAAAAAAD8/////P8CAP7/+//8//7//f/8/wEA/f8EAAMAAwD8//3//v8CAP3//f8AAAEAAwD//////P8BAP///f/+/wEA/v/9/wEAAwADAP//AQADAP//AAD//////v/5//z/AQABAPr//f/7////CAAIAP3/AgD4//n/AQADAPz/AQD9//v//f8BAAIAAgABAAUAAgADAAEAAAD///7/BQADAPz/AQABAP7////9//v/AwACAAEA//8AAAEAAgAHAAEA/v/5/wIA//8AAP///////wAA+//9/wEAAQACAAAAAQD+//v/AgD///z/+/8BAAEAAQAEAAAAAQD9/wQA/v8CAPr/AgAAAP3/BAAGAP3/AgD9//v/AQADAP3////7/wAAAgABAP////8DAAEAAQABAAAAAAD8////AAD9//z/AQADAAEAAgD9/wMA/v8AAP///v//////AwAAAAMAAQAAAPz///8EAP///P/4//r//v8AAP3/AgD//wEAAgAAAAEA/f8CAAIA/v///wEAAQD+//7/+//9//3/AQACAP//AAD8//v/AgAFAP7/BQD//wEA/v8AAP7/AwAMAAQA///+/wQA/v8CAAEA/v///wEABAD//wEABQD///3/AgAEAP//AAD//wIA/P8AAAAA///+/wAA/v8FAAIA/v/+/wIA//8AAAQA//8DAAMAAQD+//v/AQD7//z/+v/+/wAA/P///wAAAAD///7///8BAP//AAADAAIAAAACAAAA////////+v/9////AQAEAAUABAD//wMA/v8EAAEAAwAAAAMA/f8AAP3/+//+////AQD8/wIABQAIAAAA/v/9//z/BgAFAAIA///9//v///8BAP///v8AAAIAAAAEAAIABAD+/wIAAwAGAAAAAQD///r/AQD7////+v8FAP///f/8/wEAAAD//wAA+//+//3/AwAAAPz/AgACAPz//f/6//n///8DAP7//P/8////AQAHAPv/AQD6//z/AgACAPz//P/6//z/BAAHAAAA/v/8/wEAAwD///7/AAAFAAEA+//5//7/+v8EAAMA/P8CAAYA/P/8/wAAAwD8//7/AAD6//v/BQACAP7////7//3/AAADAP7//P/+/wEA//8BAP7/AAD//wEA/v/+/wAA/v////v/BQAAAAEA/v8BAAMAAAAFAAMABAD8//3//P/9//j//f8CAAEABAABAAUAAgAAAAIA/P8FAP3/AAD7//z/AgADAPv//f/+/wEAAgACAP3/AQAAAPz/BQAAAAEAAgD5//z//f8CAPz/AQABAAAABgACAAIA/f/7//z//f8GAAEA/////wMAAAAAAAAAAAD9//r////+//z//v///wAAAQD9//7///////z//P8CAAIA+//+/wIAAAADAAAAAQD6//3/CAAEAP3/BAADAP7/AwADAAAAAwD///7//f8EAAIAAwD9/wAA/f8AAP3//f8BAAEAAQD///v//P/7//3/BQD//wEA//8FAP//AAACAAMAAQAAAAIA/v8CAAAAAAACAAEAAAD8//z/+v/9//3//f8AAAMAAQD///7//f/6//3/AQACAP///f/+//////8BAAMA/v8CAAAA//8AAP7/AgABAP////8DAP//BAD9/////P/5//v//f8FAAEAAQD8//3/AwACAP7//v/9//3//v8CAP7/AgD9//7//v8AAPz/AQABAPz//f/6//v//v8DAAMA/v/+/wEA/f/9//3/AgADAP//AQD///7/AgD+//3/AAAAAP7/+////wYAAgACAAEAAQACAAMA/f/4//z/BAABAP7/AQD///z/+//+//7////+//////8DAAIAAQACAAYABgAIAAAABQABAP//AAD8//r////9//v//f/+/////f///wIAAAACAP3/AQD8//7/AQABAAEAAQAGAP3////7/wIA/////wEA/f8BAAUAAAD+////AwABAP///P/9//z//v8EAAEABgABAAMA//8AAPv/AQACAP3/+//6//z//v8DAAEA/f///wAA/f/7/wEAAAACAAMABQAFAAAAAQD+//7/AQAAAP7//P8BAAQAAgAAAAAABAABAAMA/v/9//7////+/wEA/v8AAAAA/P/8////AAD9//n/AQAAAPz//v/8/wAA/P8GAAAAAgD8//z/AgABAPz/AgACAP3//v/8//7//f8CAAMA/v/9/wEAAgD+/////v8GAAIAAAD9//3/AgD8/wAA//8DAP///v8CAAIA/f///wEA/P/7/wAAAgAEAP///f/9/wQAAAD///r////9/wIA/f///wAA/v8GAAMA/P/5//7/AgAHAPv/AgD7//v/BgD///z//P8BAPz//f8CAP//AAD8//z/AQD8//z/+//9//7/AQAKAAQA///2////AgAHAP///f///wAAAQD+//n////8/wEA///7////AAADAP//BgAGAAQAAgAEAAEA/v/7//7//P8BAAUAAAAAAAAAAQAAAAEAAwAEAP//AQD+//7/CAAFAP/////9//3/BAAFAPz////4//7//v8FAPz/AgD//wEABAAJAP//BAD+//v/BAD6//z/+/8GAAIA+/8CAAMAAwD6/wEAAgD///3//P8JAAIA/P/+/wcABgACAP//+P/3//v/AgAFAAEA/P/5/wAA/v///wIA+//+/wEA/f/9//v//v/7//3//v///wEA/f8CAAEA/v/7//7/AgADAP//AAD///7/AAACAP7//P/+/wMAAAABAAMAAwABAAAABQABAP7//v8BAP7///8AAAEAAAAAAP7/+v/9/wEA//8CAAUABQAFAAMA/v8AAP//AgD///3/AAAAAP///v//////AQD9//z/AAACAAIA+/8CAP3//f/6/wQA/v/+//3/+/8AAAQA///8//7/AAD8/wEA/P8DAAEAAgAEAAAA+P/4//7//v8DAP//BQD9//r/BwAMAP3/AAD3//f/AwABAP7/+v8BAP//BQAGAAAA///2//7//f8IAAAAAgD//wEABgD8//v//v8EAPv/+//7/wMAAAACAAEA/P///////v///wUABAAEAP//AwABAP7//P/+/wEAAAADAAEA/v/9//3////+/wAAAgADAP///v8AAPz/AQD//wEAAwACAAEA+f///wAA/v8CAAAA///4////+f/9//7////+/wIA+v/9/wEA/v8GAAQAAwD9//v/AAD+//z//v8FAAAAAAD//wIA/v///wAA+/8AAAQA/v////z/BQD9////+f/9/////v8CAAIAAAD7/wEAAQAHAPn/AQD7//3/BgAEAPv//P/7//r/AgAHAAEAAQD8//z//v8GAAMAAQD8/wEABwAKAP//AwD9/wMABAAGAPz//v/8//v/AAD+//n/+/////3/AwAAAAAA///9//v/+////wMA///7//3/BAAFAP//+v/6/wAAAwAGAP7/AgD5//z////9//f//P8CAP3/AwAEAAMA////////+/////7/AAD6////AAAMAPn/AQD8/wAAAwD+//f/AQADAAAA9//8//3/AgAIAAEA///y/wAA+/8DAP//+//+/wEA/f/8/wAA/v///wMAAAAAAAQAAwACAP7//f/5//j/AwD//////v8DAAEA/f/+/////f/5//3///8EAP//AwAAAAEABAAAAAAA/v8GAAIA/f/5////AAAAAAEA/f8DAP//AgD7////BQAEAPr///////7/AQAFAAUAAwD/////AAAEAP3/BAABAP//BAAAAPv/AwABAAEA/f8BAP//AgD///z/AwABAAMA/P8DAAMA/v8BAAIABAD//wEAAgADAP///v/9//7/AQABAAEA//////////8DAAIA/v/7//3//P/7//z/AAABAP3///8BAP7/AAABAAQAAwAJAAIABAD+/wEAAgD///z///8FAP////8CAAAAAAACAP/////8/wAAAwAEAAAA+v8CAP3/AgABAAMAAgD2//v/+/8DAP7//f8AAAUA+f/8/wEA/f/+/wIAAQAAAAAA///8//3//f8DAAIAAwAEAAIAAQD8//z/+v/7//7///8GAP///v/6/wIAAQADAAEA+v///wMAAwAAAP//AQD+/////P////v//P/+/wEA/f/7////+/8CAPr/AAD5//3/BQACAP3/9//8//z/AwAEAAAAAQD7//3/AQD+//n//v8DAAAA/v/5/wEA/f8CAAQABAALAAQA/f/4////AQAAAP3/AAAAAPv//v/7//7/AAAAAAEAAAADAAEA/v/9/wQAAQD///7//P8AAP///v/7/wIA+v8AAP7/AQAAAAIA///5//v///8FAP3/AQD7//3/CAAHAPv/AgD+//7//P8DAP3//v//////AAAAAP//AQABAP7//f/8//7/+f8AAAUA/f/9////AwADAPv/AQD//wEAAgD7//3//v8DAP7/AQAAAAMABQAGAAAA/f/9//z/+v8CAAAABQAFAAUAAAD6//7///8DAP3//v/9/wAA/P8AAAEAAQD+/wIA///9//3//f8JAAAAAAADAAcACAD///7/AQABAAAABQAFAP//AAABAAQAAgAEAAAABgAAAP7//P////v/AgABAP//AAD7/wEA/f8FAAEA/v/8/wQA+v/7/wEAAQAAAAMAAQABAP7/+//7/wAAAgAGAAAA/P/8/wMA//8HAAYA+f/4/wEA/f8DAAQA/f/8/wMAAAAAAP//AwAAAP7//v/7/wAA/v8HAPz/AQD5//v/AAABAPj/+/8CAP7/AwD+/wAA/v///wIA/f8CAAAA/v/+////BgAEAP//BQADAAEABQAGAAAAAAAAAP7/9v/6//3/BAAFAAQA/f/6////AAADAAAA/////wYA/v8EAAMA/v/6/wQA/f8AAP//+/8DAAQAAgD7//7/BAACAP////8CAPz/+f/7/wIA+P/+/wEA/P/+/wIA//8AAAEACAAFAAAA/v/9//r/AgACAAAAAwAAAP//AAAHAAIA///+/wMAAAAFAP7/AwABAAAAAAAAAPz///8EAP3/AwD6//7/AAABAAEAAAAIAAEA/f/5/wAAAgAFAAIAAwADAP3/AgAAAP///P/9//3//v8CAAIA///6//z////+//3/+v8AAP//AwADAAEA//8AAAIA/v8DAP7/AQD6//v/BAAHAP3//f/6//v/AwAEAPz//f/7/wEAAwAHAP3/AwD9//n/BAD9//n//v8AAPn/+v/+/wMAAQD9/wEA/v8EAAAAAQAAAAAABgD7/////P8AAAAA+v///wQA//8BAAIAAQD//wEAAwD9/////f8BAPz//v/+//7/AgABAPz//P/6/wAA//////7/AgAAAPn////7////AQACAAAA/v/+//z//f/+/wIA/v8EAP7/AAABAAIA/v8DAPv/AAD9//r/CAD+//r//v8BAP////8GAAUAAwD+////+f/8//r///8AAAIABwAFAP3//v/8/wAA//8EAP7//f/8//X//f/4//n/AgAFAAUAAQAGAAIA/v/7//7//f8AAAAAAgAGAP3/BAD2//7//v8FAAAA9P/2/wgA/v8EAAcA/v///wAAAAD8/wIA/v8BAP7/AQD///z/AgABAPr/AQABAP//AAAEAAAABAD//wEA/P/7/wAA/f8HAP///P/7/wMA///+/wEA/P/+/wAAAQACAP//AAD+/wIAAQACAAQA//8CAAAAAAAAAPz/AQD8/wAA//8EAAQA//8DAAMAAAD//wEA///+//3//v8AAP//AQABAAEA/v////7//f/+///////+//z//f/9//v//v8BAP///f/5////AQABAAEA+//8/wQAAgAHAAAA/v/2/wAA/P8DAAEA/P8AAAMA/f/5/////f8BAAAA////////CQABAP7//P////7/AQACAAMAAAD7//3//v8EAP7/AgD8//7//v8FAAAABAADAAIABgACAP7//f////3/AAACAAEA/f////7/+//4//7///8DAAEAAQABAP7////6//3////+//3//f8CAAQABAAEAAIA+f////r//v/6/wIAAAAAAAEA+f8BAP7/AQAAAAMABQABAP3/AAABAAAA+////wEA/P////7//P/8/////v8GAAMAAAD9/wIA///+//z////8//3/+/8AAAIACAADAAQA/P/9//7/+P8EAPz/AgD9/wEABwACAP3/AQAAAPz//P8BAAEA/////wUAAgABAP7//v8AAP////8BAP///f/+/wIAAQD+/wAA//8DAAAABAAFAAAABAD8//3/AwAEAPr/AwD9//z/AgAIAP7////8//3//v/8//3//P8AAP////8AAP////8BAAEA/P/6//7/AAD+/////v8DAP7//v/9/wEAAQADAP7//v/8////+v/9//z/AAABAP7/AwADAAEAAQADAP7////4/wAA/f8CAAIA+/8CAP7/AQD6//7/AgAAAPn/AAD///r/AAABAAAAAAD8//v///8DAP//AAABAAIABQAFAAAAAgD+/wIA/v8EAP//BgACAAEAAgD+//3//f8HAAMA///+//7//v/5//v/AAAGAAQAAQAAAP//AQABAP7/////////AQADAAEAAgACAAEAAgD+//r/+//+//r//P8AAAAAAAABAAUAAAACAP//AQACAP7//P/9//3////+//z//////////v8AAP7////+/////v///wAA+v/4/////f8CAAAA+v8AAAEAAAABAAUAAAABAAIABAACAAAAAQD9/////P8DAAQABAABAAQAAQADAP7/AAD9//3/AgAFAAMA/v8BAAAAAgD8////+////wEA/P///wIAAwAAAAIA/f8BAAEAAwD9/wMAAQAGAAIA+P/7/wQAAwADAAUAAQAEAAQA//8AAAQACAAHAP///v8BAAEAAgACAP7/AgD9//z/AAD8//3//v8EAAEAAgAFAAQAAwD6////AQAHAP3//f8CAP7/BAD+/wEA//8EAP//9f/+/wQAAwAEAAMABQD//wIAAgAEAP///P/6////+f/6//3//f8DAAIAAAD+/wIAAgAFAPz/AQD9//v//v8AAPz/AAAAAAMA//8DAP7/AQD+//3/AwABAP///f8EAAEA/v/+/wMA/f/9/wAABwAEAAEA/v8BAAEA/v8FAP/////6////AwAFAAAAAwAAAP3/BwAAAP//+f/7//z//f/9/wMA//8CAAEA+//8/wAAAQD+/wEAAAABAPv//f8AAP3///8AAP7/AgD7//3/+/8DAPr/AQD9//j/+//8//z/+/8EAAMA/v/+/wMAAAD+/////f8AAP7/AAAAAP3/AwD9//7//f8AAAAAAAADAAQA/f///wIAAQABAAMAAgAAAPr//f/9//z//P/+/wQABwABAP7///////f////+//z/+v/4//z//P8FAAQAAwABAAMAAAABAP3/BAD+//7/AgAFAAQA/v/7////BgAGAP//AQD9//z/AQD//wAA//////3/9v/8//7/AQD///z//f/+//7//P/+/wEA/P/+//v//f//////AAD//wAA//8BAAAA/v8AAAMAAAD///7/AAAAAP///v/+/wEAAAD+//3/AwACAAUAAgAIAAAA/f/6//7/BQAEAP//AQACAPz/BgACAAEA/P///wEAAAACAAIAAgD8/////P/7//n/+v8CAPz//f/6/wEA/v/8/wAAAAAFAP//AAD6/wEAAwAHAPj/AgD5//j/BAABAPb/AgD///n//f8EAAQA//8AAAQAAgACAAMAAwAAAP7//P////r//v/8/wEABgAFAAIAAgABAP7/AAD8/wAA/v8DAAAA/P8BAAUA/v/+/wIA///9//z//P/+//3/AgD//wMAAgAEAAAA+f/3////AgACAPv/+f/4//7//f8FAAUA/f/8/wMAAAD+/wEA/v8GAAAAAQD+//3/BwD///7/AAD9//r///8DAAEA/v/9//z/AAD9/wAA/f8BAAEAAAD//wAA/f/8////BAAEAAEA/f/7//r/AQAAAPv//f/4//3/AgAGAP3/AAD6//r/AgABAP7///8DAAQAAwAEAP//AgD+//z////6////+/8DAAUA/P8CAAMAAQD3////BQAEAP7/BQAEAP3/AgD9//3//v////7//P///wAA/v8AAAIAAgAHAAAABAD///7/AgAAAPj/AwD7//v/AAAEAP///v///wEAAQADAPz/BAD+/wEAAQD9////BAAKAAIA/v/6/wYAAAADAAIA+//6/wUAAgAGAAMAAgACAP7////7//z/+v/8////AAAAAAAAAAD9//3/+/8CAP7//f/9/wEA/v/9/wEABAAGAAEABAD//wAAAAADAAAABQAFAAEAAAABAP//AAD+//3//P8AAAAAAQAAAAAA///8/wEA//8DAAAAAAD7/wIA+f///wIA+v/8/wEA///7//r//P/+//z/AAD9////AQAGAP7/AwAAAP//AQD//wAAAQAFAP///f/7/wMAAAAHAAYA/f/7/wAABAAEAAEA/f///wEA/P//////AQD8/wIA/f8CAAAA/f8BAAAA+v/6//7/+//+////BAAFAAEAAAD8/wAA/P8DAP//AQD+/wEA/f//////+/8BAAAA/v/6//7/BQAIAPz/AAD9//7/BQABAP3//f////z//f///wAAAAD9//z/AwD+//7//P/5//3//f8JAAMAAgD8/wEAAQAAAPz///8HAAIA+//5//3//////wQA///+/wAAAgACAP//BwADAAMA/f8AAP7//P/8/wIA/P/+/wIA//8BAP//AgD+////AwAFAAAAAgD///7/BQADAP//AAABAAAAAgAGAAAA/f/5//7//f/+//z//v///wEAAgAHAAEABAD+//n/AQD9//7//v8AAAIAAAAFAAIAAQD4//3/BQAFAP///P8FAAIA/P8AAAYABwACAAEAAAAAAPr/BQADAAMAAgD///3///8AAAEA/f////v//f8BAP7//v/6/wAAAAAAAP7/+f/8/wIA//8DAAQAAQAAAAYAAQD8//z/+v////3/+f/9/wEAAwADAAQABAACAAAAAQD///7/+v8CAP7//f/3//3//v8DAAEA/P///wQA/v/9/wQABAAKAAEAAAD+//7/BgD+//r//P8BAPz///8AAAAA/P/6////+/8BAPz/AAABAP///v/9//7//v8AAP7/AgD//wIA//8AAAEA/P8AAAAA+v/8//z///8AAP3/+v/8//7//P8AAP3/BgD/////AwAHAAIAAgD5//r///8BAP3/+/8DAAIABwACAAIA/v///wAA+P8BAAAAAwAAAP//BgD8//7//f8EAP//9//5/wcA//8EAAQA/f/7//7//P///wIABQABAAAA/v/9////AAADAAQAAQD///7/AQACAP3//f/6//3/AQAHAP7//f/9//v/BAD/////AQAAAP7//P8DAAMAAAAEAAEAAQD8/////P/+//z//P/8/////P/9/wMAAgAHAAMAAQD8//v//P/6//z//v8GAAIA/v/+/wMA/v8BAAEA/P/9/wQA/f/+//7/BQD+/wIA+v/9//7//v///wAA/v/4//7//v8GAPv/AwD+/wAAAwADAP3/AAADAPz/BQADAP///P/5//3/+f8FAAMAAwAAAAUABQAHAP3/AAAAAAIAAwABAPv//f/7//r/AAD///r/+//7//r/BAABAAAAAAABAP7/+//9/wMAAAD+/wAAAwAEAAIA/P/7//7/AwAFAP7/AQD6//7/AwAFAPr/AAD+//z/AgADAP3//////wEA/v////z//f///wEABAAKAP7/AAD3//3/AgADAPr/AAAAAAAA//////3/AAAGAAIA/f/4//z//f/+//7/+//6//v/+/8CAP///f/3/wIA/v/9//7//f8CAAAA/f/7//v/AQD7//r//v8DAAIA+//8//z////6//7/AwACAAEAAgACAAMABgACAAAA+////wEA/f///wEAAgD//wAA/P////v////7//z/BAAGAP7/AwD+////AgADAP7/AAD+//3///8AAP7/AAADAAIAAQAAAP7/BAABAAIAAAAAAAAAAwADAP//AQD9/wMA+v8CAAIA/////wIABAAAAAQAAwAKAAMAAwD9////BQADAP7//f/9//3/AAACAP//AAD//////P/3//z//f8BAAAA+////wEAAwABAAIAAQAIAAAAAgD+/wAABgD///z/AQALAAMA/P/9/wIAAQADAAMA/f/5/wMAAQAGAP3/+P////3/AAD+/wIAAgD8//3/AAABAP//AgAFAAIA+//7/wAA/P8AAAUABQAIAAMA///9//////8CAAAA/////wIA///+//3/+//4/////P8EAAEA/v/7/wQAAQD///z/+f/+//3/AQD///7///////v//f8AAAEA//8BAAEA/f8DAAIA+f/6//3/AgD//wAA///7/wAA+/8CAAEAAgABAAIAAgD6//v/AAD//wAA/P8FAAUA/P/4//7//P/9////AwAJAAIA/f/4////AQD///v/AAABAPr/AAD+/wEA/v8CAAMA/f8BAAIAAQAAAAEAAAD9//7//P8DAAMAAwAAAAQA/P/+//7/AAADAAEA///3//3/AQAFAP3/AAD9//z/BgAGAP3/AwAAAP7//f8AAP3//v///wAABgAEAAEA///+//7///8BAAAA+v/6/wAA//8AAP//AwAEAP//AQD+/wEA///6//3//f8FAP//BAAAAAAABAACAAAA//8DAP///f///wEAAgAFAAQAAAD+/wIA//8AAAAA//8AAP3/+///////AQD6//3/AAD//////P8EAAEA/f/8/wEABAABAP//AgACAAMAAwABAP3///8BAAAA/f8EAAIACAABAP7////+//n///8CAP//AgD9/wAAAAAFAAUA/f/+/wUA/P/+/////v/8/wAA/P/+//3/+//8/wMAAgADAAAA/P/+/wMA/v8DAAMA///8////AAAFAAAA///+/wEAAQABAAEAAwAFAAIA///3//v///8DAP//AAD7//v/AQADAPv//f8BAAAAAQABAP3//f/9/wIAAQABAP//AAABAAIAAwAFAAEABAADAP//BgACAAIAAAAEAAEA9v/4/wAAAQAEAAUA/v/7/wAAAAAEAAEA//8AAAQA/f////7////3/////f8CAAAA+/8BAAMAAQD7////AgAGAAEA//8AAAAAAAD8//7/+v8BAP//+//6/wEA/v8AAAIAAwAEAAEAAQD9//3/BAABAAAAAQD+//z//v8HAAQA/P/8/wEAAQAEAP3/AgD//wAABAACAPz/AAADAP7/BAD8//7/AgABAAAAAwAIAAIA/P/6////AQACAAAAAAAAAAAABAAIAAEAAgD6//z/AQABAP//AgAAAP3//f/+//v/+v/8////AAACAAAA//8CAAEA/P/+////AgD//wEAAgAEAAAA/P/5//z/BAD+//v/+//+/wIAAgAEAAAAAwD+//v//v/9//3///8DAP//+//9/wEAAQD+/wAA/f8AAP3///8CAAEABgD7////+/8EAAMA+P/7/wMA/f/+/wEA/v8AAAIABgABAP7///////v//f////3/+//5//7//P8AAAAA/v/9/wIAAAABAPz//f/8//7/AAD+/wAA//////////8DAAYABAAEAAEA//8CAAIA/f8AAPv//v/5//3/BgAEAPz/AAD/////BAAFAAEA/f/7//n/+P8AAPz//v/6/wAABAAAAPj//v8AAP//AgABAAAA+v/6//v/+/8AAPz/BAADAAIABAABAP//AAD//wEAAQAEAAQAAwACAPz/AgD7////AAAHAAQA9v/2/wEA/f8EAAQA/P/5/wEA/f/8/wEA+v8DAAAAAwD9//3/CAAFAP3/BAAEAP//+/8AAP//AQACAAMAAgD/////+/8CAP7//P/9/wMAAQAAAAAA/P/9/wQABQAEAP///v/7////+v///////f///wMA//8AAAAAAAD//wAA+//+/wEA//8DAAIAAQD8/wEAAQABAPr//P/9//f////7//////8CAAQA/v8FAAkAAgD8//7/AAADAPz/+f8AAAAAAAD7/wIABAAHAAMAAAD7/wMAAwAIAAUA/v/7/wIA/f8DAAIAAgAAAAUA/f8BAAUAAAAEAAAAAQD7/wEABAADAPz/AAABAPz/AAD9//z/+/8AAAMA/v8CAAAAAgD9//z/+//8//r//f8CAAAAAwD/////+//8//////8EAAEA//8DAAIAAgD2//7/AgAFAPv//f/9//7//P/2/wEAAQAHAP//AgAAAAYABwAJAPv/9P/6//j/AQD9/wEAAgAEAAMA+v8CAP//BgD9/wEAAwABAPr/AgABAP7/+P/8//z/+v8AAAMA/P/6/wMA/f8DAAAA//8AAAMA/v/+/wAAAQD4//v//P8BAAIAAgADAAQA///9//3//f8DAP7/AQADAAMABQD9/wIAAQAFAAAA+/8CAAQAAQD7/wMAAwAEAAEAAAABAP//AwAAAP7/+f/9/wEAAgAFAAMA///8//3/AgACAPj/////////BAADAAEABQD8//z/AAAFAP7//v8AAP//AwD8/////f8AAAAA/P8BAAAA/v/9//////8BAAIAAwABAP///P8AAPz/+//9/wIAAQD///3/+//8//7/AAADAP3//P/3//z/AwAHAP7/AQABAP7/AgD+//3/AAAAAP7//f/8//7/AQAAAAIAAAAAAP//AQD//wAAAAADAAMABAAAAPz/AAAAAAAAAAABAAQAAAABAAAA///7//7/AgABAPz/AgAFAAQAAQAAAAAA//8EAAAA/v/8/wIAAwAHAAgAAQAFAAYAAgABAP3/AQD7//3//P8AAP//AAD+/wQA/v/8//z///8BAPz/+f/4////BQALAAQAAgD7/wMABgAKAPn////7//n/BgAFAPv/BAD+//z/AAAFAAAAAAD9/wMABAAGAAAAAAD+//3///8BAAIA/////wIA/P/8//7/AgADAAEAAQD9////AAAFAAEAAQD//wIA//8BAAQA/P/9/wIAAAADAP7//f/7//3//f8AAAAABAD9/wAAAgD///7/+v///wAA/f///wAABAAAAAEA///+//7/AQAFAAAAAQD+//7/AgAFAAIABQABAAEA/v/+//3/AwAJAAUAAQD+/wMA+/8CAAAA///+/wIAAQD9/wAAAAD9////BAAGAAIAAQAAAAAA+//9/wAAAAD///7//f8DAAAA///9/wMA//8CAAUA//8EAAMA///+//3/BAD5//z//P8BAP//+////wEAAAD7//z//v8BAPz/AwAEAAMAAQACAAMAAAAEAAEA+f/7/wAA//8CAAIAAQD6/wQA/f8GAAMABAADAAMA+//+/////P/+////AgD//wAAAwAEAP7/+v/7//z/BgADAAEA/f/5//v//f8CAAAA//8AAAAAAAAAAP//AQD9/wMAAQAIAAIAAAD///v/AwD8//z/+/8FAAEA/f/8/wMAAgABAAEA/v/9//7/AQABAPz/AgAEAAEA/v/6//3/AAADAP3/+//7/wAABQAKAPz/AAD5//z/AQAEAPn//f/9/wAACAAIAAEAAgD9/wEAAgACAAAAAAAEAAEA/f/4//3/+v8FAAIA+f/5/wMA+v/9/wAAAwAAAAIAAwD7//z/BgAGAAAAAAD7//z/AAD///3//f/9/////P8FAP7/AAABAAIA/f/6//3/AAADAP7/BAD9/wIA/v8EAAQA//8AAAUAAwD+//3/+P/8//j//f/+/wAAAgD8/wMA//8FAAYA/f8DAAAAAAD9//3/AQACAP3/AQD//wQABAACAPz///////3/BAACAAIAAAD7/wAAAAACAP//AAACAP7/AQD///7/+//1//z/AgALAAAAAAD4/wIABQAHAP7/AAD+//r/AgABAPz//v///wAA///+/wAAAAACAP//AAD+/wMA/f/+/wEA/f8GAAIAAgD7/wEABwACAP3/BgAIAP3/BAD6/wAA//8BAAEAAAAIAAYAAAD4/////f8AAPv////+/wEAAgAFAPn/AAD4//v/BgABAPz/AAAFAAMAAgAFAAEAAQD//wAA/P/8//3//v8FAAAA/P/4//3//f/+//3//f/+/wQAAQABAP///f/7////BQAHAAAA+//6/wAAAQAIAAgA/P///wAA/v8AAAEAAAD5//3//P8EAP7/AgD8/wIA/P/6//z//v8BAAAA/v/5//3/AAACAP//AAD9//3/AQADAP7/AAD+//7//v//////AAADAP///f/7//7//v8DAAYA/P/9/wIAAAD///3/BAADAP3//v8AAP//AgD9//3/AQD9//v/+f8BAAMAAAD+//7/BAAEAP///v/5//3/AQD///z//v////3//P8AAP//AgACAAIAAAADAAAAAQACAAUABwACAP7/AAADAAIAAAD8//7/AQD+//7/+//6////AQAEAAMA/v/+////AwACAAIAAQD//wIAAAAEAP3/+//6/wUA//8BAAUAAQACAAYA////////BQD//wAA/v/+//7/+P8AAAAABQD//wQAAQAAAPz///8EAP///f/8//3/AAADAAEA/P/5/wEAAAABAAQA/v8AAAMABAAEAAIA///+//7/AAABAPz//f///wIAAAABAAMABQAAAAMAAgABAAEA/v/9/wIA/v/9//7//P/+/wAAAgD+//r/AAD///r//f/8/wIA//8IAAMABAD+//3/AwACAP3/AQAFAP7//v/5//7/+/8DAAIA+v/+/wEAAwD+////+/8BAAAAAQAAAP7/AAD8/////f8DAP///P///wMA+//+/wAA+v/8/wEABAAGAP///v/9/wQAAAD+//3////9/wAA/f8AAAAA/v8DAAQA/f/7//7/AwAFAP3//v/+////BgD///z/+//+//3//f8EAAIABAD9//v/AwD9//n/+v/+////AAAFAAQA/P/4/wEA//8GAP////8CAAMAAgD+//v///8BAAEA///7//7//v8BAP//AgD//wQAAAAFAAEA/P/8/wIA/f8BAAMA//8AAP3/AQD//wEABAAEAP//AQD9//7/BAAEAPz////+//7/BQAGAP//AgD6/////v8CAPv/AgABAAUABAAHAP//AQD+//n/AQD5//3//P8FAAMA+v/9/wQAAAD5/////f/+//z//P8BAAAAAQADAAUABQAAAP7/+//8//n/AwAFAP///v/6/wAAAAACAAQA+//9////+v/5//v//v/8//7/AgABAAEA/P8BAP///f/8//7/AwD///z//v/+/////f8CAP7//v///wIA///+/wAABAADAP//AgABAP//AAACAP7//f/7/wEA//8AAAAA/P///wQA/v8BAAUABgAEAAIA/v8BAP//AAACAP7/AgD9///////+//7///8BAP//AgD//wAA+v8BAP//+//5/wEA/v////7/+//9/wMAAAABAAEAAQD6/wMA+v8BAAAAAAACAAAA+v/5//3//v8AAP7/AQD+//3/BQAHAP3//f/4//n/AwACAP3//f////3///8CAP3//v/5//////8FAP7/AQAAAAIABQD8//z/+/8FAPz//v/6/wMA/f8BAAIA//8CAAEAAQD//wQABQADAP3/AQAAAP3/+//+/wAAAAADAAIA/v/9//3/AAABAAEAAwADAAAAAAAAAP3////9/wAAAQADAAIA/P/9/////f8BAP//+//4////+v/9//3////9/wEA/P///wQAAAAFAAMAAAD7//v////+//7/AAACAP//AQAAAAIAAAABAAEA/P/+/wEA/P/+//z/BQD//wAA/P///wQAAgAEAAEAAAD6//////8HAPz/BQABAAIACQAAAPv//f/8//v///8CAAAA/v///////f8CAAEAAwD//wEAAwAJAP3/AwD9/wEAAwAFAPz/AAD8//z/AAD+//r/+/////3/AgACAAMAAgD///3//f/9/wEA/v/9////BAAFAAEA+//6////AQACAPz/AQD9//7//v/+//v/AAACAP3/AQD//////f///wEA/f8AAAEAAgD9/wEAAAAKAPr/AAD7////AwD+//r/AgAFAAEA+v/8//7/AAADAP///f/0//7//f8EAAAA/P/+/wEA/v/+/wEA/v/+/wAA/v///wMAAQACAP///v/7//n/AwD//wEAAAADAAAA/f///wAA/v/8/wEAAQAGAP//BAABAAIAAgD+//7//v8EAAEA///7//7//f/9/wAA/v8BAP7/AwD9////AgACAPv///8CAAAAAgAEAAYAAgD+/wEAAQAEAP7/AgD///7/BAAAAPv/AwAEAAIAAQABAP7/AAD+//3/AwABAAMA/v8DAAMAAAADAAIABAABAAEAAQD///3//f/9//3//v/+/wEA/v/+/wEABAAEAAEA/v/7//v/+f/7//3///////////8AAP//AQABAAIAAQAGAAEAAwD9////AwAAAPz/AAAEAAIAAAABAP//AAAAAP7////9////AwAGAAAA+v8AAP3/AQADAAIAAQD1//v/+v8DAAEA//8AAAUA/P/+/wEA/////wMAAwD////////9/////v8FAAQAAwADAAMA///8//z/+f/7////AQAGAAIAAgD8/wMAAgADAAAA+/8AAAMAAwAAAAEAAQD+//7//P8AAP3/+//+/wAA/v/+/wEA/P8AAPn//v/6//7/BQACAP3/+///////AgACAAAAAQD7//v///////v///8BAP///v/8/wAA/v8DAAQAAwAGAAMA+v/4//z/AAD8//3///8AAPv//f/7//7/AAD//wAA/v8EAAIA/////wMAAQD+//v//P8AAAEAAgD//wIA+/8AAP7/AQAAAAEA///6//z///8EAP//AgD9//7/BAADAPr/AwD///7//f8BAP3///8AAAEAAAAAAAAA//8DAP///v/8/wAA/P8BAAQA//8AAAIABAAEAP7/AQD8//7////9//7//v8CAP7/AQD+/wEAAwAEAP//AAD///////8FAAEABQAFAAMAAQD6//z///8DAP7//f/6/wAA+////wIAAwABAAQAAQD//wAA/v8EAP///f/+/wMABQAAAP//BAACAAIABgAFAP//AQACAAMAAwAFAAEABgABAP7/+//+//z/AQABAAIAAwD9/wAA/v8EAAAA///+/wIA+//9/wEAAQD//wMA//8AAP3//f/+/wMAAgAGAAAA/f/8/wIA//8FAAQA+//6/wEA/v8DAAIA/v/+/wMAAQAAAP//AwACAP/////7//////8EAP3////6//3///8CAPr//f8DAP//AwD9/////f///wAA/f8BAAEA/////wIABwAEAP//AwAAAP7/AgAEAP7////+//7/+f/9//3/AgAEAAIA/f/6//////8EAAIAAAAAAAUA/v8CAAIA/f/5/wEA/f8AAAAA/v8EAAMAAQD8////BgADAP////8DAP7//P/9/wIA+/8CAAMA/v/+/wMA/v///wEABwADAP///f/9//z/AwACAAEABAABAAAAAAAEAAEA/f/9/wMAAQAFAP7/AwABAAAAAgABAP3/AAAEAP7/AgD5//3/AQABAAAA/v8DAAAA/f/6/wEAAwAEAP//AAAAAP3/AgACAP///f/7//3//f8AAAEAAAD7//7/AQAAAP3//P8AAP//AQAAAAAA/P///wEA/v8DAAAABAD+//7/AwADAPv/+//6//v/AwADAPz//P/7/wEABAAHAP//AwD8//r/AwAAAPz/AQACAP7//P/+/wEAAAD9/////v///wAAAAACAAIABQD8/wEA/v8AAAAA+//8/wEA/v///////v/+/wAAAgD//wIA/v8BAPv//v/9//3/AQD///3//f/9/wIAAQABAAAAAgD///r//f/6//7/AQABAAAAAAD///z//P///wMAAAADAP7//////wEA/v8DAPv////9//z/CQAAAPz///8BAAAA//8EAAMAAQD9////+v/9//v/AAAAAAIABgAFAP/////9/wAA/v8CAAAA/f/7//j//v/7//r/AAAEAAQAAgADAAAAAQD8//7//P8BAAMAAwAGAP//AwD5//3///8FAAEA9//2/wUA/f8FAAYA/f/8/wEAAAD//wIA/f/9////////////AgACAP7/AQABAP7/AQACAP//AgD//wMAAAD+/wEA/f8EAP7//f/+/wIAAAD8/wAA/v/+////AwACAAEAAwAAAAMAAQACAAMA//8BAP///v////7/AwABAP//AAAAAAAAAQADAAAA/v/8////AQABAP7//v////7/AAD9////+/8AAP3/+//9/wEAAQD+//7//v////z///8CAAAAAAD7/wAAAgABAAEA/f/9/wIAAQAEAAEAAQD7/wAA/v8BAP//+//9/wAA+//7/wEAAAAAAP///v////7/AwD///3//f/+/wAAAgACAAEAAAD8//7/AAACAPz/AAD9/////v8EAP//AwABAAEABQD///7//v8BAP//AAADAAIAAAAAAAAA/P/4////AgAIAAMAAgABAP7//P/8/wAA///9//////8FAAYAAgAAAAEA+v////v//f/5/wQAAQABAAIA/f8AAP///v///wEAAgD///7//////wIA/f8CAAEA/v8AAAAAAQD+//////8DAAEA///+/wIA///9//3/AgD9//3//P8AAAEABAABAAEA+//7//3/+P8BAPr/AQD+//7/AgD9//7/AAABAPz/+v///////P/7/wIA///+//3//v//////AAABAAAA/v///wAA/v/8//////8EAAAAAAABAP//AAD8//7/AgADAP7/BAD+//z/AgADAPz//v/8//7////9//7//v8BAAAA/v8BAAAA/v/+/wAA////////AwD//wEAAAADAAAA///8/wIAAAACAP7///////7//P/6//v//v8BAAEAAgADAAIAAQABAAAAAQD+/wAA/v8BAAEA/f8AAP//AgD9////BQACAPv/AAD+/wAAAQABAP3//v/7//r///8CAP7//v8AAAIAAgAAAP3/AQD+/////v8CAP7/AwABAAAA///9//z//v8HAAUA/f/9/wAA///+//////8CAAIA/v/9//7/AgD9/////v////////8EAAMABAAAAP//AgD9//v//f8AAP3/AAACAP///v///wQAAQACAP//AAD//////v8CAP7/AgABAP7/AgD9/wMAAAABAAEA//8AAP///////wEAAAD9//////8DAAIA/P8BAAEAAQABAAIAAAD9/wEAAwAFAAQABAAAAAEA/f8BAAEAAgD+/wEA//8AAP7///////z/AQD///7///8BAP//BAACAAAA+////////v////7/AAD7/wEA/v8BAAEA//8BAAMA///9////+v/8//3///8CAP//AQABAAAA//8AAAMAAwAHAAIAAAD8//7///8AAPv/AAD///3/AQD9//v//v8EAAEAAQD//wMAAwACAAEA//8EAAEA/v8AAAIAAAD6/wEAAAADAP//+P///wUAAwD//wMABgD//wAAAAAEAP/////7////+v/8//7//f8FAAUAAgAAAAUAAgAGAAAAAgABAP3/AAD8//z//////wIAAwAGAAIAAgAAAAEABAAAAP///f8DAP///v/+/wAA/P/+/wAABQD+/wEA/f8EAAIA/v8CAAIA///4//z//f////v/AAAAAPz/BQD//////P8AAAUAAAAEAAUAAAD8//7//P///wEAAwABAAAABAACAPz//f8AAP7//v/+////AwD+/wAA/f8CAPn/AAD8//r//P/8//3/+v8AAP7/+v/9//7////+/wIAAQD+////AAADAP7/AAD//wEA/v///wAA/f/7////AAAFAAIAAQAAAAMA//8AAAMAAAACAAEA/v8AAAIABgACAP///v////z/AAD6//z//v8AAP///v8BAAIAAQADAAQA/f8AAP3/AQABAP///v/9//3//f////3/AAAAAP///v/7//3/AAD//wAA+//9//7/+f/8//3/AQD+/////v/+//3///8CAP///P/8/wAA/f8AAAIA///7/wAA//8CAP7/+//9/wAAAQAAAAEAAgAAAAAA/v8CAAIAAQAAAAEAAgABAAQA/v8BAAAAAQAAAAAAAQABAAEAAAAEAAIAAgD+/wAA/v////7//f/8//7//f/9/wQAAgAFAP///P/5//7/AgADAP///v/+////AQAFAP3////9////BQADAAIABAACAAAA/v8BAAEAAgACAAAAAAD+/wAA//8CAAIA/v8AAP7///8AAP///f/+//////////3//P/7//z//f////z//f/5/wAA//8CAAIA+////wQAAgABAAAA///6//z/+f/9/wIAAQADAAIAAwAAAP7////7//3//f8EAAEA/v8AAAAAAAD//wUA//8DAP3/AAD+/wEAAQABAPz//P8BAPv/AQD6//3/AAAAAP7//v/////////+////AQD+/wEA/v8AAAAA+v8AAAEA/v8AAAAAAQD9////AQACAAAAAAADAAEAAgD//wEAAgABAAEAAAACAAAA/f/9/wIA/f8AAAEA//8AAAIA//8AAAEA///9//////8BAP7////7/////v////7//P/8/wAAAAD//wIA/P////7/AwACAAAA//8AAP///v8DAP///f/6//7///8CAAEA/P/8/wEA/v/////////+/wAAAgACAAEAAgD//////v8CAAMAAAABAAEAAwADAAEA/v/8/wAAAAAEAAMAAQD//wEAAAD+////AAD//wIAAwAEAAEA/v/+/wEA//8AAAIA///+//7/+/8BAP//AQD+//z//v///////v8CAP//AgAAAAAAAQABAAMAAQADAAIAAAD//wAAAAABAAIAAgADAAEA/v/8//z//f/7//z//f////v/+v/8//7//P/8//3//v/9//7//f/7//v//f8AAAAA//8AAAEAAQACAAEAAQABAP////////3//P8AAAAAAwADAAIABAACAAIAAwACAAEA//8BAP7/AAD9////AQABAAMA/v8AAP7//P/8//3/AAD+//7//f////////8AAP7/AAD9//7/AgAAAP7///8CAAAA//8AAP//AAAAAAAA/v8AAAIAAQAAAP///v/6//3//f//////AQD//wIAAQD///////8AAP7////+/wAA///9//7//P/+/////v///wIA/v/+/wEAAQD8//3//////wAA///////////9////AAABAAEA///+/wAA////////AAABAAIAAAABAP7//P/8//z//f/5//j/+//5//r//P/7//z//P/8//7//v///wEAAgACAAAA///9/wAA/f///wEA/f/9//3//f/9//v//f/6//z////8//7//P/9////AgD//wAAAAAAAAEAAgAAAAIAAAADAPz////8//3//v8BAAAA///+//z/AQD///r/AQAAAAEAAgAKAAUACgAFAAwADgANAAoACQANAAoAAgAFAAcACQALAA0ABwAWABcADAAVABgAGwAFAPn/8v/z/yAA//8AAAwAAwANAOb/2v/J/87/yP/d/7v/BgD6/6n/",
                        expires_at=1729286252,
                        transcript="Yes.",
                    ),
                ),
            )
        ],
        created=1729282652,
        model="gpt-4o-audio-preview-2024-10-01",
        object="chat.completion",
        system_fingerprint="fp_4eafc16e9d",
        usage=usage_object,
        service_tier=None,
    )

    cost = completion_cost(completion, model="gpt-4o-audio-preview-2024-10-01")

    model_info = litellm.get_model_info("gpt-4o-audio-preview-2024-10-01")
    print(f"model_info: {model_info}")
    ## input cost

    input_audio_cost = (
        model_info["input_cost_per_audio_token"]
        * usage_object.prompt_tokens_details.audio_tokens
    )
    input_text_cost = (
        model_info["input_cost_per_token"]
        * usage_object.prompt_tokens_details.text_tokens
    )

    total_input_cost = input_audio_cost + input_text_cost

    ## output cost

    output_audio_cost = (
        model_info["output_cost_per_audio_token"]
        * usage_object.completion_tokens_details.audio_tokens
    )
    output_text_cost = (
        model_info["output_cost_per_token"]
        * usage_object.completion_tokens_details.text_tokens
    )

    total_output_cost = output_audio_cost + output_text_cost

    assert round(cost, 2) == round(total_input_cost + total_output_cost, 2)


@pytest.mark.parametrize(
    "response_model, custom_llm_provider",
    [
        ("azure_ai/Meta-Llama-3.1-70B-Instruct", "azure_ai"),
        ("anthropic.claude-3-5-sonnet-20240620-v1:0", "bedrock"),
    ],
)
def test_completion_cost_model_response_cost(response_model, custom_llm_provider):
    """
    Relevant issue: https://github.com/BerriAI/litellm/issues/6310
    """
    from litellm import ModelResponse

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    litellm.set_verbose = True
    response = {
        "id": "cmpl-55db75e0b05344058b0bd8ee4e00bf84",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": 'Here\'s one:\n\nWhy did the Linux kernel go to therapy?\n\nBecause it had a lot of "core" issues!\n\nHope that one made you laugh!',
                    "refusal": None,
                    "role": "assistant",
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [],
                },
            }
        ],
        "created": 1729243714,
        "model": response_model,
        "object": "chat.completion",
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 32,
            "prompt_tokens": 16,
            "total_tokens": 48,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    }

    model_response = ModelResponse(**response)
    cost = completion_cost(model_response, custom_llm_provider=custom_llm_provider)

    assert cost > 0


def test_completion_cost_azure_tts():
    from unittest.mock import MagicMock

    args = {
        "response_object": MagicMock,
        "model": "tts-1",
        "cache_hit": None,
        "custom_llm_provider": "azure",
        "base_model": None,
        "call_type": "aspeech",
        "optional_params": {},
        "custom_pricing": False,
    }
    litellm.response_cost_calculator(**args)


def test_select_model_name_for_cost_calc():
    from litellm.cost_calculator import _select_model_name_for_cost_calc
    from litellm.types.utils import ModelResponse, Choices, Usage, Message

    args = {
        "model": "Mistral-large-nmefg",
        "completion_response": ModelResponse(
            id="127f24aed4984b4c9a4c5e32ad3752f3",
            created=1734406048,
            model="azure_ai/mistral-large",
            object="chat.completion",
            system_fingerprint=None,
            choices=[
                Choices(
                    finish_reason="length",
                    index=0,
                    message=Message(
                        content="I'm an artificial intelligence and do not have an LLM (Master",
                        role="assistant",
                        tool_calls=None,
                        function_call=None,
                    ),
                )
            ],
            usage=Usage(
                completion_tokens=15,
                prompt_tokens=8,
                total_tokens=23,
                completion_tokens_details=None,
                prompt_tokens_details=None,
            ),
            service_tier=None,
        ),
        "base_model": None,
        "custom_pricing": None,
    }

    return_model = _select_model_name_for_cost_calc(**args)
    assert return_model == "azure_ai/mistral-large"


def test_moderations():
    from litellm import moderation

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.add_known_models()

    assert "omni-moderation-latest" in litellm.model_cost
    print(
        f"litellm.model_cost['omni-moderation-latest']: {litellm.model_cost['omni-moderation-latest']}"
    )
    assert "omni-moderation-latest" in litellm.open_ai_chat_completion_models

    response = moderation("I am a bad person", model="omni-moderation-latest")
    cost = completion_cost(response, model="omni-moderation-latest")
    assert cost == 0


def test_cost_calculator_azure_embedding():
    from litellm.cost_calculator import response_cost_calculator
    from litellm.types.utils import EmbeddingResponse, Usage

    kwargs = {
        "response_object": EmbeddingResponse(
            model="text-embedding-3-small",
            data=[{"embedding": [1, 2, 3]}],
            usage=Usage(prompt_tokens=10, completion_tokens=10),
        ),
        "model": "text-embedding-3-small",
        "cache_hit": None,
        "custom_llm_provider": None,
        "base_model": "azure/text-embedding-3-small",
        "call_type": "aembedding",
        "optional_params": {},
        "custom_pricing": False,
        "prompt": "Hello, world!",
    }

    try:
        response_cost_calculator(**kwargs)
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error: {e}")


def test_add_known_models():
    litellm.add_known_models()
    assert (
        "bedrock/us-west-1/meta.llama3-70b-instruct-v1:0" not in litellm.bedrock_models
    )


@pytest.mark.skip(reason="flaky test")
def test_bedrock_cost_calc_with_region():
    from litellm import completion

    from litellm import ModelResponse

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    litellm.add_known_models()

    hidden_params = {
        "custom_llm_provider": "bedrock",
        "region_name": "us-east-1",
        "optional_params": {},
        "litellm_call_id": "cf371a5d-679b-410f-b862-8084676d6d59",
        "model_id": None,
        "api_base": None,
        "response_cost": 0.0005639999999999999,
        "additional_headers": {},
    }

    litellm.set_verbose = True

    bedrock_models = litellm.bedrock_models + litellm.bedrock_converse_models

    for model in bedrock_models:
        if litellm.model_cost[model]["mode"] == "chat":
            response = {
                "id": "cmpl-55db75e0b05344058b0bd8ee4e00bf84",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "logprobs": None,
                        "message": {
                            "content": 'Here\'s one:\n\nWhy did the Linux kernel go to therapy?\n\nBecause it had a lot of "core" issues!\n\nHope that one made you laugh!',
                            "refusal": None,
                            "role": "assistant",
                            "audio": None,
                            "function_call": None,
                            "tool_calls": [],
                        },
                    }
                ],
                "created": 1729243714,
                "model": model,
                "object": "chat.completion",
                "service_tier": None,
                "system_fingerprint": None,
                "usage": {
                    "completion_tokens": 32,
                    "prompt_tokens": 16,
                    "total_tokens": 48,
                    "completion_tokens_details": None,
                    "prompt_tokens_details": None,
                },
            }

            model_response = ModelResponse(**response)
            model_response._hidden_params = hidden_params
            cost = completion_cost(model_response, custom_llm_provider="bedrock")

            assert cost > 0


# @pytest.mark.parametrize(
#     "base_model_arg", [
#         {"base_model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"},
#         {"model_info": "anthropic.claude-3-sonnet-20240229-v1:0"},
#     ]
# )
def test_cost_calculator_with_base_model():
    resp = litellm.completion(
        model="bedrock/random-model",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        base_model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        mock_response="Hello, how are you?",
    )
    assert resp.model == "random-model"
    assert resp._hidden_params["response_cost"] > 0


@pytest.fixture
def model_item():
    return {
        "model_name": "random-model",
        "litellm_params": {
            "model": "openai/my-fake-model",
            "api_key": "my-fake-key",
            "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
        },
        "model_info": {},
    }


@pytest.mark.parametrize("base_model_arg", ["litellm_param", "model_info"])
def test_cost_calculator_with_base_model_with_router(base_model_arg, model_item):
    from litellm import Router


@pytest.mark.parametrize("base_model_arg", ["litellm_param", "model_info"])
def test_cost_calculator_with_base_model_with_router(base_model_arg):
    from litellm import Router

    model_item = {
        "model_name": "random-model",
        "litellm_params": {
            "model": "bedrock/random-model",
        },
    }

    if base_model_arg == "litellm_param":
        model_item["litellm_params"][
            "base_model"
        ] = "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
    elif base_model_arg == "model_info":
        model_item["model_info"] = {
            "base_model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        }

    router = Router(model_list=[model_item])
    resp = router.completion(
        model="random-model",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="Hello, how are you?",
    )
    assert resp.model == "random-model"
    assert resp._hidden_params["response_cost"] > 0


@pytest.mark.parametrize("base_model_arg", ["litellm_param", "model_info"])
def test_cost_calculator_with_base_model_with_router_embedding(base_model_arg):
    from litellm import Router

    litellm._turn_on_debug()

    model_item = {
        "model_name": "random-model",
        "litellm_params": {
            "model": "bedrock/random-model",
        },
    }

    if base_model_arg == "litellm_param":
        model_item["litellm_params"]["base_model"] = "cohere.embed-english-v3"
    elif base_model_arg == "model_info":
        model_item["model_info"] = {
            "base_model": "cohere.embed-english-v3",
        }

    router = Router(model_list=[model_item])
    resp = router.embedding(
        model="random-model",
        input="Hello, how are you?",
        mock_response=[1, 2, 3],
    )
    assert resp.model == "random-model"
    assert resp._hidden_params["response_cost"] > 0


def test_cost_calculator_with_custom_pricing():
    resp = litellm.completion(
        model="bedrock/random-model",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="Hello, how are you?",
        input_cost_per_token=0.0000008,
        output_cost_per_token=0.0000032,
    )
    assert resp.model == "random-model"
    assert resp._hidden_params["response_cost"] > 0


@pytest.mark.parametrize(
    "custom_pricing",
    [
        "litellm_params",
        "model_info",
    ],
)
@pytest.mark.asyncio
async def test_cost_calculator_with_custom_pricing_router(model_item, custom_pricing):
    from litellm import Router

    if custom_pricing == "litellm_params":
        model_item["litellm_params"]["input_cost_per_token"] = 0.0000008
        model_item["litellm_params"]["output_cost_per_token"] = 0.0000032
    elif custom_pricing == "model_info":
        model_item["model_info"]["input_cost_per_token"] = 0.0000008
        model_item["model_info"]["output_cost_per_token"] = 0.0000032

    router = Router(model_list=[model_item])
    resp = await router.acompletion(
        model="random-model",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="Hello, how are you?",
    )
    # assert resp.model == "random-model"
    assert resp._hidden_params["response_cost"] > 0


def test_json_valid_model_cost_map():
    import json

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

    model_cost = litellm.get_model_cost_map(url="")

    try:
        # Attempt to serialize and deserialize the JSON
        json_str = json.dumps(model_cost)
        json.loads(json_str)
    except json.JSONDecodeError as e:
        assert False, f"Invalid JSON format: {str(e)}"


def test_batch_cost_calculator():

    args = {
        "completion_response": {
            "choices": [
                {
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "protected_material_code": {
                            "filtered": False,
                            "detected": False,
                        },
                        "protected_material_text": {
                            "filtered": False,
                            "detected": False,
                        },
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    },
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": 'As of my last update in October 2023, there are eight recognized planets in the solar system. They are:\n\n1. **Mercury** - The closest planet to the Sun, known for its extreme temperature fluctuations.\n2. **Venus** - Similar in size to Earth but with a thick atmosphere rich in carbon dioxide, leading to a greenhouse effect that makes it the hottest planet.\n3. **Earth** - The only planet known to support life, with a diverse environment and liquid water.\n4. **Mars** - Known as the Red Planet, it has the largest volcano and canyon in the solar system and features signs of past water.\n5. **Jupiter** - The largest planet in the solar system, known for its Great Red Spot and numerous moons.\n6. **Saturn** - Famous for its stunning rings, it is a gas giant also known for its extensive moon system.\n7. **Uranus** - An ice giant with a unique tilt, it rotates on its side and has a blue color due to methane in its atmosphere.\n8. **Neptune** - Another ice giant, known for its deep blue color and strong winds, it is the farthest planet from the Sun.\n\nPluto was previously classified as the ninth planet but was reclassified as a "dwarf planet" in 2006 by the International Astronomical Union.',
                        "refusal": None,
                        "role": "assistant",
                    },
                }
            ],
            "created": 1741135408,
            "id": "chatcmpl-B7X96teepFM4ILP7cm4Ga62eRuV8p",
            "model": "gpt-4o-mini-2024-07-18",
            "object": "chat.completion",
            "prompt_filter_results": [
                {
                    "prompt_index": 0,
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "jailbreak": {"filtered": False, "detected": False},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    },
                }
            ],
            "system_fingerprint": "fp_b705f0c291",
            "usage": {
                "completion_tokens": 278,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens": 20,
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
                "total_tokens": 298,
            },
        },
        "model": None,
    }

    cost = completion_cost(**args)
    assert cost > 0
