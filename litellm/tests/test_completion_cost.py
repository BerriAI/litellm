import os
import sys
import traceback

import litellm.cost_calculator

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import time
from typing import Optional

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
        model_name="qwen/Qwen2-72B-Instruct"
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



@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_completion_cost_hidden_params(sync_mode):
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

