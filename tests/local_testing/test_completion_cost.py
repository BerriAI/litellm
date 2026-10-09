import asyncio
import json
import os
import time
import traceback
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

import litellm
import litellm.cost_calculator
from litellm import (
    completion_cost,
)
from litellm.litellm_core_utils.litellm_logging import CustomLogger
from litellm.llms.custom_httpx.http_handler import HTTPHandler


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




    # print(results)


# test_get_gpt3_tokens()


# test_get_palm_tokens()


# test_zephyr_hf_tokens()




# test_cost_ft_gpt_35()




# test_cost_azure_gpt_35()


# test_cost_azure_embedding()


























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






def test_vertex_ai_llama_predict_cost():
    model = "meta/llama3-405b-instruct-maas"
    messages = [{"role": "user", "content": "Hey, hows it going???"}]
    custom_llm_provider = "vertex_ai"
    predictive_cost = completion_cost(
        model=model, messages=messages, custom_llm_provider=custom_llm_provider
    )

    assert predictive_cost == 0








def test_completion_cost_azure_common_deployment_name():
    from litellm.utils import (
        CallTypes,
        Choices,
        Message,
        ModelResponse,
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
    "model",
    [
        "fireworks_ai/accounts/fireworks/models/deepseek-v3p1",
    ],
)
def test_completion_cost_fireworks_ai(model):
    """
    Mocked so it does not depend on Fireworks' rotating serverless catalog.
    Validates the Fireworks cost path: a parsed response with usage yields a
    non-zero cost against the local cost map.
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    mock_response_data = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model.split("fireworks_ai/")[-1],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Going great, thanks!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = mock_response_data
    mock_response.text = json.dumps(mock_response_data)

    sync_handler = HTTPHandler()
    messages = [{"role": "user", "content": "Hey, how's it going?"}]

    with patch.object(HTTPHandler, "post", return_value=mock_response):
        resp = litellm.completion(model=model, messages=messages, client=sync_handler)

    cost = completion_cost(completion_response=resp)
    assert cost > 0


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




# @pytest.mark.parametrize(
#     "base_model_arg", [
#         {"base_model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"},
#         {"model_info": "anthropic.claude-3-sonnet-20240229-v1:0"},
#     ]
# )
