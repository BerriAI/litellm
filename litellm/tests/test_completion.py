import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import embedding, completion, completion_cost, Timeout
from litellm import RateLimitError

# litellm.num_retries = 3
litellm.cache = None
litellm.success_callback = []
user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]


def logger_fn(user_model_dict):
    print(f"user_model_dict: {user_model_dict}")


@pytest.fixture(autouse=True)
def reset_callbacks():
    print("\npytest fixture - resetting callbacks")
    litellm.success_callback = []
    litellm._async_success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []


def test_completion_custom_provider_model_name():
    try:
        litellm.cache = None
        response = completion(
            model="together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
            logger_fn=logger_fn,
        )
        # Add any assertions here to check the,response
        print(response)
        print(response["choices"][0]["finish_reason"])
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_custom_provider_model_name()


def test_completion_claude():
    litellm.set_verbose = True
    litellm.cache = None
    litellm.AnthropicTextConfig(max_tokens_to_sample=200, metadata={"user_id": "1224"})
    messages = [
        {
            "role": "system",
            "content": """You are an upbeat, enthusiastic personal fitness coach named Sam. Sam is passionate about helping clients get fit and lead healthier lifestyles. You write in an encouraging and friendly tone and always try to guide your clients toward better fitness goals. If the user asks you something unrelated to fitness, either bring the topic back to fitness, or say that you cannot answer.""",
        },
        {"content": user_message, "role": "user"},
    ]
    try:
        # test without max tokens
        response = completion(
            model="claude-instant-1", messages=messages, request_timeout=10
        )
        # Add any assertions here to check response args
        print(response)
        print(response.usage)
        print(response.usage.completion_tokens)
        print(response["usage"]["completion_tokens"])
        # print("new cost tracking")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_claude()


def test_completion_claude_3_empty_response():
    litellm.set_verbose = True
    messages = [
        {
            "role": "system",
            "content": "You are 2twNLGfqk4GMOn3ffp4p.",
        },
        {"role": "user", "content": "Hi gm!", "name": "ishaan"},
        {"role": "assistant", "content": "Good morning! How are you doing today?"},
        {
            "role": "user",
            "content": "I was hoping we could chat a bit",
        },
    ]
    response = litellm.completion(model="claude-3-opus-20240229", messages=messages)
    print(response)


def test_completion_claude_3():
    litellm.set_verbose = True
    messages = [{"role": "user", "content": "Hello, world"}]
    try:
        # test without max tokens
        response = completion(
            model="anthropic/claude-3-opus-20240229",
            messages=messages,
        )
        # Add any assertions, here to check response args
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_claude_3_function_call():
    litellm.set_verbose = True
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
    try:
        # test without max tokens
        response = completion(
            model="anthropic/claude-3-opus-20240229",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        # Add any assertions, here to check response args
        print(response)
        assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
        assert isinstance(
            response.choices[0].message.tool_calls[0].function.arguments, str
        )
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_claude_3_stream():
    litellm.set_verbose = False
    messages = [{"role": "user", "content": "Hello, world"}]
    try:
        # test without max tokens
        response = completion(
            model="anthropic/claude-3-opus-20240229",
            messages=messages,
            max_tokens=10,
            stream=True,
        )
        # Add any assertions, here to check response args
        print(response)
        for chunk in response:
            print(chunk)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def encode_image(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@pytest.mark.skip(
    reason="we already test claude-3, this is just another way to pass images"
)
def test_completion_claude_3_base64():
    try:
        litellm.set_verbose = True
        litellm.num_retries = 3
        image_path = "../proxy/cached_logo.jpg"
        # Getting the base64 string
        base64_image = encode_image(image_path)
        resp = litellm.completion(
            model="anthropic/claude-3-opus-20240229",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Whats in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + base64_image
                            },
                        },
                    ],
                }
            ],
        )
        print(f"\nResponse: {resp}")

        prompt_tokens = resp.usage.prompt_tokens
        raise Exception("it worked!")
    except Exception as e:
        if "500 Internal error encountered.'" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


@pytest.mark.skip(reason="issue getting wikipedia images in ci/cd")
def test_completion_claude_3_function_plus_image():
    litellm.set_verbose = True

    image_content = [
        {"type": "text", "text": "What’s in this image?"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            },
        },
    ]
    image_message = {"role": "user", "content": image_content}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "text",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    tool_choice = {"type": "function", "function": {"name": "get_current_weather"}}
    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]

    response = completion(
        model="claude-3-sonnet-20240229",
        messages=[image_message],
        tool_choice=tool_choice,
        tools=tools,
        stream=False,
    )

    print(response)


def test_completion_mistral_api():
    try:
        litellm.set_verbose = True
        response = completion(
            model="mistral/mistral-tiny",
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": "Hey, how's it going?",
                }
            ],
        )
        # Add any assertions here to check the response
        print(response)

        cost = litellm.completion_cost(completion_response=response)
        print("cost to make mistral completion=", cost)
        assert cost > 0.0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(
    reason="Since we already test mistral/mistral-tiny in test_completion_mistral_api. This is only for locally verifying azure mistral works"
)
def test_completion_mistral_azure():
    try:
        litellm.set_verbose = True
        response = completion(
            model="mistral/Mistral-large-nmefg",
            api_key=os.environ["MISTRAL_AZURE_API_KEY"],
            api_base=os.environ["MISTRAL_AZURE_API_BASE"],
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": "Hi from litellm",
                }
            ],
        )
        # Add any assertions here to check the response
        print(response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_mistral_api()


def test_completion_mistral_api_modified_input():
    try:
        litellm.set_verbose = True
        response = completion(
            model="mistral/mistral-tiny",
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hey, how's it going?"}],
                }
            ],
        )
        # Add any assertions here to check the response
        print(response)

        cost = litellm.completion_cost(completion_response=response)
        print("cost to make mistral completion=", cost)
        assert cost > 0.0
    except Exception as e:
        if "500" in str(e):
            pass
        else:
            pytest.fail(f"Error occurred: {e}")


def test_completion_claude2_1():
    try:
        litellm.set_verbose = True
        print("claude2.1 test request")
        messages = [
            {
                "role": "system",
                "content": "Your goal is generate a joke on the topic user gives.",
            },
            {"role": "user", "content": "Generate a 3 liner joke for me"},
        ]
        # test without max tokens
        response = completion(model="claude-2.1", messages=messages)
        # Add any assertions here to check the response
        print(response)
        print(response.usage)
        print(response.usage.completion_tokens)
        print(response["usage"]["completion_tokens"])
        # print("new cost tracking")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_claude2_1()

# def test_completion_oobabooga():
#     try:
#         response = completion(
#             model="oobabooga/vicuna-1.3b", messages=messages, api_base="http://127.0.0.1:5000"
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_oobabooga()
# aleph alpha
# def test_completion_aleph_alpha():
#     try:
#         response = completion(
#             model="luminous-base", messages=messages, logger_fn=logger_fn
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_completion_aleph_alpha()


# def test_completion_aleph_alpha_control_models():
#     try:
#         response = completion(
#             model="luminous-base-control", messages=messages, logger_fn=logger_fn
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_completion_aleph_alpha_control_models()

import openai


def test_completion_gpt4_turbo():
    try:
        response = completion(
            model="gpt-4-1106-preview",
            messages=messages,
            max_tokens=10,
        )
        print(response)
    except openai.RateLimitError:
        print("got a rate liimt error")
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_gpt4_turbo()


def test_completion_gpt4_turbo_0125():
    try:
        response = completion(
            model="gpt-4-0125-preview",
            messages=messages,
            max_tokens=10,
        )
        print(response)
    except openai.RateLimitError:
        print("got a rate liimt error")
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="this test is flaky")
def test_completion_gpt4_vision():
    try:
        litellm.set_verbose = True
        response = completion(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Whats in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                            },
                        },
                    ],
                }
            ],
        )
        print(response)
    except openai.RateLimitError:
        print("got a rate liimt error")
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_gpt4_vision()


def test_completion_azure_gpt4_vision():
    # azure/gpt-4, vision takes 5 seconds to respond
    try:
        litellm.set_verbose = True
        response = completion(
            model="azure/gpt-4-vision",
            timeout=5,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Whats in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://avatars.githubusercontent.com/u/29436595?v=4"
                            },
                        },
                    ],
                }
            ],
            base_url="https://gpt-4-vision-resource.openai.azure.com/openai/deployments/gpt-4-vision/extensions",
            api_key=os.getenv("AZURE_VISION_API_KEY"),
            enhancements={"ocr": {"enabled": True}, "grounding": {"enabled": True}},
            dataSources=[
                {
                    "type": "AzureComputerVision",
                    "parameters": {
                        "endpoint": "https://gpt-4-vision-enhancement.cognitiveservices.azure.com/",
                        "key": os.environ["AZURE_VISION_ENHANCE_KEY"],
                    },
                }
            ],
        )
        print(response)
    except openai.APIError as e:
        pass
    except openai.APITimeoutError:
        print("got a timeout error")
        pass
    except openai.RateLimitError as e:
        print("got a rate liimt error", e)
        pass
    except openai.APIStatusError as e:
        print("got an api status error", e)
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure_gpt4_vision()


@pytest.mark.skip(reason="this test is flaky")
def test_completion_perplexity_api():
    try:
        # litellm.set_verbose= True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="mistral-7b-instruct",
            messages=messages,
            api_base="https://api.perplexity.ai",
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_perplexity_api()


@pytest.mark.skip(reason="this test is flaky")
def test_completion_perplexity_api_2():
    try:
        # litellm.set_verbose=True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(model="perplexity/mistral-7b-instruct", messages=messages)
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_perplexity_api_2()

# commenting out as this is a flaky test on circle ci
# def test_completion_nlp_cloud():
#     try:
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {
#                 "role": "user",
#                 "content": "how does a court case get to the Supreme Court?",
#             },
#         ]
#         response = completion(model="dolphin", messages=messages, logger_fn=logger_fn)
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_nlp_cloud()

######### HUGGING FACE TESTS ########################
#####################################################
"""
HF Tests we should pass 
- TGI: 
    - Pro Inference API 
    - Deployed Endpoint 
- Coversational 
    - Free Inference API 
    - Deployed Endpoint 
- Neither TGI or Coversational
    - Free Inference API 
    - Deployed Endpoint 
"""


#####################################################
#####################################################
# Test util to sort models to TGI, conv, None
def test_get_hf_task_for_model():
    model = "glaiveai/glaive-coder-7b"
    model_type = litellm.llms.huggingface_restapi.get_hf_task_for_model(model)
    print(f"model:{model}, model type: {model_type}")
    assert model_type == "text-generation-inference"

    model = "meta-llama/Llama-2-7b-hf"
    model_type = litellm.llms.huggingface_restapi.get_hf_task_for_model(model)
    print(f"model:{model}, model type: {model_type}")
    assert model_type == "text-generation-inference"

    model = "facebook/blenderbot-400M-distill"
    model_type = litellm.llms.huggingface_restapi.get_hf_task_for_model(model)
    print(f"model:{model}, model type: {model_type}")
    assert model_type == "conversational"

    model = "facebook/blenderbot-3B"
    model_type = litellm.llms.huggingface_restapi.get_hf_task_for_model(model)
    print(f"model:{model}, model type: {model_type}")
    assert model_type == "conversational"

    # neither Conv or None
    model = "roneneldan/TinyStories-3M"
    model_type = litellm.llms.huggingface_restapi.get_hf_task_for_model(model)
    print(f"model:{model}, model type: {model_type}")
    assert model_type == None


# test_get_hf_task_for_model()
# litellm.set_verbose=False
# ################### Hugging Face TGI models ########################
# # TGI model
# # this is a TGI model https://huggingface.co/glaiveai/glaive-coder-7b
def hf_test_completion_tgi():
    # litellm.set_verbose=True
    try:
        response = completion(
            model="huggingface/HuggingFaceH4/zephyr-7b-beta",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
        )
        # Add any assertions here to check the response
        print(response)
    except litellm.ServiceUnavailableError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# hf_test_completion_tgi()

# ################### Hugging Face Conversational models ########################
# def hf_test_completion_conv():
#     try:
#         response = litellm.completion(
#             model="huggingface/facebook/blenderbot-3B",
#             messages=[{ "content": "Hello, how are you?","role": "user"}],
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# hf_test_completion_conv()

# ################### Hugging Face Neither TGI or Conversational models ########################
# # Neither TGI or Conversational task
# def hf_test_completion_none_task():
#     try:
#         user_message = "My name is Merve and my favorite"
#         messages = [{ "content": user_message,"role": "user"}]
#         response = completion(
#             model="huggingface/roneneldan/TinyStories-3M",
#             messages=messages,
#             api_base="https://p69xlsj6rpno5drq.us-east-1.aws.endpoints.huggingface.cloud",
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# hf_test_completion_none_task()
########################### End of Hugging Face Tests ##############################################
# def test_completion_hf_api():
# # failing on circle ci commenting out
#     try:
#         user_message = "write some code to find the sum of two numbers"
#         messages = [{ "content": user_message,"role": "user"}]
#         api_base = "https://a8l9e3ucxinyl3oj.us-east-1.aws.endpoints.huggingface.cloud"
#         response = completion(model="huggingface/meta-llama/Llama-2-7b-chat-hf", messages=messages, api_base=api_base)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         if "loading" in str(e):
#             pass
#         pytest.fail(f"Error occurred: {e}")

# test_completion_hf_api()

# def test_completion_hf_api_best_of():
# # failing on circle ci commenting out
#     try:
#         user_message = "write some code to find the sum of two numbers"
#         messages = [{ "content": user_message,"role": "user"}]
#         api_base = "https://a8l9e3ucxinyl3oj.us-east-1.aws.endpoints.huggingface.cloud"
#         response = completion(model="huggingface/meta-llama/Llama-2-7b-chat-hf", messages=messages, api_base=api_base, n=2)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         if "loading" in str(e):
#             pass
#         pytest.fail(f"Error occurred: {e}")

# test_completion_hf_api_best_of()

# def test_completion_hf_deployed_api():
#     try:
#         user_message = "There's a llama in my garden 😱 What should I do?"
#         messages = [{ "content": user_message,"role": "user"}]
#         response = completion(model="huggingface/https://ji16r2iys9a8rjk2.us-east-1.aws.endpoints.huggingface.cloud", messages=messages, logger_fn=logger_fn)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# this should throw an exception, to trigger https://logs.litellm.ai/
# def hf_test_error_logs():
#     try:
#         litellm.set_verbose=True
#         user_message = "My name is Merve and my favorite"
#         messages = [{ "content": user_message,"role": "user"}]
#         response = completion(
#             model="huggingface/roneneldan/TinyStories-3M",
#             messages=messages,
#             api_base="https://p69xlsj6rpno5drq.us-east-1.aws.endpoints.huggingface.cloud",

#         )
#         # Add any assertions here to check the response
#         print(response)

#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# hf_test_error_logs()


# def test_completion_cohere():  # commenting out,for now as the cohere endpoint is being flaky
#     try:
#         litellm.CohereConfig(max_tokens=10, stop_sequences=["a"])
#         response = completion(
#             model="command-nightly", messages=messages, logger_fn=logger_fn
#         )
#         # Add any assertions here to check the response
#         print(response)
#         response_str = response["choices"][0]["message"]["content"]
#         response_str_2 = response.choices[0].message.content
#         if type(response_str) != str:
#             pytest.fail(f"Error occurred: {e}")
#         if type(response_str_2) != str:
#             pytest.fail(f"Error occurred: {e}")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# test_completion_cohere()


def test_completion_openai():
    try:
        litellm.set_verbose = True
        litellm.drop_params = True
        print(f"api key: {os.environ['OPENAI_API_KEY']}")
        litellm.api_key = os.environ["OPENAI_API_KEY"]
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey"}],
            max_tokens=10,
            metadata={"hi": "bye"},
        )
        print("This is the response object\n", response)

        response_str = response["choices"][0]["message"]["content"]
        response_str_2 = response.choices[0].message.content

        cost = completion_cost(completion_response=response)
        print("Cost for completion call with gpt-3.5-turbo: ", f"${float(cost):.10f}")
        assert response_str == response_str_2
        assert type(response_str) == str
        assert len(response_str) > 1

        litellm.api_key = None
    except Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_openai_organization():
    try:
        litellm.set_verbose = True
        try:
            response = completion(
                model="gpt-3.5-turbo", messages=messages, organization="org-ikDc4ex8NB"
            )
            pytest.fail("Request should have failed - This organization does not exist")
        except Exception as e:
            assert "No such organization: org-ikDc4ex8NB" in str(e)

    except Exception as e:
        print(e)
        pytest.fail(f"Error occurred: {e}")


def test_completion_text_openai():
    try:
        # litellm.set_verbose =True
        response = completion(model="gpt-3.5-turbo-instruct", messages=messages)
        print(response["choices"][0]["message"]["content"])
    except Exception as e:
        print(e)
        pytest.fail(f"Error occurred: {e}")


def custom_callback(
    kwargs,  # kwargs to completion
    completion_response,  # response from completion
    start_time,
    end_time,  # start/end time
):
    # Your custom code here
    try:
        print("LITELLM: in custom callback function")
        print("\nkwargs\n", kwargs)
        model = kwargs["model"]
        messages = kwargs["messages"]
        user = kwargs.get("user")

        #################################################

        print(
            f"""
                Model: {model},
                Messages: {messages},
                User: {user},
                Seed: {kwargs["seed"]},
                temperature: {kwargs["temperature"]},
            """
        )

        assert kwargs["user"] == "ishaans app"
        assert kwargs["model"] == "gpt-3.5-turbo-1106"
        assert kwargs["seed"] == 12
        assert kwargs["temperature"] == 0.5
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_openai_with_optional_params():
    # [Proxy PROD TEST] WARNING: DO NOT DELETE THIS TEST
    # assert that `user` gets passed to the completion call
    # Note: This tests that we actually send the optional params to the completion call
    # We use custom callbacks to test this
    try:
        litellm.set_verbose = True
        litellm.success_callback = [custom_callback]
        response = completion(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "user", "content": "respond in valid, json - what is the day"}
            ],
            temperature=0.5,
            top_p=0.1,
            seed=12,
            response_format={"type": "json_object"},
            logit_bias=None,
            user="ishaans app",
        )
        # Add any assertions here to check the response

        print(response)
        litellm.success_callback = []  # unset callbacks

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_openai_with_optional_params()


def test_completion_logprobs():
    """
    This function is used to test the litellm.completion logprobs functionality.

    Parameters:
        None

    Returns:
        None
    """
    try:
        litellm.set_verbose = True
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what is the time"}],
            temperature=0.5,
            top_p=0.1,
            seed=12,
            logit_bias=None,
            user="ishaans app",
            logprobs=True,
            top_logprobs=3,
        )
        # Add any assertions here to check the response

        print(response)
        print(len(response.choices[0].logprobs["content"][0]["top_logprobs"]))
        assert "logprobs" in response.choices[0]
        assert "content" in response.choices[0]["logprobs"]
        assert len(response.choices[0].logprobs["content"][0]["top_logprobs"]) == 3

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_logprobs()


def test_completion_logprobs_stream():
    """
    This function is used to test the litellm.completion logprobs functionality.

    Parameters:
        None

    Returns:
        None
    """
    try:
        litellm.set_verbose = False
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what is the time"}],
            temperature=0.5,
            top_p=0.1,
            seed=12,
            max_tokens=5,
            logit_bias=None,
            user="ishaans app",
            logprobs=True,
            top_logprobs=3,
            stream=True,
        )
        # Add any assertions here to check the response

        print(response)

        found_logprob = False
        for chunk in response:
            # check if atleast one chunk has log probs
            print(chunk)
            if "logprobs" in chunk.choices[0]:
                # assert we got a valid logprob in the choices
                assert len(chunk.choices[0].logprobs.content[0].top_logprobs) == 3
                found_logprob = True
                break
            print(chunk)
        assert found_logprob == True
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_logprobs_stream()


def test_completion_openai_litellm_key():
    try:
        litellm.set_verbose = True
        litellm.num_retries = 0
        litellm.api_key = os.environ["OPENAI_API_KEY"]

        # ensure key is set to None in .env and in openai.api_key
        os.environ["OPENAI_API_KEY"] = ""
        import openai

        openai.api_key = ""
        ##########################################################

        response = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
            top_p=0.1,
            max_tokens=10,
            user="ishaan_dev@berri.ai",
        )
        # Add any assertions here to check the response
        print(response)

        ###### reset environ key
        os.environ["OPENAI_API_KEY"] = litellm.api_key

        ##### unset litellm var
        litellm.api_key = None
    except Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_ completion_openai_litellm_key()


def test_completion_ollama_hosted():
    try:
        litellm.request_timeout = 20  # give ollama 20 seconds to response
        litellm.set_verbose = True
        response = completion(
            model="ollama/phi",
            messages=messages,
            max_tokens=2,
            api_base="https://test-ollama-endpoint.onrender.com",
        )
        # Add any assertions here to check the response
        print(response)
    except openai.APITimeoutError as e:
        print("got a timeout error. Passed ! ")
        litellm.request_timeout = None
        pass
    except Exception as e:
        if "try pulling it first" in str(e):
            return
        pytest.fail(f"Error occurred: {e}")


# test_completion_ollama_hosted()


def test_completion_openrouter1():
    try:
        litellm.set_verbose = True
        response = completion(
            model="openrouter/mistralai/mistral-tiny",
            messages=messages,
            max_tokens=5,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_openrouter1()


def test_completion_hf_model_no_provider():
    try:
        response = completion(
            model="WizardLM/WizardLM-70B-V1.0",
            messages=messages,
            max_tokens=5,
        )
        # Add any assertions here to check the response
        print(response)
        pytest.fail(f"Error occurred: {e}")
    except Exception as e:
        pass


# test_completion_hf_model_no_provider()


def test_completion_anyscale_with_functions():
    function1 = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]
    try:
        messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
        response = completion(
            model="anyscale/mistralai/Mistral-7B-Instruct-v0.1",
            messages=messages,
            functions=function1,
        )
        # Add any assertions here to check the response
        print(response)

        cost = litellm.completion_cost(completion_response=response)
        print("cost to make anyscale completion=", cost)
        assert cost > 0.0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_anyscale_with_functions()


def test_completion_azure_key_completion_arg():
    # this tests if we can pass api_key to completion, when it's not in the env.
    # DO NOT REMOVE THIS TEST. No MATTER WHAT Happens!
    # If you want to remove it, speak to Ishaan!
    # Ishaan will be very disappointed if this test is removed -> this is a standard way to pass api_key + the router + proxy use this
    old_key = os.environ["AZURE_API_KEY"]
    os.environ.pop("AZURE_API_KEY", None)
    try:
        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = True
        ## Test azure call
        response = completion(
            model="azure/chatgpt-v-2",
            messages=messages,
            api_key=old_key,
            max_tokens=10,
        )
        print(f"response: {response}")

        print("Hidden Params", response._hidden_params)
        assert response._hidden_params["custom_llm_provider"] == "azure"
        os.environ["AZURE_API_KEY"] = old_key
    except Exception as e:
        os.environ["AZURE_API_KEY"] = old_key
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure_key_completion_arg()


def test_azure_instruct():
    litellm.set_verbose = True
    response = completion(
        model="azure_text/instruct-model",
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        max_tokens=10,
    )
    print("response", response)


@pytest.mark.asyncio
async def test_azure_instruct_stream():
    litellm.set_verbose = False
    response = await litellm.acompletion(
        model="azure_text/instruct-model",
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        max_tokens=10,
        stream=True,
    )
    print("response", response)
    async for chunk in response:
        print(chunk)


async def test_re_use_azure_async_client():
    try:
        print("azure gpt-3.5 ASYNC with clie nttest\n\n")
        litellm.set_verbose = True
        import openai

        client = openai.AsyncAzureOpenAI(
            azure_endpoint=os.environ["AZURE_API_BASE"],
            api_key=os.environ["AZURE_API_KEY"],
            api_version="2023-07-01-preview",
        )
        ## Test azure call
        for _ in range(3):
            response = await litellm.acompletion(
                model="azure/chatgpt-v-2", messages=messages, client=client
            )
            print(f"response: {response}")
    except Exception as e:
        pytest.fail("got Exception", e)


# import asyncio
# asyncio.run(
#     test_re_use_azure_async_client()
# )


def test_re_use_openaiClient():
    try:
        print("gpt-3.5  with client test\n\n")
        litellm.set_verbose = True
        import openai

        client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
        ## Test OpenAI call
        for _ in range(2):
            response = litellm.completion(
                model="gpt-3.5-turbo", messages=messages, client=client
            )
            print(f"response: {response}")
    except Exception as e:
        pytest.fail("got Exception", e)


# test_re_use_openaiClient()


def test_completion_azure():
    try:
        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = False
        ## Test azure call
        response = completion(
            model="azure/chatgpt-v-2",
            messages=messages,
            api_key="os.environ/AZURE_API_KEY",
        )
        print(f"response: {response}")
        ## Test azure flag for backwards compat
        # response = completion(
        #     model="chatgpt-v-2",
        #     messages=messages,
        #     azure=True,
        #     max_tokens=10
        # )
        # Add any assertions here to check the response
        print(response)

        cost = completion_cost(completion_response=response)
        assert cost > 0.0
        print("Cost for azure completion request", cost)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure()


def test_azure_openai_ad_token():
    # this tests if the azure ad token is set in the request header
    # the request can fail since azure ad tokens expire after 30 mins, but the header MUST have the azure ad token
    # we use litellm.input_callbacks for this test
    def tester(
        kwargs,  # kwargs to completion
    ):
        print(kwargs["additional_args"])
        if kwargs["additional_args"]["headers"]["Authorization"] != "Bearer gm":
            pytest.fail("AZURE AD TOKEN Passed but not set in request header")
        return

    litellm.input_callback = [tester]
    try:
        response = litellm.completion(
            model="azure/chatgpt-v-2",  # e.g. gpt-35-instant
            messages=[
                {
                    "role": "user",
                    "content": "what is your name",
                },
            ],
            azure_ad_token="gm",
        )
        print("azure ad token respoonse\n")
        print(response)
        litellm.input_callback = []
    except Exception as e:
        litellm.input_callback = []
        pytest.fail(f"An exception occurs - {str(e)}")


# test_azure_openai_ad_token()


# test_completion_azure()
def test_completion_azure2():
    # test if we can pass api_base, api_version and api_key in compleition()
    try:
        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = False
        api_base = os.environ["AZURE_API_BASE"]
        api_key = os.environ["AZURE_API_KEY"]
        api_version = os.environ["AZURE_API_VERSION"]

        os.environ["AZURE_API_BASE"] = ""
        os.environ["AZURE_API_VERSION"] = ""
        os.environ["AZURE_API_KEY"] = ""

        ## Test azure call
        response = completion(
            model="azure/chatgpt-v-2",
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            max_tokens=10,
        )

        # Add any assertions here to check the response
        print(response)

        os.environ["AZURE_API_BASE"] = api_base
        os.environ["AZURE_API_VERSION"] = api_version
        os.environ["AZURE_API_KEY"] = api_key

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure2()


def test_completion_azure3():
    # test if we can pass api_base, api_version and api_key in compleition()
    try:
        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = True
        litellm.api_base = os.environ["AZURE_API_BASE"]
        litellm.api_key = os.environ["AZURE_API_KEY"]
        litellm.api_version = os.environ["AZURE_API_VERSION"]

        os.environ["AZURE_API_BASE"] = ""
        os.environ["AZURE_API_VERSION"] = ""
        os.environ["AZURE_API_KEY"] = ""

        ## Test azure call
        response = completion(
            model="azure/chatgpt-v-2",
            messages=messages,
            max_tokens=10,
        )

        # Add any assertions here to check the response
        print(response)

        os.environ["AZURE_API_BASE"] = litellm.api_base
        os.environ["AZURE_API_VERSION"] = litellm.api_version
        os.environ["AZURE_API_KEY"] = litellm.api_key

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure3()


# new azure test for using litellm. vars,
# use the following vars in this test and make an azure_api_call
#  litellm.api_type = self.azure_api_type
#  litellm.api_base = self.azure_api_base
#  litellm.api_version = self.azure_api_version
#  litellm.api_key = self.api_key
def test_completion_azure_with_litellm_key():
    try:
        print("azure gpt-3.5 test\n\n")
        import openai

        #### set litellm vars
        litellm.api_type = "azure"
        litellm.api_base = os.environ["AZURE_API_BASE"]
        litellm.api_version = os.environ["AZURE_API_VERSION"]
        litellm.api_key = os.environ["AZURE_API_KEY"]

        ######### UNSET ENV VARs for this ################
        os.environ["AZURE_API_BASE"] = ""
        os.environ["AZURE_API_VERSION"] = ""
        os.environ["AZURE_API_KEY"] = ""

        ######### UNSET OpenAI vars for this ##############
        openai.api_type = ""
        openai.api_base = "gm"
        openai.api_version = "333"
        openai.api_key = "ymca"

        response = completion(
            model="azure/chatgpt-v-2",
            messages=messages,
        )
        # Add any assertions here to check the response
        print(response)

        ######### RESET ENV VARs for this ################
        os.environ["AZURE_API_BASE"] = litellm.api_base
        os.environ["AZURE_API_VERSION"] = litellm.api_version
        os.environ["AZURE_API_KEY"] = litellm.api_key

        ######### UNSET litellm vars
        litellm.api_type = None
        litellm.api_base = None
        litellm.api_version = None
        litellm.api_key = None

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure()


def test_completion_azure_deployment_id():
    try:
        litellm.set_verbose = True
        response = completion(
            deployment_id="chatgpt-v-2",
            model="gpt-3.5-turbo",
            messages=messages,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure_deployment_id()

# Only works for local endpoint
# def test_completion_anthropic_openai_proxy():
#     try:
#         response = completion(
#             model="custom_openai/claude-2",
#             messages=messages,
#             api_base="http://0.0.0.0:8000"
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_anthropic_openai_proxy()


@pytest.mark.skip(reason="replicate endpoints take +2 mins just for this request")
def test_completion_replicate_vicuna():
    print("TESTING REPLICATE")
    litellm.set_verbose = True
    model_name = "replicate/meta/llama-2-7b-chat:f1d50bb24186c52daae319ca8366e53debdaa9e0ae7ff976e918df752732ccc4"
    try:
        response = completion(
            model=model_name,
            messages=messages,
            temperature=0.5,
            top_k=20,
            repetition_penalty=1,
            min_tokens=1,
            seed=-1,
            max_tokens=2,
        )
        print(response)
        # Add any assertions here to check the response
        response_str = response["choices"][0]["message"]["content"]
        print("RESPONSE STRING\n", response_str)
        if type(response_str) != str:
            pytest.fail(f"Error occurred: {e}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_replicate_vicuna()


def test_replicate_custom_prompt_dict():
    litellm.set_verbose = True
    model_name = "replicate/meta/llama-2-7b-chat"
    litellm.register_prompt_template(
        model="replicate/meta/llama-2-7b-chat",
        initial_prompt_value="You are a good assistant",  # [OPTIONAL]
        roles={
            "system": {
                "pre_message": "[INST] <<SYS>>\n",  # [OPTIONAL]
                "post_message": "\n<</SYS>>\n [/INST]\n",  # [OPTIONAL]
            },
            "user": {
                "pre_message": "[INST] ",  # [OPTIONAL]
                "post_message": " [/INST]",  # [OPTIONAL]
            },
            "assistant": {
                "pre_message": "\n",  # [OPTIONAL]
                "post_message": "\n",  # [OPTIONAL]
            },
        },
        final_prompt_value="Now answer as best you can:",  # [OPTIONAL]
    )
    response = completion(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "what is yc write 1 paragraph",
            }
        ],
        num_retries=3,
    )
    print(f"response: {response}")
    litellm.custom_prompt_dict = {}  # reset


# test_replicate_custom_prompt_dict()

# commenthing this out since we won't be always testing a custom, replicate deployment
# def test_completion_replicate_deployments():
#     print("TESTING REPLICATE")
#     litellm.set_verbose=False
#     model_name = "replicate/deployments/ishaan-jaff/ishaan-mistral"
#     try:
#         response = completion(
#             model=model_name,
#             messages=messages,
#             temperature=0.5,
#             seed=-1,
#         )
#         print(response)
#         # Add any assertions here to check the response
#         response_str = response["choices"][0]["message"]["content"]
#         print("RESPONSE STRING\n", response_str)
#         if type(response_str) != str:
#             pytest.fail(f"Error occurred: {e}")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_completion_replicate_deployments()


######## Test TogetherAI ########
@pytest.mark.skip(reason="Skip flaky test")
def test_completion_together_ai_mixtral():
    model_name = "together_ai/DiscoResearch/DiscoLM-mixtral-8x7b-v2"
    try:
        messages = [
            {"role": "user", "content": "Who are you"},
            {"role": "assistant", "content": "I am your helpful assistant."},
            {"role": "user", "content": "Tell me a joke"},
        ]
        response = completion(
            model=model_name,
            messages=messages,
            max_tokens=256,
            n=1,
            logger_fn=logger_fn,
        )
        # Add any assertions here to check the response
        print(response)
        cost = completion_cost(completion_response=response)
        assert cost > 0.0
        print(
            "Cost for completion call together-computer/llama-2-70b: ",
            f"${float(cost):.10f}",
        )
    except litellm.Timeout as e:
        pass
    except litellm.ServiceUnavailableError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_together_ai_mixtral()


def test_completion_together_ai_yi_chat():
    litellm.set_verbose = True
    model_name = "together_ai/zero-one-ai/Yi-34B-Chat"
    try:
        messages = [
            {"role": "user", "content": "What llm are you?"},
        ]
        response = completion(model=model_name, messages=messages, max_tokens=5)
        # Add any assertions here to check the response
        print(response)
        cost = completion_cost(completion_response=response)
        assert cost > 0.0
        print(
            "Cost for completion call together-computer/llama-2-70b: ",
            f"${float(cost):.10f}",
        )
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_together_ai_yi_chat()


# test_completion_together_ai()
def test_customprompt_together_ai():
    try:
        litellm.set_verbose = False
        litellm.num_retries = 0
        print("in test_customprompt_together_ai")
        print(litellm.success_callback)
        print(litellm._async_success_callback)
        response = completion(
            model="together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
            roles={
                "system": {
                    "pre_message": "<|im_start|>system\n",
                    "post_message": "<|im_end|>",
                },
                "assistant": {
                    "pre_message": "<|im_start|>assistant\n",
                    "post_message": "<|im_end|>",
                },
                "user": {
                    "pre_message": "<|im_start|>user\n",
                    "post_message": "<|im_end|>",
                },
            },
        )
        print(response)
    except litellm.exceptions.Timeout as e:
        print(f"Timeout Error")
        pass
    except Exception as e:
        print(f"ERROR TYPE {type(e)}")
        pytest.fail(f"Error occurred: {e}")


# test_customprompt_together_ai()


@pytest.mark.skip(reason="AWS Suspended Account")
def test_completion_sagemaker():
    try:
        litellm.set_verbose = True
        print("testing sagemaker")
        response = completion(
            model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
            messages=messages,
            temperature=0.2,
            max_tokens=80,
            input_cost_per_second=0.000420,
        )
        # Add any assertions here to check the response
        print(response)
        cost = completion_cost(completion_response=response)
        print("calculated cost", cost)
        assert (
            cost > 0.0 and cost < 1.0
        )  # should never be > $1 for a single completion call
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_sagemaker()


@pytest.mark.skip(reason="AWS Suspended Account")
def test_completion_sagemaker_stream():
    try:
        litellm.set_verbose = False
        print("testing sagemaker")
        response = completion(
            model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
            messages=messages,
            temperature=0.2,
            max_tokens=80,
            stream=True,
        )

        complete_streaming_response = ""
        first_chunk_id, chunk_id = None, None
        for i, chunk in enumerate(response):
            print(chunk)
            chunk_id = chunk.id
            print(chunk_id)
            if i == 0:
                first_chunk_id = chunk_id
            else:
                assert chunk_id == first_chunk_id
            complete_streaming_response += chunk.choices[0].delta.content or ""
        # Add any assertions here to check the response
        # print(response)
        assert len(complete_streaming_response) > 0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="AWS Suspended Account")
def test_completion_chat_sagemaker():
    try:
        messages = [{"role": "user", "content": "Hey, how's it going?"}]
        litellm.set_verbose = True
        response = completion(
            model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            stream=True,
        )
        # Add any assertions here to check the response
        complete_response = ""
        for chunk in response:
            complete_response += chunk.choices[0].delta.content or ""
        print(f"complete_response: {complete_response}")
        assert len(complete_response) > 0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_chat_sagemaker()


@pytest.mark.skip(reason="AWS Suspended Account")
def test_completion_chat_sagemaker_mistral():
    try:
        messages = [{"role": "user", "content": "Hey, how's it going?"}]

        response = completion(
            model="sagemaker/jumpstart-dft-hf-llm-mistral-7b-instruct",
            messages=messages,
            max_tokens=100,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"An error occurred: {str(e)}")


# test_completion_chat_sagemaker_mistral()


def test_completion_bedrock_titan_null_response():
    try:
        response = completion(
            model="bedrock/amazon.titan-text-lite-v1",
            messages=[
                {
                    "role": "user",
                    "content": "Hello!",
                },
                {
                    "role": "assistant",
                    "content": "Hello! How can I help you?",
                },
                {
                    "role": "user",
                    "content": "What model are you?",
                },
            ],
        )
        # Add any assertions here to check the response
        print(f"response: {response}")
    except Exception as e:
        pytest.fail(f"An error occurred - {str(e)}")


def test_completion_bedrock_titan():
    try:
        response = completion(
            model="bedrock/amazon.titan-tg1-large",
            messages=messages,
            temperature=0.2,
            max_tokens=200,
            top_p=0.8,
            logger_fn=logger_fn,
        )
        # Add any assertions here to check the response
        print(response)
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_bedrock_titan()


def test_completion_bedrock_claude():
    print("calling claude")
    try:
        response = completion(
            model="anthropic.claude-instant-v1",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            logger_fn=logger_fn,
        )
        # Add any assertions here to check the response
        print(response)
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_bedrock_claude()


def test_completion_bedrock_cohere():
    print("calling bedrock cohere")
    litellm.set_verbose = True
    try:
        response = completion(
            model="bedrock/cohere.command-text-v14",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.1,
            max_tokens=10,
            stream=True,
        )
        # Add any assertions here to check the response
        print(response)
        for chunk in response:
            print(chunk)
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_bedrock_cohere()


# def test_completion_bedrock_claude_stream():
#     print("calling claude")
#     litellm.set_verbose = False
#     try:
#         response = completion(
#             model="bedrock/anthropic.claude-instant-v1",
#             messages=messages,
#             stream=True
#         )
#         # Add any assertions here to check the response
#         print(response)
#         for chunk in response:
#             print(chunk)
#     except RateLimitError:
#         pass
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_completion_bedrock_claude_stream()

# def test_completion_bedrock_ai21():
#     try:
#         litellm.set_verbose = False
#         response = completion(
#             model="bedrock/ai21.j2-mid",
#             messages=messages,
#             temperature=0.2,
#             top_p=0.2,
#             max_tokens=20
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except RateLimitError:
#         pass
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


######## Test VLLM ########
# def test_completion_vllm():
#     try:
#         response = completion(
#             model="vllm/facebook/opt-125m",
#             messages=messages,
#             temperature=0.2,
#             max_tokens=80,
#         )
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_vllm()

# def test_completion_hosted_chatCompletion():
#     # this tests calling a server where vllm is hosted
#     # this should make an openai.Completion() call to the specified api_base
#     # send a request to this proxy server: https://replit.com/@BerriAI/openai-proxy#main.py
#     # it checks if model == facebook/opt-125m and returns test passed
#     try:
#         litellm.set_verbose = True
#         response = completion(
#             model="facebook/opt-125m",
#             messages=messages,
#             temperature=0.2,
#             max_tokens=80,
#             api_base="https://openai-proxy.berriai.repl.co",
#             custom_llm_provider="openai"
#         )
#         print(response)

#         if response['choices'][0]['message']['content'] != "passed":
#             # see https://replit.com/@BerriAI/openai-proxy#main.py
#             pytest.fail(f"Error occurred: proxy server did not respond")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_hosted_chatCompletion()

# def test_completion_custom_api_base():
#     try:
#         response = completion(
#             model="custom/meta-llama/Llama-2-13b-hf",
#             messages=messages,
#             temperature=0.2,
#             max_tokens=10,
#             api_base="https://api.autoai.dev/inference",
#             request_timeout=300,
#         )
#         # Add any assertions here to check the response
#         print("got response\n", response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_custom_api_base()


def test_completion_with_fallbacks():
    print(f"RUNNING TEST COMPLETION WITH FALLBACKS -  test_completion_with_fallbacks")
    fallbacks = ["gpt-3.5-turbo", "gpt-3.5-turbo", "command-nightly"]
    try:
        response = completion(
            model="bad-model", messages=messages, force_timeout=120, fallbacks=fallbacks
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_with_fallbacks()
def test_completion_anyscale_api():
    try:
        # litellm.set_verbose=True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="anyscale/meta-llama/Llama-2-7b-chat-hf",
            messages=messages,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_anyscale_api()


@pytest.mark.skip(reason="flaky test, times out frequently")
def test_completion_cohere():
    try:
        # litellm.set_verbose=True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="command-nightly",
            messages=messages,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# FYI - cohere_chat looks quite unstable, even when testing locally
def test_chat_completion_cohere():
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="cohere_chat/command-r",
            messages=messages,
            max_tokens=10,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_chat_completion_cohere_stream():
    try:
        litellm.set_verbose = False
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="cohere_chat/command-r",
            messages=messages,
            max_tokens=10,
            stream=True,
        )
        print(response)
        for chunk in response:
            print(chunk)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_azure_cloudflare_api():
    litellm.set_verbose = True
    try:
        messages = [
            {
                "role": "user",
                "content": "How do I output all files in a directory using Python?",
            },
        ]
        response = completion(
            model="azure/gpt-turbo",
            messages=messages,
            base_url=os.getenv("CLOUDFLARE_AZURE_BASE_URL"),
            api_key=os.getenv("AZURE_FRANCE_API_KEY"),
        )
        print(f"response: {response}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        traceback.print_exc()
        pass


# test_azure_cloudflare_api()


def test_completion_anyscale_2():
    try:
        # litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="anyscale/meta-llama/Llama-2-7b-chat-hf", messages=messages
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_mistral_anyscale_stream():
    litellm.set_verbose = False
    response = completion(
        model="anyscale/mistralai/Mistral-7B-Instruct-v0.1",
        messages=[{"content": "hello, good morning", "role": "user"}],
        stream=True,
    )
    for chunk in response:
        # print(chunk)
        print(chunk["choices"][0]["delta"].get("content", ""), end="")


# test_mistral_anyscale_stream()
# test_completion_anyscale_2()
# def test_completion_with_fallbacks_multiple_keys():
#     print(f"backup key 1: {os.getenv('BACKUP_OPENAI_API_KEY_1')}")
#     print(f"backup key 2: {os.getenv('BACKUP_OPENAI_API_KEY_2')}")
#     backup_keys = [{"api_key": os.getenv("BACKUP_OPENAI_API_KEY_1")}, {"api_key": os.getenv("BACKUP_OPENAI_API_KEY_2")}]
#     try:
#         api_key = "bad-key"
#         response = completion(
#             model="gpt-3.5-turbo", messages=messages, force_timeout=120, fallbacks=backup_keys, api_key=api_key
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         error_str = traceback.format_exc()
#         pytest.fail(f"Error occurred: {error_str}")

# test_completion_with_fallbacks_multiple_keys()
# def test_petals():
#     try:
#         response = completion(model="petals-team/StableBeluga2", messages=messages)
#         # Add any assertions here to check the response
#         print(response)

#         response = completion(model="petals-team/StableBeluga2", messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# def test_baseten():
#     try:

#         response = completion(model="baseten/7qQNLDB", messages=messages, logger_fn=logger_fn)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_baseten()
# def test_baseten_falcon_7bcompletion():
#     model_name = "qvv0xeq"
#     try:
#         response = completion(model=model_name, messages=messages, custom_llm_provider="baseten")
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# def test_baseten_falcon_7bcompletion_withbase():
#     model_name = "qvv0xeq"
#     litellm.api_base = "https://app.baseten.co"
#     try:
#         response = completion(model=model_name, messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
#     litellm.api_base = None

# test_baseten_falcon_7bcompletion_withbase()


# def test_baseten_wizardLMcompletion_withbase():
#     model_name = "q841o8w"
#     litellm.api_base = "https://app.baseten.co"
#     try:
#         response = completion(model=model_name, messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_baseten_wizardLMcompletion_withbase()

# def test_baseten_mosaic_ML_completion_withbase():
#     model_name = "31dxrj3",
#     litellm.api_base = "https://app.baseten.co"
#     try:
#         response = completion(model=model_name, messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


#### Test A121 ###################
def test_completion_ai21():
    print("running ai21 j2light test")
    litellm.set_verbose = True
    model_name = "j2-light"
    try:
        response = completion(
            model=model_name, messages=messages, max_tokens=100, temperature=0.8
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_ai21()
# test_completion_ai21()
## test deep infra
def test_completion_deep_infra():
    litellm.set_verbose = False
    model_name = "deepinfra/meta-llama/Llama-2-70b-chat-hf"
    try:
        response = completion(
            model=model_name, messages=messages, temperature=0, max_tokens=10
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_deep_infra()


def test_completion_deep_infra_mistral():
    print("deep infra test with temp=0")
    model_name = "deepinfra/mistralai/Mistral-7B-Instruct-v0.1"
    try:
        response = completion(
            model=model_name,
            messages=messages,
            temperature=0.01,  # mistrail fails with temperature=0
            max_tokens=10,
        )
        # Add any assertions here to check the response
        print(response)
    except litellm.exceptions.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_deep_infra_mistral()


# Gemini tests
def test_completion_gemini():
    litellm.set_verbose = True
    model_name = "gemini/gemini-pro"
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    try:
        response = completion(model=model_name, messages=messages)
        # Add any assertions,here to check the response
        print(response)
    except litellm.APIError as e:
        pass
    except Exception as e:
        if "InternalServerError" in str(e):
            pass
        else:
            pytest.fail(f"Error occurred: {e}")


# test_completion_gemini()


@pytest.mark.asyncio
async def test_acompletion_gemini():
    litellm.set_verbose = True
    model_name = "gemini/gemini-pro"
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    try:
        response = await litellm.acompletion(model=model_name, messages=messages)
        # Add any assertions here to check the response
        print(f"response: {response}")
    except litellm.Timeout as e:
        pass
    except litellm.APIError as e:
        pass
    except Exception as e:
        if "InternalServerError" in str(e):
            pass
        else:
            pytest.fail(f"Error occurred: {e}")


# Palm tests
def test_completion_palm():
    litellm.set_verbose = True
    model_name = "palm/chat-bison"
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    try:
        response = completion(model=model_name, messages=messages)
        # Add any assertions here to check the response
        print(response)
    except litellm.APIError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_palm()


# test palm with streaming
def test_completion_palm_stream():
    # litellm.set_verbose = True
    model_name = "palm/chat-bison"
    try:
        response = completion(
            model=model_name,
            messages=messages,
            stop=["stop"],
            stream=True,
            max_tokens=20,
        )
        # Add any assertions here to check the response
        for chunk in response:
            print(chunk)
    except litellm.APIError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_palm_stream()

# test_completion_deep_infra()
# test_completion_ai21()
# test config file with completion #
# def test_completion_openai_config():
#     try:
#         litellm.config_path = "../config.json"
#         litellm.set_verbose = True
#         response = litellm.config_completion(messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#         litellm.config_path = None
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# def test_maritalk():
#     messages = [{"role": "user", "content": "Hey"}]
#     try:
#         response = completion("maritalk", messages=messages)
#         print(f"response: {response}")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_maritalk()


def test_completion_together_ai_stream():
    litellm.set_verbose = True
    user_message = "Write 1pg about YC & litellm"
    messages = [{"content": user_message, "role": "user"}]
    try:
        response = completion(
            model="together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
            stream=True,
            max_tokens=5,
        )
        print(response)
        for chunk in response:
            print(chunk)
        # print(string_response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_together_ai_stream()


# Cloud flare AI tests
@pytest.mark.skip(reason="Flaky test-cloudflare is very unstable")
def test_completion_cloudflare():
    try:
        litellm.set_verbose = True
        response = completion(
            model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
            messages=[{"content": "what llm are you", "role": "user"}],
            max_tokens=15,
            num_retries=3,
        )
        print(response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_cloudflare()


def test_moderation():
    response = litellm.moderation(input="i'm ishaan cto of litellm")
    print(response)
    output = response.results[0]
    print(output)
    return output


# test_moderation()
