import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm import (
    RateLimitError,
    Timeout,
    acompletion,
    completion,
    completion_cost,
    embedding,
)
from litellm.llms.vertex_ai import _gemini_convert_messages_with_history
from litellm.tests.test_streaming import streaming_format_tests

litellm.num_retries = 3
litellm.cache = None
user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]


def get_vertex_ai_creds_json() -> dict:
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/vertex_key.json"
    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
            # Read the file content
            print("Read vertexai file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Update the service_account_key_data with environment variables
    private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    return service_account_key_data


def load_vertex_ai_credentials():
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/vertex_key.json"

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
            # Read the file content
            print("Read vertexai file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Update the service_account_key_data with environment variables
    private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        # Write the updated content to the temporary files
        json.dump(service_account_key_data, temp_file, indent=2)

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)


@pytest.mark.asyncio
async def test_get_response():
    load_vertex_ai_credentials()
    prompt = '\ndef count_nums(arr):\n    """\n    Write a function count_nums which takes an array of integers and returns\n    the number of elements which has a sum of digits > 0.\n    If a number is negative, then its first signed digit will be negative:\n    e.g. -123 has signed digits -1, 2, and 3.\n    >>> count_nums([]) == 0\n    >>> count_nums([-1, 11, -11]) == 1\n    >>> count_nums([1, 1, 2]) == 3\n    """\n'
    try:
        response = await acompletion(
            model="gemini-pro",
            messages=[
                {
                    "role": "system",
                    "content": "Complete the given code with no more explanation. Remember that there is a 4-space indent before the first line of your generated code.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response
    except litellm.RateLimitError:
        pass
    except litellm.UnprocessableEntityError as e:
        pass
    except Exception as e:
        pytest.fail(f"An error occurred - {str(e)}")


@pytest.mark.asyncio
async def test_get_router_response():
    model = "claude-3-sonnet@20240229"
    vertex_ai_project = "adroit-crow-413218"
    vertex_ai_location = "asia-southeast1"
    json_obj = get_vertex_ai_creds_json()
    vertex_credentials = json.dumps(json_obj)

    prompt = '\ndef count_nums(arr):\n    """\n    Write a function count_nums which takes an array of integers and returns\n    the number of elements which has a sum of digits > 0.\n    If a number is negative, then its first signed digit will be negative:\n    e.g. -123 has signed digits -1, 2, and 3.\n    >>> count_nums([]) == 0\n    >>> count_nums([-1, 11, -11]) == 1\n    >>> count_nums([1, 1, 2]) == 3\n    """\n'
    try:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "sonnet",
                    "litellm_params": {
                        "model": "vertex_ai/claude-3-sonnet@20240229",
                        "vertex_ai_project": vertex_ai_project,
                        "vertex_ai_location": vertex_ai_location,
                        "vertex_credentials": vertex_credentials,
                    },
                }
            ]
        )
        response = await router.acompletion(
            model="sonnet",
            messages=[
                {
                    "role": "system",
                    "content": "Complete the given code with no more explanation. Remember that there is a 4-space indent before the first line of your generated code.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        print(f"\n\nResponse: {response}\n\n")

    except litellm.UnprocessableEntityError as e:
        pass
    except Exception as e:
        pytest.fail(f"An error occurred - {str(e)}")


# @pytest.mark.skip(
#     reason="Local test. Vertex AI Quota is low. Leads to rate limit errors on ci/cd."
# )
def test_vertex_ai_anthropic():
    model = "claude-3-sonnet@20240229"

    vertex_ai_project = "adroit-crow-413218"
    vertex_ai_location = "asia-southeast1"
    json_obj = get_vertex_ai_creds_json()
    vertex_credentials = json.dumps(json_obj)

    response = completion(
        model="vertex_ai/" + model,
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        vertex_ai_project=vertex_ai_project,
        vertex_ai_location=vertex_ai_location,
        vertex_credentials=vertex_credentials,
    )
    print("\nModel Response", response)


# @pytest.mark.skip(
#     reason="Local test. Vertex AI Quota is low. Leads to rate limit errors on ci/cd."
# )
def test_vertex_ai_anthropic_streaming():
    try:
        # load_vertex_ai_credentials()

        # litellm.set_verbose = True

        model = "claude-3-sonnet@20240229"

        vertex_ai_project = "adroit-crow-413218"
        vertex_ai_location = "asia-southeast1"
        json_obj = get_vertex_ai_creds_json()
        vertex_credentials = json.dumps(json_obj)

        response = completion(
            model="vertex_ai/" + model,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            vertex_ai_project=vertex_ai_project,
            vertex_ai_location=vertex_ai_location,
            stream=True,
        )
        # print("\nModel Response", response)
        for chunk in response:
            print(f"chunk: {chunk}")

    # raise Exception("it worked!")
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_vertex_ai_anthropic_streaming()


# @pytest.mark.skip(
#     reason="Local test. Vertex AI Quota is low. Leads to rate limit errors on ci/cd."
# )
@pytest.mark.asyncio
async def test_vertex_ai_anthropic_async():
    # load_vertex_ai_credentials()
    try:

        model = "claude-3-sonnet@20240229"

        vertex_ai_project = "adroit-crow-413218"
        vertex_ai_location = "asia-southeast1"
        json_obj = get_vertex_ai_creds_json()
        vertex_credentials = json.dumps(json_obj)

        response = await acompletion(
            model="vertex_ai/" + model,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            vertex_ai_project=vertex_ai_project,
            vertex_ai_location=vertex_ai_location,
            vertex_credentials=vertex_credentials,
        )
        print(f"Model Response: {response}")
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# asyncio.run(test_vertex_ai_anthropic_async())


# @pytest.mark.skip(
#     reason="Local test. Vertex AI Quota is low. Leads to rate limit errors on ci/cd."
# )
@pytest.mark.asyncio
async def test_vertex_ai_anthropic_async_streaming():
    # load_vertex_ai_credentials()
    try:
        litellm.set_verbose = True
        model = "claude-3-sonnet@20240229"

        vertex_ai_project = "adroit-crow-413218"
        vertex_ai_location = "asia-southeast1"
        json_obj = get_vertex_ai_creds_json()
        vertex_credentials = json.dumps(json_obj)

        response = await acompletion(
            model="vertex_ai/" + model,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            vertex_ai_project=vertex_ai_project,
            vertex_ai_location=vertex_ai_location,
            vertex_credentials=vertex_credentials,
            stream=True,
        )

        async for chunk in response:
            print(f"chunk: {chunk}")
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# asyncio.run(test_vertex_ai_anthropic_async_streaming())


def test_vertex_ai():
    import random

    litellm.num_retries = 3
    load_vertex_ai_credentials()
    test_models = (
        litellm.vertex_chat_models
        + litellm.vertex_code_chat_models
        + litellm.vertex_text_models
        + litellm.vertex_code_text_models
    )
    litellm.set_verbose = False
    vertex_ai_project = "adroit-crow-413218"
    # litellm.vertex_project = "adroit-crow-413218"

    test_models = random.sample(test_models, 1)
    test_models += litellm.vertex_language_models  # always test gemini-pro
    for model in test_models:
        try:
            if model in [
                "code-gecko",
                "code-gecko@001",
                "code-gecko@002",
                "code-gecko@latest",
                "codechat-bison@latest",
                "code-bison@001",
                "text-bison@001",
                "gemini-1.5-pro",
                "gemini-1.5-pro-preview-0215",
            ] or (
                "gecko" in model or "32k" in model or "ultra" in model or "002" in model
            ):
                # our account does not have access to this model
                continue
            print("making request", model)
            response = completion(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.7,
                vertex_ai_project=vertex_ai_project,
            )
            print("\nModel Response", response)
            print(response)
            assert type(response.choices[0].message.content) == str
            assert len(response.choices[0].message.content) > 1
            print(
                f"response.choices[0].finish_reason: {response.choices[0].finish_reason}"
            )
            assert response.choices[0].finish_reason in litellm._openai_finish_reasons
        except litellm.RateLimitError as e:
            pass
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


# test_vertex_ai()


def test_vertex_ai_stream():
    load_vertex_ai_credentials()
    litellm.set_verbose = True
    litellm.vertex_project = "adroit-crow-413218"
    import random

    test_models = (
        litellm.vertex_chat_models
        + litellm.vertex_code_chat_models
        + litellm.vertex_text_models
        + litellm.vertex_code_text_models
    )
    test_models = random.sample(test_models, 1)
    test_models += litellm.vertex_language_models  # always test gemini-pro
    for model in test_models:
        try:
            if model in [
                "code-gecko",
                "code-gecko@001",
                "code-gecko@002",
                "code-gecko@latest",
                "codechat-bison@latest",
                "code-bison@001",
                "text-bison@001",
                "gemini-1.5-pro",
                "gemini-1.5-pro-preview-0215",
            ] or (
                "gecko" in model or "32k" in model or "ultra" in model or "002" in model
            ):
                # our account does not have access to this model
                continue
            print("making request", model)
            response = completion(
                model=model,
                messages=[{"role": "user", "content": "hello tell me a short story"}],
                max_tokens=15,
                stream=True,
            )
            completed_str = ""
            for chunk in response:
                print(chunk)
                content = chunk.choices[0].delta.content or ""
                print("\n content", content)
                completed_str += content
                assert type(content) == str
                # pass
            assert len(completed_str) > 1
        except litellm.RateLimitError as e:
            pass
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


# test_vertex_ai_stream()


@pytest.mark.asyncio
async def test_async_vertexai_response():
    import random

    load_vertex_ai_credentials()
    test_models = (
        litellm.vertex_chat_models
        + litellm.vertex_code_chat_models
        + litellm.vertex_text_models
        + litellm.vertex_code_text_models
    )
    test_models = random.sample(test_models, 1)
    test_models += litellm.vertex_language_models  # always test gemini-pro
    for model in test_models:
        print(f"model being tested in async call: {model}")
        if model in [
            "code-gecko",
            "code-gecko@001",
            "code-gecko@002",
            "code-gecko@latest",
            "codechat-bison@latest",
            "code-bison@001",
            "text-bison@001",
            "gemini-1.5-pro",
            "gemini-1.5-pro-preview-0215",
        ] or ("gecko" in model or "32k" in model or "ultra" in model or "002" in model):
            # our account does not have access to this model
            continue
        try:
            user_message = "Hello, how are you?"
            messages = [{"content": user_message, "role": "user"}]
            response = await acompletion(
                model=model, messages=messages, temperature=0.7, timeout=5
            )
            print(f"response: {response}")
        except litellm.RateLimitError as e:
            pass
        except litellm.Timeout as e:
            pass
        except litellm.APIError as e:
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")


# asyncio.run(test_async_vertexai_response())


@pytest.mark.asyncio
async def test_async_vertexai_streaming_response():
    import random

    load_vertex_ai_credentials()
    test_models = (
        litellm.vertex_chat_models
        + litellm.vertex_code_chat_models
        + litellm.vertex_text_models
        + litellm.vertex_code_text_models
    )
    test_models = random.sample(test_models, 1)
    test_models += litellm.vertex_language_models  # always test gemini-pro
    for model in test_models:
        if model in [
            "code-gecko",
            "code-gecko@001",
            "code-gecko@002",
            "code-gecko@latest",
            "codechat-bison@latest",
            "code-bison@001",
            "text-bison@001",
            "gemini-1.5-pro",
            "gemini-1.5-pro-preview-0215",
        ] or ("gecko" in model or "32k" in model or "ultra" in model or "002" in model):
            # our account does not have access to this model
            continue
        try:
            user_message = "Hello, how are you?"
            messages = [{"content": user_message, "role": "user"}]
            response = await acompletion(
                model="gemini-pro",
                messages=messages,
                temperature=0.7,
                timeout=5,
                stream=True,
            )
            print(f"response: {response}")
            complete_response = ""
            async for chunk in response:
                print(f"chunk: {chunk}")
                if chunk.choices[0].delta.content is not None:
                    complete_response += chunk.choices[0].delta.content
            print(f"complete_response: {complete_response}")
            assert len(complete_response) > 0
        except litellm.RateLimitError as e:
            pass
        except litellm.Timeout as e:
            pass
        except Exception as e:
            print(e)
            pytest.fail(f"An exception occurred: {e}")


# asyncio.run(test_async_vertexai_streaming_response())


@pytest.mark.parametrize("provider", ["vertex_ai"])  # "vertex_ai_beta"
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_gemini_pro_vision(provider, sync_mode):
    try:
        load_vertex_ai_credentials()
        litellm.set_verbose = True
        litellm.num_retries = 3
        if sync_mode:
            resp = litellm.completion(
                model="{}/gemini-1.5-flash-preview-0514".format(provider),
                messages=[
                    {"role": "system", "content": "Be a good bot"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Whats in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "gs://cloud-samples-data/generative-ai/image/boats.jpeg"
                                },
                            },
                        ],
                    },
                ],
            )
        else:
            resp = await litellm.acompletion(
                model="{}/gemini-1.5-flash-preview-0514".format(provider),
                messages=[
                    {"role": "system", "content": "Be a good bot"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Whats in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "gs://cloud-samples-data/generative-ai/image/boats.jpeg"
                                },
                            },
                        ],
                    },
                ],
            )
        print(resp)

        prompt_tokens = resp.usage.prompt_tokens

        # DO Not DELETE this ASSERT
        # Google counts the prompt tokens for us, we should ensure we use the tokens from the orignal response
        assert prompt_tokens == 267  # the gemini api returns 267 to us

    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        if "500 Internal error encountered.'" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


# test_gemini_pro_vision()


def encode_image(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@pytest.mark.skip(
    reason="we already test gemini-pro-vision, this is just another way to pass images"
)
def test_gemini_pro_vision_base64():
    try:
        load_vertex_ai_credentials()
        litellm.set_verbose = True
        litellm.num_retries = 3
        image_path = "../proxy/cached_logo.jpg"
        # Getting the base64 string
        base64_image = encode_image(image_path)
        resp = litellm.completion(
            model="vertex_ai/gemini-pro-vision",
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
        print(resp)

        prompt_tokens = resp.usage.prompt_tokens
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        if "500 Internal error encountered.'" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


@pytest.mark.skip(reason="exhausted vertex quota. need to refactor to mock the call")
@pytest.mark.parametrize("provider", ["vertex_ai_beta"])  # "vertex_ai",
@pytest.mark.parametrize("sync_mode", [True])  # "vertex_ai",
@pytest.mark.asyncio
async def test_gemini_pro_function_calling_httpx(provider, sync_mode):
    try:
        load_vertex_ai_credentials()
        litellm.set_verbose = True

        messages = [
            {
                "role": "system",
                "content": "Your name is Litellm Bot, you are a helpful assistant",
            },
            # User asks for their name and weather in San Francisco
            {
                "role": "user",
                "content": "Hello, what is your name and can you tell me the weather?",
            },
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        data = {
            "model": "{}/gemini-1.5-pro".format(provider),
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
        }
        if sync_mode:
            response = litellm.completion(**data)
        else:
            response = await litellm.acompletion(**data)

        print(f"response: {response}")

        assert response.choices[0].message.tool_calls[0].function.arguments is not None
        assert isinstance(
            response.choices[0].message.tool_calls[0].function.arguments, str
        )
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        if "429 Quota exceeded" in str(e):
            pass
        else:
            pytest.fail("An unexpected exception occurred - {}".format(str(e)))


def vertex_httpx_mock_reject_prompt_post(*args, **kwargs):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "promptFeedback": {"blockReason": "OTHER"},
        "usageMetadata": {"promptTokenCount": 6285, "totalTokenCount": 6285},
    }

    return mock_response


# @pytest.mark.skip(reason="exhausted vertex quota. need to refactor to mock the call")
def vertex_httpx_mock_post(url, data=None, json=None, headers=None):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "candidates": [
            {
                "finishReason": "RECITATION",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.14965563,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.13660839,
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.16344544,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.10230471,
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.1979091,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.06052939,
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.1765296,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.18417984,
                    },
                ],
                "citationMetadata": {
                    "citations": [
                        {
                            "startIndex": 251,
                            "endIndex": 380,
                            "uri": "https://chocolatecake2023.blogspot.com/2023/02/taste-deliciousness-of-perfectly-baked.html?m=1",
                        },
                        {
                            "startIndex": 393,
                            "endIndex": 535,
                            "uri": "https://skinnymixes.co.uk/blogs/food-recipes/peanut-butter-cup-cookies",
                        },
                        {
                            "startIndex": 439,
                            "endIndex": 581,
                            "uri": "https://mast-producing-trees.org/aldis-chocolate-chips-are-peanut-and-tree-nut-free/",
                        },
                        {
                            "startIndex": 1117,
                            "endIndex": 1265,
                            "uri": "https://github.com/frdrck100/To_Do_Assignments",
                        },
                        {
                            "startIndex": 1146,
                            "endIndex": 1288,
                            "uri": "https://skinnymixes.co.uk/blogs/food-recipes/peanut-butter-cup-cookies",
                        },
                        {
                            "startIndex": 1166,
                            "endIndex": 1299,
                            "uri": "https://www.girlversusdough.com/brookies/",
                        },
                        {
                            "startIndex": 1780,
                            "endIndex": 1909,
                            "uri": "https://chocolatecake2023.blogspot.com/2023/02/taste-deliciousness-of-perfectly-baked.html?m=1",
                        },
                        {
                            "startIndex": 1834,
                            "endIndex": 1964,
                            "uri": "https://newsd.in/national-cream-cheese-brownie-day-2023-date-history-how-to-make-a-cream-cheese-brownie/",
                        },
                        {
                            "startIndex": 1846,
                            "endIndex": 1989,
                            "uri": "https://github.com/frdrck100/To_Do_Assignments",
                        },
                        {
                            "startIndex": 2121,
                            "endIndex": 2261,
                            "uri": "https://recipes.net/copycat/hardee/hardees-chocolate-chip-cookie-recipe/",
                        },
                        {
                            "startIndex": 2505,
                            "endIndex": 2671,
                            "uri": "https://www.tfrecipes.com/Oranges%20with%20dried%20cherries/",
                        },
                        {
                            "startIndex": 3390,
                            "endIndex": 3529,
                            "uri": "https://github.com/quantumcognition/Crud-palm",
                        },
                        {
                            "startIndex": 3568,
                            "endIndex": 3724,
                            "uri": "https://recipes.net/dessert/cakes/ultimate-easy-gingerbread/",
                        },
                        {
                            "startIndex": 3640,
                            "endIndex": 3770,
                            "uri": "https://recipes.net/dessert/cookies/soft-and-chewy-peanut-butter-cookies/",
                        },
                    ]
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 336, "totalTokenCount": 336},
    }
    return mock_response


@pytest.mark.parametrize("provider", ["vertex_ai_beta"])  # "vertex_ai",
@pytest.mark.parametrize("content_filter_type", ["prompt", "response"])  # "vertex_ai",
@pytest.mark.asyncio
async def test_gemini_pro_json_schema_httpx_content_policy_error(
    provider, content_filter_type
):
    load_vertex_ai_credentials()
    litellm.set_verbose = True
    messages = [
        {
            "role": "user",
            "content": """
    
List 5 popular cookie recipes.

Using this JSON schema:
```json
{'$defs': {'Recipe': {'properties': {'recipe_name': {'examples': ['Chocolate Chip Cookies', 'Peanut Butter Cookies'], 'maxLength': 100, 'title': 'The recipe name', 'type': 'string'}, 'estimated_time': {'anyOf': [{'minimum': 0, 'type': 'integer'}, {'type': 'null'}], 'default': None, 'description': 'The estimated time to make the recipe in minutes', 'examples': [30, 45], 'title': 'The estimated time'}, 'ingredients': {'examples': [['flour', 'sugar', 'chocolate chips'], ['peanut butter', 'sugar', 'eggs']], 'items': {'type': 'string'}, 'maxItems': 10, 'title': 'The ingredients', 'type': 'array'}, 'instructions': {'examples': [['mix', 'bake'], ['mix', 'chill', 'bake']], 'items': {'type': 'string'}, 'maxItems': 10, 'title': 'The instructions', 'type': 'array'}}, 'required': ['recipe_name', 'ingredients', 'instructions'], 'title': 'Recipe', 'type': 'object'}}, 'properties': {'recipes': {'items': {'$ref': '#/$defs/Recipe'}, 'maxItems': 11, 'title': 'The recipes', 'type': 'array'}}, 'required': ['recipes'], 'title': 'MyRecipes', 'type': 'object'}
```
            """,
        }
    ]
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    if content_filter_type == "prompt":
        _side_effect = vertex_httpx_mock_reject_prompt_post
    else:
        _side_effect = vertex_httpx_mock_post

    with patch.object(client, "post", side_effect=_side_effect) as mock_call:
        response = completion(
            model="vertex_ai_beta/gemini-1.5-flash",
            messages=messages,
            response_format={"type": "json_object"},
            client=client,
        )

        assert response.choices[0].finish_reason == "content_filter"

        mock_call.assert_called_once()


@pytest.mark.parametrize("provider", ["vertex_ai_beta"])  # "vertex_ai",
@pytest.mark.asyncio
async def test_gemini_pro_httpx_custom_api_base(provider):
    load_vertex_ai_credentials()
    litellm.set_verbose = True
    messages = [
        {
            "role": "user",
            "content": "Hello world",
        }
    ]
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post", new=MagicMock()) as mock_call:
        try:
            response = completion(
                model="vertex_ai_beta/gemini-1.5-flash",
                messages=messages,
                response_format={"type": "json_object"},
                client=client,
                api_base="my-custom-api-base",
                extra_headers={"hello": "world"},
            )
        except Exception as e:
            pass

        mock_call.assert_called_once()

        assert "my-custom-api-base:generateContent" == mock_call.call_args.kwargs["url"]
        assert "hello" in mock_call.call_args.kwargs["headers"]


@pytest.mark.skip(reason="exhausted vertex quota. need to refactor to mock the call")
@pytest.mark.parametrize("sync_mode", [True])
@pytest.mark.parametrize("provider", ["vertex_ai"])
@pytest.mark.asyncio
async def test_gemini_pro_function_calling(provider, sync_mode):
    try:
        load_vertex_ai_credentials()
        litellm.set_verbose = True

        messages = [
            {
                "role": "system",
                "content": "Your name is Litellm Bot, you are a helpful assistant",
            },
            # User asks for their name and weather in San Francisco
            {
                "role": "user",
                "content": "Hello, what is your name and can you tell me the weather?",
            },
            # Assistant replies with a tool call
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "index": 0,
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"San Francisco, CA"}',
                        },
                    }
                ],
            },
            # The result of the tool call is added to the history
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "get_weather",
                "content": "27 degrees celsius and clear in San Francisco, CA",
            },
            # Now the assistant can reply with the result of the tool call.
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        data = {
            "model": "{}/gemini-1.5-pro-preview-0514".format(provider),
            "messages": messages,
            "tools": tools,
        }
        if sync_mode:
            response = litellm.completion(**data)
        else:
            response = await litellm.acompletion(**data)

        print(f"response: {response}")
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        if "429 Quota exceeded" in str(e):
            pass
        else:
            pytest.fail("An unexpected exception occurred - {}".format(str(e)))


# gemini_pro_function_calling()


@pytest.mark.parametrize("sync_mode", [True])
@pytest.mark.asyncio
async def test_gemini_pro_function_calling_streaming(sync_mode):
    load_vertex_ai_credentials()
    litellm.set_verbose = True
    data = {
        "model": "vertex_ai/gemini-pro",
        "messages": [
            {
                "role": "user",
                "content": "Call the submit_cities function with San Francisco and New York",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "submit_cities",
                    "description": "Submits a list of cities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cities": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["cities"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "n": 1,
        "stream": True,
        "temperature": 0.1,
    }
    chunks = []
    try:
        if sync_mode == True:
            response = litellm.completion(**data)
            print(f"completion: {response}")

            for chunk in response:
                chunks.append(chunk)
                assert isinstance(chunk, litellm.ModelResponse)
        else:
            response = await litellm.acompletion(**data)
            print(f"completion: {response}")

            assert isinstance(response, litellm.CustomStreamWrapper)

            async for chunk in response:
                print(f"chunk: {chunk}")
                chunks.append(chunk)
                assert isinstance(chunk, litellm.ModelResponse)

        complete_response = litellm.stream_chunk_builder(chunks=chunks)
        assert (
            complete_response.choices[0].message.content is not None
            or len(complete_response.choices[0].message.tool_calls) > 0
        )
        print(f"complete_response: {complete_response}")
    except litellm.APIError as e:
        pass
    except litellm.RateLimitError as e:
        pass


@pytest.mark.asyncio
async def test_gemini_pro_async_function_calling():
    load_vertex_ai_credentials()
    try:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        messages = [
            {
                "role": "user",
                "content": "What's the weather like in Boston today in fahrenheit?",
            }
        ]
        completion = await litellm.acompletion(
            model="gemini-pro", messages=messages, tools=tools, tool_choice="auto"
        )
        print(f"completion: {completion}")
        assert completion.choices[0].message.content is None
        assert len(completion.choices[0].message.tool_calls) == 1

    # except litellm.APIError as e:
    #     pass
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")
    # raise Exception("it worked!")


# asyncio.run(gemini_pro_async_function_calling())


def test_vertexai_embedding():
    try:
        load_vertex_ai_credentials()
        # litellm.set_verbose = True
        response = embedding(
            model="textembedding-gecko@001",
            input=["good morning from litellm", "this is another item"],
        )
        print(f"response:", response)
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(
    reason="new test - works locally running into vertex version issues on ci/cd"
)
def test_vertexai_embedding_embedding_latest():
    try:
        load_vertex_ai_credentials()
        litellm.set_verbose = True

        response = embedding(
            model="vertex_ai/text-embedding-004",
            input=["hi"],
            dimensions=1,
            auto_truncate=True,
            task_type="RETRIEVAL_QUERY",
        )

        assert len(response.data[0]["embedding"]) == 1
        assert response.usage.prompt_tokens > 0
        print(f"response:", response)
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_vertexai_aembedding():
    try:
        load_vertex_ai_credentials()
        # litellm.set_verbose=True
        response = await litellm.aembedding(
            model="textembedding-gecko@001",
            input=["good morning from litellm", "this is another item"],
        )
        print(f"response: {response}")
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# Extra gemini Vision tests for completion + stream, async, async + stream
# if we run into issues with gemini, we will also add these to our ci/cd pipeline
# def test_gemini_pro_vision_stream():
#     try:
#         litellm.set_verbose = False
#         litellm.num_retries=0
#         print("streaming response from gemini-pro-vision")
#         resp = litellm.completion(
#             model = "vertex_ai/gemini-pro-vision",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                                     {
#                                         "type": "text",
#                                         "text": "Whats in this image?"
#                                     },
#                                     {
#                                         "type": "image_url",
#                                         "image_url": {
#                                         "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
#                                         }
#                                     }
#                                 ]
#                 }
#             ],
#             stream=True
#         )
#         print(resp)
#         for chunk in resp:
#             print(chunk)
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise e
# test_gemini_pro_vision_stream()


def test_gemini_pro_vision_async():
    try:
        litellm.set_verbose = True
        litellm.num_retries = 0

        async def test():
            load_vertex_ai_credentials()
            resp = await litellm.acompletion(
                model="vertex_ai/gemini-pro-vision",
                messages=[
                    {"role": "system", "content": ""},
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
                    },
                ],
            )
            print("async response gemini pro vision")
            print(resp)

        asyncio.run(test())
    except litellm.RateLimitError:
        pass
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e


# test_gemini_pro_vision_async()


# def test_gemini_pro_vision_async_stream():
#     try:
#         litellm.set_verbose = True
#         litellm.num_retries=0
#         async def test():
#             resp = await litellm.acompletion(
#                 model = "vertex_ai/gemini-pro-vision",
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                                         {
#                                             "type": "text",
#                                             "text": "Whats in this image?"
#                                         },
#                                         {
#                                             "type": "image_url",
#                                             "image_url": {
#                                             "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
#                                             }
#                                         }
#                                     ]
#                     }
#                 ],
#                 stream=True
#             )
#             print("async response gemini pro vision")
#             print(resp)
#             for chunk in resp:
#                 print(chunk)
#         asyncio.run(test())
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise e
# test_gemini_pro_vision_async()


def test_prompt_factory():
    messages = [
        {
            "role": "system",
            "content": "Your name is Litellm Bot, you are a helpful assistant",
        },
        # User asks for their name and weather in San Francisco
        {
            "role": "user",
            "content": "Hello, what is your name and can you tell me the weather?",
        },
        # Assistant replies with a tool call
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "index": 0,
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"San Francisco, CA"}',
                    },
                }
            ],
        },
        # The result of the tool call is added to the history
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "get_weather",
            "content": "27 degrees celsius and clear in San Francisco, CA",
        },
        # Now the assistant can reply with the result of the tool call.
    ]

    translated_messages = _gemini_convert_messages_with_history(messages=messages)

    print(f"\n\ntranslated_messages: {translated_messages}\ntranslated_messages")


def test_prompt_factory_nested():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hi! 👋 \n\nHow can I help you today? 😊 \n"}
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "hi 2nd time"}]},
    ]

    translated_messages = _gemini_convert_messages_with_history(messages=messages)

    print(f"\n\ntranslated_messages: {translated_messages}\ntranslated_messages")

    for message in translated_messages:
        assert len(message["parts"]) == 1
        assert "text" in message["parts"][0], "Missing 'text' from 'parts'"
        assert isinstance(
            message["parts"][0]["text"], str
        ), "'text' value not a string."
