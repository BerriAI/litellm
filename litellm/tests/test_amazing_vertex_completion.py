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
from unittest.mock import AsyncMock, MagicMock, patch

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
from litellm.llms.vertex_ai_and_google_ai_studio.vertex_and_google_ai_studio_gemini import (
    _gemini_convert_messages_with_history,
)
from litellm.tests.test_streaming import streaming_format_tests

litellm.num_retries = 3
litellm.cache = None
user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]

VERTEX_MODELS_TO_NOT_TEST = [
    "medlm-medium",
    "medlm-large",
    "code-gecko",
    "code-gecko@001",
    "code-gecko@002",
    "code-gecko@latest",
    "codechat-bison@latest",
    "code-bison@001",
    "text-bison@001",
    "gemini-1.5-pro",
    "gemini-1.5-pro-preview-0215",
]


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
@pytest.mark.flaky(retries=3, delay=1)
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
@pytest.mark.flaky(retries=3, delay=1)
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
@pytest.mark.flaky(retries=3, delay=1)
def test_vertex_ai_anthropic_streaming():
    try:
        load_vertex_ai_credentials()

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
        for idx, chunk in enumerate(response):
            print(f"chunk: {chunk}")
            streaming_format_tests(idx=idx, chunk=chunk)

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
@pytest.mark.flaky(retries=3, delay=1)
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
@pytest.mark.flaky(retries=3, delay=1)
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

        idx = 0
        async for chunk in response:
            streaming_format_tests(idx=idx, chunk=chunk)
            idx += 1
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# asyncio.run(test_vertex_ai_anthropic_async_streaming())


@pytest.mark.flaky(retries=3, delay=1)
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
            if model in VERTEX_MODELS_TO_NOT_TEST or (
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
        except litellm.InternalServerError as e:
            pass
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


# test_vertex_ai()


@pytest.mark.flaky(retries=3, delay=1)
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
            if model in VERTEX_MODELS_TO_NOT_TEST or (
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
        except litellm.InternalServerError as e:
            pass
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


# test_vertex_ai_stream()


@pytest.mark.flaky(retries=3, delay=1)
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
        if model in VERTEX_MODELS_TO_NOT_TEST or (
            "gecko" in model or "32k" in model or "ultra" in model or "002" in model
        ):
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
        except litellm.InternalServerError as e:
            pass
        except Exception as e:
            pytest.fail(f"An exception occurred: {e}")


# asyncio.run(test_async_vertexai_response())


@pytest.mark.flaky(retries=3, delay=1)
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
        if model in VERTEX_MODELS_TO_NOT_TEST or (
            "gecko" in model or "32k" in model or "ultra" in model or "002" in model
        ):
            # our account does not have access to this model
            continue
        try:
            user_message = "Hello, how are you?"
            messages = [{"content": user_message, "role": "user"}]
            response = await acompletion(
                model=model,
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
        except litellm.APIConnectionError:
            pass
        except litellm.Timeout as e:
            pass
        except litellm.InternalServerError as e:
            pass
        except Exception as e:
            print(e)
            pytest.fail(f"An exception occurred: {e}")


# asyncio.run(test_async_vertexai_streaming_response())


@pytest.mark.parametrize("provider", ["vertex_ai"])  # "vertex_ai_beta"
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.flaky(retries=3, delay=1)
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


@pytest.mark.parametrize("load_pdf", [False])  # True,
@pytest.mark.flaky(retries=3, delay=1)
def test_completion_function_plus_pdf(load_pdf):
    litellm.set_verbose = True
    load_vertex_ai_credentials()
    try:
        import base64

        import requests

        # URL of the file
        url = "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf"

        # Download the file
        if load_pdf:
            response = requests.get(url)
            file_data = response.content

            encoded_file = base64.b64encode(file_data).decode("utf-8")
            url = f"data:application/pdf;base64,{encoded_file}"

        image_content = [
            {"type": "text", "text": "What's this file about?"},
            {
                "type": "image_url",
                "image_url": {"url": url},
            },
        ]
        image_message = {"role": "user", "content": image_content}

        response = completion(
            model="vertex_ai_beta/gemini-1.5-flash-preview-0514",
            messages=[image_message],
            stream=False,
        )

        print(response)
    except litellm.InternalServerError as e:
        pass
    except Exception as e:
        pytest.fail("Got={}".format(str(e)))


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


def vertex_httpx_grounding_post(*args, **kwargs):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": "Argentina won the FIFA World Cup 2022. Argentina defeated France 4-2 on penalties in the FIFA World Cup 2022 final tournament for the first time after 36 years and the third time overall."
                        }
                    ],
                },
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.14940722,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.07477004,
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.15636235,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.015967654,
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.1943678,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.1284158,
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.09384396,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.0726367,
                    },
                ],
                "groundingMetadata": {
                    "webSearchQueries": ["who won the world cup 2022"],
                    "groundingAttributions": [
                        {
                            "segment": {"endIndex": 38},
                            "confidenceScore": 0.9919262,
                            "web": {
                                "uri": "https://www.careerpower.in/fifa-world-cup-winners-list.html",
                                "title": "FIFA World Cup Winners List from 1930 to 2022, Complete List - Career Power",
                            },
                        },
                        {
                            "segment": {"endIndex": 38},
                            "confidenceScore": 0.9919262,
                            "web": {
                                "uri": "https://www.careerpower.in/fifa-world-cup-winners-list.html",
                                "title": "FIFA World Cup Winners List from 1930 to 2022, Complete List - Career Power",
                            },
                        },
                        {
                            "segment": {"endIndex": 38},
                            "confidenceScore": 0.9919262,
                            "web": {
                                "uri": "https://www.britannica.com/sports/2022-FIFA-World-Cup",
                                "title": "2022 FIFA World Cup | Qatar, Controversy, Stadiums, Winner, & Final - Britannica",
                            },
                        },
                        {
                            "segment": {"endIndex": 38},
                            "confidenceScore": 0.9919262,
                            "web": {
                                "uri": "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_final",
                                "title": "2022 FIFA World Cup final - Wikipedia",
                            },
                        },
                        {
                            "segment": {"endIndex": 38},
                            "confidenceScore": 0.9919262,
                            "web": {
                                "uri": "https://www.transfermarkt.com/2022-world-cup/erfolge/pokalwettbewerb/WM22",
                                "title": "2022 World Cup - All winners - Transfermarkt",
                            },
                        },
                        {
                            "segment": {"startIndex": 39, "endIndex": 187},
                            "confidenceScore": 0.9919262,
                            "web": {
                                "uri": "https://www.careerpower.in/fifa-world-cup-winners-list.html",
                                "title": "FIFA World Cup Winners List from 1930 to 2022, Complete List - Career Power",
                            },
                        },
                        {
                            "segment": {"startIndex": 39, "endIndex": 187},
                            "confidenceScore": 0.9919262,
                            "web": {
                                "uri": "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_final",
                                "title": "2022 FIFA World Cup final - Wikipedia",
                            },
                        },
                    ],
                    "searchEntryPoint": {
                        "renderedContent": '\u003cstyle\u003e\n.container {\n  align-items: center;\n  border-radius: 8px;\n  display: flex;\n  font-family: Google Sans, Roboto, sans-serif;\n  font-size: 14px;\n  line-height: 20px;\n  padding: 8px 12px;\n}\n.chip {\n  display: inline-block;\n  border: solid 1px;\n  border-radius: 16px;\n  min-width: 14px;\n  padding: 5px 16px;\n  text-align: center;\n  user-select: none;\n  margin: 0 8px;\n  -webkit-tap-highlight-color: transparent;\n}\n.carousel {\n  overflow: auto;\n  scrollbar-width: none;\n  white-space: nowrap;\n  margin-right: -12px;\n}\n.headline {\n  display: flex;\n  margin-right: 4px;\n}\n.gradient-container {\n  position: relative;\n}\n.gradient {\n  position: absolute;\n  transform: translate(3px, -9px);\n  height: 36px;\n  width: 9px;\n}\n@media (prefers-color-scheme: light) {\n  .container {\n    background-color: #fafafa;\n    box-shadow: 0 0 0 1px #0000000f;\n  }\n  .headline-label {\n    color: #1f1f1f;\n  }\n  .chip {\n    background-color: #ffffff;\n    border-color: #d2d2d2;\n    color: #5e5e5e;\n    text-decoration: none;\n  }\n  .chip:hover {\n    background-color: #f2f2f2;\n  }\n  .chip:focus {\n    background-color: #f2f2f2;\n  }\n  .chip:active {\n    background-color: #d8d8d8;\n    border-color: #b6b6b6;\n  }\n  .logo-dark {\n    display: none;\n  }\n  .gradient {\n    background: linear-gradient(90deg, #fafafa 15%, #fafafa00 100%);\n  }\n}\n@media (prefers-color-scheme: dark) {\n  .container {\n    background-color: #1f1f1f;\n    box-shadow: 0 0 0 1px #ffffff26;\n  }\n  .headline-label {\n    color: #fff;\n  }\n  .chip {\n    background-color: #2c2c2c;\n    border-color: #3c4043;\n    color: #fff;\n    text-decoration: none;\n  }\n  .chip:hover {\n    background-color: #353536;\n  }\n  .chip:focus {\n    background-color: #353536;\n  }\n  .chip:active {\n    background-color: #464849;\n    border-color: #53575b;\n  }\n  .logo-light {\n    display: none;\n  }\n  .gradient {\n    background: linear-gradient(90deg, #1f1f1f 15%, #1f1f1f00 100%);\n  }\n}\n\u003c/style\u003e\n\u003cdiv class="container"\u003e\n  \u003cdiv class="headline"\u003e\n    \u003csvg class="logo-light" width="18" height="18" viewBox="9 9 35 35" fill="none" xmlns="http://www.w3.org/2000/svg"\u003e\n      \u003cpath fill-rule="evenodd" clip-rule="evenodd" d="M42.8622 27.0064C42.8622 25.7839 42.7525 24.6084 42.5487 23.4799H26.3109V30.1568H35.5897C35.1821 32.3041 33.9596 34.1222 32.1258 35.3448V39.6864H37.7213C40.9814 36.677 42.8622 32.2571 42.8622 27.0064V27.0064Z" fill="#4285F4"/\u003e\n      \u003cpath fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 43.8555C30.9659 43.8555 34.8687 42.3195 37.7213 39.6863L32.1258 35.3447C30.5898 36.3792 28.6306 37.0061 26.3109 37.0061C21.8282 37.0061 18.0195 33.9811 16.6559 29.906H10.9194V34.3573C13.7563 39.9841 19.5712 43.8555 26.3109 43.8555V43.8555Z" fill="#34A853"/\u003e\n      \u003cpath fill-rule="evenodd" clip-rule="evenodd" d="M16.6559 29.8904C16.3111 28.8559 16.1074 27.7588 16.1074 26.6146C16.1074 25.4704 16.3111 24.3733 16.6559 23.3388V18.8875H10.9194C9.74388 21.2072 9.06992 23.8247 9.06992 26.6146C9.06992 29.4045 9.74388 32.022 10.9194 34.3417L15.3864 30.8621L16.6559 29.8904V29.8904Z" fill="#FBBC05"/\u003e\n      \u003cpath fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 16.2386C28.85 16.2386 31.107 17.1164 32.9095 18.8091L37.8466 13.8719C34.853 11.082 30.9659 9.3736 26.3109 9.3736C19.5712 9.3736 13.7563 13.245 10.9194 18.8875L16.6559 23.3388C18.0195 19.2636 21.8282 16.2386 26.3109 16.2386V16.2386Z" fill="#EA4335"/\u003e\n    \u003c/svg\u003e\n    \u003csvg class="logo-dark" width="18" height="18" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"\u003e\n      \u003ccircle cx="24" cy="23" fill="#FFF" r="22"/\u003e\n      \u003cpath d="M33.76 34.26c2.75-2.56 4.49-6.37 4.49-11.26 0-.89-.08-1.84-.29-3H24.01v5.99h8.03c-.4 2.02-1.5 3.56-3.07 4.56v.75l3.91 2.97h.88z" fill="#4285F4"/\u003e\n      \u003cpath d="M15.58 25.77A8.845 8.845 0 0 0 24 31.86c1.92 0 3.62-.46 4.97-1.31l4.79 3.71C31.14 36.7 27.65 38 24 38c-5.93 0-11.01-3.4-13.45-8.36l.17-1.01 4.06-2.85h.8z" fill="#34A853"/\u003e\n      \u003cpath d="M15.59 20.21a8.864 8.864 0 0 0 0 5.58l-5.03 3.86c-.98-2-1.53-4.25-1.53-6.64 0-2.39.55-4.64 1.53-6.64l1-.22 3.81 2.98.22 1.08z" fill="#FBBC05"/\u003e\n      \u003cpath d="M24 14.14c2.11 0 4.02.75 5.52 1.98l4.36-4.36C31.22 9.43 27.81 8 24 8c-5.93 0-11.01 3.4-13.45 8.36l5.03 3.85A8.86 8.86 0 0 1 24 14.14z" fill="#EA4335"/\u003e\n    \u003c/svg\u003e\n    \u003cdiv class="gradient-container"\u003e\u003cdiv class="gradient"\u003e\u003c/div\u003e\u003c/div\u003e\n  \u003c/div\u003e\n  \u003cdiv class="carousel"\u003e\n    \u003ca class="chip" href="https://www.google.com/search?q=who+won+the+world+cup+2022&client=app-vertex-grounding&safesearch=active"\u003ewho won the world cup 2022\u003c/a\u003e\n  \u003c/div\u003e\n\u003c/div\u003e\n'
                    },
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 6,
            "candidatesTokenCount": 48,
            "totalTokenCount": 54,
        },
    }

    return mock_response


@pytest.mark.parametrize("value_in_dict", [{}, {"disable_attribution": False}])  #
def test_gemini_pro_grounding(value_in_dict):
    try:
        load_vertex_ai_credentials()
        litellm.set_verbose = True

        tools = [{"googleSearchRetrieval": value_in_dict}]

        litellm.set_verbose = True

        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        client = HTTPHandler()

        with patch.object(
            client, "post", side_effect=vertex_httpx_grounding_post
        ) as mock_call:
            resp = litellm.completion(
                model="vertex_ai_beta/gemini-1.0-pro-001",
                messages=[{"role": "user", "content": "Who won the world cup?"}],
                tools=tools,
                client=client,
            )

            mock_call.assert_called_once()

            print(mock_call.call_args.kwargs["json"]["tools"][0])

            assert (
                "googleSearchRetrieval"
                in mock_call.call_args.kwargs["json"]["tools"][0]
            )
            assert (
                mock_call.call_args.kwargs["json"]["tools"][0]["googleSearchRetrieval"]
                == value_in_dict
            )

            assert "vertex_ai_grounding_metadata" in resp._hidden_params
            assert isinstance(resp._hidden_params["vertex_ai_grounding_metadata"], list)

    except litellm.InternalServerError:
        pass
    except litellm.RateLimitError:
        pass


# @pytest.mark.skip(reason="exhausted vertex quota. need to refactor to mock the call")
@pytest.mark.parametrize(
    "model", ["vertex_ai_beta/gemini-1.5-pro", "vertex_ai/claude-3-sonnet@20240229"]
)  # "vertex_ai",
@pytest.mark.parametrize("sync_mode", [True])  # "vertex_ai",
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_gemini_pro_function_calling_httpx(model, sync_mode):
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
            "model": model,
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


from litellm.tests.test_completion import response_format_tests


@pytest.mark.parametrize(
    "model",
    [
        "vertex_ai/mistral-large@2407",
        "vertex_ai/mistral-nemo@2407",
        "vertex_ai/codestral@2405",
        "vertex_ai/meta/llama3-405b-instruct-maas",
    ],  #
)  # "vertex_ai",
@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)  #
@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.asyncio
async def test_partner_models_httpx(model, sync_mode):
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

        data = {
            "model": model,
            "messages": messages,
        }
        if sync_mode:
            response = litellm.completion(**data)
        else:
            response = await litellm.acompletion(**data)

        response_format_tests(response=response)

        print(f"response: {response}")

        assert isinstance(response._hidden_params["response_cost"], float)
    except litellm.RateLimitError as e:
        pass
    except litellm.InternalServerError as e:
        pass
    except Exception as e:
        if "429 Quota exceeded" in str(e):
            pass
        else:
            pytest.fail("An unexpected exception occurred - {}".format(str(e)))


@pytest.mark.parametrize(
    "model",
    [
        "vertex_ai/mistral-large@2407",
        "vertex_ai/meta/llama3-405b-instruct-maas",
    ],  #
)  # "vertex_ai",
@pytest.mark.parametrize(
    "sync_mode",
    [True, False],  #
)  #
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_partner_models_httpx_streaming(model, sync_mode):
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

        data = {"model": model, "messages": messages, "stream": True}
        if sync_mode:
            response = litellm.completion(**data)
            for idx, chunk in enumerate(response):
                streaming_format_tests(idx=idx, chunk=chunk)
        else:
            response = await litellm.acompletion(**data)
            idx = 0
            async for chunk in response:
                streaming_format_tests(idx=idx, chunk=chunk)
                idx += 1

        print(f"response: {response}")
    except litellm.RateLimitError as e:
        pass
    except litellm.InternalServerError as e:
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
@pytest.mark.flaky(retries=3, delay=1)
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


def vertex_httpx_mock_post_valid_response(*args, **kwargs):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": """{
                            "recipes": [
                                {"recipe_name": "Chocolate Chip Cookies"},
                                {"recipe_name": "Oatmeal Raisin Cookies"},
                                {"recipe_name": "Peanut Butter Cookies"},
                                {"recipe_name": "Sugar Cookies"},
                                {"recipe_name": "Snickerdoodles"}
                            ]
                            }"""
                        }
                    ],
                },
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.09790669,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.11736965,
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.1261379,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.08601588,
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.083441176,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.0355444,
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.071981624,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.08108212,
                    },
                ],
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 60,
            "candidatesTokenCount": 55,
            "totalTokenCount": 115,
        },
    }
    return mock_response


def vertex_httpx_mock_post_valid_response_anthropic(*args, **kwargs):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "id": "msg_vrtx_013Wki5RFQXAspL7rmxRFjZg",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20240620",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01YMnYZrToPPfcmY2myP2gEB",
                "name": "json_tool_call",
                "input": {
                    "values": {
                        "recipes": [
                            {"recipe_name": "Chocolate Chip Cookies"},
                            {"recipe_name": "Oatmeal Raisin Cookies"},
                            {"recipe_name": "Peanut Butter Cookies"},
                            {"recipe_name": "Snickerdoodle Cookies"},
                            {"recipe_name": "Sugar Cookies"},
                        ]
                    }
                },
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 368, "output_tokens": 118},
    }

    return mock_response


def vertex_httpx_mock_post_invalid_schema_response(*args, **kwargs):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": '[{"recipe_world": "Chocolate Chip Cookies"}]\n'}
                    ],
                },
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.09790669,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.11736965,
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.1261379,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.08601588,
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.083441176,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.0355444,
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.071981624,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.08108212,
                    },
                ],
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 60,
            "candidatesTokenCount": 55,
            "totalTokenCount": 115,
        },
    }
    return mock_response


def vertex_httpx_mock_post_invalid_schema_response_anthropic(*args, **kwargs):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "id": "msg_vrtx_013Wki5RFQXAspL7rmxRFjZg",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20240620",
        "content": [{"text": "Hi! My name is Claude.", "type": "text"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 368, "output_tokens": 118},
    }
    return mock_response


@pytest.mark.parametrize(
    "model, vertex_location, supports_response_schema",
    [
        ("vertex_ai_beta/gemini-1.5-pro-001", "us-central1", True),
        ("gemini/gemini-1.5-pro", None, True),
        ("vertex_ai_beta/gemini-1.5-flash", "us-central1", False),
        ("vertex_ai/claude-3-5-sonnet@20240620", "us-east5", False),
    ],
)
@pytest.mark.parametrize(
    "invalid_response",
    [True, False],
)
@pytest.mark.parametrize(
    "enforce_validation",
    [True, False],
)
@pytest.mark.asyncio
async def test_gemini_pro_json_schema_args_sent_httpx(
    model,
    supports_response_schema,
    vertex_location,
    invalid_response,
    enforce_validation,
):
    load_vertex_ai_credentials()
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    litellm.set_verbose = True
    messages = [{"role": "user", "content": "List 5 cookie recipes"}]
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    response_schema = {
        "type": "object",
        "properties": {
            "recipes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"recipe_name": {"type": "string"}},
                    "required": ["recipe_name"],
                },
            }
        },
        "required": ["recipes"],
        "additionalProperties": False,
    }

    client = HTTPHandler()

    httpx_response = MagicMock()
    if invalid_response is True:
        if "claude" in model:
            httpx_response.side_effect = (
                vertex_httpx_mock_post_invalid_schema_response_anthropic
            )
        else:
            httpx_response.side_effect = vertex_httpx_mock_post_invalid_schema_response
    else:
        if "claude" in model:
            httpx_response.side_effect = vertex_httpx_mock_post_valid_response_anthropic
        else:
            httpx_response.side_effect = vertex_httpx_mock_post_valid_response
    with patch.object(client, "post", new=httpx_response) as mock_call:
        print("SENDING CLIENT POST={}".format(client.post))
        try:
            resp = completion(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_object",
                    "response_schema": response_schema,
                    "enforce_validation": enforce_validation,
                },
                vertex_location=vertex_location,
                client=client,
            )
            print("Received={}".format(resp))
            if invalid_response is True and enforce_validation is True:
                pytest.fail("Expected this to fail")
        except litellm.JSONSchemaValidationError as e:
            if invalid_response is False:
                pytest.fail("Expected this to pass. Got={}".format(e))

        mock_call.assert_called_once()
        if "claude" not in model:
            print(mock_call.call_args.kwargs)
            print(mock_call.call_args.kwargs["json"]["generationConfig"])

            if supports_response_schema:
                assert (
                    "response_schema"
                    in mock_call.call_args.kwargs["json"]["generationConfig"]
                )
            else:
                assert (
                    "response_schema"
                    not in mock_call.call_args.kwargs["json"]["generationConfig"]
                )
                assert (
                    "Use this JSON schema:"
                    in mock_call.call_args.kwargs["json"]["contents"][0]["parts"][1][
                        "text"
                    ]
                )


@pytest.mark.parametrize(
    "model, vertex_location, supports_response_schema",
    [
        ("vertex_ai_beta/gemini-1.5-pro-001", "us-central1", True),
        ("gemini/gemini-1.5-pro", None, True),
        ("vertex_ai_beta/gemini-1.5-flash", "us-central1", False),
        ("vertex_ai/claude-3-5-sonnet@20240620", "us-east5", False),
    ],
)
@pytest.mark.parametrize(
    "invalid_response",
    [True, False],
)
@pytest.mark.parametrize(
    "enforce_validation",
    [True, False],
)
@pytest.mark.asyncio
async def test_gemini_pro_json_schema_args_sent_httpx_openai_schema(
    model,
    supports_response_schema,
    vertex_location,
    invalid_response,
    enforce_validation,
):
    from typing import List

    if enforce_validation:
        litellm.enable_json_schema_validation = True

    from pydantic import BaseModel

    load_vertex_ai_credentials()
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    litellm.set_verbose = True

    messages = [{"role": "user", "content": "List 5 cookie recipes"}]
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    class Recipe(BaseModel):
        recipe_name: str

    class ResponseSchema(BaseModel):
        recipes: List[Recipe]

    client = HTTPHandler()

    httpx_response = MagicMock()
    if invalid_response is True:
        if "claude" in model:
            httpx_response.side_effect = (
                vertex_httpx_mock_post_invalid_schema_response_anthropic
            )
        else:
            httpx_response.side_effect = vertex_httpx_mock_post_invalid_schema_response
    else:
        if "claude" in model:
            httpx_response.side_effect = vertex_httpx_mock_post_valid_response_anthropic
        else:
            httpx_response.side_effect = vertex_httpx_mock_post_valid_response
    with patch.object(client, "post", new=httpx_response) as mock_call:
        print("SENDING CLIENT POST={}".format(client.post))
        try:
            resp = completion(
                model=model,
                messages=messages,
                response_format=ResponseSchema,
                vertex_location=vertex_location,
                client=client,
            )
            print("Received={}".format(resp))
            if invalid_response is True and enforce_validation is True:
                pytest.fail("Expected this to fail")
        except litellm.JSONSchemaValidationError as e:
            if invalid_response is False:
                pytest.fail("Expected this to pass. Got={}".format(e))

        mock_call.assert_called_once()
        if "claude" not in model:
            print(mock_call.call_args.kwargs)
            print(mock_call.call_args.kwargs["json"]["generationConfig"])

            if supports_response_schema:
                assert (
                    "response_schema"
                    in mock_call.call_args.kwargs["json"]["generationConfig"]
                )
                assert (
                    "response_mime_type"
                    in mock_call.call_args.kwargs["json"]["generationConfig"]
                )
                assert (
                    mock_call.call_args.kwargs["json"]["generationConfig"][
                        "response_mime_type"
                    ]
                    == "application/json"
                )
            else:
                assert (
                    "response_schema"
                    not in mock_call.call_args.kwargs["json"]["generationConfig"]
                )
                assert (
                    "Use this JSON schema:"
                    in mock_call.call_args.kwargs["json"]["contents"][0]["parts"][1][
                        "text"
                    ]
                )


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
            traceback.print_exc()
            print("Receives error - {}".format(str(e)))

        mock_call.assert_called_once()

        assert "my-custom-api-base:generateContent" == mock_call.call_args.kwargs["url"]
        assert "hello" in mock_call.call_args.kwargs["headers"]


# @pytest.mark.skip(reason="exhausted vertex quota. need to refactor to mock the call")
@pytest.mark.parametrize("sync_mode", [True])
@pytest.mark.parametrize("provider", ["vertex_ai"])
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
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
@pytest.mark.flaky(retries=3, delay=1)
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
@pytest.mark.flaky(retries=3, delay=1)
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
        print(f"message content: {completion.choices[0].message.content}")
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


@pytest.mark.flaky(retries=3, delay=1)
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


@pytest.mark.asyncio
async def test_vertexai_multimodal_embedding():
    load_vertex_ai_credentials()
    mock_response = AsyncMock()

    def return_val():
        return {
            "predictions": [
                {
                    "imageEmbedding": [0.1, 0.2, 0.3],  # Simplified example
                    "textEmbedding": [0.4, 0.5, 0.6],  # Simplified example
                }
            ]
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    expected_payload = {
        "instances": [
            {
                "image": {
                    "gcsUri": "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"
                },
                "text": "this is a unicorn",
            }
        ]
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        # Act: Call the litellm.aembedding function
        response = await litellm.aembedding(
            model="vertex_ai/multimodalembedding@001",
            input=[
                {
                    "image": {
                        "gcsUri": "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"
                    },
                    "text": "this is a unicorn",
                },
            ],
        )

        # Assert
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        args_to_vertexai = kwargs["json"]

        print("args to vertex ai call:", args_to_vertexai)

        assert args_to_vertexai == expected_payload
        assert response.model == "multimodalembedding@001"
        assert len(response.data) == 1
        response_data = response.data[0]
        assert "imageEmbedding" in response_data
        assert "textEmbedding" in response_data

        # Optional: Print for debugging
        print("Arguments passed to Vertex AI:", args_to_vertexai)
        print("Response:", response)


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
@pytest.mark.flaky(retries=3, delay=1)
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


@pytest.mark.asyncio
def test_tool_name_conversion():
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
            "content": "27 degrees celsius and clear in San Francisco, CA",
        },
        # Now the assistant can reply with the result of the tool call.
    ]

    translated_messages = _gemini_convert_messages_with_history(messages=messages)

    print(f"\n\ntranslated_messages: {translated_messages}\ntranslated_messages")

    # assert that the last tool response has the corresponding tool name
    assert (
        translated_messages[-1]["parts"][0]["function_response"]["name"]
        == "get_weather"
    )


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


def test_get_token_url():
    from litellm.llms.vertex_ai_and_google_ai_studio.vertex_and_google_ai_studio_gemini import (
        VertexLLM,
    )

    vertex_llm = VertexLLM()
    vertex_ai_project = "adroit-crow-413218"
    vertex_ai_location = "us-central1"
    json_obj = get_vertex_ai_creds_json()
    vertex_credentials = json.dumps(json_obj)

    should_use_v1beta1_features = vertex_llm.is_using_v1beta1_features(
        optional_params={"cached_content": "hi"}
    )

    assert should_use_v1beta1_features is True

    _, url = vertex_llm._get_token_and_url(
        vertex_project=vertex_ai_project,
        vertex_location=vertex_ai_location,
        vertex_credentials=vertex_credentials,
        gemini_api_key="",
        custom_llm_provider="vertex_ai_beta",
        should_use_v1beta1_features=should_use_v1beta1_features,
        api_base=None,
        model="",
        stream=False,
    )

    print("url=", url)

    assert "/v1beta1/" in url

    should_use_v1beta1_features = vertex_llm.is_using_v1beta1_features(
        optional_params={"temperature": 0.1}
    )

    _, url = vertex_llm._get_token_and_url(
        vertex_project=vertex_ai_project,
        vertex_location=vertex_ai_location,
        vertex_credentials=vertex_credentials,
        gemini_api_key="",
        custom_llm_provider="vertex_ai_beta",
        should_use_v1beta1_features=should_use_v1beta1_features,
        api_base=None,
        model="",
        stream=False,
    )

    print("url for normal request", url)

    assert "v1beta1" not in url
    assert "/v1/" in url

    pass


@pytest.mark.asyncio
async def test_completion_fine_tuned_model():
    # load_vertex_ai_credentials()
    mock_response = AsyncMock()

    def return_val():
        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "text": "A canvas vast, a boundless blue,\nWhere clouds paint tales and winds imbue.\nThe sun descends in fiery hue,\nStars shimmer bright, a gentle few.\n\nThe moon ascends, a pearl of light,\nGuiding travelers through the night.\nThe sky embraces, holds all tight,\nA tapestry of wonder, bright."
                            }
                        ],
                    },
                    "finishReason": "STOP",
                    "safetyRatings": [
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "probability": "NEGLIGIBLE",
                            "probabilityScore": 0.028930664,
                            "severity": "HARM_SEVERITY_NEGLIGIBLE",
                            "severityScore": 0.041992188,
                        },
                        # ... other safety ratings ...
                    ],
                    "avgLogprobs": -0.95772853367765187,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 7,
                "candidatesTokenCount": 71,
                "totalTokenCount": 78,
            },
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    expected_payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "Write a short poem about the sky"}]}
        ],
        "generationConfig": {},
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        # Act: Call the litellm.completion function
        response = await litellm.acompletion(
            model="vertex_ai_beta/4965075652664360960",
            messages=[{"role": "user", "content": "Write a short poem about the sky"}],
        )

        # Assert
        mock_post.assert_called_once()
        url, kwargs = mock_post.call_args
        print("url = ", url)

        # this is the fine-tuned model endpoint
        assert (
            url[0]
            == "https://us-central1-aiplatform.googleapis.com/v1/projects/adroit-crow-413218/locations/us-central1/endpoints/4965075652664360960:generateContent"
        )

        print("call args = ", kwargs)
        args_to_vertexai = kwargs["json"]

        print("args to vertex ai call:", args_to_vertexai)

        assert args_to_vertexai == expected_payload
        assert response.choices[0].message.content.startswith("A canvas vast")
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.total_tokens == 78

        # Optional: Print for debugging
        print("Arguments passed to Vertex AI:", args_to_vertexai)
        print("Response:", response)


def mock_gemini_request(*args, **kwargs):
    print(f"kwargs: {kwargs}")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    if "cachedContents" in kwargs["url"]:
        mock_response.json.return_value = {
            "name": "cachedContents/4d2kd477o3pg",
            "model": "models/gemini-1.5-flash-001",
            "createTime": "2024-08-26T22:31:16.147190Z",
            "updateTime": "2024-08-26T22:31:16.147190Z",
            "expireTime": "2024-08-26T22:36:15.548934784Z",
            "displayName": "",
            "usageMetadata": {"totalTokenCount": 323383},
        }
    else:
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Please provide me with the text of the legal agreement"
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "MAX_TOKENS",
                    "index": 0,
                    "safetyRatings": [
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "probability": "NEGLIGIBLE",
                        },
                    ],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 40049,
                "candidatesTokenCount": 10,
                "totalTokenCount": 40059,
                "cachedContentTokenCount": 40012,
            },
        }

    return mock_response


gemini_context_caching_messages = [
    # System Message
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "Here is the full text of a complex legal agreement" * 4000,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    },
    # marked for caching with the cache_control parameter, so that this checkpoint can read from the previous cache.
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What are the key terms and conditions in this agreement?",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    },
    {
        "role": "assistant",
        "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
    },
    # The final turn is marked with cache-control, for continuing in followups.
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What are the key terms and conditions in this agreement?",
            }
        ],
    },
]


@pytest.mark.asyncio
async def test_gemini_context_caching_anthropic_format():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    litellm.set_verbose = True
    client = HTTPHandler(concurrent_limit=1)
    with patch.object(client, "post", side_effect=mock_gemini_request) as mock_client:
        try:
            response = litellm.completion(
                model="gemini/gemini-1.5-flash-001",
                messages=gemini_context_caching_messages,
                temperature=0.2,
                max_tokens=10,
                client=client,
            )

        except Exception as e:
            print(e)

        assert mock_client.call_count == 2

        first_call_args = mock_client.call_args_list[0].kwargs

        print(f"first_call_args: {first_call_args}")

        assert "cachedContents" in first_call_args["url"]

        # assert "cache_read_input_tokens" in response.usage
        # assert "cache_creation_input_tokens" in response.usage

        # # Assert either a cache entry was created or cache was read - changes depending on the anthropic api ttl
        # assert (response.usage.cache_read_input_tokens > 0) or (
        #     response.usage.cache_creation_input_tokens > 0
        # )

        check_cache_mock = MagicMock()
        client.get = check_cache_mock
        try:
            response = litellm.completion(
                model="gemini/gemini-1.5-flash-001",
                messages=gemini_context_caching_messages,
                temperature=0.2,
                max_tokens=10,
                client=client,
            )

        except Exception as e:
            print(e)

        check_cache_mock.assert_called_once()
        assert mock_client.call_count == 3


@pytest.mark.asyncio
async def test_partner_models_httpx_ai21():
    litellm.set_verbose = True
    model = "vertex_ai/jamba-1.5-mini@001"

    messages = [
        {
            "role": "system",
            "content": "Your name is Litellm Bot, you are a helpful assistant",
        },
        {
            "role": "user",
            "content": "Hello, can you tell me the weather in San Francisco?",
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
        "model": model,
        "messages": messages,
        "tools": tools,
        "top_p": 0.5,
    }

    mock_response = AsyncMock()

    def return_val():
        return {
            "id": "chat-3d11cf95eb224966937b216d9494fe73",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": " Sure, let me check that for you.",
                        "tool_calls": [
                            {
                                "id": "b5cef16b-5946-4937-b9d5-beeaea871e77",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 158,
                "completion_tokens": 36,
                "total_tokens": 194,
            },
            "meta": {"requestDurationMillis": 501},
            "model": "jamba-1.5",
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.acompletion(**data)

        # Assert
        mock_post.assert_called_once()
        url, kwargs = mock_post.call_args
        print("url = ", url)
        print("call args = ", kwargs)

        print(kwargs["data"])

        assert (
            url[0]
            == "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/adroit-crow-413218/locations/us-central1/publishers/ai21/models/jamba-1.5-mini@001:rawPredict"
        )

        # json loads kwargs
        kwargs["data"] = json.loads(kwargs["data"])

        assert kwargs["data"] == {
            "model": "jamba-1.5-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "Your name is Litellm Bot, you are a helpful assistant",
                },
                {
                    "role": "user",
                    "content": "Hello, can you tell me the weather in San Francisco?",
                },
            ],
            "top_p": 0.5,
            "tools": [
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
            ],
            "stream": False,
        }

        assert response.id == "chat-3d11cf95eb224966937b216d9494fe73"
        assert len(response.choices) == 1
        assert (
            response.choices[0].message.content == " Sure, let me check that for you."
        )
        assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
        assert (
            response.choices[0].message.tool_calls[0].function.arguments
            == '{"location": "San Francisco"}'
        )
        assert response.usage.prompt_tokens == 158
        assert response.usage.completion_tokens == 36
        assert response.usage.total_tokens == 194

        print(f"response: {response}")
