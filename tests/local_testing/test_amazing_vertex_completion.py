import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

from test_streaming import streaming_format_tests

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from respx import MockRouter
import httpx

import pytest

import litellm
from litellm import (
    RateLimitError,
    Timeout,
    acompletion,
    completion,
    completion_cost,
    embedding,
    image_generation,
)
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase


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
    "gemini-pro-experimental",
    "gemini-flash-experimental",
    "gemini-2.5-flash-lite-exp-0827",
    "gemini-2.0-pro-exp-02-05",
    "gemini-pro-flash",
    "gemini-2.5-flash-lite-exp-0827",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-thinking-exp",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.0-flash-live-preview-04-09",
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
            model="gemini-2.5-flash-lite",
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


@pytest.mark.skip(
    reason="Local test. Vertex AI Quota is low. Leads to rate limit errors on ci/cd."
)
@pytest.mark.flaky(retries=3, delay=1)
def test_vertex_ai_anthropic_streaming():
    try:
        load_vertex_ai_credentials()

        # litellm.set_verbose = True

        model = "claude-3-5-sonnet@20240620"

        vertex_ai_project = "pathrise-convert-1606954137718"
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


@pytest.mark.skip(
    reason="Local test. Vertex AI Quota is low. Leads to rate limit errors on ci/cd."
)
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_aavertex_ai_anthropic_async():
    # load_vertex_ai_credentials()
    try:

        model = "claude-3-5-sonnet@20240620"

        vertex_ai_project = "pathrise-convert-1606954137718"
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


@pytest.mark.skip(
    reason="Local test. Vertex AI Quota is low. Leads to rate limit errors on ci/cd."
)
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_aaavertex_ai_anthropic_async_streaming():
    # load_vertex_ai_credentials()
    try:
        litellm.set_verbose = True
        model = "claude-3-5-sonnet@20240620"

        vertex_ai_project = "pathrise-convert-1606954137718"
        vertex_ai_location = "asia-southeast1"
        json_obj = get_vertex_ai_creds_json()
        vertex_credentials = json.dumps(json_obj)
        print(f"vertex_credentials: {vertex_credentials}")
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


@pytest.mark.skip(
    reason="Local test. Vertex AI Quota is low. Leads to rate limit errors on ci/cd."
)
@pytest.mark.flaky(retries=3, delay=1)
def test_avertex_ai():
    import random

    litellm.num_retries = 3
    load_vertex_ai_credentials()
    test_models = (
        litellm.vertex_chat_models
        | litellm.vertex_code_chat_models
        | litellm.vertex_text_models
        | litellm.vertex_code_text_models
    )
    litellm.set_verbose = False
    vertex_ai_project = "pathrise-convert-1606954137718"

    test_models = random.sample(list(test_models), 1)
    test_models += list(litellm.vertex_language_models)  # always test gemini-pro
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


@pytest.mark.skip(
    reason="Local test. Vertex AI Quota is low. Leads to rate limit errors on ci/cd."
)
@pytest.mark.flaky(retries=3, delay=1)
def test_avertex_ai_stream():
    load_vertex_ai_credentials()
    litellm.set_verbose = True
    litellm.vertex_project = "pathrise-convert-1606954137718"
    import random

    test_models = (
        litellm.vertex_chat_models
        | litellm.vertex_code_chat_models
        | litellm.vertex_text_models
        | litellm.vertex_code_text_models
    )
    test_models = random.sample(list(test_models), 1)
    test_models += list(litellm.vertex_language_models)  # always test gemini-pro
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
async def test_async_vertexai_response_basic():

    load_vertex_ai_credentials()
    try:
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        response = await acompletion(
            model="gemini-2.5-flash",
            messages=messages, 
            temperature=0.7, 
            timeout=5
        )
        print(f"response: {response}")
    except litellm.NotFoundError as e:
        pass
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




@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.asyncio
async def test_async_vertexai_streaming_response():
    import random

    litellm._turn_on_debug()

    load_vertex_ai_credentials()
    test_models = (
        litellm.vertex_chat_models
        | litellm.vertex_code_chat_models
        | litellm.vertex_text_models
        | litellm.vertex_code_text_models
    )
    test_models = random.sample(list(test_models), 1)
    test_models += list(litellm.vertex_language_models)  # always test gemini-pro
    test_models = ["gemini-2.5-flash"]
    for model in test_models:
        if model in VERTEX_MODELS_TO_NOT_TEST or (
            "gecko" in model
            or "32k" in model
            or "ultra" in model
            or "002" in model
            or "gemini-2.0-flash-thinking-exp" in model
            or "gemini-2.0-pro-exp-02-05" in model
            or "gemini-pro" in model
            or "gemini-1.0-pro" in model
            or "image-generation" in model
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
            complete_response: str = ""
            async for chunk in response:
                print(f"chunk: {chunk}")
                if chunk.choices[0].delta.content is not None:
                    complete_response += chunk.choices[0].delta.content
            print(f"complete_response: {complete_response}")
        except litellm.NotFoundError as e:
            pass
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
            model="vertex_ai_beta/gemini-2.5-flash-lite",
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
        image_path = "../proxy/cached_logo.jpg"
        # Getting the base64 string
        base64_image = encode_image(image_path)
        resp = litellm.completion(
            model="vertex_ai/gemini-1.5-pro",
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
    except litellm.InternalServerError:
        pass
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
@pytest.mark.parametrize("model", ["vertex_ai_beta/gemini-2.5-flash-lite"])  # "vertex_ai",
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
        print(f"Model for call - {model}")
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


from test_completion import response_format_tests


@pytest.mark.parametrize(
    "model,region",
    [
        ("vertex_ai/mistral-large-2411", "us-central1"),
        ("vertex_ai/qwen/qwen3-coder-480b-a35b-instruct-maas", "us-south1"),
        ("vertex_ai/openai/gpt-oss-20b-maas", "us-central1"),
    ],
)
@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)  #
@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.asyncio
async def test_partner_models_httpx(model, region, sync_mode):
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
            "timeout": 10,
            "vertex_ai_location": region,
        }
        if sync_mode:
            response = litellm.completion(**data)
        else:
            response = await litellm.acompletion(**data)

        response_format_tests(response=response)

        print(f"response: {response}")

        assert isinstance(response._hidden_params["response_cost"], float)
    except litellm.RateLimitError as e:
        print("RateLimitError", e)
        pass
    except litellm.Timeout as e:
        print("Timeout", e)
        pass
    except litellm.InternalServerError as e:
        print("InternalServerError", e)
        pass
    except litellm.APIConnectionError as e:
        print("APIConnectionError", e)
        pass
    except litellm.ServiceUnavailableError as e:
        print("ServiceUnavailableError", e)
        pass
    except Exception as e:
        print("got generic exception", e)
        if "429 Quota exceeded" in str(e):
            pass
        else:
            pytest.fail("An unexpected exception occurred - {}".format(str(e)))


@pytest.mark.parametrize(
    "model,region",
    [
        ("vertex_ai/meta/llama-4-scout-17b-16e-instruct-maas", "us-east5"),
        ("vertex_ai/qwen/qwen3-coder-480b-a35b-instruct-maas", "us-south1"),
        (
            "vertex_ai/mistral-large-2411",
            "us-central1",
        ),  # critical - we had this issue: https://github.com/BerriAI/litellm/issues/13888
        ("vertex_ai/openai/gpt-oss-20b-maas", "us-central1"),
    ],
)
@pytest.mark.parametrize(
    "sync_mode",
    [True, False],  #
)  #
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_partner_models_httpx_streaming(model, region, sync_mode):
    try:
        load_vertex_ai_credentials()
        litellm._turn_on_debug()

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
            "stream": True,
            "vertex_ai_location": region,
        }
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
def vertex_httpx_mock_post(url, data=None, json=None, headers=None, **kwargs):
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
            model="vertex_ai_beta/gemini-2.5-flash-lite",
            messages=messages,
            response_format={"type": "json_object"},
            client=client,
            logging_obj=ANY,
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
        ("vertex_ai_beta/gemini-2.5-flash-lite", "us-central1", True),
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
    resp = None
    with patch.object(client, "post", new=httpx_response) as mock_call:
        litellm.set_verbose = True
        print(f"model entering completion: {model}")

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
        elif resp is not None:

            assert resp.model == model.split("/")[1]


@pytest.mark.asyncio
async def test_anthropic_message_via_anthropic_messages():
    from litellm.llms.custom_httpx.llm_http_handler import AsyncHTTPHandler
    from unittest.mock import MagicMock, AsyncMock

    load_vertex_ai_credentials()
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.set_verbose = True
    client = AsyncHTTPHandler()

    httpx_response = AsyncMock()
    httpx_response.side_effect = vertex_httpx_mock_post_valid_response_anthropic

    call_1_kwargs = {}
    call_2_kwargs = {}
    with patch.object(client, "post", new=httpx_response) as mock_call:
        messages = [{"role": "user", "content": "List 5 cookie recipes"}]
        response = await litellm.anthropic_messages(
            model="vertex_ai/claude-3-5-sonnet@20240620",
            messages=messages,
            max_tokens=100,
            client=client,
        )

        print(f"response: {response}")
        assert mock_call.call_count == 1
        call_1_kwargs = mock_call.call_args.kwargs

    with patch.object(client, "post", new=httpx_response) as mock_call:
        response_2 = await litellm.acompletion(
            model="vertex_ai/claude-3-5-sonnet@20240620",
            messages=messages,
            max_tokens=100,
            client=client,
        )
        print(f"response_2: {response_2}")
        call_args = mock_call.call_args
        print(f"call_args: {call_args}")
        call_2_kwargs = mock_call.call_args.kwargs
        call_2_kwargs["url"] = call_args[0][0]

    """
    Compare Call 1 and Call 2

    Expect:
        - url 
        - headers
        - data / json

        to be the same, except for the Authorization header.
    """
    print(f"call_1_kwargs: {call_1_kwargs}")
    print(f"call_2_kwargs: {call_2_kwargs}")
    assert (
        call_1_kwargs["url"] == call_2_kwargs["url"]
    ), f"Expected url to be the same, but got {call_1_kwargs['url']} and Expected {call_2_kwargs['url']}"
    assert "Authorization".lower() in [
        k.lower() for k in call_1_kwargs["headers"].keys()
    ], f"Expected Authorization header to be present in call_1_kwargs, but got {call_1_kwargs['headers'].keys()}"
    assert "content-type".lower() in [
        k.lower() for k in call_1_kwargs["headers"].keys()
    ], f"Expected Content-Type header to be present in call_1_kwargs, but got {call_1_kwargs['headers'].keys()}"

    ## validate request body
    print(f"call 1 kwargs keys: {call_1_kwargs.keys()}")
    print(f"call_2_kwargs['json']: {type(call_2_kwargs['json'])}")
    print(f"call_1_kwargs['data']: {type(call_1_kwargs['data'])}")
    call_1_kwargs_data = json.loads(call_1_kwargs["data"])
    for k, v in call_2_kwargs["json"].items():
        assert (
            k in call_1_kwargs_data
        ), f"Expected {k} to be present in call_1_kwargs['data'], but got {call_1_kwargs_data.keys()}"


@pytest.mark.parametrize(
    "model, vertex_location, supports_response_schema",
    [
        ("vertex_ai_beta/gemini-1.5-pro-001", "us-central1", True),
        ("gemini/gemini-1.5-pro", None, True),
        ("vertex_ai_beta/gemini-2.5-flash-lite", "us-central1", True),
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


@pytest.mark.parametrize(
    "model", ["gemini-2.5-flash-lite", "claude-3-5-sonnet@20240620"]
)  # "vertex_ai",
@pytest.mark.asyncio
async def test_gemini_pro_httpx_custom_api_base(model):
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
                model="vertex_ai/{}".format(model),
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

        print(f"mock_call.call_args: {mock_call.call_args}")
        print(f"mock_call.call_args.kwargs: {mock_call.call_args.kwargs}")
        if "url" in mock_call.call_args.kwargs:
            assert (
                "my-custom-api-base:generateContent"
                == mock_call.call_args.kwargs["url"]
            )
        else:
            assert "my-custom-api-base:rawPredict" == mock_call.call_args[0][0]
        if "headers" in mock_call.call_args.kwargs:
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
            "model": "{}/gemini-2.5-flash-lite".format(provider),
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
        "model": "vertex_ai/gemini-2.5-flash-lite",
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
                assert isinstance(chunk, litellm.ModelResponseStream)
        else:
            response = await litellm.acompletion(**data)
            print(f"completion: {response}")

            assert isinstance(response, litellm.CustomStreamWrapper)

            async for chunk in response:
                print(f"chunk: {chunk}")
                chunks.append(chunk)
                assert isinstance(chunk, litellm.ModelResponseStream)

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


# asyncio.run(gemini_pro_async_function_calling())


@pytest.mark.skip(reason="need to get gecko permissions on vertex ai to run this test")
@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_vertexai_embedding(sync_mode):
    try:
        load_vertex_ai_credentials()
        litellm.set_verbose = True

        input_text = ["good morning from litellm", "this is another item"]

        if sync_mode:
            response = litellm.embedding(
                model="textembedding-gecko@001", input=input_text
            )
        else:
            response = await litellm.aembedding(
                model="textembedding-gecko@001", input=input_text
            )

        print(f"response: {response}")

        # Assert that the response is not None
        assert response is not None

        # Assert that the response contains embeddings
        assert hasattr(response, "data")
        assert len(response.data) == len(input_text)

        # Assert that each embedding is a non-empty list of floats
        for embedding in response.data:
            assert "embedding" in embedding
            assert isinstance(embedding["embedding"], list)
            assert len(embedding["embedding"]) > 0
            assert all(isinstance(x, float) for x in embedding["embedding"])

    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="need to get gecko permissions on vertex ai to run this test")
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

        # Optional: Print for debugging
        print("Arguments passed to Vertex AI:", args_to_vertexai)
        print("Response:", response)


@pytest.mark.skip(reason="need to get gecko permissions on vertex ai to run this test")
@pytest.mark.asyncio
async def test_vertexai_multimodal_embedding_text_input():
    load_vertex_ai_credentials()
    mock_response = AsyncMock()

    def return_val():
        return {
            "predictions": [
                {
                    "textEmbedding": [0.4, 0.5, 0.6],  # Simplified example
                }
            ]
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    expected_payload = {
        "instances": [
            {
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
                "this is a unicorn",
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
        assert response_data["embedding"] == [0.4, 0.5, 0.6]

        # Optional: Print for debugging
        print("Arguments passed to Vertex AI:", args_to_vertexai)
        print("Response:", response)


@pytest.mark.skip(reason="need to get gecko permissions on vertex ai to run this test")
@pytest.mark.asyncio
async def test_vertexai_multimodal_embedding_image_in_input():
    load_vertex_ai_credentials()
    mock_response = AsyncMock()

    def return_val():
        return {
            "predictions": [
                {
                    "imageEmbedding": [0.1, 0.2, 0.3],  # Simplified example
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
            input=["gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"],
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

        assert response_data["embedding"] == [0.1, 0.2, 0.3]

        # Optional: Print for debugging
        print("Arguments passed to Vertex AI:", args_to_vertexai)
        print("Response:", response)


@pytest.mark.skip(reason="need to get gecko permissions on vertex ai to run this test")
@pytest.mark.asyncio
async def test_vertexai_multimodal_embedding_base64image_in_input():
    import base64

    import requests

    load_vertex_ai_credentials()
    mock_response = AsyncMock()

    url = "https://dummyimage.com/100/100/fff&text=Test+image"
    response = requests.get(url)
    file_data = response.content

    encoded_file = base64.b64encode(file_data).decode("utf-8")
    base64_image = f"data:image/png;base64,{encoded_file}"

    def return_val():
        return {
            "predictions": [
                {
                    "imageEmbedding": [0.1, 0.2, 0.3],  # Simplified example
                }
            ]
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    expected_payload = {
        "instances": [
            {
                "image": {"bytesBase64Encoded": base64_image},
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
            input=[base64_image],
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

        assert response_data["embedding"] == [0.1, 0.2, 0.3]

        # Optional: Print for debugging
        print("Arguments passed to Vertex AI:", args_to_vertexai)
        print("Response:", response)


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


def test_vertexai_multimodalembedding_embedding_latest():
    try:
        import requests, base64

        load_vertex_ai_credentials()
        litellm._turn_on_debug()

        response = embedding(
            model="vertex_ai/multimodalembedding@001",
            input=["hi"],
            dimensions=1,
            auto_truncate=True,
            task_type="RETRIEVAL_QUERY",
        )

        print(f"response.usage: {response.usage}")
        assert response.usage is not None
        assert response.usage.prompt_tokens_details is not None

        assert response._hidden_params["response_cost"] > 0
        print(f"response:", response)
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


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


@pytest.mark.skip(reason="need to get gecko permissions on vertex ai to run this test")
@pytest.mark.flaky(retries=3, delay=1)
def test_vertexai_embedding_embedding_latest_input_type():
    try:
        load_vertex_ai_credentials()
        litellm.set_verbose = True

        response = embedding(
            model="vertex_ai/text-embedding-004",
            input=["hi"],
            input_type="RETRIEVAL_QUERY",
        )
        assert response.usage.prompt_tokens > 0
        print(f"response:", response)
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="need to get gecko permissions on vertex ai to run this test")
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




@pytest.mark.asyncio
async def test_completion_fine_tuned_model():
    load_vertex_ai_credentials()
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
            == "https://us-central1-aiplatform.googleapis.com/v1/projects/pathrise-convert-1606954137718/locations/us-central1/endpoints/4965075652664360960:generateContent"
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
            "model": "models/gemini-2.5-flash-lite-001",
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


def mock_gemini_list_request(*args, **kwargs):
    from litellm.types.llms.vertex_ai import (
        CachedContent,
        CachedContentListAllResponseBody,
    )

    print(f"kwargs: {kwargs}")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = CachedContentListAllResponseBody(
        cachedContents=[CachedContent(name="test", displayName="test")]
    )

    return mock_response


from litellm._uuid import uuid


@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)
@pytest.mark.asyncio
async def test_gemini_context_caching_anthropic_format(sync_mode):
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    litellm.set_verbose = True
    gemini_context_caching_messages = [
        # System Message
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement {}".format(
                        uuid.uuid4()
                    )
                    * 4000,
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
    if sync_mode:
        client = HTTPHandler(concurrent_limit=1)
    else:
        client = AsyncHTTPHandler(concurrent_limit=1)
    with patch.object(client, "post", side_effect=mock_gemini_request) as mock_client:
        try:
            if sync_mode:
                response = litellm.completion(
                    model="gemini/gemini-2.5-flash-lite-001",
                    messages=gemini_context_caching_messages,
                    temperature=0.2,
                    max_tokens=10,
                    client=client,
                )
            else:
                response = await litellm.acompletion(
                    model="gemini/gemini-2.5-flash-lite-001",
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
            "model": "jamba-1.5-mini@001",
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
            == "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/pathrise-convert-1606954137718/locations/us-central1/publishers/ai21/models/jamba-1.5-mini@001:rawPredict"
        )

        # json loads kwargs
        kwargs["data"] = json.loads(kwargs["data"])

        assert kwargs["data"] == {
            "model": "jamba-1.5-mini@001",
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


def test_gemini_function_call_parameter_in_messages():
    litellm.set_verbose = True
    load_vertex_ai_credentials()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Executes searches.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "description": "A list of queries to search for.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["queries"],
                },
            },
        },
    ]

    # Set up the messages
    messages = [
        {"role": "system", "content": """Use search for most queries."""},
        {"role": "user", "content": """search for weather in boston (use `search`)"""},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": "search",
                "arguments": '{"queries": ["weather in boston"]}',
            },
        },
        {
            "role": "function",
            "name": "search",
            "content": "The current weather in Boston is 22°F.",
        },
    ]

    client = HTTPHandler(concurrent_limit=1)

    with patch.object(client, "post", new=MagicMock()) as mock_client:
        try:
            response_stream = completion(
                model="vertex_ai/gemini-1.5-pro",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                client=client,
            )
        except Exception as e:
            print(e)

        # mock_client.assert_any_call()

        assert {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "search for weather in boston (use `search`)"}],
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "name": "search",
                                "args": {"queries": ["weather in boston"]},
                            }
                        }
                    ],
                },
                {
                    "parts": [
                        {
                            "function_response": {
                                "name": "search",
                                "response": {
                                    "content": "The current weather in Boston is 22°F."
                                },
                            }
                        }
                    ]
                },
            ],
            "system_instruction": {"parts": [{"text": "Use search for most queries."}]},
            "tools": [
                {
                    "function_declarations": [
                        {
                            "name": "search",
                            "description": "Executes searches.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "queries": {
                                        "type": "array",
                                        "description": "A list of queries to search for.",
                                        "items": {"type": "string"},
                                    }
                                },
                                "required": ["queries"],
                            },
                        }
                    ]
                }
            ],
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
            "generationConfig": {},
        } == mock_client.call_args.kwargs["json"]


def test_gemini_function_call_parameter_in_messages_2():
    litellm.set_verbose = True
    from litellm.llms.vertex_ai.gemini.transformation import (
        _gemini_convert_messages_with_history,
    )

    messages = [
        {"role": "user", "content": "search for weather in boston (use `search`)"},
        {
            "role": "assistant",
            "content": "Sure, let me check.",
            "function_call": {
                "name": "search",
                "arguments": '{"queries": ["weather in boston"]}',
            },
        },
        {
            "role": "function",
            "name": "search",
            "content": "The weather in Boston is 100 degrees.",
        },
    ]

    returned_contents = _gemini_convert_messages_with_history(messages=messages)

    print(f"returned_contents: {returned_contents}")
    assert returned_contents == [
        {
            "role": "user",
            "parts": [{"text": "search for weather in boston (use `search`)"}],
        },
        {
            "role": "model",
            "parts": [
                {"text": "Sure, let me check."},
                {
                    "function_call": {
                        "name": "search",
                        "args": {"queries": ["weather in boston"]},
                    }
                },
            ],
        },
        {
            "parts": [
                {
                    "function_response": {
                        "name": "search",
                        "response": {
                            "content": "The weather in Boston is 100 degrees."
                        },
                    }
                }
            ]
        },
    ]


@pytest.mark.parametrize(
    "base_model, metadata",
    [
        (None, {"model_info": {"base_model": "vertex_ai/gemini-1.5-pro"}}),
        ("vertex_ai/gemini-1.5-pro", None),
    ],
)
def test_gemini_finetuned_endpoint(base_model, metadata):
    litellm.set_verbose = True
    load_vertex_ai_credentials()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    # Set up the messages
    messages = [
        {"role": "system", "content": """Use search for most queries."""},
        {"role": "user", "content": """search for weather in boston (use `search`)"""},
    ]

    client = HTTPHandler(concurrent_limit=1)

    with patch.object(client, "post", new=MagicMock()) as mock_client:
        try:
            response = completion(
                model="vertex_ai/4965075652664360960",
                messages=messages,
                tool_choice="auto",
                client=client,
                metadata=metadata,
                base_model=base_model,
            )
        except Exception as e:
            print(e)

        print(mock_client.call_args.kwargs)

        mock_client.assert_called()
        assert mock_client.call_args.kwargs["url"].endswith(
            "endpoints/4965075652664360960:generateContent"
        )


@pytest.mark.parametrize("api_base", ["", None, "my-custom-proxy-base"])
def test_custom_api_base(api_base):
    stream = None
    test_endpoint = "my-fake-endpoint"
    vertex_base = VertexBase()
    auth_header, url = vertex_base._check_custom_proxy(
        api_base=api_base,
        custom_llm_provider="gemini",
        gemini_api_key="12324",
        endpoint="",
        stream=stream,
        auth_header=None,
        url="my-fake-endpoint",
        model="gemini-1.5-pro",  # Required for Gemini custom API base URLs
    )

    if api_base:
        # For Gemini with custom API base, URL should be constructed as api_base/models/model:endpoint
        expected_url = f"{api_base}/models/gemini-1.5-pro:"
        assert url == expected_url
    else:
        assert url == test_endpoint


@pytest.mark.asyncio
@pytest.mark.respx
async def test_vertexai_embedding_finetuned(respx_mock: MockRouter):
    """
    Tests that:
    - Request URL and body are correctly formatted for Vertex AI embeddings
    - Response is properly parsed into litellm's embedding response format
    """
    load_vertex_ai_credentials()
    litellm.set_verbose = True
    litellm.disable_aiohttp_transport = (
        True  # since this uses respx, we need to set use_aiohttp_transport to False
    )

    # Test input
    input_text = ["good morning from litellm", "this is another item"]

    # Expected request/response
    expected_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/633608382793/locations/us-central1/endpoints/1004708436694269952:predict"
    expected_request = {
        "instances": [
            {"inputs": "good morning from litellm"},
            {"inputs": "this is another item"},
        ],
        "parameters": {},
    }

    mock_response = {
        "predictions": [
            [[-0.000431762, -0.04416759, -0.03443353]],  # Truncated embedding vector
            [[-0.000431762, -0.04416759, -0.03443353]],  # Truncated embedding vector
        ],
        "deployedModelId": "2275167734310371328",
        "model": "projects/633608382793/locations/us-central1/models/snowflake-arctic-embed-m-long-1731622468876",
        "modelDisplayName": "snowflake-arctic-embed-m-long-1731622468876",
        "modelVersionId": "1",
    }

    # Setup mock request
    mock_request = respx_mock.post(expected_url).mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    # Make request
    response = await litellm.aembedding(
        vertex_project="633608382793",
        model="vertex_ai/1004708436694269952",
        input=input_text,
    )

    # Assert request was made correctly
    assert mock_request.called
    request_body = json.loads(mock_request.calls[0].request.content)
    print("\n\nrequest_body", request_body)
    print("\n\nexpected_request", expected_request)
    assert request_body == expected_request

    # Assert response structure
    assert response is not None
    assert hasattr(response, "data")
    assert len(response.data) == len(input_text)

    # Assert embedding structure
    for embedding in response.data:
        assert "embedding" in embedding
        assert isinstance(embedding["embedding"], list)
        assert len(embedding["embedding"]) > 0
        assert all(isinstance(x, float) for x in embedding["embedding"])


@pytest.mark.parametrize("max_retries", [None, 3])
@pytest.mark.asyncio
@pytest.mark.respx
async def test_vertexai_model_garden_model_completion(
    respx_mock: MockRouter, max_retries
):
    """
    Relevant issue: https://github.com/BerriAI/litellm/issues/6480

    Using OpenAI compatible models from Vertex Model Garden
    """
    litellm.disable_aiohttp_transport = (
        True  # since this uses respx, we need to set use_aiohttp_transport to False
    )
    litellm.module_level_aclient = httpx.AsyncClient()
    load_vertex_ai_credentials()
    litellm.set_verbose = True

    # Test input
    messages = [
        {
            "role": "system",
            "content": "Your name is Litellm Bot, you are a helpful assistant",
        },
        {
            "role": "user",
            "content": "Hello, what is your name and can you tell me the weather?",
        },
    ]

    # Expected request/response
    expected_url = "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/633608382793/locations/us-central1/endpoints/5464397967697903616/chat/completions"
    expected_request = {"model": "", "messages": messages, "stream": False}

    mock_response = {
        "id": "chat-09940d4e99e3488aa52a6f5e2ecf35b1",
        "object": "chat.completion",
        "created": 1731702782,
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, my name is Litellm Bot. I'm a helpful assistant here to provide information and answer your questions.\n\nTo check the weather for you, I'll need to know your location. Could you please provide me with your city or zip code? That way, I can give you the most accurate and up-to-date weather information.\n\nIf you don't have your location handy, I can also suggest some popular weather websites or apps that you can use to check the weather for your area.\n\nLet me know how I can assist you!",
                    "tool_calls": [],
                },
                "logprobs": None,
                "finish_reason": "stop",
                "stop_reason": None,
            }
        ],
        "usage": {"prompt_tokens": 63, "total_tokens": 172, "completion_tokens": 109},
        "prompt_logprobs": None,
    }

    # Setup mock request
    mock_request = respx_mock.post(expected_url).mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    # Make request
    response = await litellm.acompletion(
        model="vertex_ai/openai/5464397967697903616",
        messages=messages,
        vertex_project="633608382793",
        vertex_location="us-central1",
        max_retries=max_retries,
    )

    # Assert request was made correctly
    assert mock_request.called
    request_body = json.loads(mock_request.calls[0].request.content)
    assert request_body == expected_request

    # Assert response structure
    assert response.id == "chat-09940d4e99e3488aa52a6f5e2ecf35b1"
    assert response.created == 1731702782
    assert response.model == "vertex_ai/meta-llama/Llama-3.1-8B-Instruct"
    assert len(response.choices) == 1
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content.startswith(
        "Hello, my name is Litellm Bot"
    )
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.completion_tokens == 109
    assert response.usage.prompt_tokens == 63
    assert response.usage.total_tokens == 172


def vertex_ai_anthropic_thinking_mock_response(*args, **kwargs):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "id": "msg_vrtx_011pL6Np3MKxXL3R8theMRJW",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-7-sonnet-20250219",
        "content": [
            {
                "type": "thinking",
                "thinking": 'This is a very simple and common greeting in programming and computing. "Hello, world!" is often the first program people write when learning a new programming language, where they create a program that outputs this phrase.\n\nI should respond in a friendly way and acknowledge this greeting. I can keep it simple and welcoming.',
                "signature": "EugBCkYQAhgCIkAqCkezmsp8DG9Jjoc/CD7yXavPXVvP4TAuwjc/ZgHRIgroz5FzAYxic3CnNiW5w2fx/4+1f4ZYVxWJVLmrEA46EgwFsxbpN2jxMxjIzy0aDIAbMy9rW6B5lGVETCIw4r2UW0A7m5Df991SMSMPvHU9VdL8p9S/F2wajLnLVpl5tH89csm4NqnMpxnou61yKlCLldFGIto1Kvit5W1jqn2gx2dGIOyR4YaJ0c8AIFfQa5TIXf+EChVDzhPKLWZ8D/Q3gCGxBx+m/4dLI8HMZA8Ob3iCMI23eBKmh62FCWJGuA==",
            },
            {
                "type": "text",
                "text": "Hi there! 👋 \n\nIt's nice to meet you! \"Hello, world!\" is such a classic phrase in computing - it's often the first output from someone's very first program.\n\nHow are you doing today? Is there something specific I can help you with?",
            },
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 39,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 134,
        },
    }

    return mock_response


def test_vertex_anthropic_completion():
    from litellm import completion
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    load_vertex_ai_credentials()

    with patch.object(
        client, "post", side_effect=vertex_ai_anthropic_thinking_mock_response
    ):
        response = completion(
            model="vertex_ai/claude-3-7-sonnet@20250219",
            messages=[{"role": "user", "content": "Hello, world!"}],
            vertex_ai_location="us-east5",
            vertex_ai_project="test-project",
            thinking={"type": "enabled", "budget_tokens": 1024},
            client=client,
        )
        print(response)
        assert response.model == "claude-3-7-sonnet@20250219"
        assert response._hidden_params["response_cost"] is not None
        assert response._hidden_params["response_cost"] > 0

        assert response.choices[0].message.reasoning_content is not None
        assert isinstance(response.choices[0].message.reasoning_content, str)
        assert response.choices[0].message.thinking_blocks is not None
        assert isinstance(response.choices[0].message.thinking_blocks, list)
        assert len(response.choices[0].message.thinking_blocks) > 0


def test_signed_s3_url_with_format():
    from litellm import completion
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    load_vertex_ai_credentials()

    args = {
        "model": "vertex_ai/gemini-2.0-flash-001",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://litellm-logo-aws-marketplace.s3.us-west-2.amazonaws.com/berriai-logo-github.png?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjENj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJGMEQCIHlAy6QneghdEo4Dp4rw%2BHhdInKX4MU3T0hZT1qV3AD%2FAiBGY%2FtfxmBJkj%2BK6%2FxAgek6L3tpOcq6su1mBrj87El%2FCirLAwghEAEaDDg4ODYwMjIyMzQyOCIMzds7lsxAFHHCRHmkKqgDgnsJBaEmmwXBWqzyMMe3BUKsCqfvrYupFGxBREP%2BaEz%2ByLSKiTM3xWzaRz6vrP9T4HSJ97B9wQ3dhUBT22XzdOFsaq49wZapwy9hoPNrMyZ77DIa0MlEbg0uudGOaMAw4NbVEqoERQuZmIMMbNHCeoJsZxKCttRZlTDzU%2FeNNy96ltb%2FuIkX5b3OOYdUaKj%2FUjmPz%2FEufY%2Bn%2FFHawunSYXJwL4pYuBF1IKRtPjqamaYscH%2FrzD7fubGUMqk6hvyGEo%2BLqnVyruQEmVFqAnXyWlpHGqeWazEC7xcsC2lhLO%2FKUouyVML%2FxyYtL4CuKp52qtLWWauAFGnyBZnCHtSL58KLaMTSh7inhoFFIKDN2hymrJ4D9%2Bxv%2FMOzefH5X%2B0pcdJUwyxcwgL3myggRmIYq1L6IL4I%2F54BIU%2FMctJcRXQ8NhQNP2PsaCsXYHHVMXRZxps9v8t9Ciorb0PAaLr0DIGVgEqejSjwbzNTctQf59Rj0GhZ0A6A3nFaq3nL4UvO51aPP6aelN6RnLwHh8fF80iPWII7Oj9PWn9bkON%2F7%2B5k42oPFR0KDTD0yaO%2BBjrlAouRvkyHZnCuLuJdEeqc8%2Fwm4W8SbMiYDzIEPPe2wFR2sH4%2FDlnJRqia9Or00d4N%2BOefBkPv%2Bcdt68r%2FwjeWOrulczzLGjJE%2FGw1Lb9dtGtmupGm2XKOW3geJwXkk1qcr7u5zwy6DNamLJbitB026JFKorRnPajhe5axEDv%2BRu6l1f0eailIrCwZ2iytA94Ni8LTha2GbZvX7fFHcmtyNlgJPpMcELdkOEGTCNBldGck5MFHG27xrVrlR%2F7HZIkKYlImNmsOIjuK7acDiangvVdB6GlmVbzNUKtJ7YJhS2ivwvdDIf8XuaFAkhjRNpewDl0GzPvojK%2BDTizZydyJL%2B20pVkSXptyPwrrHEeiOFWwhszW2iTZij4rlRAoZW6NEdfkWsXrGMbxJTZa3E5URejJbg%2B4QgGtjLrgJhRC1pJGP02GX7VMxVWZzomfC2Hn7WaF44wgcuqjE4HGJfpA2ZLBxde52g%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA45ZGR4NCKIUOODV3%2F20250305%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20250305T235823Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=71a900a9467eaf3811553500aaf509a10a9e743a8133cfb6a78dcbcbc6da4a05",
                            "format": "image/jpeg",
                        },
                    },
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ],
    }
    with patch.object(client, "post", new=MagicMock()) as mock_client:
        try:
            response = completion(**args, client=client)
            print(response)
        except Exception as e:
            print(e)

        print(mock_client.call_args.kwargs)

        mock_client.assert_called()

        print(mock_client.call_args.kwargs)

        json_str = json.dumps(mock_client.call_args.kwargs["json"])
        assert "image/jpeg" in json_str
        assert "image/png" not in json_str


def test_gemini_fine_tuned_model_request_consistency():
    """
    Assert the same transformation is applied to Fine tuned gemini 2.0 flash and gemini 2.0 flash

    - Request 1: Fine tuned: vertex_ai/gemini/ft-uuid
    - Request 2: vertex_ai/gemini-2.0-flash-001
    """
    litellm.set_verbose = True
    load_vertex_ai_credentials()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from unittest.mock import patch, MagicMock

    # Set up the messages
    messages = [
        {
            "role": "system",
            "content": "Your name is Litellm Bot, you are a helpful assistant",
        },
        {
            "role": "user",
            "content": "Hello, what is your name and can you tell me the weather?",
        },
    ]

    # Define tools
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

    client = HTTPHandler(concurrent_limit=1)

    # First request
    with patch.object(client, "post", new=MagicMock()) as mock_post_1:
        try:
            response_1 = completion(
                model="vertex_ai/gemini/ft-uuid",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                client=client,
            )

        except Exception as e:
            print(e)

        # Store the request body from the first call
        first_request_body = mock_post_1.call_args.kwargs["json"]
        print("first_request_body", first_request_body)

        # Validate correct `model` is added to the request to Vertex AI
        print("final URL=", mock_post_1.call_args.kwargs["url"])
        # Validate the request url
        assert (
            "publishers/google/models/ft-uuid:generateContent"
            in mock_post_1.call_args.kwargs["url"]
        )

    # Second request
    with patch.object(client, "post", new=MagicMock()) as mock_post_2:
        try:
            response_2 = completion(
                model="vertex_ai/gemini-2.0-flash-001",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                client=client,
            )
        except Exception as e:
            print(e)

        # Store the request body from the second call
        second_request_body = mock_post_2.call_args.kwargs["json"]
        print("second_request_body", second_request_body)

    # Get the diff between the two request bodies
    # Convert dictionaries to formatted JSON strings
    import json

    first_json = json.dumps(first_request_body, indent=2).splitlines()
    second_json = json.dumps(second_request_body, indent=2).splitlines()
    # Assert there is no difference between the request bodies
    assert first_json == second_json, "Request bodies should be identical"


@pytest.mark.parametrize("provider", ["vertex_ai", "gemini"])
@pytest.mark.parametrize("route", ["completion", "embedding", "image_generation"])
def test_litellm_api_base(monkeypatch, provider, route):
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    import litellm

    monkeypatch.setattr(litellm, "api_base", "https://litellm.com")

    load_vertex_ai_credentials()

    if route == "image_generation" and provider == "gemini":
        pytest.skip("Gemini does not support image generation")

    with patch.object(client, "post", new=MagicMock()) as mock_client:
        try:
            if route == "completion":
                response = completion(
                    model=f"{provider}/gemini-2.0-flash-001",
                    messages=[{"role": "user", "content": "Hello, world!"}],
                    client=client,
                )
            elif route == "embedding":
                response = embedding(
                    model=f"{provider}/gemini-2.0-flash-001",
                    input=["Hello, world!"],
                    client=client,
                )
            elif route == "image_generation":
                response = image_generation(
                    model=f"{provider}/gemini-2.0-flash-001",
                    prompt="Hello, world!",
                    client=client,
                )
        except Exception as e:
            print(e)

        mock_client.assert_called()
        assert mock_client.call_args.kwargs["url"].startswith("https://litellm.com")


def test_gemini_tool_calling_working_demo():
    load_vertex_ai_credentials()
    litellm._turn_on_debug()
    args = {
        "messages": [
            {
                "content": "\n    You are a helpful assistant who can help with questions on customers business or personal finances.\n    Use the results from the available tools to answer the question.\n    ",
                "role": "system",
            },
            {"content": "Hello", "role": "user"},
        ],
        "max_completion_tokens": 1000,
        "temperature": 0.0,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_agent",
                    "description": "This tool helps find relevant help content",
                    "parameters": {
                        "properties": {
                            "state": {
                                "properties": {
                                    "messages": {
                                        "items": {"type": "object"},
                                        "type": "array",
                                    },
                                    "conversation_id": {"type": "string"},
                                },
                                "required": ["messages", "conversation_id"],
                                "type": "object",
                            },
                            "config": {
                                "description": "Configuration for a Runnable.",
                                "properties": {
                                    "tags": {
                                        "items": {"type": "string"},
                                        "type": "array",
                                    },
                                    "metadata": {"type": "object"},
                                    "callbacks": {
                                        "anyOf": [
                                            {"type": "array"},
                                            {"type": "object"},
                                            {"type": "null"},
                                        ],
                                    },
                                    "run_name": {"type": "string"},
                                    "max_concurrency": {
                                        "anyOf": [{"type": "integer"}, {"type": "null"}]
                                    },
                                    "recursion_limit": {"type": "integer"},
                                    "configurable": {"type": "object"},
                                    "run_id": {
                                        "anyOf": [
                                            {"format": "uuid", "type": "string"},
                                            {"type": "null"},
                                        ]
                                    },
                                },
                                "type": "object",
                            },
                            "kwargs": {"default": None, "type": "object"},
                        },
                        "required": ["state", "config"],
                        "type": "object",
                    },
                },
            }
        ],
    }
    response = completion(model="vertex_ai/gemini-2.0-flash", **args)
    print(response)


def test_gemini_tool_calling_not_working():
    load_vertex_ai_credentials()
    litellm._turn_on_debug()
    args = {
        "messages": [
            {
                "content": "\n    You are a helpful assistant who can help with questions on customers business or personal finances.\n    Use the results from the available tools to answer the question.\n    ",
                "role": "system",
            },
            {"content": "Hello", "role": "user"},
        ],
        "max_completion_tokens": 1000,
        "temperature": 0.0,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_agent",
                    "description": "This tool helps find relevant help content",
                    "parameters": {
                        "properties": {
                            "state": {
                                "properties": {
                                    "messages": {"items": {}, "type": "array"},
                                    "conversation_id": {"type": "string"},
                                },
                                "required": ["messages", "conversation_id"],
                                "type": "object",
                            },
                            "config": {
                                "description": "Configuration for a Runnable.",
                                "properties": {
                                    "tags": {
                                        "items": {"type": "string"},
                                        "type": "array",
                                    },
                                    "metadata": {"type": "object"},
                                    "callbacks": {
                                        "anyOf": [
                                            {"items": {}, "type": "array"},
                                            {},
                                            {"type": "null"},
                                        ]
                                    },
                                    "run_name": {"type": "string"},
                                    "max_concurrency": {
                                        "anyOf": [{"type": "integer"}, {"type": "null"}]
                                    },
                                    "recursion_limit": {"type": "integer"},
                                    "configurable": {"type": "object"},
                                    "run_id": {
                                        "anyOf": [
                                            {"format": "uuid", "type": "string"},
                                            {"type": "null"},
                                        ]
                                    },
                                },
                                "type": "object",
                            },
                            "kwargs": {"default": None, "type": "object"},
                        },
                        "required": ["state", "config"],
                        "type": "object",
                    },
                },
            }
        ],
    }
    response = completion(model="vertex_ai/gemini-2.0-flash", **args)
    print(response)


def test_vertex_ai_llama_tool_calling():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    load_vertex_ai_credentials()
    litellm._turn_on_debug()
    args = {
        "model": "vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas",
        "messages": [
            {"role": "user", "content": "What is the weather in Boston, MA today?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current temperature for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country e.g. Bogotá, Colombia",
                            }
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "vertex_location": "us-east5",
    }
    try:
        response = completion(**args)
    except litellm.RateLimitError:
        pytest.skip("Rate limit error")
    print(response)

    assert response.choices[0].message.tool_calls is not None
    assert response.choices[0].finish_reason == "tool_calls"
    assert response._hidden_params["response_cost"] > 0


def test_vertex_schema_test():
    load_vertex_ai_credentials()
    litellm._turn_on_debug()

    def tool_call(text: str | None) -> str:
        return text or "No text provided"

    tool = {
        "type": "function",
        "function": {
            "name": "git_create_branch",
            "description": "Creates a new branch from an optional base branch",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"title": "Repo Path", "type": "string"},
                    "branch_name": {"title": "Branch Name", "type": "string"},
                    "base_branch": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "title": "Base Branch",
                    },
                },
                "required": ["repo_path", "branch_name"],
                "title": "GitCreateBranch",
            },
        },
    }

    response = litellm.completion(
        model="vertex_ai/gemini-2.5-flash",
        messages=[{"role": "user", "content": "call the tool"}],
        tools=[tool],
        tool_choice="required",
    )

    print(response)


def test_vertex_ai_response_id():
    """Test that litellm preserves the response ID from Vertex AI's API for non-streaming responses"""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    load_vertex_ai_credentials()

    client = HTTPHandler()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "responseId": "vertex_ai_response_123",
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello! How can I help you today?"}],
                },
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                    }
                ],
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 8,
            "totalTokenCount": 18,
        },
    }

    with patch.object(client, "post", return_value=mock_response) as mock_post:
        response = completion(
            model="vertex_ai/gemini-1.5-pro",
            messages=[{"role": "user", "content": "Hi!"}],
            client=client,
        )

        # Verify the response ID is preserved
        assert response.id == "vertex_ai_response_123"
        assert response.choices[0].message.content == "Hello! How can I help you today?"


def test_vertex_ai_streaming_response_id():
    """Test that litellm preserves the response ID from Vertex AI's API for streaming responses"""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        make_sync_call,
    )

    load_vertex_ai_credentials()

    client = HTTPHandler()

    def mock_post(url, **kwargs):
        def stream_response():
            chunk = {
                "responseId": "vertex_ai_response_stream_123",
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Hello streaming!"}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 18,
                },
            }
            yield json.dumps(chunk)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines = MagicMock(return_value=stream_response())
        return mock_response

    logging_obj = MagicMock()

    with patch.object(client, "post", side_effect=mock_post):
        iterator = make_sync_call(
            client=client,
            gemini_client=None,
            api_base="https://mock-vertex-ai-api.com",
            headers={},
            data="{}",
            model="gemini-pro",
            messages=[],
            logging_obj=logging_obj,
        )
        iterator = iter(iterator)
        first_chunk = next(iterator)
        assert first_chunk.id == "vertex_ai_response_stream_123"


def test_vertex_ai_gemini_2_5_pro_streaming():
    load_vertex_ai_credentials()
    # litellm._turn_on_debug()
    response = completion(
        model="vertex_ai/gemini-2.5-pro",
        messages=[{"role": "user", "content": "Hi!"}],
        vertex_location="global",
        stream=True,
    )
    has_real_content = False
    for chunk in response:
        print(chunk)
        if (
            chunk.choices[0].delta.content is not None
            and len(chunk.choices[0].delta.content) > 0
        ):
            has_real_content = True
    assert has_real_content


def test_vertex_ai_gemini_audio_ogg():
    load_vertex_ai_credentials()
    litellm._turn_on_debug()
    response = completion(
        model="vertex_ai/gemini-2.0-flash",
        messages=[
            {
                "content": [
                    {"text": "generate a transcript of the speech.", "type": "text"}
                ],
                "role": "user",
            },
            {
                "content": [
                    {
                        "file": {
                            "file_id": "https://upload.wikimedia.org/wikipedia/commons/5/5f/En-us-public.ogg"
                        },
                        "type": "file",
                    }
                ],
                "role": "user",
            },
        ],
    )
    print(response)


@pytest.mark.asyncio
async def test_vertex_ai_deepseek():
    """Test that deepseek models use the correct v1 API endpoint instead of v1beta1."""
    # load_vertex_ai_credentials()
    litellm._turn_on_debug()
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    client = AsyncHTTPHandler()

    # Create a proper mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "model": "deepseek-ai/deepseek-r1-0528-maas",
    }
    mock_response.status_code = 200

    with patch.object(client, "post", return_value=mock_response) as mock_post:
        response = await acompletion(
            model="vertex_ai/deepseek-ai/deepseek-r1-0528-maas",
            messages=[{"role": "user", "content": "Hi!"}],
            client=client,
        )

        mock_post.assert_called_once()
        # Access the URL from kwargs since the call is made with keyword arguments
        url = mock_post.call_args.kwargs["url"]
        print(f"mock_post.call_args.kwargs['url']: {url}")
        assert "v1beta1" not in url
        assert "v1" in url


def test_gemini_grounding_on_streaming():
    from litellm import completion

    load_vertex_ai_credentials()
    # litellm._turn_on_debug()
    args = {
        "model": "vertex_ai/gemini-2.0-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the weather like on San Francisco today ?",
                    }
                ],
            }
        ],
        "stream": True,
        "tools": [{"googleSearch": {}}],
        "fallbacks": [],
    }

    result = completion(**args)
    vertex_ai_grounding_metadata_shows_up = False
    for chunk in result:
        if hasattr(chunk, "vertex_ai_grounding_metadata"):
            vertex_ai_grounding_metadata_shows_up = True
        print(chunk)
    assert vertex_ai_grounding_metadata_shows_up


def test_gemini_google_maps_tool_simple():
    """
    Test googleMaps tool with just enableWidget parameter.
    """
    load_vertex_ai_credentials()
    litellm._turn_on_debug()

    tools = [{"googleMaps": {"enableWidget": True}}]
    tools_with_location = [{"googleMaps": {"enableWidget": True, "latitude": 37.7749, "longitude": -122.4194, "languageCode": "en_US"}}]
    try:
        for tools in [tools, tools_with_location]:
            response = completion(
                model="vertex_ai/gemini-2.0-flash",
                messages=[
                    {
                        "role": "user",
                        "content": "What restaurants are nearby?",
                    }
                ],
                tools=tools,
            )
        print(f"Response: {response.model_dump_json(indent=4)}")
        assert response.choices[0].message.content is not None
    except litellm.RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

