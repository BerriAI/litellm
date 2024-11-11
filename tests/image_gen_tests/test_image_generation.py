# What this tests?
## This tests the litellm support for the openai /generations endpoint

import logging
import os
import sys
import traceback

from dotenv import load_dotenv
from openai.types.image import Image

logging.basicConfig(level=logging.DEBUG)
load_dotenv()
import asyncio
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
import json
import tempfile


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


def test_image_generation_openai():
    try:
        litellm.set_verbose = True
        response = litellm.image_generation(
            prompt="A cute baby sea otter", model="dall-e-3"
        )
        print(f"response: {response}")
        assert len(response.data) > 0
    except litellm.RateLimitError as e:
        pass
    except litellm.ContentPolicyViolationError:
        pass  # OpenAI randomly raises these errors - skip when they occur
    except Exception as e:
        if "Connection error" in str(e):
            pass
        pytest.fail(f"An exception occurred - {str(e)}")


# test_image_generation_openai()


@pytest.mark.parametrize(
    "sync_mode",
    [
        True,
    ],  # False
)  #
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_image_generation_azure(sync_mode):
    try:
        if sync_mode:
            response = litellm.image_generation(
                prompt="A cute baby sea otter",
                model="azure/",
                api_version="2023-06-01-preview",
            )
        else:
            response = await litellm.aimage_generation(
                prompt="A cute baby sea otter",
                model="azure/",
                api_version="2023-06-01-preview",
            )
        print(f"response: {response}")
        assert len(response.data) > 0
    except litellm.RateLimitError as e:
        pass
    except litellm.ContentPolicyViolationError:
        pass  # Azure randomly raises these errors - skip when they occur
    except litellm.InternalServerError:
        pass
    except Exception as e:
        if "Your task failed as a result of our safety system." in str(e):
            pass
        if "Connection error" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


# test_image_generation_azure()


@pytest.mark.flaky(retries=3, delay=1)
def test_image_generation_azure_dall_e_3():
    try:
        litellm.set_verbose = True
        response = litellm.image_generation(
            prompt="A cute baby sea otter",
            model="azure/dall-e-3-test",
            api_version="2023-12-01-preview",
            api_base=os.getenv("AZURE_SWEDEN_API_BASE"),
            api_key=os.getenv("AZURE_SWEDEN_API_KEY"),
        )
        print(f"response: {response}")
        assert len(response.data) > 0
    except litellm.InternalServerError as e:
        pass
    except litellm.ContentPolicyViolationError:
        pass  # OpenAI randomly raises these errors - skip when they occur
    except litellm.InternalServerError:
        pass
    except Exception as e:
        if "Your task failed as a result of our safety system." in str(e):
            pass
        if "Connection error" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


# test_image_generation_azure_dall_e_3()
@pytest.mark.asyncio
async def test_async_image_generation_openai():
    try:
        response = litellm.image_generation(
            prompt="A cute baby sea otter", model="dall-e-3"
        )
        print(f"response: {response}")
        assert len(response.data) > 0
    except litellm.APIError:
        pass
    except litellm.RateLimitError as e:
        pass
    except litellm.ContentPolicyViolationError:
        pass  # openai randomly raises these errors - skip when they occur
    except litellm.InternalServerError:
        pass
    except Exception as e:
        if "Connection error" in str(e):
            pass
        pytest.fail(f"An exception occurred - {str(e)}")


# asyncio.run(test_async_image_generation_openai())


@pytest.mark.asyncio
async def test_async_image_generation_azure():
    try:
        response = await litellm.aimage_generation(
            prompt="A cute baby sea otter",
            model="azure/dall-e-3-test",
            api_version="2023-09-01-preview",
        )
        print(f"response: {response}")
    except litellm.RateLimitError as e:
        pass
    except litellm.ContentPolicyViolationError:
        pass  # Azure randomly raises these errors - skip when they occur
    except litellm.InternalServerError:
        pass
    except Exception as e:
        if "Your task failed as a result of our safety system." in str(e):
            pass
        if "Connection error" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["bedrock/stability.sd3-large-v1:0", "bedrock/stability.stable-diffusion-xl-v1"],
)
def test_image_generation_bedrock(model):
    try:
        litellm.set_verbose = True
        response = litellm.image_generation(
            prompt="A cute baby sea otter",
            model=model,
            aws_region_name="us-west-2",
        )

        print(f"response: {response}")
        from openai.types.images_response import ImagesResponse

        ImagesResponse.model_validate(response.model_dump())
    except litellm.RateLimitError as e:
        pass
    except litellm.ContentPolicyViolationError:
        pass  # Azure randomly raises these errors - skip when they occur
    except Exception as e:
        if "Your task failed as a result of our safety system." in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


@pytest.mark.asyncio
async def test_aimage_generation_bedrock_with_optional_params():
    try:
        response = await litellm.aimage_generation(
            prompt="A cute baby sea otter",
            model="bedrock/stability.stable-diffusion-xl-v1",
            size="256x256",
        )
        print(f"response: {response}")
    except litellm.RateLimitError as e:
        pass
    except litellm.ContentPolicyViolationError:
        pass  # Azure randomly raises these errors skip when they occur
    except Exception as e:
        if "Your task failed as a result of our safety system." in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


from openai.types.image import Image


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_aimage_generation_vertex_ai(sync_mode):

    litellm.set_verbose = True

    load_vertex_ai_credentials()
    data = {
        "prompt": "An olympic size swimming pool",
        "model": "vertex_ai/imagegeneration@006",
        "vertex_ai_project": "adroit-crow-413218",
        "vertex_ai_location": "us-central1",
        "n": 1,
    }
    try:
        if sync_mode:
            response = litellm.image_generation(**data)
        else:
            response = await litellm.aimage_generation(**data)
        assert response.data is not None
        assert len(response.data) > 0

        for d in response.data:
            assert isinstance(d, Image)
            print("data in response.data", d)
            assert d.b64_json is not None
    except litellm.ServiceUnavailableError as e:
        pass
    except litellm.RateLimitError as e:
        pass
    except litellm.InternalServerError as e:
        pass
    except litellm.ContentPolicyViolationError:
        pass  # Azure randomly raises these errors - skip when they occur
    except Exception as e:
        if "Your task failed as a result of our safety system." in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")
