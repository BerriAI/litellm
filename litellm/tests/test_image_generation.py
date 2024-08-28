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
    except Exception as e:
        if "Your task failed as a result of our safety system." in str(e):
            pass
        if "Connection error" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


def test_image_generation_bedrock():
    try:
        litellm.set_verbose = True
        response = litellm.image_generation(
            prompt="A cute baby sea otter",
            model="bedrock/stability.stable-diffusion-xl-v1",
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
    from test_amazing_vertex_completion import load_vertex_ai_credentials

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
