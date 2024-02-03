# What this tests?
## This tests the litellm support for the openai /generations endpoint

import sys, os
import traceback
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.DEBUG)
load_dotenv()
import os
import asyncio

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
        pytest.fail(f"An exception occurred - {str(e)}")


# test_image_generation_openai()


def test_image_generation_azure():
    try:
        response = litellm.image_generation(
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
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


# test_image_generation_azure()


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
    except litellm.RateLimitError as e:
        pass
    except litellm.ContentPolicyViolationError:
        pass  # OpenAI randomly raises these errors - skip when they occur
    except Exception as e:
        if "Your task failed as a result of our safety system." in str(e):
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
    except litellm.RateLimitError as e:
        pass
    except litellm.ContentPolicyViolationError:
        pass  # openai randomly raises these errors - skip when they occur
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


# asyncio.run(test_async_image_generation_openai())


@pytest.mark.asyncio
async def test_async_image_generation_azure():
    try:
        response = await litellm.aimage_generation(
            prompt="A cute baby sea otter", model="azure/dall-e-3-test"
        )
        print(f"response: {response}")
    except litellm.RateLimitError as e:
        pass
    except litellm.ContentPolicyViolationError:
        pass  # Azure randomly raises these errors - skip when they occur
    except Exception as e:
        if "Your task failed as a result of our safety system." in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


def test_image_generation_bedrock():
    try:
        litellm.set_verbose = True
        response = litellm.image_generation(
            prompt="A cute baby sea otter",
            model="bedrock/stability.stable-diffusion-xl-v0",
            aws_region_name="us-east-1",
        )
        print(f"response: {response}")
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
            model="bedrock/stability.stable-diffusion-xl-v0",
            size="128x128",
        )
        print(f"response: {response}")
    except litellm.RateLimitError as e:
        pass
    except litellm.ContentPolicyViolationError:
        pass  # Azure randomly raises these errors - skip when they occur
    except Exception as e:
        if "Your task failed as a result of our safety system." in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")
