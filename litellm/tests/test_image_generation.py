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
    litellm.set_verbose = True
    response = litellm.image_generation(prompt="A cute baby sea otter", model="dall-e-3")
    print(f"response: {response}")
    assert len(response.data) > 0

# test_image_generation_openai()

def test_image_generation_azure(): 
    response = litellm.image_generation(prompt="A cute baby sea otter", model="azure/", api_version="2023-06-01-preview")
    print(f"response: {response}")
    assert len(response.data) > 0
# test_image_generation_azure()

def test_image_generation_azure_dall_e_3(): 
    litellm.set_verbose = True
    response = litellm.image_generation(prompt="A cute baby sea otter", model="azure/dall-e-3-test", api_version="2023-12-01-preview", api_base=os.getenv("AZURE_SWEDEN_API_BASE"), api_key=os.getenv("AZURE_SWEDEN_API_KEY"))
    print(f"response: {response}")
    assert len(response.data) > 0
# test_image_generation_azure_dall_e_3()
@pytest.mark.asyncio
async def test_async_image_generation_openai(): 
    response = litellm.image_generation(prompt="A cute baby sea otter", model="dall-e-3")
    print(f"response: {response}")
    assert len(response.data) > 0

# asyncio.run(test_async_image_generation_openai())

@pytest.mark.asyncio
async def test_async_image_generation_azure(): 
    response = await litellm.aimage_generation(prompt="A cute baby sea otter", model="azure/dall-e-3-test")
    print(f"response: {response}")