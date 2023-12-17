# What this tests? 
## This tests the litellm support for the openai /generations endpoint 

import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm 

def test_image_generation_openai(): 
    litellm.set_verbose = True
    response = litellm.image_generation(prompt="A cute baby sea otter", model="dall-e-3")
    print(f"response: {response}")

# test_image_generation_openai()

# def test_image_generation_azure(): 
#     response = litellm.image_generation(prompt="A cute baby sea otter", api_version="2023-06-01-preview", custom_llm_provider="azure")
#     print(f"response: {response}")
# test_image_generation_azure()

# @pytest.mark.asyncio
# async def test_async_image_generation_openai(): 
#     response = litellm.image_generation(prompt="A cute baby sea otter", model="dall-e-3")
#     print(f"response: {response}")

# @pytest.mark.asyncio
# async def test_async_image_generation_azure(): 
#     response = litellm.image_generation(prompt="A cute baby sea otter", model="azure/dall-e-3")
#     print(f"response: {response}")