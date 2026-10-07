# What this tests?
## This tests the litellm support for the openai /generations endpoint

import logging
import os
import traceback
from unittest.mock import AsyncMock, MagicMock, patch

from dotenv import load_dotenv
from openai.types.image import Image

from litellm.caching import InMemoryCache

logging.basicConfig(level=logging.DEBUG)
load_dotenv()
import asyncio
import json
import logging
import tempfile

import pytest
from base_image_generation_test import BaseImageGenTest, TestCustomLogger

import litellm
from litellm._logging import verbose_logger

verbose_logger.setLevel(logging.DEBUG)


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


class TestVertexAIGeminiImageGeneration(BaseImageGenTest):
    """Test Gemini image generation models (Nano Banana)"""

    def get_base_image_generation_call_args(self) -> dict:
        # comment this when running locally
        load_vertex_ai_credentials()

        litellm.in_memory_llm_clients_cache = InMemoryCache()
        return {
            "model": "vertex_ai/gemini-2.5-flash-image",
            "vertex_ai_project": "litellm-ci-cd",
            "vertex_ai_location": "us-central1",
            "n": 1,
            "size": "1024x1024",
        }


class TestBedrockNovaCanvasTextToImage(BaseImageGenTest):
    def get_base_image_generation_call_args(self) -> dict:
        litellm.in_memory_llm_clients_cache = InMemoryCache()
        return {
            "model": "bedrock/amazon.nova-canvas-v1:0",
            "n": 1,
            "size": "320x320",
            "imageGenerationConfig": {"cfgScale": 6.5, "seed": 12},
            "taskType": "TEXT_IMAGE",
            "aws_region_name": "us-east-1",
        }


class TestBedrockNovaCanvasColorGuidedGeneration(BaseImageGenTest):
    def get_base_image_generation_call_args(self) -> dict:
        litellm.in_memory_llm_clients_cache = InMemoryCache()
        return {
            "model": "bedrock/amazon.nova-canvas-v1:0",
            "n": 1,
            "size": "320x320",
            "imageGenerationConfig": {"cfgScale": 6.5, "seed": 12},
            "taskType": "COLOR_GUIDED_GENERATION",
            "colorGuidedGenerationParams": {"colors": ["#FFFFFF"]},
            "aws_region_name": "us-east-1",
        }


class TestOpenAIGPTImage1(BaseImageGenTest):
    def get_base_image_generation_call_args(self) -> dict:
        return {"model": "gpt-image-1"}




class TestGoogleImageGen(BaseImageGenTest):
    def get_base_image_generation_call_args(self) -> dict:
        return {"model": "gemini/gemini-3.1-flash-image"}




## AZURE AI DALL-E 3 is deprecated and new deployments cannot be made
# class TestAzureOpenAIDalle3(BaseImageGenTest):
#     def get_base_image_generation_call_args(self) -> dict:
#         return {
#             "model": "azure/dall-e-3",
#             "api_version": "2024-02-01",
#             "api_base": os.getenv("AZURE_AI_API_BASE"),
#             "api_key": os.getenv("AZURE_AI_API_KEY"),
#             "metadata": {
#                 "model_info": {
#                     "base_model": "azure/dall-e-3",
#                 }
#             },
#         }
