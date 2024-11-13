import asyncio
import httpx
import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import CustomStreamWrapper
from openai.types.image import Image

# test_example.py
from abc import ABC, abstractmethod


class BaseImageGenTest(ABC):
    """
    Abstract base test class that enforces a common test across all test classes.
    """

    @abstractmethod
    def get_base_image_generation_call_args(self) -> dict:
        """Must return the base image generation call args"""
        pass

    @pytest.mark.asyncio(scope="module")
    async def test_basic_image_generation(self):
        """Test basic image generation"""
        try:
            base_image_generation_call_args = self.get_base_image_generation_call_args()
            litellm.set_verbose = True
            response = await litellm.aimage_generation(
                **base_image_generation_call_args, prompt="A image of a otter"
            )
            print(response)

            assert response._hidden_params["response_cost"] is not None
            print("response_cost", response._hidden_params["response_cost"])
            from openai.types.images_response import ImagesResponse

            ImagesResponse.model_validate(response.model_dump())

            for d in response.data:
                assert isinstance(d, Image)
                print("data in response.data", d)
                assert d.b64_json is not None or d.url is not None
        except litellm.RateLimitError as e:
            pass
        except litellm.ContentPolicyViolationError:
            pass  # Azure randomly raises these errors - skip when they occur
        except Exception as e:
            if "Your task failed as a result of our safety system." in str(e):
                pass
            else:
                pytest.fail(f"An exception occurred - {str(e)}")
