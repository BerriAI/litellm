import asyncio
import httpx
import json
import pytest
import sys
from typing import Any, Dict, List, Optional
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
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload


class TestCustomLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.standard_logging_payload: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.standard_logging_payload = kwargs.get("standard_logging_object")
        pass


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
            custom_logger = TestCustomLogger()
            litellm.logging_callback_manager._reset_all_callbacks()
            litellm.callbacks = [custom_logger]
            base_image_generation_call_args = self.get_base_image_generation_call_args()
            litellm.set_verbose = True
            response = await litellm.aimage_generation(
                **base_image_generation_call_args, prompt="A image of a otter"
            )
            print(response)

            await asyncio.sleep(1)

            # assert response._hidden_params["response_cost"] is not None
            # assert response._hidden_params["response_cost"] > 0
            # print("response_cost", response._hidden_params["response_cost"])

            logged_standard_logging_payload = custom_logger.standard_logging_payload
            print("logged_standard_logging_payload", logged_standard_logging_payload)
            assert logged_standard_logging_payload is not None
            assert logged_standard_logging_payload["response_cost"] is not None
            assert logged_standard_logging_payload["response_cost"] > 0
            import openai
            from openai.types.images_response import ImagesResponse

            # print openai version
            print("openai version=", openai.__version__)

            response_dict = dict(response)
            if "usage" in response_dict:
                response_dict["usage"] = dict(response_dict["usage"])
            print("response usage=", response_dict.get("usage"))
            ImagesResponse.model_validate(response_dict)

            for d in response.data:
                assert isinstance(d, Image)
                print("data in response.data", d)
                assert d.b64_json is not None or d.url is not None
        except litellm.RateLimitError as e:
            pass
        except litellm.ContentPolicyViolationError:
            pass  # Azure randomly raises these errors - skip when they occur
        except litellm.InternalServerError:
            pass
        except Exception as e:
            if "Your task failed as a result of our safety system." in str(e):
                pass
            else:
                pytest.fail(f"An exception occurred - {str(e)}")