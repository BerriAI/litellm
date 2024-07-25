# What is this?
## Handler file for a Custom Chat LLM

"""
- completion
- acompletion
- streaming
- async_streaming
"""

import copy
import json
import os
import time
import types
from enum import Enum
from functools import partial
from typing import Callable, List, Literal, Optional, Tuple, Union

import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.databricks import GenericStreamingChunk
from litellm.types.utils import ProviderField
from litellm.utils import CustomStreamWrapper, EmbeddingResponse, ModelResponse, Usage

from .base import BaseLLM
from .prompt_templates.factory import custom_prompt, prompt_factory


class CustomLLMError(Exception):  # use this for all your exceptions
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


def custom_chat_llm_router():
    """
    Routes call to CustomLLM completion/acompletion/streaming/astreaming functions, based on call type

    Validates if response is in expected format
    """
    pass


class CustomLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    def completion(self, *args, **kwargs) -> ModelResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    def streaming(self, *args, **kwargs):
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        raise CustomLLMError(status_code=500, message="Not implemented yet!")

    async def astreaming(self, *args, **kwargs):
        raise CustomLLMError(status_code=500, message="Not implemented yet!")
