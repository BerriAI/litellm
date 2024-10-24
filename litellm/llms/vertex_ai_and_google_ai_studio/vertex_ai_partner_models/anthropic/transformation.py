# What is this?
## Handler file for calling claude-3 on vertex ai
import copy
import json
import os
import time
import types
import uuid
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.anthropic import (
    AnthropicMessagesTool,
    AnthropicMessagesToolChoice,
)
from litellm.types.llms.openai import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from litellm.types.utils import ResponseFormatChunk
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage

from ....prompt_templates.factory import (
    construct_tool_use_system_prompt,
    contains_tag,
    custom_prompt,
    extract_between_tags,
    parse_xml_params,
    prompt_factory,
    response_schema_prompt,
)


class VertexAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url=" https://cloud.google.com/vertex-ai/"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class VertexAIAnthropicConfig:
    """
    Reference:https://docs.anthropic.com/claude/reference/messages_post

    Note that the API for Claude on Vertex differs from the Anthropic API documentation in the following ways:

    - `model` is not a valid parameter. The model is instead specified in the Google Cloud endpoint URL.
    - `anthropic_version` is a required parameter and must be set to "vertex-2023-10-16".

    The class `VertexAIAnthropicConfig` provides configuration for the VertexAI's Anthropic API interface. Below are the parameters:

    - `max_tokens` Required (integer) max tokens,
    - `anthropic_version` Required (string) version of anthropic for bedrock - e.g. "bedrock-2023-05-31"
    - `system` Optional (string) the system prompt, conversion from openai format to this is handled in factory.py
    - `temperature` Optional (float) The amount of randomness injected into the response
    - `top_p` Optional (float) Use nucleus sampling.
    - `top_k` Optional (int) Only sample from the top K options for each subsequent token
    - `stop_sequences` Optional (List[str]) Custom text sequences that cause the model to stop generating

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    max_tokens: Optional[int] = (
        4096  # anthropic max - setting this doesn't impact response, but is required by anthropic.
    )
    system: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        anthropic_version: Optional[str] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key == "max_tokens" and value is None:
                value = self.max_tokens
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self):
        return [
            "max_tokens",
            "max_completion_tokens",
            "tools",
            "tool_choice",
            "stream",
            "stop",
            "temperature",
            "top_p",
            "response_format",
        ]

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "max_tokens" or param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "tools":
                optional_params["tools"] = value
            if param == "tool_choice":
                _tool_choice: Optional[AnthropicMessagesToolChoice] = None
                if value == "auto":
                    _tool_choice = {"type": "auto"}
                elif value == "required":
                    _tool_choice = {"type": "any"}
                elif isinstance(value, dict):
                    _tool_choice = {"type": "tool", "name": value["function"]["name"]}

                if _tool_choice is not None:
                    optional_params["tool_choice"] = _tool_choice
            if param == "stream":
                optional_params["stream"] = value
            if param == "stop":
                optional_params["stop_sequences"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "response_format" and isinstance(value, dict):
                json_schema: Optional[dict] = None
                if "response_schema" in value:
                    json_schema = value["response_schema"]
                elif "json_schema" in value:
                    json_schema = value["json_schema"]["schema"]
                """
                When using tools in this way: - https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode
                - You usually want to provide a single tool
                - You should set tool_choice (see Forcing tool use) to instruct the model to explicitly use that tool
                - Remember that the model will pass the input to the tool, so the name of the tool and description should be from the model’s perspective.
                """
                _tool_choice = None
                _tool_choice = {"name": "json_tool_call", "type": "tool"}

                _tool = AnthropicMessagesTool(
                    name="json_tool_call",
                    input_schema={
                        "type": "object",
                        "properties": {"values": json_schema},  # type: ignore
                    },
                )

                optional_params["tools"] = [_tool]
                optional_params["tool_choice"] = _tool_choice
                optional_params["json_mode"] = True

        return optional_params

    @classmethod
    def is_supported_model(
        cls, model: str, custom_llm_provider: Optional[str] = None
    ) -> bool:
        """
        Check if the model is supported by the VertexAI Anthropic API.
        """
        if custom_llm_provider == "vertex_ai" and "claude" in model.lower():
            return True
        elif model in litellm.vertex_anthropic_models:
            return True
        return False
