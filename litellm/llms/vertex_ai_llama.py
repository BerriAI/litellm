# What is this?
## Handler for calling llama 3.1 API on Vertex AI
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

from .base import BaseLLM
from .prompt_templates.factory import (
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


class VertexAILlama3Config:
    """
    Reference:https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama#streaming

    The class `VertexAILlama3Config` provides configuration for the VertexAI's Llama API interface. Below are the parameters:

    - `max_tokens` Required (integer) max tokens,

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    max_tokens: Optional[int] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
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
            "stream",
        ]

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens"] = value

        return optional_params


class VertexAILlama3(BaseLLM):
    def __init__(self) -> None:
        pass

    def create_vertex_llama3_url(
        self, vertex_location: str, vertex_project: str
    ) -> str:
        return f"https://{vertex_location}-aiplatform.googleapis.com/v1beta1/projects/{vertex_project}/locations/{vertex_location}/endpoints/openapi"

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        optional_params: dict,
        custom_prompt_dict: dict,
        headers: Optional[dict],
        timeout: Union[float, httpx.Timeout],
        vertex_project=None,
        vertex_location=None,
        vertex_credentials=None,
        litellm_params=None,
        logger_fn=None,
        acompletion: bool = False,
        client=None,
    ):
        try:
            import vertexai
            from google.cloud import aiplatform

            from litellm.llms.openai import OpenAIChatCompletion
            from litellm.llms.vertex_httpx import VertexLLM
        except Exception:

            raise VertexAIError(
                status_code=400,
                message="""vertexai import failed please run `pip install -U "google-cloud-aiplatform>=1.38"`""",
            )

        if not (
            hasattr(vertexai, "preview") or hasattr(vertexai.preview, "language_models")
        ):
            raise VertexAIError(
                status_code=400,
                message="""Upgrade vertex ai. Run `pip install "google-cloud-aiplatform>=1.38"`""",
            )
        try:

            vertex_httpx_logic = VertexLLM()

            access_token, project_id = vertex_httpx_logic._ensure_access_token(
                credentials=vertex_credentials, project_id=vertex_project
            )

            openai_chat_completions = OpenAIChatCompletion()

            ## Load Config
            # config = litellm.VertexAILlama3.get_config()
            # for k, v in config.items():
            #     if k not in optional_params:
            #         optional_params[k] = v

            ## CONSTRUCT API BASE
            stream: bool = optional_params.get("stream", False) or False

            optional_params["stream"] = stream

            api_base = self.create_vertex_llama3_url(
                vertex_location=vertex_location or "us-central1",
                vertex_project=vertex_project or project_id,
            )

            return openai_chat_completions.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                api_key=access_token,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                logging_obj=logging_obj,
                optional_params=optional_params,
                acompletion=acompletion,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                client=client,
                timeout=timeout,
            )

        except Exception as e:
            raise VertexAIError(status_code=500, message=str(e))
