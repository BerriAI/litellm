# What is this?
## Handler file for calling claude-3 on vertex ai
import copy
import json
import os
import time
import types
import uuid
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import ResponseFormatChunk
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage

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
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "tools":
                optional_params["tools"] = value
            if param == "stream":
                optional_params["stream"] = value
            if param == "stop":
                optional_params["stop_sequences"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "response_format" and "response_schema" in value:
                optional_params["response_format"] = ResponseFormatChunk(**value)  # type: ignore
        return optional_params


"""
- Run client init 
- Support async completion, streaming
"""


def refresh_auth(
    credentials,
) -> str:  # used when user passes in credentials as json string
    from google.auth.transport.requests import Request  # type: ignore[import-untyped]

    if credentials.token is None:
        credentials.refresh(Request())

    if not credentials.token:
        raise RuntimeError("Could not resolve API token from the credentials")

    return credentials.token


def get_vertex_client(
    client: Any,
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_credentials: Optional[str],
) -> Tuple[Any, Optional[str]]:
    args = locals()
    from litellm.llms.vertex_httpx import VertexLLM

    try:
        from anthropic import AnthropicVertex
    except Exception:
        raise VertexAIError(
            status_code=400,
            message="""vertexai import failed please run `pip install -U google-cloud-aiplatform "anthropic[vertex]"`""",
        )

    access_token: Optional[str] = None

    if client is None:
        _credentials, cred_project_id = VertexLLM().load_auth(
            credentials=vertex_credentials, project_id=vertex_project
        )
        vertex_ai_client = AnthropicVertex(
            project_id=vertex_project or cred_project_id,
            region=vertex_location or "us-central1",
            access_token=_credentials.token,
        )
    else:
        vertex_ai_client = client

    return vertex_ai_client, access_token


def completion(
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    logging_obj,
    optional_params: dict,
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
        from anthropic import AnthropicVertex
    except:
        raise VertexAIError(
            status_code=400,
            message="""vertexai import failed please run `pip install -U google-cloud-aiplatform "anthropic[vertex]"`""",
        )

    if not (
        hasattr(vertexai, "preview") or hasattr(vertexai.preview, "language_models")
    ):
        raise VertexAIError(
            status_code=400,
            message="""Upgrade vertex ai. Run `pip install "google-cloud-aiplatform>=1.38"`""",
        )
    try:

        vertex_ai_client, access_token = get_vertex_client(
            client=client,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
        )

        ## Load Config
        config = litellm.VertexAIAnthropicConfig.get_config()
        for k, v in config.items():
            if k not in optional_params:
                optional_params[k] = v

        ## Format Prompt
        _is_function_call = False
        _is_json_schema = False
        messages = copy.deepcopy(messages)
        optional_params = copy.deepcopy(optional_params)
        # Separate system prompt from rest of message
        system_prompt_indices = []
        system_prompt = ""
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                system_prompt += message["content"]
                system_prompt_indices.append(idx)
        if len(system_prompt_indices) > 0:
            for idx in reversed(system_prompt_indices):
                messages.pop(idx)
        if len(system_prompt) > 0:
            optional_params["system"] = system_prompt
        # Checks for 'response_schema' support - if passed in
        if "response_format" in optional_params:
            response_format_chunk = ResponseFormatChunk(
                **optional_params["response_format"]  # type: ignore
            )
            supports_response_schema = litellm.supports_response_schema(
                model=model, custom_llm_provider="vertex_ai"
            )
            if (
                supports_response_schema is False
                and response_format_chunk["type"] == "json_object"
                and "response_schema" in response_format_chunk
            ):
                _is_json_schema = True
                user_response_schema_message = response_schema_prompt(
                    model=model,
                    response_schema=response_format_chunk["response_schema"],
                )
                messages.append(
                    {"role": "user", "content": user_response_schema_message}
                )
                messages.append({"role": "assistant", "content": "{"})
                optional_params.pop("response_format")
        # Format rest of message according to anthropic guidelines
        try:
            messages = prompt_factory(
                model=model, messages=messages, custom_llm_provider="anthropic_xml"
            )
        except Exception as e:
            raise VertexAIError(status_code=400, message=str(e))

        ## Handle Tool Calling
        if "tools" in optional_params:
            _is_function_call = True
            tool_calling_system_prompt = construct_tool_use_system_prompt(
                tools=optional_params["tools"]
            )
            optional_params["system"] = (
                optional_params.get("system", "\n") + tool_calling_system_prompt
            )  # add the anthropic tool calling prompt to the system prompt
            optional_params.pop("tools")

        stream = optional_params.pop("stream", None)

        data = {
            "model": model,
            "messages": messages,
            **optional_params,
        }
        print_verbose(f"_is_function_call: {_is_function_call}")

        ## Completion Call

        print_verbose(
            f"VERTEX AI: vertex_project={vertex_project}; vertex_location={vertex_location}; vertex_credentials={vertex_credentials}"
        )

        if acompletion == True:
            """
            - async streaming
            - async completion
            """
            if stream is not None and stream == True:
                return async_streaming(
                    model=model,
                    messages=messages,
                    data=data,
                    print_verbose=print_verbose,
                    model_response=model_response,
                    logging_obj=logging_obj,
                    vertex_project=vertex_project,
                    vertex_location=vertex_location,
                    optional_params=optional_params,
                    client=client,
                    access_token=access_token,
                )
            else:
                return async_completion(
                    model=model,
                    messages=messages,
                    data=data,
                    print_verbose=print_verbose,
                    model_response=model_response,
                    logging_obj=logging_obj,
                    vertex_project=vertex_project,
                    vertex_location=vertex_location,
                    optional_params=optional_params,
                    client=client,
                    access_token=access_token,
                )
        if stream is not None and stream == True:
            ## LOGGING
            logging_obj.pre_call(
                input=messages,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                },
            )
            response = vertex_ai_client.messages.create(**data, stream=True)  # type: ignore
            return response

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
            },
        )

        message = vertex_ai_client.messages.create(**data)  # type: ignore

        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=message,
            additional_args={"complete_input_dict": data},
        )

        text_content: str = message.content[0].text
        ## TOOL CALLING - OUTPUT PARSE
        if text_content is not None and contains_tag("invoke", text_content):
            function_name = extract_between_tags("tool_name", text_content)[0]
            function_arguments_str = extract_between_tags("invoke", text_content)[
                0
            ].strip()
            function_arguments_str = f"<invoke>{function_arguments_str}</invoke>"
            function_arguments = parse_xml_params(function_arguments_str)
            _message = litellm.Message(
                tool_calls=[
                    {
                        "id": f"call_{uuid.uuid4()}",
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(function_arguments),
                        },
                    }
                ],
                content=None,
            )
            model_response.choices[0].message = _message  # type: ignore
        else:
            if (
                _is_json_schema
            ):  # follows https://github.com/anthropics/anthropic-cookbook/blob/main/misc/how_to_enable_json_mode.ipynb
                json_response = "{" + text_content[: text_content.rfind("}") + 1]
                model_response.choices[0].message.content = json_response  # type: ignore
            else:
                model_response.choices[0].message.content = text_content  # type: ignore
        model_response.choices[0].finish_reason = map_finish_reason(message.stop_reason)

        ## CALCULATING USAGE
        prompt_tokens = message.usage.input_tokens
        completion_tokens = message.usage.output_tokens

        model_response["created"] = int(time.time())
        model_response["model"] = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response
    except Exception as e:
        raise VertexAIError(status_code=500, message=str(e))


async def async_completion(
    model: str,
    messages: list,
    data: dict,
    model_response: ModelResponse,
    print_verbose: Callable,
    logging_obj,
    vertex_project=None,
    vertex_location=None,
    optional_params=None,
    client=None,
    access_token=None,
):
    from anthropic import AsyncAnthropicVertex

    if client is None:
        vertex_ai_client = AsyncAnthropicVertex(
            project_id=vertex_project, region=vertex_location, access_token=access_token
        )
    else:
        vertex_ai_client = client

    ## LOGGING
    logging_obj.pre_call(
        input=messages,
        api_key=None,
        additional_args={
            "complete_input_dict": optional_params,
        },
    )
    message = await vertex_ai_client.messages.create(**data)  # type: ignore
    text_content = message.content[0].text
    ## TOOL CALLING - OUTPUT PARSE
    if text_content is not None and contains_tag("invoke", text_content):
        function_name = extract_between_tags("tool_name", text_content)[0]
        function_arguments_str = extract_between_tags("invoke", text_content)[0].strip()
        function_arguments_str = f"<invoke>{function_arguments_str}</invoke>"
        function_arguments = parse_xml_params(function_arguments_str)
        _message = litellm.Message(
            tool_calls=[
                {
                    "id": f"call_{uuid.uuid4()}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(function_arguments),
                    },
                }
            ],
            content=None,
        )
        model_response.choices[0].message = _message  # type: ignore
    else:
        model_response.choices[0].message.content = text_content  # type: ignore
    model_response.choices[0].finish_reason = map_finish_reason(message.stop_reason)

    ## CALCULATING USAGE
    prompt_tokens = message.usage.input_tokens
    completion_tokens = message.usage.output_tokens

    model_response["created"] = int(time.time())
    model_response["model"] = model
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    setattr(model_response, "usage", usage)
    return model_response


async def async_streaming(
    model: str,
    messages: list,
    data: dict,
    model_response: ModelResponse,
    print_verbose: Callable,
    logging_obj,
    vertex_project=None,
    vertex_location=None,
    optional_params=None,
    client=None,
    access_token=None,
):
    from anthropic import AsyncAnthropicVertex

    if client is None:
        vertex_ai_client = AsyncAnthropicVertex(
            project_id=vertex_project, region=vertex_location, access_token=access_token
        )
    else:
        vertex_ai_client = client

    ## LOGGING
    logging_obj.pre_call(
        input=messages,
        api_key=None,
        additional_args={
            "complete_input_dict": optional_params,
        },
    )
    response = await vertex_ai_client.messages.create(**data, stream=True)  # type: ignore
    logging_obj.post_call(input=messages, api_key=None, original_response=response)

    streamwrapper = CustomStreamWrapper(
        completion_stream=response,
        model=model,
        custom_llm_provider="vertex_ai",
        logging_obj=logging_obj,
    )

    return streamwrapper
