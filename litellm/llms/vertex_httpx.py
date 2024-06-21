# What is this?
## httpx client for vertex ai calls
## Initial implementation - covers gemini + image gen calls
import inspect
import json
import os
import time
import types
import uuid
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import httpx  # type: ignore
import ijson
import requests  # type: ignore

import litellm
import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging
from litellm import verbose_logger
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.prompt_templates.factory import convert_url_to_base64
from litellm.llms.vertex_ai import _gemini_convert_messages_with_history
from litellm.types.llms.openai import (
    ChatCompletionResponseMessage,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionUsageBlock,
)
from litellm.types.llms.vertex_ai import (
    ContentType,
    FunctionCallingConfig,
    FunctionDeclaration,
    GenerateContentResponseBody,
    GenerationConfig,
    PartType,
    RequestBody,
    SafetSettingsConfig,
    SystemInstructions,
    ToolConfig,
    Tools,
)
from litellm.types.utils import GenericStreamingChunk
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage

from .base import BaseLLM


class VertexGeminiConfig:
    """
    Reference: https://cloud.google.com/vertex-ai/docs/generative-ai/chat/test-chat-prompts
    Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference

    The class `VertexAIConfig` provides configuration for the VertexAI's API interface. Below are the parameters:

    - `temperature` (float): This controls the degree of randomness in token selection.

    - `max_output_tokens` (integer): This sets the limitation for the maximum amount of token in the text output. In this case, the default value is 256.

    - `top_p` (float): The tokens are selected from the most probable to the least probable until the sum of their probabilities equals the `top_p` value. Default is 0.95.

    - `top_k` (integer): The value of `top_k` determines how many of the most probable tokens are considered in the selection. For example, a `top_k` of 1 means the selected token is the most probable among all tokens. The default value is 40.

    - `response_mime_type` (str): The MIME type of the response. The default value is 'text/plain'.

    - `candidate_count` (int): Number of generated responses to return.

    - `stop_sequences` (List[str]): The set of character sequences (up to 5) that will stop output generation. If specified, the API will stop at the first appearance of a stop sequence. The stop sequence will not be included as part of the response.

    - `frequency_penalty` (float): This parameter is used to penalize the model from repeating the same output. The default value is 0.0.

    - `presence_penalty` (float): This parameter is used to penalize the model from generating the same output as the input. The default value is 0.0.

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    response_mime_type: Optional[str] = None
    candidate_count: Optional[int] = None
    stop_sequences: Optional[list] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        response_mime_type: Optional[str] = None,
        candidate_count: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
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
            "temperature",
            "top_p",
            "max_tokens",
            "stream",
            "tools",
            "tool_choice",
            "response_format",
            "n",
            "stop",
        ]

    def map_tool_choice_values(
        self, model: str, tool_choice: Union[str, dict]
    ) -> Optional[ToolConfig]:
        if tool_choice == "none":
            return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="NONE"))
        elif tool_choice == "required":
            return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="ANY"))
        elif tool_choice == "auto":
            return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="AUTO"))
        elif isinstance(tool_choice, dict):
            # only supported for anthropic + mistral models - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            name = tool_choice.get("function", {}).get("name", "")
            return ToolConfig(
                functionCallingConfig=FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[name]
                )
            )
        else:
            raise litellm.utils.UnsupportedParamsError(
                message="VertexAI doesn't support tool_choice={}. Supported tool_choice values=['auto', 'required', json object]. To drop it from the call, set `litellm.drop_params = True.".format(
                    tool_choice
                ),
                status_code=400,
            )

    def map_openai_params(
        self,
        model: str,
        non_default_params: dict,
        optional_params: dict,
    ):
        for param, value in non_default_params.items():
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if (
                param == "stream" and value is True
            ):  # sending stream = False, can cause it to get passed unchecked and raise issues
                optional_params["stream"] = value
            if param == "n":
                optional_params["candidate_count"] = value
            if param == "stop":
                if isinstance(value, str):
                    optional_params["stop_sequences"] = [value]
                elif isinstance(value, list):
                    optional_params["stop_sequences"] = value
            if param == "max_tokens":
                optional_params["max_output_tokens"] = value
            if param == "response_format" and value["type"] == "json_object":  # type: ignore
                optional_params["response_mime_type"] = "application/json"
            if param == "frequency_penalty":
                optional_params["frequency_penalty"] = value
            if param == "presence_penalty":
                optional_params["presence_penalty"] = value
            if param == "tools" and isinstance(value, list):
                gtool_func_declarations = []
                for tool in value:
                    gtool_func_declaration = FunctionDeclaration(
                        name=tool["function"]["name"],
                        description=tool["function"].get("description", ""),
                        parameters=tool["function"].get("parameters", {}),
                    )
                    gtool_func_declarations.append(gtool_func_declaration)
                optional_params["tools"] = [
                    Tools(function_declarations=gtool_func_declarations)
                ]
            if param == "tool_choice" and (
                isinstance(value, str) or isinstance(value, dict)
            ):
                _tool_choice_value = self.map_tool_choice_values(
                    model=model, tool_choice=value  # type: ignore
                )
                if _tool_choice_value is not None:
                    optional_params["tool_choice"] = _tool_choice_value
        return optional_params

    def get_mapped_special_auth_params(self) -> dict:
        """
        Common auth params across bedrock/vertex_ai/azure/watsonx
        """
        return {"project": "vertex_project", "region_name": "vertex_location"}

    def map_special_auth_params(self, non_default_params: dict, optional_params: dict):
        mapped_params = self.get_mapped_special_auth_params()

        for param, value in non_default_params.items():
            if param in mapped_params:
                optional_params[mapped_params[param]] = value
        return optional_params

    def get_eu_regions(self) -> List[str]:
        """
        Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
        """
        return [
            "europe-central2",
            "europe-north1",
            "europe-southwest1",
            "europe-west1",
            "europe-west2",
            "europe-west3",
            "europe-west4",
            "europe-west6",
            "europe-west8",
            "europe-west9",
        ]

    def get_flagged_finish_reasons(self) -> Dict[str, str]:
        """
        Return Dictionary of finish reasons which indicate response was flagged

        and what it means
        """
        return {
            "SAFETY": "The token generation was stopped as the response was flagged for safety reasons. NOTE: When streaming the Candidate.content will be empty if content filters blocked the output.",
            "RECITATION": "The token generation was stopped as the response was flagged for unauthorized citations.",
            "BLOCKLIST": "The token generation was stopped as the response was flagged for the terms which are included from the terminology blocklist.",
            "PROHIBITED_CONTENT": "The token generation was stopped as the response was flagged for the prohibited contents.",
            "SPII": "The token generation was stopped as the response was flagged for Sensitive Personally Identifiable Information (SPII) contents.",
        }


async def make_call(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
):
    if client is None:
        client = AsyncHTTPHandler()  # Create a new client if none provided

    response = await client.post(api_base, headers=headers, data=data, stream=True)

    if response.status_code != 200:
        raise VertexAIError(status_code=response.status_code, message=response.text)

    completion_stream = ModelResponseIterator(
        streaming_response=response.aiter_bytes(), sync_stream=False
    )
    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


def make_sync_call(
    client: Optional[HTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
):
    if client is None:
        client = HTTPHandler()  # Create a new client if none provided

    response = client.post(api_base, headers=headers, data=data, stream=True)

    if response.status_code != 200:
        raise VertexAIError(status_code=response.status_code, message=response.read())

    completion_stream = ModelResponseIterator(
        streaming_response=response.iter_bytes(chunk_size=2056), sync_stream=True
    )

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


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


class VertexLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__()
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self._credentials: Optional[Any] = None
        self.project_id: Optional[str] = None
        self.async_handler: Optional[AsyncHTTPHandler] = None

    def _process_response(
        self,
        model: str,
        response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: litellm.litellm_core_utils.litellm_logging.Logging,
        optional_params: dict,
        api_key: str,
        data: Union[dict, str],
        messages: List,
        print_verbose,
        encoding,
    ) -> ModelResponse:

        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )

        print_verbose(f"raw model_response: {response.text}")

        ## RESPONSE OBJECT
        try:
            completion_response = GenerateContentResponseBody(**response.json())  # type: ignore
        except Exception as e:
            raise VertexAIError(
                message="Received={}, Error converting to valid response block={}. File an issue if litellm error - https://github.com/BerriAI/litellm/issues".format(
                    response.text, str(e)
                ),
                status_code=422,
            )

        ## CHECK IF RESPONSE FLAGGED
        if len(completion_response["candidates"]) > 0:
            content_policy_violations = (
                VertexGeminiConfig().get_flagged_finish_reasons()
            )
            if (
                "finishReason" in completion_response["candidates"][0]
                and completion_response["candidates"][0]["finishReason"]
                in content_policy_violations.keys()
            ):
                ## CONTENT POLICY VIOLATION ERROR
                raise VertexAIError(
                    status_code=400,
                    message="The response was blocked. Reason={}. Raw Response={}".format(
                        content_policy_violations[
                            completion_response["candidates"][0]["finishReason"]
                        ],
                        completion_response,
                    ),
                )

        model_response.choices = []  # type: ignore

        ## GET MODEL ##
        model_response.model = model

        try:
            ## GET TEXT ##
            chat_completion_message: ChatCompletionResponseMessage = {
                "role": "assistant"
            }
            content_str = ""
            tools: List[ChatCompletionToolCallChunk] = []
            for idx, candidate in enumerate(completion_response["candidates"]):
                if "content" not in candidate:
                    continue

                if "text" in candidate["content"]["parts"][0]:
                    content_str = candidate["content"]["parts"][0]["text"]

                if "functionCall" in candidate["content"]["parts"][0]:
                    _function_chunk = ChatCompletionToolCallFunctionChunk(
                        name=candidate["content"]["parts"][0]["functionCall"]["name"],
                        arguments=json.dumps(
                            candidate["content"]["parts"][0]["functionCall"]["args"]
                        ),
                    )
                    _tool_response_chunk = ChatCompletionToolCallChunk(
                        id=f"call_{str(uuid.uuid4())}",
                        type="function",
                        function=_function_chunk,
                    )
                    tools.append(_tool_response_chunk)

                chat_completion_message["content"] = content_str
                chat_completion_message["tool_calls"] = tools

                choice = litellm.Choices(
                    finish_reason=candidate.get("finishReason", "stop"),
                    index=candidate.get("index", idx),
                    message=chat_completion_message,  # type: ignore
                    logprobs=None,
                    enhancements=None,
                )

                model_response.choices.append(choice)

            ## GET USAGE ##
            usage = litellm.Usage(
                prompt_tokens=completion_response["usageMetadata"]["promptTokenCount"],
                completion_tokens=completion_response["usageMetadata"][
                    "candidatesTokenCount"
                ],
                total_tokens=completion_response["usageMetadata"]["totalTokenCount"],
            )

            setattr(model_response, "usage", usage)
        except Exception as e:
            raise VertexAIError(
                message="Received={}, Error converting to valid response block={}. File an issue if litellm error - https://github.com/BerriAI/litellm/issues".format(
                    completion_response, str(e)
                ),
                status_code=422,
            )

        return model_response

    def get_vertex_region(self, vertex_region: Optional[str]) -> str:
        return vertex_region or "us-central1"

    def load_auth(
        self, credentials: Optional[str], project_id: Optional[str]
    ) -> Tuple[Any, str]:
        import google.auth as google_auth
        from google.auth.credentials import Credentials  # type: ignore[import-untyped]
        from google.auth.transport.requests import (
            Request,  # type: ignore[import-untyped]
        )

        if credentials is not None and isinstance(credentials, str):
            import google.oauth2.service_account

            json_obj = json.loads(credentials)

            creds = google.oauth2.service_account.Credentials.from_service_account_info(
                json_obj,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

            if project_id is None:
                project_id = creds.project_id
        else:
            creds, project_id = google_auth.default(
                quota_project_id=project_id,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        creds.refresh(Request())

        if not project_id:
            raise ValueError("Could not resolve project_id")

        if not isinstance(project_id, str):
            raise TypeError(
                f"Expected project_id to be a str but got {type(project_id)}"
            )

        return creds, project_id

    def refresh_auth(self, credentials: Any) -> None:
        from google.auth.transport.requests import (
            Request,  # type: ignore[import-untyped]
        )

        credentials.refresh(Request())

    def _ensure_access_token(
        self, credentials: Optional[str], project_id: Optional[str]
    ) -> Tuple[str, str]:
        """
        Returns auth token and project id
        """
        if self.access_token is not None and self.project_id is not None:
            return self.access_token, self.project_id

        if not self._credentials:
            self._credentials, project_id = self.load_auth(
                credentials=credentials, project_id=project_id
            )
            if not self.project_id:
                self.project_id = project_id
        else:
            self.refresh_auth(self._credentials)

            if not self.project_id:
                self.project_id = self._credentials.project_id

        if not self.project_id:
            raise ValueError("Could not resolve project_id")

        if not self._credentials or not self._credentials.token:
            raise RuntimeError("Could not resolve API token from the environment")

        return self._credentials.token, self.project_id

    def _get_token_and_url(
        self,
        model: str,
        gemini_api_key: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        api_base: Optional[str],
    ) -> Tuple[Optional[str], str]:
        """
        Internal function. Returns the token and url for the call.

        Handles logic if it's google ai studio vs. vertex ai.

        Returns
            token, url
        """
        if custom_llm_provider == "gemini":
            _gemini_model_name = "models/{}".format(model)
            auth_header = None
            endpoint = "generateContent"
            if stream is True:
                endpoint = "streamGenerateContent"

            url = (
                "https://generativelanguage.googleapis.com/v1beta/{}:{}?key={}".format(
                    _gemini_model_name, endpoint, gemini_api_key
                )
            )
        else:
            auth_header, vertex_project = self._ensure_access_token(
                credentials=vertex_credentials, project_id=vertex_project
            )
            vertex_location = self.get_vertex_region(vertex_region=vertex_location)

            ### SET RUNTIME ENDPOINT ###
            endpoint = "generateContent"
            if stream is True:
                endpoint = "streamGenerateContent"
            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:{endpoint}"

        if (
            api_base is not None
        ):  # for cloudflare ai gateway - https://github.com/BerriAI/litellm/issues/4317
            if custom_llm_provider == "gemini":
                url = "{}/{}".format(api_base, endpoint)
                auth_header = (
                    gemini_api_key  # cloudflare expects api key as bearer token
                )
            else:
                url = "{}:{}".format(api_base, endpoint)

        return auth_header, url

    async def async_streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        data: str,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        stream,
        optional_params: dict,
        litellm_params=None,
        logger_fn=None,
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
    ) -> CustomStreamWrapper:
        streaming_response = CustomStreamWrapper(
            completion_stream=None,
            make_call=partial(
                make_call,
                client=client,
                api_base=api_base,
                headers=headers,
                data=data,
                model=model,
                messages=messages,
                logging_obj=logging_obj,
            ),
            model=model,
            custom_llm_provider="vertex_ai_beta",
            logging_obj=logging_obj,
        )
        return streaming_response

    async def async_completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        data: str,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        stream,
        optional_params: dict,
        litellm_params=None,
        logger_fn=None,
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = AsyncHTTPHandler(**_params)  # type: ignore
        else:
            client = client  # type: ignore

        try:
            response = await client.post(api_base, headers=headers, json=data)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise VertexAIError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise VertexAIError(status_code=408, message="Timeout error occurred.")

        return self._process_response(
            model=model,
            response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key="",
            data=data,
            messages=messages,
            print_verbose=print_verbose,
            optional_params=optional_params,
            encoding=encoding,
        )

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
        encoding,
        logging_obj,
        optional_params: dict,
        acompletion: bool,
        timeout: Optional[Union[float, httpx.Timeout]],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[str],
        gemini_api_key: Optional[str],
        litellm_params=None,
        logger_fn=None,
        extra_headers: Optional[dict] = None,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
        api_base: Optional[str] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        stream: Optional[bool] = optional_params.pop("stream", None)  # type: ignore

        auth_header, url = self._get_token_and_url(
            model=model,
            gemini_api_key=gemini_api_key,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
        )

        ## TRANSFORMATION ##
        try:
            _custom_llm_provider = custom_llm_provider
            if custom_llm_provider == "vertex_ai_beta":
                _custom_llm_provider = "vertex_ai"
            supports_system_message = litellm.supports_system_messages(
                model=model, custom_llm_provider=_custom_llm_provider
            )
        except Exception as e:
            verbose_logger.error(
                "Unable to identify if system message supported. Defaulting to 'False'. Received error message - {}\nAdd it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json".format(
                    str(e)
                )
            )
            supports_system_message = False
        # Separate system prompt from rest of message
        system_prompt_indices = []
        system_content_blocks: List[PartType] = []
        if supports_system_message is True:
            for idx, message in enumerate(messages):
                if message["role"] == "system":
                    _system_content_block = PartType(text=message["content"])
                    system_content_blocks.append(_system_content_block)
                    system_prompt_indices.append(idx)
            if len(system_prompt_indices) > 0:
                for idx in reversed(system_prompt_indices):
                    messages.pop(idx)
        content = _gemini_convert_messages_with_history(messages=messages)
        tools: Optional[Tools] = optional_params.pop("tools", None)
        tool_choice: Optional[ToolConfig] = optional_params.pop("tool_choice", None)
        safety_settings: Optional[List[SafetSettingsConfig]] = optional_params.pop(
            "safety_settings", None
        )  # type: ignore
        generation_config: Optional[GenerationConfig] = GenerationConfig(
            **optional_params
        )
        data = RequestBody(contents=content)
        if len(system_content_blocks) > 0:
            system_instructions = SystemInstructions(parts=system_content_blocks)
            data["system_instruction"] = system_instructions
        if tools is not None:
            data["tools"] = tools
        if tool_choice is not None:
            data["toolConfig"] = tool_choice
        if safety_settings is not None:
            data["safetySettings"] = safety_settings
        if generation_config is not None:
            data["generationConfig"] = generation_config

        headers = {
            "Content-Type": "application/json; charset=utf-8",
        }
        if auth_header is not None:
            headers["Authorization"] = f"Bearer {auth_header}"
        if extra_headers is not None:
            headers.update(extra_headers)

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": headers,
            },
        )

        ### ROUTING (ASYNC, STREAMING, SYNC)
        if acompletion:
            ### ASYNC STREAMING
            if stream is True:
                return self.async_streaming(
                    model=model,
                    messages=messages,
                    data=json.dumps(data),  # type: ignore
                    api_base=url,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                    timeout=timeout,
                    client=client,  # type: ignore
                )
            ### ASYNC COMPLETION
            return self.async_completion(
                model=model,
                messages=messages,
                data=data,  # type: ignore
                api_base=url,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                stream=stream,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                headers=headers,
                timeout=timeout,
                client=client,  # type: ignore
            )

        ## SYNC STREAMING CALL ##
        if stream is not None and stream is True:
            streaming_response = CustomStreamWrapper(
                completion_stream=None,
                make_call=partial(
                    make_sync_call,
                    client=None,
                    api_base=url,
                    headers=headers,  # type: ignore
                    data=json.dumps(data),
                    model=model,
                    messages=messages,
                    logging_obj=logging_obj,
                ),
                model=model,
                custom_llm_provider="vertex_ai_beta",
                logging_obj=logging_obj,
            )

            return streaming_response
        ## COMPLETION CALL ##
        if client is None or isinstance(client, AsyncHTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = HTTPHandler(**_params)  # type: ignore
        else:
            client = client

        try:
            response = client.post(url=url, headers=headers, json=data)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise VertexAIError(status_code=error_code, message=response.text)
        except httpx.TimeoutException:
            raise VertexAIError(status_code=408, message="Timeout error occurred.")

        return self._process_response(
            model=model,
            response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params,
            api_key="",
            data=data,  # type: ignore
            messages=messages,
            print_verbose=print_verbose,
            encoding=encoding,
        )

    def image_generation(
        self,
        prompt: str,
        vertex_project: str,
        vertex_location: str,
        model: Optional[
            str
        ] = "imagegeneration",  # vertex ai uses imagegeneration as the default model
        client: Optional[AsyncHTTPHandler] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[int] = None,
        logging_obj=None,
        model_response=None,
        aimg_generation=False,
    ):
        if aimg_generation == True:
            response = self.aimage_generation(
                prompt=prompt,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                model=model,
                client=client,
                optional_params=optional_params,
                timeout=timeout,
                logging_obj=logging_obj,
                model_response=model_response,
            )
            return response

    async def aimage_generation(
        self,
        prompt: str,
        vertex_project: str,
        vertex_location: str,
        model_response: litellm.ImageResponse,
        model: Optional[
            str
        ] = "imagegeneration",  # vertex ai uses imagegeneration as the default model
        client: Optional[AsyncHTTPHandler] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[int] = None,
        logging_obj=None,
    ):
        response = None
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            self.async_handler = AsyncHTTPHandler(**_params)  # type: ignore
        else:
            self.async_handler = client  # type: ignore

        # make POST request to
        # https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/us-central1/publishers/google/models/imagegeneration:predict
        url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:predict"

        """
        Docs link: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/imagegeneration?project=adroit-crow-413218
        curl -X POST \
        -H "Authorization: Bearer $(gcloud auth print-access-token)" \
        -H "Content-Type: application/json; charset=utf-8" \
        -d {
            "instances": [
                {
                    "prompt": "a cat"
                }
            ],
            "parameters": {
                "sampleCount": 1
            }
        } \
        "https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/us-central1/publishers/google/models/imagegeneration:predict"
        """
        auth_header, _ = self._ensure_access_token(credentials=None, project_id=None)
        optional_params = optional_params or {
            "sampleCount": 1
        }  # default optional params

        request_data = {
            "instances": [{"prompt": prompt}],
            "parameters": optional_params,
        }

        request_str = f"\n curl -X POST \\\n -H \"Authorization: Bearer {auth_header[:10] + 'XXXXXXXXXX'}\" \\\n -H \"Content-Type: application/json; charset=utf-8\" \\\n -d {request_data} \\\n \"{url}\""
        logging_obj.pre_call(
            input=prompt,
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": request_str,
            },
        )

        response = await self.async_handler.post(
            url=url,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {auth_header}",
            },
            data=json.dumps(request_data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")
        """
        Vertex AI Image generation response example:
        {
            "predictions": [
                {
                "bytesBase64Encoded": "BASE64_IMG_BYTES",
                "mimeType": "image/png"
                },
                {
                "mimeType": "image/png",
                "bytesBase64Encoded": "BASE64_IMG_BYTES"
                }
            ]
        }
        """

        _json_response = response.json()
        _predictions = _json_response["predictions"]

        _response_data: List[litellm.ImageObject] = []
        for _prediction in _predictions:
            _bytes_base64_encoded = _prediction["bytesBase64Encoded"]
            image_object = litellm.ImageObject(b64_json=_bytes_base64_encoded)
            _response_data.append(image_object)

        model_response.data = _response_data

        return model_response


class ModelResponseIterator:
    def __init__(self, streaming_response, sync_stream: bool):
        self.streaming_response = streaming_response
        if sync_stream:
            self.response_iterator = iter(self.streaming_response)

        self.events = ijson.sendable_list()
        self.coro = ijson.items_coro(self.events, "item")

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            processed_chunk = GenerateContentResponseBody(**chunk)  # type: ignore
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None

            gemini_chunk = processed_chunk["candidates"][0]

            if (
                "content" in gemini_chunk
                and "text" in gemini_chunk["content"]["parts"][0]
            ):
                text = gemini_chunk["content"]["parts"][0]["text"]

            if "finishReason" in gemini_chunk:
                finish_reason = map_finish_reason(
                    finish_reason=gemini_chunk["finishReason"]
                )
                is_finished = True

            if "usageMetadata" in processed_chunk:
                usage = ChatCompletionUsageBlock(
                    prompt_tokens=processed_chunk["usageMetadata"]["promptTokenCount"],
                    completion_tokens=processed_chunk["usageMetadata"][
                        "candidatesTokenCount"
                    ],
                    total_tokens=processed_chunk["usageMetadata"]["totalTokenCount"],
                )

            returned_chunk = GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=0,
            )
            return returned_chunk
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = self.response_iterator.__next__()
            self.coro.send(chunk)
            if self.events:
                event = self.events[0]
                json_chunk = event
                self.events.clear()
                return self.chunk_parser(chunk=json_chunk)
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e}")

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            chunk = await self.async_response_iterator.__anext__()
            self.coro.send(chunk)
            if self.events:
                event = self.events[0]
                json_chunk = event
                self.events.clear()
                return self.chunk_parser(chunk=json_chunk)
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e}")
