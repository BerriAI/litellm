import json
import os
import time
import traceback
import types
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.cohere import ToolResultObject
from litellm.types.utils import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    GenericStreamingChunk,
)
from litellm.utils import Choices, Message, ModelResponse, Usage

from ..prompt_templates.factory import cohere_message_pt, cohere_messages_pt_v2


class CohereError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://api.cohere.ai/v1/chat")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class CohereChatConfig:
    """
    Configuration class for Cohere's API interface.

    Args:
        preamble (str, optional): When specified, the default Cohere preamble will be replaced with the provided one.
        chat_history (List[Dict[str, str]], optional): A list of previous messages between the user and the model.
        generation_id (str, optional): Unique identifier for the generated reply.
        response_id (str, optional): Unique identifier for the response.
        conversation_id (str, optional): An alternative to chat_history, creates or resumes a persisted conversation.
        prompt_truncation (str, optional): Dictates how the prompt will be constructed. Options: 'AUTO', 'AUTO_PRESERVE_ORDER', 'OFF'.
        connectors (List[Dict[str, str]], optional): List of connectors (e.g., web-search) to enrich the model's reply.
        search_queries_only (bool, optional): When true, the response will only contain a list of generated search queries.
        documents (List[Dict[str, str]], optional): A list of relevant documents that the model can cite.
        temperature (float, optional): A non-negative float that tunes the degree of randomness in generation.
        max_tokens (int, optional): The maximum number of tokens the model will generate as part of the response.
        k (int, optional): Ensures only the top k most likely tokens are considered for generation at each step.
        p (float, optional): Ensures that only the most likely tokens, with total probability mass of p, are considered for generation.
        frequency_penalty (float, optional): Used to reduce repetitiveness of generated tokens.
        presence_penalty (float, optional): Used to reduce repetitiveness of generated tokens.
        tools (List[Dict[str, str]], optional): A list of available tools (functions) that the model may suggest invoking.
        tool_results (List[Dict[str, Any]], optional): A list of results from invoking tools.
        seed (int, optional): A seed to assist reproducibility of the model's response.
    """

    preamble: Optional[str] = None
    chat_history: Optional[list] = None
    generation_id: Optional[str] = None
    response_id: Optional[str] = None
    conversation_id: Optional[str] = None
    prompt_truncation: Optional[str] = None
    connectors: Optional[list] = None
    search_queries_only: Optional[bool] = None
    documents: Optional[list] = None
    temperature: Optional[int] = None
    max_tokens: Optional[int] = None
    k: Optional[int] = None
    p: Optional[int] = None
    frequency_penalty: Optional[int] = None
    presence_penalty: Optional[int] = None
    tools: Optional[list] = None
    tool_results: Optional[list] = None
    seed: Optional[int] = None

    def __init__(
        self,
        preamble: Optional[str] = None,
        chat_history: Optional[list] = None,
        generation_id: Optional[str] = None,
        response_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        prompt_truncation: Optional[str] = None,
        connectors: Optional[list] = None,
        search_queries_only: Optional[bool] = None,
        documents: Optional[list] = None,
        temperature: Optional[int] = None,
        max_tokens: Optional[int] = None,
        k: Optional[int] = None,
        p: Optional[int] = None,
        frequency_penalty: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        tools: Optional[list] = None,
        tool_results: Optional[list] = None,
        seed: Optional[int] = None,
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


def validate_environment(api_key, headers: dict):
    headers.update(
        {
            "Request-Source": "unspecified:litellm",
            "accept": "application/json",
            "content-type": "application/json",
        }
    )
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def translate_openai_tool_to_cohere(openai_tool):
    # cohere tools look like this
    """
    {
       "name": "query_daily_sales_report",
       "description": "Connects to a database to retrieve overall sales volumes and sales information for a given day.",
       "parameter_definitions": {
           "day": {
               "description": "Retrieves sales data for this day, formatted as YYYY-MM-DD.",
               "type": "str",
               "required": True
           }
       }
    }
    """

    # OpenAI tools look like this
    """
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
    """
    cohere_tool = {
        "name": openai_tool["function"]["name"],
        "description": openai_tool["function"]["description"],
        "parameter_definitions": {},
    }

    for param_name, param_def in openai_tool["function"]["parameters"][
        "properties"
    ].items():
        required_params = (
            openai_tool.get("function", {}).get("parameters", {}).get("required", [])
        )
        cohere_param_def = {
            "description": param_def.get("description", ""),
            "type": param_def.get("type", ""),
            "required": param_name in required_params,
        }
        cohere_tool["parameter_definitions"][param_name] = cohere_param_def

    return cohere_tool


def construct_cohere_tool(tools=None):
    if tools is None:
        tools = []
    cohere_tools = []
    for tool in tools:
        cohere_tool = translate_openai_tool_to_cohere(tool)
        cohere_tools.append(cohere_tool)
    return cohere_tools


async def make_call(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    timeout: Optional[Union[float, httpx.Timeout]],
    json_mode: bool,
) -> Tuple[Any, httpx.Headers]:
    if client is None:
        client = litellm.module_level_aclient

    try:
        response = await client.post(
            api_base, headers=headers, data=data, stream=True, timeout=timeout
        )
    except httpx.HTTPStatusError as e:
        error_headers = getattr(e, "headers", None)
        error_response = getattr(e, "response", None)
        if error_headers is None and error_response:
            error_headers = getattr(error_response, "headers", None)
        raise CohereError(
            status_code=e.response.status_code,
            message=await e.response.aread(),
        )
    except Exception as e:
        for exception in litellm.LITELLM_EXCEPTION_TYPES:
            if isinstance(e, exception):
                raise e
        raise CohereError(status_code=500, message=str(e))

    completion_stream = ModelResponseIterator(
        streaming_response=response.aiter_lines(),
        sync_stream=False,
        json_mode=json_mode,
    )

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=completion_stream,  # Pass the completion stream for logging
        additional_args={"complete_input_dict": data},
    )

    return completion_stream, response.headers


def make_sync_call(
    client: Optional[HTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    timeout: Optional[Union[float, httpx.Timeout]],
) -> Tuple[Any, httpx.Headers]:
    if client is None:
        client = litellm.module_level_client  # re-use a module level client

    try:
        response = client.post(
            api_base, headers=headers, data=data, stream=True, timeout=timeout
        )
    except httpx.HTTPStatusError as e:
        raise CohereError(
            status_code=e.response.status_code,
            message=e.response.read(),
        )
    except Exception as e:
        for exception in litellm.LITELLM_EXCEPTION_TYPES:
            if isinstance(e, exception):
                raise e
        raise CohereError(status_code=500, message=str(e))

    if response.status_code != 200:

        raise CohereError(
            status_code=response.status_code,
            message=response.read(),
        )

    completion_stream = ModelResponseIterator(
        streaming_response=response.iter_lines(), sync_stream=True
    )

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream, response.headers


def completion(  # noqa: PLR0915
    model: str,
    messages: list,
    api_base: str,
    model_response: ModelResponse,
    print_verbose: Callable,
    optional_params: dict,
    headers: dict,
    encoding,
    api_key,
    logging_obj,
    litellm_params=None,
    logger_fn=None,
    client=None,
    timeout=None,
):
    headers = validate_environment(api_key, headers=headers)
    completion_url = api_base
    model = model
    most_recent_message, chat_history = cohere_messages_pt_v2(
        messages=messages, model=model, llm_provider="cohere_chat"
    )

    ## Load Config
    config = litellm.CohereConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > cohere_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    ## Handle Tool Calling
    if "tools" in optional_params:
        _is_function_call = True
        cohere_tools = construct_cohere_tool(tools=optional_params["tools"])
        optional_params["tools"] = cohere_tools
    if isinstance(most_recent_message, dict):
        optional_params["tool_results"] = [most_recent_message]
    elif isinstance(most_recent_message, str):
        optional_params["message"] = most_recent_message

    ## check if chat history message is 'user' and 'tool_results' is given -> force_single_step=True, else cohere api fails
    if len(chat_history) > 0 and chat_history[-1]["role"] == "USER":
        optional_params["force_single_step"] = True

    data = {
        "model": model,
        "chat_history": chat_history,
        **optional_params,
    }

    ## LOGGING
    logging_obj.pre_call(
        input=most_recent_message,
        api_key=api_key,
        additional_args={
            "complete_input_dict": data,
            "headers": headers,
            "api_base": completion_url,
        },
    )
    ## COMPLETION CALL
    response = requests.post(
        completion_url,
        headers=headers,
        data=json.dumps(data),
        stream=optional_params["stream"] if "stream" in optional_params else False,
    )
    ## error handling for cohere calls
    if response.status_code != 200:
        raise CohereError(message=response.text, status_code=response.status_code)

    if "stream" in optional_params and optional_params["stream"] is True:
        completion_stream, cohere_headers = make_sync_call(
            client=client,
            api_base=api_base,
            headers=headers,  # type: ignore
            data=json.dumps(data),
            model=model,
            messages=messages,
            logging_obj=logging_obj,
            timeout=timeout,
        )
        return CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="cohere_chat",
            logging_obj=logging_obj,
            _response_headers=dict(cohere_headers),
        )
    else:
        ## LOGGING
        logging_obj.post_call(
            input=most_recent_message,
            api_key=api_key,
            original_response=response.text,
            additional_args={"complete_input_dict": data},
        )
        print_verbose(f"raw model_response: {response.text}")
        ## RESPONSE OBJECT
        completion_response = response.json()
        try:
            model_response.choices[0].message.content = completion_response["text"]  # type: ignore
        except Exception:
            raise CohereError(message=response.text, status_code=response.status_code)

        ## ADD CITATIONS
        if "citations" in completion_response:
            setattr(model_response, "citations", completion_response["citations"])

        ## Tool calling response
        cohere_tools_response = completion_response.get("tool_calls", None)
        if cohere_tools_response is not None and cohere_tools_response != []:
            # convert cohere_tools_response to OpenAI response format
            tool_calls = []
            for tool in cohere_tools_response:
                function_name = tool.get("name", "")
                generation_id = tool.get("generation_id", "")
                parameters = tool.get("parameters", {})
                tool_call = {
                    "id": f"call_{generation_id}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(parameters),
                    },
                }
                tool_calls.append(tool_call)
            _message = litellm.Message(
                tool_calls=tool_calls,
                content=None,
            )
            model_response.choices[0].message = _message  # type: ignore

        ## CALCULATING USAGE - use cohere `billed_units` for returning usage
        billed_units = completion_response.get("meta", {}).get("billed_units", {})

        prompt_tokens = billed_units.get("input_tokens", 0)
        completion_tokens = billed_units.get("output_tokens", 0)

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response


class ModelResponseIterator:
    def __init__(
        self, streaming_response, sync_stream: bool, json_mode: Optional[bool] = False
    ):
        self.streaming_response = streaming_response
        self.response_iterator = self.streaming_response
        self.content_blocks: List = []
        self.tool_index = -1
        self.json_mode = json_mode

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None
            provider_specific_fields = None

            index = int(chunk.get("index", 0))

            if "text" in chunk:
                text = chunk["text"]
            elif "is_finished" in chunk and chunk["is_finished"] is True:
                is_finished = chunk["is_finished"]
                finish_reason = chunk["finish_reason"]

            if "citations" in chunk:
                provider_specific_fields = {"citations": chunk["citations"]}

            returned_chunk = GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=index,
                provider_specific_fields=provider_specific_fields,
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
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            str_line = chunk
            if isinstance(chunk, bytes):  # Handle binary data
                str_line = chunk.decode("utf-8")  # Convert bytes to string
                index = str_line.find("data:")
                if index != -1:
                    str_line = str_line[index:]
            data_json = json.loads(str_line)
            return self.chunk_parser(chunk=data_json)
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            chunk = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            str_line = chunk
            if isinstance(chunk, bytes):  # Handle binary data
                str_line = chunk.decode("utf-8")  # Convert bytes to string
                index = str_line.find("data:")
                if index != -1:
                    str_line = str_line[index:]

            data_json = json.loads(str_line)
            return self.chunk_parser(chunk=data_json)
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")
