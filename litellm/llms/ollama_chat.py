import json
import time
import uuid
from typing import Any, List, Optional, Union

import aiohttp
import httpx
from pydantic import BaseModel

import litellm
from litellm import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.llms.ollama import OllamaToolCall, OllamaToolCallFunction
from litellm.types.llms.openai import ChatCompletionAssistantToolCall
from litellm.types.utils import ModelResponse, StreamingChoices


class OllamaError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="http://localhost:11434")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


# ollama implementation
def get_ollama_response(  # noqa: PLR0915
    model_response: ModelResponse,
    messages: list,
    optional_params: dict,
    model: str,
    logging_obj: Any,
    api_base="http://localhost:11434",
    api_key: Optional[str] = None,
    acompletion: bool = False,
    encoding=None,
    client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
):
    if api_base.endswith("/api/chat"):
        url = api_base
    else:
        url = f"{api_base}/api/chat"

    ## Load Config
    config = litellm.OllamaChatConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > cohere_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    stream = optional_params.pop("stream", False)
    format = optional_params.pop("format", None)
    keep_alive = optional_params.pop("keep_alive", None)
    function_name = optional_params.pop("function_name", None)
    tools = optional_params.pop("tools", None)

    new_messages = []
    for m in messages:
        if isinstance(
            m, BaseModel
        ):  # avoid message serialization issues - https://github.com/BerriAI/litellm/issues/5319
            m = m.model_dump(exclude_none=True)
        if m.get("tool_calls") is not None and isinstance(m["tool_calls"], list):
            new_tools: List[OllamaToolCall] = []
            for tool in m["tool_calls"]:
                typed_tool = ChatCompletionAssistantToolCall(**tool)  # type: ignore
                if typed_tool["type"] == "function":
                    arguments = {}
                    if "arguments" in typed_tool["function"]:
                        arguments = json.loads(typed_tool["function"]["arguments"])
                    ollama_tool_call = OllamaToolCall(
                        function=OllamaToolCallFunction(
                            name=typed_tool["function"].get("name") or "",
                            arguments=arguments,
                        )
                    )
                    new_tools.append(ollama_tool_call)
            m["tool_calls"] = new_tools
        new_messages.append(m)

    data = {
        "model": model,
        "messages": new_messages,
        "options": optional_params,
        "stream": stream,
    }
    if format is not None:
        data["format"] = format
    if tools is not None:
        data["tools"] = tools
    if keep_alive is not None:
        data["keep_alive"] = keep_alive
    ## LOGGING
    logging_obj.pre_call(
        input=None,
        api_key=None,
        additional_args={
            "api_base": url,
            "complete_input_dict": data,
            "headers": {},
            "acompletion": acompletion,
        },
    )
    if acompletion is True:
        if stream is True:
            response = ollama_async_streaming(
                url=url,
                api_key=api_key,
                data=data,
                model_response=model_response,
                encoding=encoding,
                logging_obj=logging_obj,
            )
        else:
            response = ollama_acompletion(
                url=url,
                api_key=api_key,
                data=data,
                model_response=model_response,
                encoding=encoding,
                logging_obj=logging_obj,
                function_name=function_name,
            )
        return response
    elif stream is True:
        return ollama_completion_stream(
            url=url, api_key=api_key, data=data, logging_obj=logging_obj
        )

    headers: Optional[dict] = None
    if api_key is not None:
        headers = {"Authorization": "Bearer {}".format(api_key)}

    sync_client = litellm.module_level_client
    if client is not None and isinstance(client, HTTPHandler):
        sync_client = client
    response = sync_client.post(
        url=url,
        json=data,
        headers=headers,
    )
    if response.status_code != 200:
        raise OllamaError(status_code=response.status_code, message=response.text)

    ## LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=response.text,
        additional_args={
            "headers": None,
            "api_base": api_base,
        },
    )

    response_json = response.json()

    ## RESPONSE OBJECT
    model_response.choices[0].finish_reason = "stop"
    if data.get("format", "") == "json" and function_name is not None:
        function_call = json.loads(response_json["message"]["content"])
        message = litellm.Message(
            content=None,
            tool_calls=[
                {
                    "id": f"call_{str(uuid.uuid4())}",
                    "function": {
                        "name": function_call.get("name", function_name),
                        "arguments": json.dumps(
                            function_call.get("arguments", function_call)
                        ),
                    },
                    "type": "function",
                }
            ],
        )
        model_response.choices[0].message = message  # type: ignore
        model_response.choices[0].finish_reason = "tool_calls"
    else:
        _message = litellm.Message(**response_json["message"])
        model_response.choices[0].message = _message  # type: ignore
    model_response.created = int(time.time())
    model_response.model = "ollama_chat/" + model
    prompt_tokens = response_json.get("prompt_eval_count", litellm.token_counter(messages=messages))  # type: ignore
    completion_tokens = response_json.get(
        "eval_count", litellm.token_counter(text=response_json["message"]["content"])
    )
    setattr(
        model_response,
        "usage",
        litellm.Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return model_response


def ollama_completion_stream(url, api_key, data, logging_obj):
    _request = {
        "url": f"{url}",
        "json": data,
        "method": "POST",
        "timeout": litellm.request_timeout,
        "follow_redirects": True,
    }
    if api_key is not None:
        _request["headers"] = {"Authorization": "Bearer {}".format(api_key)}
    with httpx.stream(**_request) as response:
        try:
            if response.status_code != 200:
                raise OllamaError(
                    status_code=response.status_code, message=response.iter_lines()
                )

            streamwrapper = litellm.CustomStreamWrapper(
                completion_stream=response.iter_lines(),
                model=data["model"],
                custom_llm_provider="ollama_chat",
                logging_obj=logging_obj,
            )

            # If format is JSON, this was a function call
            # Gather all chunks and return the function call as one delta to simplify parsing
            if data.get("format", "") == "json":
                content_chunks = []
                for chunk in streamwrapper:
                    chunk_choice = chunk.choices[0]
                    if (
                        isinstance(chunk_choice, StreamingChoices)
                        and hasattr(chunk_choice, "delta")
                        and hasattr(chunk_choice.delta, "content")
                    ):
                        content_chunks.append(chunk_choice.delta.content)
                response_content = "".join(content_chunks)

                function_call = json.loads(response_content)
                delta = litellm.utils.Delta(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{str(uuid.uuid4())}",
                            "function": {
                                "name": function_call["name"],
                                "arguments": json.dumps(function_call["arguments"]),
                            },
                            "type": "function",
                        }
                    ],
                )
                model_response = content_chunks[0]
                model_response.choices[0].delta = delta  # type: ignore
                model_response.choices[0].finish_reason = "tool_calls"
                yield model_response
            else:
                for transformed_chunk in streamwrapper:
                    yield transformed_chunk
        except Exception as e:
            raise e


async def ollama_async_streaming(
    url, api_key, data, model_response, encoding, logging_obj
):
    try:
        _async_http_client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.OLLAMA
        )
        client = _async_http_client.client
        _request = {
            "url": f"{url}",
            "json": data,
            "method": "POST",
            "timeout": litellm.request_timeout,
        }
        if api_key is not None:
            _request["headers"] = {"Authorization": "Bearer {}".format(api_key)}
        async with client.stream(**_request) as response:
            if response.status_code != 200:
                raise OllamaError(
                    status_code=response.status_code, message=response.text
                )

            streamwrapper = litellm.CustomStreamWrapper(
                completion_stream=response.aiter_lines(),
                model=data["model"],
                custom_llm_provider="ollama_chat",
                logging_obj=logging_obj,
            )

            # If format is JSON, this was a function call
            # Gather all chunks and return the function call as one delta to simplify parsing
            if data.get("format", "") == "json":
                first_chunk = await anext(streamwrapper)  # noqa F821
                chunk_choice = first_chunk.choices[0]
                if (
                    isinstance(chunk_choice, StreamingChoices)
                    and hasattr(chunk_choice, "delta")
                    and hasattr(chunk_choice.delta, "content")
                ):
                    first_chunk_content = chunk_choice.delta.content or ""
                else:
                    first_chunk_content = ""

                content_chunks = []
                async for chunk in streamwrapper:
                    chunk_choice = chunk.choices[0]
                    if (
                        isinstance(chunk_choice, StreamingChoices)
                        and hasattr(chunk_choice, "delta")
                        and hasattr(chunk_choice.delta, "content")
                    ):
                        content_chunks.append(chunk_choice.delta.content)
                response_content = first_chunk_content + "".join(content_chunks)

                function_call = json.loads(response_content)
                delta = litellm.utils.Delta(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{str(uuid.uuid4())}",
                            "function": {
                                "name": function_call.get(
                                    "name", function_call.get("function", None)
                                ),
                                "arguments": json.dumps(function_call["arguments"]),
                            },
                            "type": "function",
                        }
                    ],
                )
                model_response = first_chunk
                model_response.choices[0].delta = delta  # type: ignore
                model_response.choices[0].finish_reason = "tool_calls"
                yield model_response
            else:
                async for transformed_chunk in streamwrapper:
                    yield transformed_chunk
    except Exception as e:
        verbose_logger.exception(
            "LiteLLM.ollama(): Exception occured - {}".format(str(e))
        )
        raise e


async def ollama_acompletion(
    url,
    api_key: Optional[str],
    data,
    model_response: litellm.ModelResponse,
    encoding,
    logging_obj,
    function_name,
):
    data["stream"] = False
    try:
        timeout = aiohttp.ClientTimeout(total=litellm.request_timeout)  # 10 minutes
        async with aiohttp.ClientSession(timeout=timeout) as session:
            _request = {
                "url": f"{url}",
                "json": data,
            }
            if api_key is not None:
                _request["headers"] = {"Authorization": "Bearer {}".format(api_key)}
            resp = await session.post(**_request)

            if resp.status != 200:
                text = await resp.text()
                raise OllamaError(status_code=resp.status, message=text)

            response_json = await resp.json()

            ## LOGGING
            logging_obj.post_call(
                input=data,
                api_key="",
                original_response=response_json,
                additional_args={
                    "headers": None,
                    "api_base": url,
                },
            )

            ## RESPONSE OBJECT
            model_response.choices[0].finish_reason = "stop"

            if data.get("format", "") == "json" and function_name is not None:
                function_call = json.loads(response_json["message"]["content"])
                message = litellm.Message(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{str(uuid.uuid4())}",
                            "function": {
                                "name": function_call.get("name", function_name),
                                "arguments": json.dumps(
                                    function_call.get("arguments", function_call)
                                ),
                            },
                            "type": "function",
                        }
                    ],
                )
                model_response.choices[0].message = message  # type: ignore
                model_response.choices[0].finish_reason = "tool_calls"
            else:
                _message = litellm.Message(**response_json["message"])
                model_response.choices[0].message = _message  # type: ignore

            model_response.created = int(time.time())
            model_response.model = "ollama_chat/" + data["model"]
            prompt_tokens = response_json.get("prompt_eval_count", litellm.token_counter(messages=data["messages"]))  # type: ignore
            completion_tokens = response_json.get(
                "eval_count",
                litellm.token_counter(
                    text=response_json["message"]["content"], count_response_tokens=True
                ),
            )
            setattr(
                model_response,
                "usage",
                litellm.Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            return model_response
    except Exception as e:
        raise e  # don't use verbose_logger.exception, if exception is raised
