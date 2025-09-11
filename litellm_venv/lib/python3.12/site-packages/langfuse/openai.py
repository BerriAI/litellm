"""If you use the OpenAI Python SDK, you can use the Langfuse drop-in replacement to get full logging by changing only the import.

```diff
- import openai
+ from langfuse.openai import openai
```

Langfuse automatically tracks:

- All prompts/completions with support for streaming, async and functions
- Latencies
- API Errors
- Model usage (tokens) and cost (USD)

The integration is fully interoperable with the `observe()` decorator and the low-level tracing SDK.

See docs for more details: https://langfuse.com/docs/integrations/openai
"""

import logging
import types
from collections import defaultdict
from dataclasses import dataclass
from inspect import isclass
from typing import Optional

import openai.resources
from openai._types import NotGiven
from packaging.version import Version
from pydantic import BaseModel
from wrapt import wrap_function_wrapper

from langfuse import Langfuse
from langfuse.client import StatefulGenerationClient
from langfuse.decorators import langfuse_context
from langfuse.media import LangfuseMedia
from langfuse.utils import _get_timestamp
from langfuse.utils.langfuse_singleton import LangfuseSingleton

try:
    import openai
except ImportError:
    raise ModuleNotFoundError(
        "Please install OpenAI to use this feature: 'pip install openai'"
    )

try:
    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI  # noqa: F401
except ImportError:
    AsyncAzureOpenAI = None
    AsyncOpenAI = None
    AzureOpenAI = None
    OpenAI = None

log = logging.getLogger("langfuse")


@dataclass
class OpenAiDefinition:
    module: str
    object: str
    method: str
    type: str
    sync: bool
    min_version: Optional[str] = None


OPENAI_METHODS_V0 = [
    OpenAiDefinition(
        module="openai",
        object="ChatCompletion",
        method="create",
        type="chat",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai",
        object="Completion",
        method="create",
        type="completion",
        sync=True,
    ),
]


OPENAI_METHODS_V1 = [
    OpenAiDefinition(
        module="openai.resources.chat.completions",
        object="Completions",
        method="create",
        type="chat",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai.resources.completions",
        object="Completions",
        method="create",
        type="completion",
        sync=True,
    ),
    OpenAiDefinition(
        module="openai.resources.chat.completions",
        object="AsyncCompletions",
        method="create",
        type="chat",
        sync=False,
    ),
    OpenAiDefinition(
        module="openai.resources.completions",
        object="AsyncCompletions",
        method="create",
        type="completion",
        sync=False,
    ),
    OpenAiDefinition(
        module="openai.resources.beta.chat.completions",
        object="Completions",
        method="parse",
        type="chat",
        sync=True,
        min_version="1.50.0",
    ),
    OpenAiDefinition(
        module="openai.resources.beta.chat.completions",
        object="AsyncCompletions",
        method="parse",
        type="chat",
        sync=False,
        min_version="1.50.0",
    ),
]


class OpenAiArgsExtractor:
    def __init__(
        self,
        name=None,
        metadata=None,
        trace_id=None,
        session_id=None,
        user_id=None,
        tags=None,
        parent_observation_id=None,
        langfuse_prompt=None,  # we cannot use prompt because it's an argument of the old OpenAI completions API
        **kwargs,
    ):
        self.args = {}
        self.args["name"] = name
        self.args["metadata"] = (
            metadata
            if "response_format" not in kwargs
            else {
                **(metadata or {}),
                "response_format": kwargs["response_format"].model_json_schema()
                if isclass(kwargs["response_format"])
                and issubclass(kwargs["response_format"], BaseModel)
                else kwargs["response_format"],
            }
        )
        self.args["trace_id"] = trace_id
        self.args["session_id"] = session_id
        self.args["user_id"] = user_id
        self.args["tags"] = tags
        self.args["parent_observation_id"] = parent_observation_id
        self.args["langfuse_prompt"] = langfuse_prompt
        self.kwargs = kwargs

    def get_langfuse_args(self):
        return {**self.args, **self.kwargs}

    def get_openai_args(self):
        # If OpenAI model distillation is enabled, we need to add the metadata to the kwargs
        # https://platform.openai.com/docs/guides/distillation
        if self.kwargs.get("store", False):
            self.kwargs["metadata"] = (
                {} if self.args.get("metadata", None) is None else self.args["metadata"]
            )

            # OpenAI does not support non-string type values in metadata when using
            # model distillation feature
            self.kwargs["metadata"].pop("response_format", None)

        return self.kwargs


def _langfuse_wrapper(func):
    def _with_langfuse(open_ai_definitions, initialize):
        def wrapper(wrapped, instance, args, kwargs):
            return func(open_ai_definitions, initialize, wrapped, args, kwargs)

        return wrapper

    return _with_langfuse


def _extract_chat_prompt(kwargs: any):
    """Extracts the user input from prompts. Returns an array of messages or dict with messages and functions"""
    prompt = {}

    if kwargs.get("functions") is not None:
        prompt.update({"functions": kwargs["functions"]})

    if kwargs.get("function_call") is not None:
        prompt.update({"function_call": kwargs["function_call"]})

    if kwargs.get("tools") is not None:
        prompt.update({"tools": kwargs["tools"]})

    if prompt:
        # uf user provided functions, we need to send these together with messages to langfuse
        prompt.update(
            {
                "messages": [
                    _process_message(message) for message in kwargs.get("messages", [])
                ],
            }
        )
        return prompt
    else:
        # vanilla case, only send messages in openai format to langfuse
        return [_process_message(message) for message in kwargs.get("messages", [])]


def _process_message(message):
    if not isinstance(message, dict):
        return message

    processed_message = {**message}

    content = processed_message.get("content", None)
    if not isinstance(content, list):
        return processed_message

    processed_content = []

    for content_part in content:
        if content_part.get("type") == "input_audio":
            audio_base64 = content_part.get("input_audio", {}).get("data", None)
            format = content_part.get("input_audio", {}).get("format", "wav")

            if audio_base64 is not None:
                base64_data_uri = f"data:audio/{format};base64,{audio_base64}"

                processed_content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": LangfuseMedia(base64_data_uri=base64_data_uri),
                            "format": format,
                        },
                    }
                )
        else:
            processed_content.append(content_part)

    processed_message["content"] = processed_content

    return processed_message


def _extract_chat_response(kwargs: any):
    """Extracts the llm output from the response."""
    response = {
        "role": kwargs.get("role", None),
    }

    audio = None

    if kwargs.get("function_call") is not None:
        response.update({"function_call": kwargs["function_call"]})

    if kwargs.get("tool_calls") is not None:
        response.update({"tool_calls": kwargs["tool_calls"]})

    if kwargs.get("audio") is not None:
        audio = kwargs["audio"].__dict__

        if "data" in audio and audio["data"] is not None:
            base64_data_uri = f"data:audio/{audio.get('format', 'wav')};base64,{audio.get('data', None)}"
            audio["data"] = LangfuseMedia(base64_data_uri=base64_data_uri)

    response.update(
        {
            "content": kwargs.get("content", None),
        }
    )

    if audio is not None:
        response.update({"audio": audio})

    return response


def _get_langfuse_data_from_kwargs(
    resource: OpenAiDefinition, langfuse: Langfuse, start_time, kwargs
):
    name = kwargs.get("name", "OpenAI-generation")

    if name is None:
        name = "OpenAI-generation"

    if name is not None and not isinstance(name, str):
        raise TypeError("name must be a string")

    decorator_context_observation_id = langfuse_context.get_current_observation_id()
    decorator_context_trace_id = langfuse_context.get_current_trace_id()

    trace_id = kwargs.get("trace_id", None) or decorator_context_trace_id
    if trace_id is not None and not isinstance(trace_id, str):
        raise TypeError("trace_id must be a string")

    session_id = kwargs.get("session_id", None)
    if session_id is not None and not isinstance(session_id, str):
        raise TypeError("session_id must be a string")

    user_id = kwargs.get("user_id", None)
    if user_id is not None and not isinstance(user_id, str):
        raise TypeError("user_id must be a string")

    tags = kwargs.get("tags", None)
    if tags is not None and (
        not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags)
    ):
        raise TypeError("tags must be a list of strings")

    # Update trace params in decorator context if specified in openai call
    if decorator_context_trace_id:
        langfuse_context.update_current_trace(
            session_id=session_id, user_id=user_id, tags=tags
        )

    parent_observation_id = kwargs.get("parent_observation_id", None) or (
        decorator_context_observation_id
        if decorator_context_observation_id != decorator_context_trace_id
        else None
    )
    if parent_observation_id is not None and not isinstance(parent_observation_id, str):
        raise TypeError("parent_observation_id must be a string")
    if parent_observation_id is not None and trace_id is None:
        raise ValueError("parent_observation_id requires trace_id to be set")

    metadata = kwargs.get("metadata", {})

    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    model = kwargs.get("model", None) or None

    prompt = None

    if resource.type == "completion":
        prompt = kwargs.get("prompt", None)
    elif resource.type == "chat":
        prompt = _extract_chat_prompt(kwargs)

    is_nested_trace = False
    if trace_id:
        is_nested_trace = True
        langfuse.trace(id=trace_id, session_id=session_id, user_id=user_id, tags=tags)
    else:
        trace_id = (
            decorator_context_trace_id
            or langfuse.trace(
                session_id=session_id,
                user_id=user_id,
                tags=tags,
                name=name,
                input=prompt,
                metadata=metadata,
            ).id
        )

    parsed_temperature = (
        kwargs.get("temperature", 1)
        if not isinstance(kwargs.get("temperature", 1), NotGiven)
        else 1
    )

    parsed_max_tokens = (
        kwargs.get("max_tokens", float("inf"))
        if not isinstance(kwargs.get("max_tokens", float("inf")), NotGiven)
        else float("inf")
    )

    parsed_top_p = (
        kwargs.get("top_p", 1)
        if not isinstance(kwargs.get("top_p", 1), NotGiven)
        else 1
    )

    parsed_frequency_penalty = (
        kwargs.get("frequency_penalty", 0)
        if not isinstance(kwargs.get("frequency_penalty", 0), NotGiven)
        else 0
    )

    parsed_presence_penalty = (
        kwargs.get("presence_penalty", 0)
        if not isinstance(kwargs.get("presence_penalty", 0), NotGiven)
        else 0
    )

    parsed_seed = (
        kwargs.get("seed", None)
        if not isinstance(kwargs.get("seed", None), NotGiven)
        else None
    )

    parsed_n = kwargs.get("n", 1) if not isinstance(kwargs.get("n", 1), NotGiven) else 1

    modelParameters = {
        "temperature": parsed_temperature,
        "max_tokens": parsed_max_tokens,  # casing?
        "top_p": parsed_top_p,
        "frequency_penalty": parsed_frequency_penalty,
        "presence_penalty": parsed_presence_penalty,
    }
    if parsed_n is not None and parsed_n > 1:
        modelParameters["n"] = parsed_n

    if parsed_seed is not None:
        modelParameters["seed"] = parsed_seed

    langfuse_prompt = kwargs.get("langfuse_prompt", None)

    return {
        "name": name,
        "metadata": metadata,
        "trace_id": trace_id,
        "parent_observation_id": parent_observation_id,
        "user_id": user_id,
        "start_time": start_time,
        "input": prompt,
        "model_parameters": modelParameters,
        "model": model or None,
        "prompt": langfuse_prompt,
    }, is_nested_trace


def _create_langfuse_update(
    completion,
    generation: StatefulGenerationClient,
    completion_start_time,
    model=None,
    usage=None,
):
    update = {
        "end_time": _get_timestamp(),
        "output": completion,
        "completion_start_time": completion_start_time,
    }
    if model is not None:
        update["model"] = model

    if usage is not None:
        update["usage"] = _parse_usage(usage)

    generation.update(**update)


def _parse_usage(usage=None):
    if usage is None:
        return

    usage_dict = usage.copy() if isinstance(usage, dict) else usage.__dict__.copy()

    for tokens_details in ["prompt_tokens_details", "completion_tokens_details"]:
        if tokens_details in usage_dict and usage_dict[tokens_details] is not None:
            tokens_details_dict = (
                usage_dict[tokens_details]
                if isinstance(usage_dict[tokens_details], dict)
                else usage_dict[tokens_details].__dict__
            )
            usage_dict[tokens_details] = {
                k: v for k, v in tokens_details_dict.items() if v is not None
            }

    return usage_dict


def _extract_streamed_openai_response(resource, chunks):
    completion = defaultdict(str) if resource.type == "chat" else ""
    model, usage = None, None

    for chunk in chunks:
        if _is_openai_v1():
            chunk = chunk.__dict__

        model = model or chunk.get("model", None) or None
        usage = chunk.get("usage", None)

        choices = chunk.get("choices", [])

        for choice in choices:
            if _is_openai_v1():
                choice = choice.__dict__
            if resource.type == "chat":
                delta = choice.get("delta", None)

                if _is_openai_v1():
                    delta = delta.__dict__

                if delta.get("role", None) is not None:
                    completion["role"] = delta["role"]

                if delta.get("content", None) is not None:
                    completion["content"] = (
                        delta.get("content", None)
                        if completion["content"] is None
                        else completion["content"] + delta.get("content", None)
                    )
                elif delta.get("function_call", None) is not None:
                    curr = completion["function_call"]
                    tool_call_chunk = delta.get("function_call", None)

                    if not curr:
                        completion["function_call"] = {
                            "name": getattr(tool_call_chunk, "name", ""),
                            "arguments": getattr(tool_call_chunk, "arguments", ""),
                        }

                    else:
                        curr["name"] = curr["name"] or getattr(
                            tool_call_chunk, "name", None
                        )
                        curr["arguments"] += getattr(tool_call_chunk, "arguments", "")

                elif delta.get("tool_calls", None) is not None:
                    curr = completion["tool_calls"]
                    tool_call_chunk = getattr(
                        delta.get("tool_calls", None)[0], "function", None
                    )

                    if not curr:
                        completion["tool_calls"] = [
                            {
                                "name": getattr(tool_call_chunk, "name", ""),
                                "arguments": getattr(tool_call_chunk, "arguments", ""),
                            }
                        ]

                    elif getattr(tool_call_chunk, "name", None) is not None:
                        curr.append(
                            {
                                "name": getattr(tool_call_chunk, "name", None),
                                "arguments": getattr(
                                    tool_call_chunk, "arguments", None
                                ),
                            }
                        )

                    else:
                        curr[-1]["name"] = curr[-1]["name"] or getattr(
                            tool_call_chunk, "name", None
                        )
                        curr[-1]["arguments"] += getattr(
                            tool_call_chunk, "arguments", None
                        )

            if resource.type == "completion":
                completion += choice.get("text", "")

    def get_response_for_chat():
        return (
            completion["content"]
            or (
                completion["function_call"]
                and {
                    "role": "assistant",
                    "function_call": completion["function_call"],
                }
            )
            or (
                completion["tool_calls"]
                and {
                    "role": "assistant",
                    # "tool_calls": [{"function": completion["tool_calls"]}],
                    "tool_calls": [
                        {"function": data} for data in completion["tool_calls"]
                    ],
                }
            )
            or None
        )

    return (
        model,
        get_response_for_chat() if resource.type == "chat" else completion,
        usage,
    )


def _get_langfuse_data_from_default_response(resource: OpenAiDefinition, response):
    if response is None:
        return None, "<NoneType response returned from OpenAI>", None

    model = response.get("model", None) or None

    completion = None
    if resource.type == "completion":
        choices = response.get("choices", [])
        if len(choices) > 0:
            choice = choices[-1]

            completion = choice.text if _is_openai_v1() else choice.get("text", None)
    elif resource.type == "chat":
        choices = response.get("choices", [])
        if len(choices) > 0:
            # If multiple choices were generated, we'll show all of them in the UI as a list.
            if len(choices) > 1:
                completion = [
                    _extract_chat_response(choice.message.__dict__)
                    if _is_openai_v1()
                    else choice.get("message", None)
                    for choice in choices
                ]
            else:
                choice = choices[0]
                completion = (
                    _extract_chat_response(choice.message.__dict__)
                    if _is_openai_v1()
                    else choice.get("message", None)
                )

    usage = _parse_usage(response.get("usage", None))

    return (model, completion, usage)


def _is_openai_v1():
    return Version(openai.__version__) >= Version("1.0.0")


def _is_streaming_response(response):
    return (
        isinstance(response, types.GeneratorType)
        or isinstance(response, types.AsyncGeneratorType)
        or (_is_openai_v1() and isinstance(response, openai.Stream))
        or (_is_openai_v1() and isinstance(response, openai.AsyncStream))
    )


@_langfuse_wrapper
def _wrap(open_ai_resource: OpenAiDefinition, initialize, wrapped, args, kwargs):
    new_langfuse: Langfuse = initialize()

    start_time = _get_timestamp()
    arg_extractor = OpenAiArgsExtractor(*args, **kwargs)

    generation, is_nested_trace = _get_langfuse_data_from_kwargs(
        open_ai_resource, new_langfuse, start_time, arg_extractor.get_langfuse_args()
    )
    generation = new_langfuse.generation(**generation)
    try:
        openai_response = wrapped(**arg_extractor.get_openai_args())

        if _is_streaming_response(openai_response):
            return LangfuseResponseGeneratorSync(
                resource=open_ai_resource,
                response=openai_response,
                generation=generation,
                langfuse=new_langfuse,
                is_nested_trace=is_nested_trace,
            )

        else:
            model, completion, usage = _get_langfuse_data_from_default_response(
                open_ai_resource,
                (openai_response and openai_response.__dict__)
                if _is_openai_v1()
                else openai_response,
            )
            generation.update(
                model=model,
                output=completion,
                end_time=_get_timestamp(),
                usage=usage,  # backward compat for all V2 self hosters
                usage_details=usage,
            )

            # Avoiding the trace-update if trace-id is provided by user.
            if not is_nested_trace:
                new_langfuse.trace(id=generation.trace_id, output=completion)

        return openai_response
    except Exception as ex:
        log.warning(ex)
        model = kwargs.get("model", None) or None
        generation.update(
            end_time=_get_timestamp(),
            status_message=str(ex),
            level="ERROR",
            model=model,
            usage={
                "input_cost": 0,
                "output_cost": 0,
                "total_cost": 0,
            },  # backward compat for all V2 self hosters
            cost_details={"input": 0, "output": 0, "total": 0},
        )
        raise ex


@_langfuse_wrapper
async def _wrap_async(
    open_ai_resource: OpenAiDefinition, initialize, wrapped, args, kwargs
):
    new_langfuse = initialize()
    start_time = _get_timestamp()
    arg_extractor = OpenAiArgsExtractor(*args, **kwargs)

    generation, is_nested_trace = _get_langfuse_data_from_kwargs(
        open_ai_resource, new_langfuse, start_time, arg_extractor.get_langfuse_args()
    )
    generation = new_langfuse.generation(**generation)
    try:
        openai_response = await wrapped(**arg_extractor.get_openai_args())

        if _is_streaming_response(openai_response):
            return LangfuseResponseGeneratorAsync(
                resource=open_ai_resource,
                response=openai_response,
                generation=generation,
                langfuse=new_langfuse,
                is_nested_trace=is_nested_trace,
            )

        else:
            model, completion, usage = _get_langfuse_data_from_default_response(
                open_ai_resource,
                (openai_response and openai_response.__dict__)
                if _is_openai_v1()
                else openai_response,
            )
            generation.update(
                model=model,
                output=completion,
                end_time=_get_timestamp(),
                usage=usage,  # backward compat for all V2 self hosters
                usage_details=usage,
            )
            # Avoiding the trace-update if trace-id is provided by user.
            if not is_nested_trace:
                new_langfuse.trace(id=generation.trace_id, output=completion)

        return openai_response
    except Exception as ex:
        model = kwargs.get("model", None) or None
        generation.update(
            end_time=_get_timestamp(),
            status_message=str(ex),
            level="ERROR",
            model=model,
            usage={
                "input_cost": 0,
                "output_cost": 0,
                "total_cost": 0,
            },  # Backward compat for all V2 self hosters
            cost_details={"input": 0, "output": 0, "total": 0},
        )
        raise ex


class OpenAILangfuse:
    _langfuse: Optional[Langfuse] = None

    def initialize(self):
        self._langfuse = LangfuseSingleton().get(
            public_key=openai.langfuse_public_key,
            secret_key=openai.langfuse_secret_key,
            host=openai.langfuse_host,
            debug=openai.langfuse_debug,
            enabled=openai.langfuse_enabled,
            sdk_integration="openai",
            sample_rate=openai.langfuse_sample_rate,
            environment=openai.langfuse_environment,
            mask=openai.langfuse_mask,
        )

        return self._langfuse

    def flush(cls):
        cls._langfuse.flush()

    def langfuse_auth_check(self):
        """Check if the provided Langfuse credentials (public and secret key) are valid.

        Raises:
            Exception: If no projects were found for the provided credentials.

        Note:
            This method is blocking. It is discouraged to use it in production code.
        """
        if self._langfuse is None:
            self.initialize()

        return self._langfuse.auth_check()

    def register_tracing(self):
        resources = OPENAI_METHODS_V1 if _is_openai_v1() else OPENAI_METHODS_V0

        for resource in resources:
            if resource.min_version is not None and Version(
                openai.__version__
            ) < Version(resource.min_version):
                continue

            wrap_function_wrapper(
                resource.module,
                f"{resource.object}.{resource.method}",
                _wrap(resource, self.initialize)
                if resource.sync
                else _wrap_async(resource, self.initialize),
            )

        setattr(openai, "langfuse_public_key", None)
        setattr(openai, "langfuse_secret_key", None)
        setattr(openai, "langfuse_host", None)
        setattr(openai, "langfuse_debug", None)
        setattr(openai, "langfuse_enabled", True)
        setattr(openai, "langfuse_sample_rate", None)
        setattr(openai, "langfuse_environment", None)
        setattr(openai, "langfuse_mask", None)
        setattr(openai, "langfuse_auth_check", self.langfuse_auth_check)
        setattr(openai, "flush_langfuse", self.flush)


modifier = OpenAILangfuse()
modifier.register_tracing()


# DEPRECATED: Use `openai.langfuse_auth_check()` instead
def auth_check():
    if modifier._langfuse is None:
        modifier.initialize()

    return modifier._langfuse.auth_check()


class LangfuseResponseGeneratorSync:
    def __init__(
        self,
        *,
        resource,
        response,
        generation,
        langfuse,
        is_nested_trace,
    ):
        self.items = []

        self.resource = resource
        self.response = response
        self.generation = generation
        self.langfuse = langfuse
        self.is_nested_trace = is_nested_trace
        self.completion_start_time = None

    def __iter__(self):
        try:
            for i in self.response:
                self.items.append(i)

                if self.completion_start_time is None:
                    self.completion_start_time = _get_timestamp()

                yield i
        finally:
            self._finalize()

    def __next__(self):
        try:
            item = self.response.__next__()
            self.items.append(item)

            if self.completion_start_time is None:
                self.completion_start_time = _get_timestamp()

            return item

        except StopIteration:
            self._finalize()

            raise

    def __enter__(self):
        return self.__iter__()

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _finalize(self):
        model, completion, usage = _extract_streamed_openai_response(
            self.resource, self.items
        )

        # Avoiding the trace-update if trace-id is provided by user.
        if not self.is_nested_trace:
            self.langfuse.trace(id=self.generation.trace_id, output=completion)

        _create_langfuse_update(
            completion,
            self.generation,
            self.completion_start_time,
            model=model,
            usage=usage,
        )


class LangfuseResponseGeneratorAsync:
    def __init__(
        self,
        *,
        resource,
        response,
        generation,
        langfuse,
        is_nested_trace,
    ):
        self.items = []

        self.resource = resource
        self.response = response
        self.generation = generation
        self.langfuse = langfuse
        self.is_nested_trace = is_nested_trace
        self.completion_start_time = None

    async def __aiter__(self):
        try:
            async for i in self.response:
                self.items.append(i)

                if self.completion_start_time is None:
                    self.completion_start_time = _get_timestamp()

                yield i
        finally:
            await self._finalize()

    async def __anext__(self):
        try:
            item = await self.response.__anext__()
            self.items.append(item)

            if self.completion_start_time is None:
                self.completion_start_time = _get_timestamp()

            return item

        except StopAsyncIteration:
            await self._finalize()

            raise

    async def __aenter__(self):
        return self.__aiter__()

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

    async def _finalize(self):
        model, completion, usage = _extract_streamed_openai_response(
            self.resource, self.items
        )

        # Avoiding the trace-update if trace-id is provided by user.
        if not self.is_nested_trace:
            self.langfuse.trace(id=self.generation.trace_id, output=completion)

        _create_langfuse_update(
            completion,
            self.generation,
            self.completion_start_time,
            model=model,
            usage=usage,
        )

    async def close(self) -> None:
        """Close the response and release the connection.

        Automatically called if the response body is read to completion.
        """
        await self.response.close()

    async def aclose(self) -> None:
        """Close the response and release the connection.

        Automatically called if the response body is read to completion.
        """
        await self.response.aclose()
