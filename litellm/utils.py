# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan
import sys, re, binascii, struct
import litellm
import dotenv, json, traceback, threading, base64, ast
import subprocess, os
from os.path import abspath, join, dirname
import litellm, openai
import itertools
import random, uuid, requests  # type: ignore
from functools import wraps, lru_cache
import datetime, time
import tiktoken
import uuid
from pydantic import BaseModel, ConfigDict
import aiohttp
import textwrap
import logging
import asyncio, httpx, inspect
from inspect import iscoroutine
import copy
from tokenizers import Tokenizer
from dataclasses import (
    dataclass,
    field,
)

import litellm._service_logger  # for storing API inputs, outputs, and metadata
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
from litellm.caching import DualCache
from litellm.types.utils import CostPerToken, ProviderField, ModelInfo

oidc_cache = DualCache()

try:
    # New and recommended way to access resources
    from importlib import resources

    filename = str(resources.files(litellm).joinpath("llms/tokenizers"))
except (ImportError, AttributeError):
    # Old way to access resources, which setuptools deprecated some time ago
    import pkg_resources  # type: ignore

    filename = pkg_resources.resource_filename(__name__, "llms/tokenizers")

os.environ["TIKTOKEN_CACHE_DIR"] = (
    filename  # use local copy of tiktoken b/c of - https://github.com/BerriAI/litellm/issues/1071
)

encoding = tiktoken.get_encoding("cl100k_base")
from importlib import resources

with resources.open_text("litellm.llms.tokenizers", "anthropic_tokenizer.json") as f:
    json_data = json.load(f)
# Convert to str (if necessary)
claude_json_str = json.dumps(json_data)
import importlib.metadata
from ._logging import verbose_logger
from .types.router import LiteLLM_Params
from .integrations.traceloop import TraceloopLogger
from .integrations.athina import AthinaLogger
from .integrations.helicone import HeliconeLogger
from .integrations.aispend import AISpendLogger
from .integrations.berrispend import BerriSpendLogger
from .integrations.supabase import Supabase
from .integrations.lunary import LunaryLogger
from .integrations.prompt_layer import PromptLayerLogger
from .integrations.langsmith import LangsmithLogger
from .integrations.logfire_logger import LogfireLogger, LogfireLevel
from .integrations.weights_biases import WeightsBiasesLogger
from .integrations.custom_logger import CustomLogger
from .integrations.langfuse import LangFuseLogger
from .integrations.openmeter import OpenMeterLogger
from .integrations.lago import LagoLogger
from .integrations.datadog import DataDogLogger
from .integrations.prometheus import PrometheusLogger
from .integrations.prometheus_services import PrometheusServicesLogger
from .integrations.dynamodb import DyanmoDBLogger
from .integrations.s3 import S3Logger
from .integrations.clickhouse import ClickhouseLogger
from .integrations.greenscale import GreenscaleLogger
from .integrations.litedebugger import LiteDebugger
from .proxy._types import KeyManagementSystem
from openai import OpenAIError as OriginalError
from openai._models import BaseModel as OpenAIObject
from .caching import S3Cache, RedisSemanticCache, RedisCache, DualSemanticCache
from .exceptions import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    OpenAIError,
    PermissionDeniedError,
    ContextWindowExceededError,
    ContentPolicyViolationError,
    Timeout,
    APIConnectionError,
    APIError,
    BudgetExceededError,
    UnprocessableEntityError,
)

try:
    from .proxy.enterprise.enterprise_callbacks.generic_api_callback import (
        GenericAPILogger,
    )
except Exception as e:
    verbose_logger.debug(f"Exception import enterprise features {str(e)}")

from typing import (
    cast,
    List,
    Dict,
    Union,
    Optional,
    Literal,
    Any,
    BinaryIO,
    Iterable,
    Tuple,
    Callable,
)
from .caching import Cache
from concurrent.futures import ThreadPoolExecutor

####### ENVIRONMENT VARIABLES ####################
# Adjust to your specific application needs / system capabilities.
MAX_THREADS = 100

# Create a ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)
sentry_sdk_instance = None
capture_exception = None
add_breadcrumb = None
posthog = None
slack_app = None
alerts_channel = None
heliconeLogger = None
athinaLogger = None
promptLayerLogger = None
langsmithLogger = None
logfireLogger = None
weightsBiasesLogger = None
customLogger = None
langFuseLogger = None
openMeterLogger = None
lagoLogger = None
dataDogLogger = None
prometheusLogger = None
dynamoLogger = None
s3Logger = None
genericAPILogger = None
clickHouseLogger = None
greenscaleLogger = None
lunaryLogger = None
aispendLogger = None
berrispendLogger = None
supabaseClient = None
liteDebuggerClient = None
callback_list: Optional[List[str]] = []
user_logger_fn = None
additional_details: Optional[Dict[str, str]] = {}
local_cache: Optional[Dict[str, str]] = {}
last_fetched_at = None
last_fetched_at_keys = None
######## Model Response #########################
# All liteLLM Model responses will be in this format, Follows the OpenAI Format
# https://docs.litellm.ai/docs/completion/output
# {
#   'choices': [
#      {
#         'finish_reason': 'stop',
#         'index': 0,
#         'message': {
#            'role': 'assistant',
#             'content': " I'm doing well, thank you for asking. I am Claude, an AI assistant created by Anthropic."
#         }
#       }
#     ],
#  'created': 1691429984.3852863,
#  'model': 'claude-instant-1',
#  'usage': {'prompt_tokens': 18, 'completion_tokens': 23, 'total_tokens': 41}
# }


class UnsupportedParamsError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url=" https://openai.api.com/v1/")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


def _generate_id():  # private helper function
    return "chatcmpl-" + str(uuid.uuid4())


def map_finish_reason(
    finish_reason: str,
):  # openai supports 5 stop sequences - 'stop', 'length', 'function_call', 'content_filter', 'null'
    # anthropic mapping
    if finish_reason == "stop_sequence":
        return "stop"
    # cohere mapping - https://docs.cohere.com/reference/generate
    elif finish_reason == "COMPLETE":
        return "stop"
    elif finish_reason == "MAX_TOKENS":  # cohere + vertex ai
        return "length"
    elif finish_reason == "ERROR_TOXIC":
        return "content_filter"
    elif (
        finish_reason == "ERROR"
    ):  # openai currently doesn't support an 'error' finish reason
        return "stop"
    # huggingface mapping https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/generate_stream
    elif finish_reason == "eos_token" or finish_reason == "stop_sequence":
        return "stop"
    elif (
        finish_reason == "FINISH_REASON_UNSPECIFIED" or finish_reason == "STOP"
    ):  # vertex ai - got from running `print(dir(response_obj.candidates[0].finish_reason))`: ['FINISH_REASON_UNSPECIFIED', 'MAX_TOKENS', 'OTHER', 'RECITATION', 'SAFETY', 'STOP',]
        return "stop"
    elif finish_reason == "SAFETY":  # vertex ai
        return "content_filter"
    elif finish_reason == "STOP":  # vertex ai
        return "stop"
    elif finish_reason == "end_turn" or finish_reason == "stop_sequence":  # anthropic
        return "stop"
    elif finish_reason == "max_tokens":  # anthropic
        return "length"
    elif finish_reason == "tool_use":  # anthropic
        return "tool_calls"
    elif finish_reason == "content_filtered":
        return "content_filter"
    return finish_reason


class TopLogprob(OpenAIObject):
    token: str
    """The token."""

    bytes: Optional[List[int]] = None
    """A list of integers representing the UTF-8 bytes representation of the token.

    Useful in instances where characters are represented by multiple tokens and
    their byte representations must be combined to generate the correct text
    representation. Can be `null` if there is no bytes representation for the token.
    """

    logprob: float
    """The log probability of this token, if it is within the top 20 most likely
    tokens.

    Otherwise, the value `-9999.0` is used to signify that the token is very
    unlikely.
    """


class ChatCompletionTokenLogprob(OpenAIObject):
    token: str
    """The token."""

    bytes: Optional[List[int]] = None
    """A list of integers representing the UTF-8 bytes representation of the token.

    Useful in instances where characters are represented by multiple tokens and
    their byte representations must be combined to generate the correct text
    representation. Can be `null` if there is no bytes representation for the token.
    """

    logprob: float
    """The log probability of this token, if it is within the top 20 most likely
    tokens.

    Otherwise, the value `-9999.0` is used to signify that the token is very
    unlikely.
    """

    top_logprobs: List[TopLogprob]
    """List of the most likely tokens and their log probability, at this token
    position.

    In rare cases, there may be fewer than the number of requested `top_logprobs`
    returned.
    """


class ChoiceLogprobs(OpenAIObject):
    content: Optional[List[ChatCompletionTokenLogprob]] = None
    """A list of message content tokens with log probability information."""


class FunctionCall(OpenAIObject):
    arguments: str
    name: Optional[str] = None


class Function(OpenAIObject):
    arguments: str
    name: Optional[str] = None

    def __init__(
        self,
        arguments: Union[Dict, str],
        name: Optional[str] = None,
        **params,
    ):
        if isinstance(arguments, Dict):
            arguments = json.dumps(arguments)
        else:
            arguments = arguments

        name = name

        # Build a dictionary with the structure your BaseModel expects
        data = {"arguments": arguments, "name": name, **params}

        super(Function, self).__init__(**data)

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class ChatCompletionDeltaToolCall(OpenAIObject):
    id: Optional[str] = None
    function: Function
    type: Optional[str] = None
    index: int


class HiddenParams(OpenAIObject):
    original_response: Optional[str] = None
    model_id: Optional[str] = None  # used in Router for individual deployments
    api_base: Optional[str] = None  # returns api base used for making completion call

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()  # noqa
        except:
            # if using pydantic v1
            return self.dict()


class ChatCompletionMessageToolCall(OpenAIObject):
    def __init__(
        self,
        function: Union[Dict, Function],
        id: Optional[str] = None,
        type: Optional[str] = None,
        **params,
    ):
        super(ChatCompletionMessageToolCall, self).__init__(**params)
        if isinstance(function, Dict):
            self.function = Function(**function)
        else:
            self.function = function

        if id is not None:
            self.id = id
        else:
            self.id = f"{uuid.uuid4()}"

        if type is not None:
            self.type = type
        else:
            self.type = "function"

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class Message(OpenAIObject):
    def __init__(
        self,
        content: Optional[str] = "default",
        role="assistant",
        logprobs=None,
        function_call=None,
        tool_calls=None,
        **params,
    ):
        super(Message, self).__init__(**params)
        self.content = content
        self.role = role
        if function_call is not None:
            self.function_call = FunctionCall(**function_call)

        if tool_calls is not None:
            self.tool_calls = []
            for tool_call in tool_calls:
                self.tool_calls.append(ChatCompletionMessageToolCall(**tool_call))

        if logprobs is not None:
            self._logprobs = ChoiceLogprobs(**logprobs)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()  # noqa
        except:
            # if using pydantic v1
            return self.dict()


class Delta(OpenAIObject):
    def __init__(
        self,
        content=None,
        role=None,
        function_call=None,
        tool_calls=None,
        **params,
    ):
        super(Delta, self).__init__(**params)
        self.content = content
        self.role = role

        if function_call is not None and isinstance(function_call, dict):
            self.function_call = FunctionCall(**function_call)
        else:
            self.function_call = function_call
        if tool_calls is not None and isinstance(tool_calls, list):
            self.tool_calls = []
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    if tool_call.get("index", None) is None:
                        tool_call["index"] = 0
                    self.tool_calls.append(ChatCompletionDeltaToolCall(**tool_call))
                elif isinstance(tool_call, ChatCompletionDeltaToolCall):
                    self.tool_calls.append(tool_call)
        else:
            self.tool_calls = tool_calls

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class Choices(OpenAIObject):
    def __init__(
        self,
        finish_reason=None,
        index=0,
        message=None,
        logprobs=None,
        enhancements=None,
        **params,
    ):
        super(Choices, self).__init__(**params)
        self.finish_reason = (
            map_finish_reason(finish_reason) or "stop"
        )  # set finish_reason for all responses
        self.index = index
        if message is None:
            self.message = Message()
        else:
            if isinstance(message, Message):
                self.message = message
            elif isinstance(message, dict):
                self.message = Message(**message)
        if logprobs is not None:
            self.logprobs = logprobs
        if enhancements is not None:
            self.enhancements = enhancements

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class Usage(OpenAIObject):
    def __init__(
        self, prompt_tokens=None, completion_tokens=None, total_tokens=None, **params
    ):
        super(Usage, self).__init__(**params)
        if prompt_tokens:
            self.prompt_tokens = prompt_tokens
        if completion_tokens:
            self.completion_tokens = completion_tokens
        if total_tokens:
            self.total_tokens = total_tokens

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class StreamingChoices(OpenAIObject):
    def __init__(
        self,
        finish_reason=None,
        index=0,
        delta: Optional[Delta] = None,
        logprobs=None,
        enhancements=None,
        **params,
    ):
        super(StreamingChoices, self).__init__(**params)
        if finish_reason:
            self.finish_reason = finish_reason
        else:
            self.finish_reason = None
        self.index = index
        if delta is not None:
            if isinstance(delta, Delta):
                self.delta = delta
            elif isinstance(delta, dict):
                self.delta = Delta(**delta)
        else:
            self.delta = Delta()
        if enhancements is not None:
            self.enhancements = enhancements

        if logprobs is not None and isinstance(logprobs, dict):
            self.logprobs = ChoiceLogprobs(**logprobs)
        else:
            self.logprobs = logprobs  # type: ignore

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class ModelResponse(OpenAIObject):
    id: str
    """A unique identifier for the completion."""

    choices: List[Union[Choices, StreamingChoices]]
    """The list of completion choices the model generated for the input prompt."""

    created: int
    """The Unix timestamp (in seconds) of when the completion was created."""

    model: Optional[str] = None
    """The model used for completion."""

    object: str
    """The object type, which is always "text_completion" """

    system_fingerprint: Optional[str] = None
    """This fingerprint represents the backend configuration that the model runs with.

    Can be used in conjunction with the `seed` request parameter to understand when
    backend changes have been made that might impact determinism.
    """

    _hidden_params: dict = {}

    def __init__(
        self,
        id=None,
        choices=None,
        created=None,
        model=None,
        object=None,
        system_fingerprint=None,
        usage=None,
        stream=None,
        stream_options=None,
        response_ms=None,
        hidden_params=None,
        **params,
    ):
        if stream is not None and stream == True:
            object = "chat.completion.chunk"
            if choices is not None and isinstance(choices, list):
                new_choices = []
                for choice in choices:
                    if isinstance(choice, StreamingChoices):
                        _new_choice = choice
                    elif isinstance(choice, dict):
                        _new_choice = StreamingChoices(**choice)
                    new_choices.append(_new_choice)
                choices = new_choices
            else:
                choices = [StreamingChoices()]
        else:
            if model in litellm.open_ai_embedding_models:
                object = "embedding"
            else:
                object = "chat.completion"
            if choices is not None and isinstance(choices, list):
                new_choices = []
                for choice in choices:
                    if isinstance(choice, Choices):
                        _new_choice = choice
                    elif isinstance(choice, dict):
                        _new_choice = Choices(**choice)
                    new_choices.append(_new_choice)
                choices = new_choices
            else:
                choices = [Choices()]
        if id is None:
            id = _generate_id()
        else:
            id = id
        if created is None:
            created = int(time.time())
        else:
            created = created
        model = model
        if usage is not None:
            if isinstance(usage, dict):
                usage = Usage(**usage)
            else:
                usage = usage
        elif stream is None or stream == False:
            usage = Usage()
        if hidden_params:
            self._hidden_params = hidden_params

        init_values = {
            "id": id,
            "choices": choices,
            "created": created,
            "model": model,
            "object": object,
            "system_fingerprint": system_fingerprint,
        }

        if usage is not None:
            init_values["usage"] = usage

        super().__init__(
            **init_values,
            **params,
        )

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()  # noqa
        except:
            # if using pydantic v1
            return self.dict()


class Embedding(OpenAIObject):
    embedding: Union[list, str] = []
    index: int
    object: str

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class EmbeddingResponse(OpenAIObject):
    model: Optional[str] = None
    """The model used for embedding."""

    data: Optional[List] = None
    """The actual embedding value"""

    object: str
    """The object type, which is always "embedding" """

    usage: Optional[Usage] = None
    """Usage statistics for the embedding request."""

    _hidden_params: dict = {}

    def __init__(
        self,
        model=None,
        usage=None,
        stream=False,
        response_ms=None,
        data=None,
        **params,
    ):
        object = "list"
        if response_ms:
            _response_ms = response_ms
        else:
            _response_ms = None
        if data:
            data = data
        else:
            data = None

        if usage:
            usage = usage
        else:
            usage = Usage()

        model = model
        super().__init__(model=model, object=object, data=data, usage=usage)

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()  # noqa
        except:
            # if using pydantic v1
            return self.dict()


class Logprobs(OpenAIObject):
    text_offset: List[int]
    token_logprobs: List[float]
    tokens: List[str]
    top_logprobs: List[Dict[str, float]]


class TextChoices(OpenAIObject):
    def __init__(self, finish_reason=None, index=0, text=None, logprobs=None, **params):
        super(TextChoices, self).__init__(**params)
        if finish_reason:
            self.finish_reason = map_finish_reason(finish_reason)
        else:
            self.finish_reason = None
        self.index = index
        if text is not None:
            self.text = text
        else:
            self.text = None
        if logprobs is None:
            self.logprobs = None
        else:
            if isinstance(logprobs, dict):
                self.logprobs = Logprobs(**logprobs)
            else:
                self.logprobs = logprobs

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()  # noqa
        except:
            # if using pydantic v1
            return self.dict()


class TextCompletionResponse(OpenAIObject):
    """
    {
        "id": response["id"],
        "object": "text_completion",
        "created": response["created"],
        "model": response["model"],
        "choices": [
        {
            "text": response["choices"][0]["message"]["content"],
            "index": response["choices"][0]["index"],
            "logprobs": transformed_logprobs,
            "finish_reason": response["choices"][0]["finish_reason"]
        }
        ],
        "usage": response["usage"]
    }
    """

    id: str
    object: str
    created: int
    model: Optional[str]
    choices: List[TextChoices]
    usage: Optional[Usage]
    _response_ms: Optional[int] = None
    _hidden_params: HiddenParams

    def __init__(
        self,
        id=None,
        choices=None,
        created=None,
        model=None,
        usage=None,
        stream=False,
        response_ms=None,
        object=None,
        **params,
    ):

        if stream:
            object = "text_completion.chunk"
            choices = [TextChoices()]
        else:
            object = "text_completion"
            if choices is not None and isinstance(choices, list):
                new_choices = []
                for choice in choices:

                    if isinstance(choice, TextChoices):
                        _new_choice = choice
                    elif isinstance(choice, dict):
                        _new_choice = TextChoices(**choice)
                    new_choices.append(_new_choice)
                choices = new_choices
            else:
                choices = [TextChoices()]
        if object is not None:
            object = object
        if id is None:
            id = _generate_id()
        else:
            id = id
        if created is None:
            created = int(time.time())
        else:
            created = created

        model = model
        if usage:
            usage = usage
        else:
            usage = Usage()

        super(TextCompletionResponse, self).__init__(
            id=id,
            object=object,
            created=created,
            model=model,
            choices=choices,
            usage=usage,
            **params,
        )

        if response_ms:
            self._response_ms = response_ms
        else:
            self._response_ms = None
        self._hidden_params = HiddenParams()

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class ImageObject(OpenAIObject):
    """
    Represents the url or the content of an image generated by the OpenAI API.

    Attributes:
    b64_json: The base64-encoded JSON of the generated image, if response_format is b64_json.
    url: The URL of the generated image, if response_format is url (default).
    revised_prompt: The prompt that was used to generate the image, if there was any revision to the prompt.

    https://platform.openai.com/docs/api-reference/images/object
    """

    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None

    def __init__(self, b64_json=None, url=None, revised_prompt=None):

        super().__init__(b64_json=b64_json, url=url, revised_prompt=revised_prompt)

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()  # noqa
        except:
            # if using pydantic v1
            return self.dict()


class ImageResponse(OpenAIObject):
    created: Optional[int] = None

    data: Optional[List[ImageObject]] = None

    usage: Optional[dict] = None

    _hidden_params: dict = {}

    def __init__(self, created=None, data=None, response_ms=None):
        if response_ms:
            _response_ms = response_ms
        else:
            _response_ms = None
        if data:
            data = data
        else:
            data = None

        if created:
            created = created
        else:
            created = None

        super().__init__(data=data, created=created)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()  # noqa
        except:
            # if using pydantic v1
            return self.dict()


class TranscriptionResponse(OpenAIObject):
    text: Optional[str] = None

    _hidden_params: dict = {}

    def __init__(self, text=None):
        super().__init__(text=text)

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()  # noqa
        except:
            # if using pydantic v1
            return self.dict()


############################################################
def print_verbose(
    print_statement,
    logger_only: bool = False,
    log_level: Literal["DEBUG", "INFO"] = "DEBUG",
):
    try:
        if log_level == "DEBUG":
            verbose_logger.debug(print_statement)
        elif log_level == "INFO":
            verbose_logger.info(print_statement)
        if litellm.set_verbose == True and logger_only == False:
            print(print_statement)  # noqa
    except:
        pass


####### LOGGING ###################
from enum import Enum


class CallTypes(Enum):
    embedding = "embedding"
    aembedding = "aembedding"
    completion = "completion"
    acompletion = "acompletion"
    atext_completion = "atext_completion"
    text_completion = "text_completion"
    image_generation = "image_generation"
    aimage_generation = "aimage_generation"
    moderation = "moderation"
    amoderation = "amoderation"
    atranscription = "atranscription"
    transcription = "transcription"
    aspeech = "aspeech"
    speech = "speech"


# Logging function -> log the exact model details + what's being sent | Non-BlockingP
class Logging:
    global supabaseClient, liteDebuggerClient, promptLayerLogger, weightsBiasesLogger, langsmithLogger, logfireLogger, capture_exception, add_breadcrumb, lunaryLogger

    custom_pricing: bool = False
    stream_options = None

    def __init__(
        self,
        model,
        messages,
        stream,
        call_type,
        start_time,
        litellm_call_id,
        function_id,
        dynamic_success_callbacks=None,
        dynamic_failure_callbacks=None,
        dynamic_async_success_callbacks=None,
        langfuse_public_key=None,
        langfuse_secret=None,
    ):
        if call_type not in [item.value for item in CallTypes]:
            allowed_values = ", ".join([item.value for item in CallTypes])
            raise ValueError(
                f"Invalid call_type {call_type}. Allowed values: {allowed_values}"
            )
        if messages is not None:
            if isinstance(messages, str):
                messages = [
                    {"role": "user", "content": messages}
                ]  # convert text completion input to the chat completion format
            elif (
                isinstance(messages, list)
                and len(messages) > 0
                and isinstance(messages[0], str)
            ):
                new_messages = []
                for m in messages:
                    new_messages.append({"role": "user", "content": m})
                messages = new_messages
        self.model = model
        self.messages = messages
        self.stream = stream
        self.start_time = start_time  # log the call start time
        self.call_type = call_type
        self.litellm_call_id = litellm_call_id
        self.function_id = function_id
        self.streaming_chunks = []  # for generating complete stream response
        self.sync_streaming_chunks = []  # for generating complete stream response
        self.model_call_details = {}
        self.dynamic_input_callbacks = []  # [TODO] callbacks set for just that call
        self.dynamic_failure_callbacks = dynamic_failure_callbacks
        self.dynamic_success_callbacks = (
            dynamic_success_callbacks  # callbacks set for just that call
        )
        self.dynamic_async_success_callbacks = (
            dynamic_async_success_callbacks  # callbacks set for just that call
        )
        ## DYNAMIC LANGFUSE KEYS ##
        self.langfuse_public_key = langfuse_public_key
        self.langfuse_secret = langfuse_secret
        ## TIME TO FIRST TOKEN LOGGING ##
        self.completion_start_time: Optional[datetime.datetime] = None

    def update_environment_variables(
        self, model, user, optional_params, litellm_params, **additional_params
    ):
        self.optional_params = optional_params
        self.model = model
        self.user = user
        self.litellm_params = litellm_params
        self.logger_fn = litellm_params.get("logger_fn", None)
        print_verbose(f"self.optional_params: {self.optional_params}")

        self.model_call_details = {
            "model": self.model,
            "messages": self.messages,
            "optional_params": self.optional_params,
            "litellm_params": self.litellm_params,
            "start_time": self.start_time,
            "stream": self.stream,
            "user": user,
            "call_type": str(self.call_type),
            "litellm_call_id": self.litellm_call_id,
            "completion_start_time": self.completion_start_time,
            **self.optional_params,
            **additional_params,
        }

        ## check if stream options is set ##  - used by CustomStreamWrapper for easy instrumentation
        if "stream_options" in additional_params:
            self.stream_options = additional_params["stream_options"]
        ## check if custom pricing set ##
        if (
            litellm_params.get("input_cost_per_token") is not None
            or litellm_params.get("input_cost_per_second") is not None
            or litellm_params.get("output_cost_per_token") is not None
            or litellm_params.get("output_cost_per_second") is not None
        ):
            self.custom_pricing = True

    def _pre_call(self, input, api_key, model=None, additional_args={}):
        """
        Common helper function across the sync + async pre-call function
        """
        # print_verbose(f"logging pre call for model: {self.model} with call type: {self.call_type}")
        self.model_call_details["input"] = input
        self.model_call_details["api_key"] = api_key
        self.model_call_details["additional_args"] = additional_args
        self.model_call_details["log_event_type"] = "pre_api_call"
        if (
            model
        ):  # if model name was changes pre-call, overwrite the initial model call name with the new one
            self.model_call_details["model"] = model

    def pre_call(self, input, api_key, model=None, additional_args={}):
        # Log the exact input to the LLM API
        litellm.error_logs["PRE_CALL"] = locals()
        try:
            self._pre_call(
                input=input,
                api_key=api_key,
                model=model,
                additional_args=additional_args,
            )

            # User Logging -> if you pass in a custom logging function
            headers = additional_args.get("headers", {})
            if headers is None:
                headers = {}
            data = additional_args.get("complete_input_dict", {})
            api_base = additional_args.get("api_base", "")
            self.model_call_details["litellm_params"]["api_base"] = str(
                api_base
            )  # used for alerting
            masked_headers = {
                k: (
                    (v[:-44] + "*" * 44)
                    if (isinstance(v, str) and len(v) > 44)
                    else "*****"
                )
                for k, v in headers.items()
            }
            formatted_headers = " ".join(
                [f"-H '{k}: {v}'" for k, v in masked_headers.items()]
            )

            verbose_logger.debug(f"PRE-API-CALL ADDITIONAL ARGS: {additional_args}")

            curl_command = "\n\nPOST Request Sent from LiteLLM:\n"
            curl_command += "curl -X POST \\\n"
            curl_command += f"{api_base} \\\n"
            curl_command += (
                f"{formatted_headers} \\\n" if formatted_headers.strip() != "" else ""
            )
            curl_command += f"-d '{str(data)}'\n"
            if additional_args.get("request_str", None) is not None:
                # print the sagemaker / bedrock client request
                curl_command = "\nRequest Sent from LiteLLM:\n"
                curl_command += additional_args.get("request_str", None)
            elif api_base == "":
                curl_command = self.model_call_details

            # only print verbose if verbose logger is not set
            if verbose_logger.level == 0:
                # this means verbose logger was not switched on - user is in litellm.set_verbose=True
                print_verbose(f"\033[92m{curl_command}\033[0m\n")

            if litellm.json_logs:
                verbose_logger.debug(
                    "POST Request Sent from LiteLLM",
                    extra={"api_base": {api_base}, **masked_headers},
                )
            else:
                verbose_logger.debug(f"\033[92m{curl_command}\033[0m\n")
            # log raw request to provider (like LangFuse)
            try:
                # [Non-blocking Extra Debug Information in metadata]
                _litellm_params = self.model_call_details.get("litellm_params", {})
                _metadata = _litellm_params.get("metadata", {}) or {}
                if (
                    litellm.turn_off_message_logging is not None
                    and litellm.turn_off_message_logging is True
                ):
                    _metadata["raw_request"] = (
                        "redacted by litellm. \
                        'litellm.turn_off_message_logging=True'"
                    )
                else:
                    _metadata["raw_request"] = str(curl_command)
            except Exception as e:
                _metadata["raw_request"] = (
                    "Unable to Log \
                    raw request: {}".format(
                        str(e)
                    )
                )
            if self.logger_fn and callable(self.logger_fn):
                try:
                    self.logger_fn(
                        self.model_call_details
                    )  # Expectation: any logger function passed in by the user should accept a dict object
                except Exception as e:
                    print_verbose(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
                    )
            # Input Integration Logging -> If you want to log the fact that an attempt to call the model was made
            callbacks = litellm.input_callback + self.dynamic_input_callbacks
            for callback in callbacks:
                try:
                    if callback == "supabase":
                        print_verbose("reaches supabase for logging!")
                        model = self.model_call_details["model"]
                        messages = self.model_call_details["input"]
                        print_verbose(f"supabaseClient: {supabaseClient}")
                        supabaseClient.input_log_event(
                            model=model,
                            messages=messages,
                            end_user=self.model_call_details.get("user", "default"),
                            litellm_call_id=self.litellm_params["litellm_call_id"],
                            print_verbose=print_verbose,
                        )
                    elif callback == "sentry" and add_breadcrumb:
                        try:
                            details_to_log = copy.deepcopy(self.model_call_details)
                        except:
                            details_to_log = self.model_call_details
                        if litellm.turn_off_message_logging:
                            # make a copy of the _model_Call_details and log it
                            details_to_log.pop("messages", None)
                            details_to_log.pop("input", None)
                            details_to_log.pop("prompt", None)

                        add_breadcrumb(
                            category="litellm.llm_call",
                            message=f"Model Call Details pre-call: {details_to_log}",
                            level="info",
                        )
                    elif isinstance(callback, CustomLogger):  # custom logger class
                        callback.log_pre_api_call(
                            model=self.model,
                            messages=self.messages,
                            kwargs=self.model_call_details,
                        )
                    elif callable(callback):  # custom logger functions
                        customLogger.log_input_event(
                            model=self.model,
                            messages=self.messages,
                            kwargs=self.model_call_details,
                            print_verbose=print_verbose,
                            callback_func=callback,
                        )
                except Exception as e:
                    verbose_logger.error(
                        "litellm.Logging.pre_call(): Exception occured - {}".format(
                            str(e)
                        )
                    )
                    verbose_logger.debug(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while input logging with integrations {traceback.format_exc()}"
                    )
                    print_verbose(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except:
            print_verbose(
                f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
            )
            print_verbose(
                f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
            )
            if capture_exception:  # log this error to sentry for debugging
                capture_exception(e)

    def post_call(
        self, original_response, input=None, api_key=None, additional_args={}
    ):
        # Log the exact result from the LLM API, for streaming - log the type of response received
        litellm.error_logs["POST_CALL"] = locals()
        if isinstance(original_response, dict):
            original_response = json.dumps(original_response)
        try:
            self.model_call_details["input"] = input
            self.model_call_details["api_key"] = api_key
            self.model_call_details["original_response"] = original_response
            self.model_call_details["additional_args"] = additional_args
            self.model_call_details["log_event_type"] = "post_api_call"
            # User Logging -> if you pass in a custom logging function
            print_verbose(
                f"RAW RESPONSE:\n{self.model_call_details.get('original_response', self.model_call_details)}\n\n",
                log_level="DEBUG",
            )
            if self.logger_fn and callable(self.logger_fn):
                try:
                    self.logger_fn(
                        self.model_call_details
                    )  # Expectation: any logger function passed in by the user should accept a dict object
                except Exception as e:
                    print_verbose(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
                    )
            self.redact_message_input_output_from_logging(result=original_response)
            # Input Integration Logging -> If you want to log the fact that an attempt to call the model was made

            callbacks = litellm.input_callback + self.dynamic_input_callbacks
            for callback in callbacks:
                try:
                    if callback == "lite_debugger":
                        print_verbose("reaches litedebugger for post-call logging!")
                        print_verbose(f"liteDebuggerClient: {liteDebuggerClient}")
                        liteDebuggerClient.post_call_log_event(
                            original_response=original_response,
                            litellm_call_id=self.litellm_params["litellm_call_id"],
                            print_verbose=print_verbose,
                            call_type=self.call_type,
                            stream=self.stream,
                        )
                    elif callback == "sentry" and add_breadcrumb:
                        print_verbose("reaches sentry breadcrumbing")
                        try:
                            details_to_log = copy.deepcopy(self.model_call_details)
                        except:
                            details_to_log = self.model_call_details
                        if litellm.turn_off_message_logging:
                            # make a copy of the _model_Call_details and log it
                            details_to_log.pop("messages", None)
                            details_to_log.pop("input", None)
                            details_to_log.pop("prompt", None)

                        add_breadcrumb(
                            category="litellm.llm_call",
                            message=f"Model Call Details post-call: {details_to_log}",
                            level="info",
                        )
                    elif isinstance(callback, CustomLogger):  # custom logger class
                        callback.log_post_api_call(
                            kwargs=self.model_call_details,
                            response_obj=None,
                            start_time=self.start_time,
                            end_time=None,
                        )
                except Exception as e:
                    print_verbose(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while post-call logging with integrations {traceback.format_exc()}"
                    )
                    print_verbose(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except:
            print_verbose(
                f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
            )
            pass

    def _success_handler_helper_fn(
        self, result=None, start_time=None, end_time=None, cache_hit=None
    ):
        try:
            if start_time is None:
                start_time = self.start_time
            if end_time is None:
                end_time = datetime.datetime.now()
            if self.completion_start_time is None:
                self.completion_start_time = end_time
                self.model_call_details["completion_start_time"] = (
                    self.completion_start_time
                )
            self.model_call_details["log_event_type"] = "successful_api_call"
            self.model_call_details["end_time"] = end_time
            self.model_call_details["cache_hit"] = cache_hit
            ## if model in model cost map - log the response cost
            ## else set cost to None
            verbose_logger.debug(f"Model={self.model};")
            if (
                result is not None
                and (
                    isinstance(result, ModelResponse)
                    or isinstance(result, EmbeddingResponse)
                    or isinstance(result, ImageResponse)
                    or isinstance(result, TranscriptionResponse)
                    or isinstance(result, TextCompletionResponse)
                )
                and self.stream != True
            ):  # handle streaming separately
                self.model_call_details["response_cost"] = (
                    litellm.response_cost_calculator(
                        response_object=result,
                        model=self.model,
                        cache_hit=self.model_call_details.get("cache_hit", False),
                        custom_llm_provider=self.model_call_details.get(
                            "custom_llm_provider", None
                        ),
                        base_model=_get_base_model_from_metadata(
                            model_call_details=self.model_call_details
                        ),
                        call_type=self.call_type,
                        optional_params=self.optional_params,
                    )
                )
            else:  # streaming chunks + image gen.
                self.model_call_details["response_cost"] = None

            if (
                litellm.max_budget
                and self.stream == False
                and result is not None
                and "content" in result
            ):
                time_diff = (end_time - start_time).total_seconds()
                float_diff = float(time_diff)
                litellm._current_cost += litellm.completion_cost(
                    model=self.model,
                    prompt="",
                    completion=result["content"],
                    total_time=float_diff,
                )

            return start_time, end_time, result
        except Exception as e:
            raise Exception(f"[Non-Blocking] LiteLLM.Success_Call Error: {str(e)}")

    def success_handler(
        self, result=None, start_time=None, end_time=None, cache_hit=None, **kwargs
    ):
        print_verbose(f"Logging Details LiteLLM-Success Call: {cache_hit}")
        start_time, end_time, result = self._success_handler_helper_fn(
            start_time=start_time,
            end_time=end_time,
            result=result,
            cache_hit=cache_hit,
        )
        # print(f"original response in success handler: {self.model_call_details['original_response']}")
        try:
            print_verbose(f"success callbacks: {litellm.success_callback}")
            ## BUILD COMPLETE STREAMED RESPONSE
            complete_streaming_response = None
            if self.stream and isinstance(result, ModelResponse):
                if (
                    result.choices[0].finish_reason is not None
                ):  # if it's the last chunk
                    self.sync_streaming_chunks.append(result)
                    # print_verbose(f"final set of received chunks: {self.sync_streaming_chunks}")
                    try:
                        complete_streaming_response = litellm.stream_chunk_builder(
                            self.sync_streaming_chunks,
                            messages=self.model_call_details.get("messages", None),
                            start_time=start_time,
                            end_time=end_time,
                        )
                    except Exception as e:

                        complete_streaming_response = None
                else:
                    self.sync_streaming_chunks.append(result)

            if complete_streaming_response is not None:
                print_verbose(
                    f"Logging Details LiteLLM-Success Call streaming complete"
                )
                self.model_call_details["complete_streaming_response"] = (
                    complete_streaming_response
                )
                self.model_call_details["response_cost"] = (
                    litellm.response_cost_calculator(
                        response_object=complete_streaming_response,
                        model=self.model,
                        cache_hit=self.model_call_details.get("cache_hit", False),
                        custom_llm_provider=self.model_call_details.get(
                            "custom_llm_provider", None
                        ),
                        base_model=_get_base_model_from_metadata(
                            model_call_details=self.model_call_details
                        ),
                        call_type=self.call_type,
                        optional_params=self.optional_params,
                    )
                )
            if self.dynamic_success_callbacks is not None and isinstance(
                self.dynamic_success_callbacks, list
            ):
                callbacks = self.dynamic_success_callbacks
                ## keep the internal functions ##
                for callback in litellm.success_callback:
                    if (
                        isinstance(callback, CustomLogger)
                        and "_PROXY_" in callback.__class__.__name__
                    ):
                        callbacks.append(callback)
            else:
                callbacks = litellm.success_callback

            self.redact_message_input_output_from_logging(result=result)

            for callback in callbacks:
                try:
                    litellm_params = self.model_call_details.get("litellm_params", {})
                    if litellm_params.get("no-log", False) == True:
                        # proxy cost tracking cal backs should run
                        if not (
                            isinstance(callback, CustomLogger)
                            and "_PROXY_" in callback.__class__.__name__
                        ):
                            print_verbose("no-log request, skipping logging")
                            continue
                    if callback == "lite_debugger":
                        print_verbose("reaches lite_debugger for logging!")
                        print_verbose(f"liteDebuggerClient: {liteDebuggerClient}")
                        print_verbose(
                            f"liteDebuggerClient details function {self.call_type} and stream set to {self.stream}"
                        )
                        liteDebuggerClient.log_event(
                            end_user=kwargs.get("user", "default"),
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            litellm_call_id=self.litellm_call_id,
                            print_verbose=print_verbose,
                            call_type=self.call_type,
                            stream=self.stream,
                        )
                    if callback == "promptlayer":
                        print_verbose("reaches promptlayer for logging!")
                        promptLayerLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "supabase":
                        print_verbose("reaches supabase for logging!")
                        kwargs = self.model_call_details

                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            if "complete_streaming_response" not in kwargs:
                                continue
                            else:
                                print_verbose("reaches supabase for streaming logging!")
                                result = kwargs["complete_streaming_response"]

                        model = kwargs["model"]
                        messages = kwargs["messages"]
                        optional_params = kwargs.get("optional_params", {})
                        litellm_params = kwargs.get("litellm_params", {})
                        supabaseClient.log_event(
                            model=model,
                            messages=messages,
                            end_user=optional_params.get("user", "default"),
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            litellm_call_id=litellm_params.get(
                                "litellm_call_id", str(uuid.uuid4())
                            ),
                            print_verbose=print_verbose,
                        )
                    if callback == "wandb":
                        print_verbose("reaches wandb for logging!")
                        weightsBiasesLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "langsmith":
                        print_verbose("reaches langsmith for logging!")
                        if self.stream:
                            if "complete_streaming_response" not in kwargs:
                                continue
                            else:
                                print_verbose(
                                    "reaches langsmith for streaming logging!"
                                )
                                result = kwargs["complete_streaming_response"]
                        langsmithLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "logfire":
                        global logfireLogger
                        verbose_logger.debug("reaches logfire for success logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v

                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            if "complete_streaming_response" not in kwargs:
                                continue
                            else:
                                print_verbose("reaches logfire for streaming logging!")
                                result = kwargs["complete_streaming_response"]

                        logfireLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                            level=LogfireLevel.INFO.value,
                        )

                    if callback == "lunary":
                        print_verbose("reaches lunary for logging!")
                        model = self.model
                        kwargs = self.model_call_details

                        input = kwargs.get("messages", kwargs.get("input", None))

                        type = (
                            "embed"
                            if self.call_type == CallTypes.embedding.value
                            else "llm"
                        )

                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            if "complete_streaming_response" not in kwargs:
                                continue
                            else:
                                result = kwargs["complete_streaming_response"]

                        lunaryLogger.log_event(
                            type=type,
                            kwargs=kwargs,
                            event="end",
                            model=model,
                            input=input,
                            user_id=kwargs.get("user", None),
                            # user_props=self.model_call_details.get("user_props", None),
                            extra=kwargs.get("optional_params", {}),
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            run_id=self.litellm_call_id,
                            print_verbose=print_verbose,
                        )
                    if callback == "helicone":
                        print_verbose("reaches helicone for logging!")
                        model = self.model
                        messages = self.model_call_details["input"]
                        heliconeLogger.log_success(
                            model=model,
                            messages=messages,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "langfuse":
                        global langFuseLogger
                        verbose_logger.debug("reaches langfuse for success logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            verbose_logger.debug(
                                f"is complete_streaming_response in kwargs: {kwargs.get('complete_streaming_response', None)}"
                            )
                            if complete_streaming_response is None:
                                continue
                            else:
                                print_verbose("reaches langfuse for streaming logging!")
                                result = kwargs["complete_streaming_response"]
                        if langFuseLogger is None or (
                            (
                                self.langfuse_public_key is not None
                                and self.langfuse_public_key
                                != langFuseLogger.public_key
                            )
                            and (
                                self.langfuse_public_key is not None
                                and self.langfuse_public_key
                                != langFuseLogger.public_key
                            )
                        ):
                            langFuseLogger = LangFuseLogger(
                                langfuse_public_key=self.langfuse_public_key,
                                langfuse_secret=self.langfuse_secret,
                            )
                        langFuseLogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                        )
                    if callback == "datadog":
                        global dataDogLogger
                        verbose_logger.debug("reaches datadog for success logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            verbose_logger.debug(
                                f"datadog: is complete_streaming_response in kwargs: {kwargs.get('complete_streaming_response', None)}"
                            )
                            if complete_streaming_response is None:
                                continue
                            else:
                                print_verbose("reaches datadog for streaming logging!")
                                result = kwargs["complete_streaming_response"]
                        dataDogLogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                        )
                    if callback == "prometheus":
                        global prometheusLogger
                        verbose_logger.debug("reaches prometheus for success logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            verbose_logger.debug(
                                f"prometheus: is complete_streaming_response in kwargs: {kwargs.get('complete_streaming_response', None)}"
                            )
                            if complete_streaming_response is None:
                                continue
                            else:
                                print_verbose(
                                    "reaches prometheus for streaming logging!"
                                )
                                result = kwargs["complete_streaming_response"]
                        prometheusLogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                        )
                    if callback == "generic":
                        global genericAPILogger
                        verbose_logger.debug("reaches langfuse for success logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            verbose_logger.debug(
                                f"is complete_streaming_response in kwargs: {kwargs.get('complete_streaming_response', None)}"
                            )
                            if complete_streaming_response is None:
                                continue
                            else:
                                print_verbose("reaches langfuse for streaming logging!")
                                result = kwargs["complete_streaming_response"]
                        if genericAPILogger is None:
                            genericAPILogger = GenericAPILogger()
                        genericAPILogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                        )
                    if callback == "clickhouse":
                        global clickHouseLogger
                        verbose_logger.debug("reaches clickhouse for success logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            verbose_logger.debug(
                                f"is complete_streaming_response in kwargs: {kwargs.get('complete_streaming_response', None)}"
                            )
                            if complete_streaming_response is None:
                                continue
                            else:
                                print_verbose(
                                    "reaches clickhouse for streaming logging!"
                                )
                                result = kwargs["complete_streaming_response"]
                        if clickHouseLogger is None:
                            clickHouseLogger = ClickhouseLogger()
                        clickHouseLogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                        )
                    if callback == "greenscale":
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if self.stream:
                            verbose_logger.debug(
                                f"is complete_streaming_response in kwargs: {kwargs.get('complete_streaming_response', None)}"
                            )
                            if complete_streaming_response is None:
                                continue
                            else:
                                print_verbose(
                                    "reaches greenscale for streaming logging!"
                                )
                                result = kwargs["complete_streaming_response"]

                        greenscaleLogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "cache" and litellm.cache is not None:
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        print_verbose("success_callback: reaches cache for logging!")
                        kwargs = self.model_call_details
                        if self.stream:
                            if "complete_streaming_response" not in kwargs:
                                print_verbose(
                                    f"success_callback: reaches cache for logging, there is no complete_streaming_response. Kwargs={kwargs}\n\n"
                                )
                                pass
                            else:
                                print_verbose(
                                    "success_callback: reaches cache for logging, there is a complete_streaming_response. Adding to cache"
                                )
                                result = kwargs["complete_streaming_response"]
                                # only add to cache once we have a complete streaming response
                                litellm.cache.add_cache(result, **kwargs)
                    if callback == "athina":
                        deep_copy = {}
                        for k, v in self.model_call_details.items():
                            deep_copy[k] = v
                        athinaLogger.log_event(
                            kwargs=deep_copy,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "traceloop":
                        deep_copy = {}
                        for k, v in self.model_call_details.items():
                            if k != "original_response":
                                deep_copy[k] = v
                        traceloopLogger.log_event(
                            kwargs=deep_copy,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                        )
                    if callback == "s3":
                        global s3Logger
                        if s3Logger is None:
                            s3Logger = S3Logger()
                        if self.stream:
                            if "complete_streaming_response" in self.model_call_details:
                                print_verbose(
                                    "S3Logger Logger: Got Stream Event - Completed Stream Response"
                                )
                                s3Logger.log_event(
                                    kwargs=self.model_call_details,
                                    response_obj=self.model_call_details[
                                        "complete_streaming_response"
                                    ],
                                    start_time=start_time,
                                    end_time=end_time,
                                    print_verbose=print_verbose,
                                )
                            else:
                                print_verbose(
                                    "S3Logger Logger: Got Stream Event - No complete stream response as yet"
                                )
                        else:
                            s3Logger.log_event(
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                                print_verbose=print_verbose,
                            )
                    if (
                        callback == "openmeter"
                        and self.model_call_details.get("litellm_params", {}).get(
                            "acompletion", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aembedding", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aimage_generation", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "atranscription", False
                        )
                        == False
                    ):
                        global openMeterLogger
                        if openMeterLogger is None:
                            print_verbose("Instantiates openmeter client")
                            openMeterLogger = OpenMeterLogger()
                        if self.stream and complete_streaming_response is None:
                            openMeterLogger.log_stream_event(
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                            )
                        else:
                            if self.stream and complete_streaming_response:
                                self.model_call_details["complete_response"] = (
                                    self.model_call_details.get(
                                        "complete_streaming_response", {}
                                    )
                                )
                                result = self.model_call_details["complete_response"]
                            openMeterLogger.log_success_event(
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                            )

                    if (
                        isinstance(callback, CustomLogger)
                        and self.model_call_details.get("litellm_params", {}).get(
                            "acompletion", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aembedding", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aimage_generation", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "atranscription", False
                        )
                        == False
                    ):  # custom logger class
                        if self.stream and complete_streaming_response is None:
                            callback.log_stream_event(
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                            )
                        else:
                            if self.stream and complete_streaming_response:
                                self.model_call_details["complete_response"] = (
                                    self.model_call_details.get(
                                        "complete_streaming_response", {}
                                    )
                                )
                                result = self.model_call_details["complete_response"]
                            callback.log_success_event(
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                            )
                    if (
                        callable(callback) == True
                        and self.model_call_details.get("litellm_params", {}).get(
                            "acompletion", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aembedding", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aimage_generation", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "atranscription", False
                        )
                        == False
                    ):  # custom logger functions
                        print_verbose(
                            f"success callbacks: Running Custom Callback Function"
                        )
                        customLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                            callback_func=callback,
                        )

                except Exception as e:
                    print_verbose(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success logging with integrations {traceback.format_exc()}"
                    )
                    print_verbose(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except:
            print_verbose(
                f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success logging {traceback.format_exc()}"
            )
            pass

    async def async_success_handler(
        self, result=None, start_time=None, end_time=None, cache_hit=None, **kwargs
    ):
        """
        Implementing async callbacks, to handle asyncio event loop issues when custom integrations need to use async functions.
        """
        print_verbose(f"Logging Details LiteLLM-Async Success Call")
        start_time, end_time, result = self._success_handler_helper_fn(
            start_time=start_time, end_time=end_time, result=result, cache_hit=cache_hit
        )
        ## BUILD COMPLETE STREAMED RESPONSE
        complete_streaming_response = None
        if self.stream:
            if result.choices[0].finish_reason is not None:  # if it's the last chunk
                self.streaming_chunks.append(result)
                # verbose_logger.debug(f"final set of received chunks: {self.streaming_chunks}")
                try:
                    complete_streaming_response = litellm.stream_chunk_builder(
                        self.streaming_chunks,
                        messages=self.model_call_details.get("messages", None),
                        start_time=start_time,
                        end_time=end_time,
                    )
                except Exception as e:
                    print_verbose(
                        f"Error occurred building stream chunk: {traceback.format_exc()}"
                    )
                    complete_streaming_response = None
            else:
                self.streaming_chunks.append(result)
        if complete_streaming_response is not None:
            print_verbose("Async success callbacks: Got a complete streaming response")
            self.model_call_details["async_complete_streaming_response"] = (
                complete_streaming_response
            )
            try:
                if self.model_call_details.get("cache_hit", False) == True:
                    self.model_call_details["response_cost"] = 0.0
                else:
                    # check if base_model set on azure
                    base_model = _get_base_model_from_metadata(
                        model_call_details=self.model_call_details
                    )
                    # base_model defaults to None if not set on model_info
                    self.model_call_details["response_cost"] = litellm.completion_cost(
                        completion_response=complete_streaming_response,
                        model=base_model,
                    )
                verbose_logger.debug(
                    f"Model={self.model}; cost={self.model_call_details['response_cost']}"
                )
            except litellm.NotFoundError as e:
                verbose_logger.debug(
                    f"Model={self.model} not found in completion cost map."
                )
                self.model_call_details["response_cost"] = None

        if self.dynamic_async_success_callbacks is not None and isinstance(
            self.dynamic_async_success_callbacks, list
        ):
            callbacks = self.dynamic_async_success_callbacks
            ## keep the internal functions ##
            for callback in litellm._async_success_callback:
                callback_name = ""
                if isinstance(callback, CustomLogger):
                    callback_name = callback.__class__.__name__
                if callable(callback):
                    callback_name = callback.__name__
                if "_PROXY_" in callback_name:
                    callbacks.append(callback)
        else:
            callbacks = litellm._async_success_callback

        self.redact_message_input_output_from_logging(result=result)

        for callback in callbacks:
            # check if callback can run for this request
            litellm_params = self.model_call_details.get("litellm_params", {})
            if litellm_params.get("no-log", False) == True:
                # proxy cost tracking cal backs should run
                if not (
                    isinstance(callback, CustomLogger)
                    and "_PROXY_" in callback.__class__.__name__
                ):
                    print_verbose("no-log request, skipping logging")
                    continue
            try:
                if kwargs.get("no-log", False) == True:
                    print_verbose("no-log request, skipping logging")
                    continue
                if callback == "cache" and litellm.cache is not None:
                    # set_cache once complete streaming response is built
                    print_verbose("async success_callback: reaches cache for logging!")
                    kwargs = self.model_call_details
                    if self.stream:
                        if "async_complete_streaming_response" not in kwargs:
                            print_verbose(
                                f"async success_callback: reaches cache for logging, there is no async_complete_streaming_response. Kwargs={kwargs}\n\n"
                            )
                            pass
                        else:
                            print_verbose(
                                "async success_callback: reaches cache for logging, there is a async_complete_streaming_response. Adding to cache"
                            )
                            result = kwargs["async_complete_streaming_response"]
                            # only add to cache once we have a complete streaming response
                            if litellm.cache is not None and not isinstance(
                                litellm.cache.cache, S3Cache
                            ):
                                await litellm.cache.async_add_cache(result, **kwargs)
                            else:
                                litellm.cache.add_cache(result, **kwargs)
                if callback == "openmeter":
                    global openMeterLogger
                    if self.stream == True:
                        if (
                            "async_complete_streaming_response"
                            in self.model_call_details
                        ):
                            await openMeterLogger.async_log_success_event(
                                kwargs=self.model_call_details,
                                response_obj=self.model_call_details[
                                    "async_complete_streaming_response"
                                ],
                                start_time=start_time,
                                end_time=end_time,
                            )
                        else:
                            await openMeterLogger.async_log_stream_event(  # [TODO]: move this to being an async log stream event function
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                            )
                    else:
                        await openMeterLogger.async_log_success_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                        )
                if isinstance(callback, CustomLogger):  # custom logger class
                    if self.stream == True:
                        if (
                            "async_complete_streaming_response"
                            in self.model_call_details
                        ):
                            await callback.async_log_success_event(
                                kwargs=self.model_call_details,
                                response_obj=self.model_call_details[
                                    "async_complete_streaming_response"
                                ],
                                start_time=start_time,
                                end_time=end_time,
                            )
                        else:
                            await callback.async_log_stream_event(  # [TODO]: move this to being an async log stream event function
                                kwargs=self.model_call_details,
                                response_obj=result,
                                start_time=start_time,
                                end_time=end_time,
                            )
                    else:
                        await callback.async_log_success_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                        )
                if callable(callback):  # custom logger functions
                    if self.stream:
                        if (
                            "async_complete_streaming_response"
                            in self.model_call_details
                        ):

                            await customLogger.async_log_event(
                                kwargs=self.model_call_details,
                                response_obj=self.model_call_details[
                                    "async_complete_streaming_response"
                                ],
                                start_time=start_time,
                                end_time=end_time,
                                print_verbose=print_verbose,
                                callback_func=callback,
                            )
                    else:
                        await customLogger.async_log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                            callback_func=callback,
                        )
                if callback == "dynamodb":
                    global dynamoLogger
                    if dynamoLogger is None:
                        dynamoLogger = DyanmoDBLogger()
                    if self.stream:
                        if (
                            "async_complete_streaming_response"
                            in self.model_call_details
                        ):
                            print_verbose(
                                "DynamoDB Logger: Got Stream Event - Completed Stream Response"
                            )
                            await dynamoLogger._async_log_event(
                                kwargs=self.model_call_details,
                                response_obj=self.model_call_details[
                                    "async_complete_streaming_response"
                                ],
                                start_time=start_time,
                                end_time=end_time,
                                print_verbose=print_verbose,
                            )
                        else:
                            print_verbose(
                                "DynamoDB Logger: Got Stream Event - No complete stream response as yet"
                            )
                    else:
                        await dynamoLogger._async_log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
            except Exception as e:
                verbose_logger.error(
                    f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success logging {traceback.format_exc()}"
                )
                pass

    def _failure_handler_helper_fn(
        self, exception, traceback_exception, start_time=None, end_time=None
    ):
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = datetime.datetime.now()

        # on some exceptions, model_call_details is not always initialized, this ensures that we still log those exceptions
        if not hasattr(self, "model_call_details"):
            self.model_call_details = {}

        self.model_call_details["log_event_type"] = "failed_api_call"
        self.model_call_details["exception"] = exception
        self.model_call_details["traceback_exception"] = traceback_exception
        self.model_call_details["end_time"] = end_time
        self.model_call_details.setdefault("original_response", None)
        return start_time, end_time

    def failure_handler(
        self, exception, traceback_exception, start_time=None, end_time=None
    ):
        print_verbose(
            f"Logging Details LiteLLM-Failure Call: {litellm.failure_callback}"
        )
        try:
            start_time, end_time = self._failure_handler_helper_fn(
                exception=exception,
                traceback_exception=traceback_exception,
                start_time=start_time,
                end_time=end_time,
            )
            callbacks = []  # init this to empty incase it's not created

            if self.dynamic_failure_callbacks is not None and isinstance(
                self.dynamic_failure_callbacks, list
            ):
                callbacks = self.dynamic_failure_callbacks
                ## keep the internal functions ##
                for callback in litellm.failure_callback:
                    if (
                        isinstance(callback, CustomLogger)
                        and "_PROXY_" in callback.__class__.__name__
                    ):
                        callbacks.append(callback)
            else:
                callbacks = litellm.failure_callback

            result = None  # result sent to all loggers, init this to None incase it's not created

            self.redact_message_input_output_from_logging(result=result)
            for callback in callbacks:
                try:
                    if callback == "lite_debugger":
                        print_verbose("reaches lite_debugger for logging!")
                        print_verbose(f"liteDebuggerClient: {liteDebuggerClient}")
                        result = {
                            "model": self.model,
                            "created": time.time(),
                            "error": traceback_exception,
                            "usage": {
                                "prompt_tokens": prompt_token_calculator(
                                    self.model, messages=self.messages
                                ),
                                "completion_tokens": 0,
                            },
                        }
                        liteDebuggerClient.log_event(
                            model=self.model,
                            messages=self.messages,
                            end_user=self.model_call_details.get("user", "default"),
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            litellm_call_id=self.litellm_call_id,
                            print_verbose=print_verbose,
                            call_type=self.call_type,
                            stream=self.stream,
                        )
                    if callback == "lunary":
                        print_verbose("reaches lunary for logging error!")

                        model = self.model

                        input = self.model_call_details["input"]

                        _type = (
                            "embed"
                            if self.call_type == CallTypes.embedding.value
                            else "llm"
                        )

                        lunaryLogger.log_event(
                            type=_type,
                            event="error",
                            user_id=self.model_call_details.get("user", "default"),
                            model=model,
                            input=input,
                            error=traceback_exception,
                            run_id=self.litellm_call_id,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                        )
                    if callback == "sentry":
                        print_verbose("sending exception to sentry")
                        if capture_exception:
                            capture_exception(exception)
                        else:
                            print_verbose(
                                f"capture exception not initialized: {capture_exception}"
                            )
                    if callable(callback):  # custom logger functions
                        customLogger.log_event(
                            kwargs=self.model_call_details,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            print_verbose=print_verbose,
                            callback_func=callback,
                        )
                    if (
                        isinstance(callback, CustomLogger)
                        and self.model_call_details.get("litellm_params", {}).get(
                            "acompletion", False
                        )
                        == False
                        and self.model_call_details.get("litellm_params", {}).get(
                            "aembedding", False
                        )
                        == False
                    ):  # custom logger class
                        callback.log_failure_event(
                            start_time=start_time,
                            end_time=end_time,
                            response_obj=result,
                            kwargs=self.model_call_details,
                        )
                    if callback == "langfuse":
                        global langFuseLogger
                        verbose_logger.debug("reaches langfuse for logging failure")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        # this only logs streaming once, complete_streaming_response exists i.e when stream ends
                        if langFuseLogger is None or (
                            (
                                self.langfuse_public_key is not None
                                and self.langfuse_public_key
                                != langFuseLogger.public_key
                            )
                            and (
                                self.langfuse_public_key is not None
                                and self.langfuse_public_key
                                != langFuseLogger.public_key
                            )
                        ):
                            langFuseLogger = LangFuseLogger(
                                langfuse_public_key=self.langfuse_public_key,
                                langfuse_secret=self.langfuse_secret,
                            )
                        langFuseLogger.log_event(
                            start_time=start_time,
                            end_time=end_time,
                            response_obj=None,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                            status_message=str(exception),
                            level="ERROR",
                            kwargs=self.model_call_details,
                        )
                    if callback == "traceloop":
                        traceloopLogger.log_event(
                            start_time=start_time,
                            end_time=end_time,
                            response_obj=None,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                            status_message=str(exception),
                            level="ERROR",
                            kwargs=self.model_call_details,
                        )
                    if callback == "prometheus":
                        global prometheusLogger
                        verbose_logger.debug("reaches prometheus for success logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        kwargs["exception"] = str(exception)
                        prometheusLogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            user_id=kwargs.get("user", None),
                            print_verbose=print_verbose,
                        )

                    if callback == "logfire":
                        global logfireLogger
                        verbose_logger.debug("reaches logfire for failure logging!")
                        kwargs = {}
                        for k, v in self.model_call_details.items():
                            if (
                                k != "original_response"
                            ):  # copy.deepcopy raises errors as this could be a coroutine
                                kwargs[k] = v
                        kwargs["exception"] = exception

                        logfireLogger.log_event(
                            kwargs=kwargs,
                            response_obj=result,
                            start_time=start_time,
                            end_time=end_time,
                            level=LogfireLevel.ERROR.value,
                            print_verbose=print_verbose,
                        )
                except Exception as e:
                    print_verbose(
                        f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while failure logging with integrations {str(e)}"
                    )
                    print_verbose(
                        f"LiteLLM.Logging: is sentry capture exception initialized {capture_exception}"
                    )
                    if capture_exception:  # log this error to sentry for debugging
                        capture_exception(e)
        except Exception as e:
            print_verbose(
                f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while failure logging {traceback.format_exc()}"
            )
            pass

    async def async_failure_handler(
        self, exception, traceback_exception, start_time=None, end_time=None
    ):
        """
        Implementing async callbacks, to handle asyncio event loop issues when custom integrations need to use async functions.
        """
        start_time, end_time = self._failure_handler_helper_fn(
            exception=exception,
            traceback_exception=traceback_exception,
            start_time=start_time,
            end_time=end_time,
        )
        result = None  # result sent to all loggers, init this to None incase it's not created
        for callback in litellm._async_failure_callback:
            try:
                if isinstance(callback, CustomLogger):  # custom logger class
                    await callback.async_log_failure_event(
                        kwargs=self.model_call_details,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                    )  # type: ignore
                if callable(callback):  # custom logger functions
                    await customLogger.async_log_event(
                        kwargs=self.model_call_details,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        print_verbose=print_verbose,
                        callback_func=callback,
                    )
            except Exception as e:
                print_verbose(
                    f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while success logging {traceback.format_exc()}"
                )

    def redact_message_input_output_from_logging(self, result):
        """
        Removes messages, prompts, input, response from logging. This modifies the data in-place
        only redacts when litellm.turn_off_message_logging == True
        """
        # check if user opted out of logging message/response to callbacks
        if litellm.turn_off_message_logging == True:
            # remove messages, prompts, input, response from logging
            self.model_call_details["messages"] = [
                {"role": "user", "content": "redacted-by-litellm"}
            ]
            self.model_call_details["prompt"] = ""
            self.model_call_details["input"] = ""

            # response cleaning
            # ChatCompletion Responses
            if self.stream and "complete_streaming_response" in self.model_call_details:
                _streaming_response = self.model_call_details[
                    "complete_streaming_response"
                ]
                for choice in _streaming_response.choices:
                    if isinstance(choice, litellm.Choices):
                        choice.message.content = "redacted-by-litellm"
                    elif isinstance(choice, litellm.utils.StreamingChoices):
                        choice.delta.content = "redacted-by-litellm"
            else:
                if result is not None:
                    if isinstance(result, litellm.ModelResponse):
                        if hasattr(result, "choices") and result.choices is not None:
                            for choice in result.choices:
                                if isinstance(choice, litellm.Choices):
                                    choice.message.content = "redacted-by-litellm"
                                elif isinstance(choice, litellm.utils.StreamingChoices):
                                    choice.delta.content = "redacted-by-litellm"


def exception_logging(
    additional_args={},
    logger_fn=None,
    exception=None,
):
    try:
        model_call_details = {}
        if exception:
            model_call_details["exception"] = exception
        model_call_details["additional_args"] = additional_args
        # User Logging -> if you pass in a custom logging function or want to use sentry breadcrumbs
        print_verbose(
            f"Logging Details: logger_fn - {logger_fn} | callable(logger_fn) - {callable(logger_fn)}"
        )
        if logger_fn and callable(logger_fn):
            try:
                logger_fn(
                    model_call_details
                )  # Expectation: any logger function passed in by the user should accept a dict object
            except Exception as e:
                print_verbose(
                    f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
                )
    except Exception as e:
        print_verbose(
            f"LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging {traceback.format_exc()}"
        )
        pass


####### RULES ###################


class Rules:
    """
    Fail calls based on the input or llm api output

    Example usage:
    import litellm
    def my_custom_rule(input): # receives the model response
            if "i don't think i can answer" in input: # trigger fallback if the model refuses to answer
                    return False
            return True

    litellm.post_call_rules = [my_custom_rule] # have these be functions that can be called to fail a call

    response = litellm.completion(model="gpt-3.5-turbo", messages=[{"role": "user",
        "content": "Hey, how's it going?"}], fallbacks=["openrouter/mythomax"])
    """

    def __init__(self) -> None:
        pass

    def pre_call_rules(self, input: str, model: str):
        for rule in litellm.pre_call_rules:
            if callable(rule):
                decision = rule(input)
                if decision is False:
                    raise litellm.APIResponseValidationError(message="LLM Response failed post-call-rule check", llm_provider="", model=model)  # type: ignore
        return True

    def post_call_rules(self, input: str, model: str):
        for rule in litellm.post_call_rules:
            if callable(rule):
                decision = rule(input)
                if type(decision) == bool:
                    if decision is False:
                        raise litellm.APIResponseValidationError(message="LLM Response failed post-call-rule check", llm_provider="", model=model)  # type: ignore
                elif type(decision) == dict:
                    decision_val = decision.get("decision", True)
                    decision_message = decision.get(
                        "message", "LLM Response failed post-call-rule check"
                    )
                    if decision_val is False:
                        raise litellm.APIResponseValidationError(message=decision_message, llm_provider="", model=model)  # type: ignore
        return True


def _init_custom_logger_compatible_class(
    logging_integration: litellm._custom_logger_compatible_callbacks_literal,
) -> Callable:
    if logging_integration == "lago":
        return LagoLogger()  # type: ignore
    elif logging_integration == "openmeter":
        return OpenMeterLogger()  # type: ignore


####### CLIENT ###################
# make it easy to log if completion/embedding runs succeeded or failed + see what happened | Non-Blocking
def function_setup(
    original_function: str, rules_obj, start_time, *args, **kwargs
):  # just run once to check if user wants to send their data anywhere - PostHog/Sentry/Slack/etc.
    try:
        global callback_list, add_breadcrumb, user_logger_fn, Logging
        function_id = kwargs["id"] if "id" in kwargs else None
        if len(litellm.callbacks) > 0:
            for callback in litellm.callbacks:
                # check if callback is a string - e.g. "lago", "openmeter"
                if isinstance(callback, str):
                    callback = _init_custom_logger_compatible_class(callback)
                    if any(
                        isinstance(cb, type(callback))
                        for cb in litellm._async_success_callback
                    ):  # don't double add a callback
                        continue
                if callback not in litellm.input_callback:
                    litellm.input_callback.append(callback)  # type: ignore
                if callback not in litellm.success_callback:
                    litellm.success_callback.append(callback)  # type: ignore
                if callback not in litellm.failure_callback:
                    litellm.failure_callback.append(callback)  # type: ignore
                if callback not in litellm._async_success_callback:
                    litellm._async_success_callback.append(callback)  # type: ignore
                if callback not in litellm._async_failure_callback:
                    litellm._async_failure_callback.append(callback)  # type: ignore
            print_verbose(
                f"Initialized litellm callbacks, Async Success Callbacks: {litellm._async_success_callback}"
            )
        if (
            len(litellm.input_callback) > 0
            or len(litellm.success_callback) > 0
            or len(litellm.failure_callback) > 0
        ) and len(
            callback_list  # type: ignore
        ) == 0:  # type: ignore
            callback_list = list(
                set(
                    litellm.input_callback  # type: ignore
                    + litellm.success_callback
                    + litellm.failure_callback
                )
            )
            set_callbacks(callback_list=callback_list, function_id=function_id)
        ## ASYNC CALLBACKS
        if len(litellm.input_callback) > 0:
            removed_async_items = []
            for index, callback in enumerate(litellm.input_callback):  # type: ignore
                if inspect.iscoroutinefunction(callback):
                    litellm._async_input_callback.append(callback)
                    removed_async_items.append(index)

            # Pop the async items from input_callback in reverse order to avoid index issues
            for index in reversed(removed_async_items):
                litellm.input_callback.pop(index)

        if len(litellm.success_callback) > 0:
            removed_async_items = []
            for index, callback in enumerate(litellm.success_callback):  # type: ignore
                if inspect.iscoroutinefunction(callback):
                    litellm._async_success_callback.append(callback)
                    removed_async_items.append(index)
                elif callback == "dynamodb" or callback == "openmeter":
                    # dynamo is an async callback, it's used for the proxy and needs to be async
                    # we only support async dynamo db logging for acompletion/aembedding since that's used on proxy
                    litellm._async_success_callback.append(callback)
                    removed_async_items.append(index)

            # Pop the async items from success_callback in reverse order to avoid index issues
            for index in reversed(removed_async_items):
                litellm.success_callback.pop(index)

        if len(litellm.failure_callback) > 0:
            removed_async_items = []
            for index, callback in enumerate(litellm.failure_callback):  # type: ignore
                if inspect.iscoroutinefunction(callback):
                    litellm._async_failure_callback.append(callback)
                    removed_async_items.append(index)

            # Pop the async items from failure_callback in reverse order to avoid index issues
            for index in reversed(removed_async_items):
                litellm.failure_callback.pop(index)
        ### DYNAMIC CALLBACKS ###
        dynamic_success_callbacks = None
        dynamic_async_success_callbacks = None
        dynamic_failure_callbacks = None
        if kwargs.get("success_callback", None) is not None and isinstance(
            kwargs["success_callback"], list
        ):
            removed_async_items = []
            for index, callback in enumerate(kwargs["success_callback"]):
                if (
                    inspect.iscoroutinefunction(callback)
                    or callback == "dynamodb"
                    or callback == "s3"
                ):
                    if dynamic_async_success_callbacks is not None and isinstance(
                        dynamic_async_success_callbacks, list
                    ):
                        dynamic_async_success_callbacks.append(callback)
                    else:
                        dynamic_async_success_callbacks = [callback]
                    removed_async_items.append(index)
            # Pop the async items from success_callback in reverse order to avoid index issues
            for index in reversed(removed_async_items):
                kwargs["success_callback"].pop(index)
            dynamic_success_callbacks = kwargs.pop("success_callback")
        if kwargs.get("failure_callback", None) is not None and isinstance(
            kwargs["failure_callback"], list
        ):
            dynamic_failure_callbacks = kwargs.pop("failure_callback")

        if add_breadcrumb:
            try:
                details_to_log = copy.deepcopy(kwargs)
            except:
                details_to_log = kwargs

            if litellm.turn_off_message_logging:
                # make a copy of the _model_Call_details and log it
                details_to_log.pop("messages", None)
                details_to_log.pop("input", None)
                details_to_log.pop("prompt", None)
            add_breadcrumb(
                category="litellm.llm_call",
                message=f"Positional Args: {args}, Keyword Args: {details_to_log}",
                level="info",
            )
        if "logger_fn" in kwargs:
            user_logger_fn = kwargs["logger_fn"]
        # INIT LOGGER - for user-specified integrations
        model = args[0] if len(args) > 0 else kwargs.get("model", None)
        call_type = original_function
        if (
            call_type == CallTypes.completion.value
            or call_type == CallTypes.acompletion.value
        ):
            messages = None
            if len(args) > 1:
                messages = args[1]
            elif kwargs.get("messages", None):
                messages = kwargs["messages"]
            ### PRE-CALL RULES ###
            if (
                isinstance(messages, list)
                and len(messages) > 0
                and isinstance(messages[0], dict)
                and "content" in messages[0]
            ):
                rules_obj.pre_call_rules(
                    input="".join(
                        m.get("content", "")
                        for m in messages
                        if "content" in m and isinstance(m["content"], str)
                    ),
                    model=model,
                )
        elif (
            call_type == CallTypes.embedding.value
            or call_type == CallTypes.aembedding.value
        ):
            messages = args[1] if len(args) > 1 else kwargs["input"]
        elif (
            call_type == CallTypes.image_generation.value
            or call_type == CallTypes.aimage_generation.value
        ):
            messages = args[0] if len(args) > 0 else kwargs["prompt"]
        elif (
            call_type == CallTypes.moderation.value
            or call_type == CallTypes.amoderation.value
        ):
            messages = args[1] if len(args) > 1 else kwargs["input"]
        elif (
            call_type == CallTypes.atext_completion.value
            or call_type == CallTypes.text_completion.value
        ):
            messages = args[0] if len(args) > 0 else kwargs["prompt"]
        elif (
            call_type == CallTypes.atranscription.value
            or call_type == CallTypes.transcription.value
        ):
            _file_name: BinaryIO = args[1] if len(args) > 1 else kwargs["file"]
            messages = "audio_file"
        elif (
            call_type == CallTypes.aspeech.value or call_type == CallTypes.speech.value
        ):
            messages = kwargs.get("input", "speech")
        stream = True if "stream" in kwargs and kwargs["stream"] == True else False
        logging_obj = Logging(
            model=model,
            messages=messages,
            stream=stream,
            litellm_call_id=kwargs["litellm_call_id"],
            function_id=function_id,
            call_type=call_type,
            start_time=start_time,
            dynamic_success_callbacks=dynamic_success_callbacks,
            dynamic_failure_callbacks=dynamic_failure_callbacks,
            dynamic_async_success_callbacks=dynamic_async_success_callbacks,
            langfuse_public_key=kwargs.pop("langfuse_public_key", None),
            langfuse_secret=kwargs.pop("langfuse_secret", None)
            or kwargs.pop("langfuse_secret_key", None),
        )
        ## check if metadata is passed in
        litellm_params = {"api_base": ""}
        if "metadata" in kwargs:
            litellm_params["metadata"] = kwargs["metadata"]
        logging_obj.update_environment_variables(
            model=model,
            user="",
            optional_params={},
            litellm_params=litellm_params,
            stream_options=kwargs.get("stream_options", None),
        )
        return logging_obj, kwargs
    except Exception as e:
        import logging

        logging.debug(
            f"[Non-Blocking] {traceback.format_exc()}; args - {args}; kwargs - {kwargs}"
        )
        raise e


def client(original_function):
    global liteDebuggerClient, get_all_keys
    rules_obj = Rules()

    def check_coroutine(value) -> bool:
        if inspect.iscoroutine(value):
            return True
        elif inspect.iscoroutinefunction(value):
            return True
        else:
            return False

    def post_call_processing(original_response, model):
        try:
            if original_response is None:
                pass
            else:
                call_type = original_function.__name__
                if (
                    call_type == CallTypes.completion.value
                    or call_type == CallTypes.acompletion.value
                ):
                    is_coroutine = check_coroutine(original_function)
                    if is_coroutine == True:
                        pass
                    else:
                        if isinstance(original_response, ModelResponse):
                            model_response = original_response["choices"][0]["message"][
                                "content"
                            ]
                            ### POST-CALL RULES ###
                            rules_obj.post_call_rules(input=model_response, model=model)
        except Exception as e:
            raise e

    @wraps(original_function)
    def wrapper(*args, **kwargs):
        # DO NOT MOVE THIS. It always needs to run first
        # Check if this is an async function. If so only execute the async function
        if (
            kwargs.get("acompletion", False) == True
            or kwargs.get("aembedding", False) == True
            or kwargs.get("aimg_generation", False) == True
            or kwargs.get("amoderation", False) == True
            or kwargs.get("atext_completion", False) == True
            or kwargs.get("atranscription", False) == True
        ):
            # [OPTIONAL] CHECK MAX RETRIES / REQUEST
            if litellm.num_retries_per_request is not None:
                # check if previous_models passed in as ['litellm_params']['metadata]['previous_models']
                previous_models = kwargs.get("metadata", {}).get(
                    "previous_models", None
                )
                if previous_models is not None:
                    if litellm.num_retries_per_request <= len(previous_models):
                        raise Exception(f"Max retries per request hit!")

            # MODEL CALL
            result = original_function(*args, **kwargs)
            if "stream" in kwargs and kwargs["stream"] == True:
                if (
                    "complete_response" in kwargs
                    and kwargs["complete_response"] == True
                ):
                    chunks = []
                    for idx, chunk in enumerate(result):
                        chunks.append(chunk)
                    return litellm.stream_chunk_builder(
                        chunks, messages=kwargs.get("messages", None)
                    )
                else:
                    return result

            return result

        # Prints Exactly what was passed to litellm function - don't execute any logic here - it should just print
        print_args_passed_to_litellm(original_function, args, kwargs)
        start_time = datetime.datetime.now()
        result = None
        logging_obj = kwargs.get("litellm_logging_obj", None)

        # only set litellm_call_id if its not in kwargs
        call_type = original_function.__name__
        if "litellm_call_id" not in kwargs:
            kwargs["litellm_call_id"] = str(uuid.uuid4())
        try:
            model = args[0] if len(args) > 0 else kwargs["model"]
        except:
            model = None
            if (
                call_type != CallTypes.image_generation.value
                and call_type != CallTypes.text_completion.value
            ):
                raise ValueError("model param not passed in.")

        try:
            if logging_obj is None:
                logging_obj, kwargs = function_setup(
                    original_function.__name__, rules_obj, start_time, *args, **kwargs
                )
            kwargs["litellm_logging_obj"] = logging_obj

            # CHECK FOR 'os.environ/' in kwargs
            for k, v in kwargs.items():
                if v is not None and isinstance(v, str) and v.startswith("os.environ/"):
                    kwargs[k] = litellm.get_secret(v)
            # [OPTIONAL] CHECK BUDGET
            if litellm.max_budget:
                if litellm._current_cost > litellm.max_budget:
                    raise BudgetExceededError(
                        current_cost=litellm._current_cost,
                        max_budget=litellm.max_budget,
                    )

            # [OPTIONAL] CHECK MAX RETRIES / REQUEST
            if litellm.num_retries_per_request is not None:
                # check if previous_models passed in as ['litellm_params']['metadata]['previous_models']
                previous_models = kwargs.get("metadata", {}).get(
                    "previous_models", None
                )
                if previous_models is not None:
                    if litellm.num_retries_per_request <= len(previous_models):
                        raise Exception(f"Max retries per request hit!")

            # [OPTIONAL] CHECK CACHE
            print_verbose(
                f"SYNC kwargs[caching]: {kwargs.get('caching', False)}; litellm.cache: {litellm.cache}; kwargs.get('cache')['no-cache']: {kwargs.get('cache', {}).get('no-cache', False)}"
            )
            # if caching is false or cache["no-cache"]==True, don't run this
            if (
                (
                    (
                        (
                            kwargs.get("caching", None) is None
                            and litellm.cache is not None
                        )
                        or kwargs.get("caching", False) == True
                    )
                    and kwargs.get("cache", {}).get("no-cache", False) != True
                )
                and kwargs.get("aembedding", False) != True
                and kwargs.get("atext_completion", False) != True
                and kwargs.get("acompletion", False) != True
                and kwargs.get("aimg_generation", False) != True
                and kwargs.get("atranscription", False) != True
            ):  # allow users to control returning cached responses from the completion function
                # checking cache
                print_verbose(f"INSIDE CHECKING CACHE")
                if (
                    litellm.cache is not None
                    and str(original_function.__name__)
                    in litellm.cache.supported_call_types
                ):
                    print_verbose(f"Checking Cache")
                    preset_cache_key = litellm.cache.get_cache_key(*args, **kwargs)
                    kwargs["preset_cache_key"] = (
                        preset_cache_key  # for streaming calls, we need to pass the preset_cache_key
                    )
                    cached_result = litellm.cache.get_cache(*args, **kwargs)
                    if cached_result != None:
                        if "detail" in cached_result:
                            # implies an error occurred
                            pass
                        else:
                            call_type = original_function.__name__
                            print_verbose(
                                f"Cache Response Object routing: call_type - {call_type}; cached_result instace: {type(cached_result)}"
                            )
                            if call_type == CallTypes.completion.value and isinstance(
                                cached_result, dict
                            ):
                                cached_result = convert_to_model_response_object(
                                    response_object=cached_result,
                                    model_response_object=ModelResponse(),
                                    stream=kwargs.get("stream", False),
                                )

                                if kwargs.get("stream", False) == True:
                                    cached_result = CustomStreamWrapper(
                                        completion_stream=cached_result,
                                        model=model,
                                        custom_llm_provider="cached_response",
                                        logging_obj=logging_obj,
                                    )
                            elif call_type == CallTypes.embedding.value and isinstance(
                                cached_result, dict
                            ):
                                cached_result = convert_to_model_response_object(
                                    response_object=cached_result,
                                    response_type="embedding",
                                )

                            # LOG SUCCESS
                            cache_hit = True
                            end_time = datetime.datetime.now()
                            (
                                model,
                                custom_llm_provider,
                                dynamic_api_key,
                                api_base,
                            ) = litellm.get_llm_provider(
                                model=model,
                                custom_llm_provider=kwargs.get(
                                    "custom_llm_provider", None
                                ),
                                api_base=kwargs.get("api_base", None),
                                api_key=kwargs.get("api_key", None),
                            )
                            print_verbose(
                                f"Async Wrapper: Completed Call, calling async_success_handler: {logging_obj.async_success_handler}"
                            )
                            logging_obj.update_environment_variables(
                                model=model,
                                user=kwargs.get("user", None),
                                optional_params={},
                                litellm_params={
                                    "logger_fn": kwargs.get("logger_fn", None),
                                    "acompletion": False,
                                    "metadata": kwargs.get("metadata", {}),
                                    "model_info": kwargs.get("model_info", {}),
                                    "proxy_server_request": kwargs.get(
                                        "proxy_server_request", None
                                    ),
                                    "preset_cache_key": kwargs.get(
                                        "preset_cache_key", None
                                    ),
                                    "stream_response": kwargs.get(
                                        "stream_response", {}
                                    ),
                                },
                                input=kwargs.get("messages", ""),
                                api_key=kwargs.get("api_key", None),
                                original_response=str(cached_result),
                                additional_args=None,
                                stream=kwargs.get("stream", False),
                            )
                            threading.Thread(
                                target=logging_obj.success_handler,
                                args=(cached_result, start_time, end_time, cache_hit),
                            ).start()
                            return cached_result

            # CHECK MAX TOKENS
            if (
                kwargs.get("max_tokens", None) is not None
                and model is not None
                and litellm.modify_params
                == True  # user is okay with params being modified
                and (
                    call_type == CallTypes.acompletion.value
                    or call_type == CallTypes.completion.value
                )
            ):
                try:
                    base_model = model
                    if kwargs.get("hf_model_name", None) is not None:
                        base_model = f"huggingface/{kwargs.get('hf_model_name')}"
                    max_output_tokens = (
                        get_max_tokens(model=base_model) or 4096
                    )  # assume min context window is 4k tokens
                    user_max_tokens = kwargs.get("max_tokens")
                    ## Scenario 1: User limit + prompt > model limit
                    messages = None
                    if len(args) > 1:
                        messages = args[1]
                    elif kwargs.get("messages", None):
                        messages = kwargs["messages"]
                    input_tokens = token_counter(model=base_model, messages=messages)
                    input_tokens += max(
                        0.1 * input_tokens, 10
                    )  # give at least a 10 token buffer. token counting can be imprecise.
                    if input_tokens > max_output_tokens:
                        pass  # allow call to fail normally
                    elif user_max_tokens + input_tokens > max_output_tokens:
                        user_max_tokens = max_output_tokens - input_tokens
                    print_verbose(f"user_max_tokens: {user_max_tokens}")
                    kwargs["max_tokens"] = int(
                        round(user_max_tokens)
                    )  # make sure max tokens is always an int
                except Exception as e:
                    print_verbose(f"Error while checking max token limit: {str(e)}")
            # MODEL CALL
            result = original_function(*args, **kwargs)
            end_time = datetime.datetime.now()
            if "stream" in kwargs and kwargs["stream"] == True:
                if (
                    "complete_response" in kwargs
                    and kwargs["complete_response"] == True
                ):
                    chunks = []
                    for idx, chunk in enumerate(result):
                        chunks.append(chunk)
                    return litellm.stream_chunk_builder(
                        chunks, messages=kwargs.get("messages", None)
                    )
                else:
                    return result
            elif "acompletion" in kwargs and kwargs["acompletion"] == True:
                return result
            elif "aembedding" in kwargs and kwargs["aembedding"] == True:
                return result
            elif "aimg_generation" in kwargs and kwargs["aimg_generation"] == True:
                return result
            elif "atranscription" in kwargs and kwargs["atranscription"] == True:
                return result
            elif "aspeech" in kwargs and kwargs["aspeech"] == True:
                return result

            ### POST-CALL RULES ###
            post_call_processing(original_response=result, model=model or None)

            # [OPTIONAL] ADD TO CACHE
            if (
                litellm.cache is not None
                and str(original_function.__name__)
                in litellm.cache.supported_call_types
            ) and (kwargs.get("cache", {}).get("no-store", False) != True):
                litellm.cache.add_cache(result, *args, **kwargs)

            # LOG SUCCESS - handle streaming success logging in the _next_ object, remove `handle_success` once it's deprecated
            verbose_logger.info(f"Wrapper: Completed Call, calling success_handler")
            threading.Thread(
                target=logging_obj.success_handler, args=(result, start_time, end_time)
            ).start()
            # RETURN RESULT
            if hasattr(result, "_hidden_params"):
                result._hidden_params["model_id"] = kwargs.get("model_info", {}).get(
                    "id", None
                )
                result._hidden_params["api_base"] = get_api_base(
                    model=model,
                    optional_params=getattr(logging_obj, "optional_params", {}),
                )
            result._response_ms = (
                end_time - start_time
            ).total_seconds() * 1000  # return response latency in ms like openai
            return result
        except Exception as e:
            call_type = original_function.__name__
            if call_type == CallTypes.completion.value:
                num_retries = (
                    kwargs.get("num_retries", None) or litellm.num_retries or None
                )
                litellm.num_retries = (
                    None  # set retries to None to prevent infinite loops
                )
                context_window_fallback_dict = kwargs.get(
                    "context_window_fallback_dict", {}
                )

                _is_litellm_router_call = "model_group" in kwargs.get(
                    "metadata", {}
                )  # check if call from litellm.router/proxy
                if (
                    num_retries and not _is_litellm_router_call
                ):  # only enter this if call is not from litellm router/proxy. router has it's own logic for retrying
                    if (
                        isinstance(e, openai.APIError)
                        or isinstance(e, openai.Timeout)
                        or isinstance(e, openai.APIConnectionError)
                    ):
                        kwargs["num_retries"] = num_retries
                        return litellm.completion_with_retries(*args, **kwargs)
                elif (
                    isinstance(e, litellm.exceptions.ContextWindowExceededError)
                    and context_window_fallback_dict
                    and model in context_window_fallback_dict
                ):
                    if len(args) > 0:
                        args[0] = context_window_fallback_dict[model]
                    else:
                        kwargs["model"] = context_window_fallback_dict[model]
                    return original_function(*args, **kwargs)
            traceback_exception = traceback.format_exc()
            end_time = datetime.datetime.now()

            # LOG FAILURE - handle streaming failure logging in the _next_ object, remove `handle_failure` once it's deprecated
            if logging_obj:
                logging_obj.failure_handler(
                    e, traceback_exception, start_time, end_time
                )  # DO NOT MAKE THREADED - router retry fallback relies on this!
                my_thread = threading.Thread(
                    target=handle_failure,
                    args=(e, traceback_exception, start_time, end_time, args, kwargs),
                )  # don't interrupt execution of main thread
                my_thread.start()
                if hasattr(e, "message"):
                    if (
                        liteDebuggerClient and liteDebuggerClient.dashboard_url != None
                    ):  # make it easy to get to the debugger logs if you've initialized it
                        e.message += f"\n Check the log in your dashboard - {liteDebuggerClient.dashboard_url}"
            raise e

    @wraps(original_function)
    async def wrapper_async(*args, **kwargs):
        print_args_passed_to_litellm(original_function, args, kwargs)
        start_time = datetime.datetime.now()
        result = None
        logging_obj = kwargs.get("litellm_logging_obj", None)
        # only set litellm_call_id if its not in kwargs
        call_type = original_function.__name__
        if "litellm_call_id" not in kwargs:
            kwargs["litellm_call_id"] = str(uuid.uuid4())

        model = ""
        try:
            model = args[0] if len(args) > 0 else kwargs["model"]
        except:
            if (
                call_type != CallTypes.aimage_generation.value  # model optional
                and call_type != CallTypes.atext_completion.value  # can also be engine
            ):
                raise ValueError("model param not passed in.")

        try:
            if logging_obj is None:
                logging_obj, kwargs = function_setup(
                    original_function.__name__, rules_obj, start_time, *args, **kwargs
                )
            kwargs["litellm_logging_obj"] = logging_obj

            # [OPTIONAL] CHECK BUDGET
            if litellm.max_budget:
                if litellm._current_cost > litellm.max_budget:
                    raise BudgetExceededError(
                        current_cost=litellm._current_cost,
                        max_budget=litellm.max_budget,
                    )

            # [OPTIONAL] CHECK CACHE
            print_verbose(
                f"ASYNC kwargs[caching]: {kwargs.get('caching', False)}; litellm.cache: {litellm.cache}; kwargs.get('cache'): {kwargs.get('cache', None)}"
            )
            # if caching is false, don't run this
            final_embedding_cached_response = None

            if (
                (kwargs.get("caching", None) is None and litellm.cache is not None)
                or kwargs.get("caching", False) == True
            ) and (
                kwargs.get("cache", {}).get("no-cache", False) != True
            ):  # allow users to control returning cached responses from the completion function
                # checking cache
                print_verbose("INSIDE CHECKING CACHE")
                if (
                    litellm.cache is not None
                    and str(original_function.__name__)
                    in litellm.cache.supported_call_types
                ):
                    print_verbose(f"Checking Cache")
                    if call_type == CallTypes.aembedding.value and isinstance(
                        kwargs["input"], list
                    ):
                        tasks = []
                        for idx, i in enumerate(kwargs["input"]):
                            preset_cache_key = litellm.cache.get_cache_key(
                                *args, **{**kwargs, "input": i}
                            )
                            tasks.append(
                                litellm.cache.async_get_cache(
                                    cache_key=preset_cache_key
                                )
                            )
                        cached_result = await asyncio.gather(*tasks)
                        ## check if cached result is None ##
                        if cached_result is not None and isinstance(
                            cached_result, list
                        ):
                            if len(cached_result) == 1 and cached_result[0] is None:
                                cached_result = None
                    elif (
                        isinstance(litellm.cache.cache, RedisSemanticCache)
                        or isinstance(litellm.cache.cache, RedisCache)
                        or isinstance(litellm.cache.cache, DualSemanticCache)
                    ):
                        preset_cache_key = litellm.cache.get_cache_key(*args, **kwargs)
                        kwargs["preset_cache_key"] = (
                            preset_cache_key  # for streaming calls, we need to pass the preset_cache_key
                        )
                        cached_result = await litellm.cache.async_get_cache(
                            *args, **kwargs
                        )
                    else:  # for s3 caching. [NOT RECOMMENDED IN PROD - this will slow down responses since boto3 is sync]
                        preset_cache_key = litellm.cache.get_cache_key(*args, **kwargs)
                        kwargs["preset_cache_key"] = (
                            preset_cache_key  # for streaming calls, we need to pass the preset_cache_key
                        )
                        cached_result = litellm.cache.get_cache(*args, **kwargs)
                    if cached_result is not None and not isinstance(
                        cached_result, list
                    ):
                        print_verbose(f"Cache Hit!")
                        cache_hit = True
                        end_time = datetime.datetime.now()
                        (
                            model,
                            custom_llm_provider,
                            dynamic_api_key,
                            api_base,
                        ) = litellm.get_llm_provider(
                            model=model,
                            custom_llm_provider=kwargs.get("custom_llm_provider", None),
                            api_base=kwargs.get("api_base", None),
                            api_key=kwargs.get("api_key", None),
                        )
                        print_verbose(
                            f"Async Wrapper: Completed Call, calling async_success_handler: {logging_obj.async_success_handler}"
                        )
                        logging_obj.update_environment_variables(
                            model=model,
                            user=kwargs.get("user", None),
                            optional_params={},
                            litellm_params={
                                "logger_fn": kwargs.get("logger_fn", None),
                                "acompletion": True,
                                "metadata": kwargs.get("metadata", {}),
                                "model_info": kwargs.get("model_info", {}),
                                "proxy_server_request": kwargs.get(
                                    "proxy_server_request", None
                                ),
                                "preset_cache_key": kwargs.get(
                                    "preset_cache_key", None
                                ),
                                "stream_response": kwargs.get("stream_response", {}),
                                "api_base": kwargs.get("api_base", ""),
                            },
                            input=kwargs.get("messages", ""),
                            api_key=kwargs.get("api_key", None),
                            original_response=str(cached_result),
                            additional_args=None,
                            stream=kwargs.get("stream", False),
                        )
                        call_type = original_function.__name__
                        if call_type == CallTypes.acompletion.value and isinstance(
                            cached_result, dict
                        ):
                            if kwargs.get("stream", False) == True:
                                cached_result = convert_to_streaming_response_async(
                                    response_object=cached_result,
                                )
                                cached_result = CustomStreamWrapper(
                                    completion_stream=cached_result,
                                    model=model,
                                    custom_llm_provider="cached_response",
                                    logging_obj=logging_obj,
                                )
                            else:
                                cached_result = convert_to_model_response_object(
                                    response_object=cached_result,
                                    model_response_object=ModelResponse(),
                                )
                        if (
                            call_type == CallTypes.atext_completion.value
                            and isinstance(cached_result, dict)
                        ):
                            if kwargs.get("stream", False) == True:
                                cached_result = convert_to_streaming_response_async(
                                    response_object=cached_result,
                                )
                                cached_result = CustomStreamWrapper(
                                    completion_stream=cached_result,
                                    model=model,
                                    custom_llm_provider="cached_response",
                                    logging_obj=logging_obj,
                                )
                            else:
                                cached_result = TextCompletionResponse(**cached_result)
                        elif call_type == CallTypes.aembedding.value and isinstance(
                            cached_result, dict
                        ):
                            cached_result = convert_to_model_response_object(
                                response_object=cached_result,
                                model_response_object=EmbeddingResponse(),
                                response_type="embedding",
                            )
                        elif call_type == CallTypes.atranscription.value and isinstance(
                            cached_result, dict
                        ):
                            hidden_params = {
                                "model": "whisper-1",
                                "custom_llm_provider": custom_llm_provider,
                            }
                            cached_result = convert_to_model_response_object(
                                response_object=cached_result,
                                model_response_object=TranscriptionResponse(),
                                response_type="audio_transcription",
                                hidden_params=hidden_params,
                            )
                        if kwargs.get("stream", False) == False:
                            # LOG SUCCESS
                            asyncio.create_task(
                                logging_obj.async_success_handler(
                                    cached_result, start_time, end_time, cache_hit
                                )
                            )
                            threading.Thread(
                                target=logging_obj.success_handler,
                                args=(cached_result, start_time, end_time, cache_hit),
                            ).start()
                        cache_key = kwargs.get("preset_cache_key", None)
                        cached_result._hidden_params["cache_key"] = cache_key
                        return cached_result
                    elif (
                        call_type == CallTypes.aembedding.value
                        and cached_result is not None
                        and isinstance(cached_result, list)
                        and litellm.cache is not None
                        and not isinstance(
                            litellm.cache.cache, S3Cache
                        )  # s3 doesn't support bulk writing. Exclude.
                    ):
                        remaining_list = []
                        non_null_list = []
                        for idx, cr in enumerate(cached_result):
                            if cr is None:
                                remaining_list.append(kwargs["input"][idx])
                            else:
                                non_null_list.append((idx, cr))
                        original_kwargs_input = kwargs["input"]
                        kwargs["input"] = remaining_list
                        if len(non_null_list) > 0:
                            print_verbose(
                                f"EMBEDDING CACHE HIT! - {len(non_null_list)}"
                            )
                            final_embedding_cached_response = EmbeddingResponse(
                                model=kwargs.get("model"),
                                data=[None] * len(original_kwargs_input),
                            )
                            final_embedding_cached_response._hidden_params[
                                "cache_hit"
                            ] = True

                            for val in non_null_list:
                                idx, cr = val  # (idx, cr) tuple
                                if cr is not None:
                                    final_embedding_cached_response.data[idx] = (
                                        Embedding(
                                            embedding=cr["embedding"],
                                            index=idx,
                                            object="embedding",
                                        )
                                    )
                        if len(remaining_list) == 0:
                            # LOG SUCCESS
                            cache_hit = True
                            end_time = datetime.datetime.now()
                            (
                                model,
                                custom_llm_provider,
                                dynamic_api_key,
                                api_base,
                            ) = litellm.get_llm_provider(
                                model=model,
                                custom_llm_provider=kwargs.get(
                                    "custom_llm_provider", None
                                ),
                                api_base=kwargs.get("api_base", None),
                                api_key=kwargs.get("api_key", None),
                            )
                            print_verbose(
                                f"Async Wrapper: Completed Call, calling async_success_handler: {logging_obj.async_success_handler}"
                            )
                            logging_obj.update_environment_variables(
                                model=model,
                                user=kwargs.get("user", None),
                                optional_params={},
                                litellm_params={
                                    "logger_fn": kwargs.get("logger_fn", None),
                                    "acompletion": True,
                                    "metadata": kwargs.get("metadata", {}),
                                    "model_info": kwargs.get("model_info", {}),
                                    "proxy_server_request": kwargs.get(
                                        "proxy_server_request", None
                                    ),
                                    "preset_cache_key": kwargs.get(
                                        "preset_cache_key", None
                                    ),
                                    "stream_response": kwargs.get(
                                        "stream_response", {}
                                    ),
                                    "api_base": "",
                                },
                                input=kwargs.get("messages", ""),
                                api_key=kwargs.get("api_key", None),
                                original_response=str(final_embedding_cached_response),
                                additional_args=None,
                                stream=kwargs.get("stream", False),
                            )
                            asyncio.create_task(
                                logging_obj.async_success_handler(
                                    final_embedding_cached_response,
                                    start_time,
                                    end_time,
                                    cache_hit,
                                )
                            )
                            threading.Thread(
                                target=logging_obj.success_handler,
                                args=(
                                    final_embedding_cached_response,
                                    start_time,
                                    end_time,
                                    cache_hit,
                                ),
                            ).start()
                            return final_embedding_cached_response
            # MODEL CALL
            result = await original_function(*args, **kwargs)
            end_time = datetime.datetime.now()
            if "stream" in kwargs and kwargs["stream"] == True:
                if (
                    "complete_response" in kwargs
                    and kwargs["complete_response"] == True
                ):
                    chunks = []
                    for idx, chunk in enumerate(result):
                        chunks.append(chunk)
                    return litellm.stream_chunk_builder(
                        chunks, messages=kwargs.get("messages", None)
                    )
                else:
                    return result

            # ADD HIDDEN PARAMS - additional call metadata
            if hasattr(result, "_hidden_params"):
                result._hidden_params["model_id"] = kwargs.get("model_info", {}).get(
                    "id", None
                )
                result._hidden_params["api_base"] = get_api_base(
                    model=model,
                    optional_params=kwargs,
                )
            if (
                isinstance(result, ModelResponse)
                or isinstance(result, EmbeddingResponse)
                or isinstance(result, TranscriptionResponse)
            ):
                result._response_ms = (
                    end_time - start_time
                ).total_seconds() * 1000  # return response latency in ms like openai

            ### POST-CALL RULES ###
            post_call_processing(original_response=result, model=model)

            # [OPTIONAL] ADD TO CACHE
            if (
                (litellm.cache is not None)
                and (
                    str(original_function.__name__)
                    in litellm.cache.supported_call_types
                )
                and (kwargs.get("cache", {}).get("no-store", False) != True)
            ):
                if (
                    isinstance(result, litellm.ModelResponse)
                    or isinstance(result, litellm.EmbeddingResponse)
                    or isinstance(result, TranscriptionResponse)
                ):
                    if (
                        isinstance(result, EmbeddingResponse)
                        and isinstance(kwargs["input"], list)
                        and litellm.cache is not None
                        and not isinstance(
                            litellm.cache.cache, S3Cache
                        )  # s3 doesn't support bulk writing. Exclude.
                    ):
                        asyncio.create_task(
                            litellm.cache.async_add_cache_pipeline(
                                result, *args, **kwargs
                            )
                        )
                    elif isinstance(litellm.cache.cache, S3Cache):
                        threading.Thread(
                            target=litellm.cache.add_cache,
                            args=(result,) + args,
                            kwargs=kwargs,
                        ).start()
                    else:
                        asyncio.create_task(
                            litellm.cache.async_add_cache(
                                result.json(), *args, **kwargs
                            )
                        )
                else:
                    asyncio.create_task(
                        litellm.cache.async_add_cache(result, *args, **kwargs)
                    )
            # LOG SUCCESS - handle streaming success logging in the _next_ object
            print_verbose(
                f"Async Wrapper: Completed Call, calling async_success_handler: {logging_obj.async_success_handler}"
            )
            # check if user does not want this to be logged
            asyncio.create_task(
                logging_obj.async_success_handler(result, start_time, end_time)
            )
            threading.Thread(
                target=logging_obj.success_handler,
                args=(result, start_time, end_time),
            ).start()

            # REBUILD EMBEDDING CACHING
            if (
                isinstance(result, EmbeddingResponse)
                and final_embedding_cached_response is not None
            ):
                idx = 0
                final_data_list = []
                for item in final_embedding_cached_response.data:
                    if item is None:
                        final_data_list.append(result.data[idx])
                        idx += 1
                    else:
                        final_data_list.append(item)

                final_embedding_cached_response.data = final_data_list
                final_embedding_cached_response._hidden_params["cache_hit"] = True
                final_embedding_cached_response._response_ms = (
                    end_time - start_time
                ).total_seconds() * 1000
                return final_embedding_cached_response

            return result
        except Exception as e:
            traceback_exception = traceback.format_exc()
            end_time = datetime.datetime.now()
            if logging_obj:
                try:
                    logging_obj.failure_handler(
                        e, traceback_exception, start_time, end_time
                    )  # DO NOT MAKE THREADED - router retry fallback relies on this!
                except Exception as e:
                    raise e
                try:
                    await logging_obj.async_failure_handler(
                        e, traceback_exception, start_time, end_time
                    )
                except Exception as e:
                    raise e

            call_type = original_function.__name__
            if call_type == CallTypes.acompletion.value:
                num_retries = (
                    kwargs.get("num_retries", None) or litellm.num_retries or None
                )
                litellm.num_retries = (
                    None  # set retries to None to prevent infinite loops
                )
                context_window_fallback_dict = kwargs.get(
                    "context_window_fallback_dict", {}
                )

                _is_litellm_router_call = "model_group" in kwargs.get(
                    "metadata", {}
                )  # check if call from litellm.router/proxy
                if (
                    num_retries and not _is_litellm_router_call
                ):  # only enter this if call is not from litellm router/proxy. router has it's own logic for retrying
                    try:
                        kwargs["num_retries"] = num_retries
                        kwargs["original_function"] = original_function
                        if isinstance(
                            e, openai.RateLimitError
                        ):  # rate limiting specific error
                            kwargs["retry_strategy"] = "exponential_backoff_retry"
                        elif isinstance(e, openai.APIError):  # generic api error
                            kwargs["retry_strategy"] = "constant_retry"
                        return await litellm.acompletion_with_retries(*args, **kwargs)
                    except:
                        pass
                elif (
                    isinstance(e, litellm.exceptions.ContextWindowExceededError)
                    and context_window_fallback_dict
                    and model in context_window_fallback_dict
                ):
                    if len(args) > 0:
                        args[0] = context_window_fallback_dict[model]
                    else:
                        kwargs["model"] = context_window_fallback_dict[model]
                    return await original_function(*args, **kwargs)
            raise e

    is_coroutine = inspect.iscoroutinefunction(original_function)

    # Return the appropriate wrapper based on the original function type
    if is_coroutine:
        return wrapper_async
    else:
        return wrapper


@lru_cache(maxsize=128)
def _select_tokenizer(model: str):
    if model in litellm.cohere_models and "command-r" in model:
        # cohere
        cohere_tokenizer = Tokenizer.from_pretrained(
            "Xenova/c4ai-command-r-v01-tokenizer"
        )
        return {"type": "huggingface_tokenizer", "tokenizer": cohere_tokenizer}
    # anthropic
    elif model in litellm.anthropic_models and "claude-3" not in model:
        claude_tokenizer = Tokenizer.from_str(claude_json_str)
        return {"type": "huggingface_tokenizer", "tokenizer": claude_tokenizer}
    # llama2
    elif "llama-2" in model.lower() or "replicate" in model.lower():
        tokenizer = Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        return {"type": "huggingface_tokenizer", "tokenizer": tokenizer}
    # llama3
    elif "llama-3" in model.lower():
        tokenizer = Tokenizer.from_pretrained("Xenova/llama-3-tokenizer")
        return {"type": "huggingface_tokenizer", "tokenizer": tokenizer}
    # default - tiktoken
    else:
        tokenizer = None
        try:
            tokenizer = Tokenizer.from_pretrained(model)
            return {"type": "huggingface_tokenizer", "tokenizer": tokenizer}
        except:
            return {"type": "openai_tokenizer", "tokenizer": encoding}


def encode(model="", text="", custom_tokenizer: Optional[dict] = None):
    """
    Encodes the given text using the specified model.

    Args:
        model (str): The name of the model to use for tokenization.
        custom_tokenizer (Optional[dict]): A custom tokenizer created with the `create_pretrained_tokenizer` or `create_tokenizer` method. Must be a dictionary with a string value for `type` and Tokenizer for `tokenizer`. Default is None.
        text (str): The text to be encoded.

    Returns:
        enc: The encoded text.
    """
    tokenizer_json = custom_tokenizer or _select_tokenizer(model=model)
    enc = tokenizer_json["tokenizer"].encode(text)
    return enc


def decode(model="", tokens: List[int] = [], custom_tokenizer: Optional[dict] = None):
    tokenizer_json = custom_tokenizer or _select_tokenizer(model=model)
    dec = tokenizer_json["tokenizer"].decode(tokens)
    return dec


def openai_token_counter(
    messages: Optional[list] = None,
    model="gpt-3.5-turbo-0613",
    text: Optional[str] = None,
    is_tool_call: Optional[bool] = False,
    count_response_tokens: Optional[
        bool
    ] = False,  # Flag passed from litellm.stream_chunk_builder, to indicate counting tokens for LLM Response. We need this because for LLM input we add +3 tokens per message - based on OpenAI's token counter
):
    """
    Return the number of tokens used by a list of messages.

    Borrowed from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb.
    """
    print_verbose(f"LiteLLM: Utils - Counting tokens for OpenAI model={model}")
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print_verbose("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model in litellm.open_ai_chat_completion_models:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model in litellm.azure_llms:
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0

    if is_tool_call and text is not None:
        # if it's a tool call we assembled 'text' in token_counter()
        num_tokens = len(encoding.encode(text, disallowed_special=()))
    elif messages is not None:
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value, disallowed_special=()))
                    if key == "name":
                        num_tokens += tokens_per_name
                elif isinstance(value, List):
                    for c in value:
                        if c["type"] == "text":
                            text += c["text"]
                            num_tokens += len(
                                encoding.encode(c["text"], disallowed_special=())
                            )
                        elif c["type"] == "image_url":
                            if isinstance(c["image_url"], dict):
                                image_url_dict = c["image_url"]
                                detail = image_url_dict.get("detail", "auto")
                                url = image_url_dict.get("url")
                                num_tokens += calculage_img_tokens(
                                    data=url, mode=detail
                                )
                            elif isinstance(c["image_url"], str):
                                image_url_str = c["image_url"]
                                num_tokens += calculage_img_tokens(
                                    data=image_url_str, mode="auto"
                                )
    elif text is not None and count_response_tokens == True:
        # This is the case where we need to count tokens for a streamed response. We should NOT add +3 tokens per message in this branch
        num_tokens = len(encoding.encode(text, disallowed_special=()))
        return num_tokens
    elif text is not None:
        num_tokens = len(encoding.encode(text, disallowed_special=()))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def resize_image_high_res(width, height):
    # Maximum dimensions for high res mode
    max_short_side = 768
    max_long_side = 2000

    # Determine the longer and shorter sides
    longer_side = max(width, height)
    shorter_side = min(width, height)

    # Calculate the aspect ratio
    aspect_ratio = longer_side / shorter_side

    # Resize based on the short side being 768px
    if width <= height:  # Portrait or square
        resized_width = max_short_side
        resized_height = int(resized_width * aspect_ratio)
        # if the long side exceeds the limit after resizing, adjust both sides accordingly
        if resized_height > max_long_side:
            resized_height = max_long_side
            resized_width = int(resized_height / aspect_ratio)
    else:  # Landscape
        resized_height = max_short_side
        resized_width = int(resized_height * aspect_ratio)
        # if the long side exceeds the limit after resizing, adjust both sides accordingly
        if resized_width > max_long_side:
            resized_width = max_long_side
            resized_height = int(resized_width / aspect_ratio)

    return resized_width, resized_height


# Test the function with the given example
def calculate_tiles_needed(
    resized_width, resized_height, tile_width=512, tile_height=512
):
    tiles_across = (resized_width + tile_width - 1) // tile_width
    tiles_down = (resized_height + tile_height - 1) // tile_height
    total_tiles = tiles_across * tiles_down
    return total_tiles


def get_image_dimensions(data):
    img_data = None

    # Check if data is a URL by trying to parse it
    try:
        response = requests.get(data)
        response.raise_for_status()  # Check if the request was successful
        img_data = response.content
    except Exception:
        # Data is not a URL, handle as base64
        header, encoded = data.split(",", 1)
        img_data = base64.b64decode(encoded)

    # Try to determine dimensions from headers
    # This is a very simplistic check, primarily works with PNG and non-progressive JPEG
    if img_data[:8] == b"\x89PNG\r\n\x1a\n":
        # PNG Image; width and height are 4 bytes each and start at offset 16
        width, height = struct.unpack(">ii", img_data[16:24])
        return width, height
    elif img_data[:2] == b"\xff\xd8":
        # JPEG Image; for dimensions, SOF0 block (0xC0) gives dimensions at offset 3 for length, and then 5 and 7 for height and width
        # This will NOT find dimensions for all JPEGs (e.g., progressive JPEGs)
        # Find SOF0 marker (0xFF followed by 0xC0)
        sof = re.search(b"\xff\xc0....", img_data)
        if sof:
            # Parse SOF0 block to find dimensions
            height, width = struct.unpack(">HH", sof.group()[5:9])
            return width, height
        else:
            return None, None
    else:
        # Unsupported format
        return None, None


def calculage_img_tokens(
    data,
    mode: Literal["low", "high", "auto"] = "auto",
    base_tokens: int = 85,  # openai default - https://openai.com/pricing
):
    if mode == "low" or mode == "auto":
        return base_tokens
    elif mode == "high":
        width, height = get_image_dimensions(data=data)
        resized_width, resized_height = resize_image_high_res(
            width=width, height=height
        )
        tiles_needed_high_res = calculate_tiles_needed(resized_width, resized_height)
        tile_tokens = (base_tokens * 2) * tiles_needed_high_res
        total_tokens = base_tokens + tile_tokens
        return total_tokens


def create_pretrained_tokenizer(
    identifier: str, revision="main", auth_token: Optional[str] = None
):
    """
    Creates a tokenizer from an existing file on a HuggingFace repository to be used with `token_counter`.

    Args:
    identifier (str): The identifier of a Model on the Hugging Face Hub, that contains a tokenizer.json file
    revision (str, defaults to main): A branch or commit id
    auth_token (str, optional, defaults to None): An optional auth token used to access private repositories on the Hugging Face Hub

    Returns:
    dict: A dictionary with the tokenizer and its type.
    """

    tokenizer = Tokenizer.from_pretrained(
        identifier, revision=revision, auth_token=auth_token
    )
    return {"type": "huggingface_tokenizer", "tokenizer": tokenizer}


def create_tokenizer(json: str):
    """
    Creates a tokenizer from a valid JSON string for use with `token_counter`.

    Args:
    json (str): A valid JSON string representing a previously serialized tokenizer

    Returns:
    dict: A dictionary with the tokenizer and its type.
    """

    tokenizer = Tokenizer.from_str(json)
    return {"type": "huggingface_tokenizer", "tokenizer": tokenizer}


def token_counter(
    model="",
    custom_tokenizer: Optional[dict] = None,
    text: Optional[Union[str, List[str]]] = None,
    messages: Optional[List] = None,
    count_response_tokens: Optional[bool] = False,
) -> int:
    """
    Count the number of tokens in a given text using a specified model.

    Args:
    model (str): The name of the model to use for tokenization. Default is an empty string.
    custom_tokenizer (Optional[dict]): A custom tokenizer created with the `create_pretrained_tokenizer` or `create_tokenizer` method. Must be a dictionary with a string value for `type` and Tokenizer for `tokenizer`. Default is None.
    text (str): The raw text string to be passed to the model. Default is None.
    messages (Optional[List[Dict[str, str]]]): Alternative to passing in text. A list of dictionaries representing messages with "role" and "content" keys. Default is None.

    Returns:
    int: The number of tokens in the text.
    """
    # use tiktoken, anthropic, cohere, llama2, or llama3's tokenizer depending on the model
    is_tool_call = False
    num_tokens = 0
    if text == None:
        if messages is not None:
            print_verbose(f"token_counter messages received: {messages}")
            text = ""
            for message in messages:
                if message.get("content", None) is not None:
                    content = message.get("content")
                    if isinstance(content, str):
                        text += message["content"]
                    elif isinstance(content, List):
                        for c in content:
                            if c["type"] == "text":
                                text += c["text"]
                            elif c["type"] == "image_url":
                                if isinstance(c["image_url"], dict):
                                    image_url_dict = c["image_url"]
                                    detail = image_url_dict.get("detail", "auto")
                                    url = image_url_dict.get("url")
                                    num_tokens += calculage_img_tokens(
                                        data=url, mode=detail
                                    )
                                elif isinstance(c["image_url"], str):
                                    image_url_str = c["image_url"]
                                    num_tokens += calculage_img_tokens(
                                        data=image_url_str, mode="auto"
                                    )
                if "tool_calls" in message:
                    is_tool_call = True
                    for tool_call in message["tool_calls"]:
                        if "function" in tool_call:
                            function_arguments = tool_call["function"]["arguments"]
                            text += function_arguments
        else:
            raise ValueError("text and messages cannot both be None")
    elif isinstance(text, List):
        text = "".join(t for t in text if isinstance(t, str))
    elif isinstance(text, str):
        count_response_tokens = True  # user just trying to count tokens for a text. don't add the chat_ml +3 tokens to this

    if model is not None or custom_tokenizer is not None:
        tokenizer_json = custom_tokenizer or _select_tokenizer(model=model)
        if tokenizer_json["type"] == "huggingface_tokenizer":
            enc = tokenizer_json["tokenizer"].encode(text)
            num_tokens = len(enc.ids)
        elif tokenizer_json["type"] == "openai_tokenizer":
            if (
                model in litellm.open_ai_chat_completion_models
                or model in litellm.azure_llms
            ):
                if model in litellm.azure_llms:
                    # azure llms use gpt-35-turbo instead of gpt-3.5-turbo 🙃
                    model = model.replace("-35", "-3.5")

                print_verbose(
                    f"Token Counter - using OpenAI token counter, for model={model}"
                )
                num_tokens = openai_token_counter(
                    text=text,  # type: ignore
                    model=model,
                    messages=messages,
                    is_tool_call=is_tool_call,
                    count_response_tokens=count_response_tokens,
                )
            else:
                print_verbose(
                    f"Token Counter - using generic token counter, for model={model}"
                )
                num_tokens = openai_token_counter(
                    text=text,  # type: ignore
                    model="gpt-3.5-turbo",
                    messages=messages,
                    is_tool_call=is_tool_call,
                    count_response_tokens=count_response_tokens,
                )
    else:
        num_tokens = len(encoding.encode(text, disallowed_special=()))  # type: ignore
    return num_tokens


def _cost_per_token_custom_pricing_helper(
    prompt_tokens=0,
    completion_tokens=0,
    response_time_ms=None,
    ### CUSTOM PRICING ###
    custom_cost_per_token: Optional[CostPerToken] = None,
    custom_cost_per_second: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Internal helper function for calculating cost, if custom pricing given"""
    if custom_cost_per_token is None and custom_cost_per_second is None:
        return None

    if custom_cost_per_token is not None:
        input_cost = custom_cost_per_token["input_cost_per_token"] * prompt_tokens
        output_cost = custom_cost_per_token["output_cost_per_token"] * completion_tokens
        return input_cost, output_cost
    elif custom_cost_per_second is not None:
        output_cost = custom_cost_per_second * response_time_ms / 1000  # type: ignore
        return 0, output_cost

    return None


def cost_per_token(
    model: str = "",
    prompt_tokens=0,
    completion_tokens=0,
    response_time_ms=None,
    custom_llm_provider=None,
    region_name=None,
    ### CUSTOM PRICING ###
    custom_cost_per_token: Optional[CostPerToken] = None,
    custom_cost_per_second: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Parameters:
        model (str): The name of the model to use. Default is ""
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens in the completion.
        response_time (float): The amount of time, in milliseconds, it took the call to complete.
        custom_llm_provider (str): The llm provider to whom the call was made (see init.py for full list)
        custom_cost_per_token: Optional[CostPerToken]: the cost per input + output token for the llm api call.
        custom_cost_per_second: Optional[float]: the cost per second for the llm api call.

    Returns:
        tuple: A tuple containing the cost in USD dollars for prompt tokens and completion tokens, respectively.
    """
    if model is None:
        raise Exception("Invalid arg. Model cannot be none.")
    ## CUSTOM PRICING ##
    response_cost = _cost_per_token_custom_pricing_helper(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_time_ms=response_time_ms,
        custom_cost_per_second=custom_cost_per_second,
        custom_cost_per_token=custom_cost_per_token,
    )
    if response_cost is not None:
        return response_cost[0], response_cost[1]

    # given
    prompt_tokens_cost_usd_dollar: float = 0
    completion_tokens_cost_usd_dollar: float = 0
    model_cost_ref = litellm.model_cost
    model_with_provider = model
    if custom_llm_provider is not None:
        model_with_provider = custom_llm_provider + "/" + model
        if region_name is not None:
            model_with_provider_and_region = (
                f"{custom_llm_provider}/{region_name}/{model}"
            )
            if (
                model_with_provider_and_region in model_cost_ref
            ):  # use region based pricing, if it's available
                model_with_provider = model_with_provider_and_region

    model_without_prefix = model
    model_parts = model.split("/")
    if len(model_parts) > 1:
        model_without_prefix = model_parts[1]
    else:
        model_without_prefix = model
    """
    Code block that formats model to lookup in litellm.model_cost
    Option1. model = "bedrock/ap-northeast-1/anthropic.claude-instant-v1". This is the most accurate since it is region based. Should always be option 1
    Option2. model = "openai/gpt-4"       - model = provider/model
    Option3. model = "anthropic.claude-3" - model = model
    """
    if (
        model_with_provider in model_cost_ref
    ):  # Option 2. use model with provider, model = "openai/gpt-4"
        model = model_with_provider
    elif model in model_cost_ref:  # Option 1. use model passed, model="gpt-4"
        model = model
    elif (
        model_without_prefix in model_cost_ref
    ):  # Option 3. if user passed model="bedrock/anthropic.claude-3", use model="anthropic.claude-3"
        model = model_without_prefix

    # see this https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models
    print_verbose(f"Looking up model={model} in model_cost_map")
    if model in model_cost_ref:
        print_verbose(f"Success: model={model} in model_cost_map")
        print_verbose(
            f"prompt_tokens={prompt_tokens}; completion_tokens={completion_tokens}"
        )
        if (
            model_cost_ref[model].get("input_cost_per_token", None) is not None
            and model_cost_ref[model].get("output_cost_per_token", None) is not None
        ):
            ## COST PER TOKEN ##
            prompt_tokens_cost_usd_dollar = (
                model_cost_ref[model]["input_cost_per_token"] * prompt_tokens
            )
            completion_tokens_cost_usd_dollar = (
                model_cost_ref[model]["output_cost_per_token"] * completion_tokens
            )
        elif (
            model_cost_ref[model].get("output_cost_per_second", None) is not None
            and response_time_ms is not None
        ):
            print_verbose(
                f"For model={model} - output_cost_per_second: {model_cost_ref[model].get('output_cost_per_second')}; response time: {response_time_ms}"
            )
            ## COST PER SECOND ##
            prompt_tokens_cost_usd_dollar = 0
            completion_tokens_cost_usd_dollar = (
                model_cost_ref[model]["output_cost_per_second"]
                * response_time_ms
                / 1000
            )
        elif (
            model_cost_ref[model].get("input_cost_per_second", None) is not None
            and response_time_ms is not None
        ):
            print_verbose(
                f"For model={model} - input_cost_per_second: {model_cost_ref[model].get('input_cost_per_second')}; response time: {response_time_ms}"
            )
            ## COST PER SECOND ##
            prompt_tokens_cost_usd_dollar = (
                model_cost_ref[model]["input_cost_per_second"] * response_time_ms / 1000
            )
            completion_tokens_cost_usd_dollar = 0.0
        print_verbose(
            f"Returned custom cost for model={model} - prompt_tokens_cost_usd_dollar: {prompt_tokens_cost_usd_dollar}, completion_tokens_cost_usd_dollar: {completion_tokens_cost_usd_dollar}"
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    elif "ft:gpt-3.5-turbo" in model:
        print_verbose(f"Cost Tracking: {model} is an OpenAI FinteTuned LLM")
        # fuzzy match ft:gpt-3.5-turbo:abcd-id-cool-litellm
        prompt_tokens_cost_usd_dollar = (
            model_cost_ref["ft:gpt-3.5-turbo"]["input_cost_per_token"] * prompt_tokens
        )
        completion_tokens_cost_usd_dollar = (
            model_cost_ref["ft:gpt-3.5-turbo"]["output_cost_per_token"]
            * completion_tokens
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    elif "ft:davinci-002" in model:
        print_verbose(f"Cost Tracking: {model} is an OpenAI FinteTuned LLM")
        # fuzzy match ft:davinci-002:abcd-id-cool-litellm
        prompt_tokens_cost_usd_dollar = (
            model_cost_ref["ft:davinci-002"]["input_cost_per_token"] * prompt_tokens
        )
        completion_tokens_cost_usd_dollar = (
            model_cost_ref["ft:davinci-002"]["output_cost_per_token"]
            * completion_tokens
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    elif "ft:babbage-002" in model:
        print_verbose(f"Cost Tracking: {model} is an OpenAI FinteTuned LLM")
        # fuzzy match ft:babbage-002:abcd-id-cool-litellm
        prompt_tokens_cost_usd_dollar = (
            model_cost_ref["ft:babbage-002"]["input_cost_per_token"] * prompt_tokens
        )
        completion_tokens_cost_usd_dollar = (
            model_cost_ref["ft:babbage-002"]["output_cost_per_token"]
            * completion_tokens
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    elif model in litellm.azure_llms:
        verbose_logger.debug(f"Cost Tracking: {model} is an Azure LLM")
        model = litellm.azure_llms[model]
        verbose_logger.debug(
            f"applying cost={model_cost_ref[model]['input_cost_per_token']} for prompt_tokens={prompt_tokens}"
        )
        prompt_tokens_cost_usd_dollar = (
            model_cost_ref[model]["input_cost_per_token"] * prompt_tokens
        )
        verbose_logger.debug(
            f"applying cost={model_cost_ref[model]['output_cost_per_token']} for completion_tokens={completion_tokens}"
        )
        completion_tokens_cost_usd_dollar = (
            model_cost_ref[model]["output_cost_per_token"] * completion_tokens
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    elif model in litellm.azure_embedding_models:
        verbose_logger.debug(f"Cost Tracking: {model} is an Azure Embedding Model")
        model = litellm.azure_embedding_models[model]
        prompt_tokens_cost_usd_dollar = (
            model_cost_ref[model]["input_cost_per_token"] * prompt_tokens
        )
        completion_tokens_cost_usd_dollar = (
            model_cost_ref[model]["output_cost_per_token"] * completion_tokens
        )
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    else:
        # if model is not in model_prices_and_context_window.json. Raise an exception-let users know
        error_str = f"Model not in model_prices_and_context_window.json. You passed model={model}. Register pricing for model - https://docs.litellm.ai/docs/proxy/custom_pricing\n"
        raise litellm.exceptions.NotFoundError(  # type: ignore
            message=error_str,
            model=model,
            response=httpx.Response(
                status_code=404,
                content=error_str,
                request=httpx.Request(method="cost_per_token", url="https://github.com/BerriAI/litellm"),  # type: ignore
            ),
            llm_provider="",
        )


def supports_httpx_timeout(custom_llm_provider: str) -> bool:
    """
    Helper function to know if a provider implementation supports httpx timeout
    """
    supported_providers = ["openai", "azure", "bedrock"]

    if custom_llm_provider in supported_providers:
        return True

    return False


def supports_function_calling(model: str) -> bool:
    """
    Check if the given model supports function calling and return a boolean value.

    Parameters:
    model (str): The model name to be checked.

    Returns:
    bool: True if the model supports function calling, False otherwise.

    Raises:
    Exception: If the given model is not found in model_prices_and_context_window.json.
    """
    if model in litellm.model_cost:
        model_info = litellm.model_cost[model]
        if model_info.get("supports_function_calling", False):
            return True
        return False
    else:
        raise Exception(
            f"Model not in model_prices_and_context_window.json. You passed model={model}."
        )


def supports_vision(model: str):
    """
    Check if the given model supports vision and return a boolean value.

    Parameters:
    model (str): The model name to be checked.

    Returns:
    bool: True if the model supports vision, False otherwise.

    Raises:
    Exception: If the given model is not found in model_prices_and_context_window.json.
    """
    if model in litellm.model_cost:
        model_info = litellm.model_cost[model]
        if model_info.get("supports_vision", False):
            return True
        return False
    else:
        return False


def supports_parallel_function_calling(model: str):
    """
    Check if the given model supports parallel function calling and return True if it does, False otherwise.

    Parameters:
        model (str): The model to check for support of parallel function calling.

    Returns:
        bool: True if the model supports parallel function calling, False otherwise.

    Raises:
        Exception: If the model is not found in the model_cost dictionary.
    """
    if model in litellm.model_cost:
        model_info = litellm.model_cost[model]
        if model_info.get("supports_parallel_function_calling", False):
            return True
        return False
    else:
        raise Exception(
            f"Model not in model_prices_and_context_window.json. You passed model={model}."
        )


####### HELPER FUNCTIONS ################
def register_model(model_cost: Union[str, dict]):
    """
    Register new / Override existing models (and their pricing) to specific providers.
    Provide EITHER a model cost dictionary or a url to a hosted json blob
    Example usage:
    model_cost_dict = {
        "gpt-4": {
            "max_tokens": 8192,
            "input_cost_per_token": 0.00003,
            "output_cost_per_token": 0.00006,
            "litellm_provider": "openai",
            "mode": "chat"
        },
    }
    """
    loaded_model_cost = {}
    if isinstance(model_cost, dict):
        loaded_model_cost = model_cost
    elif isinstance(model_cost, str):
        loaded_model_cost = litellm.get_model_cost_map(url=model_cost)

    for key, value in loaded_model_cost.items():
        ## override / add new keys to the existing model cost dictionary
        litellm.model_cost.setdefault(key, {}).update(value)
        verbose_logger.debug(f"{key} added to model cost map")
        # add new model names to provider lists
        if value.get("litellm_provider") == "openai":
            if key not in litellm.open_ai_chat_completion_models:
                litellm.open_ai_chat_completion_models.append(key)
        elif value.get("litellm_provider") == "text-completion-openai":
            if key not in litellm.open_ai_text_completion_models:
                litellm.open_ai_text_completion_models.append(key)
        elif value.get("litellm_provider") == "cohere":
            if key not in litellm.cohere_models:
                litellm.cohere_models.append(key)
        elif value.get("litellm_provider") == "anthropic":
            if key not in litellm.anthropic_models:
                litellm.anthropic_models.append(key)
        elif value.get("litellm_provider") == "openrouter":
            split_string = key.split("/", 1)
            if key not in litellm.openrouter_models:
                litellm.openrouter_models.append(split_string[1])
        elif value.get("litellm_provider") == "vertex_ai-text-models":
            if key not in litellm.vertex_text_models:
                litellm.vertex_text_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-code-text-models":
            if key not in litellm.vertex_code_text_models:
                litellm.vertex_code_text_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-chat-models":
            if key not in litellm.vertex_chat_models:
                litellm.vertex_chat_models.append(key)
        elif value.get("litellm_provider") == "vertex_ai-code-chat-models":
            if key not in litellm.vertex_code_chat_models:
                litellm.vertex_code_chat_models.append(key)
        elif value.get("litellm_provider") == "ai21":
            if key not in litellm.ai21_models:
                litellm.ai21_models.append(key)
        elif value.get("litellm_provider") == "nlp_cloud":
            if key not in litellm.nlp_cloud_models:
                litellm.nlp_cloud_models.append(key)
        elif value.get("litellm_provider") == "aleph_alpha":
            if key not in litellm.aleph_alpha_models:
                litellm.aleph_alpha_models.append(key)
        elif value.get("litellm_provider") == "bedrock":
            if key not in litellm.bedrock_models:
                litellm.bedrock_models.append(key)
    return model_cost


def get_litellm_params(
    api_key=None,
    force_timeout=600,
    azure=False,
    logger_fn=None,
    verbose=False,
    hugging_face=False,
    replicate=False,
    together_ai=False,
    custom_llm_provider=None,
    api_base=None,
    litellm_call_id=None,
    model_alias_map=None,
    completion_call_id=None,
    metadata=None,
    model_info=None,
    proxy_server_request=None,
    acompletion=None,
    preset_cache_key=None,
    no_log=None,
    input_cost_per_second=None,
    input_cost_per_token=None,
    output_cost_per_token=None,
    output_cost_per_second=None,
):
    litellm_params = {
        "acompletion": acompletion,
        "api_key": api_key,
        "force_timeout": force_timeout,
        "logger_fn": logger_fn,
        "verbose": verbose,
        "custom_llm_provider": custom_llm_provider,
        "api_base": api_base,
        "litellm_call_id": litellm_call_id,
        "model_alias_map": model_alias_map,
        "completion_call_id": completion_call_id,
        "metadata": metadata,
        "model_info": model_info,
        "proxy_server_request": proxy_server_request,
        "preset_cache_key": preset_cache_key,
        "no-log": no_log,
        "stream_response": {},  # litellm_call_id: ModelResponse Dict
        "input_cost_per_token": input_cost_per_token,
        "input_cost_per_second": input_cost_per_second,
        "output_cost_per_token": output_cost_per_token,
        "output_cost_per_second": output_cost_per_second,
    }

    return litellm_params


def get_optional_params_image_gen(
    n: Optional[int] = None,
    quality: Optional[str] = None,
    response_format: Optional[str] = None,
    size: Optional[str] = None,
    style: Optional[str] = None,
    user: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
):
    # retrieve all parameters passed to the function
    passed_params = locals()
    custom_llm_provider = passed_params.pop("custom_llm_provider")
    special_params = passed_params.pop("kwargs")
    for k, v in special_params.items():
        passed_params[k] = v

    default_params = {
        "n": None,
        "quality": None,
        "response_format": None,
        "size": None,
        "style": None,
        "user": None,
    }

    non_default_params = {
        k: v
        for k, v in passed_params.items()
        if (k in default_params and v != default_params[k])
    }
    optional_params = {}

    ## raise exception if non-default value passed for non-openai/azure embedding calls
    def _check_valid_arg(supported_params):
        if len(non_default_params.keys()) > 0:
            keys = list(non_default_params.keys())
            for k in keys:
                if (
                    litellm.drop_params is True and k not in supported_params
                ):  # drop the unsupported non-default values
                    non_default_params.pop(k, None)
                elif k not in supported_params:
                    raise UnsupportedParamsError(
                        status_code=500,
                        message=f"Setting user/encoding format is not supported by {custom_llm_provider}. To drop it from the call, set `litellm.drop_params = True`.",
                    )
            return non_default_params

    if (
        custom_llm_provider == "openai"
        or custom_llm_provider == "azure"
        or custom_llm_provider in litellm.openai_compatible_providers
    ):
        optional_params = non_default_params
    elif custom_llm_provider == "bedrock":
        supported_params = ["size"]
        _check_valid_arg(supported_params=supported_params)
        if size is not None:
            width, height = size.split("x")
            optional_params["width"] = int(width)
            optional_params["height"] = int(height)
    elif custom_llm_provider == "vertex_ai":
        supported_params = ["n"]
        """
        All params here: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/imagegeneration?project=adroit-crow-413218
        """
        _check_valid_arg(supported_params=supported_params)
        if n is not None:
            optional_params["sampleCount"] = int(n)

    for k in passed_params.keys():
        if k not in default_params.keys():
            optional_params[k] = passed_params[k]
    return optional_params


def get_optional_params_embeddings(
    # 2 optional params
    model=None,
    user=None,
    encoding_format=None,
    dimensions=None,
    custom_llm_provider="",
    **kwargs,
):
    # retrieve all parameters passed to the function
    passed_params = locals()
    custom_llm_provider = passed_params.pop("custom_llm_provider", None)
    special_params = passed_params.pop("kwargs")
    for k, v in special_params.items():
        passed_params[k] = v

    default_params = {"user": None, "encoding_format": None, "dimensions": None}

    def _check_valid_arg(supported_params: Optional[list]):
        if supported_params is None:
            return
        unsupported_params = {}
        for k in non_default_params.keys():
            if k not in supported_params:
                unsupported_params[k] = non_default_params[k]
        if unsupported_params and not litellm.drop_params:
            raise UnsupportedParamsError(
                status_code=500,
                message=f"{custom_llm_provider} does not support parameters: {unsupported_params}, for model={model}. To drop these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n",
            )

    non_default_params = {
        k: v
        for k, v in passed_params.items()
        if (k in default_params and v != default_params[k])
    }

    ## raise exception if non-default value passed for non-openai/azure embedding calls
    if custom_llm_provider == "openai":
        # 'dimensions` is only supported in `text-embedding-3` and later models

        if (
            model is not None
            and "text-embedding-3" not in model
            and "dimensions" in non_default_params.keys()
        ):
            raise UnsupportedParamsError(
                status_code=500,
                message=f"Setting dimensions is not supported for OpenAI `text-embedding-3` and later models. To drop it from the call, set `litellm.drop_params = True`.",
            )
    if custom_llm_provider == "triton":
        keys = list(non_default_params.keys())
        for k in keys:
            non_default_params.pop(k, None)
        final_params = {**non_default_params, **kwargs}
        return final_params
    if custom_llm_provider == "databricks":
        supported_params = get_supported_openai_params(
            model=model or "",
            custom_llm_provider="databricks",
            request_type="embeddings",
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.DatabricksEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params, optional_params={}
        )
        final_params = {**optional_params, **kwargs}
        return final_params
    if custom_llm_provider == "vertex_ai":
        if len(non_default_params.keys()) > 0:
            if litellm.drop_params is True:  # drop the unsupported non-default values
                keys = list(non_default_params.keys())
                for k in keys:
                    non_default_params.pop(k, None)
                final_params = {**non_default_params, **kwargs}
                return final_params
            raise UnsupportedParamsError(
                status_code=500,
                message=f"Setting user/encoding format is not supported by {custom_llm_provider}. To drop it from the call, set `litellm.drop_params = True`.",
            )
    if custom_llm_provider == "bedrock":
        # if dimensions is in non_default_params -> pass it for model=bedrock/amazon.titan-embed-text-v2
        if (
            "dimensions" in non_default_params.keys()
            and "amazon.titan-embed-text-v2" in model
        ):
            kwargs["dimensions"] = non_default_params["dimensions"]
            non_default_params.pop("dimensions", None)

        if len(non_default_params.keys()) > 0:
            if litellm.drop_params is True:  # drop the unsupported non-default values
                keys = list(non_default_params.keys())
                for k in keys:
                    non_default_params.pop(k, None)
                final_params = {**non_default_params, **kwargs}
                return final_params
            raise UnsupportedParamsError(
                status_code=500,
                message=f"Setting user/encoding format is not supported by {custom_llm_provider}. To drop it from the call, set `litellm.drop_params = True`.",
            )
        return {**non_default_params, **kwargs}

    if (
        custom_llm_provider != "openai"
        and custom_llm_provider != "azure"
        and custom_llm_provider not in litellm.openai_compatible_providers
    ):
        if len(non_default_params.keys()) > 0:
            if litellm.drop_params is True:  # drop the unsupported non-default values
                keys = list(non_default_params.keys())
                for k in keys:
                    non_default_params.pop(k, None)
            else:
                raise UnsupportedParamsError(
                    status_code=500,
                    message=f"Setting user/encoding format is not supported by {custom_llm_provider}. To drop it from the call, set `litellm.drop_params = True`.",
                )

    final_params = {**non_default_params, **kwargs}
    return final_params


def get_optional_params(
    # use the openai defaults
    # https://platform.openai.com/docs/api-reference/chat/create
    model: str,
    functions=None,
    function_call=None,
    temperature=None,
    top_p=None,
    n=None,
    stream=False,
    stream_options=None,
    stop=None,
    max_tokens=None,
    presence_penalty=None,
    frequency_penalty=None,
    logit_bias=None,
    user=None,
    custom_llm_provider="",
    response_format=None,
    seed=None,
    tools=None,
    tool_choice=None,
    max_retries=None,
    logprobs=None,
    top_logprobs=None,
    extra_headers=None,
    api_version=None,
    drop_params=None,
    **kwargs,
):
    # retrieve all parameters passed to the function
    passed_params = locals().copy()
    special_params = passed_params.pop("kwargs")
    for k, v in special_params.items():
        if k.startswith("aws_") and (
            custom_llm_provider != "bedrock" and custom_llm_provider != "sagemaker"
        ):  # allow dynamically setting boto3 init logic
            continue
        elif k == "hf_model_name" and custom_llm_provider != "sagemaker":
            continue
        elif (
            k.startswith("vertex_") and custom_llm_provider != "vertex_ai"
        ):  # allow dynamically setting vertex ai init logic
            continue

        passed_params[k] = v

    optional_params: Dict = {}

    common_auth_dict = litellm.common_cloud_provider_auth_params
    if custom_llm_provider in common_auth_dict["providers"]:
        """
        Check if params = ["project", "region_name", "token"]
        and correctly translate for = ["azure", "vertex_ai", "watsonx", "aws"]
        """
        if custom_llm_provider == "azure":
            optional_params = litellm.AzureOpenAIConfig().map_special_auth_params(
                non_default_params=passed_params, optional_params=optional_params
            )
        elif custom_llm_provider == "bedrock":
            optional_params = (
                litellm.AmazonBedrockGlobalConfig().map_special_auth_params(
                    non_default_params=passed_params, optional_params=optional_params
                )
            )
        elif custom_llm_provider == "vertex_ai":
            optional_params = litellm.VertexAIConfig().map_special_auth_params(
                non_default_params=passed_params, optional_params=optional_params
            )
        elif custom_llm_provider == "watsonx":
            optional_params = litellm.IBMWatsonXAIConfig().map_special_auth_params(
                non_default_params=passed_params, optional_params=optional_params
            )

    default_params = {
        "functions": None,
        "function_call": None,
        "temperature": None,
        "top_p": None,
        "n": None,
        "stream": None,
        "stream_options": None,
        "stop": None,
        "max_tokens": None,
        "presence_penalty": None,
        "frequency_penalty": None,
        "logit_bias": None,
        "user": None,
        "model": None,
        "custom_llm_provider": "",
        "response_format": None,
        "seed": None,
        "tools": None,
        "tool_choice": None,
        "max_retries": None,
        "logprobs": None,
        "top_logprobs": None,
        "extra_headers": None,
        "api_version": None,
        "drop_params": None,
    }
    # filter out those parameters that were passed with non-default values
    non_default_params = {
        k: v
        for k, v in passed_params.items()
        if (
            k != "model"
            and k != "custom_llm_provider"
            and k != "api_version"
            and k != "drop_params"
            and k in default_params
            and v != default_params[k]
        )
    }

    ## raise exception if function calling passed in for a provider that doesn't support it
    if (
        "functions" in non_default_params
        or "function_call" in non_default_params
        or "tools" in non_default_params
    ):
        if (
            custom_llm_provider == "ollama"
            and custom_llm_provider != "text-completion-openai"
            and custom_llm_provider != "azure"
            and custom_llm_provider != "vertex_ai"
            and custom_llm_provider != "anyscale"
            and custom_llm_provider != "together_ai"
            and custom_llm_provider != "groq"
            and custom_llm_provider != "deepseek"
            and custom_llm_provider != "mistral"
            and custom_llm_provider != "anthropic"
            and custom_llm_provider != "cohere_chat"
            and custom_llm_provider != "cohere"
            and custom_llm_provider != "bedrock"
            and custom_llm_provider != "ollama_chat"
            and custom_llm_provider != "openrouter"
            and custom_llm_provider not in litellm.openai_compatible_providers
        ):
            if custom_llm_provider == "ollama":
                # ollama actually supports json output
                optional_params["format"] = "json"
                litellm.add_function_to_prompt = (
                    True  # so that main.py adds the function call to the prompt
                )
                if "tools" in non_default_params:
                    optional_params["functions_unsupported_model"] = (
                        non_default_params.pop("tools")
                    )
                    non_default_params.pop(
                        "tool_choice", None
                    )  # causes ollama requests to hang
                elif "functions" in non_default_params:
                    optional_params["functions_unsupported_model"] = (
                        non_default_params.pop("functions")
                    )
            elif (
                litellm.add_function_to_prompt
            ):  # if user opts to add it to prompt instead
                optional_params["functions_unsupported_model"] = non_default_params.pop(
                    "tools", non_default_params.pop("functions", None)
                )
            else:
                raise UnsupportedParamsError(
                    status_code=500,
                    message=f"Function calling is not supported by {custom_llm_provider}.",
                )

    def _check_valid_arg(supported_params):
        verbose_logger.debug(
            f"\nLiteLLM completion() model= {model}; provider = {custom_llm_provider}"
        )
        verbose_logger.debug(
            f"\nLiteLLM: Params passed to completion() {passed_params}"
        )
        verbose_logger.debug(
            f"\nLiteLLM: Non-Default params passed to completion() {non_default_params}"
        )
        unsupported_params = {}
        for k in non_default_params.keys():
            if k not in supported_params:
                if k == "user" or k == "stream_options":
                    continue
                if k == "n" and n == 1:  # langchain sends n=1 as a default value
                    continue  # skip this param
                if (
                    k == "max_retries"
                ):  # TODO: This is a patch. We support max retries for OpenAI, Azure. For non OpenAI LLMs we need to add support for max retries
                    continue  # skip this param
                # Always keeps this in elif code blocks
                else:
                    unsupported_params[k] = non_default_params[k]
        if unsupported_params:
            if litellm.drop_params == True or (
                drop_params is not None and drop_params == True
            ):
                pass
            else:
                raise UnsupportedParamsError(
                    status_code=500,
                    message=f"{custom_llm_provider} does not support parameters: {unsupported_params}, for model={model}. To drop these, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n",
                )

    def _map_and_modify_arg(supported_params: dict, provider: str, model: str):
        """
        filter params to fit the required provider format, drop those that don't fit if user sets `litellm.drop_params = True`.
        """
        filtered_stop = None
        if "stop" in supported_params and litellm.drop_params:
            if provider == "bedrock" and "amazon" in model:
                filtered_stop = []
                if isinstance(stop, list):
                    for s in stop:
                        if re.match(r"^(\|+|User:)$", s):
                            filtered_stop.append(s)
        if filtered_stop is not None:
            supported_params["stop"] = filtered_stop

        return supported_params

    ## raise exception if provider doesn't support passed in param
    if custom_llm_provider == "anthropic":
        ## check if unsupported param passed in
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.AnthropicConfig().map_openai_params(
            non_default_params=non_default_params, optional_params=optional_params
        )
    elif custom_llm_provider == "cohere":
        ## check if unsupported param passed in
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)
        # handle cohere params
        if stream:
            optional_params["stream"] = stream
        if temperature is not None:
            optional_params["temperature"] = temperature
        if max_tokens is not None:
            optional_params["max_tokens"] = max_tokens
        if n is not None:
            optional_params["num_generations"] = n
        if logit_bias is not None:
            optional_params["logit_bias"] = logit_bias
        if top_p is not None:
            optional_params["p"] = top_p
        if frequency_penalty is not None:
            optional_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            optional_params["presence_penalty"] = presence_penalty
        if stop is not None:
            optional_params["stop_sequences"] = stop
    elif custom_llm_provider == "cohere_chat":
        ## check if unsupported param passed in
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)
        # handle cohere params
        if stream:
            optional_params["stream"] = stream
        if temperature is not None:
            optional_params["temperature"] = temperature
        if max_tokens is not None:
            optional_params["max_tokens"] = max_tokens
        if n is not None:
            optional_params["num_generations"] = n
        if top_p is not None:
            optional_params["p"] = top_p
        if frequency_penalty is not None:
            optional_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            optional_params["presence_penalty"] = presence_penalty
        if stop is not None:
            optional_params["stop_sequences"] = stop
        if tools is not None:
            optional_params["tools"] = tools
        if seed is not None:
            optional_params["seed"] = seed
    elif custom_llm_provider == "maritalk":
        ## check if unsupported param passed in
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)
        # handle cohere params
        if stream:
            optional_params["stream"] = stream
        if temperature is not None:
            optional_params["temperature"] = temperature
        if max_tokens is not None:
            optional_params["max_tokens"] = max_tokens
        if logit_bias is not None:
            optional_params["logit_bias"] = logit_bias
        if top_p is not None:
            optional_params["p"] = top_p
        if presence_penalty is not None:
            optional_params["repetition_penalty"] = presence_penalty
        if stop is not None:
            optional_params["stopping_tokens"] = stop
    elif custom_llm_provider == "replicate":
        ## check if unsupported param passed in
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)

        if stream:
            optional_params["stream"] = stream
            return optional_params
        if max_tokens is not None:
            if "vicuna" in model or "flan" in model:
                optional_params["max_length"] = max_tokens
            elif "meta/codellama-13b" in model:
                optional_params["max_tokens"] = max_tokens
            else:
                optional_params["max_new_tokens"] = max_tokens
        if temperature is not None:
            optional_params["temperature"] = temperature
        if top_p is not None:
            optional_params["top_p"] = top_p
        if stop is not None:
            optional_params["stop_sequences"] = stop
    elif custom_llm_provider == "huggingface" or custom_llm_provider == "predibase":
        ## check if unsupported param passed in
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.HuggingfaceConfig().map_openai_params(
            non_default_params=non_default_params, optional_params=optional_params
        )
    elif custom_llm_provider == "together_ai":
        ## check if unsupported param passed in
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)

        if stream:
            optional_params["stream"] = stream
        if temperature is not None:
            optional_params["temperature"] = temperature
        if top_p is not None:
            optional_params["top_p"] = top_p
        if max_tokens is not None:
            optional_params["max_tokens"] = max_tokens
        if frequency_penalty is not None:
            optional_params["frequency_penalty"] = frequency_penalty
        if stop is not None:
            optional_params["stop"] = stop
        if tools is not None:
            optional_params["tools"] = tools
        if tool_choice is not None:
            optional_params["tool_choice"] = tool_choice
    elif custom_llm_provider == "ai21":
        ## check if unsupported param passed in
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)

        if stream:
            optional_params["stream"] = stream
        if n is not None:
            optional_params["numResults"] = n
        if max_tokens is not None:
            optional_params["maxTokens"] = max_tokens
        if temperature is not None:
            optional_params["temperature"] = temperature
        if top_p is not None:
            optional_params["topP"] = top_p
        if stop is not None:
            optional_params["stopSequences"] = stop
        if frequency_penalty is not None:
            optional_params["frequencyPenalty"] = {"scale": frequency_penalty}
        if presence_penalty is not None:
            optional_params["presencePenalty"] = {"scale": presence_penalty}
    elif (
        custom_llm_provider == "palm" or custom_llm_provider == "gemini"
    ):  # https://developers.generativeai.google/tutorials/curl_quickstart
        ## check if unsupported param passed in
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)

        if temperature is not None:
            optional_params["temperature"] = temperature
        if top_p is not None:
            optional_params["top_p"] = top_p
        if stream:
            optional_params["stream"] = stream
        if n is not None:
            optional_params["candidate_count"] = n
        if stop is not None:
            if isinstance(stop, str):
                optional_params["stop_sequences"] = [stop]
            elif isinstance(stop, list):
                optional_params["stop_sequences"] = stop
        if max_tokens is not None:
            optional_params["max_output_tokens"] = max_tokens
    elif custom_llm_provider == "vertex_ai" and (
        model in litellm.vertex_chat_models
        or model in litellm.vertex_code_chat_models
        or model in litellm.vertex_text_models
        or model in litellm.vertex_code_text_models
        or model in litellm.vertex_language_models
        or model in litellm.vertex_embedding_models
        or model in litellm.vertex_vision_models
    ):
        print_verbose(f"(start) INSIDE THE VERTEX AI OPTIONAL PARAM BLOCK")
        ## check if unsupported param passed in
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)

        optional_params = litellm.VertexAIConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
        )

        print_verbose(
            f"(end) INSIDE THE VERTEX AI OPTIONAL PARAM BLOCK - optional_params: {optional_params}"
        )
    elif (
        custom_llm_provider == "vertex_ai" and model in litellm.vertex_anthropic_models
    ):
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.VertexAIAnthropicConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
        )
    elif custom_llm_provider == "sagemaker":
        ## check if unsupported param passed in
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)
        # temperature, top_p, n, stream, stop, max_tokens, n, presence_penalty default to None
        if temperature is not None:
            if temperature == 0.0 or temperature == 0:
                # hugging face exception raised when temp==0
                # Failed: Error occurred: HuggingfaceException - Input validation error: `temperature` must be strictly positive
                temperature = 0.01
            optional_params["temperature"] = temperature
        if top_p is not None:
            optional_params["top_p"] = top_p
        if n is not None:
            optional_params["best_of"] = n
            optional_params["do_sample"] = (
                True  # Need to sample if you want best of for hf inference endpoints
            )
        if stream is not None:
            optional_params["stream"] = stream
        if stop is not None:
            optional_params["stop"] = stop
        if max_tokens is not None:
            # HF TGI raises the following exception when max_new_tokens==0
            # Failed: Error occurred: HuggingfaceException - Input validation error: `max_new_tokens` must be strictly positive
            if max_tokens == 0:
                max_tokens = 1
            optional_params["max_new_tokens"] = max_tokens
    elif custom_llm_provider == "bedrock":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        if "ai21" in model:
            _check_valid_arg(supported_params=supported_params)
            # params "maxTokens":200,"temperature":0,"topP":250,"stop_sequences":[],
            # https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=j2-ultra
            if max_tokens is not None:
                optional_params["maxTokens"] = max_tokens
            if temperature is not None:
                optional_params["temperature"] = temperature
            if top_p is not None:
                optional_params["topP"] = top_p
            if stream:
                optional_params["stream"] = stream
        elif "anthropic" in model:
            _check_valid_arg(supported_params=supported_params)
            if "aws_bedrock_client" in passed_params:  # deprecated boto3.invoke route.
                if model.startswith("anthropic.claude-3"):
                    optional_params = (
                        litellm.AmazonAnthropicClaude3Config().map_openai_params(
                            non_default_params=non_default_params,
                            optional_params=optional_params,
                        )
                    )
                else:
                    optional_params = litellm.AmazonAnthropicConfig().map_openai_params(
                        non_default_params=non_default_params,
                        optional_params=optional_params,
                    )
            else:  # bedrock httpx route
                optional_params = litellm.AmazonConverseConfig().map_openai_params(
                    model=model,
                    non_default_params=non_default_params,
                    optional_params=optional_params,
                    drop_params=(
                        drop_params
                        if drop_params is not None and isinstance(drop_params, bool)
                        else False
                    ),
                )
        elif "amazon" in model:  # amazon titan llms
            _check_valid_arg(supported_params=supported_params)
            # see https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=titan-large
            if max_tokens is not None:
                optional_params["maxTokenCount"] = max_tokens
            if temperature is not None:
                optional_params["temperature"] = temperature
            if stop is not None:
                filtered_stop = _map_and_modify_arg(
                    {"stop": stop}, provider="bedrock", model=model
                )
                optional_params["stopSequences"] = filtered_stop["stop"]
            if top_p is not None:
                optional_params["topP"] = top_p
            if stream:
                optional_params["stream"] = stream
        elif "meta" in model:  # amazon / meta llms
            _check_valid_arg(supported_params=supported_params)
            # see https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=titan-large
            if max_tokens is not None:
                optional_params["max_gen_len"] = max_tokens
            if temperature is not None:
                optional_params["temperature"] = temperature
            if top_p is not None:
                optional_params["top_p"] = top_p
            if stream:
                optional_params["stream"] = stream
        elif "cohere" in model:  # cohere models on bedrock
            _check_valid_arg(supported_params=supported_params)
            # handle cohere params
            if stream:
                optional_params["stream"] = stream
            if temperature is not None:
                optional_params["temperature"] = temperature
            if max_tokens is not None:
                optional_params["max_tokens"] = max_tokens
        elif "mistral" in model:
            _check_valid_arg(supported_params=supported_params)
            # mistral params on bedrock
            # \"max_tokens\":400,\"temperature\":0.7,\"top_p\":0.7,\"stop\":[\"\\\\n\\\\nHuman:\"]}"
            if max_tokens is not None:
                optional_params["max_tokens"] = max_tokens
            if temperature is not None:
                optional_params["temperature"] = temperature
            if top_p is not None:
                optional_params["top_p"] = top_p
            if stop is not None:
                optional_params["stop"] = stop
            if stream is not None:
                optional_params["stream"] = stream
    elif custom_llm_provider == "aleph_alpha":
        supported_params = [
            "max_tokens",
            "stream",
            "top_p",
            "temperature",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "stop",
        ]
        _check_valid_arg(supported_params=supported_params)
        if max_tokens is not None:
            optional_params["maximum_tokens"] = max_tokens
        if stream:
            optional_params["stream"] = stream
        if temperature is not None:
            optional_params["temperature"] = temperature
        if top_p is not None:
            optional_params["top_p"] = top_p
        if presence_penalty is not None:
            optional_params["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            optional_params["frequency_penalty"] = frequency_penalty
        if n is not None:
            optional_params["n"] = n
        if stop is not None:
            optional_params["stop_sequences"] = stop
    elif custom_llm_provider == "cloudflare":
        # https://developers.cloudflare.com/workers-ai/models/text-generation/#input
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)

        if max_tokens is not None:
            optional_params["max_tokens"] = max_tokens
        if stream is not None:
            optional_params["stream"] = stream
    elif custom_llm_provider == "ollama":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)

        if max_tokens is not None:
            optional_params["num_predict"] = max_tokens
        if stream:
            optional_params["stream"] = stream
        if temperature is not None:
            optional_params["temperature"] = temperature
        if seed is not None:
            optional_params["seed"] = seed
        if top_p is not None:
            optional_params["top_p"] = top_p
        if frequency_penalty is not None:
            optional_params["repeat_penalty"] = frequency_penalty
        if stop is not None:
            optional_params["stop"] = stop
        if response_format is not None and response_format["type"] == "json_object":
            optional_params["format"] = "json"
    elif custom_llm_provider == "ollama_chat":
        supported_params = litellm.OllamaChatConfig().get_supported_openai_params()

        _check_valid_arg(supported_params=supported_params)

        optional_params = litellm.OllamaChatConfig().map_openai_params(
            non_default_params=non_default_params, optional_params=optional_params
        )
    elif custom_llm_provider == "nlp_cloud":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)

        if max_tokens is not None:
            optional_params["max_length"] = max_tokens
        if stream:
            optional_params["stream"] = stream
        if temperature is not None:
            optional_params["temperature"] = temperature
        if top_p is not None:
            optional_params["top_p"] = top_p
        if presence_penalty is not None:
            optional_params["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            optional_params["frequency_penalty"] = frequency_penalty
        if n is not None:
            optional_params["num_return_sequences"] = n
        if stop is not None:
            optional_params["stop_sequences"] = stop
    elif custom_llm_provider == "petals":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)
        # max_new_tokens=1,temperature=0.9, top_p=0.6
        if max_tokens is not None:
            optional_params["max_new_tokens"] = max_tokens
        if temperature is not None:
            optional_params["temperature"] = temperature
        if top_p is not None:
            optional_params["top_p"] = top_p
        if stream:
            optional_params["stream"] = stream
    elif custom_llm_provider == "deepinfra":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.DeepInfraConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
        )
    elif custom_llm_provider == "perplexity":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)
        if temperature is not None:
            if (
                temperature == 0 and model == "mistral-7b-instruct"
            ):  # this model does no support temperature == 0
                temperature = 0.0001  # close to 0
            optional_params["temperature"] = temperature
        if top_p:
            optional_params["top_p"] = top_p
        if stream:
            optional_params["stream"] = stream
        if max_tokens:
            optional_params["max_tokens"] = max_tokens
        if presence_penalty:
            optional_params["presence_penalty"] = presence_penalty
        if frequency_penalty:
            optional_params["frequency_penalty"] = frequency_penalty
    elif custom_llm_provider == "anyscale":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        if model in [
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ]:
            supported_params += [  # type: ignore
                "functions",
                "function_call",
                "tools",
                "tool_choice",
                "response_format",
            ]
        _check_valid_arg(supported_params=supported_params)
        optional_params = non_default_params
        if temperature is not None:
            if temperature == 0 and model in [
                "mistralai/Mistral-7B-Instruct-v0.1",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
            ]:  # this model does no support temperature == 0
                temperature = 0.0001  # close to 0
            optional_params["temperature"] = temperature
        if top_p:
            optional_params["top_p"] = top_p
        if stream:
            optional_params["stream"] = stream
        if max_tokens:
            optional_params["max_tokens"] = max_tokens
    elif custom_llm_provider == "mistral":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.MistralConfig().map_openai_params(
            non_default_params=non_default_params, optional_params=optional_params
        )
    elif custom_llm_provider == "databricks":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.DatabricksConfig().map_openai_params(
            non_default_params=non_default_params, optional_params=optional_params
        )
    elif custom_llm_provider == "groq":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)

        if temperature is not None:
            optional_params["temperature"] = temperature
        if max_tokens is not None:
            optional_params["max_tokens"] = max_tokens
        if top_p is not None:
            optional_params["top_p"] = top_p
        if stream is not None:
            optional_params["stream"] = stream
        if stop is not None:
            optional_params["stop"] = stop
        if tools is not None:
            optional_params["tools"] = tools
        if tool_choice is not None:
            optional_params["tool_choice"] = tool_choice
        if response_format is not None:
            optional_params["response_format"] = response_format
        if seed is not None:
            optional_params["seed"] = seed

    elif custom_llm_provider == "deepseek":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)

        if frequency_penalty is not None:
            optional_params["frequency_penalty"] = frequency_penalty
        if max_tokens is not None:
            optional_params["max_tokens"] = max_tokens
        if presence_penalty is not None:
            optional_params["presence_penalty"] = presence_penalty
        if stop is not None:
            optional_params["stop"] = stop
        if stream is not None:
            optional_params["stream"] = stream
        if temperature is not None:
            optional_params["temperature"] = temperature
        if logprobs is not None:
            optional_params["logprobs"] = logprobs
        if top_logprobs is not None:
            optional_params["top_logprobs"] = top_logprobs
    elif custom_llm_provider == "openrouter":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)

        if functions is not None:
            optional_params["functions"] = functions
        if function_call is not None:
            optional_params["function_call"] = function_call
        if temperature is not None:
            optional_params["temperature"] = temperature
        if top_p is not None:
            optional_params["top_p"] = top_p
        if n is not None:
            optional_params["n"] = n
        if stream is not None:
            optional_params["stream"] = stream
        if stop is not None:
            optional_params["stop"] = stop
        if max_tokens is not None:
            optional_params["max_tokens"] = max_tokens
        if presence_penalty is not None:
            optional_params["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            optional_params["frequency_penalty"] = frequency_penalty
        if logit_bias is not None:
            optional_params["logit_bias"] = logit_bias
        if user is not None:
            optional_params["user"] = user
        if response_format is not None:
            optional_params["response_format"] = response_format
        if seed is not None:
            optional_params["seed"] = seed
        if tools is not None:
            optional_params["tools"] = tools
        if tool_choice is not None:
            optional_params["tool_choice"] = tool_choice
        if max_retries is not None:
            optional_params["max_retries"] = max_retries

        # OpenRouter-only parameters
        extra_body = {}
        transforms = passed_params.pop("transforms", None)
        models = passed_params.pop("models", None)
        route = passed_params.pop("route", None)
        if transforms is not None:
            extra_body["transforms"] = transforms
        if models is not None:
            extra_body["models"] = models
        if route is not None:
            extra_body["route"] = route
        optional_params["extra_body"] = (
            extra_body  # openai client supports `extra_body` param
        )
    elif custom_llm_provider == "watsonx":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        _check_valid_arg(supported_params=supported_params)
        if max_tokens is not None:
            optional_params["max_new_tokens"] = max_tokens
        if stream:
            optional_params["stream"] = stream
        if temperature is not None:
            optional_params["temperature"] = temperature
        if top_p is not None:
            optional_params["top_p"] = top_p
        if frequency_penalty is not None:
            optional_params["repetition_penalty"] = frequency_penalty
        if seed is not None:
            optional_params["random_seed"] = seed
        if stop is not None:
            optional_params["stop_sequences"] = stop

        # WatsonX-only parameters
        extra_body = {}
        if "decoding_method" in passed_params:
            extra_body["decoding_method"] = passed_params.pop("decoding_method")
        if "min_tokens" in passed_params or "min_new_tokens" in passed_params:
            extra_body["min_new_tokens"] = passed_params.pop(
                "min_tokens", passed_params.pop("min_new_tokens")
            )
        if "top_k" in passed_params:
            extra_body["top_k"] = passed_params.pop("top_k")
        if "truncate_input_tokens" in passed_params:
            extra_body["truncate_input_tokens"] = passed_params.pop(
                "truncate_input_tokens"
            )
        if "length_penalty" in passed_params:
            extra_body["length_penalty"] = passed_params.pop("length_penalty")
        if "time_limit" in passed_params:
            extra_body["time_limit"] = passed_params.pop("time_limit")
        if "return_options" in passed_params:
            extra_body["return_options"] = passed_params.pop("return_options")
        optional_params["extra_body"] = (
            extra_body  # openai client supports `extra_body` param
        )
    elif custom_llm_provider == "openai":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider="openai"
        )
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.OpenAIConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
        )
    elif custom_llm_provider == "azure":
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider="azure"
        )
        _check_valid_arg(supported_params=supported_params)
        api_version = (
            api_version or litellm.api_version or get_secret("AZURE_API_VERSION")
        )
        optional_params = litellm.AzureOpenAIConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            api_version=api_version,  # type: ignore
            drop_params=drop_params,
        )
    else:  # assume passing in params for text-completion openai
        supported_params = get_supported_openai_params(
            model=model, custom_llm_provider="custom_openai"
        )
        _check_valid_arg(supported_params=supported_params)
        if functions is not None:
            optional_params["functions"] = functions
        if function_call is not None:
            optional_params["function_call"] = function_call
        if temperature is not None:
            optional_params["temperature"] = temperature
        if top_p is not None:
            optional_params["top_p"] = top_p
        if n is not None:
            optional_params["n"] = n
        if stream is not None:
            optional_params["stream"] = stream
        if stream_options is not None:
            optional_params["stream_options"] = stream_options
        if stop is not None:
            optional_params["stop"] = stop
        if max_tokens is not None:
            optional_params["max_tokens"] = max_tokens
        if presence_penalty is not None:
            optional_params["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            optional_params["frequency_penalty"] = frequency_penalty
        if logit_bias is not None:
            optional_params["logit_bias"] = logit_bias
        if user is not None:
            optional_params["user"] = user
        if response_format is not None:
            optional_params["response_format"] = response_format
        if seed is not None:
            optional_params["seed"] = seed
        if tools is not None:
            optional_params["tools"] = tools
        if tool_choice is not None:
            optional_params["tool_choice"] = tool_choice
        if max_retries is not None:
            optional_params["max_retries"] = max_retries
        if logprobs is not None:
            optional_params["logprobs"] = logprobs
        if top_logprobs is not None:
            optional_params["top_logprobs"] = top_logprobs
        if extra_headers is not None:
            optional_params["extra_headers"] = extra_headers
    if custom_llm_provider in ["openai", "azure"] + litellm.openai_compatible_providers:
        # for openai, azure we should pass the extra/passed params within `extra_body` https://github.com/openai/openai-python/blob/ac33853ba10d13ac149b1fa3ca6dba7d613065c9/src/openai/resources/models.py#L46
        extra_body = passed_params.pop("extra_body", {})
        for k in passed_params.keys():
            if k not in default_params.keys():
                extra_body[k] = passed_params[k]
        optional_params.setdefault("extra_body", {})
        optional_params["extra_body"] = {**optional_params["extra_body"], **extra_body}
    else:
        # if user passed in non-default kwargs for specific providers/models, pass them along
        for k in passed_params.keys():
            if k not in default_params.keys():
                optional_params[k] = passed_params[k]
    print_verbose(f"Final returned optional params: {optional_params}")
    return optional_params


def get_non_default_params(passed_params: dict) -> dict:
    default_params = {
        "functions": None,
        "function_call": None,
        "temperature": None,
        "top_p": None,
        "n": None,
        "stream": None,
        "stream_options": None,
        "stop": None,
        "max_tokens": None,
        "presence_penalty": None,
        "frequency_penalty": None,
        "logit_bias": None,
        "user": None,
        "model": None,
        "custom_llm_provider": "",
        "response_format": None,
        "seed": None,
        "tools": None,
        "tool_choice": None,
        "max_retries": None,
        "logprobs": None,
        "top_logprobs": None,
        "extra_headers": None,
    }
    # filter out those parameters that were passed with non-default values
    non_default_params = {
        k: v
        for k, v in passed_params.items()
        if (
            k != "model"
            and k != "custom_llm_provider"
            and k in default_params
            and v != default_params[k]
        )
    }

    return non_default_params


def calculate_max_parallel_requests(
    max_parallel_requests: Optional[int],
    rpm: Optional[int],
    tpm: Optional[int],
    default_max_parallel_requests: Optional[int],
) -> Optional[int]:
    """
    Returns the max parallel requests to send to a deployment.

    Used in semaphore for async requests on router.

    Parameters:
    - max_parallel_requests - Optional[int] - max_parallel_requests allowed for that deployment
    - rpm - Optional[int] - requests per minute allowed for that deployment
    - tpm - Optional[int] - tokens per minute allowed for that deployment
    - default_max_parallel_requests - Optional[int] - default_max_parallel_requests allowed for any deployment

    Returns:
    - int or None (if all params are None)

    Order:
    max_parallel_requests > rpm > tpm / 6 (azure formula) > default max_parallel_requests

    Azure RPM formula:
    6 rpm per 1000 TPM
    https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits


    """
    if max_parallel_requests is not None:
        return max_parallel_requests
    elif rpm is not None:
        return rpm
    elif tpm is not None:
        calculated_rpm = int(tpm / 1000 / 6)
        if calculated_rpm == 0:
            calculated_rpm = 1
        return calculated_rpm
    elif default_max_parallel_requests is not None:
        return default_max_parallel_requests
    return None


def _get_order_filtered_deployments(healthy_deployments: List[Dict]) -> List:
    min_order = min(
        (
            deployment["litellm_params"]["order"]
            for deployment in healthy_deployments
            if "order" in deployment["litellm_params"]
        ),
        default=None,
    )

    if min_order is not None:
        filtered_deployments = [
            deployment
            for deployment in healthy_deployments
            if deployment["litellm_params"].get("order") == min_order
        ]

        return filtered_deployments
    return healthy_deployments


def _get_model_region(
    custom_llm_provider: str, litellm_params: LiteLLM_Params
) -> Optional[str]:
    """
    Return the region for a model, for a given provider
    """
    if custom_llm_provider == "vertex_ai":
        # check 'vertex_location'
        vertex_ai_location = (
            litellm_params.vertex_location
            or litellm.vertex_location
            or get_secret("VERTEXAI_LOCATION")
            or get_secret("VERTEX_LOCATION")
        )
        if vertex_ai_location is not None and isinstance(vertex_ai_location, str):
            return vertex_ai_location
    elif custom_llm_provider == "bedrock":
        aws_region_name = litellm_params.aws_region_name
        if aws_region_name is not None:
            return aws_region_name
    elif custom_llm_provider == "watsonx":
        watsonx_region_name = litellm_params.watsonx_region_name
        if watsonx_region_name is not None:
            return watsonx_region_name
    return litellm_params.region_name


def _is_region_eu(litellm_params: LiteLLM_Params) -> bool:
    """
    Return true/false if a deployment is in the EU
    """
    if litellm_params.region_name == "eu":
        return True

    ## ELSE ##
    """
    - get provider 
    - get provider regions 
    - return true if given region (get_provider_region) in eu region (config.get_eu_regions())
    """
    model, custom_llm_provider, _, _ = litellm.get_llm_provider(
        model=litellm_params.model, litellm_params=litellm_params
    )

    model_region = _get_model_region(
        custom_llm_provider=custom_llm_provider, litellm_params=litellm_params
    )

    if model_region is None:
        return False

    if custom_llm_provider == "azure":
        eu_regions = litellm.AzureOpenAIConfig().get_eu_regions()
    elif custom_llm_provider == "vertex_ai":
        eu_regions = litellm.VertexAIConfig().get_eu_regions()
    elif custom_llm_provider == "bedrock":
        eu_regions = litellm.AmazonBedrockGlobalConfig().get_eu_regions()
    elif custom_llm_provider == "watsonx":
        eu_regions = litellm.IBMWatsonXAIConfig().get_eu_regions()
    else:
        return False

    for region in eu_regions:
        if region in model_region.lower():
            return True
    return False


def get_model_region(
    litellm_params: LiteLLM_Params, mode: Optional[str]
) -> Optional[str]:
    """
    Pass the litellm params for an azure model, and get back the region
    """
    if (
        "azure" in litellm_params.model
        and isinstance(litellm_params.api_key, str)
        and isinstance(litellm_params.api_base, str)
    ):
        _model = litellm_params.model.replace("azure/", "")
        response: dict = litellm.AzureChatCompletion().get_headers(
            model=_model,
            api_key=litellm_params.api_key,
            api_base=litellm_params.api_base,
            api_version=litellm_params.api_version or "2023-07-01-preview",
            timeout=10,
            mode=mode or "chat",
        )

        region: Optional[str] = response.get("x-ms-region", None)
        return region
    return None


def get_api_base(
    model: str, optional_params: Union[dict, LiteLLM_Params]
) -> Optional[str]:
    """
    Returns the api base used for calling the model.

    Parameters:
    - model: str - the model passed to litellm.completion()
    - optional_params - the 'litellm_params' in router.completion *OR* additional params passed to litellm.completion - eg. api_base, api_key, etc. See `LiteLLM_Params` - https://github.com/BerriAI/litellm/blob/f09e6ba98d65e035a79f73bc069145002ceafd36/litellm/router.py#L67

    Returns:
    - string (api_base) or None

    Example:
    ```
    from litellm import get_api_base

    get_api_base(model="gemini/gemini-pro")
    ```
    """

    try:
        if isinstance(optional_params, LiteLLM_Params):
            _optional_params = optional_params
        elif "model" in optional_params:
            _optional_params = LiteLLM_Params(**optional_params)
        else:  # prevent needing to copy and pop the dict
            _optional_params = LiteLLM_Params(
                model=model, **optional_params
            )  # convert to pydantic object
    except Exception as e:
        verbose_logger.debug("Error occurred in getting api base - {}".format(str(e)))
        return None
    # get llm provider

    if _optional_params.api_base is not None:
        return _optional_params.api_base

    if litellm.model_alias_map and model in litellm.model_alias_map:
        model = litellm.model_alias_map[model]
    try:
        model, custom_llm_provider, dynamic_api_key, dynamic_api_base = (
            get_llm_provider(
                model=model,
                custom_llm_provider=_optional_params.custom_llm_provider,
                api_base=_optional_params.api_base,
                api_key=_optional_params.api_key,
            )
        )
    except Exception as e:
        verbose_logger.debug("Error occurred in getting api base - {}".format(str(e)))
        custom_llm_provider = None
        dynamic_api_key = None
        dynamic_api_base = None

    if dynamic_api_base is not None:
        return dynamic_api_base

    if (
        _optional_params.vertex_location is not None
        and _optional_params.vertex_project is not None
    ):
        _api_base = "{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:streamGenerateContent".format(
            _optional_params.vertex_location,
            _optional_params.vertex_project,
            _optional_params.vertex_location,
            model,
        )
        return _api_base

    if custom_llm_provider is None:
        return None

    if custom_llm_provider == "gemini":
        _api_base = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent".format(
            model
        )
        return _api_base
    elif custom_llm_provider == "openai":
        _api_base = "https://api.openai.com"
        return _api_base
    return None


def get_first_chars_messages(kwargs: dict) -> str:
    try:
        _messages = kwargs.get("messages")
        _messages = str(_messages)[:100]
        return _messages
    except:
        return ""


def get_supported_openai_params(
    model: str,
    custom_llm_provider: str,
    request_type: Literal["chat_completion", "embeddings"] = "chat_completion",
) -> Optional[list]:
    """
    Returns the supported openai params for a given model + provider

    Example:
    ```
    get_supported_openai_params(model="anthropic.claude-3", custom_llm_provider="bedrock")
    ```

    Returns:
    - List if custom_llm_provider is mapped
    - None if unmapped
    """
    if custom_llm_provider == "bedrock":
        return litellm.AmazonConverseConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "ollama":
        return litellm.OllamaConfig().get_supported_openai_params()
    elif custom_llm_provider == "ollama_chat":
        return litellm.OllamaChatConfig().get_supported_openai_params()
    elif custom_llm_provider == "anthropic":
        return litellm.AnthropicConfig().get_supported_openai_params()
    elif custom_llm_provider == "groq":
        return [
            "temperature",
            "max_tokens",
            "top_p",
            "stream",
            "stop",
            "tools",
            "tool_choice",
            "response_format",
            "seed",
        ]
    elif custom_llm_provider == "deepseek":
        return [
            # https://platform.deepseek.com/api-docs/api/create-chat-completion
            "frequency_penalty",
            "max_tokens",
            "presence_penalty",
            "stop",
            "stream",
            "temperature",
            "top_p",
            "logprobs",
            "top_logprobs",
        ]
    elif custom_llm_provider == "cohere":
        return [
            "stream",
            "temperature",
            "max_tokens",
            "logit_bias",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
        ]
    elif custom_llm_provider == "cohere_chat":
        return [
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "tools",
            "tool_choice",
            "seed",
        ]
    elif custom_llm_provider == "maritalk":
        return [
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "presence_penalty",
            "stop",
        ]
    elif custom_llm_provider == "openai":
        return litellm.OpenAIConfig().get_supported_openai_params(model=model)
    elif custom_llm_provider == "azure":
        return litellm.AzureOpenAIConfig().get_supported_openai_params()
    elif custom_llm_provider == "openrouter":
        return [
            "functions",
            "function_call",
            "temperature",
            "top_p",
            "n",
            "stream",
            "stop",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "max_retries",
        ]
    elif custom_llm_provider == "mistral":
        return litellm.MistralConfig().get_supported_openai_params()
    elif custom_llm_provider == "replicate":
        return [
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "seed",
        ]
    elif custom_llm_provider == "huggingface":
        return litellm.HuggingfaceConfig().get_supported_openai_params()
    elif custom_llm_provider == "together_ai":
        return [
            "stream",
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "frequency_penalty",
            "tools",
            "tool_choice",
        ]
    elif custom_llm_provider == "ai21":
        return [
            "stream",
            "n",
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "frequency_penalty",
            "presence_penalty",
        ]
    elif custom_llm_provider == "databricks":
        if request_type == "chat_completion":
            return litellm.DatabricksConfig().get_supported_openai_params()
        elif request_type == "embeddings":
            return litellm.DatabricksEmbeddingConfig().get_supported_openai_params()
    elif custom_llm_provider == "palm" or custom_llm_provider == "gemini":
        return ["temperature", "top_p", "stream", "n", "stop", "max_tokens"]
    elif custom_llm_provider == "vertex_ai":
        return litellm.VertexAIConfig().get_supported_openai_params()
    elif custom_llm_provider == "sagemaker":
        return ["stream", "temperature", "max_tokens", "top_p", "stop", "n"]
    elif custom_llm_provider == "aleph_alpha":
        return [
            "max_tokens",
            "stream",
            "top_p",
            "temperature",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "stop",
        ]
    elif custom_llm_provider == "cloudflare":
        return ["max_tokens", "stream"]
    elif custom_llm_provider == "nlp_cloud":
        return [
            "max_tokens",
            "stream",
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "stop",
        ]
    elif custom_llm_provider == "petals":
        return ["max_tokens", "temperature", "top_p", "stream"]
    elif custom_llm_provider == "deepinfra":
        return litellm.DeepInfraConfig().get_supported_openai_params()
    elif custom_llm_provider == "perplexity":
        return [
            "temperature",
            "top_p",
            "stream",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
        ]
    elif custom_llm_provider == "anyscale":
        return [
            "temperature",
            "top_p",
            "stream",
            "max_tokens",
            "stop",
            "frequency_penalty",
            "presence_penalty",
        ]
    elif custom_llm_provider == "watsonx":
        return litellm.IBMWatsonXAIConfig().get_supported_openai_params()
    elif custom_llm_provider == "custom_openai" or "text-completion-openai":
        return [
            "functions",
            "function_call",
            "temperature",
            "top_p",
            "n",
            "stream",
            "stream_options",
            "stop",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "max_retries",
            "logprobs",
            "top_logprobs",
            "extra_headers",
        ]
    return None


def get_formatted_prompt(
    data: dict,
    call_type: Literal[
        "completion",
        "embedding",
        "image_generation",
        "audio_transcription",
        "moderation",
        "text_completion",
    ],
) -> str:
    """
    Extracts the prompt from the input data based on the call type.

    Returns a string.
    """
    prompt = ""
    if call_type == "completion":
        for m in data["messages"]:
            if "content" in m and isinstance(m["content"], str):
                prompt += m["content"]
    elif call_type == "text_completion":
        prompt = data["prompt"]
    elif call_type == "embedding" or call_type == "moderation":
        if isinstance(data["input"], str):
            prompt = data["input"]
        elif isinstance(data["input"], list):
            for m in data["input"]:
                prompt += m
    elif call_type == "image_generation":
        prompt = data["prompt"]
    elif call_type == "audio_transcription":
        if "prompt" in data:
            prompt = data["prompt"]
    return prompt


def _is_non_openai_azure_model(model: str) -> bool:
    try:
        model_name = model.split("/", 1)[1]
        if (
            model_name in litellm.cohere_chat_models
            or f"mistral/{model_name}" in litellm.mistral_chat_models
        ):
            return True
    except:
        return False
    return False


def get_llm_provider(
    model: str,
    custom_llm_provider: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    litellm_params: Optional[LiteLLM_Params] = None,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Returns the provider for a given model name - e.g. 'azure/chatgpt-v-2' -> 'azure'

    For router -> Can also give the whole litellm param dict -> this function will extract the relevant details

    Raises Error - if unable to map model to a provider
    """
    try:
        ## IF LITELLM PARAMS GIVEN ##
        if litellm_params is not None:
            assert (
                custom_llm_provider is None and api_base is None and api_key is None
            ), "Either pass in litellm_params or the custom_llm_provider/api_base/api_key. Otherwise, these values will be overriden."
            custom_llm_provider = litellm_params.custom_llm_provider
            api_base = litellm_params.api_base
            api_key = litellm_params.api_key

        dynamic_api_key = None
        # check if llm provider provided
        # AZURE AI-Studio Logic - Azure AI Studio supports AZURE/Cohere
        # If User passes azure/command-r-plus -> we should send it to cohere_chat/command-r-plus
        if model.split("/", 1)[0] == "azure":
            if _is_non_openai_azure_model(model):
                custom_llm_provider = "openai"
                return model, custom_llm_provider, dynamic_api_key, api_base

        if custom_llm_provider:
            return model, custom_llm_provider, dynamic_api_key, api_base

        if api_key and api_key.startswith("os.environ/"):
            dynamic_api_key = get_secret(api_key)
        # check if llm provider part of model name
        if (
            model.split("/", 1)[0] in litellm.provider_list
            and model.split("/", 1)[0] not in litellm.model_list
            and len(model.split("/"))
            > 1  # handle edge case where user passes in `litellm --model mistral` https://github.com/BerriAI/litellm/issues/1351
        ):
            custom_llm_provider = model.split("/", 1)[0]
            model = model.split("/", 1)[1]
            if custom_llm_provider == "perplexity":
                # perplexity is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.perplexity.ai
                api_base = "https://api.perplexity.ai"
                dynamic_api_key = get_secret("PERPLEXITYAI_API_KEY")
            elif custom_llm_provider == "anyscale":
                # anyscale is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
                api_base = "https://api.endpoints.anyscale.com/v1"
                dynamic_api_key = get_secret("ANYSCALE_API_KEY")
            elif custom_llm_provider == "deepinfra":
                # deepinfra is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
                api_base = "https://api.deepinfra.com/v1/openai"
                dynamic_api_key = get_secret("DEEPINFRA_API_KEY")
            elif custom_llm_provider == "groq":
                # groq is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.groq.com/openai/v1
                api_base = "https://api.groq.com/openai/v1"
                dynamic_api_key = get_secret("GROQ_API_KEY")
            elif custom_llm_provider == "deepseek":
                # deepseek is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.deepseek.com/v1
                api_base = "https://api.deepseek.com/v1"
                dynamic_api_key = get_secret("DEEPSEEK_API_KEY")
            elif custom_llm_provider == "fireworks_ai":
                # fireworks is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.fireworks.ai/inference/v1
                if not model.startswith("accounts/fireworks/models"):
                    model = f"accounts/fireworks/models/{model}"
                api_base = "https://api.fireworks.ai/inference/v1"
                dynamic_api_key = (
                    get_secret("FIREWORKS_API_KEY")
                    or get_secret("FIREWORKS_AI_API_KEY")
                    or get_secret("FIREWORKSAI_API_KEY")
                    or get_secret("FIREWORKS_AI_TOKEN")
                )
            elif custom_llm_provider == "mistral":
                # mistral is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.mistral.ai
                api_base = (
                    api_base
                    or get_secret("MISTRAL_AZURE_API_BASE")  # for Azure AI Mistral
                    or "https://api.mistral.ai/v1"
                )  # type: ignore

                # if api_base does not end with /v1 we add it
                if api_base is not None and not api_base.endswith(
                    "/v1"
                ):  # Mistral always needs a /v1 at the end
                    api_base = api_base + "/v1"
                dynamic_api_key = (
                    api_key
                    or get_secret("MISTRAL_AZURE_API_KEY")  # for Azure AI Mistral
                    or get_secret("MISTRAL_API_KEY")
                )
            elif custom_llm_provider == "voyage":
                # voyage is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.voyageai.com/v1
                api_base = "https://api.voyageai.com/v1"
                dynamic_api_key = get_secret("VOYAGE_API_KEY")
            elif custom_llm_provider == "together_ai":
                api_base = "https://api.together.xyz/v1"
                dynamic_api_key = (
                    get_secret("TOGETHER_API_KEY")
                    or get_secret("TOGETHER_AI_API_KEY")
                    or get_secret("TOGETHERAI_API_KEY")
                    or get_secret("TOGETHER_AI_TOKEN")
                )
            if api_base is not None and not isinstance(api_base, str):
                raise Exception(
                    "api base needs to be a string. api_base={}".format(api_base)
                )
            if dynamic_api_key is not None and not isinstance(dynamic_api_key, str):
                raise Exception(
                    "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                        dynamic_api_key
                    )
                )
            return model, custom_llm_provider, dynamic_api_key, api_base
        elif model.split("/", 1)[0] in litellm.provider_list:
            custom_llm_provider = model.split("/", 1)[0]
            model = model.split("/", 1)[1]
            if api_base is not None and not isinstance(api_base, str):
                raise Exception(
                    "api base needs to be a string. api_base={}".format(api_base)
                )
            if dynamic_api_key is not None and not isinstance(dynamic_api_key, str):
                raise Exception(
                    "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                        dynamic_api_key
                    )
                )
            return model, custom_llm_provider, dynamic_api_key, api_base
        # check if api base is a known openai compatible endpoint
        if api_base:
            for endpoint in litellm.openai_compatible_endpoints:
                if endpoint in api_base:
                    if endpoint == "api.perplexity.ai":
                        custom_llm_provider = "perplexity"
                        dynamic_api_key = get_secret("PERPLEXITYAI_API_KEY")
                    elif endpoint == "api.endpoints.anyscale.com/v1":
                        custom_llm_provider = "anyscale"
                        dynamic_api_key = get_secret("ANYSCALE_API_KEY")
                    elif endpoint == "api.deepinfra.com/v1/openai":
                        custom_llm_provider = "deepinfra"
                        dynamic_api_key = get_secret("DEEPINFRA_API_KEY")
                    elif endpoint == "api.mistral.ai/v1":
                        custom_llm_provider = "mistral"
                        dynamic_api_key = get_secret("MISTRAL_API_KEY")
                    elif endpoint == "api.groq.com/openai/v1":
                        custom_llm_provider = "groq"
                        dynamic_api_key = get_secret("GROQ_API_KEY")
                    elif endpoint == "api.deepseek.com/v1":
                        custom_llm_provider = "deepseek"
                        dynamic_api_key = get_secret("DEEPSEEK_API_KEY")

                    if api_base is not None and not isinstance(api_base, str):
                        raise Exception(
                            "api base needs to be a string. api_base={}".format(
                                api_base
                            )
                        )
                    if dynamic_api_key is not None and not isinstance(
                        dynamic_api_key, str
                    ):
                        raise Exception(
                            "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                                dynamic_api_key
                            )
                        )
                    return model, custom_llm_provider, dynamic_api_key, api_base  # type: ignore

        # check if model in known model provider list  -> for huggingface models, raise exception as they don't have a fixed provider (can be togetherai, anyscale, baseten, runpod, et.)
        ## openai - chatcompletion + text completion
        if (
            model in litellm.open_ai_chat_completion_models
            or "ft:gpt-3.5-turbo" in model
            or model in litellm.openai_image_generation_models
        ):
            custom_llm_provider = "openai"
        elif model in litellm.open_ai_text_completion_models:
            custom_llm_provider = "text-completion-openai"
        ## anthropic
        elif model in litellm.anthropic_models:
            custom_llm_provider = "anthropic"
        ## cohere
        elif model in litellm.cohere_models or model in litellm.cohere_embedding_models:
            custom_llm_provider = "cohere"
        ## cohere chat models
        elif model in litellm.cohere_chat_models:
            custom_llm_provider = "cohere_chat"
        ## replicate
        elif model in litellm.replicate_models or (":" in model and len(model) > 64):
            model_parts = model.split(":")
            if (
                len(model_parts) > 1 and len(model_parts[1]) == 64
            ):  ## checks if model name has a 64 digit code - e.g. "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
                custom_llm_provider = "replicate"
            elif model in litellm.replicate_models:
                custom_llm_provider = "replicate"
        ## openrouter
        elif model in litellm.openrouter_models:
            custom_llm_provider = "openrouter"
        ## openrouter
        elif model in litellm.maritalk_models:
            custom_llm_provider = "maritalk"
        ## vertex - text + chat + language (gemini) models
        elif (
            model in litellm.vertex_chat_models
            or model in litellm.vertex_code_chat_models
            or model in litellm.vertex_text_models
            or model in litellm.vertex_code_text_models
            or model in litellm.vertex_language_models
            or model in litellm.vertex_embedding_models
            or model in litellm.vertex_vision_models
        ):
            custom_llm_provider = "vertex_ai"
        ## ai21
        elif model in litellm.ai21_models:
            custom_llm_provider = "ai21"
        ## aleph_alpha
        elif model in litellm.aleph_alpha_models:
            custom_llm_provider = "aleph_alpha"
        ## baseten
        elif model in litellm.baseten_models:
            custom_llm_provider = "baseten"
        ## nlp_cloud
        elif model in litellm.nlp_cloud_models:
            custom_llm_provider = "nlp_cloud"
        ## petals
        elif model in litellm.petals_models:
            custom_llm_provider = "petals"
        ## bedrock
        elif (
            model in litellm.bedrock_models or model in litellm.bedrock_embedding_models
        ):
            custom_llm_provider = "bedrock"
        elif model in litellm.watsonx_models:
            custom_llm_provider = "watsonx"
        # openai embeddings
        elif model in litellm.open_ai_embedding_models:
            custom_llm_provider = "openai"
        if custom_llm_provider is None or custom_llm_provider == "":
            if litellm.suppress_debug_info == False:
                print()  # noqa
                print(  # noqa
                    "\033[1;31mProvider List: https://docs.litellm.ai/docs/providers\033[0m"  # noqa
                )  # noqa
                print()  # noqa
            error_str = f"LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model={model}\n Pass model as E.g. For 'Huggingface' inference endpoints pass in `completion(model='huggingface/starcoder',..)` Learn more: https://docs.litellm.ai/docs/providers"
            # maps to openai.NotFoundError, this is raised when openai does not recognize the llm
            raise litellm.exceptions.BadRequestError(  # type: ignore
                message=error_str,
                model=model,
                response=httpx.Response(
                    status_code=400,
                    content=error_str,
                    request=httpx.Request(method="completion", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
                llm_provider="",
            )
        if api_base is not None and not isinstance(api_base, str):
            raise Exception(
                "api base needs to be a string. api_base={}".format(api_base)
            )
        if dynamic_api_key is not None and not isinstance(dynamic_api_key, str):
            raise Exception(
                "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                    dynamic_api_key
                )
            )
        return model, custom_llm_provider, dynamic_api_key, api_base
    except Exception as e:
        if isinstance(e, litellm.exceptions.BadRequestError):
            raise e
        else:
            error_str = (
                f"GetLLMProvider Exception - {str(e)}\n\noriginal model: {model}"
            )
            raise litellm.exceptions.BadRequestError(  # type: ignore
                message=f"GetLLMProvider Exception - {str(e)}\n\noriginal model: {model}",
                model=model,
                response=httpx.Response(
                    status_code=400,
                    content=error_str,
                    request=httpx.Request(method="completion", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
                llm_provider="",
            )


def get_api_key(llm_provider: str, dynamic_api_key: Optional[str]):
    api_key = dynamic_api_key or litellm.api_key
    # openai
    if llm_provider == "openai" or llm_provider == "text-completion-openai":
        api_key = api_key or litellm.openai_key or get_secret("OPENAI_API_KEY")
    # anthropic
    elif llm_provider == "anthropic":
        api_key = api_key or litellm.anthropic_key or get_secret("ANTHROPIC_API_KEY")
    # ai21
    elif llm_provider == "ai21":
        api_key = api_key or litellm.ai21_key or get_secret("AI211_API_KEY")
    # aleph_alpha
    elif llm_provider == "aleph_alpha":
        api_key = (
            api_key or litellm.aleph_alpha_key or get_secret("ALEPH_ALPHA_API_KEY")
        )
    # baseten
    elif llm_provider == "baseten":
        api_key = api_key or litellm.baseten_key or get_secret("BASETEN_API_KEY")
    # cohere
    elif llm_provider == "cohere" or llm_provider == "cohere_chat":
        api_key = api_key or litellm.cohere_key or get_secret("COHERE_API_KEY")
    # huggingface
    elif llm_provider == "huggingface":
        api_key = (
            api_key or litellm.huggingface_key or get_secret("HUGGINGFACE_API_KEY")
        )
    # nlp_cloud
    elif llm_provider == "nlp_cloud":
        api_key = api_key or litellm.nlp_cloud_key or get_secret("NLP_CLOUD_API_KEY")
    # replicate
    elif llm_provider == "replicate":
        api_key = api_key or litellm.replicate_key or get_secret("REPLICATE_API_KEY")
    # together_ai
    elif llm_provider == "together_ai":
        api_key = (
            api_key
            or litellm.togetherai_api_key
            or get_secret("TOGETHERAI_API_KEY")
            or get_secret("TOGETHER_AI_TOKEN")
        )
    return api_key


def get_utc_datetime():
    import datetime as dt
    from datetime import datetime

    if hasattr(dt, "UTC"):
        return datetime.now(dt.UTC)  # type: ignore
    else:
        return datetime.utcnow()  # type: ignore


def get_max_tokens(model: str):
    """
    Get the maximum number of output tokens allowed for a given model.

    Parameters:
    model (str): The name of the model.

    Returns:
        int: The maximum number of tokens allowed for the given model.

    Raises:
        Exception: If the model is not mapped yet.

    Example:
        >>> get_max_tokens("gpt-4")
        8192
    """

    def _get_max_position_embeddings(model_name):
        # Construct the URL for the config.json file
        config_url = f"https://huggingface.co/{model_name}/raw/main/config.json"
        try:
            # Make the HTTP request to get the raw JSON file
            response = requests.get(config_url)
            response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)

            # Parse the JSON response
            config_json = response.json()
            # Extract and return the max_position_embeddings
            max_position_embeddings = config_json.get("max_position_embeddings")
            if max_position_embeddings is not None:
                return max_position_embeddings
            else:
                return None
        except requests.exceptions.RequestException as e:
            return None

    try:
        if model in litellm.model_cost:
            if "max_output_tokens" in litellm.model_cost[model]:
                return litellm.model_cost[model]["max_output_tokens"]
            elif "max_tokens" in litellm.model_cost[model]:
                return litellm.model_cost[model]["max_tokens"]
        model, custom_llm_provider, _, _ = get_llm_provider(model=model)
        if custom_llm_provider == "huggingface":
            max_tokens = _get_max_position_embeddings(model_name=model)
            return max_tokens
        if model in litellm.model_cost:  # check if extracted model is in model_list
            if "max_output_tokens" in litellm.model_cost[model]:
                return litellm.model_cost[model]["max_output_tokens"]
            elif "max_tokens" in litellm.model_cost[model]:
                return litellm.model_cost[model]["max_tokens"]
        else:
            raise Exception()
    except:
        raise Exception(
            f"Model {model} isn't mapped yet. Add it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
        )


def get_model_info(model: str) -> ModelInfo:
    """
    Get a dict for the maximum tokens (context window),
    input_cost_per_token, output_cost_per_token  for a given model.

    Parameters:
    model (str): The name of the model.

    Returns:
        dict: A dictionary containing the following information:
            - max_tokens (int): The maximum number of tokens allowed for the given model.
            - input_cost_per_token (float): The cost per token for input.
            - output_cost_per_token (float): The cost per token for output.
            - litellm_provider (str): The provider of the model (e.g., "openai").
            - mode (str): The mode of the model (e.g., "chat" or "completion").
            - supported_openai_params (List[str]): A list of supported OpenAI parameters for the model.

    Raises:
        Exception: If the model is not mapped yet.

    Example:
        >>> get_model_info("gpt-4")
        {
            "max_tokens": 8192,
            "input_cost_per_token": 0.00003,
            "output_cost_per_token": 0.00006,
            "litellm_provider": "openai",
            "mode": "chat",
            "supported_openai_params": ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]
        }
    """
    supported_openai_params: Union[List[str], None] = []

    def _get_max_position_embeddings(model_name):
        # Construct the URL for the config.json file
        config_url = f"https://huggingface.co/{model_name}/raw/main/config.json"

        try:
            # Make the HTTP request to get the raw JSON file
            response = requests.get(config_url)
            response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)

            # Parse the JSON response
            config_json = response.json()

            # Extract and return the max_position_embeddings
            max_position_embeddings = config_json.get("max_position_embeddings")

            if max_position_embeddings is not None:
                return max_position_embeddings
            else:
                return None
        except requests.exceptions.RequestException as e:
            return None

    try:
        azure_llms = litellm.azure_llms
        if model in azure_llms:
            model = azure_llms[model]
        ##########################
        # Get custom_llm_provider
        split_model, custom_llm_provider = model, ""
        try:
            split_model, custom_llm_provider, _, _ = get_llm_provider(model=model)
        except:
            pass
        #########################

        supported_openai_params = litellm.get_supported_openai_params(
            model=model, custom_llm_provider=custom_llm_provider
        )
        if custom_llm_provider == "huggingface":
            max_tokens = _get_max_position_embeddings(model_name=model)
            return {
                "max_tokens": max_tokens,  # type: ignore
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
                "litellm_provider": "huggingface",
                "mode": "chat",
                "supported_openai_params": supported_openai_params,
            }
        else:
            """
            Check if:
            1. 'model' in litellm.model_cost. Checks "groq/llama3-8b-8192" in  litellm.model_cost
            2. 'split_model' in litellm.model_cost. Checks "llama3-8b-8192" in litellm.model_cost
            """
            if model in litellm.model_cost:
                _model_info = litellm.model_cost[model]
                _model_info["supported_openai_params"] = supported_openai_params
                return _model_info
            if split_model in litellm.model_cost:
                _model_info = litellm.model_cost[split_model]
                _model_info["supported_openai_params"] = supported_openai_params
                return _model_info
            else:
                raise ValueError(
                    "This model isn't mapped yet. Add it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
                )
    except:
        raise Exception(
            "This model isn't mapped yet. Add it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
        )


def json_schema_type(python_type_name: str):
    """Converts standard python types to json schema types

    Parameters
    ----------
    python_type_name : str
        __name__ of type

    Returns
    -------
    str
        a standard JSON schema type, "string" if not recognized.
    """
    python_to_json_schema_types = {
        str.__name__: "string",
        int.__name__: "integer",
        float.__name__: "number",
        bool.__name__: "boolean",
        list.__name__: "array",
        dict.__name__: "object",
        "NoneType": "null",
    }

    return python_to_json_schema_types.get(python_type_name, "string")


def function_to_dict(input_function):  # noqa: C901
    """Using type hints and numpy-styled docstring,
    produce a dictionnary usable for OpenAI function calling

    Parameters
    ----------
    input_function : function
        A function with a numpy-style docstring

    Returns
    -------
    dictionnary
        A dictionnary to add to the list passed to `functions` parameter of `litellm.completion`
    """
    # Get function name and docstring
    try:
        import inspect
        from numpydoc.docscrape import NumpyDocString
        from ast import literal_eval
    except Exception as e:
        raise e

    name = input_function.__name__
    docstring = inspect.getdoc(input_function)
    numpydoc = NumpyDocString(docstring)
    description = "\n".join([s.strip() for s in numpydoc["Summary"]])

    # Get function parameters and their types from annotations and docstring
    parameters = {}
    required_params = []
    param_info = inspect.signature(input_function).parameters

    for param_name, param in param_info.items():
        if hasattr(param, "annotation"):
            param_type = json_schema_type(param.annotation.__name__)
        else:
            param_type = None
        param_description = None
        param_enum = None

        # Try to extract param description from docstring using numpydoc
        for param_data in numpydoc["Parameters"]:
            if param_data.name == param_name:
                if hasattr(param_data, "type"):
                    # replace type from docstring rather than annotation
                    param_type = param_data.type
                    if "optional" in param_type:
                        param_type = param_type.split(",")[0]
                    elif "{" in param_type:
                        # may represent a set of acceptable values
                        # translating as enum for function calling
                        try:
                            param_enum = str(list(literal_eval(param_type)))
                            param_type = "string"
                        except Exception:
                            pass
                    param_type = json_schema_type(param_type)
                param_description = "\n".join([s.strip() for s in param_data.desc])

        param_dict = {
            "type": param_type,
            "description": param_description,
            "enum": param_enum,
        }

        parameters[param_name] = dict(
            [(k, v) for k, v in param_dict.items() if isinstance(v, str)]
        )

        # Check if the parameter has no default value (i.e., it's required)
        if param.default == param.empty:
            required_params.append(param_name)

    # Create the dictionary
    result = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": parameters,
        },
    }

    # Add "required" key if there are required parameters
    if required_params:
        result["parameters"]["required"] = required_params

    return result


def load_test_model(
    model: str,
    custom_llm_provider: str = "",
    api_base: str = "",
    prompt: str = "",
    num_calls: int = 0,
    force_timeout: int = 0,
):
    test_prompt = "Hey, how's it going"
    test_calls = 100
    if prompt:
        test_prompt = prompt
    if num_calls:
        test_calls = num_calls
    messages = [[{"role": "user", "content": test_prompt}] for _ in range(test_calls)]
    start_time = time.time()
    try:
        litellm.batch_completion(
            model=model,
            messages=messages,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            force_timeout=force_timeout,
        )
        end_time = time.time()
        response_time = end_time - start_time
        return {
            "total_response_time": response_time,
            "calls_made": 100,
            "status": "success",
            "exception": None,
        }
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        return {
            "total_response_time": response_time,
            "calls_made": 100,
            "status": "failed",
            "exception": e,
        }


def get_provider_fields(custom_llm_provider: str) -> List[ProviderField]:
    """Return the fields required for each provider"""

    if custom_llm_provider == "databricks":
        return litellm.DatabricksConfig().get_required_params()

    elif custom_llm_provider == "ollama":
        return litellm.OllamaConfig().get_required_params()

    else:
        return []


def validate_environment(model: Optional[str] = None) -> dict:
    """
    Checks if the environment variables are valid for the given model.

    Args:
        model (Optional[str]): The name of the model. Defaults to None.

    Returns:
        dict: A dictionary containing the following keys:
            - keys_in_environment (bool): True if all the required keys are present in the environment, False otherwise.
            - missing_keys (List[str]): A list of missing keys in the environment.
    """
    keys_in_environment = False
    missing_keys: List[str] = []

    if model is None:
        return {
            "keys_in_environment": keys_in_environment,
            "missing_keys": missing_keys,
        }
    ## EXTRACT LLM PROVIDER - if model name provided
    try:
        _, custom_llm_provider, _, _ = get_llm_provider(model=model)
    except:
        custom_llm_provider = None
    # # check if llm provider part of model name
    # if model.split("/",1)[0] in litellm.provider_list:
    #     custom_llm_provider = model.split("/", 1)[0]
    #     model = model.split("/", 1)[1]
    #     custom_llm_provider_passed_in = True

    if custom_llm_provider:
        if custom_llm_provider == "openai":
            if "OPENAI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("OPENAI_API_KEY")
        elif custom_llm_provider == "azure":
            if (
                "AZURE_API_BASE" in os.environ
                and "AZURE_API_VERSION" in os.environ
                and "AZURE_API_KEY" in os.environ
            ):
                keys_in_environment = True
            else:
                missing_keys.extend(
                    ["AZURE_API_BASE", "AZURE_API_VERSION", "AZURE_API_KEY"]
                )
        elif custom_llm_provider == "anthropic":
            if "ANTHROPIC_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("ANTHROPIC_API_KEY")
        elif custom_llm_provider == "cohere":
            if "COHERE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("COHERE_API_KEY")
        elif custom_llm_provider == "replicate":
            if "REPLICATE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("REPLICATE_API_KEY")
        elif custom_llm_provider == "openrouter":
            if "OPENROUTER_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("OPENROUTER_API_KEY")
        elif custom_llm_provider == "vertex_ai":
            if "VERTEXAI_PROJECT" in os.environ and "VERTEXAI_LOCATION" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.extend(["VERTEXAI_PROJECT", "VERTEXAI_LOCATION"])
        elif custom_llm_provider == "huggingface":
            if "HUGGINGFACE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("HUGGINGFACE_API_KEY")
        elif custom_llm_provider == "ai21":
            if "AI21_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("AI21_API_KEY")
        elif custom_llm_provider == "together_ai":
            if "TOGETHERAI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("TOGETHERAI_API_KEY")
        elif custom_llm_provider == "aleph_alpha":
            if "ALEPH_ALPHA_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("ALEPH_ALPHA_API_KEY")
        elif custom_llm_provider == "baseten":
            if "BASETEN_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("BASETEN_API_KEY")
        elif custom_llm_provider == "nlp_cloud":
            if "NLP_CLOUD_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("NLP_CLOUD_API_KEY")
        elif custom_llm_provider == "bedrock" or custom_llm_provider == "sagemaker":
            if (
                "AWS_ACCESS_KEY_ID" in os.environ
                and "AWS_SECRET_ACCESS_KEY" in os.environ
            ):
                keys_in_environment = True
            else:
                missing_keys.append("AWS_ACCESS_KEY_ID")
                missing_keys.append("AWS_SECRET_ACCESS_KEY")
        elif custom_llm_provider in ["ollama", "ollama_chat"]:
            if "OLLAMA_API_BASE" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("OLLAMA_API_BASE")
        elif custom_llm_provider == "anyscale":
            if "ANYSCALE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("ANYSCALE_API_KEY")
        elif custom_llm_provider == "deepinfra":
            if "DEEPINFRA_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("DEEPINFRA_API_KEY")
        elif custom_llm_provider == "gemini":
            if "GEMINI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("GEMINI_API_KEY")
        elif custom_llm_provider == "groq":
            if "GROQ_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("GROQ_API_KEY")
        elif custom_llm_provider == "deepseek":
            if "DEEPSEEK_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("DEEPSEEK_API_KEY")
        elif custom_llm_provider == "mistral":
            if "MISTRAL_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("MISTRAL_API_KEY")
        elif custom_llm_provider == "palm":
            if "PALM_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("PALM_API_KEY")
        elif custom_llm_provider == "perplexity":
            if "PERPLEXITYAI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("PERPLEXITYAI_API_KEY")
        elif custom_llm_provider == "voyage":
            if "VOYAGE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("VOYAGE_API_KEY")
        elif custom_llm_provider == "fireworks_ai":
            if (
                "FIREWORKS_AI_API_KEY" in os.environ
                or "FIREWORKS_API_KEY" in os.environ
                or "FIREWORKSAI_API_KEY" in os.environ
                or "FIREWORKS_AI_TOKEN" in os.environ
            ):
                keys_in_environment = True
            else:
                missing_keys.append("FIREWORKS_AI_API_KEY")
        elif custom_llm_provider == "cloudflare":
            if "CLOUDFLARE_API_KEY" in os.environ and (
                "CLOUDFLARE_ACCOUNT_ID" in os.environ
                or "CLOUDFLARE_API_BASE" in os.environ
            ):
                keys_in_environment = True
            else:
                missing_keys.append("CLOUDFLARE_API_KEY")
                missing_keys.append("CLOUDFLARE_API_BASE")
    else:
        ## openai - chatcompletion + text completion
        if (
            model in litellm.open_ai_chat_completion_models
            or model in litellm.open_ai_text_completion_models
            or model in litellm.open_ai_embedding_models
            or model in litellm.openai_image_generation_models
        ):
            if "OPENAI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("OPENAI_API_KEY")
        ## anthropic
        elif model in litellm.anthropic_models:
            if "ANTHROPIC_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("ANTHROPIC_API_KEY")
        ## cohere
        elif model in litellm.cohere_models:
            if "COHERE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("COHERE_API_KEY")
        ## replicate
        elif model in litellm.replicate_models:
            if "REPLICATE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("REPLICATE_API_KEY")
        ## openrouter
        elif model in litellm.openrouter_models:
            if "OPENROUTER_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("OPENROUTER_API_KEY")
        ## vertex - text + chat models
        elif (
            model in litellm.vertex_chat_models
            or model in litellm.vertex_text_models
            or model in litellm.models_by_provider["vertex_ai"]
        ):
            if "VERTEXAI_PROJECT" in os.environ and "VERTEXAI_LOCATION" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.extend(["VERTEXAI_PROJECT", "VERTEXAI_PROJECT"])
        ## huggingface
        elif model in litellm.huggingface_models:
            if "HUGGINGFACE_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("HUGGINGFACE_API_KEY")
        ## ai21
        elif model in litellm.ai21_models:
            if "AI21_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("AI21_API_KEY")
        ## together_ai
        elif model in litellm.together_ai_models:
            if "TOGETHERAI_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("TOGETHERAI_API_KEY")
        ## aleph_alpha
        elif model in litellm.aleph_alpha_models:
            if "ALEPH_ALPHA_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("ALEPH_ALPHA_API_KEY")
        ## baseten
        elif model in litellm.baseten_models:
            if "BASETEN_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("BASETEN_API_KEY")
        ## nlp_cloud
        elif model in litellm.nlp_cloud_models:
            if "NLP_CLOUD_API_KEY" in os.environ:
                keys_in_environment = True
            else:
                missing_keys.append("NLP_CLOUD_API_KEY")
    return {"keys_in_environment": keys_in_environment, "missing_keys": missing_keys}


def set_callbacks(callback_list, function_id=None):

    global sentry_sdk_instance, capture_exception, add_breadcrumb, posthog, slack_app, alerts_channel, traceloopLogger, athinaLogger, heliconeLogger, aispendLogger, berrispendLogger, supabaseClient, liteDebuggerClient, lunaryLogger, promptLayerLogger, langFuseLogger, customLogger, weightsBiasesLogger, langsmithLogger, logfireLogger, dynamoLogger, s3Logger, dataDogLogger, prometheusLogger, greenscaleLogger, openMeterLogger

    try:
        for callback in callback_list:
            print_verbose(f"init callback list: {callback}")
            if callback == "sentry":
                try:
                    import sentry_sdk
                except ImportError:
                    print_verbose("Package 'sentry_sdk' is missing. Installing it...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "sentry_sdk"]
                    )
                    import sentry_sdk
                sentry_sdk_instance = sentry_sdk
                sentry_trace_rate = (
                    os.environ.get("SENTRY_API_TRACE_RATE")
                    if "SENTRY_API_TRACE_RATE" in os.environ
                    else "1.0"
                )
                sentry_sdk_instance.init(
                    dsn=os.environ.get("SENTRY_DSN"),
                    traces_sample_rate=float(sentry_trace_rate),
                )
                capture_exception = sentry_sdk_instance.capture_exception
                add_breadcrumb = sentry_sdk_instance.add_breadcrumb
            elif callback == "posthog":
                try:
                    from posthog import Posthog
                except ImportError:
                    print_verbose("Package 'posthog' is missing. Installing it...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "posthog"]
                    )
                    from posthog import Posthog
                posthog = Posthog(
                    project_api_key=os.environ.get("POSTHOG_API_KEY"),
                    host=os.environ.get("POSTHOG_API_URL"),
                )
            elif callback == "slack":
                try:
                    from slack_bolt import App
                except ImportError:
                    print_verbose("Package 'slack_bolt' is missing. Installing it...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "slack_bolt"]
                    )
                    from slack_bolt import App
                slack_app = App(
                    token=os.environ.get("SLACK_API_TOKEN"),
                    signing_secret=os.environ.get("SLACK_API_SECRET"),
                )
                alerts_channel = os.environ["SLACK_API_CHANNEL"]
                print_verbose(f"Initialized Slack App: {slack_app}")
            elif callback == "traceloop":
                traceloopLogger = TraceloopLogger()
            elif callback == "athina":
                athinaLogger = AthinaLogger()
                print_verbose("Initialized Athina Logger")
            elif callback == "helicone":
                heliconeLogger = HeliconeLogger()
            elif callback == "lunary":
                lunaryLogger = LunaryLogger()
            elif callback == "promptlayer":
                promptLayerLogger = PromptLayerLogger()
            elif callback == "langfuse":
                langFuseLogger = LangFuseLogger()
            elif callback == "openmeter":
                openMeterLogger = OpenMeterLogger()
            elif callback == "datadog":
                dataDogLogger = DataDogLogger()
            elif callback == "prometheus":
                if prometheusLogger is None:
                    prometheusLogger = PrometheusLogger()
            elif callback == "dynamodb":
                dynamoLogger = DyanmoDBLogger()
            elif callback == "s3":
                s3Logger = S3Logger()
            elif callback == "wandb":
                weightsBiasesLogger = WeightsBiasesLogger()
            elif callback == "langsmith":
                langsmithLogger = LangsmithLogger()
            elif callback == "logfire":
                logfireLogger = LogfireLogger()
            elif callback == "aispend":
                aispendLogger = AISpendLogger()
            elif callback == "berrispend":
                berrispendLogger = BerriSpendLogger()
            elif callback == "supabase":
                print_verbose(f"instantiating supabase")
                supabaseClient = Supabase()
            elif callback == "greenscale":
                greenscaleLogger = GreenscaleLogger()
                print_verbose("Initialized Greenscale Logger")
            elif callback == "lite_debugger":
                print_verbose(f"instantiating lite_debugger")
                if function_id:
                    liteDebuggerClient = LiteDebugger(email=function_id)
                elif litellm.token:
                    liteDebuggerClient = LiteDebugger(email=litellm.token)
                elif litellm.email:
                    liteDebuggerClient = LiteDebugger(email=litellm.email)
                else:
                    liteDebuggerClient = LiteDebugger(email=str(uuid.uuid4()))
            elif callable(callback):
                customLogger = CustomLogger()
    except Exception as e:
        raise e


# NOTE: DEPRECATING this in favor of using failure_handler() in Logging:
def handle_failure(exception, traceback_exception, start_time, end_time, args, kwargs):
    global sentry_sdk_instance, capture_exception, add_breadcrumb, posthog, slack_app, alerts_channel, aispendLogger, berrispendLogger, supabaseClient, liteDebuggerClient, lunaryLogger
    try:
        # print_verbose(f"handle_failure args: {args}")
        # print_verbose(f"handle_failure kwargs: {kwargs}")

        success_handler = additional_details.pop("success_handler", None)
        failure_handler = additional_details.pop("failure_handler", None)

        additional_details["Event_Name"] = additional_details.pop(
            "failed_event_name", "litellm.failed_query"
        )
        print_verbose(f"self.failure_callback: {litellm.failure_callback}")
        for callback in litellm.failure_callback:
            try:
                if callback == "slack":
                    slack_msg = ""
                    if len(kwargs) > 0:
                        for key in kwargs:
                            slack_msg += f"{key}: {kwargs[key]}\n"
                    if len(args) > 0:
                        for i, arg in enumerate(args):
                            slack_msg += f"LiteLLM_Args_{str(i)}: {arg}"
                    for detail in additional_details:
                        slack_msg += f"{detail}: {additional_details[detail]}\n"
                    slack_msg += f"Traceback: {traceback_exception}"
                    truncated_slack_msg = textwrap.shorten(
                        slack_msg, width=512, placeholder="..."
                    )
                    slack_app.client.chat_postMessage(
                        channel=alerts_channel, text=truncated_slack_msg
                    )
                elif callback == "sentry":
                    capture_exception(exception)
                elif callback == "posthog":
                    print_verbose(
                        f"inside posthog, additional_details: {len(additional_details.keys())}"
                    )
                    ph_obj = {}
                    if len(kwargs) > 0:
                        ph_obj = kwargs
                    if len(args) > 0:
                        for i, arg in enumerate(args):
                            ph_obj["litellm_args_" + str(i)] = arg
                    for detail in additional_details:
                        ph_obj[detail] = additional_details[detail]
                    event_name = additional_details["Event_Name"]
                    print_verbose(f"ph_obj: {ph_obj}")
                    print_verbose(f"PostHog Event Name: {event_name}")
                    if "user_id" in additional_details:
                        posthog.capture(
                            additional_details["user_id"], event_name, ph_obj
                        )
                    else:  # PostHog calls require a unique id to identify a user - https://posthog.com/docs/libraries/python
                        unique_id = str(uuid.uuid4())
                        posthog.capture(unique_id, event_name)
                        print_verbose(f"successfully logged to PostHog!")
                elif callback == "berrispend":
                    print_verbose("reaches berrispend for logging!")
                    model = args[0] if len(args) > 0 else kwargs["model"]
                    messages = args[1] if len(args) > 1 else kwargs["messages"]
                    result = {
                        "model": model,
                        "created": time.time(),
                        "error": traceback_exception,
                        "usage": {
                            "prompt_tokens": prompt_token_calculator(
                                model, messages=messages
                            ),
                            "completion_tokens": 0,
                        },
                    }
                    berrispendLogger.log_event(
                        model=model,
                        messages=messages,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        print_verbose=print_verbose,
                    )
                elif callback == "aispend":
                    print_verbose("reaches aispend for logging!")
                    model = args[0] if len(args) > 0 else kwargs["model"]
                    messages = args[1] if len(args) > 1 else kwargs["messages"]
                    result = {
                        "model": model,
                        "created": time.time(),
                        "usage": {
                            "prompt_tokens": prompt_token_calculator(
                                model, messages=messages
                            ),
                            "completion_tokens": 0,
                        },
                    }
                    aispendLogger.log_event(
                        model=model,
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        print_verbose=print_verbose,
                    )
                elif callback == "supabase":
                    print_verbose("reaches supabase for logging!")
                    print_verbose(f"supabaseClient: {supabaseClient}")
                    model = args[0] if len(args) > 0 else kwargs["model"]
                    messages = args[1] if len(args) > 1 else kwargs["messages"]
                    result = {
                        "model": model,
                        "created": time.time(),
                        "error": traceback_exception,
                        "usage": {
                            "prompt_tokens": prompt_token_calculator(
                                model, messages=messages
                            ),
                            "completion_tokens": 0,
                        },
                    }
                    supabaseClient.log_event(
                        model=model,
                        messages=messages,
                        end_user=kwargs.get("user", "default"),
                        response_obj=result,
                        start_time=start_time,
                        end_time=end_time,
                        litellm_call_id=kwargs["litellm_call_id"],
                        print_verbose=print_verbose,
                    )
            except:
                print_verbose(
                    f"Error Occurred while logging failure: {traceback.format_exc()}"
                )
                pass

        if failure_handler and callable(failure_handler):
            call_details = {
                "exception": exception,
                "additional_details": additional_details,
            }
            failure_handler(call_details)
        pass
    except Exception as e:
        # LOGGING
        exception_logging(logger_fn=user_logger_fn, exception=e)
        pass


async def convert_to_streaming_response_async(response_object: Optional[dict] = None):
    """
    Asynchronously converts a response object to a streaming response.

    Args:
        response_object (Optional[dict]): The response object to be converted. Defaults to None.

    Raises:
        Exception: If the response object is None.

    Yields:
        ModelResponse: The converted streaming response object.

    Returns:
        None
    """
    if response_object is None:
        raise Exception("Error in response object format")

    model_response_object = ModelResponse(stream=True)

    if model_response_object is None:
        raise Exception("Error in response creating model response object")

    choice_list = []

    for idx, choice in enumerate(response_object["choices"]):
        if (
            choice["message"].get("tool_calls", None) is not None
            and isinstance(choice["message"]["tool_calls"], list)
            and len(choice["message"]["tool_calls"]) > 0
            and isinstance(choice["message"]["tool_calls"][0], dict)
        ):
            pydantic_tool_calls = []
            for index, t in enumerate(choice["message"]["tool_calls"]):
                if "index" not in t:
                    t["index"] = index
                pydantic_tool_calls.append(ChatCompletionDeltaToolCall(**t))
            choice["message"]["tool_calls"] = pydantic_tool_calls
        delta = Delta(
            content=choice["message"].get("content", None),
            role=choice["message"]["role"],
            function_call=choice["message"].get("function_call", None),
            tool_calls=choice["message"].get("tool_calls", None),
        )
        finish_reason = choice.get("finish_reason", None)

        if finish_reason is None:
            finish_reason = choice.get("finish_details")

        logprobs = choice.get("logprobs", None)

        choice = StreamingChoices(
            finish_reason=finish_reason, index=idx, delta=delta, logprobs=logprobs
        )
        choice_list.append(choice)

    model_response_object.choices = choice_list

    if "usage" in response_object and response_object["usage"] is not None:
        setattr(
            model_response_object,
            "usage",
            Usage(
                completion_tokens=response_object["usage"].get("completion_tokens", 0),
                prompt_tokens=response_object["usage"].get("prompt_tokens", 0),
                total_tokens=response_object["usage"].get("total_tokens", 0),
            ),
        )

    if "id" in response_object:
        model_response_object.id = response_object["id"]

    if "created" in response_object:
        model_response_object.created = response_object["created"]

    if "system_fingerprint" in response_object:
        model_response_object.system_fingerprint = response_object["system_fingerprint"]

    if "model" in response_object:
        model_response_object.model = response_object["model"]

    yield model_response_object
    await asyncio.sleep(0)


def convert_to_streaming_response(response_object: Optional[dict] = None):
    # used for yielding Cache hits when stream == True
    if response_object is None:
        raise Exception("Error in response object format")

    model_response_object = ModelResponse(stream=True)
    choice_list = []
    for idx, choice in enumerate(response_object["choices"]):
        delta = Delta(
            content=choice["message"].get("content", None),
            role=choice["message"]["role"],
            function_call=choice["message"].get("function_call", None),
            tool_calls=choice["message"].get("tool_calls", None),
        )
        finish_reason = choice.get("finish_reason", None)
        if finish_reason == None:
            # gpt-4 vision can return 'finish_reason' or 'finish_details'
            finish_reason = choice.get("finish_details")
        logprobs = choice.get("logprobs", None)
        enhancements = choice.get("enhancements", None)
        choice = StreamingChoices(
            finish_reason=finish_reason,
            index=idx,
            delta=delta,
            logprobs=logprobs,
            enhancements=enhancements,
        )

        choice_list.append(choice)
    model_response_object.choices = choice_list

    if "usage" in response_object and response_object["usage"] is not None:
        setattr(model_response_object, "usage", Usage())
        model_response_object.usage.completion_tokens = response_object["usage"].get("completion_tokens", 0)  # type: ignore
        model_response_object.usage.prompt_tokens = response_object["usage"].get("prompt_tokens", 0)  # type: ignore
        model_response_object.usage.total_tokens = response_object["usage"].get("total_tokens", 0)  # type: ignore

    if "id" in response_object:
        model_response_object.id = response_object["id"]

    if "created" in response_object:
        model_response_object.created = response_object["created"]

    if "system_fingerprint" in response_object:
        model_response_object.system_fingerprint = response_object["system_fingerprint"]

    if "model" in response_object:
        model_response_object.model = response_object["model"]
    yield model_response_object


def convert_to_model_response_object(
    response_object: Optional[dict] = None,
    model_response_object: Optional[
        Union[ModelResponse, EmbeddingResponse, ImageResponse, TranscriptionResponse]
    ] = None,
    response_type: Literal[
        "completion", "embedding", "image_generation", "audio_transcription"
    ] = "completion",
    stream=False,
    start_time=None,
    end_time=None,
    hidden_params: Optional[dict] = None,
):
    received_args = locals()
    try:
        if response_type == "completion" and (
            model_response_object is None
            or isinstance(model_response_object, ModelResponse)
        ):
            if response_object is None or model_response_object is None:
                raise Exception("Error in response object format")
            if stream == True:
                # for returning cached responses, we need to yield a generator
                return convert_to_streaming_response(response_object=response_object)
            choice_list = []

            assert response_object["choices"] is not None and isinstance(
                response_object["choices"], Iterable
            )

            for idx, choice in enumerate(response_object["choices"]):
                message = Message(
                    content=choice["message"].get("content", None),
                    role=choice["message"]["role"] or "assistant",
                    function_call=choice["message"].get("function_call", None),
                    tool_calls=choice["message"].get("tool_calls", None),
                )
                finish_reason = choice.get("finish_reason", None)
                if finish_reason == None:
                    # gpt-4 vision can return 'finish_reason' or 'finish_details'
                    finish_reason = choice.get("finish_details")
                logprobs = choice.get("logprobs", None)
                enhancements = choice.get("enhancements", None)
                choice = Choices(
                    finish_reason=finish_reason,
                    index=idx,
                    message=message,
                    logprobs=logprobs,
                    enhancements=enhancements,
                )
                choice_list.append(choice)
            model_response_object.choices = choice_list

            if "usage" in response_object and response_object["usage"] is not None:
                model_response_object.usage.completion_tokens = response_object["usage"].get("completion_tokens", 0)  # type: ignore
                model_response_object.usage.prompt_tokens = response_object["usage"].get("prompt_tokens", 0)  # type: ignore
                model_response_object.usage.total_tokens = response_object["usage"].get("total_tokens", 0)  # type: ignore

            if "created" in response_object:
                model_response_object.created = response_object["created"]

            if "id" in response_object:
                model_response_object.id = response_object["id"]

            if "system_fingerprint" in response_object:
                model_response_object.system_fingerprint = response_object[
                    "system_fingerprint"
                ]

            if "model" in response_object and model_response_object.model is None:
                model_response_object.model = response_object["model"]

            if start_time is not None and end_time is not None:
                if isinstance(start_time, type(end_time)):
                    model_response_object._response_ms = (  # type: ignore
                        end_time - start_time
                    ).total_seconds() * 1000

            if hidden_params is not None:
                model_response_object._hidden_params = hidden_params

            return model_response_object
        elif response_type == "embedding" and (
            model_response_object is None
            or isinstance(model_response_object, EmbeddingResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = EmbeddingResponse()

            if "model" in response_object:
                model_response_object.model = response_object["model"]

            if "object" in response_object:
                model_response_object.object = response_object["object"]

            model_response_object.data = response_object["data"]

            if "usage" in response_object and response_object["usage"] is not None:
                model_response_object.usage.completion_tokens = response_object["usage"].get("completion_tokens", 0)  # type: ignore
                model_response_object.usage.prompt_tokens = response_object["usage"].get("prompt_tokens", 0)  # type: ignore
                model_response_object.usage.total_tokens = response_object["usage"].get("total_tokens", 0)  # type: ignore

            if start_time is not None and end_time is not None:
                model_response_object._response_ms = (  # type: ignore
                    end_time - start_time
                ).total_seconds() * 1000  # return response latency in ms like openai

            if hidden_params is not None:
                model_response_object._hidden_params = hidden_params

            return model_response_object
        elif response_type == "image_generation" and (
            model_response_object is None
            or isinstance(model_response_object, ImageResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = ImageResponse()

            if "created" in response_object:
                model_response_object.created = response_object["created"]

            if "data" in response_object:
                model_response_object.data = response_object["data"]

            if hidden_params is not None:
                model_response_object._hidden_params = hidden_params

            return model_response_object
        elif response_type == "audio_transcription" and (
            model_response_object is None
            or isinstance(model_response_object, TranscriptionResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = TranscriptionResponse()

            if "text" in response_object:
                model_response_object.text = response_object["text"]

            if hidden_params is not None:
                model_response_object._hidden_params = hidden_params
            return model_response_object
    except Exception as e:
        raise Exception(
            f"Invalid response object {traceback.format_exc()}\n\nreceived_args={received_args}"
        )


def acreate(*args, **kwargs):  ## Thin client to handle the acreate langchain call
    return litellm.acompletion(*args, **kwargs)


def prompt_token_calculator(model, messages):
    # use tiktoken or anthropic's tokenizer depending on the model
    text = " ".join(message["content"] for message in messages)
    num_tokens = 0
    if "claude" in model:
        try:
            import anthropic
        except:
            Exception("Anthropic import failed please run `pip install anthropic`")
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

        anthropic = Anthropic()
        num_tokens = anthropic.count_tokens(text)
    else:
        num_tokens = len(encoding.encode(text))
    return num_tokens


def valid_model(model):
    try:
        # for a given model name, check if the user has the right permissions to access the model
        if (
            model in litellm.open_ai_chat_completion_models
            or model in litellm.open_ai_text_completion_models
        ):
            openai.Model.retrieve(model)
        else:
            messages = [{"role": "user", "content": "Hello World"}]
            litellm.completion(model=model, messages=messages)
    except:
        raise BadRequestError(message="", model=model, llm_provider="")


def check_valid_key(model: str, api_key: str):
    """
    Checks if a given API key is valid for a specific model by making a litellm.completion call with max_tokens=10

    Args:
        model (str): The name of the model to check the API key against.
        api_key (str): The API key to be checked.

    Returns:
        bool: True if the API key is valid for the model, False otherwise.
    """
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    try:
        litellm.completion(
            model=model, messages=messages, api_key=api_key, max_tokens=10
        )
        return True
    except AuthenticationError as e:
        return False
    except Exception as e:
        return False


def _should_retry(status_code: int):
    """
    Reimplementation of openai's should retry logic, since that one can't be imported.
    https://github.com/openai/openai-python/blob/af67cfab4210d8e497c05390ce14f39105c77519/src/openai/_base_client.py#L639
    """
    # If the server explicitly says whether or not to retry, obey.
    # Retry on request timeouts.
    if status_code == 408:
        return True

    # Retry on lock timeouts.
    if status_code == 409:
        return True

    # Retry on rate limits.
    if status_code == 429:
        return True

    # Retry internal errors.
    if status_code >= 500:
        return True

    return False


def _get_retry_after_from_exception_header(
    response_headers: Optional[httpx.Headers] = None,
):
    """
    Reimplementation of openai's calculate retry after, since that one can't be imported.
    https://github.com/openai/openai-python/blob/af67cfab4210d8e497c05390ce14f39105c77519/src/openai/_base_client.py#L631
    """
    try:
        import email  # openai import

        # About the Retry-After header: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After
        #
        # <http-date>". See https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After#syntax for
        # details.
        if response_headers is not None:
            retry_header = response_headers.get("retry-after")
            try:
                retry_after = int(retry_header)
            except Exception:
                retry_date_tuple = email.utils.parsedate_tz(retry_header)  # type: ignore
                if retry_date_tuple is None:
                    retry_after = -1
                else:
                    retry_date = email.utils.mktime_tz(retry_date_tuple)  # type: ignore
                    retry_after = int(retry_date - time.time())
        else:
            retry_after = -1
        return retry_after

    except Exception as e:
        retry_after = -1


def _calculate_retry_after(
    remaining_retries: int,
    max_retries: int,
    response_headers: Optional[httpx.Headers] = None,
    min_timeout: int = 0,
):
    retry_after = _get_retry_after_from_exception_header(response_headers)

    # If the API asks us to wait a certain amount of time (and it's a reasonable amount), just do what it says.
    if 0 < retry_after <= 60:
        return retry_after

    initial_retry_delay = 0.5
    max_retry_delay = 8.0
    nb_retries = max_retries - remaining_retries

    # Apply exponential backoff, but not more than the max.
    sleep_seconds = min(initial_retry_delay * pow(2.0, nb_retries), max_retry_delay)

    # Apply some jitter, plus-or-minus half a second.
    jitter = 1 - 0.25 * random.random()
    timeout = sleep_seconds * jitter
    return timeout if timeout >= min_timeout else min_timeout


# integration helper function
def modify_integration(integration_name, integration_params):
    global supabaseClient
    if integration_name == "supabase":
        if "table_name" in integration_params:
            Supabase.supabase_table_name = integration_params["table_name"]


# custom prompt helper function
def register_prompt_template(
    model: str,
    roles: dict,
    initial_prompt_value: str = "",
    final_prompt_value: str = "",
):
    """
    Register a prompt template to follow your custom format for a given model

    Args:
        model (str): The name of the model.
        roles (dict): A dictionary mapping roles to their respective prompt values.
        initial_prompt_value (str, optional): The initial prompt value. Defaults to "".
        final_prompt_value (str, optional): The final prompt value. Defaults to "".

    Returns:
        dict: The updated custom prompt dictionary.
    Example usage:
    ```
    import litellm
    litellm.register_prompt_template(
            model="llama-2",
        initial_prompt_value="You are a good assistant" # [OPTIONAL]
            roles={
            "system": {
                "pre_message": "[INST] <<SYS>>\n", # [OPTIONAL]
                "post_message": "\n<</SYS>>\n [/INST]\n" # [OPTIONAL]
            },
            "user": {
                "pre_message": "[INST] ", # [OPTIONAL]
                "post_message": " [/INST]" # [OPTIONAL]
            },
            "assistant": {
                "pre_message": "\n" # [OPTIONAL]
                "post_message": "\n" # [OPTIONAL]
            }
        }
        final_prompt_value="Now answer as best you can:" # [OPTIONAL]
    )
    ```
    """
    model = get_llm_provider(model=model)[0]
    litellm.custom_prompt_dict[model] = {
        "roles": roles,
        "initial_prompt_value": initial_prompt_value,
        "final_prompt_value": final_prompt_value,
    }
    return litellm.custom_prompt_dict


####### DEPRECATED ################


def get_all_keys(llm_provider=None):
    try:
        global last_fetched_at_keys
        # if user is using hosted product -> instantiate their env with their hosted api keys - refresh every 5 minutes
        print_verbose(f"Reaches get all keys, llm_provider: {llm_provider}")
        user_email = (
            os.getenv("LITELLM_EMAIL")
            or litellm.email
            or litellm.token
            or os.getenv("LITELLM_TOKEN")
        )
        if user_email:
            time_delta = 0
            if last_fetched_at_keys != None:
                current_time = time.time()
                time_delta = current_time - last_fetched_at_keys
            if (
                time_delta > 300 or last_fetched_at_keys == None or llm_provider
            ):  # if the llm provider is passed in , assume this happening due to an AuthError for that provider
                # make the api call
                last_fetched_at = time.time()
                print_verbose(f"last_fetched_at: {last_fetched_at}")
                response = requests.post(
                    url="http://api.litellm.ai/get_all_keys",
                    headers={"content-type": "application/json"},
                    data=json.dumps({"user_email": user_email}),
                )
                print_verbose(f"get model key response: {response.text}")
                data = response.json()
                # update model list
                for key, value in data[
                    "model_keys"
                ].items():  # follows the LITELLM API KEY format - <UPPERCASE_PROVIDER_NAME>_API_KEY - e.g. HUGGINGFACE_API_KEY
                    os.environ[key] = value
                # set model alias map
                for model_alias, value in data["model_alias_map"].items():
                    litellm.model_alias_map[model_alias] = value
                return "it worked!"
            return None
        return None
    except:
        print_verbose(
            f"[Non-Blocking Error] get_all_keys error - {traceback.format_exc()}"
        )
        pass


def get_model_list():
    global last_fetched_at, print_verbose
    try:
        # if user is using hosted product -> get their updated model list
        user_email = (
            os.getenv("LITELLM_EMAIL")
            or litellm.email
            or litellm.token
            or os.getenv("LITELLM_TOKEN")
        )
        if user_email:
            # make the api call
            last_fetched_at = time.time()
            print_verbose(f"last_fetched_at: {last_fetched_at}")
            response = requests.post(
                url="http://api.litellm.ai/get_model_list",
                headers={"content-type": "application/json"},
                data=json.dumps({"user_email": user_email}),
            )
            print_verbose(f"get_model_list response: {response.text}")
            data = response.json()
            # update model list
            model_list = data["model_list"]
            # # check if all model providers are in environment
            # model_providers = data["model_providers"]
            # missing_llm_provider = None
            # for item in model_providers:
            #     if f"{item.upper()}_API_KEY" not in os.environ:
            #         missing_llm_provider = item
            #         break
            # # update environment - if required
            # threading.Thread(target=get_all_keys, args=(missing_llm_provider)).start()
            return model_list
        return []  # return empty list by default
    except:
        print_verbose(
            f"[Non-Blocking Error] get_model_list error - {traceback.format_exc()}"
        )


####### EXCEPTION MAPPING ################
def exception_type(
    model,
    original_exception,
    custom_llm_provider,
    completion_kwargs={},
    extra_kwargs={},
):
    global user_logger_fn, liteDebuggerClient
    exception_mapping_worked = False
    if litellm.suppress_debug_info is False:
        print()  # noqa
        print(  # noqa
            "\033[1;31mGive Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new\033[0m"  # noqa
        )  # noqa
        print(  # noqa
            "LiteLLM.Info: If you need to debug this error, use `litellm.set_verbose=True'."  # noqa
        )  # noqa
        print()  # noqa
    try:
        if model:
            error_str = str(original_exception)
            if isinstance(original_exception, BaseException):
                exception_type = type(original_exception).__name__
            else:
                exception_type = ""

            ################################################################################
            # Common Extra information needed for all providers
            # We pass num retries, api_base, vertex_deployment etc to the exception here
            ################################################################################
            extra_information = ""
            try:
                _api_base = litellm.get_api_base(
                    model=model, optional_params=extra_kwargs
                )
                messages = litellm.get_first_chars_messages(kwargs=completion_kwargs)
                _vertex_project = extra_kwargs.get("vertex_project")
                _vertex_location = extra_kwargs.get("vertex_location")
                _metadata = extra_kwargs.get("metadata", {}) or {}
                _model_group = _metadata.get("model_group")
                _deployment = _metadata.get("deployment")
                extra_information = f"\nModel: {model}"
                if _api_base:
                    extra_information += f"\nAPI Base: `{_api_base}`"
                if (
                    messages
                    and len(messages) > 0
                    and litellm.redact_messages_in_exceptions is False
                ):
                    extra_information += f"\nMessages: `{messages}`"

                if _model_group is not None:
                    extra_information += f"\nmodel_group: `{_model_group}`\n"
                if _deployment is not None:
                    extra_information += f"\ndeployment: `{_deployment}`\n"
                if _vertex_project is not None:
                    extra_information += f"\nvertex_project: `{_vertex_project}`\n"
                if _vertex_location is not None:
                    extra_information += f"\nvertex_location: `{_vertex_location}`\n"

                # on litellm proxy add key name + team to exceptions
                extra_information = _add_key_name_and_team_to_alert(
                    request_info=extra_information, metadata=_metadata
                )
            except:
                # DO NOT LET this Block raising the original exception
                pass

            ################################################################################
            # End of Common Extra information Needed for all providers
            ################################################################################

            ################################################################################
            #################### Start of Provider Exception mapping ####################
            ################################################################################

            if "Request Timeout Error" in error_str or "Request timed out" in error_str:
                exception_mapping_worked = True
                raise Timeout(
                    message=f"APITimeoutError - Request timed out. \nerror_str: {error_str}",
                    model=model,
                    llm_provider=custom_llm_provider,
                    litellm_debug_info=extra_information,
                )

            if (
                custom_llm_provider == "openai"
                or custom_llm_provider == "text-completion-openai"
                or custom_llm_provider == "custom_openai"
                or custom_llm_provider in litellm.openai_compatible_providers
            ):
                # custom_llm_provider is openai, make it OpenAI
                if hasattr(original_exception, "message"):
                    message = original_exception.message
                else:
                    message = str(original_exception)
                if message is not None and isinstance(message, str):
                    message = message.replace("OPENAI", custom_llm_provider.upper())
                    message = message.replace("openai", custom_llm_provider)
                    message = message.replace("OpenAI", custom_llm_provider)
                if custom_llm_provider == "openai":
                    exception_provider = "OpenAI" + "Exception"
                else:
                    exception_provider = (
                        custom_llm_provider[0].upper()
                        + custom_llm_provider[1:]
                        + "Exception"
                    )

                if "This model's maximum context length is" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"ContextWindowExceededError: {exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        response=original_exception.response,
                        litellm_debug_info=extra_information,
                    )
                elif (
                    "invalid_request_error" in error_str
                    and "model_not_found" in error_str
                ):
                    exception_mapping_worked = True
                    raise NotFoundError(
                        message=f"{exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        response=original_exception.response,
                        litellm_debug_info=extra_information,
                    )
                elif (
                    "invalid_request_error" in error_str
                    and "content_policy_violation" in error_str
                ):
                    exception_mapping_worked = True
                    raise ContentPolicyViolationError(
                        message=f"ContentPolicyViolationError: {exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        response=original_exception.response,
                        litellm_debug_info=extra_information,
                    )
                elif (
                    "invalid_request_error" in error_str
                    and "Incorrect API key provided" not in error_str
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"BadRequestError: {exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        response=original_exception.response,
                        litellm_debug_info=extra_information,
                    )
                elif "Request too large" in error_str:
                    raise RateLimitError(
                        message=f"RateLimitError: {exception_provider} - {message}",
                        model=model,
                        llm_provider=custom_llm_provider,
                        response=original_exception.response,
                        litellm_debug_info=extra_information,
                    )
                elif (
                    "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"AuthenticationError: {exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        response=original_exception.response,
                        litellm_debug_info=extra_information,
                    )
                elif "Mistral API raised a streaming error" in error_str:
                    exception_mapping_worked = True
                    _request = httpx.Request(
                        method="POST", url="https://api.openai.com/v1"
                    )
                    raise APIError(
                        status_code=500,
                        message=f"{exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        request=_request,
                        litellm_debug_info=extra_information,
                    )
                elif hasattr(original_exception, "status_code"):
                    exception_mapping_worked = True
                    if original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"{exception_provider} - {message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            response=original_exception.response,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AuthenticationError: {exception_provider} - {message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            response=original_exception.response,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 404:
                        exception_mapping_worked = True
                        raise NotFoundError(
                            message=f"NotFoundError: {exception_provider} - {message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=original_exception.response,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"Timeout Error: {exception_provider} - {message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"BadRequestError: {exception_provider} - {message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=original_exception.response,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"RateLimitError: {exception_provider} - {message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=original_exception.response,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"ServiceUnavailableError: {exception_provider} - {message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=original_exception.response,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 504:  # gateway timeout error
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"Timeout Error: {exception_provider} - {message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"APIError: {exception_provider} - {message}",
                            llm_provider=custom_llm_provider,
                            model=model,
                            request=original_exception.request,
                            litellm_debug_info=extra_information,
                        )
                else:
                    # if no status code then it is an APIConnectionError: https://github.com/openai/openai-python#handling-errors
                    raise APIConnectionError(
                        message=f"APIConnectionError: {exception_provider} - {message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        litellm_debug_info=extra_information,
                        request=httpx.Request(
                            method="POST", url="https://api.openai.com/v1/"
                        ),
                    )
            elif custom_llm_provider == "anthropic":  # one of the anthropics
                if hasattr(original_exception, "message"):
                    if (
                        "prompt is too long" in original_exception.message
                        or "prompt: length" in original_exception.message
                    ):
                        exception_mapping_worked = True
                        raise ContextWindowExceededError(
                            message=original_exception.message,
                            model=model,
                            llm_provider="anthropic",
                            response=original_exception.response,
                        )
                    if "Invalid API Key" in original_exception.message:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=original_exception.message,
                            model=model,
                            llm_provider="anthropic",
                            response=original_exception.response,
                        )
                if hasattr(original_exception, "status_code"):
                    print_verbose(f"status_code: {original_exception.status_code}")
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AnthropicException - {original_exception.message}",
                            llm_provider="anthropic",
                            model=model,
                            response=original_exception.response,
                        )
                    elif (
                        original_exception.status_code == 400
                        or original_exception.status_code == 413
                    ):
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"AnthropicException - {original_exception.message}",
                            model=model,
                            llm_provider="anthropic",
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"AnthropicException - {original_exception.message}",
                            model=model,
                            llm_provider="anthropic",
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"AnthropicException - {original_exception.message}",
                            llm_provider="anthropic",
                            model=model,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=500,
                            message=f"AnthropicException - {original_exception.message}. Handle with `litellm.APIError`.",
                            llm_provider="anthropic",
                            model=model,
                            request=original_exception.request,
                        )
            elif custom_llm_provider == "replicate":
                if "Incorrect authentication token" in error_str:
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"ReplicateException - {error_str}",
                        llm_provider="replicate",
                        model=model,
                        response=original_exception.response,
                    )
                elif "input is too long" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"ReplicateException - {error_str}",
                        model=model,
                        llm_provider="replicate",
                        response=original_exception.response,
                    )
                elif exception_type == "ModelError":
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"ReplicateException - {error_str}",
                        model=model,
                        llm_provider="replicate",
                        response=original_exception.response,
                    )
                elif "Request was throttled" in error_str:
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"ReplicateException - {error_str}",
                        llm_provider="replicate",
                        model=model,
                        response=original_exception.response,
                    )
                elif hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"ReplicateException - {original_exception.message}",
                            llm_provider="replicate",
                            model=model,
                            response=original_exception.response,
                        )
                    elif (
                        original_exception.status_code == 400
                        or original_exception.status_code == 422
                        or original_exception.status_code == 413
                    ):
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"ReplicateException - {original_exception.message}",
                            model=model,
                            llm_provider="replicate",
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"ReplicateException - {original_exception.message}",
                            model=model,
                            llm_provider="replicate",
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"ReplicateException - {original_exception.message}",
                            llm_provider="replicate",
                            model=model,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"ReplicateException - {original_exception.message}",
                            llm_provider="replicate",
                            model=model,
                            response=original_exception.response,
                        )
                exception_mapping_worked = True
                raise APIError(
                    status_code=500,
                    message=f"ReplicateException - {str(original_exception)}",
                    llm_provider="replicate",
                    model=model,
                    request=httpx.Request(
                        method="POST",
                        url="https://api.replicate.com/v1/deployments",
                    ),
                )
            elif custom_llm_provider == "watsonx":
                if "token_quota_reached" in error_str:
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"WatsonxException: Rate Limit Errror - {error_str}",
                        llm_provider="watsonx",
                        model=model,
                        response=original_exception.response,
                    )
            elif custom_llm_provider == "predibase":
                if "authorization denied for" in error_str:
                    exception_mapping_worked = True

                    # Predibase returns the raw API Key in the response - this block ensures it's not returned in the exception
                    if (
                        error_str is not None
                        and isinstance(error_str, str)
                        and "bearer" in error_str.lower()
                    ):
                        # only keep the first 10 chars after the occurnence of "bearer"
                        _bearer_token_start_index = error_str.lower().find("bearer")
                        error_str = error_str[: _bearer_token_start_index + 14]
                        error_str += "XXXXXXX" + '"'

                    raise AuthenticationError(
                        message=f"PredibaseException: Authentication Error - {error_str}",
                        llm_provider="predibase",
                        model=model,
                        response=original_exception.response,
                        litellm_debug_info=extra_information,
                    )
                elif hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise litellm.InternalServerError(
                            message=f"PredibaseException - {original_exception.message}",
                            llm_provider="predibase",
                            model=model,
                        )
                    elif original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"PredibaseException - {original_exception.message}",
                            llm_provider="predibase",
                            model=model,
                        )
                    elif original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"PredibaseException - {original_exception.message}",
                            llm_provider="predibase",
                            model=model,
                        )
                    elif original_exception.status_code == 404:
                        exception_mapping_worked = True
                        raise NotFoundError(
                            message=f"PredibaseException - {original_exception.message}",
                            llm_provider="predibase",
                            model=model,
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"PredibaseException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"PredibaseException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"PredibaseException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"PredibaseException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 504:  # gateway timeout error
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"PredibaseException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
            elif custom_llm_provider == "bedrock":
                if (
                    "too many tokens" in error_str
                    or "expected maxLength:" in error_str
                    or "Input is too long" in error_str
                    or "prompt: length: 1.." in error_str
                    or "Too many input tokens" in error_str
                ):
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"BedrockException: Context Window Error - {error_str}",
                        model=model,
                        llm_provider="bedrock",
                        response=original_exception.response,
                    )
                elif "Malformed input request" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"BedrockException - {error_str}",
                        model=model,
                        llm_provider="bedrock",
                        response=original_exception.response,
                    )
                elif (
                    "Unable to locate credentials" in error_str
                    or "The security token included in the request is invalid"
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"BedrockException Invalid Authentication - {error_str}",
                        model=model,
                        llm_provider="bedrock",
                        response=original_exception.response,
                    )
                elif "AccessDeniedException" in error_str:
                    exception_mapping_worked = True
                    raise PermissionDeniedError(
                        message=f"BedrockException PermissionDeniedError - {error_str}",
                        model=model,
                        llm_provider="bedrock",
                        response=original_exception.response,
                    )
                elif (
                    "throttlingException" in error_str
                    or "ThrottlingException" in error_str
                ):
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"BedrockException: Rate Limit Error - {error_str}",
                        model=model,
                        llm_provider="bedrock",
                        response=original_exception.response,
                    )
                elif (
                    "Connect timeout on endpoint URL" in error_str
                    or "timed out" in error_str
                ):
                    exception_mapping_worked = True
                    raise Timeout(
                        message=f"BedrockException: Timeout Error - {error_str}",
                        model=model,
                        llm_provider="bedrock",
                    )
                elif hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"BedrockException - {original_exception.message}",
                            llm_provider="bedrock",
                            model=model,
                            response=httpx.Response(
                                status_code=500,
                                request=httpx.Request(
                                    method="POST", url="https://api.openai.com/v1/"
                                ),
                            ),
                        )
                    elif original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"BedrockException - {original_exception.message}",
                            llm_provider="bedrock",
                            model=model,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"BedrockException - {original_exception.message}",
                            llm_provider="bedrock",
                            model=model,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 404:
                        exception_mapping_worked = True
                        raise NotFoundError(
                            message=f"BedrockException - {original_exception.message}",
                            llm_provider="bedrock",
                            model=model,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"BedrockException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"BedrockException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=original_exception.response,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"BedrockException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=original_exception.response,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"BedrockException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            response=original_exception.response,
                            litellm_debug_info=extra_information,
                        )
                    elif original_exception.status_code == 504:  # gateway timeout error
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"BedrockException - {original_exception.message}",
                            model=model,
                            llm_provider=custom_llm_provider,
                            litellm_debug_info=extra_information,
                        )
            elif custom_llm_provider == "sagemaker":
                if "Unable to locate credentials" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"litellm.BadRequestError: SagemakerException - {error_str}",
                        model=model,
                        llm_provider="sagemaker",
                        response=original_exception.response,
                    )
                elif (
                    "Input validation error: `best_of` must be > 0 and <= 2"
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"SagemakerException - the value of 'n' must be > 0 and <= 2 for sagemaker endpoints",
                        model=model,
                        llm_provider="sagemaker",
                        response=original_exception.response,
                    )
                elif (
                    "`inputs` tokens + `max_new_tokens` must be <=" in error_str
                    or "instance type with more CPU capacity or memory" in error_str
                ):
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"SagemakerException - {error_str}",
                        model=model,
                        llm_provider="sagemaker",
                        response=original_exception.response,
                    )
            elif custom_llm_provider == "vertex_ai":
                if (
                    "Vertex AI API has not been used in project" in error_str
                    or "Unable to find your project" in error_str
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"litellm.BadRequestError: VertexAIException - {error_str}",
                        model=model,
                        llm_provider="vertex_ai",
                        response=httpx.Response(
                            status_code=400,
                            request=httpx.Request(
                                method="POST",
                                url=" https://cloud.google.com/vertex-ai/",
                            ),
                        ),
                        litellm_debug_info=extra_information,
                    )
                elif (
                    "None Unknown Error." in error_str
                    or "Content has no parts." in error_str
                ):
                    exception_mapping_worked = True
                    raise litellm.InternalServerError(
                        message=f"litellm.InternalServerError: VertexAIException - {error_str}",
                        model=model,
                        llm_provider="vertex_ai",
                        response=httpx.Response(
                            status_code=500,
                            content=str(original_exception),
                            request=httpx.Request(method="completion", url="https://github.com/BerriAI/litellm"),  # type: ignore
                        ),
                        litellm_debug_info=extra_information,
                    )
                elif "403" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"VertexAIException BadRequestError - {error_str}",
                        model=model,
                        llm_provider="vertex_ai",
                        response=httpx.Response(
                            status_code=403,
                            request=httpx.Request(
                                method="POST",
                                url=" https://cloud.google.com/vertex-ai/",
                            ),
                        ),
                        litellm_debug_info=extra_information,
                    )
                elif "The response was blocked." in error_str:
                    exception_mapping_worked = True
                    raise UnprocessableEntityError(
                        message=f"VertexAIException UnprocessableEntityError - {error_str}",
                        model=model,
                        llm_provider="vertex_ai",
                        litellm_debug_info=extra_information,
                        response=httpx.Response(
                            status_code=422,
                            request=httpx.Request(
                                method="POST",
                                url=" https://cloud.google.com/vertex-ai/",
                            ),
                        ),
                    )
                elif (
                    "429 Quota exceeded" in error_str
                    or "IndexError: list index out of range" in error_str
                    or "429 Unable to submit request because the service is temporarily out of capacity."
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"VertexAIException RateLimitError - {error_str}",
                        model=model,
                        llm_provider="vertex_ai",
                        litellm_debug_info=extra_information,
                        response=httpx.Response(
                            status_code=429,
                            request=httpx.Request(
                                method="POST",
                                url=" https://cloud.google.com/vertex-ai/",
                            ),
                        ),
                    )

                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"VertexAIException BadRequestError - {error_str}",
                            model=model,
                            llm_provider="vertex_ai",
                            litellm_debug_info=extra_information,
                            response=httpx.Response(
                                status_code=400,
                                request=httpx.Request(
                                    method="POST",
                                    url="https://cloud.google.com/vertex-ai/",
                                ),
                            ),
                        )
                    if original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise litellm.InternalServerError(
                            message=f"VertexAIException InternalServerError - {error_str}",
                            model=model,
                            llm_provider="vertex_ai",
                            litellm_debug_info=extra_information,
                            response=httpx.Response(
                                status_code=500,
                                content=str(original_exception),
                                request=httpx.Request(method="completion", url="https://github.com/BerriAI/litellm"),  # type: ignore
                            ),
                        )
            elif custom_llm_provider == "palm" or custom_llm_provider == "gemini":
                if "503 Getting metadata" in error_str:
                    # auth errors look like this
                    # 503 Getting metadata from plugin failed with error: Reauthentication is needed. Please run `gcloud auth application-default login` to reauthenticate.
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"GeminiException - Invalid api key",
                        model=model,
                        llm_provider="palm",
                        response=original_exception.response,
                    )
                if (
                    "504 Deadline expired before operation could complete." in error_str
                    or "504 Deadline Exceeded" in error_str
                ):
                    exception_mapping_worked = True
                    raise Timeout(
                        message=f"GeminiException - {original_exception.message}",
                        model=model,
                        llm_provider="palm",
                    )
                if "400 Request payload size exceeds" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"GeminiException - {error_str}",
                        model=model,
                        llm_provider="palm",
                        response=original_exception.response,
                    )
                if (
                    "500 An internal error has occurred." in error_str
                    or "list index out of range" in error_str
                ):
                    exception_mapping_worked = True
                    raise APIError(
                        status_code=getattr(original_exception, "status_code", 500),
                        message=f"GeminiException - {original_exception.message}",
                        llm_provider="palm",
                        model=model,
                        request=httpx.Response(
                            status_code=429,
                            request=httpx.Request(
                                method="POST",
                                url=" https://cloud.google.com/vertex-ai/",
                            ),
                        ),
                    )
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"GeminiException - {error_str}",
                            model=model,
                            llm_provider="palm",
                            response=original_exception.response,
                        )
                # Dailed: Error occurred: 400 Request payload size exceeds the limit: 20000 bytes
            elif custom_llm_provider == "cloudflare":
                if "Authentication error" in error_str:
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"Cloudflare Exception - {original_exception.message}",
                        llm_provider="cloudflare",
                        model=model,
                        response=original_exception.response,
                    )
                if "must have required property" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"Cloudflare Exception - {original_exception.message}",
                        llm_provider="cloudflare",
                        model=model,
                        response=original_exception.response,
                    )
            elif (
                custom_llm_provider == "cohere" or custom_llm_provider == "cohere_chat"
            ):  # Cohere
                if (
                    "invalid api token" in error_str
                    or "No API key provided." in error_str
                ):
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"CohereException - {original_exception.message}",
                        llm_provider="cohere",
                        model=model,
                        response=original_exception.response,
                    )
                elif "too many tokens" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"CohereException - {original_exception.message}",
                        model=model,
                        llm_provider="cohere",
                        response=original_exception.response,
                    )
                elif hasattr(original_exception, "status_code"):
                    if (
                        original_exception.status_code == 400
                        or original_exception.status_code == 498
                    ):
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"CohereException - {original_exception.message}",
                            llm_provider="cohere",
                            model=model,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"CohereException - {original_exception.message}",
                            llm_provider="cohere",
                            model=model,
                            response=original_exception.response,
                        )
                elif (
                    "CohereConnectionError" in exception_type
                ):  # cohere seems to fire these errors when we load test it (1k+ messages / min)
                    exception_mapping_worked = True
                    raise RateLimitError(
                        message=f"CohereException - {original_exception.message}",
                        llm_provider="cohere",
                        model=model,
                        response=original_exception.response,
                    )
                elif "invalid type:" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"CohereException - {original_exception.message}",
                        llm_provider="cohere",
                        model=model,
                        response=original_exception.response,
                    )
                elif "Unexpected server error" in error_str:
                    exception_mapping_worked = True
                    raise ServiceUnavailableError(
                        message=f"CohereException - {original_exception.message}",
                        llm_provider="cohere",
                        model=model,
                        response=original_exception.response,
                    )
                else:
                    if hasattr(original_exception, "status_code"):
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"CohereException - {original_exception.message}",
                            llm_provider="cohere",
                            model=model,
                            request=original_exception.request,
                        )
                    raise original_exception
            elif custom_llm_provider == "huggingface":
                if "length limit exceeded" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=error_str,
                        model=model,
                        llm_provider="huggingface",
                        response=original_exception.response,
                    )
                elif "A valid user token is required" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=error_str,
                        llm_provider="huggingface",
                        model=model,
                        response=original_exception.response,
                    )
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"HuggingfaceException - {original_exception.message}",
                            llm_provider="huggingface",
                            model=model,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"HuggingfaceException - {original_exception.message}",
                            model=model,
                            llm_provider="huggingface",
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"HuggingfaceException - {original_exception.message}",
                            model=model,
                            llm_provider="huggingface",
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"HuggingfaceException - {original_exception.message}",
                            llm_provider="huggingface",
                            model=model,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"HuggingfaceException - {original_exception.message}",
                            llm_provider="huggingface",
                            model=model,
                            response=original_exception.response,
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"HuggingfaceException - {original_exception.message}",
                            llm_provider="huggingface",
                            model=model,
                            request=original_exception.request,
                        )
            elif custom_llm_provider == "ai21":
                if hasattr(original_exception, "message"):
                    if "Prompt has too many tokens" in original_exception.message:
                        exception_mapping_worked = True
                        raise ContextWindowExceededError(
                            message=f"AI21Exception - {original_exception.message}",
                            model=model,
                            llm_provider="ai21",
                            response=original_exception.response,
                        )
                    if "Bad or missing API token." in original_exception.message:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"AI21Exception - {original_exception.message}",
                            model=model,
                            llm_provider="ai21",
                            response=original_exception.response,
                        )
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AI21Exception - {original_exception.message}",
                            llm_provider="ai21",
                            model=model,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"AI21Exception - {original_exception.message}",
                            model=model,
                            llm_provider="ai21",
                        )
                    if original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"AI21Exception - {original_exception.message}",
                            model=model,
                            llm_provider="ai21",
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"AI21Exception - {original_exception.message}",
                            llm_provider="ai21",
                            model=model,
                            response=original_exception.response,
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"AI21Exception - {original_exception.message}",
                            llm_provider="ai21",
                            model=model,
                            request=original_exception.request,
                        )
            elif custom_llm_provider == "nlp_cloud":
                if "detail" in error_str:
                    if "Input text length should not exceed" in error_str:
                        exception_mapping_worked = True
                        raise ContextWindowExceededError(
                            message=f"NLPCloudException - {error_str}",
                            model=model,
                            llm_provider="nlp_cloud",
                            response=original_exception.response,
                        )
                    elif "value is not a valid" in error_str:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"NLPCloudException - {error_str}",
                            model=model,
                            llm_provider="nlp_cloud",
                            response=original_exception.response,
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=500,
                            message=f"NLPCloudException - {error_str}",
                            model=model,
                            llm_provider="nlp_cloud",
                            request=original_exception.request,
                        )
                if hasattr(
                    original_exception, "status_code"
                ):  # https://docs.nlpcloud.com/?shell#errors
                    if (
                        original_exception.status_code == 400
                        or original_exception.status_code == 406
                        or original_exception.status_code == 413
                        or original_exception.status_code == 422
                    ):
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model,
                            response=original_exception.response,
                        )
                    elif (
                        original_exception.status_code == 401
                        or original_exception.status_code == 403
                    ):
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model,
                            response=original_exception.response,
                        )
                    elif (
                        original_exception.status_code == 522
                        or original_exception.status_code == 524
                    ):
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"NLPCloudException - {original_exception.message}",
                            model=model,
                            llm_provider="nlp_cloud",
                        )
                    elif (
                        original_exception.status_code == 429
                        or original_exception.status_code == 402
                    ):
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model,
                            response=original_exception.response,
                        )
                    elif (
                        original_exception.status_code == 500
                        or original_exception.status_code == 503
                    ):
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model,
                            request=original_exception.request,
                        )
                    elif (
                        original_exception.status_code == 504
                        or original_exception.status_code == 520
                    ):
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"NLPCloudException - {original_exception.message}",
                            model=model,
                            llm_provider="nlp_cloud",
                            response=original_exception.response,
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"NLPCloudException - {original_exception.message}",
                            llm_provider="nlp_cloud",
                            model=model,
                            request=original_exception.request,
                        )
            elif custom_llm_provider == "together_ai":
                import json

                try:
                    error_response = json.loads(error_str)
                except:
                    error_response = {"error": error_str}
                if (
                    "error" in error_response
                    and "`inputs` tokens + `max_new_tokens` must be <="
                    in error_response["error"]
                ):
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"TogetherAIException - {error_response['error']}",
                        model=model,
                        llm_provider="together_ai",
                        response=original_exception.response,
                    )
                elif (
                    "error" in error_response
                    and "invalid private key" in error_response["error"]
                ):
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"TogetherAIException - {error_response['error']}",
                        llm_provider="together_ai",
                        model=model,
                        response=original_exception.response,
                    )
                elif (
                    "error" in error_response
                    and "INVALID_ARGUMENT" in error_response["error"]
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"TogetherAIException - {error_response['error']}",
                        model=model,
                        llm_provider="together_ai",
                        response=original_exception.response,
                    )

                elif (
                    "error" in error_response
                    and "API key doesn't match expected format."
                    in error_response["error"]
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"TogetherAIException - {error_response['error']}",
                        model=model,
                        llm_provider="together_ai",
                        response=original_exception.response,
                    )
                elif (
                    "error_type" in error_response
                    and error_response["error_type"] == "validation"
                ):
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"TogetherAIException - {error_response['error']}",
                        model=model,
                        llm_provider="together_ai",
                        response=original_exception.response,
                    )
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"TogetherAIException - {original_exception.message}",
                            model=model,
                            llm_provider="together_ai",
                        )
                    elif original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"TogetherAIException - {error_response['error']}",
                            model=model,
                            llm_provider="together_ai",
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"TogetherAIException - {original_exception.message}",
                            llm_provider="together_ai",
                            model=model,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 524:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"TogetherAIException - {original_exception.message}",
                            llm_provider="together_ai",
                            model=model,
                        )
                else:
                    exception_mapping_worked = True
                    raise APIError(
                        status_code=original_exception.status_code,
                        message=f"TogetherAIException - {original_exception.message}",
                        llm_provider="together_ai",
                        model=model,
                        request=original_exception.request,
                    )
            elif custom_llm_provider == "aleph_alpha":
                if (
                    "This is longer than the model's maximum context length"
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"AlephAlphaException - {original_exception.message}",
                        llm_provider="aleph_alpha",
                        model=model,
                        response=original_exception.response,
                    )
                elif "InvalidToken" in error_str or "No token provided" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"AlephAlphaException - {original_exception.message}",
                        llm_provider="aleph_alpha",
                        model=model,
                        response=original_exception.response,
                    )
                elif hasattr(original_exception, "status_code"):
                    print_verbose(f"status code: {original_exception.status_code}")
                    if original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AlephAlphaException - {original_exception.message}",
                            llm_provider="aleph_alpha",
                            model=model,
                        )
                    elif original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"AlephAlphaException - {original_exception.message}",
                            llm_provider="aleph_alpha",
                            model=model,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"AlephAlphaException - {original_exception.message}",
                            llm_provider="aleph_alpha",
                            model=model,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 500:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"AlephAlphaException - {original_exception.message}",
                            llm_provider="aleph_alpha",
                            model=model,
                            response=original_exception.response,
                        )
                    raise original_exception
                raise original_exception
            elif (
                custom_llm_provider == "ollama" or custom_llm_provider == "ollama_chat"
            ):
                if isinstance(original_exception, dict):
                    error_str = original_exception.get("error", "")
                else:
                    error_str = str(original_exception)
                if "no such file or directory" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"OllamaException: Invalid Model/Model not loaded - {original_exception}",
                        model=model,
                        llm_provider="ollama",
                        response=original_exception.response,
                    )
                elif "Failed to establish a new connection" in error_str:
                    exception_mapping_worked = True
                    raise ServiceUnavailableError(
                        message=f"OllamaException: {original_exception}",
                        llm_provider="ollama",
                        model=model,
                        response=original_exception.response,
                    )
                elif "Invalid response object from API" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"OllamaException: {original_exception}",
                        llm_provider="ollama",
                        model=model,
                        response=original_exception.response,
                    )
                elif "Read timed out" in error_str:
                    exception_mapping_worked = True
                    raise Timeout(
                        message=f"OllamaException: {original_exception}",
                        llm_provider="ollama",
                        model=model,
                    )
            elif custom_llm_provider == "vllm":
                if hasattr(original_exception, "status_code"):
                    if original_exception.status_code == 0:
                        exception_mapping_worked = True
                        raise APIConnectionError(
                            message=f"VLLMException - {original_exception.message}",
                            llm_provider="vllm",
                            model=model,
                            request=original_exception.request,
                        )
            elif custom_llm_provider == "azure":
                if "Internal server error" in error_str:
                    exception_mapping_worked = True
                    raise litellm.InternalServerError(
                        message=f"AzureException Internal server error - {original_exception.message}",
                        llm_provider="azure",
                        model=model,
                        litellm_debug_info=extra_information,
                        response=httpx.Response(
                            status_code=400,
                            content=str(original_exception),
                            request=httpx.Request(method="completion", url="https://github.com/BerriAI/litellm"),  # type: ignore
                        ),
                    )
                elif "This model's maximum context length is" in error_str:
                    exception_mapping_worked = True
                    raise ContextWindowExceededError(
                        message=f"AzureException ContextWindowExceededError - {original_exception.message}",
                        llm_provider="azure",
                        model=model,
                        litellm_debug_info=extra_information,
                        response=original_exception.response,
                    )
                elif "DeploymentNotFound" in error_str:
                    exception_mapping_worked = True
                    raise NotFoundError(
                        message=f"AzureException NotFoundError - {original_exception.message}",
                        llm_provider="azure",
                        model=model,
                        litellm_debug_info=extra_information,
                        response=original_exception.response,
                    )
                elif (
                    (
                        "invalid_request_error" in error_str
                        and "content_policy_violation" in error_str
                    )
                    or (
                        "The response was filtered due to the prompt triggering Azure OpenAI's content management"
                        in error_str
                    )
                    or "Your task failed as a result of our safety system" in error_str
                ):
                    exception_mapping_worked = True
                    raise ContentPolicyViolationError(
                        message=f"litellm.ContentPolicyViolationError: AzureException - {original_exception.message}",
                        llm_provider="azure",
                        model=model,
                        litellm_debug_info=extra_information,
                        response=original_exception.response,
                    )
                elif "invalid_request_error" in error_str:
                    exception_mapping_worked = True
                    raise BadRequestError(
                        message=f"AzureException BadRequestError - {original_exception.message}",
                        llm_provider="azure",
                        model=model,
                        litellm_debug_info=extra_information,
                        response=original_exception.response,
                    )
                elif (
                    "The api_key client option must be set either by passing api_key to the client or by setting"
                    in error_str
                ):
                    exception_mapping_worked = True
                    raise AuthenticationError(
                        message=f"{exception_provider} AuthenticationError - {original_exception.message}",
                        llm_provider=custom_llm_provider,
                        model=model,
                        litellm_debug_info=extra_information,
                        response=original_exception.response,
                    )
                elif hasattr(original_exception, "status_code"):
                    exception_mapping_worked = True
                    if original_exception.status_code == 400:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"AzureException - {original_exception.message}",
                            llm_provider="azure",
                            model=model,
                            litellm_debug_info=extra_information,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 401:
                        exception_mapping_worked = True
                        raise AuthenticationError(
                            message=f"AzureException AuthenticationError - {original_exception.message}",
                            llm_provider="azure",
                            model=model,
                            litellm_debug_info=extra_information,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 408:
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"AzureException Timeout - {original_exception.message}",
                            model=model,
                            litellm_debug_info=extra_information,
                            llm_provider="azure",
                        )
                    elif original_exception.status_code == 422:
                        exception_mapping_worked = True
                        raise BadRequestError(
                            message=f"AzureException BadRequestError - {original_exception.message}",
                            model=model,
                            llm_provider="azure",
                            litellm_debug_info=extra_information,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 429:
                        exception_mapping_worked = True
                        raise RateLimitError(
                            message=f"AzureException RateLimitError - {original_exception.message}",
                            model=model,
                            llm_provider="azure",
                            litellm_debug_info=extra_information,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 503:
                        exception_mapping_worked = True
                        raise ServiceUnavailableError(
                            message=f"AzureException ServiceUnavailableError - {original_exception.message}",
                            model=model,
                            llm_provider="azure",
                            litellm_debug_info=extra_information,
                            response=original_exception.response,
                        )
                    elif original_exception.status_code == 504:  # gateway timeout error
                        exception_mapping_worked = True
                        raise Timeout(
                            message=f"AzureException Timeout - {original_exception.message}",
                            model=model,
                            litellm_debug_info=extra_information,
                            llm_provider="azure",
                        )
                    else:
                        exception_mapping_worked = True
                        raise APIError(
                            status_code=original_exception.status_code,
                            message=f"AzureException APIError - {original_exception.message}",
                            llm_provider="azure",
                            litellm_debug_info=extra_information,
                            model=model,
                            request=httpx.Request(
                                method="POST", url="https://openai.com/"
                            ),
                        )
                else:
                    # if no status code then it is an APIConnectionError: https://github.com/openai/openai-python#handling-errors
                    raise APIConnectionError(
                        message=f"{exception_provider} APIConnectionError - {message}",
                        llm_provider="azure",
                        model=model,
                        litellm_debug_info=extra_information,
                        request=httpx.Request(method="POST", url="https://openai.com/"),
                    )
        if (
            "BadRequestError.__init__() missing 1 required positional argument: 'param'"
            in str(original_exception)
        ):  # deal with edge-case invalid request error bug in openai-python sdk
            exception_mapping_worked = True
            raise BadRequestError(
                message=f"{exception_provider} BadRequestError : This can happen due to missing AZURE_API_VERSION: {str(original_exception)}",
                model=model,
                llm_provider=custom_llm_provider,
                response=original_exception.response,
            )
        else:  # ensure generic errors always return APIConnectionError=
            exception_mapping_worked = True
            if hasattr(original_exception, "request"):
                raise APIConnectionError(
                    message=f"{str(original_exception)}",
                    llm_provider=custom_llm_provider,
                    model=model,
                    request=original_exception.request,
                )
            else:
                raise APIConnectionError(
                    message=f"{str(original_exception)}",
                    llm_provider=custom_llm_provider,
                    model=model,
                    request=httpx.Request(
                        method="POST", url="https://api.openai.com/v1/"
                    ),  # stub the request
                )
    except Exception as e:
        # LOGGING
        exception_logging(
            logger_fn=user_logger_fn,
            additional_args={
                "exception_mapping_worked": exception_mapping_worked,
                "original_exception": original_exception,
            },
            exception=e,
        )
        ## AUTH ERROR
        if isinstance(e, AuthenticationError) and (
            litellm.email or "LITELLM_EMAIL" in os.environ
        ):
            threading.Thread(target=get_all_keys, args=(e.llm_provider,)).start()
        # don't let an error with mapping interrupt the user from receiving an error from the llm api calls
        if exception_mapping_worked:
            raise e
        else:
            raise original_exception


def get_or_generate_uuid():
    temp_dir = os.path.join(os.path.abspath(os.sep), "tmp")
    uuid_file = os.path.join(temp_dir, "litellm_uuid.txt")
    try:
        # Try to open the file and load the UUID
        with open(uuid_file, "r") as file:
            uuid_value = file.read()
            if uuid_value:
                uuid_value = uuid_value.strip()
            else:
                raise FileNotFoundError

    except FileNotFoundError:
        # Generate a new UUID if the file doesn't exist or is empty
        try:
            new_uuid = uuid.uuid4()
            uuid_value = str(new_uuid)
            with open(uuid_file, "w") as file:
                file.write(uuid_value)
        except:  # if writing to tmp/litellm_uuid.txt then retry writing to litellm_uuid.txt
            try:
                new_uuid = uuid.uuid4()
                uuid_value = str(new_uuid)
                with open("litellm_uuid.txt", "w") as file:
                    file.write(uuid_value)
            except:  # if this 3rd attempt fails just pass
                # Good first issue for someone to improve this function :)
                return
    except:
        # [Non-Blocking Error]
        return
    return uuid_value


######### Secret Manager ############################
# checks if user has passed in a secret manager client
# if passed in then checks the secret there
def _is_base64(s):
    try:
        return base64.b64encode(base64.b64decode(s)).decode() == s
    except binascii.Error:
        return False


def get_secret(
    secret_name: str,
    default_value: Optional[Union[str, bool]] = None,
):
    key_management_system = litellm._key_management_system
    key_management_settings = litellm._key_management_settings
    args = locals()

    if secret_name.startswith("os.environ/"):
        secret_name = secret_name.replace("os.environ/", "")

    # Example: oidc/google/https://bedrock-runtime.us-east-1.amazonaws.com/model/stability.stable-diffusion-xl-v1/invoke
    if secret_name.startswith("oidc/"):
        secret_name_split = secret_name.replace("oidc/", "")
        oidc_provider, oidc_aud = secret_name_split.split("/", 1)
        # TODO: Add caching for HTTP requests
        if oidc_provider == "google":
            oidc_token = oidc_cache.get_cache(key=secret_name)
            if oidc_token is not None:
                return oidc_token

            oidc_client = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))
            # https://cloud.google.com/compute/docs/instances/verifying-instance-identity#request_signature
            response = oidc_client.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity",
                params={"audience": oidc_aud},
                headers={"Metadata-Flavor": "Google"},
            )
            if response.status_code == 200:
                oidc_token = response.text
                oidc_cache.set_cache(key=secret_name, value=oidc_token, ttl=3600 - 60)
                return oidc_token
            else:
                raise ValueError("Google OIDC provider failed")
        elif oidc_provider == "circleci":
            # https://circleci.com/docs/openid-connect-tokens/
            env_secret = os.getenv("CIRCLE_OIDC_TOKEN")
            if env_secret is None:
                raise ValueError("CIRCLE_OIDC_TOKEN not found in environment")
            return env_secret
        elif oidc_provider == "circleci_v2":
            # https://circleci.com/docs/openid-connect-tokens/
            env_secret = os.getenv("CIRCLE_OIDC_TOKEN_V2")
            if env_secret is None:
                raise ValueError("CIRCLE_OIDC_TOKEN_V2 not found in environment")
            return env_secret
        elif oidc_provider == "github":
            # https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-cloud-providers#using-custom-actions
            actions_id_token_request_url = os.getenv("ACTIONS_ID_TOKEN_REQUEST_URL")
            actions_id_token_request_token = os.getenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
            if (
                actions_id_token_request_url is None
                or actions_id_token_request_token is None
            ):
                raise ValueError(
                    "ACTIONS_ID_TOKEN_REQUEST_URL or ACTIONS_ID_TOKEN_REQUEST_TOKEN not found in environment"
                )

            oidc_token = oidc_cache.get_cache(key=secret_name)
            if oidc_token is not None:
                return oidc_token

            oidc_client = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))
            response = oidc_client.get(
                actions_id_token_request_url,
                params={"audience": oidc_aud},
                headers={
                    "Authorization": f"Bearer {actions_id_token_request_token}",
                    "Accept": "application/json; api-version=2.0",
                },
            )
            if response.status_code == 200:
                oidc_token = response.text["value"]
                oidc_cache.set_cache(key=secret_name, value=oidc_token, ttl=300 - 5)
                return oidc_token
            else:
                raise ValueError("Github OIDC provider failed")
        else:
            raise ValueError("Unsupported OIDC provider")

    try:
        if litellm.secret_manager_client is not None:
            try:
                client = litellm.secret_manager_client
                key_manager = "local"
                if key_management_system is not None:
                    key_manager = key_management_system.value

                if key_management_settings is not None:
                    if (
                        secret_name not in key_management_settings.hosted_keys
                    ):  # allow user to specify which keys to check in hosted key manager
                        key_manager = "local"

                if (
                    key_manager == KeyManagementSystem.AZURE_KEY_VAULT.value
                    or type(client).__module__ + "." + type(client).__name__
                    == "azure.keyvault.secrets._client.SecretClient"
                ):  # support Azure Secret Client - from azure.keyvault.secrets import SecretClient
                    secret = client.get_secret(secret_name).value
                elif (
                    key_manager == KeyManagementSystem.GOOGLE_KMS.value
                    or client.__class__.__name__ == "KeyManagementServiceClient"
                ):
                    encrypted_secret: Any = os.getenv(secret_name)
                    if encrypted_secret is None:
                        raise ValueError(
                            f"Google KMS requires the encrypted secret to be in the environment!"
                        )
                    b64_flag = _is_base64(encrypted_secret)
                    if b64_flag == True:  # if passed in as encoded b64 string
                        encrypted_secret = base64.b64decode(encrypted_secret)
                    if not isinstance(encrypted_secret, bytes):
                        # If it's not, assume it's a string and encode it to bytes
                        ciphertext = eval(
                            encrypted_secret.encode()
                        )  # assuming encrypted_secret is something like - b'\n$\x00D\xac\xb4/t)07\xe5\xf6..'
                    else:
                        ciphertext = encrypted_secret

                    response = client.decrypt(
                        request={
                            "name": litellm._google_kms_resource_name,
                            "ciphertext": ciphertext,
                        }
                    )
                    secret = response.plaintext.decode(
                        "utf-8"
                    )  # assumes the original value was encoded with utf-8
                elif key_manager == KeyManagementSystem.AWS_KMS.value:
                    """
                    Only check the tokens which start with 'aws_kms/'. This prevents latency impact caused by checking all keys.
                    """
                    encrypted_value = os.getenv(secret_name, None)
                    if encrypted_value is None:
                        raise Exception("encrypted value for AWS KMS cannot be None.")
                    # Decode the base64 encoded ciphertext
                    ciphertext_blob = base64.b64decode(encrypted_value)

                    # Set up the parameters for the decrypt call
                    params = {"CiphertextBlob": ciphertext_blob}

                    # Perform the decryption
                    response = client.decrypt(**params)

                    # Extract and decode the plaintext
                    plaintext = response["Plaintext"]
                    secret = plaintext.decode("utf-8")
                elif key_manager == KeyManagementSystem.AWS_SECRET_MANAGER.value:
                    try:
                        get_secret_value_response = client.get_secret_value(
                            SecretId=secret_name
                        )
                        print_verbose(
                            f"get_secret_value_response: {get_secret_value_response}"
                        )
                    except Exception as e:
                        print_verbose(f"An error occurred - {str(e)}")
                        # For a list of exceptions thrown, see
                        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
                        raise e

                    # assume there is 1 secret per secret_name
                    secret_dict = json.loads(get_secret_value_response["SecretString"])
                    print_verbose(f"secret_dict: {secret_dict}")
                    for k, v in secret_dict.items():
                        secret = v
                    print_verbose(f"secret: {secret}")
                elif key_manager == "local":
                    secret = os.getenv(secret_name)
                else:  # assume the default is infisicial client
                    secret = client.get_secret(secret_name).secret_value
            except Exception as e:  # check if it's in os.environ
                verbose_logger.error(
                    f"An exception occurred - {str(e)}\n\n{traceback.format_exc()}"
                )
                secret = os.getenv(secret_name)
            try:
                secret_value_as_bool = ast.literal_eval(secret)
                if isinstance(secret_value_as_bool, bool):
                    return secret_value_as_bool
                else:
                    return secret
            except:
                return secret
        else:
            secret = os.environ.get(secret_name)
            try:
                secret_value_as_bool = (
                    ast.literal_eval(secret) if secret is not None else None
                )
                if isinstance(secret_value_as_bool, bool):
                    return secret_value_as_bool
                else:
                    return secret
            except:
                return secret
    except Exception as e:
        if default_value is not None:
            return default_value
        else:
            raise e


######## Streaming Class ############################
# wraps the completion stream to return the correct format for the model
# replicate/anthropic/cohere
class CustomStreamWrapper:
    def __init__(
        self,
        completion_stream,
        model,
        custom_llm_provider=None,
        logging_obj=None,
        stream_options=None,
        make_call: Optional[Callable] = None,
    ):
        self.model = model
        self.make_call = make_call
        self.custom_llm_provider = custom_llm_provider
        self.logging_obj = logging_obj
        self.completion_stream = completion_stream
        self.sent_first_chunk = False
        self.sent_last_chunk = False
        self.system_fingerprint: Optional[str] = None
        self.received_finish_reason: Optional[str] = None
        self.special_tokens = [
            "<|assistant|>",
            "<|system|>",
            "<|user|>",
            "<s>",
            "</s>",
            "<|im_end|>",
        ]
        self.holding_chunk = ""
        self.complete_response = ""
        self.response_uptil_now = ""
        _model_info = (
            self.logging_obj.model_call_details.get("litellm_params", {}).get(
                "model_info", {}
            )
            or {}
        )
        self._hidden_params = {
            "model_id": (_model_info.get("id", None))
        }  # returned as x-litellm-model-id response header in proxy
        self.response_id = None
        self.logging_loop = None
        self.rules = Rules()
        self.stream_options = stream_options or getattr(
            logging_obj, "stream_options", None
        )
        self.messages = getattr(logging_obj, "messages", None)
        self.sent_stream_usage = False
        self.chunks: List = (
            []
        )  # keep track of the returned chunks - used for calculating the input/output tokens for stream options

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def process_chunk(self, chunk: str):
        """
        NLP Cloud streaming returns the entire response, for each chunk. Process this, to only return the delta.
        """
        try:
            chunk = chunk.strip()
            self.complete_response = self.complete_response.strip()

            if chunk.startswith(self.complete_response):
                # Remove last_sent_chunk only if it appears at the start of the new chunk
                chunk = chunk[len(self.complete_response) :]

            self.complete_response += chunk
            return chunk
        except Exception as e:
            raise e

    def check_special_tokens(self, chunk: str, finish_reason: Optional[str]):
        """
        Output parse <s> / </s> special tokens for sagemaker + hf streaming.
        """
        hold = False
        if (
            self.custom_llm_provider != "huggingface"
            and self.custom_llm_provider != "sagemaker"
        ):
            return hold, chunk

        if finish_reason:
            for token in self.special_tokens:
                if token in chunk:
                    chunk = chunk.replace(token, "")
            return hold, chunk

        if self.sent_first_chunk is True:
            return hold, chunk

        curr_chunk = self.holding_chunk + chunk
        curr_chunk = curr_chunk.strip()

        for token in self.special_tokens:
            if len(curr_chunk) < len(token) and curr_chunk in token:
                hold = True
                self.holding_chunk = curr_chunk
            elif len(curr_chunk) >= len(token):
                if token in curr_chunk:
                    self.holding_chunk = curr_chunk.replace(token, "")
                    hold = True
            else:
                pass

        if hold is False:  # reset
            self.holding_chunk = ""
        return hold, curr_chunk

    def handle_anthropic_text_chunk(self, chunk):
        str_line = chunk
        if isinstance(chunk, bytes):  # Handle binary data
            str_line = chunk.decode("utf-8")  # Convert bytes to string
        text = ""
        is_finished = False
        finish_reason = None
        if str_line.startswith("data:"):
            data_json = json.loads(str_line[5:])
            type_chunk = data_json.get("type", None)
            if type_chunk == "completion":
                text = data_json.get("completion")
                finish_reason = data_json.get("stop_reason")
                if finish_reason is not None:
                    is_finished = True
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        elif "error" in str_line:
            raise ValueError(f"Unable to parse response. Original response: {str_line}")
        else:
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }

    def handle_anthropic_chunk(self, chunk):
        str_line = chunk
        if isinstance(chunk, bytes):  # Handle binary data
            str_line = chunk.decode("utf-8")  # Convert bytes to string
        text = ""
        is_finished = False
        finish_reason = None
        if str_line.startswith("data:"):
            data_json = json.loads(str_line[5:])
            type_chunk = data_json.get("type", None)
            if type_chunk == "content_block_delta":
                """
                Anthropic content chunk
                chunk = {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Hello'}}
                """
                text = data_json.get("delta", {}).get("text", "")
            elif type_chunk == "message_delta":
                """
                Anthropic
                chunk = {'type': 'message_delta', 'delta': {'stop_reason': 'max_tokens', 'stop_sequence': None}, 'usage': {'output_tokens': 10}}
                """
                # TODO - get usage from this chunk, set in response
                finish_reason = data_json.get("delta", {}).get("stop_reason", None)
                is_finished = True
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        elif "error" in str_line:
            raise ValueError(f"Unable to parse response. Original response: {str_line}")
        else:
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }

    def handle_vertexai_anthropic_chunk(self, chunk):
        """
        - MessageStartEvent(message=Message(id='msg_01LeRRgvX4gwkX3ryBVgtuYZ', content=[], model='claude-3-sonnet-20240229', role='assistant', stop_reason=None, stop_sequence=None, type='message', usage=Usage(input_tokens=8, output_tokens=1)), type='message_start'); custom_llm_provider: vertex_ai
        - ContentBlockStartEvent(content_block=ContentBlock(text='', type='text'), index=0, type='content_block_start'); custom_llm_provider: vertex_ai
        - ContentBlockDeltaEvent(delta=TextDelta(text='Hello', type='text_delta'), index=0, type='content_block_delta'); custom_llm_provider: vertex_ai
        """
        text = ""
        prompt_tokens = None
        completion_tokens = None
        is_finished = False
        finish_reason = None
        type_chunk = getattr(chunk, "type", None)
        if type_chunk == "message_start":
            message = getattr(chunk, "message", None)
            text = ""  # lets us return a chunk with usage to user
            _usage = getattr(message, "usage", None)
            if _usage is not None:
                prompt_tokens = getattr(_usage, "input_tokens", None)
                completion_tokens = getattr(_usage, "output_tokens", None)
        elif type_chunk == "content_block_delta":
            """
            Anthropic content chunk
            chunk = {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Hello'}}
            """
            delta = getattr(chunk, "delta", None)
            if delta is not None:
                text = getattr(delta, "text", "")
            else:
                text = ""
        elif type_chunk == "message_delta":
            """
            Anthropic
            chunk = {'type': 'message_delta', 'delta': {'stop_reason': 'max_tokens', 'stop_sequence': None}, 'usage': {'output_tokens': 10}}
            """
            # TODO - get usage from this chunk, set in response
            delta = getattr(chunk, "delta", None)
            if delta is not None:
                finish_reason = getattr(delta, "stop_reason", "stop")
                is_finished = True
            _usage = getattr(chunk, "usage", None)
            if _usage is not None:
                prompt_tokens = getattr(_usage, "input_tokens", None)
                completion_tokens = getattr(_usage, "output_tokens", None)

        return {
            "text": text,
            "is_finished": is_finished,
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def handle_together_ai_chunk(self, chunk):
        chunk = chunk.decode("utf-8")
        text = ""
        is_finished = False
        finish_reason = None
        if "text" in chunk:
            text_index = chunk.find('"text":"')  # this checks if text: exists
            text_start = text_index + len('"text":"')
            text_end = chunk.find('"}', text_start)
            if text_index != -1 and text_end != -1:
                extracted_text = chunk[text_start:text_end]
                text = extracted_text
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        elif "[DONE]" in chunk:
            return {"text": text, "is_finished": True, "finish_reason": "stop"}
        elif "error" in chunk:
            raise litellm.together_ai.TogetherAIError(
                status_code=422, message=f"{str(chunk)}"
            )
        else:
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }

    def handle_predibase_chunk(self, chunk):
        try:
            if type(chunk) != str:
                chunk = chunk.decode(
                    "utf-8"
                )  # DO NOT REMOVE this: This is required for HF inference API + Streaming
            text = ""
            is_finished = False
            finish_reason = ""
            print_verbose(f"chunk: {chunk}")
            if chunk.startswith("data:"):
                data_json = json.loads(chunk[5:])
                print_verbose(f"data json: {data_json}")
                if "token" in data_json and "text" in data_json["token"]:
                    text = data_json["token"]["text"]
                if data_json.get("details", False) and data_json["details"].get(
                    "finish_reason", False
                ):
                    is_finished = True
                    finish_reason = data_json["details"]["finish_reason"]
                elif data_json.get(
                    "generated_text", False
                ):  # if full generated text exists, then stream is complete
                    text = ""  # don't return the final bos token
                    is_finished = True
                    finish_reason = "stop"
                elif data_json.get("error", False):
                    raise Exception(data_json.get("error"))
                return {
                    "text": text,
                    "is_finished": is_finished,
                    "finish_reason": finish_reason,
                }
            elif "error" in chunk:
                raise ValueError(chunk)
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except Exception as e:
            verbose_logger.error(
                "litellm.CustomStreamWrapper.handle_predibase_chunk(): Exception occured - {}".format(
                    str(e)
                )
            )
            verbose_logger.debug(traceback.format_exc())
            raise e

    def handle_huggingface_chunk(self, chunk):
        try:
            if type(chunk) != str:
                chunk = chunk.decode(
                    "utf-8"
                )  # DO NOT REMOVE this: This is required for HF inference API + Streaming
            text = ""
            is_finished = False
            finish_reason = ""
            print_verbose(f"chunk: {chunk}")
            if chunk.startswith("data:"):
                data_json = json.loads(chunk[5:])
                print_verbose(f"data json: {data_json}")
                if "token" in data_json and "text" in data_json["token"]:
                    text = data_json["token"]["text"]
                if data_json.get("details", False) and data_json["details"].get(
                    "finish_reason", False
                ):
                    is_finished = True
                    finish_reason = data_json["details"]["finish_reason"]
                elif data_json.get(
                    "generated_text", False
                ):  # if full generated text exists, then stream is complete
                    text = ""  # don't return the final bos token
                    is_finished = True
                    finish_reason = "stop"
                elif data_json.get("error", False):
                    raise Exception(data_json.get("error"))
                return {
                    "text": text,
                    "is_finished": is_finished,
                    "finish_reason": finish_reason,
                }
            elif "error" in chunk:
                raise ValueError(chunk)
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except Exception as e:
            verbose_logger.error(
                "litellm.CustomStreamWrapper.handle_huggingface_chunk(): Exception occured - {}".format(
                    str(e)
                )
            )
            verbose_logger.debug(traceback.format_exc())
            raise e

    def handle_ai21_chunk(self, chunk):  # fake streaming
        chunk = chunk.decode("utf-8")
        data_json = json.loads(chunk)
        try:
            text = data_json["completions"][0]["data"]["text"]
            is_finished = True
            finish_reason = "stop"
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def handle_maritalk_chunk(self, chunk):  # fake streaming
        chunk = chunk.decode("utf-8")
        data_json = json.loads(chunk)
        try:
            text = data_json["answer"]
            is_finished = True
            finish_reason = "stop"
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def handle_nlp_cloud_chunk(self, chunk):
        text = ""
        is_finished = False
        finish_reason = ""
        try:
            if "dolphin" in self.model:
                chunk = self.process_chunk(chunk=chunk)
            else:
                data_json = json.loads(chunk)
                chunk = data_json["generated_text"]
            text = chunk
            if "[DONE]" in text:
                text = text.replace("[DONE]", "")
                is_finished = True
                finish_reason = "stop"
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except Exception as e:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def handle_aleph_alpha_chunk(self, chunk):
        chunk = chunk.decode("utf-8")
        data_json = json.loads(chunk)
        try:
            text = data_json["completions"][0]["completion"]
            is_finished = True
            finish_reason = "stop"
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def handle_cohere_chunk(self, chunk):
        chunk = chunk.decode("utf-8")
        data_json = json.loads(chunk)
        try:
            text = ""
            is_finished = False
            finish_reason = ""
            if "text" in data_json:
                text = data_json["text"]
            elif "is_finished" in data_json:
                is_finished = data_json["is_finished"]
                finish_reason = data_json["finish_reason"]
            else:
                raise Exception(data_json)
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def handle_cohere_chat_chunk(self, chunk):
        chunk = chunk.decode("utf-8")
        data_json = json.loads(chunk)
        print_verbose(f"chunk: {chunk}")
        try:
            text = ""
            is_finished = False
            finish_reason = ""
            if "text" in data_json:
                text = data_json["text"]
            elif "is_finished" in data_json and data_json["is_finished"] == True:
                is_finished = data_json["is_finished"]
                finish_reason = data_json["finish_reason"]
            else:
                return
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def handle_azure_chunk(self, chunk):
        is_finished = False
        finish_reason = ""
        text = ""
        print_verbose(f"chunk: {chunk}")
        if "data: [DONE]" in chunk:
            text = ""
            is_finished = True
            finish_reason = "stop"
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        elif chunk.startswith("data:"):
            data_json = json.loads(chunk[5:])  # chunk.startswith("data:"):
            try:
                if len(data_json["choices"]) > 0:
                    delta = data_json["choices"][0]["delta"]
                    text = "" if delta is None else delta.get("content", "")
                    if data_json["choices"][0].get("finish_reason", None):
                        is_finished = True
                        finish_reason = data_json["choices"][0]["finish_reason"]
                print_verbose(
                    f"text: {text}; is_finished: {is_finished}; finish_reason: {finish_reason}"
                )
                return {
                    "text": text,
                    "is_finished": is_finished,
                    "finish_reason": finish_reason,
                }
            except:
                raise ValueError(
                    f"Unable to parse response. Original response: {chunk}"
                )
        elif "error" in chunk:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")
        else:
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }

    def handle_replicate_chunk(self, chunk):
        try:
            text = ""
            is_finished = False
            finish_reason = ""
            if "output" in chunk:
                text = chunk["output"]
            if "status" in chunk:
                if chunk["status"] == "succeeded":
                    is_finished = True
                    finish_reason = "stop"
            elif chunk.get("error", None):
                raise Exception(chunk["error"])
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        except:
            raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def handle_openai_chat_completion_chunk(self, chunk):
        try:
            print_verbose(f"\nRaw OpenAI Chunk\n{chunk}\n")
            str_line = chunk
            text = ""
            is_finished = False
            finish_reason = None
            logprobs = None
            usage = None
            original_chunk = None  # this is used for function/tool calling
            if len(str_line.choices) > 0:
                if (
                    str_line.choices[0].delta is not None
                    and str_line.choices[0].delta.content is not None
                ):
                    text = str_line.choices[0].delta.content
                else:  # function/tool calling chunk - when content is None. in this case we just return the original chunk from openai
                    original_chunk = str_line
                if str_line.choices[0].finish_reason:
                    is_finished = True
                    finish_reason = str_line.choices[0].finish_reason
                    if finish_reason == "content_filter":
                        if hasattr(str_line.choices[0], "content_filter_result"):
                            error_message = json.dumps(
                                str_line.choices[0].content_filter_result
                            )
                        else:
                            error_message = "Azure Response={}".format(
                                str(dict(str_line))
                            )
                        raise litellm.AzureOpenAIError(
                            status_code=400, message=error_message
                        )

                # checking for logprobs
                if (
                    hasattr(str_line.choices[0], "logprobs")
                    and str_line.choices[0].logprobs is not None
                ):
                    logprobs = str_line.choices[0].logprobs
                else:
                    logprobs = None

            usage = getattr(str_line, "usage", None)

            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
                "logprobs": logprobs,
                "original_chunk": str_line,
                "usage": usage,
            }
        except Exception as e:
            verbose_logger.error(
                "litellm.CustomStreamWrapper.handle_openai_chat_completion_chunk(): Exception occured - {}".format(
                    str(e)
                )
            )
            verbose_logger.debug(traceback.format_exc())
            raise e

    def handle_azure_text_completion_chunk(self, chunk):
        try:
            print_verbose(f"\nRaw OpenAI Chunk\n{chunk}\n")
            text = ""
            is_finished = False
            finish_reason = None
            choices = getattr(chunk, "choices", [])
            if len(choices) > 0:
                text = choices[0].text
                if choices[0].finish_reason is not None:
                    is_finished = True
                    finish_reason = choices[0].finish_reason
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }

        except Exception as e:
            raise e

    def handle_openai_text_completion_chunk(self, chunk):
        try:
            print_verbose(f"\nRaw OpenAI Chunk\n{chunk}\n")
            text = ""
            is_finished = False
            finish_reason = None
            usage = None
            choices = getattr(chunk, "choices", [])
            if len(choices) > 0:
                text = choices[0].text
                if choices[0].finish_reason is not None:
                    is_finished = True
                    finish_reason = choices[0].finish_reason
            usage = getattr(chunk, "usage", None)
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
                "usage": usage,
            }

        except Exception as e:
            raise e

    def handle_baseten_chunk(self, chunk):
        try:
            chunk = chunk.decode("utf-8")
            if len(chunk) > 0:
                if chunk.startswith("data:"):
                    data_json = json.loads(chunk[5:])
                    if "token" in data_json and "text" in data_json["token"]:
                        return data_json["token"]["text"]
                    else:
                        return ""
                data_json = json.loads(chunk)
                if "model_output" in data_json:
                    if (
                        isinstance(data_json["model_output"], dict)
                        and "data" in data_json["model_output"]
                        and isinstance(data_json["model_output"]["data"], list)
                    ):
                        return data_json["model_output"]["data"][0]
                    elif isinstance(data_json["model_output"], str):
                        return data_json["model_output"]
                    elif "completion" in data_json and isinstance(
                        data_json["completion"], str
                    ):
                        return data_json["completion"]
                    else:
                        raise ValueError(
                            f"Unable to parse response. Original response: {chunk}"
                        )
                else:
                    return ""
            else:
                return ""
        except:
            verbose_logger.error(
                "litellm.CustomStreamWrapper.handle_baseten_chunk(): Exception occured - {}".format(
                    str(e)
                )
            )
            verbose_logger.debug(traceback.format_exc())
            return ""

    def handle_cloudlfare_stream(self, chunk):
        try:
            print_verbose(f"\nRaw OpenAI Chunk\n{chunk}\n")
            chunk = chunk.decode("utf-8")
            str_line = chunk
            text = ""
            is_finished = False
            finish_reason = None

            if "[DONE]" in chunk:
                return {"text": text, "is_finished": True, "finish_reason": "stop"}
            elif str_line.startswith("data:"):
                data_json = json.loads(str_line[5:])
                print_verbose(f"delta content: {data_json}")
                text = data_json["response"]
                return {
                    "text": text,
                    "is_finished": is_finished,
                    "finish_reason": finish_reason,
                }
            else:
                return {
                    "text": text,
                    "is_finished": is_finished,
                    "finish_reason": finish_reason,
                }

        except Exception as e:
            raise e

    def handle_ollama_stream(self, chunk):
        try:
            if isinstance(chunk, dict):
                json_chunk = chunk
            else:
                json_chunk = json.loads(chunk)
            if "error" in json_chunk:
                raise Exception(f"Ollama Error - {json_chunk}")

            text = ""
            is_finished = False
            finish_reason = None
            if json_chunk["done"] == True:
                text = ""
                is_finished = True
                finish_reason = "stop"
                return {
                    "text": text,
                    "is_finished": is_finished,
                    "finish_reason": finish_reason,
                }
            elif json_chunk["response"]:
                print_verbose(f"delta content: {json_chunk}")
                text = json_chunk["response"]
                return {
                    "text": text,
                    "is_finished": is_finished,
                    "finish_reason": finish_reason,
                }
            else:
                raise Exception(f"Ollama Error - {json_chunk}")
        except Exception as e:
            raise e

    def handle_ollama_chat_stream(self, chunk):
        # for ollama_chat/ provider
        try:
            if isinstance(chunk, dict):
                json_chunk = chunk
            else:
                json_chunk = json.loads(chunk)
            if "error" in json_chunk:
                raise Exception(f"Ollama Error - {json_chunk}")

            text = ""
            is_finished = False
            finish_reason = None
            if json_chunk["done"] == True:
                text = ""
                is_finished = True
                finish_reason = "stop"
                return {
                    "text": text,
                    "is_finished": is_finished,
                    "finish_reason": finish_reason,
                }
            elif "message" in json_chunk:
                print_verbose(f"delta content: {json_chunk}")
                text = json_chunk["message"]["content"]
                return {
                    "text": text,
                    "is_finished": is_finished,
                    "finish_reason": finish_reason,
                }
            else:
                raise Exception(f"Ollama Error - {json_chunk}")
        except Exception as e:
            raise e

    def handle_bedrock_stream(self, chunk):
        return {
            "text": chunk["text"],
            "is_finished": chunk["is_finished"],
            "finish_reason": chunk["finish_reason"],
        }

    def handle_sagemaker_stream(self, chunk):
        if "data: [DONE]" in chunk:
            text = ""
            is_finished = True
            finish_reason = "stop"
            return {
                "text": text,
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        elif isinstance(chunk, dict):
            if chunk["is_finished"] == True:
                finish_reason = "stop"
            else:
                finish_reason = ""
            return {
                "text": chunk["text"],
                "is_finished": chunk["is_finished"],
                "finish_reason": finish_reason,
            }

    def handle_watsonx_stream(self, chunk):
        try:
            if isinstance(chunk, dict):
                parsed_response = chunk
            elif isinstance(chunk, (str, bytes)):
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8")
                if "generated_text" in chunk:
                    response = chunk.replace("data: ", "").strip()
                    parsed_response = json.loads(response)
                else:
                    return {
                        "text": "",
                        "is_finished": False,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    }
            else:
                print_verbose(f"chunk: {chunk} (Type: {type(chunk)})")
                raise ValueError(
                    f"Unable to parse response. Original response: {chunk}"
                )
            results = parsed_response.get("results", [])
            if len(results) > 0:
                text = results[0].get("generated_text", "")
                finish_reason = results[0].get("stop_reason")
                is_finished = finish_reason != "not_finished"
                return {
                    "text": text,
                    "is_finished": is_finished,
                    "finish_reason": finish_reason,
                    "prompt_tokens": results[0].get("input_token_count", 0),
                    "completion_tokens": results[0].get("generated_token_count", 0),
                }
            return {"text": "", "is_finished": False}
        except Exception as e:
            raise e

    def handle_clarifai_completion_chunk(self, chunk):
        try:
            if isinstance(chunk, dict):
                parsed_response = chunk
            if isinstance(chunk, (str, bytes)):
                if isinstance(chunk, bytes):
                    parsed_response = chunk.decode("utf-8")
                else:
                    parsed_response = chunk
            data_json = json.loads(parsed_response)
            text = (
                data_json.get("outputs", "")[0]
                .get("data", "")
                .get("text", "")
                .get("raw", "")
            )
            prompt_tokens = len(
                encoding.encode(
                    data_json.get("outputs", "")[0]
                    .get("input", "")
                    .get("data", "")
                    .get("text", "")
                    .get("raw", "")
                )
            )
            completion_tokens = len(encoding.encode(text))
            return {
                "text": text,
                "is_finished": True,
            }
        except:
            verbose_logger.error(
                "litellm.CustomStreamWrapper.handle_clarifai_chunk(): Exception occured - {}".format(
                    str(e)
                )
            )
            verbose_logger.debug(traceback.format_exc())
            return ""

    def model_response_creator(self):
        _model = self.model
        _received_llm_provider = self.custom_llm_provider
        _logging_obj_llm_provider = self.logging_obj.model_call_details.get("custom_llm_provider", None)  # type: ignore
        if (
            _received_llm_provider == "openai"
            and _received_llm_provider != _logging_obj_llm_provider
        ):
            _model = "{}/{}".format(_logging_obj_llm_provider, _model)
        model_response = ModelResponse(
            stream=True, model=_model, stream_options=self.stream_options
        )
        if self.response_id is not None:
            model_response.id = self.response_id
        else:
            self.response_id = model_response.id
        if self.system_fingerprint is not None:
            model_response.system_fingerprint = self.system_fingerprint
        model_response._hidden_params["custom_llm_provider"] = _logging_obj_llm_provider
        model_response._hidden_params["created_at"] = time.time()
        model_response.choices = [StreamingChoices(finish_reason=None)]
        return model_response

    def is_delta_empty(self, delta: Delta) -> bool:
        is_empty = True
        if delta.content is not None:
            is_empty = False
        elif delta.tool_calls is not None:
            is_empty = False
        elif delta.function_call is not None:
            is_empty = False
        return is_empty

    def chunk_creator(self, chunk):
        model_response = self.model_response_creator()
        response_obj = {}
        try:
            # return this for all models
            completion_obj = {"content": ""}
            if self.custom_llm_provider and self.custom_llm_provider == "anthropic":
                response_obj = self.handle_anthropic_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif (
                self.custom_llm_provider
                and self.custom_llm_provider == "anthropic_text"
            ):
                response_obj = self.handle_anthropic_text_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider and self.custom_llm_provider == "clarifai":
                response_obj = self.handle_clarifai_completion_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.model == "replicate" or self.custom_llm_provider == "replicate":
                response_obj = self.handle_replicate_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider and self.custom_llm_provider == "together_ai":
                response_obj = self.handle_together_ai_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider and self.custom_llm_provider == "huggingface":
                response_obj = self.handle_huggingface_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider and self.custom_llm_provider == "predibase":
                response_obj = self.handle_predibase_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif (
                self.custom_llm_provider and self.custom_llm_provider == "baseten"
            ):  # baseten doesn't provide streaming
                completion_obj["content"] = self.handle_baseten_chunk(chunk)
            elif (
                self.custom_llm_provider and self.custom_llm_provider == "ai21"
            ):  # ai21 doesn't provide streaming
                response_obj = self.handle_ai21_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider and self.custom_llm_provider == "maritalk":
                response_obj = self.handle_maritalk_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider and self.custom_llm_provider == "vllm":
                completion_obj["content"] = chunk[0].outputs[0].text
            elif (
                self.custom_llm_provider and self.custom_llm_provider == "aleph_alpha"
            ):  # aleph alpha doesn't provide streaming
                response_obj = self.handle_aleph_alpha_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider == "nlp_cloud":
                try:
                    response_obj = self.handle_nlp_cloud_chunk(chunk)
                    completion_obj["content"] = response_obj["text"]
                    if response_obj["is_finished"]:
                        self.received_finish_reason = response_obj["finish_reason"]
                except Exception as e:
                    if self.received_finish_reason:
                        raise e
                    else:
                        if self.sent_first_chunk is False:
                            raise Exception("An unknown error occurred with the stream")
                        self.received_finish_reason = "stop"
            elif self.custom_llm_provider == "gemini":
                if hasattr(chunk, "parts") == True:
                    try:
                        if len(chunk.parts) > 0:
                            completion_obj["content"] = chunk.parts[0].text
                        if len(chunk.parts) > 0 and hasattr(
                            chunk.parts[0], "finish_reason"
                        ):
                            self.received_finish_reason = chunk.parts[
                                0
                            ].finish_reason.name
                    except:
                        if chunk.parts[0].finish_reason.name == "SAFETY":
                            raise Exception(
                                f"The response was blocked by VertexAI. {str(chunk)}"
                            )
                else:
                    completion_obj["content"] = str(chunk)
            elif self.custom_llm_provider and (self.custom_llm_provider == "vertex_ai"):
                import proto  # type: ignore

                if self.model.startswith("claude-3"):
                    response_obj = self.handle_vertexai_anthropic_chunk(chunk=chunk)
                    if response_obj is None:
                        return
                    completion_obj["content"] = response_obj["text"]
                    setattr(model_response, "usage", Usage())
                    if response_obj.get("prompt_tokens", None) is not None:
                        model_response.usage.prompt_tokens = response_obj[
                            "prompt_tokens"
                        ]
                    if response_obj.get("completion_tokens", None) is not None:
                        model_response.usage.completion_tokens = response_obj[
                            "completion_tokens"
                        ]
                    if hasattr(model_response.usage, "prompt_tokens"):
                        model_response.usage.total_tokens = (
                            getattr(model_response.usage, "total_tokens", 0)
                            + model_response.usage.prompt_tokens
                        )
                    if hasattr(model_response.usage, "completion_tokens"):
                        model_response.usage.total_tokens = (
                            getattr(model_response.usage, "total_tokens", 0)
                            + model_response.usage.completion_tokens
                        )

                    if response_obj["is_finished"]:
                        self.received_finish_reason = response_obj["finish_reason"]
                elif hasattr(chunk, "candidates") == True:
                    try:
                        try:
                            completion_obj["content"] = chunk.text
                        except Exception as e:
                            if "Part has no text." in str(e):
                                ## check for function calling
                                function_call = (
                                    chunk.candidates[0].content.parts[0].function_call
                                )

                                args_dict = {}

                                # Check if it's a RepeatedComposite instance
                                for key, val in function_call.args.items():
                                    if isinstance(
                                        val,
                                        proto.marshal.collections.repeated.RepeatedComposite,
                                    ):
                                        # If so, convert to list
                                        args_dict[key] = [v for v in val]
                                    else:
                                        args_dict[key] = val

                                try:
                                    args_str = json.dumps(args_dict)
                                except Exception as e:
                                    raise e
                                _delta_obj = litellm.utils.Delta(
                                    content=None,
                                    tool_calls=[
                                        {
                                            "id": f"call_{str(uuid.uuid4())}",
                                            "function": {
                                                "arguments": args_str,
                                                "name": function_call.name,
                                            },
                                            "type": "function",
                                        }
                                    ],
                                )
                                _streaming_response = StreamingChoices(delta=_delta_obj)
                                _model_response = ModelResponse(stream=True)
                                _model_response.choices = [_streaming_response]
                                response_obj = {"original_chunk": _model_response}
                            else:
                                raise e
                        if (
                            hasattr(chunk.candidates[0], "finish_reason")
                            and chunk.candidates[0].finish_reason.name
                            != "FINISH_REASON_UNSPECIFIED"
                        ):  # every non-final chunk in vertex ai has this
                            self.received_finish_reason = chunk.candidates[
                                0
                            ].finish_reason.name
                    except Exception as e:
                        if chunk.candidates[0].finish_reason.name == "SAFETY":
                            raise Exception(
                                f"The response was blocked by VertexAI. {str(chunk)}"
                            )
                else:
                    completion_obj["content"] = str(chunk)
            elif self.custom_llm_provider == "cohere":
                response_obj = self.handle_cohere_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider == "cohere_chat":
                response_obj = self.handle_cohere_chat_chunk(chunk)
                if response_obj is None:
                    return
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider == "bedrock":
                from litellm.types.llms.bedrock import GenericStreamingChunk

                if self.received_finish_reason is not None:
                    raise StopIteration
                response_obj: GenericStreamingChunk = chunk
                completion_obj["content"] = response_obj["text"]

                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]

                if (
                    self.stream_options
                    and self.stream_options.get("include_usage", False) is True
                    and response_obj["usage"] is not None
                ):
                    self.sent_stream_usage = True
                    model_response.usage = litellm.Usage(
                        prompt_tokens=response_obj["usage"]["inputTokens"],
                        completion_tokens=response_obj["usage"]["outputTokens"],
                        total_tokens=response_obj["usage"]["totalTokens"],
                    )
            elif self.custom_llm_provider == "sagemaker":
                print_verbose(f"ENTERS SAGEMAKER STREAMING for chunk {chunk}")
                response_obj = self.handle_sagemaker_stream(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider == "petals":
                if len(self.completion_stream) == 0:
                    if self.received_finish_reason is not None:
                        raise StopIteration
                    else:
                        self.received_finish_reason = "stop"
                chunk_size = 30
                new_chunk = self.completion_stream[:chunk_size]
                completion_obj["content"] = new_chunk
                self.completion_stream = self.completion_stream[chunk_size:]
                time.sleep(0.05)
            elif self.custom_llm_provider == "palm":
                # fake streaming
                response_obj = {}
                if len(self.completion_stream) == 0:
                    if self.received_finish_reason is not None:
                        raise StopIteration
                    else:
                        self.received_finish_reason = "stop"
                chunk_size = 30
                new_chunk = self.completion_stream[:chunk_size]
                completion_obj["content"] = new_chunk
                self.completion_stream = self.completion_stream[chunk_size:]
                time.sleep(0.05)
            elif self.custom_llm_provider == "ollama":
                response_obj = self.handle_ollama_stream(chunk)
                completion_obj["content"] = response_obj["text"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider == "ollama_chat":
                response_obj = self.handle_ollama_chat_stream(chunk)
                completion_obj["content"] = response_obj["text"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider == "cloudflare":
                response_obj = self.handle_cloudlfare_stream(chunk)
                completion_obj["content"] = response_obj["text"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider == "watsonx":
                response_obj = self.handle_watsonx_stream(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider == "text-completion-openai":
                response_obj = self.handle_openai_text_completion_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
                if (
                    self.stream_options
                    and self.stream_options.get("include_usage", False) == True
                    and response_obj["usage"] is not None
                ):
                    self.sent_stream_usage = True
                    model_response.usage = litellm.Usage(
                        prompt_tokens=response_obj["usage"].prompt_tokens,
                        completion_tokens=response_obj["usage"].completion_tokens,
                        total_tokens=response_obj["usage"].total_tokens,
                    )
            elif self.custom_llm_provider == "databricks":
                response_obj = litellm.DatabricksConfig()._chunk_parser(chunk)
                completion_obj["content"] = response_obj["text"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
                if (
                    self.stream_options
                    and self.stream_options.get("include_usage", False) == True
                    and response_obj["usage"] is not None
                ):
                    self.sent_stream_usage = True
                    model_response.usage = litellm.Usage(
                        prompt_tokens=response_obj["usage"].prompt_tokens,
                        completion_tokens=response_obj["usage"].completion_tokens,
                        total_tokens=response_obj["usage"].total_tokens,
                    )
            elif self.custom_llm_provider == "azure_text":
                response_obj = self.handle_azure_text_completion_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            elif self.custom_llm_provider == "cached_response":
                response_obj = {
                    "text": chunk.choices[0].delta.content,
                    "is_finished": True,
                    "finish_reason": chunk.choices[0].finish_reason,
                    "original_chunk": chunk,
                }

                completion_obj["content"] = response_obj["text"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if hasattr(chunk, "id"):
                    model_response.id = chunk.id
                    self.response_id = chunk.id
                if hasattr(chunk, "system_fingerprint"):
                    self.system_fingerprint = chunk.system_fingerprint
                if response_obj["is_finished"]:
                    self.received_finish_reason = response_obj["finish_reason"]
            else:  # openai / azure chat model
                if self.custom_llm_provider == "azure":
                    if hasattr(chunk, "model"):
                        # for azure, we need to pass the model from the orignal chunk
                        self.model = chunk.model
                response_obj = self.handle_openai_chat_completion_chunk(chunk)
                if response_obj == None:
                    return
                completion_obj["content"] = response_obj["text"]
                print_verbose(f"completion obj content: {completion_obj['content']}")
                if response_obj["is_finished"]:
                    if response_obj["finish_reason"] == "error":
                        raise Exception(
                            "Mistral API raised a streaming error - finish_reason: error, no content string given."
                        )
                    self.received_finish_reason = response_obj["finish_reason"]
                if response_obj.get("original_chunk", None) is not None:
                    if hasattr(response_obj["original_chunk"], "id"):
                        model_response.id = response_obj["original_chunk"].id
                        self.response_id = model_response.id
                    if hasattr(response_obj["original_chunk"], "system_fingerprint"):
                        model_response.system_fingerprint = response_obj[
                            "original_chunk"
                        ].system_fingerprint
                        self.system_fingerprint = response_obj[
                            "original_chunk"
                        ].system_fingerprint
                if response_obj["logprobs"] is not None:
                    model_response.choices[0].logprobs = response_obj["logprobs"]

                if (
                    self.stream_options is not None
                    and self.stream_options["include_usage"] == True
                    and response_obj["usage"] is not None
                ):
                    self.sent_stream_usage = True
                    model_response.usage = litellm.Usage(
                        prompt_tokens=response_obj["usage"].prompt_tokens,
                        completion_tokens=response_obj["usage"].completion_tokens,
                        total_tokens=response_obj["usage"].total_tokens,
                    )

            model_response.model = self.model
            print_verbose(
                f"model_response finish reason 3: {self.received_finish_reason}; response_obj={response_obj}"
            )
            ## FUNCTION CALL PARSING
            if (
                response_obj is not None
                and response_obj.get("original_chunk", None) is not None
            ):  # function / tool calling branch - only set for openai/azure compatible endpoints
                # enter this branch when no content has been passed in response
                original_chunk = response_obj.get("original_chunk", None)
                model_response.id = original_chunk.id
                self.response_id = original_chunk.id
                if len(original_chunk.choices) > 0:
                    delta = original_chunk.choices[0].delta
                    if delta is not None and (
                        delta.function_call is not None or delta.tool_calls is not None
                    ):
                        try:
                            model_response.system_fingerprint = (
                                original_chunk.system_fingerprint
                            )
                            ## AZURE - check if arguments is not None
                            if (
                                original_chunk.choices[0].delta.function_call
                                is not None
                            ):
                                if (
                                    getattr(
                                        original_chunk.choices[0].delta.function_call,
                                        "arguments",
                                    )
                                    is None
                                ):
                                    original_chunk.choices[
                                        0
                                    ].delta.function_call.arguments = ""
                            elif original_chunk.choices[0].delta.tool_calls is not None:
                                if isinstance(
                                    original_chunk.choices[0].delta.tool_calls, list
                                ):
                                    for t in original_chunk.choices[0].delta.tool_calls:
                                        if hasattr(t, "functions") and hasattr(
                                            t.functions, "arguments"
                                        ):
                                            if (
                                                getattr(
                                                    t.function,
                                                    "arguments",
                                                )
                                                is None
                                            ):
                                                t.function.arguments = ""
                            _json_delta = delta.model_dump()
                            print_verbose(f"_json_delta: {_json_delta}")
                            if "role" not in _json_delta or _json_delta["role"] is None:
                                _json_delta["role"] = (
                                    "assistant"  # mistral's api returns role as None
                                )
                            if "tool_calls" in _json_delta and isinstance(
                                _json_delta["tool_calls"], list
                            ):
                                for tool in _json_delta["tool_calls"]:
                                    if (
                                        isinstance(tool, dict)
                                        and "function" in tool
                                        and isinstance(tool["function"], dict)
                                        and ("type" not in tool or tool["type"] is None)
                                    ):
                                        # if function returned but type set to None - mistral's api returns type: None
                                        tool["type"] = "function"
                            model_response.choices[0].delta = Delta(**_json_delta)
                        except Exception as e:
                            verbose_logger.error(
                                "litellm.CustomStreamWrapper.chunk_creator(): Exception occured - {}".format(
                                    str(e)
                                )
                            )
                            verbose_logger.debug(traceback.format_exc())
                            model_response.choices[0].delta = Delta()
                    else:
                        try:
                            delta = (
                                dict()
                                if original_chunk.choices[0].delta is None
                                else dict(original_chunk.choices[0].delta)
                            )
                            print_verbose(f"original delta: {delta}")
                            model_response.choices[0].delta = Delta(**delta)
                            print_verbose(
                                f"new delta: {model_response.choices[0].delta}"
                            )
                        except Exception as e:
                            model_response.choices[0].delta = Delta()
                else:
                    if (
                        self.stream_options is not None
                        and self.stream_options["include_usage"] == True
                    ):
                        return model_response
                    return
            print_verbose(
                f"model_response.choices[0].delta: {model_response.choices[0].delta}; completion_obj: {completion_obj}"
            )
            print_verbose(f"self.sent_first_chunk: {self.sent_first_chunk}")

            ## RETURN ARG
            if (
                "content" in completion_obj
                and isinstance(completion_obj["content"], str)
                and len(completion_obj["content"]) == 0
                and hasattr(model_response, "usage")
                and hasattr(model_response.usage, "prompt_tokens")
            ):
                if self.sent_first_chunk is False:
                    completion_obj["role"] = "assistant"
                    self.sent_first_chunk = True
                model_response.choices[0].delta = Delta(**completion_obj)
                print_verbose(f"returning model_response: {model_response}")
                return model_response
            elif (
                "content" in completion_obj
                and isinstance(completion_obj["content"], str)
                and len(completion_obj["content"]) > 0
            ):  # cannot set content of an OpenAI Object to be an empty string
                hold, model_response_str = self.check_special_tokens(
                    chunk=completion_obj["content"],
                    finish_reason=model_response.choices[0].finish_reason,
                )  # filter out bos/eos tokens from openai-compatible hf endpoints
                print_verbose(
                    f"hold - {hold}, model_response_str - {model_response_str}"
                )
                if hold is False:
                    ## check if openai/azure chunk
                    original_chunk = response_obj.get("original_chunk", None)
                    if original_chunk:
                        model_response.id = original_chunk.id
                        self.response_id = original_chunk.id
                        if len(original_chunk.choices) > 0:
                            choices = []
                            for idx, choice in enumerate(original_chunk.choices):
                                try:
                                    if isinstance(choice, BaseModel):
                                        try:
                                            choice_json = choice.model_dump()
                                        except Exception as e:
                                            choice_json = choice.dict()
                                        choice_json.pop(
                                            "finish_reason", None
                                        )  # for mistral etc. which return a value in their last chunk (not-openai compatible).
                                        print_verbose(f"choice_json: {choice_json}")
                                        choices.append(StreamingChoices(**choice_json))
                                except Exception as e:
                                    choices.append(StreamingChoices())
                            print_verbose(f"choices in streaming: {choices}")
                            model_response.choices = choices
                        else:
                            return
                        model_response.system_fingerprint = (
                            original_chunk.system_fingerprint
                        )
                        print_verbose(f"self.sent_first_chunk: {self.sent_first_chunk}")
                        if self.sent_first_chunk == False:
                            model_response.choices[0].delta["role"] = "assistant"
                            self.sent_first_chunk = True
                        elif self.sent_first_chunk == True and hasattr(
                            model_response.choices[0].delta, "role"
                        ):
                            _initial_delta = model_response.choices[
                                0
                            ].delta.model_dump()
                            _initial_delta.pop("role", None)
                            model_response.choices[0].delta = Delta(**_initial_delta)
                        print_verbose(
                            f"model_response.choices[0].delta: {model_response.choices[0].delta}"
                        )
                    else:
                        ## else
                        completion_obj["content"] = model_response_str
                        if self.sent_first_chunk == False:
                            completion_obj["role"] = "assistant"
                            self.sent_first_chunk = True
                        model_response.choices[0].delta = Delta(**completion_obj)
                    print_verbose(f"returning model_response: {model_response}")
                    return model_response
                else:
                    return
            elif self.received_finish_reason is not None:
                if self.sent_last_chunk == True:
                    raise StopIteration
                # flush any remaining holding chunk
                if len(self.holding_chunk) > 0:
                    if model_response.choices[0].delta.content is None:
                        model_response.choices[0].delta.content = self.holding_chunk
                    else:
                        model_response.choices[0].delta.content = (
                            self.holding_chunk + model_response.choices[0].delta.content
                        )
                    self.holding_chunk = ""
                # if delta is None
                _is_delta_empty = self.is_delta_empty(
                    delta=model_response.choices[0].delta
                )

                if _is_delta_empty:
                    # get any function call arguments
                    model_response.choices[0].finish_reason = map_finish_reason(
                        finish_reason=self.received_finish_reason
                    )  # ensure consistent output to openai
                    self.sent_last_chunk = True

                return model_response
            elif (
                model_response.choices[0].delta.tool_calls is not None
                or model_response.choices[0].delta.function_call is not None
            ):
                if self.sent_first_chunk == False:
                    model_response.choices[0].delta["role"] = "assistant"
                    self.sent_first_chunk = True
                return model_response
            else:
                return
        except StopIteration:
            raise StopIteration
        except Exception as e:
            traceback_exception = traceback.format_exc()
            e.message = str(e)
            raise exception_type(
                model=self.model,
                custom_llm_provider=self.custom_llm_provider,
                original_exception=e,
            )

    def set_logging_event_loop(self, loop):
        """
        import litellm, asyncio

        loop = asyncio.get_event_loop() # 👈 gets the current event loop

        response = litellm.completion(.., stream=True)

        response.set_logging_event_loop(loop=loop) # 👈 enables async_success callbacks for sync logging

        for chunk in response:
            ...
        """
        self.logging_loop = loop

    def run_success_logging_in_thread(self, processed_chunk):
        if litellm.disable_streaming_logging == True:
            """
            [NOT RECOMMENDED]
            Set this via `litellm.disable_streaming_logging = True`.

            Disables streaming logging.
            """
            return
        ## ASYNC LOGGING
        # Create an event loop for the new thread
        if self.logging_loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self.logging_obj.async_success_handler(processed_chunk),
                loop=self.logging_loop,
            )
            result = future.result()
        else:
            asyncio.run(self.logging_obj.async_success_handler(processed_chunk))
        ## SYNC LOGGING
        self.logging_obj.success_handler(processed_chunk)

    def finish_reason_handler(self):
        model_response = self.model_response_creator()
        if self.received_finish_reason is not None:
            model_response.choices[0].finish_reason = map_finish_reason(
                finish_reason=self.received_finish_reason
            )
        else:
            model_response.choices[0].finish_reason = "stop"
        return model_response

    def __next__(self):
        try:
            if self.completion_stream is None:
                self.fetch_sync_stream()
            while True:
                if (
                    isinstance(self.completion_stream, str)
                    or isinstance(self.completion_stream, bytes)
                    or isinstance(self.completion_stream, ModelResponse)
                ):
                    chunk = self.completion_stream
                else:
                    chunk = next(self.completion_stream)
                if chunk is not None and chunk != b"":
                    print_verbose(
                        f"PROCESSED CHUNK PRE CHUNK CREATOR: {chunk}; custom_llm_provider: {self.custom_llm_provider}"
                    )
                    response: Optional[ModelResponse] = self.chunk_creator(chunk=chunk)
                    print_verbose(f"PROCESSED CHUNK POST CHUNK CREATOR: {response}")

                    if response is None:
                        continue
                    ## LOGGING
                    threading.Thread(
                        target=self.run_success_logging_in_thread, args=(response,)
                    ).start()  # log response
                    self.response_uptil_now += (
                        response.choices[0].delta.get("content", "") or ""
                    )
                    self.rules.post_call_rules(
                        input=self.response_uptil_now, model=self.model
                    )
                    # RETURN RESULT
                    self.chunks.append(response)
                    return response
        except StopIteration:
            if self.sent_last_chunk == True:
                if (
                    self.sent_stream_usage == False
                    and self.stream_options is not None
                    and self.stream_options.get("include_usage", False) == True
                ):
                    # send the final chunk with stream options
                    complete_streaming_response = litellm.stream_chunk_builder(
                        chunks=self.chunks, messages=self.messages
                    )
                    response = self.model_response_creator()
                    response.usage = complete_streaming_response.usage  # type: ignore
                    ## LOGGING
                    threading.Thread(
                        target=self.logging_obj.success_handler, args=(response,)
                    ).start()  # log response
                    self.sent_stream_usage = True
                    return response
                raise  # Re-raise StopIteration
            else:
                self.sent_last_chunk = True
                processed_chunk = self.finish_reason_handler()
                ## LOGGING
                threading.Thread(
                    target=self.logging_obj.success_handler, args=(processed_chunk,)
                ).start()  # log response
                return processed_chunk
        except Exception as e:
            traceback_exception = traceback.format_exc()
            # LOG FAILURE - handle streaming failure logging in the _next_ object, remove `handle_failure` once it's deprecated
            threading.Thread(
                target=self.logging_obj.failure_handler, args=(e, traceback_exception)
            ).start()
            if isinstance(e, OpenAIError):
                raise e
            else:
                raise exception_type(
                    model=self.model,
                    original_exception=e,
                    custom_llm_provider=self.custom_llm_provider,
                )

    def fetch_sync_stream(self):
        if self.completion_stream is None and self.make_call is not None:
            # Call make_call to get the completion stream
            self.completion_stream = self.make_call(client=litellm.module_level_client)
            self._stream_iter = self.completion_stream.__iter__()

        return self.completion_stream

    async def fetch_stream(self):
        if self.completion_stream is None and self.make_call is not None:
            # Call make_call to get the completion stream
            self.completion_stream = await self.make_call(
                client=litellm.module_level_aclient
            )
            self._stream_iter = self.completion_stream.__aiter__()

        return self.completion_stream

    async def __anext__(self):
        try:
            if self.completion_stream is None:
                await self.fetch_stream()
            if (
                self.custom_llm_provider == "openai"
                or self.custom_llm_provider == "azure"
                or self.custom_llm_provider == "custom_openai"
                or self.custom_llm_provider == "text-completion-openai"
                or self.custom_llm_provider == "azure_text"
                or self.custom_llm_provider == "anthropic"
                or self.custom_llm_provider == "anthropic_text"
                or self.custom_llm_provider == "huggingface"
                or self.custom_llm_provider == "ollama"
                or self.custom_llm_provider == "ollama_chat"
                or self.custom_llm_provider == "vertex_ai"
                or self.custom_llm_provider == "sagemaker"
                or self.custom_llm_provider == "gemini"
                or self.custom_llm_provider == "replicate"
                or self.custom_llm_provider == "cached_response"
                or self.custom_llm_provider == "predibase"
                or self.custom_llm_provider == "databricks"
                or self.custom_llm_provider == "bedrock"
                or self.custom_llm_provider in litellm.openai_compatible_endpoints
            ):
                async for chunk in self.completion_stream:
                    print_verbose(f"value of async chunk: {chunk}")
                    if chunk == "None" or chunk is None:
                        raise Exception
                    elif (
                        self.custom_llm_provider == "gemini"
                        and hasattr(chunk, "parts")
                        and len(chunk.parts) == 0
                    ):
                        continue
                    # chunk_creator() does logging/stream chunk building. We need to let it know its being called in_async_func, so we don't double add chunks.
                    # __anext__ also calls async_success_handler, which does logging
                    print_verbose(f"PROCESSED ASYNC CHUNK PRE CHUNK CREATOR: {chunk}")

                    processed_chunk: Optional[ModelResponse] = self.chunk_creator(
                        chunk=chunk
                    )
                    print_verbose(
                        f"PROCESSED ASYNC CHUNK POST CHUNK CREATOR: {processed_chunk}"
                    )
                    if processed_chunk is None:
                        continue
                    ## LOGGING
                    threading.Thread(
                        target=self.logging_obj.success_handler, args=(processed_chunk,)
                    ).start()  # log response
                    asyncio.create_task(
                        self.logging_obj.async_success_handler(
                            processed_chunk,
                        )
                    )
                    self.response_uptil_now += (
                        processed_chunk.choices[0].delta.get("content", "") or ""
                    )
                    self.rules.post_call_rules(
                        input=self.response_uptil_now, model=self.model
                    )
                    print_verbose(f"final returned processed chunk: {processed_chunk}")
                    self.chunks.append(processed_chunk)
                    return processed_chunk
                raise StopAsyncIteration
            else:  # temporary patch for non-aiohttp async calls
                # example - boto3 bedrock llms
                while True:
                    if isinstance(self.completion_stream, str) or isinstance(
                        self.completion_stream, bytes
                    ):
                        chunk = self.completion_stream
                    else:
                        chunk = next(self.completion_stream)
                    if chunk is not None and chunk != b"":
                        print_verbose(f"PROCESSED CHUNK PRE CHUNK CREATOR: {chunk}")
                        processed_chunk: Optional[ModelResponse] = self.chunk_creator(
                            chunk=chunk
                        )
                        print_verbose(
                            f"PROCESSED CHUNK POST CHUNK CREATOR: {processed_chunk}"
                        )
                        if processed_chunk is None:
                            continue
                        ## LOGGING
                        threading.Thread(
                            target=self.logging_obj.success_handler,
                            args=(processed_chunk,),
                        ).start()  # log processed_chunk
                        asyncio.create_task(
                            self.logging_obj.async_success_handler(
                                processed_chunk,
                            )
                        )

                        self.response_uptil_now += (
                            processed_chunk.choices[0].delta.get("content", "") or ""
                        )
                        self.rules.post_call_rules(
                            input=self.response_uptil_now, model=self.model
                        )
                        # RETURN RESULT
                        self.chunks.append(processed_chunk)
                        return processed_chunk
        except StopAsyncIteration:
            if self.sent_last_chunk == True:
                if (
                    self.sent_stream_usage == False
                    and self.stream_options is not None
                    and self.stream_options.get("include_usage", False) == True
                ):
                    # send the final chunk with stream options
                    complete_streaming_response = litellm.stream_chunk_builder(
                        chunks=self.chunks, messages=self.messages
                    )
                    response = self.model_response_creator()
                    response.usage = complete_streaming_response.usage
                    ## LOGGING
                    threading.Thread(
                        target=self.logging_obj.success_handler, args=(response,)
                    ).start()  # log response
                    asyncio.create_task(
                        self.logging_obj.async_success_handler(
                            response,
                        )
                    )
                    self.sent_stream_usage = True
                    return response
                raise  # Re-raise StopIteration
            else:
                self.sent_last_chunk = True
                processed_chunk = self.finish_reason_handler()
                ## LOGGING
                threading.Thread(
                    target=self.logging_obj.success_handler, args=(processed_chunk,)
                ).start()  # log response
                asyncio.create_task(
                    self.logging_obj.async_success_handler(
                        processed_chunk,
                    )
                )
                return processed_chunk
        except StopIteration:
            if self.sent_last_chunk == True:
                raise StopAsyncIteration
            else:
                self.sent_last_chunk = True
                processed_chunk = self.finish_reason_handler()
                ## LOGGING
                threading.Thread(
                    target=self.logging_obj.success_handler, args=(processed_chunk,)
                ).start()  # log response
                asyncio.create_task(
                    self.logging_obj.async_success_handler(
                        processed_chunk,
                    )
                )
                return processed_chunk
        except httpx.TimeoutException as e:  # if httpx read timeout error occues
            traceback_exception = traceback.format_exc()
            ## ADD DEBUG INFORMATION - E.G. LITELLM REQUEST TIMEOUT
            traceback_exception += "\nLiteLLM Default Request Timeout - {}".format(
                litellm.request_timeout
            )
            if self.logging_obj is not None:
                # Handle any exceptions that might occur during streaming
                asyncio.create_task(
                    self.logging_obj.async_failure_handler(e, traceback_exception)
                )
            raise e
        except Exception as e:
            traceback_exception = traceback.format_exc()
            # Handle any exceptions that might occur during streaming
            asyncio.create_task(
                self.logging_obj.async_failure_handler(e, traceback_exception)  # type: ignore
            )
            raise e


class TextCompletionStreamWrapper:
    def __init__(self, completion_stream, model, stream_options: Optional[dict] = None):
        self.completion_stream = completion_stream
        self.model = model
        self.stream_options = stream_options

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def convert_to_text_completion_object(self, chunk: ModelResponse):
        try:
            response = TextCompletionResponse()
            response["id"] = chunk.get("id", None)
            response["object"] = "text_completion"
            response["created"] = chunk.get("created", None)
            response["model"] = chunk.get("model", None)
            text_choices = TextChoices()
            if isinstance(
                chunk, Choices
            ):  # chunk should always be of type StreamingChoices
                raise Exception
            text_choices["text"] = chunk["choices"][0]["delta"]["content"]
            text_choices["index"] = chunk["choices"][0]["index"]
            text_choices["finish_reason"] = chunk["choices"][0]["finish_reason"]
            response["choices"] = [text_choices]

            # only pass usage when stream_options["include_usage"] is True
            if (
                self.stream_options
                and self.stream_options.get("include_usage", False) == True
            ):
                response["usage"] = chunk.get("usage", None)

            return response
        except Exception as e:
            raise Exception(
                f"Error occurred converting to text completion object - chunk: {chunk}; Error: {str(e)}"
            )

    def __next__(self):
        # model_response = ModelResponse(stream=True, model=self.model)
        response = TextCompletionResponse()
        try:
            for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception
                processed_chunk = self.convert_to_text_completion_object(chunk=chunk)
                return processed_chunk
            raise StopIteration
        except StopIteration:
            raise StopIteration
        except Exception as e:
            print(f"got exception {e}")  # noqa

    async def __anext__(self):
        try:
            async for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception
                processed_chunk = self.convert_to_text_completion_object(chunk=chunk)
                return processed_chunk
            raise StopIteration
        except StopIteration:
            raise StopAsyncIteration


def mock_completion_streaming_obj(model_response, mock_response, model):
    for i in range(0, len(mock_response), 3):
        completion_obj = {"role": "assistant", "content": mock_response[i : i + 3]}
        model_response.choices[0].delta = completion_obj
        yield model_response


async def async_mock_completion_streaming_obj(model_response, mock_response, model):
    for i in range(0, len(mock_response), 3):
        completion_obj = Delta(role="assistant", content=mock_response)
        model_response.choices[0].delta = completion_obj
        model_response.choices[0].finish_reason = "stop"
        yield model_response


########## Reading Config File ############################
def read_config_args(config_path) -> dict:
    try:
        import os

        current_path = os.getcwd()
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        # read keys/ values from config file and return them
        return config
    except Exception as e:
        raise e


########## experimental completion variants ############################


def completion_with_fallbacks(**kwargs):
    nested_kwargs = kwargs.pop("kwargs", {})
    response = None
    rate_limited_models = set()
    model_expiration_times = {}
    start_time = time.time()
    original_model = kwargs["model"]
    fallbacks = [kwargs["model"]] + nested_kwargs.get("fallbacks", [])
    if "fallbacks" in nested_kwargs:
        del nested_kwargs["fallbacks"]  # remove fallbacks so it's not recursive
    litellm_call_id = str(uuid.uuid4())

    # max time to process a request with fallbacks: default 45s
    while response == None and time.time() - start_time < 45:
        for model in fallbacks:
            # loop thru all models
            try:
                # check if it's dict or new model string
                if isinstance(
                    model, dict
                ):  # completion(model="gpt-4", fallbacks=[{"api_key": "", "api_base": ""}, {"api_key": "", "api_base": ""}])
                    kwargs["api_key"] = model.get("api_key", None)
                    kwargs["api_base"] = model.get("api_base", None)
                    model = model.get("model", original_model)
                elif (
                    model in rate_limited_models
                ):  # check if model is currently cooling down
                    if (
                        model_expiration_times.get(model)
                        and time.time() >= model_expiration_times[model]
                    ):
                        rate_limited_models.remove(
                            model
                        )  # check if it's been 60s of cool down and remove model
                    else:
                        continue  # skip model

                # delete model from kwargs if it exists
                if kwargs.get("model"):
                    del kwargs["model"]

                print_verbose(f"trying to make completion call with model: {model}")
                kwargs["litellm_call_id"] = litellm_call_id
                kwargs = {
                    **kwargs,
                    **nested_kwargs,
                }  # combine the openai + litellm params at the same level
                response = litellm.completion(**kwargs, model=model)
                print_verbose(f"response: {response}")
                if response != None:
                    return response

            except Exception as e:
                print_verbose(e)
                rate_limited_models.add(model)
                model_expiration_times[model] = (
                    time.time() + 60
                )  # cool down this selected model
                pass
    return response


def process_system_message(system_message, max_tokens, model):
    system_message_event = {"role": "system", "content": system_message}
    system_message_tokens = get_token_count([system_message_event], model)

    if system_message_tokens > max_tokens:
        print_verbose(
            "`tokentrimmer`: Warning, system message exceeds token limit. Trimming..."
        )
        # shorten system message to fit within max_tokens
        new_system_message = shorten_message_to_fit_limit(
            system_message_event, max_tokens, model
        )
        system_message_tokens = get_token_count([new_system_message], model)

    return system_message_event, max_tokens - system_message_tokens


def process_messages(messages, max_tokens, model):
    # Process messages from older to more recent
    messages = messages[::-1]
    final_messages = []

    for message in messages:
        used_tokens = get_token_count(final_messages, model)
        available_tokens = max_tokens - used_tokens
        if available_tokens <= 3:
            break
        final_messages = attempt_message_addition(
            final_messages=final_messages,
            message=message,
            available_tokens=available_tokens,
            max_tokens=max_tokens,
            model=model,
        )

    return final_messages


def attempt_message_addition(
    final_messages, message, available_tokens, max_tokens, model
):
    temp_messages = [message] + final_messages
    temp_message_tokens = get_token_count(messages=temp_messages, model=model)

    if temp_message_tokens <= max_tokens:
        return temp_messages

    # if temp_message_tokens > max_tokens, try shortening temp_messages
    elif "function_call" not in message:
        # fit updated_message to be within temp_message_tokens - max_tokens (aka the amount temp_message_tokens is greate than max_tokens)
        updated_message = shorten_message_to_fit_limit(message, available_tokens, model)
        if can_add_message(updated_message, final_messages, max_tokens, model):
            return [updated_message] + final_messages

    return final_messages


def can_add_message(message, messages, max_tokens, model):
    if get_token_count(messages + [message], model) <= max_tokens:
        return True
    return False


def get_token_count(messages, model):
    return token_counter(model=model, messages=messages)


def shorten_message_to_fit_limit(message, tokens_needed, model):
    """
    Shorten a message to fit within a token limit by removing characters from the middle.
    """

    # For OpenAI models, even blank messages cost 7 token,
    # and if the buffer is less than 3, the while loop will never end,
    # hence the value 10.
    if "gpt" in model and tokens_needed <= 10:
        return message

    content = message["content"]

    while True:
        total_tokens = get_token_count([message], model)

        if total_tokens <= tokens_needed:
            break

        ratio = (tokens_needed) / total_tokens

        new_length = int(len(content) * ratio) - 1
        new_length = max(0, new_length)

        half_length = new_length // 2
        left_half = content[:half_length]
        right_half = content[-half_length:]

        trimmed_content = left_half + ".." + right_half
        message["content"] = trimmed_content
        content = trimmed_content

    return message


# LiteLLM token trimmer
# this code is borrowed from https://github.com/KillianLucas/tokentrim/blob/main/tokentrim/tokentrim.py
# Credits for this code go to Killian Lucas
def trim_messages(
    messages,
    model: Optional[str] = None,
    trim_ratio: float = 0.75,
    return_response_tokens: bool = False,
    max_tokens=None,
):
    """
    Trim a list of messages to fit within a model's token limit.

    Args:
        messages: Input messages to be trimmed. Each message is a dictionary with 'role' and 'content'.
        model: The LiteLLM model being used (determines the token limit).
        trim_ratio: Target ratio of tokens to use after trimming. Default is 0.75, meaning it will trim messages so they use about 75% of the model's token limit.
        return_response_tokens: If True, also return the number of tokens left available for the response after trimming.
        max_tokens: Instead of specifying a model or trim_ratio, you can specify this directly.

    Returns:
        Trimmed messages and optionally the number of tokens available for response.
    """
    # Initialize max_tokens
    # if users pass in max tokens, trim to this amount
    messages = copy.deepcopy(messages)
    try:
        print_verbose(f"trimming messages")
        if max_tokens is None:
            # Check if model is valid
            if model in litellm.model_cost:
                max_tokens_for_model = litellm.model_cost[model].get(
                    "max_input_tokens", litellm.model_cost[model]["max_tokens"]
                )
                max_tokens = int(max_tokens_for_model * trim_ratio)
            else:
                # if user did not specify max (input) tokens
                # or passed an llm litellm does not know
                # do nothing, just return messages
                return messages

        system_message = ""
        for message in messages:
            if message["role"] == "system":
                system_message += "\n" if system_message else ""
                system_message += message["content"]

        current_tokens = token_counter(model=model, messages=messages)
        print_verbose(f"Current tokens: {current_tokens}, max tokens: {max_tokens}")

        # Do nothing if current tokens under messages
        if current_tokens < max_tokens:
            return messages

        #### Trimming messages if current_tokens > max_tokens
        print_verbose(
            f"Need to trim input messages: {messages}, current_tokens{current_tokens}, max_tokens: {max_tokens}"
        )
        if system_message:
            system_message_event, max_tokens = process_system_message(
                system_message=system_message, max_tokens=max_tokens, model=model
            )

            if max_tokens == 0:  # the system messages are too long
                return [system_message_event]

            # Since all system messages are combined and trimmed to fit the max_tokens,
            # we remove all system messages from the messages list
            messages = [message for message in messages if message["role"] != "system"]

        final_messages = process_messages(
            messages=messages, max_tokens=max_tokens, model=model
        )

        # Add system message to the beginning of the final messages
        if system_message:
            final_messages = [system_message_event] + final_messages

        if (
            return_response_tokens
        ):  # if user wants token count with new trimmed messages
            response_tokens = max_tokens - get_token_count(final_messages, model)
            return final_messages, response_tokens
        return final_messages
    except Exception as e:  # [NON-Blocking, if error occurs just return final_messages
        print_verbose(f"Got exception while token trimming{e}")
        return messages


def get_valid_models() -> List[str]:
    """
    Returns a list of valid LLMs based on the set environment variables

    Args:
        None

    Returns:
        A list of valid LLMs
    """
    try:
        # get keys set in .env
        environ_keys = os.environ.keys()
        valid_providers = []
        # for all valid providers, make a list of supported llms
        valid_models = []

        for provider in litellm.provider_list:
            # edge case litellm has together_ai as a provider, it should be togetherai
            provider = provider.replace("_", "")

            # litellm standardizes expected provider keys to
            # PROVIDER_API_KEY. Example: OPENAI_API_KEY, COHERE_API_KEY
            expected_provider_key = f"{provider.upper()}_API_KEY"
            if expected_provider_key in environ_keys:
                # key is set
                valid_providers.append(provider)

        for provider in valid_providers:
            if provider == "azure":
                valid_models.append("Azure-LLM")
            else:
                models_for_provider = litellm.models_by_provider.get(provider, [])
                valid_models.extend(models_for_provider)
        return valid_models
    except:
        return []  # NON-Blocking


# used for litellm.text_completion() to transform HF logprobs to OpenAI.Completion() format
def transform_logprobs(hf_response):
    # Initialize an empty list for the transformed logprobs
    transformed_logprobs = []

    # For each Hugging Face response, transform the logprobs
    for response in hf_response:
        # Extract the relevant information from the response
        response_details = response["details"]
        top_tokens = response_details.get("top_tokens", {})

        # Initialize an empty list for the token information
        token_info = {
            "tokens": [],
            "token_logprobs": [],
            "text_offset": [],
            "top_logprobs": [],
        }

        for i, token in enumerate(response_details["prefill"]):
            # Extract the text of the token
            token_text = token["text"]

            # Extract the logprob of the token
            token_logprob = token["logprob"]

            # Add the token information to the 'token_info' list
            token_info["tokens"].append(token_text)
            token_info["token_logprobs"].append(token_logprob)

            # stub this to work with llm eval harness
            top_alt_tokens = {"": -1, "": -2, "": -3}
            token_info["top_logprobs"].append(top_alt_tokens)

        # For each element in the 'tokens' list, extract the relevant information
        for i, token in enumerate(response_details["tokens"]):
            # Extract the text of the token
            token_text = token["text"]

            # Extract the logprob of the token
            token_logprob = token["logprob"]

            top_alt_tokens = {}
            temp_top_logprobs = []
            if top_tokens != {}:
                temp_top_logprobs = top_tokens[i]

            # top_alt_tokens should look like this: { "alternative_1": -1, "alternative_2": -2, "alternative_3": -3 }
            for elem in temp_top_logprobs:
                text = elem["text"]
                logprob = elem["logprob"]
                top_alt_tokens[text] = logprob

            # Add the token information to the 'token_info' list
            token_info["tokens"].append(token_text)
            token_info["token_logprobs"].append(token_logprob)
            token_info["top_logprobs"].append(top_alt_tokens)

            # Add the text offset of the token
            # This is computed as the sum of the lengths of all previous tokens
            token_info["text_offset"].append(
                sum(len(t["text"]) for t in response_details["tokens"][:i])
            )

        # Add the 'token_info' list to the 'transformed_logprobs' list
        transformed_logprobs = token_info

    return transformed_logprobs


def print_args_passed_to_litellm(original_function, args, kwargs):
    try:
        # we've already printed this for acompletion, don't print for completion
        if (
            "acompletion" in kwargs
            and kwargs["acompletion"] == True
            and original_function.__name__ == "completion"
        ):
            return
        elif (
            "aembedding" in kwargs
            and kwargs["aembedding"] == True
            and original_function.__name__ == "embedding"
        ):
            return
        elif (
            "aimg_generation" in kwargs
            and kwargs["aimg_generation"] == True
            and original_function.__name__ == "img_generation"
        ):
            return

        args_str = ", ".join(map(repr, args))
        kwargs_str = ", ".join(f"{key}={repr(value)}" for key, value in kwargs.items())
        print_verbose(
            "\n",
        )  # new line before
        print_verbose(
            "\033[92mRequest to litellm:\033[0m",
        )
        if args and kwargs:
            print_verbose(
                f"\033[92mlitellm.{original_function.__name__}({args_str}, {kwargs_str})\033[0m"
            )
        elif args:
            print_verbose(
                f"\033[92mlitellm.{original_function.__name__}({args_str})\033[0m"
            )
        elif kwargs:
            print_verbose(
                f"\033[92mlitellm.{original_function.__name__}({kwargs_str})\033[0m"
            )
        else:
            print_verbose(f"\033[92mlitellm.{original_function.__name__}()\033[0m")
        print_verbose("\n")  # new line after
    except:
        # This should always be non blocking
        pass


def get_logging_id(start_time, response_obj):
    try:
        response_id = (
            "time-" + start_time.strftime("%H-%M-%S-%f") + "_" + response_obj.get("id")
        )
        return response_id
    except:
        return None


def _get_base_model_from_metadata(model_call_details=None):
    if model_call_details is None:
        return None
    litellm_params = model_call_details.get("litellm_params", {})

    if litellm_params is not None:
        metadata = litellm_params.get("metadata", {})

        if metadata is not None:
            model_info = metadata.get("model_info", {})

            if model_info is not None:
                base_model = model_info.get("base_model", None)
                if base_model is not None:
                    return base_model
    return None


def _add_key_name_and_team_to_alert(request_info: str, metadata: dict) -> str:
    """
    Internal helper function for litellm proxy
    Add the Key Name + Team Name to the error
    Only gets added if the metadata contains the user_api_key_alias and user_api_key_team_alias

    [Non-Blocking helper function]
    """
    try:
        _api_key_name = metadata.get("user_api_key_alias", None)
        _user_api_key_team_alias = metadata.get("user_api_key_team_alias", None)
        if _api_key_name is not None:
            request_info = (
                f"\n\nKey Name: `{_api_key_name}`\nTeam: `{_user_api_key_team_alias}`"
                + request_info
            )

        return request_info
    except:
        return request_info


class ModelResponseIterator:
    def __init__(self, model_response: ModelResponse, convert_to_delta: bool = False):
        if convert_to_delta == True:
            self.model_response = ModelResponse(stream=True)
            _delta = self.model_response.choices[0].delta  # type: ignore
            _delta.content = model_response.choices[0].message.content  # type: ignore
        else:
            self.model_response = model_response
        self.is_done = False

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        if self.is_done:
            raise StopIteration
        self.is_done = True
        return self.model_response

    # Async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.is_done:
            raise StopAsyncIteration
        self.is_done = True
        return self.model_response
