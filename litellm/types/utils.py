import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from aiohttp import FormData
from openai._models import BaseModel as OpenAIObject
from openai.types.audio.transcription_create_params import FileTypes  # type: ignore
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from openai.types.moderation import (
    Categories,
    CategoryAppliedInputTypes,
    CategoryScores,
)
from openai.types.moderation_create_response import Moderation, ModerationCreateResponse
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from typing_extensions import Callable, Dict, Required, TypedDict, override

from ..litellm_core_utils.core_helpers import map_finish_reason
from .guardrails import GuardrailEventHooks
from .llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    OpenAIChatCompletionChunk,
)
from .rerank import RerankResponse


def _generate_id():  # private helper function
    return "chatcmpl-" + str(uuid.uuid4())


class LiteLLMPydanticObjectBase(BaseModel):
    """
    Implements default functions, all pydantic objects should have.
    """

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump(**kwargs)  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict(**kwargs)

    def fields_set(self):
        try:
            return self.model_fields_set  # noqa
        except Exception:
            # if using pydantic v1
            return self.__fields_set__

    model_config = ConfigDict(protected_namespaces=())


class LiteLLMCommonStrings(Enum):
    redacted_by_litellm = "redacted by litellm. 'litellm.turn_off_message_logging=True'"


SupportedCacheControls = ["ttl", "s-maxage", "no-cache", "no-store"]


class CostPerToken(TypedDict):
    input_cost_per_token: float
    output_cost_per_token: float


class ProviderField(TypedDict):
    field_name: str
    field_type: Literal["string"]
    field_description: str
    field_value: str


class ProviderSpecificModelInfo(TypedDict, total=False):
    supports_system_messages: Optional[bool]
    supports_response_schema: Optional[bool]
    supports_vision: Optional[bool]
    supports_function_calling: Optional[bool]
    supports_assistant_prefill: Optional[bool]
    supports_prompt_caching: Optional[bool]
    supports_audio_input: Optional[bool]
    supports_embedding_image_input: Optional[bool]
    supports_audio_output: Optional[bool]
    supports_pdf_input: Optional[bool]
    supports_native_streaming: Optional[bool]
    supports_parallel_function_calling: Optional[bool]


class ModelInfoBase(ProviderSpecificModelInfo, total=False):
    key: Required[str]  # the key in litellm.model_cost which is returned

    max_tokens: Required[Optional[int]]
    max_input_tokens: Required[Optional[int]]
    max_output_tokens: Required[Optional[int]]
    input_cost_per_token: Required[float]
    cache_creation_input_token_cost: Optional[float]
    cache_read_input_token_cost: Optional[float]
    input_cost_per_character: Optional[float]  # only for vertex ai models
    input_cost_per_audio_token: Optional[float]
    input_cost_per_token_above_128k_tokens: Optional[float]  # only for vertex ai models
    input_cost_per_character_above_128k_tokens: Optional[
        float
    ]  # only for vertex ai models
    input_cost_per_query: Optional[float]  # only for rerank models
    input_cost_per_image: Optional[float]  # only for vertex ai models
    input_cost_per_audio_per_second: Optional[float]  # only for vertex ai models
    input_cost_per_video_per_second: Optional[float]  # only for vertex ai models
    input_cost_per_second: Optional[float]  # for OpenAI Speech models
    output_cost_per_token: Required[float]
    output_cost_per_character: Optional[float]  # only for vertex ai models
    output_cost_per_audio_token: Optional[float]
    output_cost_per_token_above_128k_tokens: Optional[
        float
    ]  # only for vertex ai models
    output_cost_per_character_above_128k_tokens: Optional[
        float
    ]  # only for vertex ai models
    output_cost_per_image: Optional[float]
    output_vector_size: Optional[int]
    output_cost_per_video_per_second: Optional[float]  # only for vertex ai models
    output_cost_per_audio_per_second: Optional[float]  # only for vertex ai models
    output_cost_per_second: Optional[float]  # for OpenAI Speech models

    litellm_provider: Required[str]
    mode: Required[
        Literal[
            "completion", "embedding", "image_generation", "chat", "audio_transcription"
        ]
    ]
    tpm: Optional[int]
    rpm: Optional[int]


class ModelInfo(ModelInfoBase, total=False):
    """
    Model info for a given model, this is information found in litellm.model_prices_and_context_window.json
    """

    supported_openai_params: Required[Optional[List[str]]]


class GenericStreamingChunk(TypedDict, total=False):
    text: Required[str]
    tool_use: Optional[ChatCompletionToolCallChunk]
    is_finished: Required[bool]
    finish_reason: Required[str]
    usage: Required[Optional[ChatCompletionUsageBlock]]
    index: int

    # use this dict if you want to return any provider specific fields in the response
    provider_specific_fields: Optional[Dict[str, Any]]


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
    rerank = "rerank"
    arerank = "arerank"
    arealtime = "_arealtime"
    create_batch = "create_batch"
    acreate_batch = "acreate_batch"
    pass_through = "pass_through_endpoint"


CallTypesLiteral = Literal[
    "embedding",
    "aembedding",
    "completion",
    "acompletion",
    "atext_completion",
    "text_completion",
    "image_generation",
    "aimage_generation",
    "moderation",
    "amoderation",
    "atranscription",
    "transcription",
    "aspeech",
    "speech",
    "rerank",
    "arerank",
    "_arealtime",
    "create_batch",
    "acreate_batch",
    "pass_through_endpoint",
]


class PassthroughCallTypes(Enum):
    passthrough_image_generation = "passthrough-image-generation"


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

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)


class ChoiceLogprobs(OpenAIObject):
    content: Optional[List[ChatCompletionTokenLogprob]] = None
    """A list of message content tokens with log probability information."""

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)


class FunctionCall(OpenAIObject):
    arguments: str
    name: Optional[str] = None


class Function(OpenAIObject):
    arguments: str
    name: Optional[
        str
    ]  # can be None - openai e.g.: ChoiceDeltaToolCallFunction(arguments='{"', name=None), type=None)

    def __init__(
        self,
        arguments: Optional[Union[Dict, str]],
        name: Optional[str] = None,
        **params,
    ):
        if arguments is None:
            arguments = ""
        elif isinstance(arguments, Dict):
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
    original_response: Optional[Union[str, Any]] = None
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

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
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


from openai.types.chat.chat_completion_audio import ChatCompletionAudio


class ChatCompletionAudioResponse(ChatCompletionAudio):

    def __init__(
        self,
        data: str,
        expires_at: int,
        transcript: str,
        id: Optional[str] = None,
        **params,
    ):
        if id is not None:
            id = id
        else:
            id = f"{uuid.uuid4()}"
        super(ChatCompletionAudioResponse, self).__init__(
            data=data, expires_at=expires_at, transcript=transcript, id=id, **params
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


"""
Reference:
ChatCompletionMessage(content='This is a test', role='assistant', function_call=None, tool_calls=None))
"""


class Message(OpenAIObject):
    content: Optional[str]
    role: Literal["assistant", "user", "system", "tool", "function"]
    tool_calls: Optional[List[ChatCompletionMessageToolCall]]
    function_call: Optional[FunctionCall]
    audio: Optional[ChatCompletionAudioResponse] = None
    provider_specific_fields: Optional[Dict[str, Any]] = Field(
        default=None, exclude=True
    )

    def __init__(
        self,
        content: Optional[str] = None,
        role: Literal["assistant"] = "assistant",
        function_call=None,
        tool_calls: Optional[list] = None,
        audio: Optional[ChatCompletionAudioResponse] = None,
        provider_specific_fields: Optional[Dict[str, Any]] = None,
        **params,
    ):
        init_values: Dict[str, Any] = {
            "content": content,
            "role": role or "assistant",  # handle null input
            "function_call": (
                FunctionCall(**function_call) if function_call is not None else None
            ),
            "tool_calls": (
                [
                    (
                        ChatCompletionMessageToolCall(**tool_call)
                        if isinstance(tool_call, dict)
                        else tool_call
                    )
                    for tool_call in tool_calls
                ]
                if tool_calls is not None and len(tool_calls) > 0
                else None
            ),
        }

        if audio is not None:
            init_values["audio"] = audio

        super(Message, self).__init__(
            **init_values,  # type: ignore
            **params,
        )

        if audio is None:
            # delete audio from self
            # OpenAI compatible APIs like mistral API will raise an error if audio is passed in
            del self.audio

        if provider_specific_fields:  # set if provider_specific_fields is not empty
            self.provider_specific_fields = provider_specific_fields
            for k, v in provider_specific_fields.items():
                setattr(self, k, v)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


class Delta(OpenAIObject):
    provider_specific_fields: Optional[Dict[str, Any]] = Field(
        default=None, exclude=True
    )

    def __init__(
        self,
        content=None,
        role=None,
        function_call=None,
        tool_calls=None,
        audio: Optional[ChatCompletionAudioResponse] = None,
        **params,
    ):
        super(Delta, self).__init__(**params)
        provider_specific_fields: Dict[str, Any] = {}
        if "reasoning_content" in params:
            provider_specific_fields["reasoning_content"] = params["reasoning_content"]
            setattr(self, "reasoning_content", params["reasoning_content"])
        self.content = content
        self.role = role
        # Set default values and correct types
        self.function_call: Optional[Union[FunctionCall, Any]] = None
        self.tool_calls: Optional[List[Union[ChatCompletionDeltaToolCall, Any]]] = None
        self.audio: Optional[ChatCompletionAudioResponse] = None

        if provider_specific_fields:  # set if provider_specific_fields is not empty
            self.provider_specific_fields = provider_specific_fields

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

        self.audio = audio

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
        message: Optional[Union[Message, dict]] = None,
        logprobs=None,
        enhancements=None,
        **params,
    ):
        super(Choices, self).__init__(**params)
        if finish_reason is not None:
            self.finish_reason = map_finish_reason(
                finish_reason
            )  # set finish_reason for all responses
        else:
            self.finish_reason = "stop"
        self.index = index
        if message is None:
            self.message = Message()
        else:
            if isinstance(message, Message):
                self.message = message
            elif isinstance(message, dict):
                self.message = Message(**message)
        if logprobs is not None:
            if isinstance(logprobs, dict):
                self.logprobs = ChoiceLogprobs(**logprobs)
            else:
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


class CompletionTokensDetailsWrapper(
    CompletionTokensDetails
):  # wrapper for older openai versions
    text_tokens: Optional[int] = None
    """Text tokens generated by the model."""


class PromptTokensDetailsWrapper(
    PromptTokensDetails
):  # wrapper for older openai versions
    text_tokens: Optional[int] = None
    """Text tokens sent to the model."""

    image_tokens: Optional[int] = None
    """Image tokens sent to the model."""


class Usage(CompletionUsage):
    _cache_creation_input_tokens: int = PrivateAttr(
        0
    )  # hidden param for prompt caching. Might change, once openai introduces their equivalent.
    _cache_read_input_tokens: int = PrivateAttr(
        0
    )  # hidden param for prompt caching. Might change, once openai introduces their equivalent.

    def __init__(
        self,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        prompt_tokens_details: Optional[Union[PromptTokensDetailsWrapper, dict]] = None,
        completion_tokens_details: Optional[
            Union[CompletionTokensDetailsWrapper, dict]
        ] = None,
        **params,
    ):
        # handle reasoning_tokens
        _completion_tokens_details: Optional[CompletionTokensDetailsWrapper] = None
        if reasoning_tokens:
            completion_tokens_details = CompletionTokensDetailsWrapper(
                reasoning_tokens=reasoning_tokens
            )

        # Ensure completion_tokens_details is properly handled
        if completion_tokens_details:
            if isinstance(completion_tokens_details, dict):
                _completion_tokens_details = CompletionTokensDetailsWrapper(
                    **completion_tokens_details
                )
            elif isinstance(completion_tokens_details, CompletionTokensDetails):
                _completion_tokens_details = completion_tokens_details

        ## DEEPSEEK MAPPING ##
        if "prompt_cache_hit_tokens" in params and isinstance(
            params["prompt_cache_hit_tokens"], int
        ):
            if prompt_tokens_details is None:
                prompt_tokens_details = PromptTokensDetailsWrapper(
                    cached_tokens=params["prompt_cache_hit_tokens"]
                )

        ## ANTHROPIC MAPPING ##
        if "cache_read_input_tokens" in params and isinstance(
            params["cache_read_input_tokens"], int
        ):
            if prompt_tokens_details is None:
                prompt_tokens_details = PromptTokensDetailsWrapper(
                    cached_tokens=params["cache_read_input_tokens"]
                )

        # handle prompt_tokens_details
        _prompt_tokens_details: Optional[PromptTokensDetailsWrapper] = None
        if prompt_tokens_details:
            if isinstance(prompt_tokens_details, dict):
                _prompt_tokens_details = PromptTokensDetailsWrapper(
                    **prompt_tokens_details
                )
            elif isinstance(prompt_tokens_details, PromptTokensDetails):
                _prompt_tokens_details = prompt_tokens_details

        super().__init__(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=total_tokens or 0,
            completion_tokens_details=_completion_tokens_details or None,
            prompt_tokens_details=_prompt_tokens_details or None,
        )

        ## ANTHROPIC MAPPING ##
        if "cache_creation_input_tokens" in params and isinstance(
            params["cache_creation_input_tokens"], int
        ):
            self._cache_creation_input_tokens = params["cache_creation_input_tokens"]

        if "cache_read_input_tokens" in params and isinstance(
            params["cache_read_input_tokens"], int
        ):
            self._cache_read_input_tokens = params["cache_read_input_tokens"]

        ## DEEPSEEK MAPPING ##
        if "prompt_cache_hit_tokens" in params and isinstance(
            params["prompt_cache_hit_tokens"], int
        ):
            self._cache_read_input_tokens = params["prompt_cache_hit_tokens"]

        for k, v in params.items():
            setattr(self, k, v)

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
            self.finish_reason = map_finish_reason(finish_reason)
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


class StreamingChatCompletionChunk(OpenAIChatCompletionChunk):
    def __init__(self, **kwargs):

        new_choices = []
        for choice in kwargs["choices"]:
            new_choice = StreamingChoices(**choice).model_dump()
            new_choices.append(new_choice)
        kwargs["choices"] = new_choices

        super().__init__(**kwargs)


from openai.types.chat import ChatCompletionChunk


class ModelResponseBase(OpenAIObject):
    id: str
    """A unique identifier for the completion."""

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

    _response_headers: Optional[dict] = None


class ModelResponseStream(ModelResponseBase):
    choices: List[StreamingChoices]

    def __init__(
        self,
        choices: Optional[List[Union[StreamingChoices, dict, BaseModel]]] = None,
        id: Optional[str] = None,
        created: Optional[int] = None,
        **kwargs,
    ):
        if choices is not None and isinstance(choices, list):
            new_choices = []
            for choice in choices:
                _new_choice = None
                if isinstance(choice, StreamingChoices):
                    _new_choice = choice
                elif isinstance(choice, dict):
                    _new_choice = StreamingChoices(**choice)
                elif isinstance(choice, BaseModel):
                    _new_choice = StreamingChoices(**choice.model_dump())
                new_choices.append(_new_choice)
            kwargs["choices"] = new_choices
        else:
            kwargs["choices"] = [StreamingChoices()]

        if id is None:
            id = _generate_id()
        else:
            id = id
        if created is None:
            created = int(time.time())
        else:
            created = created

        if (
            "usage" in kwargs
            and kwargs["usage"] is not None
            and isinstance(kwargs["usage"], dict)
        ):
            kwargs["usage"] = Usage(**kwargs["usage"])

        kwargs["id"] = id
        kwargs["created"] = created
        kwargs["object"] = "chat.completion.chunk"

        super().__init__(**kwargs)

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


class ModelResponse(ModelResponseBase):
    choices: List[Union[Choices, StreamingChoices]]
    """The list of completion choices the model generated for the input prompt."""

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
        _response_headers=None,
        **params,
    ) -> None:
        if stream is not None and stream is True:
            object = "chat.completion.chunk"
            if choices is not None and isinstance(choices, list):
                new_choices = []
                for choice in choices:
                    _new_choice = None
                    if isinstance(choice, StreamingChoices):
                        _new_choice = choice
                    elif isinstance(choice, dict):
                        _new_choice = StreamingChoices(**choice)
                    elif isinstance(choice, BaseModel):
                        _new_choice = StreamingChoices(**choice.model_dump())
                    new_choices.append(_new_choice)
                choices = new_choices
            else:
                choices = [StreamingChoices()]
        else:
            object = "chat.completion"
            if choices is not None and isinstance(choices, list):
                new_choices = []
                for choice in choices:
                    if isinstance(choice, Choices):
                        _new_choice = choice  # type: ignore
                    elif isinstance(choice, dict):
                        _new_choice = Choices(**choice)  # type: ignore
                    else:
                        _new_choice = choice
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
        elif stream is None or stream is False:
            usage = Usage()
        if hidden_params:
            self._hidden_params = hidden_params

        if _response_headers:
            self._response_headers = _response_headers

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

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


class Embedding(OpenAIObject):
    embedding: Union[list, str] = []
    index: int
    object: Literal["embedding"]

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

    data: List
    """The actual embedding value"""

    object: Literal["list"]
    """The object type, which is always "list" """

    usage: Optional[Usage] = None
    """Usage statistics for the embedding request."""

    _hidden_params: dict = {}
    _response_headers: Optional[Dict] = None
    _response_ms: Optional[float] = None

    def __init__(
        self,
        model: Optional[str] = None,
        usage: Optional[Usage] = None,
        response_ms=None,
        data: Optional[Union[List, List[Embedding]]] = None,
        hidden_params=None,
        _response_headers=None,
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
            data = []

        if usage:
            usage = usage
        else:
            usage = Usage()

        if _response_headers:
            self._response_headers = _response_headers

        model = model
        super().__init__(model=model, object=object, data=data, usage=usage)  # type: ignore

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

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


class Logprobs(OpenAIObject):
    text_offset: Optional[List[int]]
    token_logprobs: Optional[List[Union[float, None]]]
    tokens: Optional[List[str]]
    top_logprobs: Optional[List[Union[Dict[str, float], None]]]


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

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
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
                    _new_choice = None
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
            id=id,  # type: ignore
            object=object,  # type: ignore
            created=created,  # type: ignore
            model=model,  # type: ignore
            choices=choices,  # type: ignore
            usage=usage,  # type: ignore
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


from openai.types.images_response import Image as OpenAIImage


class ImageObject(OpenAIImage):
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

    def __init__(self, b64_json=None, url=None, revised_prompt=None, **kwargs):
        super().__init__(b64_json=b64_json, url=url, revised_prompt=revised_prompt)  # type: ignore

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

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


from openai.types.images_response import ImagesResponse as OpenAIImageResponse


class ImageResponse(OpenAIImageResponse):
    _hidden_params: dict = {}
    usage: Usage

    def __init__(
        self,
        created: Optional[int] = None,
        data: Optional[List[ImageObject]] = None,
        response_ms=None,
        usage: Optional[Usage] = None,
        hidden_params: Optional[dict] = None,
    ):
        if response_ms:
            _response_ms = response_ms
        else:
            _response_ms = None
        if data:
            data = data
        else:
            data = []

        if created:
            created = created
        else:
            created = int(time.time())

        _data: List[OpenAIImage] = []
        for d in data:
            if isinstance(d, dict):
                _data.append(ImageObject(**d))
            elif isinstance(d, BaseModel):
                _data.append(ImageObject(**d.model_dump()))
        _usage = usage or Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        super().__init__(created=created, data=_data, usage=_usage)  # type: ignore
        self._hidden_params = hidden_params or {}

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

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


class TranscriptionResponse(OpenAIObject):
    text: Optional[str] = None

    _hidden_params: dict = {}
    _response_headers: Optional[dict] = None

    def __init__(self, text=None):
        super().__init__(text=text)  # type: ignore

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

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


class GenericImageParsingChunk(TypedDict):
    type: str
    media_type: str
    data: str


class ResponseFormatChunk(TypedDict, total=False):
    type: Required[Literal["json_object", "text"]]
    response_schema: dict


class LoggedLiteLLMParams(TypedDict, total=False):
    force_timeout: Optional[float]
    custom_llm_provider: Optional[str]
    api_base: Optional[str]
    litellm_call_id: Optional[str]
    model_alias_map: Optional[dict]
    metadata: Optional[dict]
    model_info: Optional[dict]
    proxy_server_request: Optional[dict]
    acompletion: Optional[bool]
    preset_cache_key: Optional[str]
    no_log: Optional[bool]
    input_cost_per_second: Optional[float]
    input_cost_per_token: Optional[float]
    output_cost_per_token: Optional[float]
    output_cost_per_second: Optional[float]
    cooldown_time: Optional[float]


class AdapterCompletionStreamWrapper:
    def __init__(self, completion_stream):
        self.completion_stream = completion_stream

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def __next__(self):
        try:
            for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception
                return chunk
            raise StopIteration
        except StopIteration:
            raise StopIteration
        except Exception as e:
            print(f"AdapterCompletionStreamWrapper - {e}")  # noqa

    async def __anext__(self):
        try:
            async for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception
                return chunk
            raise StopIteration
        except StopIteration:
            raise StopAsyncIteration


class StandardLoggingUserAPIKeyMetadata(TypedDict):
    user_api_key_hash: Optional[str]  # hash of the litellm virtual key used
    user_api_key_alias: Optional[str]
    user_api_key_org_id: Optional[str]
    user_api_key_team_id: Optional[str]
    user_api_key_user_id: Optional[str]
    user_api_key_team_alias: Optional[str]
    user_api_key_end_user_id: Optional[str]


class StandardLoggingPromptManagementMetadata(TypedDict):
    prompt_id: str
    prompt_variables: Optional[dict]
    prompt_integration: str


class StandardLoggingMetadata(StandardLoggingUserAPIKeyMetadata):
    """
    Specific metadata k,v pairs logged to integration for easier cost tracking and prompt management
    """

    spend_logs_metadata: Optional[
        dict
    ]  # special param to log k,v pairs to spendlogs for a call
    requester_ip_address: Optional[str]
    requester_metadata: Optional[dict]
    prompt_management_metadata: Optional[StandardLoggingPromptManagementMetadata]


class StandardLoggingAdditionalHeaders(TypedDict, total=False):
    x_ratelimit_limit_requests: int
    x_ratelimit_limit_tokens: int
    x_ratelimit_remaining_requests: int
    x_ratelimit_remaining_tokens: int


class StandardLoggingHiddenParams(TypedDict):
    model_id: Optional[str]
    cache_key: Optional[str]
    api_base: Optional[str]
    response_cost: Optional[str]
    litellm_overhead_time_ms: Optional[float]
    additional_headers: Optional[StandardLoggingAdditionalHeaders]


class StandardLoggingModelInformation(TypedDict):
    model_map_key: str
    model_map_value: Optional[ModelInfo]


class StandardLoggingModelCostFailureDebugInformation(TypedDict, total=False):
    """
    Debug information, if cost tracking fails.

    Avoid logging sensitive information like response or optional params
    """

    error_str: Required[str]
    traceback_str: Required[str]
    model: str
    cache_hit: Optional[bool]
    custom_llm_provider: Optional[str]
    base_model: Optional[str]
    call_type: str
    custom_pricing: Optional[bool]


class StandardLoggingPayloadErrorInformation(TypedDict, total=False):
    error_code: Optional[str]
    error_class: Optional[str]
    llm_provider: Optional[str]


class StandardLoggingGuardrailInformation(TypedDict, total=False):
    guardrail_name: Optional[str]
    guardrail_mode: Optional[Union[GuardrailEventHooks, List[GuardrailEventHooks]]]
    guardrail_response: Optional[Union[dict, str]]
    guardrail_status: Literal["success", "failure"]


StandardLoggingPayloadStatus = Literal["success", "failure"]


class StandardLoggingPayload(TypedDict):
    id: str
    trace_id: str  # Trace multiple LLM calls belonging to same overall request (e.g. fallbacks/retries)
    call_type: str
    stream: Optional[bool]
    response_cost: float
    response_cost_failure_debug_info: Optional[
        StandardLoggingModelCostFailureDebugInformation
    ]
    status: StandardLoggingPayloadStatus
    custom_llm_provider: Optional[str]
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    startTime: float  # Note: making this camelCase was a mistake, everything should be snake case
    endTime: float
    completionStartTime: float
    response_time: float
    model_map_information: StandardLoggingModelInformation
    model: str
    model_id: Optional[str]
    model_group: Optional[str]
    api_base: str
    metadata: StandardLoggingMetadata
    cache_hit: Optional[bool]
    cache_key: Optional[str]
    saved_cache_cost: float
    request_tags: list
    end_user: Optional[str]
    requester_ip_address: Optional[str]
    messages: Optional[Union[str, list, dict]]
    response: Optional[Union[str, list, dict]]
    error_str: Optional[str]
    error_information: Optional[StandardLoggingPayloadErrorInformation]
    model_parameters: dict
    hidden_params: StandardLoggingHiddenParams
    guardrail_information: Optional[StandardLoggingGuardrailInformation]


from typing import AsyncIterator, Iterator


class CustomStreamingDecoder:
    async def aiter_bytes(
        self, iterator: AsyncIterator[bytes]
    ) -> AsyncIterator[
        Optional[Union[GenericStreamingChunk, StreamingChatCompletionChunk]]
    ]:
        raise NotImplementedError

    def iter_bytes(
        self, iterator: Iterator[bytes]
    ) -> Iterator[Optional[Union[GenericStreamingChunk, StreamingChatCompletionChunk]]]:
        raise NotImplementedError


class StandardPassThroughResponseObject(TypedDict):
    response: str


OPENAI_RESPONSE_HEADERS = [
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
]


class StandardCallbackDynamicParams(TypedDict, total=False):
    # Langfuse dynamic params
    langfuse_public_key: Optional[str]
    langfuse_secret: Optional[str]
    langfuse_secret_key: Optional[str]
    langfuse_host: Optional[str]

    # GCS dynamic params
    gcs_bucket_name: Optional[str]
    gcs_path_service_account: Optional[str]

    # Langsmith dynamic params
    langsmith_api_key: Optional[str]
    langsmith_project: Optional[str]
    langsmith_base_url: Optional[str]

    # Humanloop dynamic params
    humanloop_api_key: Optional[str]

    # Logging settings
    turn_off_message_logging: Optional[bool]  # when true will not log messages


all_litellm_params = [
    "metadata",
    "litellm_metadata",
    "litellm_trace_id",
    "tags",
    "acompletion",
    "aimg_generation",
    "atext_completion",
    "text_completion",
    "caching",
    "mock_response",
    "mock_timeout",
    "disable_add_transform_inline_image_block",
    "api_key",
    "api_version",
    "prompt_id",
    "provider_specific_header",
    "prompt_variables",
    "api_base",
    "force_timeout",
    "logger_fn",
    "verbose",
    "custom_llm_provider",
    "litellm_logging_obj",
    "litellm_call_id",
    "use_client",
    "id",
    "fallbacks",
    "azure",
    "headers",
    "model_list",
    "num_retries",
    "context_window_fallback_dict",
    "retry_policy",
    "retry_strategy",
    "roles",
    "final_prompt_value",
    "bos_token",
    "eos_token",
    "request_timeout",
    "complete_response",
    "self",
    "client",
    "rpm",
    "tpm",
    "max_parallel_requests",
    "input_cost_per_token",
    "output_cost_per_token",
    "input_cost_per_second",
    "output_cost_per_second",
    "hf_model_name",
    "model_info",
    "proxy_server_request",
    "preset_cache_key",
    "caching_groups",
    "ttl",
    "cache",
    "no-log",
    "base_model",
    "stream_timeout",
    "supports_system_message",
    "region_name",
    "allowed_model_region",
    "model_config",
    "fastest_response",
    "cooldown_time",
    "cache_key",
    "max_retries",
    "azure_ad_token_provider",
    "tenant_id",
    "client_id",
    "azure_username",
    "azure_password",
    "client_secret",
    "user_continue_message",
    "configurable_clientside_auth_params",
    "weight",
    "ensure_alternating_roles",
    "assistant_continue_message",
    "user_continue_message",
    "fallback_depth",
    "max_fallbacks",
    "max_budget",
    "budget_duration",
    "use_in_pass_through",
] + list(StandardCallbackDynamicParams.__annotations__.keys())


class KeyGenerationConfig(TypedDict, total=False):
    required_params: List[
        str
    ]  # specify params that must be present in the key generation request


class TeamUIKeyGenerationConfig(KeyGenerationConfig):
    allowed_team_member_roles: List[str]


class PersonalUIKeyGenerationConfig(KeyGenerationConfig):
    allowed_user_roles: List[str]


class StandardKeyGenerationConfig(TypedDict, total=False):
    team_key_generation: TeamUIKeyGenerationConfig
    personal_key_generation: PersonalUIKeyGenerationConfig


class BudgetConfig(BaseModel):
    max_budget: Optional[float] = None
    budget_duration: Optional[str] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None

    def __init__(self, **data: Any) -> None:
        # Map time_period to budget_duration if present
        if "time_period" in data:
            data["budget_duration"] = data.pop("time_period")

        # Map budget_limit to max_budget if present
        if "budget_limit" in data:
            data["max_budget"] = data.pop("budget_limit")

        super().__init__(**data)


GenericBudgetConfigType = Dict[str, BudgetConfig]


class LlmProviders(str, Enum):
    OPENAI = "openai"
    OPENAI_LIKE = "openai_like"  # embedding only
    JINA_AI = "jina_ai"
    XAI = "xai"
    CUSTOM_OPENAI = "custom_openai"
    TEXT_COMPLETION_OPENAI = "text-completion-openai"
    COHERE = "cohere"
    COHERE_CHAT = "cohere_chat"
    CLARIFAI = "clarifai"
    ANTHROPIC = "anthropic"
    ANTHROPIC_TEXT = "anthropic_text"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"
    TOGETHER_AI = "together_ai"
    OPENROUTER = "openrouter"
    VERTEX_AI = "vertex_ai"
    VERTEX_AI_BETA = "vertex_ai_beta"
    GEMINI = "gemini"
    AI21 = "ai21"
    BASETEN = "baseten"
    AZURE = "azure"
    AZURE_TEXT = "azure_text"
    AZURE_AI = "azure_ai"
    SAGEMAKER = "sagemaker"
    SAGEMAKER_CHAT = "sagemaker_chat"
    BEDROCK = "bedrock"
    VLLM = "vllm"
    NLP_CLOUD = "nlp_cloud"
    PETALS = "petals"
    OOBABOOGA = "oobabooga"
    OLLAMA = "ollama"
    OLLAMA_CHAT = "ollama_chat"
    DEEPINFRA = "deepinfra"
    PERPLEXITY = "perplexity"
    MISTRAL = "mistral"
    GROQ = "groq"
    NVIDIA_NIM = "nvidia_nim"
    CEREBRAS = "cerebras"
    AI21_CHAT = "ai21_chat"
    VOLCENGINE = "volcengine"
    CODESTRAL = "codestral"
    TEXT_COMPLETION_CODESTRAL = "text-completion-codestral"
    DEEPSEEK = "deepseek"
    SAMBANOVA = "sambanova"
    MARITALK = "maritalk"
    VOYAGE = "voyage"
    CLOUDFLARE = "cloudflare"
    XINFERENCE = "xinference"
    FIREWORKS_AI = "fireworks_ai"
    FRIENDLIAI = "friendliai"
    WATSONX = "watsonx"
    WATSONX_TEXT = "watsonx_text"
    TRITON = "triton"
    PREDIBASE = "predibase"
    DATABRICKS = "databricks"
    EMPOWER = "empower"
    GITHUB = "github"
    CUSTOM = "custom"
    LITELLM_PROXY = "litellm_proxy"
    HOSTED_VLLM = "hosted_vllm"
    LM_STUDIO = "lm_studio"
    GALADRIEL = "galadriel"
    INFINITY = "infinity"
    DEEPGRAM = "deepgram"
    AIOHTTP_OPENAI = "aiohttp_openai"
    LANGFUSE = "langfuse"
    HUMANLOOP = "humanloop"
    TOPAZ = "topaz"


# Create a set of all provider values for quick lookup
LlmProvidersSet = {provider.value for provider in LlmProviders}


class LiteLLMLoggingBaseClass:
    """
    Base class for logging pre and post call

    Meant to simplify type checking for logging obj.
    """

    def pre_call(self, input, api_key, model=None, additional_args={}):
        pass

    def post_call(
        self, original_response, input=None, api_key=None, additional_args={}
    ):
        pass


class CustomHuggingfaceTokenizer(TypedDict):
    identifier: str
    revision: str  # usually 'main'
    auth_token: Optional[str]


class LITELLM_IMAGE_VARIATION_PROVIDERS(Enum):
    """
    Try using an enum for endpoints. This should make it easier to track what provider is supported for what endpoint.
    """

    OPENAI = LlmProviders.OPENAI.value
    TOPAZ = LlmProviders.TOPAZ.value


class HttpHandlerRequestFields(TypedDict, total=False):
    data: dict  # request body
    params: dict  # query params
    files: dict  # file uploads
    content: Any  # raw content


class ProviderSpecificHeader(TypedDict):
    custom_llm_provider: str
    extra_headers: dict


class SelectTokenizerResponse(TypedDict):
    type: Literal["openai_tokenizer", "huggingface_tokenizer"]
    tokenizer: Any
