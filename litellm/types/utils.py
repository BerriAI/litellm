import json
import time
import uuid
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Union

from openai._models import BaseModel as OpenAIObject
from pydantic import ConfigDict
from typing_extensions import Dict, Required, TypedDict, override

from ..litellm_core_utils.core_helpers import map_finish_reason
from .llms.openai import ChatCompletionToolCallChunk, ChatCompletionUsageBlock


def _generate_id():  # private helper function
    return "chatcmpl-" + str(uuid.uuid4())


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


class ModelInfo(TypedDict, total=False):
    """
    Model info for a given model, this is information found in litellm.model_prices_and_context_window.json
    """

    max_tokens: Required[Optional[int]]
    max_input_tokens: Required[Optional[int]]
    max_output_tokens: Required[Optional[int]]
    input_cost_per_token: Required[float]
    input_cost_per_character: Optional[float]  # only for vertex ai models
    input_cost_per_token_above_128k_tokens: Optional[float]  # only for vertex ai models
    input_cost_per_character_above_128k_tokens: Optional[
        float
    ]  # only for vertex ai models
    input_cost_per_image: Optional[float]  # only for vertex ai models
    input_cost_per_audio_per_second: Optional[float]  # only for vertex ai models
    input_cost_per_video_per_second: Optional[float]  # only for vertex ai models
    output_cost_per_token: Required[float]
    output_cost_per_character: Optional[float]  # only for vertex ai models
    output_cost_per_token_above_128k_tokens: Optional[
        float
    ]  # only for vertex ai models
    output_cost_per_character_above_128k_tokens: Optional[
        float
    ]  # only for vertex ai models
    output_cost_per_image: Optional[float]
    output_cost_per_video_per_second: Optional[float]  # only for vertex ai models
    output_cost_per_audio_per_second: Optional[float]  # only for vertex ai models
    litellm_provider: Required[str]
    mode: Required[
        Literal[
            "completion", "embedding", "image_generation", "chat", "audio_transcription"
        ]
    ]
    supported_openai_params: Required[Optional[List[str]]]
    supports_system_messages: Optional[bool]
    supports_response_schema: Optional[bool]


class GenericStreamingChunk(TypedDict):
    text: Required[str]
    tool_use: Optional[ChatCompletionToolCallChunk]
    is_finished: Required[bool]
    finish_reason: Required[str]
    usage: Optional[ChatCompletionUsageBlock]
    index: int


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
    ) -> None:
        if stream is not None and stream is True:
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
            object = "chat.completion"
            if choices is not None and isinstance(choices, list):
                new_choices = []
                for choice in choices:
                    if isinstance(choice, Choices):
                        _new_choice = choice  # type: ignore
                    elif isinstance(choice, dict):
                        _new_choice = Choices(**choice)  # type: ignore
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


class GenericImageParsingChunk(TypedDict):
    # {
    #         "type": "base64",
    #         "media_type": f"image/{image_format}",
    #         "data": base64_data,
    #     }
    type: str
    media_type: str
    data: str


class ResponseFormatChunk(TypedDict, total=False):
    type: Required[Literal["json_object", "text"]]
    response_schema: dict
