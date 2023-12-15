# coding=utf-8
# Copyright 2023-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original implementation taken from the `text-generation` Python client (see https://pypi.org/project/text-generation/
# and https://github.com/huggingface/text-generation-inference/tree/main/clients/python)
#
# Changes compared to original implementation:
# - use pydantic.dataclasses instead of BaseModel
# - default to Python's dataclasses if Pydantic is not installed (same implementation but no validation)
# - added default values for all parameters (not needed in BaseModel but dataclasses yes)
# - integrated in `huggingface_hub.InferenceClient``
# - added `stream: bool` and `details: bool` in the `text_generation` method instead of having different methods for each use case
# - NO asyncio support yet => TODO soon

from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional

from requests import HTTPError

from ..utils import is_pydantic_available


if is_pydantic_available():
    from pydantic import validator
    from pydantic.dataclasses import dataclass
else:
    # No validation if Pydantic is not installed
    from dataclasses import dataclass  # type: ignore

    def validator(x):  # type: ignore
        return lambda y: y


@dataclass
class TextGenerationParameters:
    """
    Parameters for text generation.

    Args:
        do_sample (`bool`, *optional*):
            Activate logits sampling. Defaults to False.
        max_new_tokens (`int`, *optional*):
            Maximum number of generated tokens. Defaults to 20.
        repetition_penalty (`Optional[float]`, *optional*):
            The parameter for repetition penalty. A value of 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf)
            for more details. Defaults to None.
        return_full_text (`bool`, *optional*):
            Whether to prepend the prompt to the generated text. Defaults to False.
        stop (`List[str]`, *optional*):
            Stop generating tokens if a member of `stop_sequences` is generated. Defaults to an empty list.
        seed (`Optional[int]`, *optional*):
            Random sampling seed. Defaults to None.
        temperature (`Optional[float]`, *optional*):
            The value used to modulate the logits distribution. Defaults to None.
        top_k (`Optional[int]`, *optional*):
            The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
        top_p (`Optional[float]`, *optional*):
            If set to a value less than 1, only the smallest set of most probable tokens with probabilities that add up
            to `top_p` or higher are kept for generation. Defaults to None.
        truncate (`Optional[int]`, *optional*):
            Truncate input tokens to the given size. Defaults to None.
        typical_p (`Optional[float]`, *optional*):
            Typical Decoding mass. See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666)
            for more information. Defaults to None.
        best_of (`Optional[int]`, *optional*):
            Generate `best_of` sequences and return the one with the highest token logprobs. Defaults to None.
        watermark (`bool`, *optional*):
            Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226). Defaults to False.
        details (`bool`, *optional*):
            Get generation details. Defaults to False.
        decoder_input_details (`bool`, *optional*):
            Get decoder input token logprobs and ids. Defaults to False.
    """

    # Activate logits sampling
    do_sample: bool = False
    # Maximum number of generated tokens
    max_new_tokens: int = 20
    # The parameter for repetition penalty. 1.0 means no penalty.
    # See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    repetition_penalty: Optional[float] = None
    # Whether to prepend the prompt to the generated text
    return_full_text: bool = False
    # Stop generating tokens if a member of `stop_sequences` is generated
    stop: List[str] = field(default_factory=lambda: [])
    # Random sampling seed
    seed: Optional[int] = None
    # The value used to module the logits distribution.
    temperature: Optional[float] = None
    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: Optional[int] = None
    # If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
    # higher are kept for generation.
    top_p: Optional[float] = None
    # truncate inputs tokens to the given size
    truncate: Optional[int] = None
    # Typical Decoding mass
    # See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
    typical_p: Optional[float] = None
    # Generate best_of sequences and return the one if the highest token logprobs
    best_of: Optional[int] = None
    # Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
    watermark: bool = False
    # Get generation details
    details: bool = False
    # Get decoder input token logprobs and ids
    decoder_input_details: bool = False

    @validator("best_of")
    def valid_best_of(cls, field_value, values):
        if field_value is not None:
            if field_value <= 0:
                raise ValueError("`best_of` must be strictly positive")
            if field_value > 1 and values["seed"] is not None:
                raise ValueError("`seed` must not be set when `best_of` is > 1")
            sampling = (
                values["do_sample"]
                | (values["temperature"] is not None)
                | (values["top_k"] is not None)
                | (values["top_p"] is not None)
                | (values["typical_p"] is not None)
            )
            if field_value > 1 and not sampling:
                raise ValueError("you must use sampling when `best_of` is > 1")

        return field_value

    @validator("repetition_penalty")
    def valid_repetition_penalty(cls, v):
        if v is not None and v <= 0:
            raise ValueError("`repetition_penalty` must be strictly positive")
        return v

    @validator("seed")
    def valid_seed(cls, v):
        if v is not None and v < 0:
            raise ValueError("`seed` must be positive")
        return v

    @validator("temperature")
    def valid_temp(cls, v):
        if v is not None and v <= 0:
            raise ValueError("`temperature` must be strictly positive")
        return v

    @validator("top_k")
    def valid_top_k(cls, v):
        if v is not None and v <= 0:
            raise ValueError("`top_k` must be strictly positive")
        return v

    @validator("top_p")
    def valid_top_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            raise ValueError("`top_p` must be > 0.0 and < 1.0")
        return v

    @validator("truncate")
    def valid_truncate(cls, v):
        if v is not None and v <= 0:
            raise ValueError("`truncate` must be strictly positive")
        return v

    @validator("typical_p")
    def valid_typical_p(cls, v):
        if v is not None and (v <= 0 or v >= 1.0):
            raise ValueError("`typical_p` must be > 0.0 and < 1.0")
        return v


@dataclass
class TextGenerationRequest:
    """
    Request object for text generation (only for internal use).

    Args:
        inputs (`str`):
            The prompt for text generation.
        parameters (`Optional[TextGenerationParameters]`, *optional*):
            Generation parameters.
        stream (`bool`, *optional*):
            Whether to stream output tokens. Defaults to False.
    """

    # Prompt
    inputs: str
    # Generation parameters
    parameters: Optional[TextGenerationParameters] = None
    # Whether to stream output tokens
    stream: bool = False

    @validator("inputs")
    def valid_input(cls, v):
        if not v:
            raise ValueError("`inputs` cannot be empty")
        return v

    @validator("stream")
    def valid_best_of_stream(cls, field_value, values):
        parameters = values["parameters"]
        if parameters is not None and parameters.best_of is not None and parameters.best_of > 1 and field_value:
            raise ValueError("`best_of` != 1 is not supported when `stream` == True")
        return field_value


# Decoder input tokens
@dataclass
class InputToken:
    """
    Represents an input token.

    Args:
        id (`int`):
            Token ID from the model tokenizer.
        text (`str`):
            Token text.
        logprob (`float` or `None`):
            Log probability of the token. Optional since the logprob of the first token cannot be computed.
    """

    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    # Optional since the logprob of the first token cannot be computed
    logprob: Optional[float] = None


# Generated tokens
@dataclass
class Token:
    """
    Represents a token.

    Args:
        id (`int`):
            Token ID from the model tokenizer.
        text (`str`):
            Token text.
        logprob (`float`):
            Log probability of the token.
        special (`bool`):
            Indicates whether the token is a special token. It can be used to ignore
            tokens when concatenating.
    """

    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    logprob: float
    # Is the token a special token
    # Can be used to ignore tokens when concatenating
    special: bool


# Generation finish reason
class FinishReason(str, Enum):
    # number of generated tokens == `max_new_tokens`
    Length = "length"
    # the model generated its end of sequence token
    EndOfSequenceToken = "eos_token"
    # the model generated a text included in `stop_sequences`
    StopSequence = "stop_sequence"


# Additional sequences when using the `best_of` parameter
@dataclass
class BestOfSequence:
    """
    Represents a best-of sequence generated during text generation.

    Args:
        generated_text (`str`):
            The generated text.
        finish_reason (`FinishReason`):
            The reason for the generation to finish, represented by a `FinishReason` value.
        generated_tokens (`int`):
            The number of generated tokens in the sequence.
        seed (`Optional[int]`):
            The sampling seed if sampling was activated.
        prefill (`List[InputToken]`):
            The decoder input tokens. Empty if `decoder_input_details` is False. Defaults to an empty list.
        tokens (`List[Token]`):
            The generated tokens. Defaults to an empty list.
    """

    # Generated text
    generated_text: str
    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int] = None
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[InputToken] = field(default_factory=lambda: [])
    # Generated tokens
    tokens: List[Token] = field(default_factory=lambda: [])


# `generate` details
@dataclass
class Details:
    """
    Represents details of a text generation.

    Args:
        finish_reason (`FinishReason`):
            The reason for the generation to finish, represented by a `FinishReason` value.
        generated_tokens (`int`):
            The number of generated tokens.
        seed (`Optional[int]`):
            The sampling seed if sampling was activated.
        prefill (`List[InputToken]`, *optional*):
            The decoder input tokens. Empty if `decoder_input_details` is False. Defaults to an empty list.
        tokens (`List[Token]`):
            The generated tokens. Defaults to an empty list.
        best_of_sequences (`Optional[List[BestOfSequence]]`):
            Additional sequences when using the `best_of` parameter.
    """

    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int] = None
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[InputToken] = field(default_factory=lambda: [])
    # Generated tokens
    tokens: List[Token] = field(default_factory=lambda: [])
    # Additional sequences when using the `best_of` parameter
    best_of_sequences: Optional[List[BestOfSequence]] = None


# `generate` return value
@dataclass
class TextGenerationResponse:
    """
    Represents a response for text generation.

    In practice, if `details=False` is passed (default), only the generated text is returned.

    Args:
        generated_text (`str`):
            The generated text.
        details (`Optional[Details]`):
            Generation details. Returned only if `details=True` is sent to the server.
    """

    # Generated text
    generated_text: str
    # Generation details
    details: Optional[Details] = None


# `generate_stream` details
@dataclass
class StreamDetails:
    """
    Represents details of a text generation stream.

    Args:
        finish_reason (`FinishReason`):
            The reason for the generation to finish, represented by a `FinishReason` value.
        generated_tokens (`int`):
            The number of generated tokens.
        seed (`Optional[int]`):
            The sampling seed if sampling was activated.
    """

    # Generation finish reason
    finish_reason: FinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int] = None


# `generate_stream` return value
@dataclass
class TextGenerationStreamResponse:
    """
    Represents a response for text generation when `stream=True` is passed

    Args:
        token (`Token`):
            The generated token.
        generated_text (`Optional[str]`, *optional*):
            The complete generated text. Only available when the generation is finished.
        details (`Optional[StreamDetails]`, *optional*):
            Generation details. Only available when the generation is finished.
    """

    # Generated token
    token: Token
    # Complete generated text
    # Only available when the generation is finished
    generated_text: Optional[str] = None
    # Generation details
    # Only available when the generation is finished
    details: Optional[StreamDetails] = None


# TEXT GENERATION ERRORS
# ----------------------
# Text-generation errors are parsed separately to handle as much as possible the errors returned by the text generation
# inference project (https://github.com/huggingface/text-generation-inference).
# ----------------------


class TextGenerationError(HTTPError):
    """Generic error raised if text-generation went wrong."""


# Text Generation Inference Errors
class ValidationError(TextGenerationError):
    """Server-side validation error."""


class GenerationError(TextGenerationError):
    pass


class OverloadedError(TextGenerationError):
    pass


class IncompleteGenerationError(TextGenerationError):
    pass


def raise_text_generation_error(http_error: HTTPError) -> NoReturn:
    """
    Try to parse text-generation-inference error message and raise HTTPError in any case.

    Args:
        error (`HTTPError`):
            The HTTPError that have been raised.
    """
    # Try to parse a Text Generation Inference error

    try:
        # Hacky way to retrieve payload in case of aiohttp error
        payload = getattr(http_error, "response_error_payload", None) or http_error.response.json()
        message = payload.get("error")
        error_type = payload.get("error_type")
    except Exception:  # no payload
        raise http_error

    # If error_type => more information than `hf_raise_for_status`
    if error_type is not None:
        if error_type == "generation":
            raise GenerationError(message) from http_error
        if error_type == "incomplete_generation":
            raise IncompleteGenerationError(message) from http_error
        if error_type == "overloaded":
            raise OverloadedError(message) from http_error
        if error_type == "validation":
            raise ValidationError(message) from http_error

    # Otherwise, fallback to default error
    raise http_error
