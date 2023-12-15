# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
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
from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
import itertools
from typing import Iterable, overload, TypeVar

import google.ai.generativelanguage as glm

from google.generativeai.client import get_default_text_client
from google.generativeai import string_utils
from google.generativeai.types import text_types
from google.generativeai.types import model_types
from google.generativeai import models
from google.generativeai.types import safety_types

DEFAULT_TEXT_MODEL = "models/text-bison-001"
EMBEDDING_MAX_BATCH_SIZE = 100

try:
    # python 3.12+
    _batched = itertools.batched  # type: ignore
except AttributeError:
    T = TypeVar("T")

    def _batched(iterable: Iterable[T], n: int) -> Iterable[list[T]]:
        if n < 1:
            raise ValueError(f"Batch size `n` must be >1, got: {n}")
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == n:
                yield batch
                batch = []

        if batch:
            yield batch


def _make_text_prompt(prompt: str | dict[str, str]) -> glm.TextPrompt:
    """
    Creates a `glm.TextPrompt` object based on the provided prompt input.

    Args:
        prompt: The prompt input, either a string or a dictionary.

    Returns:
        glm.TextPrompt: A TextPrompt object containing the prompt text.

    Raises:
        TypeError: If the provided prompt is neither a string nor a dictionary.
    """
    if isinstance(prompt, str):
        return glm.TextPrompt(text=prompt)
    elif isinstance(prompt, dict):
        return glm.TextPrompt(prompt)
    else:
        TypeError("Expected string or dictionary for text prompt.")


def _make_generate_text_request(
    *,
    model: model_types.AnyModelNameOptions = DEFAULT_TEXT_MODEL,
    prompt: str | None = None,
    temperature: float | None = None,
    candidate_count: int | None = None,
    max_output_tokens: int | None = None,
    top_p: int | None = None,
    top_k: int | None = None,
    safety_settings: safety_types.SafetySettingOptions | None = None,
    stop_sequences: str | Iterable[str] | None = None,
) -> glm.GenerateTextRequest:
    """
    Creates a `glm.GenerateTextRequest` object based on the provided parameters.

    This function generates a `glm.GenerateTextRequest` object with the specified
    parameters. It prepares the input parameters and creates a request that can be
    used for generating text using the chosen model.

    Args:
        model: The model to use for text generation.
        prompt: The prompt for text generation. Defaults to None.
        temperature: The temperature for randomness in generation. Defaults to None.
        candidate_count: The number of candidates to consider. Defaults to None.
        max_output_tokens: The maximum number of output tokens. Defaults to None.
        top_p: The nucleus sampling probability threshold. Defaults to None.
        top_k: The top-k sampling parameter. Defaults to None.
        safety_settings: Safety settings for generated text. Defaults to None.
        stop_sequences: Stop sequences to halt text generation. Can be a string
             or iterable of strings. Defaults to None.

    Returns:
        `glm.GenerateTextRequest`: A `GenerateTextRequest` object configured with the specified parameters.
    """
    model = model_types.make_model_name(model)
    prompt = _make_text_prompt(prompt=prompt)
    safety_settings = safety_types.normalize_safety_settings(safety_settings)
    if isinstance(stop_sequences, str):
        stop_sequences = [stop_sequences]
    if stop_sequences:
        stop_sequences = list(stop_sequences)

    return glm.GenerateTextRequest(
        model=model,
        prompt=prompt,
        temperature=temperature,
        candidate_count=candidate_count,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        top_k=top_k,
        safety_settings=safety_settings,
        stop_sequences=stop_sequences,
    )


def generate_text(
    *,
    model: model_types.AnyModelNameOptions = DEFAULT_TEXT_MODEL,
    prompt: str,
    temperature: float | None = None,
    candidate_count: int | None = None,
    max_output_tokens: int | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    safety_settings: safety_types.SafetySettingOptions | None = None,
    stop_sequences: str | Iterable[str] | None = None,
    client: glm.TextServiceClient | None = None,
) -> text_types.Completion:
    """Calls the API and returns a `types.Completion` containing the response.

    Args:
        model: Which model to call, as a string or a `types.Model`.
        prompt: Free-form input text given to the model. Given a prompt, the model will
                generate text that completes the input text.
        temperature: Controls the randomness of the output. Must be positive.
            Typical values are in the range: `[0.0,1.0]`. Higher values produce a
            more random and varied response. A temperature of zero will be deterministic.
        candidate_count: The **maximum** number of generated response messages to return.
            This value must be between `[1, 8]`, inclusive. If unset, this
            will default to `1`.

            Note: Only unique candidates are returned. Higher temperatures are more
            likely to produce unique candidates. Setting `temperature=0.0` will always
            return 1 candidate regardless of the `candidate_count`.
        max_output_tokens: Maximum number of tokens to include in a candidate. Must be greater
                           than zero. If unset, will default to 64.
        top_k: The API uses combined [nucleus](https://arxiv.org/abs/1904.09751) and top-k sampling.
            `top_k` sets the maximum number of tokens to sample from on each step.
        top_p: The API uses combined [nucleus](https://arxiv.org/abs/1904.09751) and top-k sampling.
            `top_p` configures the nucleus sampling. It sets the maximum cumulative
            probability of tokens to sample from.
            For example, if the sorted probabilities are
            `[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]` a `top_p` of `0.8` will sample
            as `[0.625, 0.25, 0.125, 0, 0, 0]`.
        safety_settings: A list of unique `types.SafetySetting` instances for blocking unsafe content.
           These will be enforced on the `prompt` and
           `candidates`. There should not be more than one
           setting for each `types.SafetyCategory` type. The API will block any prompts and
           responses that fail to meet the thresholds set by these settings. This list
           overrides the default settings for each `SafetyCategory` specified in the
           safety_settings. If there is no `types.SafetySetting` for a given
           `SafetyCategory` provided in the list, the API will use the default safety
           setting for that category.
        stop_sequences: A set of up to 5 character sequences that will stop output generation.
          If specified, the API will stop at the first appearance of a stop
          sequence. The stop sequence will not be included as part of the response.
        client: If you're not relying on a default client, you pass a `glm.TextServiceClient` instead.

    Returns:
        A `types.Completion` containing the model's text completion response.
    """
    request = _make_generate_text_request(
        model=model,
        prompt=prompt,
        temperature=temperature,
        candidate_count=candidate_count,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        top_k=top_k,
        safety_settings=safety_settings,
        stop_sequences=stop_sequences,
    )

    return _generate_response(client=client, request=request)


@string_utils.prettyprint
@dataclasses.dataclass(init=False)
class Completion(text_types.Completion):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.result = None
        if self.candidates:
            self.result = self.candidates[0]["output"]


def _generate_response(
    request: glm.GenerateTextRequest, client: glm.TextServiceClient = None
) -> Completion:
    """
    Generates a response using the provided `glm.GenerateTextRequest` and client.

    Args:
        request: The text generation request.
        client: The client to use for text generation. Defaults to None, in which
            case the default text client is used.

    Returns:
        `Completion`: A `Completion` object with the generated text and response information.
    """
    if client is None:
        client = get_default_text_client()

    response = client.generate_text(request)
    response = type(response).to_dict(response)

    response["filters"] = safety_types.convert_filters_to_enums(response["filters"])
    response["safety_feedback"] = safety_types.convert_safety_feedback_to_enums(
        response["safety_feedback"]
    )
    response["candidates"] = safety_types.convert_candidate_enums(response["candidates"])

    return Completion(_client=client, **response)


def count_text_tokens(
    model: model_types.AnyModelNameOptions,
    prompt: str,
    client: glm.TextServiceClient | None = None,
) -> text_types.TokenCount:
    base_model = models.get_base_model_name(model)

    if client is None:
        client = get_default_text_client()

    result = client.count_text_tokens(
        glm.CountTextTokensRequest(model=base_model, prompt={"text": prompt})
    )

    return type(result).to_dict(result)


@overload
def generate_embeddings(
    model: model_types.BaseModelNameOptions,
    text: str,
    client: glm.TextServiceClient = None,
) -> text_types.EmbeddingDict:
    ...


@overload
def generate_embeddings(
    model: model_types.BaseModelNameOptions,
    text: Sequence[str],
    client: glm.TextServiceClient = None,
) -> text_types.BatchEmbeddingDict:
    ...


def generate_embeddings(
    model: model_types.BaseModelNameOptions,
    text: str | Sequence[str],
    client: glm.TextServiceClient = None,
) -> text_types.EmbeddingDict | text_types.BatchEmbeddingDict:
    """Calls the API to create an embedding for the text passed in.

    Args:
        model: Which model to call, as a string or a `types.Model`.

        text: Free-form input text given to the model. Given a string, the model will
              generate an embedding based on the input text.

        client: If you're not relying on a default client, you pass a `glm.TextServiceClient` instead.

    Returns:
        Dictionary containing the embedding (list of float values) for the input text.
    """
    model = model_types.make_model_name(model)

    if client is None:
        client = get_default_text_client()

    if isinstance(text, str):
        embedding_request = glm.EmbedTextRequest(model=model, text=text)
        embedding_response = client.embed_text(embedding_request)
        embedding_dict = type(embedding_response).to_dict(embedding_response)
        embedding_dict["embedding"] = embedding_dict["embedding"]["value"]
    else:
        result = {"embedding": []}
        for batch in _batched(text, EMBEDDING_MAX_BATCH_SIZE):
            # TODO(markdaoust): This could use an option for returning an iterator or wait-bar.
            embedding_request = glm.BatchEmbedTextRequest(model=model, texts=batch)
            embedding_response = client.batch_embed_text(embedding_request)
            embedding_dict = type(embedding_response).to_dict(embedding_response)
            result["embedding"].extend(e["value"] for e in embedding_dict["embeddings"])
        return result

    return embedding_dict
