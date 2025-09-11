# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore

from google.ai.generativelanguage_v1beta.types import citation, safety

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1beta",
    manifest={
        "GenerateMessageRequest",
        "GenerateMessageResponse",
        "Message",
        "MessagePrompt",
        "Example",
        "CountMessageTokensRequest",
        "CountMessageTokensResponse",
    },
)


class GenerateMessageRequest(proto.Message):
    r"""Request to generate a message response from the model.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        model (str):
            Required. The name of the model to use.

            Format: ``name=models/{model}``.
        prompt (google.ai.generativelanguage_v1beta.types.MessagePrompt):
            Required. The structured textual input given
            to the model as a prompt.
            Given a
            prompt, the model will return what it predicts
            is the next message in the discussion.
        temperature (float):
            Optional. Controls the randomness of the output.

            Values can range over ``[0.0,1.0]``, inclusive. A value
            closer to ``1.0`` will produce responses that are more
            varied, while a value closer to ``0.0`` will typically
            result in less surprising responses from the model.

            This field is a member of `oneof`_ ``_temperature``.
        candidate_count (int):
            Optional. The number of generated response messages to
            return.

            This value must be between ``[1, 8]``, inclusive. If unset,
            this will default to ``1``.

            This field is a member of `oneof`_ ``_candidate_count``.
        top_p (float):
            Optional. The maximum cumulative probability of tokens to
            consider when sampling.

            The model uses combined Top-k and nucleus sampling.

            Nucleus sampling considers the smallest set of tokens whose
            probability sum is at least ``top_p``.

            This field is a member of `oneof`_ ``_top_p``.
        top_k (int):
            Optional. The maximum number of tokens to consider when
            sampling.

            The model uses combined Top-k and nucleus sampling.

            Top-k sampling considers the set of ``top_k`` most probable
            tokens.

            This field is a member of `oneof`_ ``_top_k``.
    """

    model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    prompt: "MessagePrompt" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="MessagePrompt",
    )
    temperature: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )
    candidate_count: int = proto.Field(
        proto.INT32,
        number=4,
        optional=True,
    )
    top_p: float = proto.Field(
        proto.FLOAT,
        number=5,
        optional=True,
    )
    top_k: int = proto.Field(
        proto.INT32,
        number=6,
        optional=True,
    )


class GenerateMessageResponse(proto.Message):
    r"""The response from the model.

    This includes candidate messages and
    conversation history in the form of chronologically-ordered
    messages.

    Attributes:
        candidates (MutableSequence[google.ai.generativelanguage_v1beta.types.Message]):
            Candidate response messages from the model.
        messages (MutableSequence[google.ai.generativelanguage_v1beta.types.Message]):
            The conversation history used by the model.
        filters (MutableSequence[google.ai.generativelanguage_v1beta.types.ContentFilter]):
            A set of content filtering metadata for the prompt and
            response text.

            This indicates which ``SafetyCategory``\ (s) blocked a
            candidate from this response, the lowest ``HarmProbability``
            that triggered a block, and the HarmThreshold setting for
            that category.
    """

    candidates: MutableSequence["Message"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Message",
    )
    messages: MutableSequence["Message"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="Message",
    )
    filters: MutableSequence[safety.ContentFilter] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=safety.ContentFilter,
    )


class Message(proto.Message):
    r"""The base unit of structured text.

    A ``Message`` includes an ``author`` and the ``content`` of the
    ``Message``.

    The ``author`` is used to tag messages when they are fed to the
    model as text.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        author (str):
            Optional. The author of this Message.

            This serves as a key for tagging
            the content of this Message when it is fed to
            the model as text.

            The author can be any alphanumeric string.
        content (str):
            Required. The text content of the structured ``Message``.
        citation_metadata (google.ai.generativelanguage_v1beta.types.CitationMetadata):
            Output only. Citation information for model-generated
            ``content`` in this ``Message``.

            If this ``Message`` was generated as output from the model,
            this field may be populated with attribution information for
            any text included in the ``content``. This field is used
            only on output.

            This field is a member of `oneof`_ ``_citation_metadata``.
    """

    author: str = proto.Field(
        proto.STRING,
        number=1,
    )
    content: str = proto.Field(
        proto.STRING,
        number=2,
    )
    citation_metadata: citation.CitationMetadata = proto.Field(
        proto.MESSAGE,
        number=3,
        optional=True,
        message=citation.CitationMetadata,
    )


class MessagePrompt(proto.Message):
    r"""All of the structured input text passed to the model as a prompt.

    A ``MessagePrompt`` contains a structured set of fields that provide
    context for the conversation, examples of user input/model output
    message pairs that prime the model to respond in different ways, and
    the conversation history or list of messages representing the
    alternating turns of the conversation between the user and the
    model.

    Attributes:
        context (str):
            Optional. Text that should be provided to the model first to
            ground the response.

            If not empty, this ``context`` will be given to the model
            first before the ``examples`` and ``messages``. When using a
            ``context`` be sure to provide it with every request to
            maintain continuity.

            This field can be a description of your prompt to the model
            to help provide context and guide the responses. Examples:
            "Translate the phrase from English to French." or "Given a
            statement, classify the sentiment as happy, sad or neutral."

            Anything included in this field will take precedence over
            message history if the total input size exceeds the model's
            ``input_token_limit`` and the input request is truncated.
        examples (MutableSequence[google.ai.generativelanguage_v1beta.types.Example]):
            Optional. Examples of what the model should generate.

            This includes both user input and the response that the
            model should emulate.

            These ``examples`` are treated identically to conversation
            messages except that they take precedence over the history
            in ``messages``: If the total input size exceeds the model's
            ``input_token_limit`` the input will be truncated. Items
            will be dropped from ``messages`` before ``examples``.
        messages (MutableSequence[google.ai.generativelanguage_v1beta.types.Message]):
            Required. A snapshot of the recent conversation history
            sorted chronologically.

            Turns alternate between two authors.

            If the total input size exceeds the model's
            ``input_token_limit`` the input will be truncated: The
            oldest items will be dropped from ``messages``.
    """

    context: str = proto.Field(
        proto.STRING,
        number=1,
    )
    examples: MutableSequence["Example"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="Example",
    )
    messages: MutableSequence["Message"] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="Message",
    )


class Example(proto.Message):
    r"""An input/output example used to instruct the Model.

    It demonstrates how the model should respond or format its
    response.

    Attributes:
        input (google.ai.generativelanguage_v1beta.types.Message):
            Required. An example of an input ``Message`` from the user.
        output (google.ai.generativelanguage_v1beta.types.Message):
            Required. An example of what the model should
            output given the input.
    """

    input: "Message" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="Message",
    )
    output: "Message" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="Message",
    )


class CountMessageTokensRequest(proto.Message):
    r"""Counts the number of tokens in the ``prompt`` sent to a model.

    Models may tokenize text differently, so each model may return a
    different ``token_count``.

    Attributes:
        model (str):
            Required. The model's resource name. This serves as an ID
            for the Model to use.

            This name should match a model name returned by the
            ``ListModels`` method.

            Format: ``models/{model}``
        prompt (google.ai.generativelanguage_v1beta.types.MessagePrompt):
            Required. The prompt, whose token count is to
            be returned.
    """

    model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    prompt: "MessagePrompt" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="MessagePrompt",
    )


class CountMessageTokensResponse(proto.Message):
    r"""A response from ``CountMessageTokens``.

    It returns the model's ``token_count`` for the ``prompt``.

    Attributes:
        token_count (int):
            The number of tokens that the ``model`` tokenizes the
            ``prompt`` into.

            Always non-negative.
    """

    token_count: int = proto.Field(
        proto.INT32,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
