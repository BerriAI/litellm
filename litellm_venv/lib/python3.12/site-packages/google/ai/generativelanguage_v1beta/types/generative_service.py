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

from google.ai.generativelanguage_v1beta.types import citation
from google.ai.generativelanguage_v1beta.types import content as gag_content
from google.ai.generativelanguage_v1beta.types import retriever, safety

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1beta",
    manifest={
        "TaskType",
        "GenerateContentRequest",
        "GenerationConfig",
        "SemanticRetrieverConfig",
        "GenerateContentResponse",
        "Candidate",
        "AttributionSourceId",
        "GroundingAttribution",
        "GenerateAnswerRequest",
        "GenerateAnswerResponse",
        "EmbedContentRequest",
        "ContentEmbedding",
        "EmbedContentResponse",
        "BatchEmbedContentsRequest",
        "BatchEmbedContentsResponse",
        "CountTokensRequest",
        "CountTokensResponse",
    },
)


class TaskType(proto.Enum):
    r"""Type of task for which the embedding will be used.

    Values:
        TASK_TYPE_UNSPECIFIED (0):
            Unset value, which will default to one of the
            other enum values.
        RETRIEVAL_QUERY (1):
            Specifies the given text is a query in a
            search/retrieval setting.
        RETRIEVAL_DOCUMENT (2):
            Specifies the given text is a document from
            the corpus being searched.
        SEMANTIC_SIMILARITY (3):
            Specifies the given text will be used for
            STS.
        CLASSIFICATION (4):
            Specifies that the given text will be
            classified.
        CLUSTERING (5):
            Specifies that the embeddings will be used
            for clustering.
    """
    TASK_TYPE_UNSPECIFIED = 0
    RETRIEVAL_QUERY = 1
    RETRIEVAL_DOCUMENT = 2
    SEMANTIC_SIMILARITY = 3
    CLASSIFICATION = 4
    CLUSTERING = 5


class GenerateContentRequest(proto.Message):
    r"""Request to generate a completion from the model.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        model (str):
            Required. The name of the ``Model`` to use for generating
            the completion.

            Format: ``name=models/{model}``.
        system_instruction (google.ai.generativelanguage_v1beta.types.Content):
            Optional. Developer set system instruction.
            Currently, text only.

            This field is a member of `oneof`_ ``_system_instruction``.
        contents (MutableSequence[google.ai.generativelanguage_v1beta.types.Content]):
            Required. The content of the current
            conversation with the model.
            For single-turn queries, this is a single
            instance. For multi-turn queries, this is a
            repeated field that contains conversation
            history + latest request.
        tools (MutableSequence[google.ai.generativelanguage_v1beta.types.Tool]):
            Optional. A list of ``Tools`` the model may use to generate
            the next response.

            A ``Tool`` is a piece of code that enables the system to
            interact with external systems to perform an action, or set
            of actions, outside of knowledge and scope of the model. The
            only supported tool is currently ``Function``.
        tool_config (google.ai.generativelanguage_v1beta.types.ToolConfig):
            Optional. Tool configuration for any ``Tool`` specified in
            the request.
        safety_settings (MutableSequence[google.ai.generativelanguage_v1beta.types.SafetySetting]):
            Optional. A list of unique ``SafetySetting`` instances for
            blocking unsafe content.

            This will be enforced on the
            ``GenerateContentRequest.contents`` and
            ``GenerateContentResponse.candidates``. There should not be
            more than one setting for each ``SafetyCategory`` type. The
            API will block any contents and responses that fail to meet
            the thresholds set by these settings. This list overrides
            the default settings for each ``SafetyCategory`` specified
            in the safety_settings. If there is no ``SafetySetting`` for
            a given ``SafetyCategory`` provided in the list, the API
            will use the default safety setting for that category. Harm
            categories HARM_CATEGORY_HATE_SPEECH,
            HARM_CATEGORY_SEXUALLY_EXPLICIT,
            HARM_CATEGORY_DANGEROUS_CONTENT, HARM_CATEGORY_HARASSMENT
            are supported.
        generation_config (google.ai.generativelanguage_v1beta.types.GenerationConfig):
            Optional. Configuration options for model
            generation and outputs.

            This field is a member of `oneof`_ ``_generation_config``.
    """

    model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    system_instruction: gag_content.Content = proto.Field(
        proto.MESSAGE,
        number=8,
        optional=True,
        message=gag_content.Content,
    )
    contents: MutableSequence[gag_content.Content] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=gag_content.Content,
    )
    tools: MutableSequence[gag_content.Tool] = proto.RepeatedField(
        proto.MESSAGE,
        number=5,
        message=gag_content.Tool,
    )
    tool_config: gag_content.ToolConfig = proto.Field(
        proto.MESSAGE,
        number=7,
        message=gag_content.ToolConfig,
    )
    safety_settings: MutableSequence[safety.SafetySetting] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=safety.SafetySetting,
    )
    generation_config: "GenerationConfig" = proto.Field(
        proto.MESSAGE,
        number=4,
        optional=True,
        message="GenerationConfig",
    )


class GenerationConfig(proto.Message):
    r"""Configuration options for model generation and outputs. Not
    all parameters may be configurable for every model.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        candidate_count (int):
            Optional. Number of generated responses to
            return.
            Currently, this value can only be set to 1. If
            unset, this will default to 1.

            This field is a member of `oneof`_ ``_candidate_count``.
        stop_sequences (MutableSequence[str]):
            Optional. The set of character sequences (up
            to 5) that will stop output generation. If
            specified, the API will stop at the first
            appearance of a stop sequence. The stop sequence
            will not be included as part of the response.
        max_output_tokens (int):
            Optional. The maximum number of tokens to include in a
            candidate.

            Note: The default value varies by model, see the
            ``Model.output_token_limit`` attribute of the ``Model``
            returned from the ``getModel`` function.

            This field is a member of `oneof`_ ``_max_output_tokens``.
        temperature (float):
            Optional. Controls the randomness of the output.

            Note: The default value varies by model, see the
            ``Model.temperature`` attribute of the ``Model`` returned
            from the ``getModel`` function.

            Values can range from [0.0, infinity).

            This field is a member of `oneof`_ ``_temperature``.
        top_p (float):
            Optional. The maximum cumulative probability of tokens to
            consider when sampling.

            The model uses combined Top-k and nucleus sampling.

            Tokens are sorted based on their assigned probabilities so
            that only the most likely tokens are considered. Top-k
            sampling directly limits the maximum number of tokens to
            consider, while Nucleus sampling limits number of tokens
            based on the cumulative probability.

            Note: The default value varies by model, see the
            ``Model.top_p`` attribute of the ``Model`` returned from the
            ``getModel`` function.

            This field is a member of `oneof`_ ``_top_p``.
        top_k (int):
            Optional. The maximum number of tokens to consider when
            sampling.

            The model uses combined Top-k and nucleus sampling.

            Top-k sampling considers the set of ``top_k`` most probable
            tokens.

            Note: The default value varies by model, see the
            ``Model.top_k`` attribute of the ``Model`` returned from the
            ``getModel`` function.

            This field is a member of `oneof`_ ``_top_k``.
    """

    candidate_count: int = proto.Field(
        proto.INT32,
        number=1,
        optional=True,
    )
    stop_sequences: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )
    max_output_tokens: int = proto.Field(
        proto.INT32,
        number=4,
        optional=True,
    )
    temperature: float = proto.Field(
        proto.FLOAT,
        number=5,
        optional=True,
    )
    top_p: float = proto.Field(
        proto.FLOAT,
        number=6,
        optional=True,
    )
    top_k: int = proto.Field(
        proto.INT32,
        number=7,
        optional=True,
    )


class SemanticRetrieverConfig(proto.Message):
    r"""Configuration for retrieving grounding content from a ``Corpus`` or
    ``Document`` created using the Semantic Retriever API.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        source (str):
            Required. Name of the resource for retrieval,
            e.g. corpora/123 or corpora/123/documents/abc.
        query (google.ai.generativelanguage_v1beta.types.Content):
            Required. Query to use for similarity matching ``Chunk``\ s
            in the given resource.
        metadata_filters (MutableSequence[google.ai.generativelanguage_v1beta.types.MetadataFilter]):
            Optional. Filters for selecting ``Document``\ s and/or
            ``Chunk``\ s from the resource.
        max_chunks_count (int):
            Optional. Maximum number of relevant ``Chunk``\ s to
            retrieve.

            This field is a member of `oneof`_ ``_max_chunks_count``.
        minimum_relevance_score (float):
            Optional. Minimum relevance score for retrieved relevant
            ``Chunk``\ s.

            This field is a member of `oneof`_ ``_minimum_relevance_score``.
    """

    source: str = proto.Field(
        proto.STRING,
        number=1,
    )
    query: gag_content.Content = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gag_content.Content,
    )
    metadata_filters: MutableSequence[retriever.MetadataFilter] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=retriever.MetadataFilter,
    )
    max_chunks_count: int = proto.Field(
        proto.INT32,
        number=4,
        optional=True,
    )
    minimum_relevance_score: float = proto.Field(
        proto.FLOAT,
        number=5,
        optional=True,
    )


class GenerateContentResponse(proto.Message):
    r"""Response from the model supporting multiple candidates.

    Note on safety ratings and content filtering. They are reported for
    both prompt in ``GenerateContentResponse.prompt_feedback`` and for
    each candidate in ``finish_reason`` and in ``safety_ratings``. The
    API contract is that:

    -  either all requested candidates are returned or no candidates at
       all
    -  no candidates are returned only if there was something wrong with
       the prompt (see ``prompt_feedback``)
    -  feedback on each candidate is reported on ``finish_reason`` and
       ``safety_ratings``.

    Attributes:
        candidates (MutableSequence[google.ai.generativelanguage_v1beta.types.Candidate]):
            Candidate responses from the model.
        prompt_feedback (google.ai.generativelanguage_v1beta.types.GenerateContentResponse.PromptFeedback):
            Returns the prompt's feedback related to the
            content filters.
    """

    class PromptFeedback(proto.Message):
        r"""A set of the feedback metadata the prompt specified in
        ``GenerateContentRequest.content``.

        Attributes:
            block_reason (google.ai.generativelanguage_v1beta.types.GenerateContentResponse.PromptFeedback.BlockReason):
                Optional. If set, the prompt was blocked and
                no candidates are returned. Rephrase your
                prompt.
            safety_ratings (MutableSequence[google.ai.generativelanguage_v1beta.types.SafetyRating]):
                Ratings for safety of the prompt.
                There is at most one rating per category.
        """

        class BlockReason(proto.Enum):
            r"""Specifies what was the reason why prompt was blocked.

            Values:
                BLOCK_REASON_UNSPECIFIED (0):
                    Default value. This value is unused.
                SAFETY (1):
                    Prompt was blocked due to safety reasons. You can inspect
                    ``safety_ratings`` to understand which safety category
                    blocked it.
                OTHER (2):
                    Prompt was blocked due to unknown reaasons.
            """
            BLOCK_REASON_UNSPECIFIED = 0
            SAFETY = 1
            OTHER = 2

        block_reason: "GenerateContentResponse.PromptFeedback.BlockReason" = (
            proto.Field(
                proto.ENUM,
                number=1,
                enum="GenerateContentResponse.PromptFeedback.BlockReason",
            )
        )
        safety_ratings: MutableSequence[safety.SafetyRating] = proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message=safety.SafetyRating,
        )

    candidates: MutableSequence["Candidate"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Candidate",
    )
    prompt_feedback: PromptFeedback = proto.Field(
        proto.MESSAGE,
        number=2,
        message=PromptFeedback,
    )


class Candidate(proto.Message):
    r"""A response candidate generated from the model.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        index (int):
            Output only. Index of the candidate in the
            list of candidates.

            This field is a member of `oneof`_ ``_index``.
        content (google.ai.generativelanguage_v1beta.types.Content):
            Output only. Generated content returned from
            the model.
        finish_reason (google.ai.generativelanguage_v1beta.types.Candidate.FinishReason):
            Optional. Output only. The reason why the
            model stopped generating tokens.
            If empty, the model has not stopped generating
            the tokens.
        safety_ratings (MutableSequence[google.ai.generativelanguage_v1beta.types.SafetyRating]):
            List of ratings for the safety of a response
            candidate.
            There is at most one rating per category.
        citation_metadata (google.ai.generativelanguage_v1beta.types.CitationMetadata):
            Output only. Citation information for model-generated
            candidate.

            This field may be populated with recitation information for
            any text included in the ``content``. These are passages
            that are "recited" from copyrighted material in the
            foundational LLM's training data.
        token_count (int):
            Output only. Token count for this candidate.
        grounding_attributions (MutableSequence[google.ai.generativelanguage_v1beta.types.GroundingAttribution]):
            Output only. Attribution information for sources that
            contributed to a grounded answer.

            This field is populated for ``GenerateAnswer`` calls.
    """

    class FinishReason(proto.Enum):
        r"""Defines the reason why the model stopped generating tokens.

        Values:
            FINISH_REASON_UNSPECIFIED (0):
                Default value. This value is unused.
            STOP (1):
                Natural stop point of the model or provided
                stop sequence.
            MAX_TOKENS (2):
                The maximum number of tokens as specified in
                the request was reached.
            SAFETY (3):
                The candidate content was flagged for safety
                reasons.
            RECITATION (4):
                The candidate content was flagged for
                recitation reasons.
            OTHER (5):
                Unknown reason.
        """
        FINISH_REASON_UNSPECIFIED = 0
        STOP = 1
        MAX_TOKENS = 2
        SAFETY = 3
        RECITATION = 4
        OTHER = 5

    index: int = proto.Field(
        proto.INT32,
        number=3,
        optional=True,
    )
    content: gag_content.Content = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gag_content.Content,
    )
    finish_reason: FinishReason = proto.Field(
        proto.ENUM,
        number=2,
        enum=FinishReason,
    )
    safety_ratings: MutableSequence[safety.SafetyRating] = proto.RepeatedField(
        proto.MESSAGE,
        number=5,
        message=safety.SafetyRating,
    )
    citation_metadata: citation.CitationMetadata = proto.Field(
        proto.MESSAGE,
        number=6,
        message=citation.CitationMetadata,
    )
    token_count: int = proto.Field(
        proto.INT32,
        number=7,
    )
    grounding_attributions: MutableSequence[
        "GroundingAttribution"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=8,
        message="GroundingAttribution",
    )


class AttributionSourceId(proto.Message):
    r"""Identifier for the source contributing to this attribution.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        grounding_passage (google.ai.generativelanguage_v1beta.types.AttributionSourceId.GroundingPassageId):
            Identifier for an inline passage.

            This field is a member of `oneof`_ ``source``.
        semantic_retriever_chunk (google.ai.generativelanguage_v1beta.types.AttributionSourceId.SemanticRetrieverChunk):
            Identifier for a ``Chunk`` fetched via Semantic Retriever.

            This field is a member of `oneof`_ ``source``.
    """

    class GroundingPassageId(proto.Message):
        r"""Identifier for a part within a ``GroundingPassage``.

        Attributes:
            passage_id (str):
                Output only. ID of the passage matching the
                ``GenerateAnswerRequest``'s ``GroundingPassage.id``.
            part_index (int):
                Output only. Index of the part within the
                ``GenerateAnswerRequest``'s ``GroundingPassage.content``.
        """

        passage_id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        part_index: int = proto.Field(
            proto.INT32,
            number=2,
        )

    class SemanticRetrieverChunk(proto.Message):
        r"""Identifier for a ``Chunk`` retrieved via Semantic Retriever
        specified in the ``GenerateAnswerRequest`` using
        ``SemanticRetrieverConfig``.

        Attributes:
            source (str):
                Output only. Name of the source matching the request's
                ``SemanticRetrieverConfig.source``. Example: ``corpora/123``
                or ``corpora/123/documents/abc``
            chunk (str):
                Output only. Name of the ``Chunk`` containing the attributed
                text. Example: ``corpora/123/documents/abc/chunks/xyz``
        """

        source: str = proto.Field(
            proto.STRING,
            number=1,
        )
        chunk: str = proto.Field(
            proto.STRING,
            number=2,
        )

    grounding_passage: GroundingPassageId = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="source",
        message=GroundingPassageId,
    )
    semantic_retriever_chunk: SemanticRetrieverChunk = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="source",
        message=SemanticRetrieverChunk,
    )


class GroundingAttribution(proto.Message):
    r"""Attribution for a source that contributed to an answer.

    Attributes:
        source_id (google.ai.generativelanguage_v1beta.types.AttributionSourceId):
            Output only. Identifier for the source
            contributing to this attribution.
        content (google.ai.generativelanguage_v1beta.types.Content):
            Grounding source content that makes up this
            attribution.
    """

    source_id: "AttributionSourceId" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="AttributionSourceId",
    )
    content: gag_content.Content = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gag_content.Content,
    )


class GenerateAnswerRequest(proto.Message):
    r"""Request to generate a grounded answer from the model.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        inline_passages (google.ai.generativelanguage_v1beta.types.GroundingPassages):
            Passages provided inline with the request.

            This field is a member of `oneof`_ ``grounding_source``.
        semantic_retriever (google.ai.generativelanguage_v1beta.types.SemanticRetrieverConfig):
            Content retrieved from resources created via
            the Semantic Retriever API.

            This field is a member of `oneof`_ ``grounding_source``.
        model (str):
            Required. The name of the ``Model`` to use for generating
            the grounded response.

            Format: ``model=models/{model}``.
        contents (MutableSequence[google.ai.generativelanguage_v1beta.types.Content]):
            Required. The content of the current conversation with the
            model. For single-turn queries, this is a single question to
            answer. For multi-turn queries, this is a repeated field
            that contains conversation history and the last ``Content``
            in the list containing the question.

            Note: GenerateAnswer currently only supports queries in
            English.
        answer_style (google.ai.generativelanguage_v1beta.types.GenerateAnswerRequest.AnswerStyle):
            Required. Style in which answers should be
            returned.
        safety_settings (MutableSequence[google.ai.generativelanguage_v1beta.types.SafetySetting]):
            Optional. A list of unique ``SafetySetting`` instances for
            blocking unsafe content.

            This will be enforced on the
            ``GenerateAnswerRequest.contents`` and
            ``GenerateAnswerResponse.candidate``. There should not be
            more than one setting for each ``SafetyCategory`` type. The
            API will block any contents and responses that fail to meet
            the thresholds set by these settings. This list overrides
            the default settings for each ``SafetyCategory`` specified
            in the safety_settings. If there is no ``SafetySetting`` for
            a given ``SafetyCategory`` provided in the list, the API
            will use the default safety setting for that category. Harm
            categories HARM_CATEGORY_HATE_SPEECH,
            HARM_CATEGORY_SEXUALLY_EXPLICIT,
            HARM_CATEGORY_DANGEROUS_CONTENT, HARM_CATEGORY_HARASSMENT
            are supported.
        temperature (float):
            Optional. Controls the randomness of the output.

            Values can range from [0.0,1.0], inclusive. A value closer
            to 1.0 will produce responses that are more varied and
            creative, while a value closer to 0.0 will typically result
            in more straightforward responses from the model. A low
            temperature (~0.2) is usually recommended for
            Attributed-Question-Answering use cases.

            This field is a member of `oneof`_ ``_temperature``.
    """

    class AnswerStyle(proto.Enum):
        r"""Style for grounded answers.

        Values:
            ANSWER_STYLE_UNSPECIFIED (0):
                Unspecified answer style.
            ABSTRACTIVE (1):
                Succint but abstract style.
            EXTRACTIVE (2):
                Very brief and extractive style.
            VERBOSE (3):
                Verbose style including extra details. The
                response may be formatted as a sentence,
                paragraph, multiple paragraphs, or bullet
                points, etc.
        """
        ANSWER_STYLE_UNSPECIFIED = 0
        ABSTRACTIVE = 1
        EXTRACTIVE = 2
        VERBOSE = 3

    inline_passages: gag_content.GroundingPassages = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="grounding_source",
        message=gag_content.GroundingPassages,
    )
    semantic_retriever: "SemanticRetrieverConfig" = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="grounding_source",
        message="SemanticRetrieverConfig",
    )
    model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    contents: MutableSequence[gag_content.Content] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=gag_content.Content,
    )
    answer_style: AnswerStyle = proto.Field(
        proto.ENUM,
        number=5,
        enum=AnswerStyle,
    )
    safety_settings: MutableSequence[safety.SafetySetting] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=safety.SafetySetting,
    )
    temperature: float = proto.Field(
        proto.FLOAT,
        number=4,
        optional=True,
    )


class GenerateAnswerResponse(proto.Message):
    r"""Response from the model for a grounded answer.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        answer (google.ai.generativelanguage_v1beta.types.Candidate):
            Candidate answer from the model.

            Note: The model *always* attempts to provide a grounded
            answer, even when the answer is unlikely to be answerable
            from the given passages. In that case, a low-quality or
            ungrounded answer may be provided, along with a low
            ``answerable_probability``.
        answerable_probability (float):
            Output only. The model's estimate of the probability that
            its answer is correct and grounded in the input passages.

            A low answerable_probability indicates that the answer might
            not be grounded in the sources.

            When ``answerable_probability`` is low, some clients may
            wish to:

            -  Display a message to the effect of "We couldn’t answer
               that question" to the user.
            -  Fall back to a general-purpose LLM that answers the
               question from world knowledge. The threshold and nature
               of such fallbacks will depend on individual clients’ use
               cases. 0.5 is a good starting threshold.

            This field is a member of `oneof`_ ``_answerable_probability``.
        input_feedback (google.ai.generativelanguage_v1beta.types.GenerateAnswerResponse.InputFeedback):
            Output only. Feedback related to the input data used to
            answer the question, as opposed to model-generated response
            to the question.

            "Input data" can be one or more of the following:

            -  Question specified by the last entry in
               ``GenerateAnswerRequest.content``
            -  Conversation history specified by the other entries in
               ``GenerateAnswerRequest.content``
            -  Grounding sources
               (``GenerateAnswerRequest.semantic_retriever`` or
               ``GenerateAnswerRequest.inline_passages``)

            This field is a member of `oneof`_ ``_input_feedback``.
    """

    class InputFeedback(proto.Message):
        r"""Feedback related to the input data used to answer the
        question, as opposed to model-generated response to the
        question.


        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            block_reason (google.ai.generativelanguage_v1beta.types.GenerateAnswerResponse.InputFeedback.BlockReason):
                Optional. If set, the input was blocked and
                no candidates are returned. Rephrase your input.

                This field is a member of `oneof`_ ``_block_reason``.
            safety_ratings (MutableSequence[google.ai.generativelanguage_v1beta.types.SafetyRating]):
                Ratings for safety of the input.
                There is at most one rating per category.
        """

        class BlockReason(proto.Enum):
            r"""Specifies what was the reason why input was blocked.

            Values:
                BLOCK_REASON_UNSPECIFIED (0):
                    Default value. This value is unused.
                SAFETY (1):
                    Input was blocked due to safety reasons. You can inspect
                    ``safety_ratings`` to understand which safety category
                    blocked it.
                OTHER (2):
                    Input was blocked due to other reasons.
            """
            BLOCK_REASON_UNSPECIFIED = 0
            SAFETY = 1
            OTHER = 2

        block_reason: "GenerateAnswerResponse.InputFeedback.BlockReason" = proto.Field(
            proto.ENUM,
            number=1,
            optional=True,
            enum="GenerateAnswerResponse.InputFeedback.BlockReason",
        )
        safety_ratings: MutableSequence[safety.SafetyRating] = proto.RepeatedField(
            proto.MESSAGE,
            number=2,
            message=safety.SafetyRating,
        )

    answer: "Candidate" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="Candidate",
    )
    answerable_probability: float = proto.Field(
        proto.FLOAT,
        number=2,
        optional=True,
    )
    input_feedback: InputFeedback = proto.Field(
        proto.MESSAGE,
        number=3,
        optional=True,
        message=InputFeedback,
    )


class EmbedContentRequest(proto.Message):
    r"""Request containing the ``Content`` for the model to embed.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        model (str):
            Required. The model's resource name. This serves as an ID
            for the Model to use.

            This name should match a model name returned by the
            ``ListModels`` method.

            Format: ``models/{model}``
        content (google.ai.generativelanguage_v1beta.types.Content):
            Required. The content to embed. Only the ``parts.text``
            fields will be counted.
        task_type (google.ai.generativelanguage_v1beta.types.TaskType):
            Optional. Optional task type for which the embeddings will
            be used. Can only be set for ``models/embedding-001``.

            This field is a member of `oneof`_ ``_task_type``.
        title (str):
            Optional. An optional title for the text. Only applicable
            when TaskType is ``RETRIEVAL_DOCUMENT``.

            Note: Specifying a ``title`` for ``RETRIEVAL_DOCUMENT``
            provides better quality embeddings for retrieval.

            This field is a member of `oneof`_ ``_title``.
    """

    model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    content: gag_content.Content = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gag_content.Content,
    )
    task_type: "TaskType" = proto.Field(
        proto.ENUM,
        number=3,
        optional=True,
        enum="TaskType",
    )
    title: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class ContentEmbedding(proto.Message):
    r"""A list of floats representing an embedding.

    Attributes:
        values (MutableSequence[float]):
            The embedding values.
    """

    values: MutableSequence[float] = proto.RepeatedField(
        proto.FLOAT,
        number=1,
    )


class EmbedContentResponse(proto.Message):
    r"""The response to an ``EmbedContentRequest``.

    Attributes:
        embedding (google.ai.generativelanguage_v1beta.types.ContentEmbedding):
            Output only. The embedding generated from the
            input content.
    """

    embedding: "ContentEmbedding" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ContentEmbedding",
    )


class BatchEmbedContentsRequest(proto.Message):
    r"""Batch request to get embeddings from the model for a list of
    prompts.

    Attributes:
        model (str):
            Required. The model's resource name. This serves as an ID
            for the Model to use.

            This name should match a model name returned by the
            ``ListModels`` method.

            Format: ``models/{model}``
        requests (MutableSequence[google.ai.generativelanguage_v1beta.types.EmbedContentRequest]):
            Required. Embed requests for the batch. The model in each of
            these requests must match the model specified
            ``BatchEmbedContentsRequest.model``.
    """

    model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    requests: MutableSequence["EmbedContentRequest"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="EmbedContentRequest",
    )


class BatchEmbedContentsResponse(proto.Message):
    r"""The response to a ``BatchEmbedContentsRequest``.

    Attributes:
        embeddings (MutableSequence[google.ai.generativelanguage_v1beta.types.ContentEmbedding]):
            Output only. The embeddings for each request,
            in the same order as provided in the batch
            request.
    """

    embeddings: MutableSequence["ContentEmbedding"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="ContentEmbedding",
    )


class CountTokensRequest(proto.Message):
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
        contents (MutableSequence[google.ai.generativelanguage_v1beta.types.Content]):
            Required. The input given to the model as a
            prompt.
    """

    model: str = proto.Field(
        proto.STRING,
        number=1,
    )
    contents: MutableSequence[gag_content.Content] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=gag_content.Content,
    )


class CountTokensResponse(proto.Message):
    r"""A response from ``CountTokens``.

    It returns the model's ``token_count`` for the ``prompt``.

    Attributes:
        total_tokens (int):
            The number of tokens that the ``model`` tokenizes the
            ``prompt`` into.

            Always non-negative.
    """

    total_tokens: int = proto.Field(
        proto.INT32,
        number=1,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
