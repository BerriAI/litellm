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


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "PairwiseChoice",
        "EvaluateInstancesRequest",
        "EvaluateInstancesResponse",
        "ExactMatchInput",
        "ExactMatchInstance",
        "ExactMatchSpec",
        "ExactMatchResults",
        "ExactMatchMetricValue",
        "BleuInput",
        "BleuInstance",
        "BleuSpec",
        "BleuResults",
        "BleuMetricValue",
        "RougeInput",
        "RougeInstance",
        "RougeSpec",
        "RougeResults",
        "RougeMetricValue",
        "CoherenceInput",
        "CoherenceInstance",
        "CoherenceSpec",
        "CoherenceResult",
        "FluencyInput",
        "FluencyInstance",
        "FluencySpec",
        "FluencyResult",
        "SafetyInput",
        "SafetyInstance",
        "SafetySpec",
        "SafetyResult",
        "GroundednessInput",
        "GroundednessInstance",
        "GroundednessSpec",
        "GroundednessResult",
        "FulfillmentInput",
        "FulfillmentInstance",
        "FulfillmentSpec",
        "FulfillmentResult",
        "SummarizationQualityInput",
        "SummarizationQualityInstance",
        "SummarizationQualitySpec",
        "SummarizationQualityResult",
        "PairwiseSummarizationQualityInput",
        "PairwiseSummarizationQualityInstance",
        "PairwiseSummarizationQualitySpec",
        "PairwiseSummarizationQualityResult",
        "SummarizationHelpfulnessInput",
        "SummarizationHelpfulnessInstance",
        "SummarizationHelpfulnessSpec",
        "SummarizationHelpfulnessResult",
        "SummarizationVerbosityInput",
        "SummarizationVerbosityInstance",
        "SummarizationVerbositySpec",
        "SummarizationVerbosityResult",
        "QuestionAnsweringQualityInput",
        "QuestionAnsweringQualityInstance",
        "QuestionAnsweringQualitySpec",
        "QuestionAnsweringQualityResult",
        "PairwiseQuestionAnsweringQualityInput",
        "PairwiseQuestionAnsweringQualityInstance",
        "PairwiseQuestionAnsweringQualitySpec",
        "PairwiseQuestionAnsweringQualityResult",
        "QuestionAnsweringRelevanceInput",
        "QuestionAnsweringRelevanceInstance",
        "QuestionAnsweringRelevanceSpec",
        "QuestionAnsweringRelevanceResult",
        "QuestionAnsweringHelpfulnessInput",
        "QuestionAnsweringHelpfulnessInstance",
        "QuestionAnsweringHelpfulnessSpec",
        "QuestionAnsweringHelpfulnessResult",
        "QuestionAnsweringCorrectnessInput",
        "QuestionAnsweringCorrectnessInstance",
        "QuestionAnsweringCorrectnessSpec",
        "QuestionAnsweringCorrectnessResult",
        "ToolCallValidInput",
        "ToolCallValidSpec",
        "ToolCallValidInstance",
        "ToolCallValidResults",
        "ToolCallValidMetricValue",
        "ToolNameMatchInput",
        "ToolNameMatchSpec",
        "ToolNameMatchInstance",
        "ToolNameMatchResults",
        "ToolNameMatchMetricValue",
        "ToolParameterKeyMatchInput",
        "ToolParameterKeyMatchSpec",
        "ToolParameterKeyMatchInstance",
        "ToolParameterKeyMatchResults",
        "ToolParameterKeyMatchMetricValue",
        "ToolParameterKVMatchInput",
        "ToolParameterKVMatchSpec",
        "ToolParameterKVMatchInstance",
        "ToolParameterKVMatchResults",
        "ToolParameterKVMatchMetricValue",
    },
)


class PairwiseChoice(proto.Enum):
    r"""Pairwise prediction autorater preference.

    Values:
        PAIRWISE_CHOICE_UNSPECIFIED (0):
            Unspecified prediction choice.
        BASELINE (1):
            Baseline prediction wins
        CANDIDATE (2):
            Candidate prediction wins
        TIE (3):
            Winner cannot be determined
    """
    PAIRWISE_CHOICE_UNSPECIFIED = 0
    BASELINE = 1
    CANDIDATE = 2
    TIE = 3


class EvaluateInstancesRequest(proto.Message):
    r"""Request message for EvaluationService.EvaluateInstances.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        exact_match_input (google.cloud.aiplatform_v1beta1.types.ExactMatchInput):
            Auto metric instances.
            Instances and metric spec for exact match
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        bleu_input (google.cloud.aiplatform_v1beta1.types.BleuInput):
            Instances and metric spec for bleu metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        rouge_input (google.cloud.aiplatform_v1beta1.types.RougeInput):
            Instances and metric spec for rouge metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        fluency_input (google.cloud.aiplatform_v1beta1.types.FluencyInput):
            LLM-based metric instance.
            General text generation metrics, applicable to
            other categories. Input for fluency metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        coherence_input (google.cloud.aiplatform_v1beta1.types.CoherenceInput):
            Input for coherence metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        safety_input (google.cloud.aiplatform_v1beta1.types.SafetyInput):
            Input for safety metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        groundedness_input (google.cloud.aiplatform_v1beta1.types.GroundednessInput):
            Input for groundedness metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        fulfillment_input (google.cloud.aiplatform_v1beta1.types.FulfillmentInput):
            Input for fulfillment metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        summarization_quality_input (google.cloud.aiplatform_v1beta1.types.SummarizationQualityInput):
            Input for summarization quality metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        pairwise_summarization_quality_input (google.cloud.aiplatform_v1beta1.types.PairwiseSummarizationQualityInput):
            Input for pairwise summarization quality
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        summarization_helpfulness_input (google.cloud.aiplatform_v1beta1.types.SummarizationHelpfulnessInput):
            Input for summarization helpfulness metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        summarization_verbosity_input (google.cloud.aiplatform_v1beta1.types.SummarizationVerbosityInput):
            Input for summarization verbosity metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        question_answering_quality_input (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringQualityInput):
            Input for question answering quality metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        pairwise_question_answering_quality_input (google.cloud.aiplatform_v1beta1.types.PairwiseQuestionAnsweringQualityInput):
            Input for pairwise question answering quality
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        question_answering_relevance_input (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringRelevanceInput):
            Input for question answering relevance
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        question_answering_helpfulness_input (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringHelpfulnessInput):
            Input for question answering helpfulness
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        question_answering_correctness_input (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringCorrectnessInput):
            Input for question answering correctness
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        tool_call_valid_input (google.cloud.aiplatform_v1beta1.types.ToolCallValidInput):
            Tool call metric instances.
            Input for tool call valid metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        tool_name_match_input (google.cloud.aiplatform_v1beta1.types.ToolNameMatchInput):
            Input for tool name match metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        tool_parameter_key_match_input (google.cloud.aiplatform_v1beta1.types.ToolParameterKeyMatchInput):
            Input for tool parameter key match metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        tool_parameter_kv_match_input (google.cloud.aiplatform_v1beta1.types.ToolParameterKVMatchInput):
            Input for tool parameter key value match
            metric.

            This field is a member of `oneof`_ ``metric_inputs``.
        location (str):
            Required. The resource name of the Location to evaluate the
            instances. Format:
            ``projects/{project}/locations/{location}``
    """

    exact_match_input: "ExactMatchInput" = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="metric_inputs",
        message="ExactMatchInput",
    )
    bleu_input: "BleuInput" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="metric_inputs",
        message="BleuInput",
    )
    rouge_input: "RougeInput" = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="metric_inputs",
        message="RougeInput",
    )
    fluency_input: "FluencyInput" = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="metric_inputs",
        message="FluencyInput",
    )
    coherence_input: "CoherenceInput" = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="metric_inputs",
        message="CoherenceInput",
    )
    safety_input: "SafetyInput" = proto.Field(
        proto.MESSAGE,
        number=8,
        oneof="metric_inputs",
        message="SafetyInput",
    )
    groundedness_input: "GroundednessInput" = proto.Field(
        proto.MESSAGE,
        number=9,
        oneof="metric_inputs",
        message="GroundednessInput",
    )
    fulfillment_input: "FulfillmentInput" = proto.Field(
        proto.MESSAGE,
        number=12,
        oneof="metric_inputs",
        message="FulfillmentInput",
    )
    summarization_quality_input: "SummarizationQualityInput" = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="metric_inputs",
        message="SummarizationQualityInput",
    )
    pairwise_summarization_quality_input: "PairwiseSummarizationQualityInput" = (
        proto.Field(
            proto.MESSAGE,
            number=23,
            oneof="metric_inputs",
            message="PairwiseSummarizationQualityInput",
        )
    )
    summarization_helpfulness_input: "SummarizationHelpfulnessInput" = proto.Field(
        proto.MESSAGE,
        number=14,
        oneof="metric_inputs",
        message="SummarizationHelpfulnessInput",
    )
    summarization_verbosity_input: "SummarizationVerbosityInput" = proto.Field(
        proto.MESSAGE,
        number=15,
        oneof="metric_inputs",
        message="SummarizationVerbosityInput",
    )
    question_answering_quality_input: "QuestionAnsweringQualityInput" = proto.Field(
        proto.MESSAGE,
        number=10,
        oneof="metric_inputs",
        message="QuestionAnsweringQualityInput",
    )
    pairwise_question_answering_quality_input: "PairwiseQuestionAnsweringQualityInput" = proto.Field(
        proto.MESSAGE,
        number=24,
        oneof="metric_inputs",
        message="PairwiseQuestionAnsweringQualityInput",
    )
    question_answering_relevance_input: "QuestionAnsweringRelevanceInput" = proto.Field(
        proto.MESSAGE,
        number=16,
        oneof="metric_inputs",
        message="QuestionAnsweringRelevanceInput",
    )
    question_answering_helpfulness_input: "QuestionAnsweringHelpfulnessInput" = (
        proto.Field(
            proto.MESSAGE,
            number=17,
            oneof="metric_inputs",
            message="QuestionAnsweringHelpfulnessInput",
        )
    )
    question_answering_correctness_input: "QuestionAnsweringCorrectnessInput" = (
        proto.Field(
            proto.MESSAGE,
            number=18,
            oneof="metric_inputs",
            message="QuestionAnsweringCorrectnessInput",
        )
    )
    tool_call_valid_input: "ToolCallValidInput" = proto.Field(
        proto.MESSAGE,
        number=19,
        oneof="metric_inputs",
        message="ToolCallValidInput",
    )
    tool_name_match_input: "ToolNameMatchInput" = proto.Field(
        proto.MESSAGE,
        number=20,
        oneof="metric_inputs",
        message="ToolNameMatchInput",
    )
    tool_parameter_key_match_input: "ToolParameterKeyMatchInput" = proto.Field(
        proto.MESSAGE,
        number=21,
        oneof="metric_inputs",
        message="ToolParameterKeyMatchInput",
    )
    tool_parameter_kv_match_input: "ToolParameterKVMatchInput" = proto.Field(
        proto.MESSAGE,
        number=22,
        oneof="metric_inputs",
        message="ToolParameterKVMatchInput",
    )
    location: str = proto.Field(
        proto.STRING,
        number=1,
    )


class EvaluateInstancesResponse(proto.Message):
    r"""Response message for EvaluationService.EvaluateInstances.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        exact_match_results (google.cloud.aiplatform_v1beta1.types.ExactMatchResults):
            Auto metric evaluation results.
            Results for exact match metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        bleu_results (google.cloud.aiplatform_v1beta1.types.BleuResults):
            Results for bleu metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        rouge_results (google.cloud.aiplatform_v1beta1.types.RougeResults):
            Results for rouge metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        fluency_result (google.cloud.aiplatform_v1beta1.types.FluencyResult):
            LLM-based metric evaluation result.
            General text generation metrics, applicable to
            other categories. Result for fluency metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        coherence_result (google.cloud.aiplatform_v1beta1.types.CoherenceResult):
            Result for coherence metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        safety_result (google.cloud.aiplatform_v1beta1.types.SafetyResult):
            Result for safety metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        groundedness_result (google.cloud.aiplatform_v1beta1.types.GroundednessResult):
            Result for groundedness metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        fulfillment_result (google.cloud.aiplatform_v1beta1.types.FulfillmentResult):
            Result for fulfillment metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        summarization_quality_result (google.cloud.aiplatform_v1beta1.types.SummarizationQualityResult):
            Summarization only metrics.
            Result for summarization quality metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        pairwise_summarization_quality_result (google.cloud.aiplatform_v1beta1.types.PairwiseSummarizationQualityResult):
            Result for pairwise summarization quality
            metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        summarization_helpfulness_result (google.cloud.aiplatform_v1beta1.types.SummarizationHelpfulnessResult):
            Result for summarization helpfulness metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        summarization_verbosity_result (google.cloud.aiplatform_v1beta1.types.SummarizationVerbosityResult):
            Result for summarization verbosity metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        question_answering_quality_result (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringQualityResult):
            Question answering only metrics.
            Result for question answering quality metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        pairwise_question_answering_quality_result (google.cloud.aiplatform_v1beta1.types.PairwiseQuestionAnsweringQualityResult):
            Result for pairwise question answering
            quality metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        question_answering_relevance_result (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringRelevanceResult):
            Result for question answering relevance
            metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        question_answering_helpfulness_result (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringHelpfulnessResult):
            Result for question answering helpfulness
            metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        question_answering_correctness_result (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringCorrectnessResult):
            Result for question answering correctness
            metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        tool_call_valid_results (google.cloud.aiplatform_v1beta1.types.ToolCallValidResults):
            Tool call metrics.
            Results for tool call valid metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        tool_name_match_results (google.cloud.aiplatform_v1beta1.types.ToolNameMatchResults):
            Results for tool name match metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        tool_parameter_key_match_results (google.cloud.aiplatform_v1beta1.types.ToolParameterKeyMatchResults):
            Results for tool parameter key match  metric.

            This field is a member of `oneof`_ ``evaluation_results``.
        tool_parameter_kv_match_results (google.cloud.aiplatform_v1beta1.types.ToolParameterKVMatchResults):
            Results for tool parameter key value match
            metric.

            This field is a member of `oneof`_ ``evaluation_results``.
    """

    exact_match_results: "ExactMatchResults" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="evaluation_results",
        message="ExactMatchResults",
    )
    bleu_results: "BleuResults" = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="evaluation_results",
        message="BleuResults",
    )
    rouge_results: "RougeResults" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="evaluation_results",
        message="RougeResults",
    )
    fluency_result: "FluencyResult" = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="evaluation_results",
        message="FluencyResult",
    )
    coherence_result: "CoherenceResult" = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="evaluation_results",
        message="CoherenceResult",
    )
    safety_result: "SafetyResult" = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="evaluation_results",
        message="SafetyResult",
    )
    groundedness_result: "GroundednessResult" = proto.Field(
        proto.MESSAGE,
        number=8,
        oneof="evaluation_results",
        message="GroundednessResult",
    )
    fulfillment_result: "FulfillmentResult" = proto.Field(
        proto.MESSAGE,
        number=11,
        oneof="evaluation_results",
        message="FulfillmentResult",
    )
    summarization_quality_result: "SummarizationQualityResult" = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="evaluation_results",
        message="SummarizationQualityResult",
    )
    pairwise_summarization_quality_result: "PairwiseSummarizationQualityResult" = (
        proto.Field(
            proto.MESSAGE,
            number=22,
            oneof="evaluation_results",
            message="PairwiseSummarizationQualityResult",
        )
    )
    summarization_helpfulness_result: "SummarizationHelpfulnessResult" = proto.Field(
        proto.MESSAGE,
        number=13,
        oneof="evaluation_results",
        message="SummarizationHelpfulnessResult",
    )
    summarization_verbosity_result: "SummarizationVerbosityResult" = proto.Field(
        proto.MESSAGE,
        number=14,
        oneof="evaluation_results",
        message="SummarizationVerbosityResult",
    )
    question_answering_quality_result: "QuestionAnsweringQualityResult" = proto.Field(
        proto.MESSAGE,
        number=9,
        oneof="evaluation_results",
        message="QuestionAnsweringQualityResult",
    )
    pairwise_question_answering_quality_result: "PairwiseQuestionAnsweringQualityResult" = proto.Field(
        proto.MESSAGE,
        number=23,
        oneof="evaluation_results",
        message="PairwiseQuestionAnsweringQualityResult",
    )
    question_answering_relevance_result: "QuestionAnsweringRelevanceResult" = (
        proto.Field(
            proto.MESSAGE,
            number=15,
            oneof="evaluation_results",
            message="QuestionAnsweringRelevanceResult",
        )
    )
    question_answering_helpfulness_result: "QuestionAnsweringHelpfulnessResult" = (
        proto.Field(
            proto.MESSAGE,
            number=16,
            oneof="evaluation_results",
            message="QuestionAnsweringHelpfulnessResult",
        )
    )
    question_answering_correctness_result: "QuestionAnsweringCorrectnessResult" = (
        proto.Field(
            proto.MESSAGE,
            number=17,
            oneof="evaluation_results",
            message="QuestionAnsweringCorrectnessResult",
        )
    )
    tool_call_valid_results: "ToolCallValidResults" = proto.Field(
        proto.MESSAGE,
        number=18,
        oneof="evaluation_results",
        message="ToolCallValidResults",
    )
    tool_name_match_results: "ToolNameMatchResults" = proto.Field(
        proto.MESSAGE,
        number=19,
        oneof="evaluation_results",
        message="ToolNameMatchResults",
    )
    tool_parameter_key_match_results: "ToolParameterKeyMatchResults" = proto.Field(
        proto.MESSAGE,
        number=20,
        oneof="evaluation_results",
        message="ToolParameterKeyMatchResults",
    )
    tool_parameter_kv_match_results: "ToolParameterKVMatchResults" = proto.Field(
        proto.MESSAGE,
        number=21,
        oneof="evaluation_results",
        message="ToolParameterKVMatchResults",
    )


class ExactMatchInput(proto.Message):
    r"""Input for exact match metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.ExactMatchSpec):
            Required. Spec for exact match metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.ExactMatchInstance]):
            Required. Repeated exact match instances.
    """

    metric_spec: "ExactMatchSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ExactMatchSpec",
    )
    instances: MutableSequence["ExactMatchInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="ExactMatchInstance",
    )


class ExactMatchInstance(proto.Message):
    r"""Spec for exact match instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class ExactMatchSpec(proto.Message):
    r"""Spec for exact match metric - returns 1 if prediction and
    reference exactly matches, otherwise 0.

    """


class ExactMatchResults(proto.Message):
    r"""Results for exact match metric.

    Attributes:
        exact_match_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.ExactMatchMetricValue]):
            Output only. Exact match metric values.
    """

    exact_match_metric_values: MutableSequence[
        "ExactMatchMetricValue"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="ExactMatchMetricValue",
    )


class ExactMatchMetricValue(proto.Message):
    r"""Exact match metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Exact match score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class BleuInput(proto.Message):
    r"""Input for bleu metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.BleuSpec):
            Required. Spec for bleu score metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.BleuInstance]):
            Required. Repeated bleu instances.
    """

    metric_spec: "BleuSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="BleuSpec",
    )
    instances: MutableSequence["BleuInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="BleuInstance",
    )


class BleuInstance(proto.Message):
    r"""Spec for bleu instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class BleuSpec(proto.Message):
    r"""Spec for bleu score metric - calculates the precision of
    n-grams in the prediction as compared to reference - returns a
    score ranging between 0 to 1.

    """


class BleuResults(proto.Message):
    r"""Results for bleu metric.

    Attributes:
        bleu_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.BleuMetricValue]):
            Output only. Bleu metric values.
    """

    bleu_metric_values: MutableSequence["BleuMetricValue"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="BleuMetricValue",
    )


class BleuMetricValue(proto.Message):
    r"""Bleu metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Bleu score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class RougeInput(proto.Message):
    r"""Input for rouge metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.RougeSpec):
            Required. Spec for rouge score metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.RougeInstance]):
            Required. Repeated rouge instances.
    """

    metric_spec: "RougeSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="RougeSpec",
    )
    instances: MutableSequence["RougeInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="RougeInstance",
    )


class RougeInstance(proto.Message):
    r"""Spec for rouge instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class RougeSpec(proto.Message):
    r"""Spec for rouge score metric - calculates the recall of
    n-grams in prediction as compared to reference - returns a score
    ranging between 0 and 1.

    Attributes:
        rouge_type (str):
            Optional. Supported rouge types are rougen[1-9], rougeL and
            rougeLsum.
        use_stemmer (bool):
            Optional. Whether to use stemmer to compute
            rouge score.
        split_summaries (bool):
            Optional. Whether to split summaries while
            using rougeLsum.
    """

    rouge_type: str = proto.Field(
        proto.STRING,
        number=1,
    )
    use_stemmer: bool = proto.Field(
        proto.BOOL,
        number=2,
    )
    split_summaries: bool = proto.Field(
        proto.BOOL,
        number=3,
    )


class RougeResults(proto.Message):
    r"""Results for rouge metric.

    Attributes:
        rouge_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.RougeMetricValue]):
            Output only. Rouge metric values.
    """

    rouge_metric_values: MutableSequence["RougeMetricValue"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="RougeMetricValue",
    )


class RougeMetricValue(proto.Message):
    r"""Rouge metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Rouge score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class CoherenceInput(proto.Message):
    r"""Input for coherence metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.CoherenceSpec):
            Required. Spec for coherence score metric.
        instance (google.cloud.aiplatform_v1beta1.types.CoherenceInstance):
            Required. Coherence instance.
    """

    metric_spec: "CoherenceSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="CoherenceSpec",
    )
    instance: "CoherenceInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="CoherenceInstance",
    )


class CoherenceInstance(proto.Message):
    r"""Spec for coherence instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )


class CoherenceSpec(proto.Message):
    r"""Spec for coherence score metric.

    Attributes:
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    version: int = proto.Field(
        proto.INT32,
        number=1,
    )


class CoherenceResult(proto.Message):
    r"""Spec for coherence result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Coherence score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for coherence score.
        confidence (float):
            Output only. Confidence for coherence score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class FluencyInput(proto.Message):
    r"""Input for fluency metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.FluencySpec):
            Required. Spec for fluency score metric.
        instance (google.cloud.aiplatform_v1beta1.types.FluencyInstance):
            Required. Fluency instance.
    """

    metric_spec: "FluencySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="FluencySpec",
    )
    instance: "FluencyInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="FluencyInstance",
    )


class FluencyInstance(proto.Message):
    r"""Spec for fluency instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )


class FluencySpec(proto.Message):
    r"""Spec for fluency score metric.

    Attributes:
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    version: int = proto.Field(
        proto.INT32,
        number=1,
    )


class FluencyResult(proto.Message):
    r"""Spec for fluency result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Fluency score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for fluency score.
        confidence (float):
            Output only. Confidence for fluency score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class SafetyInput(proto.Message):
    r"""Input for safety metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.SafetySpec):
            Required. Spec for safety metric.
        instance (google.cloud.aiplatform_v1beta1.types.SafetyInstance):
            Required. Safety instance.
    """

    metric_spec: "SafetySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="SafetySpec",
    )
    instance: "SafetyInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="SafetyInstance",
    )


class SafetyInstance(proto.Message):
    r"""Spec for safety instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )


class SafetySpec(proto.Message):
    r"""Spec for safety metric.

    Attributes:
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    version: int = proto.Field(
        proto.INT32,
        number=1,
    )


class SafetyResult(proto.Message):
    r"""Spec for safety result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Safety score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for safety score.
        confidence (float):
            Output only. Confidence for safety score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class GroundednessInput(proto.Message):
    r"""Input for groundedness metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.GroundednessSpec):
            Required. Spec for groundedness metric.
        instance (google.cloud.aiplatform_v1beta1.types.GroundednessInstance):
            Required. Groundedness instance.
    """

    metric_spec: "GroundednessSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="GroundednessSpec",
    )
    instance: "GroundednessInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="GroundednessInstance",
    )


class GroundednessInstance(proto.Message):
    r"""Spec for groundedness instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        context (str):
            Required. Background information provided in
            context used to compare against the prediction.

            This field is a member of `oneof`_ ``_context``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class GroundednessSpec(proto.Message):
    r"""Spec for groundedness metric.

    Attributes:
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    version: int = proto.Field(
        proto.INT32,
        number=1,
    )


class GroundednessResult(proto.Message):
    r"""Spec for groundedness result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Groundedness score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for groundedness
            score.
        confidence (float):
            Output only. Confidence for groundedness
            score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class FulfillmentInput(proto.Message):
    r"""Input for fulfillment metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.FulfillmentSpec):
            Required. Spec for fulfillment score metric.
        instance (google.cloud.aiplatform_v1beta1.types.FulfillmentInstance):
            Required. Fulfillment instance.
    """

    metric_spec: "FulfillmentSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="FulfillmentSpec",
    )
    instance: "FulfillmentInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="FulfillmentInstance",
    )


class FulfillmentInstance(proto.Message):
    r"""Spec for fulfillment instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        instruction (str):
            Required. Inference instruction prompt to
            compare prediction with.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class FulfillmentSpec(proto.Message):
    r"""Spec for fulfillment metric.

    Attributes:
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    version: int = proto.Field(
        proto.INT32,
        number=1,
    )


class FulfillmentResult(proto.Message):
    r"""Spec for fulfillment result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Fulfillment score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for fulfillment
            score.
        confidence (float):
            Output only. Confidence for fulfillment
            score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class SummarizationQualityInput(proto.Message):
    r"""Input for summarization quality metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.SummarizationQualitySpec):
            Required. Spec for summarization quality
            score metric.
        instance (google.cloud.aiplatform_v1beta1.types.SummarizationQualityInstance):
            Required. Summarization quality instance.
    """

    metric_spec: "SummarizationQualitySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="SummarizationQualitySpec",
    )
    instance: "SummarizationQualityInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="SummarizationQualityInstance",
    )


class SummarizationQualityInstance(proto.Message):
    r"""Spec for summarization quality instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Required. Text to be summarized.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. Summarization prompt for LLM.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class SummarizationQualitySpec(proto.Message):
    r"""Spec for summarization quality score metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute summarization quality.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class SummarizationQualityResult(proto.Message):
    r"""Spec for summarization quality result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Summarization Quality score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for summarization
            quality score.
        confidence (float):
            Output only. Confidence for summarization
            quality score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class PairwiseSummarizationQualityInput(proto.Message):
    r"""Input for pairwise summarization quality metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.PairwiseSummarizationQualitySpec):
            Required. Spec for pairwise summarization
            quality score metric.
        instance (google.cloud.aiplatform_v1beta1.types.PairwiseSummarizationQualityInstance):
            Required. Pairwise summarization quality
            instance.
    """

    metric_spec: "PairwiseSummarizationQualitySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="PairwiseSummarizationQualitySpec",
    )
    instance: "PairwiseSummarizationQualityInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="PairwiseSummarizationQualityInstance",
    )


class PairwiseSummarizationQualityInstance(proto.Message):
    r"""Spec for pairwise summarization quality instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the candidate model.

            This field is a member of `oneof`_ ``_prediction``.
        baseline_prediction (str):
            Required. Output of the baseline model.

            This field is a member of `oneof`_ ``_baseline_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Required. Text to be summarized.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. Summarization prompt for LLM.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    baseline_prediction: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=5,
        optional=True,
    )


class PairwiseSummarizationQualitySpec(proto.Message):
    r"""Spec for pairwise summarization quality score metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute pairwise summarization quality.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class PairwiseSummarizationQualityResult(proto.Message):
    r"""Spec for pairwise summarization quality result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        pairwise_choice (google.cloud.aiplatform_v1beta1.types.PairwiseChoice):
            Output only. Pairwise summarization
            prediction choice.
        explanation (str):
            Output only. Explanation for summarization
            quality score.
        confidence (float):
            Output only. Confidence for summarization
            quality score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    pairwise_choice: "PairwiseChoice" = proto.Field(
        proto.ENUM,
        number=1,
        enum="PairwiseChoice",
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class SummarizationHelpfulnessInput(proto.Message):
    r"""Input for summarization helpfulness metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.SummarizationHelpfulnessSpec):
            Required. Spec for summarization helpfulness
            score metric.
        instance (google.cloud.aiplatform_v1beta1.types.SummarizationHelpfulnessInstance):
            Required. Summarization helpfulness instance.
    """

    metric_spec: "SummarizationHelpfulnessSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="SummarizationHelpfulnessSpec",
    )
    instance: "SummarizationHelpfulnessInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="SummarizationHelpfulnessInstance",
    )


class SummarizationHelpfulnessInstance(proto.Message):
    r"""Spec for summarization helpfulness instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Required. Text to be summarized.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Optional. Summarization prompt for LLM.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class SummarizationHelpfulnessSpec(proto.Message):
    r"""Spec for summarization helpfulness score metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute summarization helpfulness.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class SummarizationHelpfulnessResult(proto.Message):
    r"""Spec for summarization helpfulness result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Summarization Helpfulness score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for summarization
            helpfulness score.
        confidence (float):
            Output only. Confidence for summarization
            helpfulness score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class SummarizationVerbosityInput(proto.Message):
    r"""Input for summarization verbosity metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.SummarizationVerbositySpec):
            Required. Spec for summarization verbosity
            score metric.
        instance (google.cloud.aiplatform_v1beta1.types.SummarizationVerbosityInstance):
            Required. Summarization verbosity instance.
    """

    metric_spec: "SummarizationVerbositySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="SummarizationVerbositySpec",
    )
    instance: "SummarizationVerbosityInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="SummarizationVerbosityInstance",
    )


class SummarizationVerbosityInstance(proto.Message):
    r"""Spec for summarization verbosity instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Required. Text to be summarized.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Optional. Summarization prompt for LLM.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class SummarizationVerbositySpec(proto.Message):
    r"""Spec for summarization verbosity score metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute summarization verbosity.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class SummarizationVerbosityResult(proto.Message):
    r"""Spec for summarization verbosity result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Summarization Verbosity score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for summarization
            verbosity score.
        confidence (float):
            Output only. Confidence for summarization
            verbosity score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class QuestionAnsweringQualityInput(proto.Message):
    r"""Input for question answering quality metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringQualitySpec):
            Required. Spec for question answering quality
            score metric.
        instance (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringQualityInstance):
            Required. Question answering quality
            instance.
    """

    metric_spec: "QuestionAnsweringQualitySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="QuestionAnsweringQualitySpec",
    )
    instance: "QuestionAnsweringQualityInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="QuestionAnsweringQualityInstance",
    )


class QuestionAnsweringQualityInstance(proto.Message):
    r"""Spec for question answering quality instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Required. Text to answer the question.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. Question Answering prompt for LLM.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class QuestionAnsweringQualitySpec(proto.Message):
    r"""Spec for question answering quality score metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute question answering quality.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class QuestionAnsweringQualityResult(proto.Message):
    r"""Spec for question answering quality result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Question Answering Quality
            score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for question
            answering quality score.
        confidence (float):
            Output only. Confidence for question
            answering quality score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class PairwiseQuestionAnsweringQualityInput(proto.Message):
    r"""Input for pairwise question answering quality metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.PairwiseQuestionAnsweringQualitySpec):
            Required. Spec for pairwise question
            answering quality score metric.
        instance (google.cloud.aiplatform_v1beta1.types.PairwiseQuestionAnsweringQualityInstance):
            Required. Pairwise question answering quality
            instance.
    """

    metric_spec: "PairwiseQuestionAnsweringQualitySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="PairwiseQuestionAnsweringQualitySpec",
    )
    instance: "PairwiseQuestionAnsweringQualityInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="PairwiseQuestionAnsweringQualityInstance",
    )


class PairwiseQuestionAnsweringQualityInstance(proto.Message):
    r"""Spec for pairwise question answering quality instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the candidate model.

            This field is a member of `oneof`_ ``_prediction``.
        baseline_prediction (str):
            Required. Output of the baseline model.

            This field is a member of `oneof`_ ``_baseline_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Required. Text to answer the question.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. Question Answering prompt for LLM.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    baseline_prediction: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=5,
        optional=True,
    )


class PairwiseQuestionAnsweringQualitySpec(proto.Message):
    r"""Spec for pairwise question answering quality score metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute question answering quality.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class PairwiseQuestionAnsweringQualityResult(proto.Message):
    r"""Spec for pairwise question answering quality result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        pairwise_choice (google.cloud.aiplatform_v1beta1.types.PairwiseChoice):
            Output only. Pairwise question answering
            prediction choice.
        explanation (str):
            Output only. Explanation for question
            answering quality score.
        confidence (float):
            Output only. Confidence for question
            answering quality score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    pairwise_choice: "PairwiseChoice" = proto.Field(
        proto.ENUM,
        number=1,
        enum="PairwiseChoice",
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class QuestionAnsweringRelevanceInput(proto.Message):
    r"""Input for question answering relevance metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringRelevanceSpec):
            Required. Spec for question answering
            relevance score metric.
        instance (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringRelevanceInstance):
            Required. Question answering relevance
            instance.
    """

    metric_spec: "QuestionAnsweringRelevanceSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="QuestionAnsweringRelevanceSpec",
    )
    instance: "QuestionAnsweringRelevanceInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="QuestionAnsweringRelevanceInstance",
    )


class QuestionAnsweringRelevanceInstance(proto.Message):
    r"""Spec for question answering relevance instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Optional. Text provided as context to answer
            the question.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. The question asked and other
            instruction in the inference prompt.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class QuestionAnsweringRelevanceSpec(proto.Message):
    r"""Spec for question answering relevance metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute question answering relevance.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class QuestionAnsweringRelevanceResult(proto.Message):
    r"""Spec for question answering relevance result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Question Answering Relevance
            score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for question
            answering relevance score.
        confidence (float):
            Output only. Confidence for question
            answering relevance score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class QuestionAnsweringHelpfulnessInput(proto.Message):
    r"""Input for question answering helpfulness metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringHelpfulnessSpec):
            Required. Spec for question answering
            helpfulness score metric.
        instance (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringHelpfulnessInstance):
            Required. Question answering helpfulness
            instance.
    """

    metric_spec: "QuestionAnsweringHelpfulnessSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="QuestionAnsweringHelpfulnessSpec",
    )
    instance: "QuestionAnsweringHelpfulnessInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="QuestionAnsweringHelpfulnessInstance",
    )


class QuestionAnsweringHelpfulnessInstance(proto.Message):
    r"""Spec for question answering helpfulness instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Optional. Text provided as context to answer
            the question.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. The question asked and other
            instruction in the inference prompt.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class QuestionAnsweringHelpfulnessSpec(proto.Message):
    r"""Spec for question answering helpfulness metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute question answering helpfulness.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class QuestionAnsweringHelpfulnessResult(proto.Message):
    r"""Spec for question answering helpfulness result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Question Answering Helpfulness
            score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for question
            answering helpfulness score.
        confidence (float):
            Output only. Confidence for question
            answering helpfulness score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class QuestionAnsweringCorrectnessInput(proto.Message):
    r"""Input for question answering correctness metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringCorrectnessSpec):
            Required. Spec for question answering
            correctness score metric.
        instance (google.cloud.aiplatform_v1beta1.types.QuestionAnsweringCorrectnessInstance):
            Required. Question answering correctness
            instance.
    """

    metric_spec: "QuestionAnsweringCorrectnessSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="QuestionAnsweringCorrectnessSpec",
    )
    instance: "QuestionAnsweringCorrectnessInstance" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="QuestionAnsweringCorrectnessInstance",
    )


class QuestionAnsweringCorrectnessInstance(proto.Message):
    r"""Spec for question answering correctness instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Optional. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
        context (str):
            Optional. Text provided as context to answer
            the question.

            This field is a member of `oneof`_ ``_context``.
        instruction (str):
            Required. The question asked and other
            instruction in the inference prompt.

            This field is a member of `oneof`_ ``_instruction``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )
    context: str = proto.Field(
        proto.STRING,
        number=3,
        optional=True,
    )
    instruction: str = proto.Field(
        proto.STRING,
        number=4,
        optional=True,
    )


class QuestionAnsweringCorrectnessSpec(proto.Message):
    r"""Spec for question answering correctness metric.

    Attributes:
        use_reference (bool):
            Optional. Whether to use instance.reference
            to compute question answering correctness.
        version (int):
            Optional. Which version to use for
            evaluation.
    """

    use_reference: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    version: int = proto.Field(
        proto.INT32,
        number=2,
    )


class QuestionAnsweringCorrectnessResult(proto.Message):
    r"""Spec for question answering correctness result.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Question Answering Correctness
            score.

            This field is a member of `oneof`_ ``_score``.
        explanation (str):
            Output only. Explanation for question
            answering correctness score.
        confidence (float):
            Output only. Confidence for question
            answering correctness score.

            This field is a member of `oneof`_ ``_confidence``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )
    explanation: str = proto.Field(
        proto.STRING,
        number=2,
    )
    confidence: float = proto.Field(
        proto.FLOAT,
        number=3,
        optional=True,
    )


class ToolCallValidInput(proto.Message):
    r"""Input for tool call valid metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.ToolCallValidSpec):
            Required. Spec for tool call valid metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolCallValidInstance]):
            Required. Repeated tool call valid instances.
    """

    metric_spec: "ToolCallValidSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ToolCallValidSpec",
    )
    instances: MutableSequence["ToolCallValidInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="ToolCallValidInstance",
    )


class ToolCallValidSpec(proto.Message):
    r"""Spec for tool call valid metric."""


class ToolCallValidInstance(proto.Message):
    r"""Spec for tool call valid instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class ToolCallValidResults(proto.Message):
    r"""Results for tool call valid metric.

    Attributes:
        tool_call_valid_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolCallValidMetricValue]):
            Output only. Tool call valid metric values.
    """

    tool_call_valid_metric_values: MutableSequence[
        "ToolCallValidMetricValue"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="ToolCallValidMetricValue",
    )


class ToolCallValidMetricValue(proto.Message):
    r"""Tool call valid metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Tool call valid score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class ToolNameMatchInput(proto.Message):
    r"""Input for tool name match metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.ToolNameMatchSpec):
            Required. Spec for tool name match metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolNameMatchInstance]):
            Required. Repeated tool name match instances.
    """

    metric_spec: "ToolNameMatchSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ToolNameMatchSpec",
    )
    instances: MutableSequence["ToolNameMatchInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="ToolNameMatchInstance",
    )


class ToolNameMatchSpec(proto.Message):
    r"""Spec for tool name match metric."""


class ToolNameMatchInstance(proto.Message):
    r"""Spec for tool name match instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class ToolNameMatchResults(proto.Message):
    r"""Results for tool name match metric.

    Attributes:
        tool_name_match_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolNameMatchMetricValue]):
            Output only. Tool name match metric values.
    """

    tool_name_match_metric_values: MutableSequence[
        "ToolNameMatchMetricValue"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="ToolNameMatchMetricValue",
    )


class ToolNameMatchMetricValue(proto.Message):
    r"""Tool name match metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Tool name match score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class ToolParameterKeyMatchInput(proto.Message):
    r"""Input for tool parameter key match metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.ToolParameterKeyMatchSpec):
            Required. Spec for tool parameter key match
            metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolParameterKeyMatchInstance]):
            Required. Repeated tool parameter key match
            instances.
    """

    metric_spec: "ToolParameterKeyMatchSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ToolParameterKeyMatchSpec",
    )
    instances: MutableSequence["ToolParameterKeyMatchInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="ToolParameterKeyMatchInstance",
    )


class ToolParameterKeyMatchSpec(proto.Message):
    r"""Spec for tool parameter key match metric."""


class ToolParameterKeyMatchInstance(proto.Message):
    r"""Spec for tool parameter key match instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class ToolParameterKeyMatchResults(proto.Message):
    r"""Results for tool parameter key match metric.

    Attributes:
        tool_parameter_key_match_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolParameterKeyMatchMetricValue]):
            Output only. Tool parameter key match metric
            values.
    """

    tool_parameter_key_match_metric_values: MutableSequence[
        "ToolParameterKeyMatchMetricValue"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="ToolParameterKeyMatchMetricValue",
    )


class ToolParameterKeyMatchMetricValue(proto.Message):
    r"""Tool parameter key match metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Tool parameter key match score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


class ToolParameterKVMatchInput(proto.Message):
    r"""Input for tool parameter key value match metric.

    Attributes:
        metric_spec (google.cloud.aiplatform_v1beta1.types.ToolParameterKVMatchSpec):
            Required. Spec for tool parameter key value
            match metric.
        instances (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolParameterKVMatchInstance]):
            Required. Repeated tool parameter key value
            match instances.
    """

    metric_spec: "ToolParameterKVMatchSpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ToolParameterKVMatchSpec",
    )
    instances: MutableSequence["ToolParameterKVMatchInstance"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="ToolParameterKVMatchInstance",
    )


class ToolParameterKVMatchSpec(proto.Message):
    r"""Spec for tool parameter key value match metric.

    Attributes:
        use_strict_string_match (bool):
            Optional. Whether to use STRCIT string match
            on parameter values.
    """

    use_strict_string_match: bool = proto.Field(
        proto.BOOL,
        number=1,
    )


class ToolParameterKVMatchInstance(proto.Message):
    r"""Spec for tool parameter key value match instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        prediction (str):
            Required. Output of the evaluated model.

            This field is a member of `oneof`_ ``_prediction``.
        reference (str):
            Required. Ground truth used to compare
            against the prediction.

            This field is a member of `oneof`_ ``_reference``.
    """

    prediction: str = proto.Field(
        proto.STRING,
        number=1,
        optional=True,
    )
    reference: str = proto.Field(
        proto.STRING,
        number=2,
        optional=True,
    )


class ToolParameterKVMatchResults(proto.Message):
    r"""Results for tool parameter key value match metric.

    Attributes:
        tool_parameter_kv_match_metric_values (MutableSequence[google.cloud.aiplatform_v1beta1.types.ToolParameterKVMatchMetricValue]):
            Output only. Tool parameter key value match
            metric values.
    """

    tool_parameter_kv_match_metric_values: MutableSequence[
        "ToolParameterKVMatchMetricValue"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="ToolParameterKVMatchMetricValue",
    )


class ToolParameterKVMatchMetricValue(proto.Message):
    r"""Tool parameter key value match metric value for an instance.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        score (float):
            Output only. Tool parameter key value match
            score.

            This field is a member of `oneof`_ ``_score``.
    """

    score: float = proto.Field(
        proto.FLOAT,
        number=1,
        optional=True,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
