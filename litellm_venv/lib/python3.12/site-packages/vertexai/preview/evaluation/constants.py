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
"""Constants for evaluation."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class Metric:
    """Namespace for Metrics."""

    # Automatic Metrics.
    EXACT_MATCH = "exact_match"
    BLEU = "bleu"
    ROUGE_1 = "rouge_1"
    ROUGE_2 = "rouge_2"
    ROUGE_L = "rouge_l"
    ROUGE_L_SUM = "rouge_l_sum"
    TOOL_CALL_VALID = "tool_call_valid"
    TOOL_NAME_MATCH = "tool_name_match"
    TOOL_PARAMETER_KEY_MATCH = "tool_parameter_key_match"
    TOOL_PARAMETER_KV_MATCH = "tool_parameter_kv_match"
    # Model-based Pointwise Metrics.
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    SAFETY = "safety"
    GROUNDEDNESS = "groundedness"
    FULFILLMENT = "fulfillment"
    RESPONSE_RECALL = "response_recall"
    SUMMARIZATION_QUALITY = "summarization_quality"
    SUMMARIZATION_HELPFULNESS = "summarization_helpfulness"
    SUMMARIZATION_VERBOSITY = "summarization_verbosity"
    QUESTION_ANSWERING_QUALITY = "question_answering_quality"
    QUESTION_ANSWERING_RELEVANCE = "question_answering_relevance"
    QUESTION_ANSWERING_HELPFULNESS = "question_answering_helpfulness"
    QUESTION_ANSWERING_CORRECTNESS = "question_answering_correctness"
    RAG_CONTEXT_RECALL = "rag_context_recall"
    # Side-by-side(SxS) Pairwise Metrics.
    PAIRWISE_SUMMARIZATION_QUALITY = "pairwise_summarization_quality"
    PAIRWISE_QUESTION_ANSWERING_QUALITY = "pairwise_question_answering_quality"

    AUTOMATIC_METRIC_LIST = (
        EXACT_MATCH,
        BLEU,
        ROUGE_1,
        ROUGE_2,
        ROUGE_L,
        ROUGE_L_SUM,
        TOOL_CALL_VALID,
        TOOL_NAME_MATCH,
        TOOL_PARAMETER_KEY_MATCH,
        TOOL_PARAMETER_KV_MATCH,
    )
    MODEL_BASED_METRIC_LIST = (
        COHERENCE,
        FLUENCY,
        SAFETY,
        GROUNDEDNESS,
        FULFILLMENT,
        RESPONSE_RECALL,
        SUMMARIZATION_QUALITY,
        SUMMARIZATION_HELPFULNESS,
        SUMMARIZATION_VERBOSITY,
        QUESTION_ANSWERING_QUALITY,
        QUESTION_ANSWERING_RELEVANCE,
        QUESTION_ANSWERING_HELPFULNESS,
        QUESTION_ANSWERING_CORRECTNESS,
        RAG_CONTEXT_RECALL,
    )
    PAIRWISE_METRIC_LIST = (
        PAIRWISE_SUMMARIZATION_QUALITY,
        PAIRWISE_QUESTION_ANSWERING_QUALITY,
    )


@dataclasses.dataclass(frozen=True)
class MetricResult:
    ROW_COUNT_KEY = "row_count"
    SCORE_KEY = "score"
    EXPLANATION_KEY = "explanation"
    CONFIDENCE_KEY = "confidence"
    PAIRWISE_CHOICE_KEY = "pairwise_choice"

    # Automatic Metrics.
    EXACT_MATCH_RESULTS = "exact_match_results"
    BLEU_RESULTS = "bleu_results"
    ROUGE_RESULTS = "rouge_results"
    TOOL_CALL_VALID_RESULTS = "tool_call_valid_results"
    TOOL_NAME_MATCH_RESULTS = "tool_name_match_results"
    TOOL_PARAMETER_KEY_MATCH_RESULTS = "tool_parameter_key_match_results"
    TOOL_PARAMETER_KV_MATCH_RESULTS = "tool_parameter_kv_match_results"
    # Model-based Pointwise Metrics.
    COHERENCE_RESULT = "coherence_result"
    FLUENCY_RESULT = "fluency_result"
    SAFETY_RESULT = "safety_result"
    GROUNDEDNESS_RESULT = "groundedness_result"
    FULFILLMENT_RESULT = "fulfillment_result"
    RESPONSE_RECALL_RESULT = "response_recall_result"
    SUMMARIZATION_QUALITY_RESULT = "summarization_quality_result"
    SUMMARIZATION_HELPFULNESS_RESULT = "summarization_helpfulness_result"
    SUMMARIZATION_VERBOSITY_RESULT = "summarization_verbosity_result"
    QUESTION_ANSWERING_QUALITY_RESULT = "question_answering_quality_result"
    QUESTION_ANSWERING_RELEVANCE_RESULT = "question_answering_relevance_result"
    QUESTION_ANSWERING_HELPFULNESS_RESULT = "question_answering_helpfulness_result"
    QUESTION_ANSWERING_CORRECTNESS_RESULT = "question_answering_correctness_result"
    RAG_CONTEXT_RECALL_RESULT = "rag_context_recall_result"
    # Side-by-side(SxS) Pairwise Metrics.
    PAIRWISE_SUMMARIZATION_QUALITY_RESULT = "pairwise_summarization_quality_result"
    PAIRWISE_QUESTION_ANSWERING_QUALITY_RESULT = (
        "pairwise_question_answering_quality_result"
    )

    AUTOMATIC_METRIC_RESULTS_LIST = (
        EXACT_MATCH_RESULTS,
        BLEU_RESULTS,
        ROUGE_RESULTS,
        TOOL_CALL_VALID_RESULTS,
        TOOL_NAME_MATCH_RESULTS,
        TOOL_PARAMETER_KEY_MATCH_RESULTS,
        TOOL_PARAMETER_KV_MATCH_RESULTS,
    )
    MODEL_BASED_METRIC_RESULT_LIST = (
        COHERENCE_RESULT,
        FLUENCY_RESULT,
        SAFETY_RESULT,
        GROUNDEDNESS_RESULT,
        FULFILLMENT_RESULT,
        RESPONSE_RECALL_RESULT,
        SUMMARIZATION_QUALITY_RESULT,
        SUMMARIZATION_HELPFULNESS_RESULT,
        SUMMARIZATION_VERBOSITY_RESULT,
        QUESTION_ANSWERING_QUALITY_RESULT,
        QUESTION_ANSWERING_RELEVANCE_RESULT,
        QUESTION_ANSWERING_HELPFULNESS_RESULT,
        QUESTION_ANSWERING_CORRECTNESS_RESULT,
        RAG_CONTEXT_RECALL_RESULT,
    )
    PAIRWISE_METRIC_RESULT_LIST = (
        PAIRWISE_SUMMARIZATION_QUALITY_RESULT,
        PAIRWISE_QUESTION_ANSWERING_QUALITY_RESULT,
    )


@dataclasses.dataclass(frozen=True)
class MetricBundle:
    """Namespace for MetricBundle."""

    TEXT_GENERATION_SIMILARITY = "text_generation_similarity"
    TEXT_GENERATION_QUALITY = "text_generation_quality"
    TOOL_CALL_QUALITY = "tool_call_quality"
    TEXT_GENERATION_INSTRUCTION_FOLLOWING = "text_generation_instruction_following"
    TEXT_GENERATION_SAFETY = "text_generation_safety"
    TEXT_GENERATION_FACTUALITY = "text_generation_factuality"
    SUMMARIZATION_POINTWISE_REFERENCE_FREE = "summarization_pointwise_reference_free"
    QA_POINTWISE_REFERENCE_FREE = "qa_pointwise_reference_free"
    QA_POINTWISE_REFERENCE_BASED = "qa_pointwise_reference_based"


@dataclasses.dataclass(frozen=True)
class Dataset:
    COMPLETED_PROMPT_COLUMN = "completed_prompt"
    MODEL_RESPONSE_COLUMN = "response"
    BASELINE_MODEL_RESPONSE_COLUMN = "baseline_model_response"
    CONTEXT_COLUMN = "context"
    REFERENCE_COLUMN = "reference"
    CONTENT_COLUMN = "content"
    INSTRUCTION_COLUMN = "instruction"
