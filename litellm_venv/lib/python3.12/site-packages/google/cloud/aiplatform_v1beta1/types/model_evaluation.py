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

from google.cloud.aiplatform_v1beta1.types import explanation
from google.cloud.aiplatform_v1beta1.types import model_evaluation_slice
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ModelEvaluation",
    },
)


class ModelEvaluation(proto.Message):
    r"""A collection of metrics calculated by comparing Model's
    predictions on all of the test data against annotations from the
    test data.

    Attributes:
        name (str):
            Output only. The resource name of the
            ModelEvaluation.
        display_name (str):
            The display name of the ModelEvaluation.
        metrics_schema_uri (str):
            Points to a YAML file stored on Google Cloud Storage
            describing the
            [metrics][google.cloud.aiplatform.v1beta1.ModelEvaluation.metrics]
            of this ModelEvaluation. The schema is defined as an OpenAPI
            3.0.2 `Schema
            Object <https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#schemaObject>`__.
        metrics (google.protobuf.struct_pb2.Value):
            Evaluation metrics of the Model. The schema of the metrics
            is stored in
            [metrics_schema_uri][google.cloud.aiplatform.v1beta1.ModelEvaluation.metrics_schema_uri]
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            ModelEvaluation was created.
        slice_dimensions (MutableSequence[str]):
            All possible
            [dimensions][google.cloud.aiplatform.v1beta1.ModelEvaluationSlice.Slice.dimension]
            of ModelEvaluationSlices. The dimensions can be used as the
            filter of the
            [ModelService.ListModelEvaluationSlices][google.cloud.aiplatform.v1beta1.ModelService.ListModelEvaluationSlices]
            request, in the form of ``slice.dimension = <dimension>``.
        model_explanation (google.cloud.aiplatform_v1beta1.types.ModelExplanation):
            Aggregated explanation metrics for the
            Model's prediction output over the data this
            ModelEvaluation uses. This field is populated
            only if the Model is evaluated with
            explanations, and only for AutoML tabular
            Models.
        explanation_specs (MutableSequence[google.cloud.aiplatform_v1beta1.types.ModelEvaluation.ModelEvaluationExplanationSpec]):
            Describes the values of
            [ExplanationSpec][google.cloud.aiplatform.v1beta1.ExplanationSpec]
            that are used for explaining the predicted values on the
            evaluated data.
        metadata (google.protobuf.struct_pb2.Value):
            The metadata of the ModelEvaluation. For the ModelEvaluation
            uploaded from Managed Pipeline, metadata contains a
            structured value with keys of "pipeline_job_id",
            "evaluation_dataset_type", "evaluation_dataset_path",
            "row_based_metrics_path".
        bias_configs (google.cloud.aiplatform_v1beta1.types.ModelEvaluation.BiasConfig):
            Specify the configuration for bias detection.
    """

    class ModelEvaluationExplanationSpec(proto.Message):
        r"""

        Attributes:
            explanation_type (str):
                Explanation type.

                For AutoML Image Classification models, possible values are:

                -  ``image-integrated-gradients``
                -  ``image-xrai``
            explanation_spec (google.cloud.aiplatform_v1beta1.types.ExplanationSpec):
                Explanation spec details.
        """

        explanation_type: str = proto.Field(
            proto.STRING,
            number=1,
        )
        explanation_spec: explanation.ExplanationSpec = proto.Field(
            proto.MESSAGE,
            number=2,
            message=explanation.ExplanationSpec,
        )

    class BiasConfig(proto.Message):
        r"""Configuration for bias detection.

        Attributes:
            bias_slices (google.cloud.aiplatform_v1beta1.types.ModelEvaluationSlice.Slice.SliceSpec):
                Specification for how the data should be sliced for bias. It
                contains a list of slices, with limitation of two slices.
                The first slice of data will be the slice_a. The second
                slice in the list (slice_b) will be compared against the
                first slice. If only a single slice is provided, then
                slice_a will be compared against "not slice_a". Below are
                examples with feature "education" with value "low",
                "medium", "high" in the dataset:

                Example 1:

                ::

                    bias_slices = [{'education': 'low'}]

                A single slice provided. In this case, slice_a is the
                collection of data with 'education' equals 'low', and
                slice_b is the collection of data with 'education' equals
                'medium' or 'high'.

                Example 2:

                ::

                    bias_slices = [{'education': 'low'},
                                   {'education': 'high'}]

                Two slices provided. In this case, slice_a is the collection
                of data with 'education' equals 'low', and slice_b is the
                collection of data with 'education' equals 'high'.
            labels (MutableSequence[str]):
                Positive labels selection on the target
                field.
        """

        bias_slices: model_evaluation_slice.ModelEvaluationSlice.Slice.SliceSpec = (
            proto.Field(
                proto.MESSAGE,
                number=1,
                message=model_evaluation_slice.ModelEvaluationSlice.Slice.SliceSpec,
            )
        )
        labels: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=2,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=10,
    )
    metrics_schema_uri: str = proto.Field(
        proto.STRING,
        number=2,
    )
    metrics: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=3,
        message=struct_pb2.Value,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    slice_dimensions: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=5,
    )
    model_explanation: explanation.ModelExplanation = proto.Field(
        proto.MESSAGE,
        number=8,
        message=explanation.ModelExplanation,
    )
    explanation_specs: MutableSequence[
        ModelEvaluationExplanationSpec
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=9,
        message=ModelEvaluationExplanationSpec,
    )
    metadata: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=11,
        message=struct_pb2.Value,
    )
    bias_configs: BiasConfig = proto.Field(
        proto.MESSAGE,
        number=12,
        message=BiasConfig,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
