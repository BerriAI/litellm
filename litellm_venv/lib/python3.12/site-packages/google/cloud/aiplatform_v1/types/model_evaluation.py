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

from google.cloud.aiplatform_v1.types import explanation
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
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
            [metrics][google.cloud.aiplatform.v1.ModelEvaluation.metrics]
            of this ModelEvaluation. The schema is defined as an OpenAPI
            3.0.2 `Schema
            Object <https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#schemaObject>`__.
        metrics (google.protobuf.struct_pb2.Value):
            Evaluation metrics of the Model. The schema of the metrics
            is stored in
            [metrics_schema_uri][google.cloud.aiplatform.v1.ModelEvaluation.metrics_schema_uri]
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            ModelEvaluation was created.
        slice_dimensions (MutableSequence[str]):
            All possible
            [dimensions][google.cloud.aiplatform.v1.ModelEvaluationSlice.Slice.dimension]
            of ModelEvaluationSlices. The dimensions can be used as the
            filter of the
            [ModelService.ListModelEvaluationSlices][google.cloud.aiplatform.v1.ModelService.ListModelEvaluationSlices]
            request, in the form of ``slice.dimension = <dimension>``.
        data_item_schema_uri (str):
            Points to a YAML file stored on Google Cloud Storage
            describing [EvaluatedDataItemView.data_item_payload][] and
            [EvaluatedAnnotation.data_item_payload][google.cloud.aiplatform.v1.EvaluatedAnnotation.data_item_payload].
            The schema is defined as an OpenAPI 3.0.2 `Schema
            Object <https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#schemaObject>`__.

            This field is not populated if there are neither
            EvaluatedDataItemViews nor EvaluatedAnnotations under this
            ModelEvaluation.
        annotation_schema_uri (str):
            Points to a YAML file stored on Google Cloud Storage
            describing [EvaluatedDataItemView.predictions][],
            [EvaluatedDataItemView.ground_truths][],
            [EvaluatedAnnotation.predictions][google.cloud.aiplatform.v1.EvaluatedAnnotation.predictions],
            and
            [EvaluatedAnnotation.ground_truths][google.cloud.aiplatform.v1.EvaluatedAnnotation.ground_truths].
            The schema is defined as an OpenAPI 3.0.2 `Schema
            Object <https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#schemaObject>`__.

            This field is not populated if there are neither
            EvaluatedDataItemViews nor EvaluatedAnnotations under this
            ModelEvaluation.
        model_explanation (google.cloud.aiplatform_v1.types.ModelExplanation):
            Aggregated explanation metrics for the
            Model's prediction output over the data this
            ModelEvaluation uses. This field is populated
            only if the Model is evaluated with
            explanations, and only for AutoML tabular
            Models.
        explanation_specs (MutableSequence[google.cloud.aiplatform_v1.types.ModelEvaluation.ModelEvaluationExplanationSpec]):
            Describes the values of
            [ExplanationSpec][google.cloud.aiplatform.v1.ExplanationSpec]
            that are used for explaining the predicted values on the
            evaluated data.
        metadata (google.protobuf.struct_pb2.Value):
            The metadata of the ModelEvaluation. For the ModelEvaluation
            uploaded from Managed Pipeline, metadata contains a
            structured value with keys of "pipeline_job_id",
            "evaluation_dataset_type", "evaluation_dataset_path",
            "row_based_metrics_path".
    """

    class ModelEvaluationExplanationSpec(proto.Message):
        r"""

        Attributes:
            explanation_type (str):
                Explanation type.

                For AutoML Image Classification models, possible values are:

                -  ``image-integrated-gradients``
                -  ``image-xrai``
            explanation_spec (google.cloud.aiplatform_v1.types.ExplanationSpec):
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
    data_item_schema_uri: str = proto.Field(
        proto.STRING,
        number=6,
    )
    annotation_schema_uri: str = proto.Field(
        proto.STRING,
        number=7,
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


__all__ = tuple(sorted(__protobuf__.manifest))
