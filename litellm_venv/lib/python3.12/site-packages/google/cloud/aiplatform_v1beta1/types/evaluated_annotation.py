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

from google.cloud.aiplatform_v1beta1.types import explanation as gca_explanation
from google.protobuf import struct_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "EvaluatedAnnotation",
        "EvaluatedAnnotationExplanation",
        "ErrorAnalysisAnnotation",
    },
)


class EvaluatedAnnotation(proto.Message):
    r"""True positive, false positive, or false negative.

    EvaluatedAnnotation is only available under ModelEvaluationSlice
    with slice of ``annotationSpec`` dimension.

    Attributes:
        type_ (google.cloud.aiplatform_v1beta1.types.EvaluatedAnnotation.EvaluatedAnnotationType):
            Output only. Type of the EvaluatedAnnotation.
        predictions (MutableSequence[google.protobuf.struct_pb2.Value]):
            Output only. The model predicted annotations.

            For true positive, there is one and only one prediction,
            which matches the only one ground truth annotation in
            [ground_truths][google.cloud.aiplatform.v1beta1.EvaluatedAnnotation.ground_truths].

            For false positive, there is one and only one prediction,
            which doesn't match any ground truth annotation of the
            corresponding
            [data_item_view_id][EvaluatedAnnotation.data_item_view_id].

            For false negative, there are zero or more predictions which
            are similar to the only ground truth annotation in
            [ground_truths][google.cloud.aiplatform.v1beta1.EvaluatedAnnotation.ground_truths]
            but not enough for a match.

            The schema of the prediction is stored in
            [ModelEvaluation.annotation_schema_uri][]
        ground_truths (MutableSequence[google.protobuf.struct_pb2.Value]):
            Output only. The ground truth Annotations, i.e. the
            Annotations that exist in the test data the Model is
            evaluated on.

            For true positive, there is one and only one ground truth
            annotation, which matches the only prediction in
            [predictions][google.cloud.aiplatform.v1beta1.EvaluatedAnnotation.predictions].

            For false positive, there are zero or more ground truth
            annotations that are similar to the only prediction in
            [predictions][google.cloud.aiplatform.v1beta1.EvaluatedAnnotation.predictions],
            but not enough for a match.

            For false negative, there is one and only one ground truth
            annotation, which doesn't match any predictions created by
            the model.

            The schema of the ground truth is stored in
            [ModelEvaluation.annotation_schema_uri][]
        data_item_payload (google.protobuf.struct_pb2.Value):
            Output only. The data item payload that the
            Model predicted this EvaluatedAnnotation on.
        evaluated_data_item_view_id (str):
            Output only. ID of the EvaluatedDataItemView under the same
            ancestor ModelEvaluation. The EvaluatedDataItemView consists
            of all ground truths and predictions on
            [data_item_payload][google.cloud.aiplatform.v1beta1.EvaluatedAnnotation.data_item_payload].
        explanations (MutableSequence[google.cloud.aiplatform_v1beta1.types.EvaluatedAnnotationExplanation]):
            Explanations of
            [predictions][google.cloud.aiplatform.v1beta1.EvaluatedAnnotation.predictions].
            Each element of the explanations indicates the explanation
            for one explanation Method.

            The attributions list in the
            [EvaluatedAnnotationExplanation.explanation][google.cloud.aiplatform.v1beta1.EvaluatedAnnotationExplanation.explanation]
            object corresponds to the
            [predictions][google.cloud.aiplatform.v1beta1.EvaluatedAnnotation.predictions]
            list. For example, the second element in the attributions
            list explains the second element in the predictions list.
        error_analysis_annotations (MutableSequence[google.cloud.aiplatform_v1beta1.types.ErrorAnalysisAnnotation]):
            Annotations of model error analysis results.
    """

    class EvaluatedAnnotationType(proto.Enum):
        r"""Describes the type of the EvaluatedAnnotation. The type is
        determined

        Values:
            EVALUATED_ANNOTATION_TYPE_UNSPECIFIED (0):
                Invalid value.
            TRUE_POSITIVE (1):
                The EvaluatedAnnotation is a true positive.
                It has a prediction created by the Model and a
                ground truth Annotation which the prediction
                matches.
            FALSE_POSITIVE (2):
                The EvaluatedAnnotation is false positive. It
                has a prediction created by the Model which does
                not match any ground truth annotation.
            FALSE_NEGATIVE (3):
                The EvaluatedAnnotation is false negative. It
                has a ground truth annotation which is not
                matched by any of the model created predictions.
        """
        EVALUATED_ANNOTATION_TYPE_UNSPECIFIED = 0
        TRUE_POSITIVE = 1
        FALSE_POSITIVE = 2
        FALSE_NEGATIVE = 3

    type_: EvaluatedAnnotationType = proto.Field(
        proto.ENUM,
        number=1,
        enum=EvaluatedAnnotationType,
    )
    predictions: MutableSequence[struct_pb2.Value] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message=struct_pb2.Value,
    )
    ground_truths: MutableSequence[struct_pb2.Value] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message=struct_pb2.Value,
    )
    data_item_payload: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=5,
        message=struct_pb2.Value,
    )
    evaluated_data_item_view_id: str = proto.Field(
        proto.STRING,
        number=6,
    )
    explanations: MutableSequence[
        "EvaluatedAnnotationExplanation"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=8,
        message="EvaluatedAnnotationExplanation",
    )
    error_analysis_annotations: MutableSequence[
        "ErrorAnalysisAnnotation"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=9,
        message="ErrorAnalysisAnnotation",
    )


class EvaluatedAnnotationExplanation(proto.Message):
    r"""Explanation result of the prediction produced by the Model.

    Attributes:
        explanation_type (str):
            Explanation type.

            For AutoML Image Classification models, possible values are:

            -  ``image-integrated-gradients``
            -  ``image-xrai``
        explanation (google.cloud.aiplatform_v1beta1.types.Explanation):
            Explanation attribution response details.
    """

    explanation_type: str = proto.Field(
        proto.STRING,
        number=1,
    )
    explanation: gca_explanation.Explanation = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_explanation.Explanation,
    )


class ErrorAnalysisAnnotation(proto.Message):
    r"""Model error analysis for each annotation.

    Attributes:
        attributed_items (MutableSequence[google.cloud.aiplatform_v1beta1.types.ErrorAnalysisAnnotation.AttributedItem]):
            Attributed items for a given annotation,
            typically representing neighbors from the
            training sets constrained by the query type.
        query_type (google.cloud.aiplatform_v1beta1.types.ErrorAnalysisAnnotation.QueryType):
            The query type used for finding the
            attributed items.
        outlier_score (float):
            The outlier score of this annotated item.
            Usually defined as the min of all distances from
            attributed items.
        outlier_threshold (float):
            The threshold used to determine if this
            annotation is an outlier or not.
    """

    class QueryType(proto.Enum):
        r"""The query type used for finding the attributed items.

        Values:
            QUERY_TYPE_UNSPECIFIED (0):
                Unspecified query type for model error
                analysis.
            ALL_SIMILAR (1):
                Query similar samples across all classes in
                the dataset.
            SAME_CLASS_SIMILAR (2):
                Query similar samples from the same class of
                the input sample.
            SAME_CLASS_DISSIMILAR (3):
                Query dissimilar samples from the same class
                of the input sample.
        """
        QUERY_TYPE_UNSPECIFIED = 0
        ALL_SIMILAR = 1
        SAME_CLASS_SIMILAR = 2
        SAME_CLASS_DISSIMILAR = 3

    class AttributedItem(proto.Message):
        r"""Attributed items for a given annotation, typically
        representing neighbors from the training sets constrained by the
        query type.

        Attributes:
            annotation_resource_name (str):
                The unique ID for each annotation. Used by FE
                to allocate the annotation in DB.
            distance (float):
                The distance of this item to the annotation.
        """

        annotation_resource_name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        distance: float = proto.Field(
            proto.DOUBLE,
            number=2,
        )

    attributed_items: MutableSequence[AttributedItem] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=AttributedItem,
    )
    query_type: QueryType = proto.Field(
        proto.ENUM,
        number=2,
        enum=QueryType,
    )
    outlier_score: float = proto.Field(
        proto.DOUBLE,
        number=3,
    )
    outlier_threshold: float = proto.Field(
        proto.DOUBLE,
        number=4,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
