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
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ModelEvaluationSlice",
    },
)


class ModelEvaluationSlice(proto.Message):
    r"""A collection of metrics calculated by comparing Model's
    predictions on a slice of the test data against ground truth
    annotations.

    Attributes:
        name (str):
            Output only. The resource name of the
            ModelEvaluationSlice.
        slice_ (google.cloud.aiplatform_v1beta1.types.ModelEvaluationSlice.Slice):
            Output only. The slice of the test data that
            is used to evaluate the Model.
        metrics_schema_uri (str):
            Output only. Points to a YAML file stored on Google Cloud
            Storage describing the
            [metrics][google.cloud.aiplatform.v1beta1.ModelEvaluationSlice.metrics]
            of this ModelEvaluationSlice. The schema is defined as an
            OpenAPI 3.0.2 `Schema
            Object <https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#schemaObject>`__.
        metrics (google.protobuf.struct_pb2.Value):
            Output only. Sliced evaluation metrics of the Model. The
            schema of the metrics is stored in
            [metrics_schema_uri][google.cloud.aiplatform.v1beta1.ModelEvaluationSlice.metrics_schema_uri]
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            ModelEvaluationSlice was created.
        model_explanation (google.cloud.aiplatform_v1beta1.types.ModelExplanation):
            Output only. Aggregated explanation metrics
            for the Model's prediction output over the data
            this ModelEvaluation uses. This field is
            populated only if the Model is evaluated with
            explanations, and only for tabular Models.
    """

    class Slice(proto.Message):
        r"""Definition of a slice.

        Attributes:
            dimension (str):
                Output only. The dimension of the slice. Well-known
                dimensions are:

                -  ``annotationSpec``: This slice is on the test data that
                   has either ground truth or prediction with
                   [AnnotationSpec.display_name][google.cloud.aiplatform.v1beta1.AnnotationSpec.display_name]
                   equals to
                   [value][google.cloud.aiplatform.v1beta1.ModelEvaluationSlice.Slice.value].
                -  ``slice``: This slice is a user customized slice defined
                   by its SliceSpec.
            value (str):
                Output only. The value of the dimension in
                this slice.
            slice_spec (google.cloud.aiplatform_v1beta1.types.ModelEvaluationSlice.Slice.SliceSpec):
                Output only. Specification for how the data
                was sliced.
        """

        class SliceSpec(proto.Message):
            r"""Specification for how the data should be sliced.

            Attributes:
                configs (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.ModelEvaluationSlice.Slice.SliceSpec.SliceConfig]):
                    Mapping configuration for this SliceSpec.
                    The key is the name of the feature.
                    By default, the key will be prefixed by
                    "instance" as a dictionary prefix for Vertex
                    Batch Predictions output format.
            """

            class SliceConfig(proto.Message):
                r"""Specification message containing the config for this SliceSpec. When
                ``kind`` is selected as ``value`` and/or ``range``, only a single
                slice will be computed. When ``all_values`` is present, a separate
                slice will be computed for each possible label/value for the
                corresponding key in ``config``. Examples, with feature zip_code
                with values 12345, 23334, 88888 and feature country with values
                "US", "Canada", "Mexico" in the dataset:

                Example 1:

                ::

                    {
                      "zip_code": { "value": { "float_value": 12345.0 } }
                    }

                A single slice for any data with zip_code 12345 in the dataset.

                Example 2:

                ::

                    {
                      "zip_code": { "range": { "low": 12345, "high": 20000 } }
                    }

                A single slice containing data where the zip_codes between 12345 and
                20000 For this example, data with the zip_code of 12345 will be in
                this slice.

                Example 3:

                ::

                    {
                      "zip_code": { "range": { "low": 10000, "high": 20000 } },
                      "country": { "value": { "string_value": "US" } }
                    }

                A single slice containing data where the zip_codes between 10000 and
                20000 has the country "US". For this example, data with the zip_code
                of 12345 and country "US" will be in this slice.

                Example 4:

                ::

                    { "country": {"all_values": { "value": true } } }

                Three slices are computed, one for each unique country in the
                dataset.

                Example 5:

                ::

                    {
                      "country": { "all_values": { "value": true } },
                      "zip_code": { "value": { "float_value": 12345.0 } }
                    }

                Three slices are computed, one for each unique country in the
                dataset where the zip_code is also 12345. For this example, data
                with zip_code 12345 and country "US" will be in one slice, zip_code
                12345 and country "Canada" in another slice, and zip_code 12345 and
                country "Mexico" in another slice, totaling 3 slices.

                This message has `oneof`_ fields (mutually exclusive fields).
                For each oneof, at most one member field can be set at the same time.
                Setting any member of the oneof automatically clears all other
                members.

                .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

                Attributes:
                    value (google.cloud.aiplatform_v1beta1.types.ModelEvaluationSlice.Slice.SliceSpec.Value):
                        A unique specific value for a given feature. Example:
                        ``{ "value": { "string_value": "12345" } }``

                        This field is a member of `oneof`_ ``kind``.
                    range_ (google.cloud.aiplatform_v1beta1.types.ModelEvaluationSlice.Slice.SliceSpec.Range):
                        A range of values for a numerical feature. Example:
                        ``{"range":{"low":10000.0,"high":50000.0}}`` will capture
                        12345 and 23334 in the slice.

                        This field is a member of `oneof`_ ``kind``.
                    all_values (google.protobuf.wrappers_pb2.BoolValue):
                        If all_values is set to true, then all possible labels of
                        the keyed feature will have another slice computed. Example:
                        ``{"all_values":{"value":true}}``

                        This field is a member of `oneof`_ ``kind``.
                """

                value: "ModelEvaluationSlice.Slice.SliceSpec.Value" = proto.Field(
                    proto.MESSAGE,
                    number=1,
                    oneof="kind",
                    message="ModelEvaluationSlice.Slice.SliceSpec.Value",
                )
                range_: "ModelEvaluationSlice.Slice.SliceSpec.Range" = proto.Field(
                    proto.MESSAGE,
                    number=2,
                    oneof="kind",
                    message="ModelEvaluationSlice.Slice.SliceSpec.Range",
                )
                all_values: wrappers_pb2.BoolValue = proto.Field(
                    proto.MESSAGE,
                    number=3,
                    oneof="kind",
                    message=wrappers_pb2.BoolValue,
                )

            class Range(proto.Message):
                r"""A range of values for slice(s). ``low`` is inclusive, ``high`` is
                exclusive.

                Attributes:
                    low (float):
                        Inclusive low value for the range.
                    high (float):
                        Exclusive high value for the range.
                """

                low: float = proto.Field(
                    proto.FLOAT,
                    number=1,
                )
                high: float = proto.Field(
                    proto.FLOAT,
                    number=2,
                )

            class Value(proto.Message):
                r"""Single value that supports strings and floats.

                This message has `oneof`_ fields (mutually exclusive fields).
                For each oneof, at most one member field can be set at the same time.
                Setting any member of the oneof automatically clears all other
                members.

                .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

                Attributes:
                    string_value (str):
                        String type.

                        This field is a member of `oneof`_ ``kind``.
                    float_value (float):
                        Float type.

                        This field is a member of `oneof`_ ``kind``.
                """

                string_value: str = proto.Field(
                    proto.STRING,
                    number=1,
                    oneof="kind",
                )
                float_value: float = proto.Field(
                    proto.FLOAT,
                    number=2,
                    oneof="kind",
                )

            configs: MutableMapping[
                str, "ModelEvaluationSlice.Slice.SliceSpec.SliceConfig"
            ] = proto.MapField(
                proto.STRING,
                proto.MESSAGE,
                number=1,
                message="ModelEvaluationSlice.Slice.SliceSpec.SliceConfig",
            )

        dimension: str = proto.Field(
            proto.STRING,
            number=1,
        )
        value: str = proto.Field(
            proto.STRING,
            number=2,
        )
        slice_spec: "ModelEvaluationSlice.Slice.SliceSpec" = proto.Field(
            proto.MESSAGE,
            number=3,
            message="ModelEvaluationSlice.Slice.SliceSpec",
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    slice_: Slice = proto.Field(
        proto.MESSAGE,
        number=2,
        message=Slice,
    )
    metrics_schema_uri: str = proto.Field(
        proto.STRING,
        number=3,
    )
    metrics: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=4,
        message=struct_pb2.Value,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    model_explanation: explanation.ModelExplanation = proto.Field(
        proto.MESSAGE,
        number=6,
        message=explanation.ModelExplanation,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
