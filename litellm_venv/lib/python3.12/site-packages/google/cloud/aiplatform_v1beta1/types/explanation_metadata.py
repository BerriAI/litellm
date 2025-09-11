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

from google.protobuf import struct_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "ExplanationMetadata",
    },
)


class ExplanationMetadata(proto.Message):
    r"""Metadata describing the Model's input and output for
    explanation.

    Attributes:
        inputs (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.ExplanationMetadata.InputMetadata]):
            Required. Map from feature names to feature input metadata.
            Keys are the name of the features. Values are the
            specification of the feature.

            An empty InputMetadata is valid. It describes a text feature
            which has the name specified as the key in
            [ExplanationMetadata.inputs][google.cloud.aiplatform.v1beta1.ExplanationMetadata.inputs].
            The baseline of the empty feature is chosen by Vertex AI.

            For Vertex AI-provided Tensorflow images, the key can be any
            friendly name of the feature. Once specified,
            [featureAttributions][google.cloud.aiplatform.v1beta1.Attribution.feature_attributions]
            are keyed by this key (if not grouped with another feature).

            For custom images, the key must match with the key in
            [instance][google.cloud.aiplatform.v1beta1.ExplainRequest.instances].
        outputs (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.ExplanationMetadata.OutputMetadata]):
            Required. Map from output names to output
            metadata.
            For Vertex AI-provided Tensorflow images, keys
            can be any user defined string that consists of
            any UTF-8 characters.

            For custom images, keys are the name of the
            output field in the prediction to be explained.

            Currently only one key is allowed.
        feature_attributions_schema_uri (str):
            Points to a YAML file stored on Google Cloud Storage
            describing the format of the [feature
            attributions][google.cloud.aiplatform.v1beta1.Attribution.feature_attributions].
            The schema is defined as an OpenAPI 3.0.2 `Schema
            Object <https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#schemaObject>`__.
            AutoML tabular Models always have this field populated by
            Vertex AI. Note: The URI given on output may be different,
            including the URI scheme, than the one given on input. The
            output URI will point to a location where the user only has
            a read access.
        latent_space_source (str):
            Name of the source to generate embeddings for
            example based explanations.
    """

    class InputMetadata(proto.Message):
        r"""Metadata of the input of a feature.

        Fields other than
        [InputMetadata.input_baselines][google.cloud.aiplatform.v1beta1.ExplanationMetadata.InputMetadata.input_baselines]
        are applicable only for Models that are using Vertex AI-provided
        images for Tensorflow.

        Attributes:
            input_baselines (MutableSequence[google.protobuf.struct_pb2.Value]):
                Baseline inputs for this feature.

                If no baseline is specified, Vertex AI chooses the baseline
                for this feature. If multiple baselines are specified,
                Vertex AI returns the average attributions across them in
                [Attribution.feature_attributions][google.cloud.aiplatform.v1beta1.Attribution.feature_attributions].

                For Vertex AI-provided Tensorflow images (both 1.x and 2.x),
                the shape of each baseline must match the shape of the input
                tensor. If a scalar is provided, we broadcast to the same
                shape as the input tensor.

                For custom images, the element of the baselines must be in
                the same format as the feature's input in the
                [instance][google.cloud.aiplatform.v1beta1.ExplainRequest.instances][].
                The schema of any single instance may be specified via
                Endpoint's DeployedModels'
                [Model's][google.cloud.aiplatform.v1beta1.DeployedModel.model]
                [PredictSchemata's][google.cloud.aiplatform.v1beta1.Model.predict_schemata]
                [instance_schema_uri][google.cloud.aiplatform.v1beta1.PredictSchemata.instance_schema_uri].
            input_tensor_name (str):
                Name of the input tensor for this feature.
                Required and is only applicable to Vertex
                AI-provided images for Tensorflow.
            encoding (google.cloud.aiplatform_v1beta1.types.ExplanationMetadata.InputMetadata.Encoding):
                Defines how the feature is encoded into the
                input tensor. Defaults to IDENTITY.
            modality (str):
                Modality of the feature. Valid values are:
                numeric, image. Defaults to numeric.
            feature_value_domain (google.cloud.aiplatform_v1beta1.types.ExplanationMetadata.InputMetadata.FeatureValueDomain):
                The domain details of the input feature
                value. Like min/max, original mean or standard
                deviation if normalized.
            indices_tensor_name (str):
                Specifies the index of the values of the input tensor.
                Required when the input tensor is a sparse representation.
                Refer to Tensorflow documentation for more details:
                https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor.
            dense_shape_tensor_name (str):
                Specifies the shape of the values of the input if the input
                is a sparse representation. Refer to Tensorflow
                documentation for more details:
                https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor.
            index_feature_mapping (MutableSequence[str]):
                A list of feature names for each index in the input tensor.
                Required when the input
                [InputMetadata.encoding][google.cloud.aiplatform.v1beta1.ExplanationMetadata.InputMetadata.encoding]
                is BAG_OF_FEATURES, BAG_OF_FEATURES_SPARSE, INDICATOR.
            encoded_tensor_name (str):
                Encoded tensor is a transformation of the input tensor. Must
                be provided if choosing [Integrated Gradients
                attribution][google.cloud.aiplatform.v1beta1.ExplanationParameters.integrated_gradients_attribution]
                or [XRAI
                attribution][google.cloud.aiplatform.v1beta1.ExplanationParameters.xrai_attribution]
                and the input tensor is not differentiable.

                An encoded tensor is generated if the input tensor is
                encoded by a lookup table.
            encoded_baselines (MutableSequence[google.protobuf.struct_pb2.Value]):
                A list of baselines for the encoded tensor.

                The shape of each baseline should match the
                shape of the encoded tensor. If a scalar is
                provided, Vertex AI broadcasts to the same shape
                as the encoded tensor.
            visualization (google.cloud.aiplatform_v1beta1.types.ExplanationMetadata.InputMetadata.Visualization):
                Visualization configurations for image
                explanation.
            group_name (str):
                Name of the group that the input belongs to. Features with
                the same group name will be treated as one feature when
                computing attributions. Features grouped together can have
                different shapes in value. If provided, there will be one
                single attribution generated in
                [Attribution.feature_attributions][google.cloud.aiplatform.v1beta1.Attribution.feature_attributions],
                keyed by the group name.
        """

        class Encoding(proto.Enum):
            r"""Defines how a feature is encoded. Defaults to IDENTITY.

            Values:
                ENCODING_UNSPECIFIED (0):
                    Default value. This is the same as IDENTITY.
                IDENTITY (1):
                    The tensor represents one feature.
                BAG_OF_FEATURES (2):
                    The tensor represents a bag of features where each index
                    maps to a feature.
                    [InputMetadata.index_feature_mapping][google.cloud.aiplatform.v1beta1.ExplanationMetadata.InputMetadata.index_feature_mapping]
                    must be provided for this encoding. For example:

                    ::

                       input = [27, 6.0, 150]
                       index_feature_mapping = ["age", "height", "weight"]
                BAG_OF_FEATURES_SPARSE (3):
                    The tensor represents a bag of features where each index
                    maps to a feature. Zero values in the tensor indicates
                    feature being non-existent.
                    [InputMetadata.index_feature_mapping][google.cloud.aiplatform.v1beta1.ExplanationMetadata.InputMetadata.index_feature_mapping]
                    must be provided for this encoding. For example:

                    ::

                       input = [2, 0, 5, 0, 1]
                       index_feature_mapping = ["a", "b", "c", "d", "e"]
                INDICATOR (4):
                    The tensor is a list of binaries representing whether a
                    feature exists or not (1 indicates existence).
                    [InputMetadata.index_feature_mapping][google.cloud.aiplatform.v1beta1.ExplanationMetadata.InputMetadata.index_feature_mapping]
                    must be provided for this encoding. For example:

                    ::

                       input = [1, 0, 1, 0, 1]
                       index_feature_mapping = ["a", "b", "c", "d", "e"]
                COMBINED_EMBEDDING (5):
                    The tensor is encoded into a 1-dimensional array represented
                    by an encoded tensor.
                    [InputMetadata.encoded_tensor_name][google.cloud.aiplatform.v1beta1.ExplanationMetadata.InputMetadata.encoded_tensor_name]
                    must be provided for this encoding. For example:

                    ::

                       input = ["This", "is", "a", "test", "."]
                       encoded = [0.1, 0.2, 0.3, 0.4, 0.5]
                CONCAT_EMBEDDING (6):
                    Select this encoding when the input tensor is encoded into a
                    2-dimensional array represented by an encoded tensor.
                    [InputMetadata.encoded_tensor_name][google.cloud.aiplatform.v1beta1.ExplanationMetadata.InputMetadata.encoded_tensor_name]
                    must be provided for this encoding. The first dimension of
                    the encoded tensor's shape is the same as the input tensor's
                    shape. For example:

                    ::

                       input = ["This", "is", "a", "test", "."]
                       encoded = [[0.1, 0.2, 0.3, 0.4, 0.5],
                                  [0.2, 0.1, 0.4, 0.3, 0.5],
                                  [0.5, 0.1, 0.3, 0.5, 0.4],
                                  [0.5, 0.3, 0.1, 0.2, 0.4],
                                  [0.4, 0.3, 0.2, 0.5, 0.1]]
            """
            ENCODING_UNSPECIFIED = 0
            IDENTITY = 1
            BAG_OF_FEATURES = 2
            BAG_OF_FEATURES_SPARSE = 3
            INDICATOR = 4
            COMBINED_EMBEDDING = 5
            CONCAT_EMBEDDING = 6

        class FeatureValueDomain(proto.Message):
            r"""Domain details of the input feature value. Provides numeric
            information about the feature, such as its range (min, max). If the
            feature has been pre-processed, for example with z-scoring, then it
            provides information about how to recover the original feature. For
            example, if the input feature is an image and it has been
            pre-processed to obtain 0-mean and stddev = 1 values, then
            original_mean, and original_stddev refer to the mean and stddev of
            the original feature (e.g. image tensor) from which input feature
            (with mean = 0 and stddev = 1) was obtained.

            Attributes:
                min_value (float):
                    The minimum permissible value for this
                    feature.
                max_value (float):
                    The maximum permissible value for this
                    feature.
                original_mean (float):
                    If this input feature has been normalized to a mean value of
                    0, the original_mean specifies the mean value of the domain
                    prior to normalization.
                original_stddev (float):
                    If this input feature has been normalized to a standard
                    deviation of 1.0, the original_stddev specifies the standard
                    deviation of the domain prior to normalization.
            """

            min_value: float = proto.Field(
                proto.FLOAT,
                number=1,
            )
            max_value: float = proto.Field(
                proto.FLOAT,
                number=2,
            )
            original_mean: float = proto.Field(
                proto.FLOAT,
                number=3,
            )
            original_stddev: float = proto.Field(
                proto.FLOAT,
                number=4,
            )

        class Visualization(proto.Message):
            r"""Visualization configurations for image explanation.

            Attributes:
                type_ (google.cloud.aiplatform_v1beta1.types.ExplanationMetadata.InputMetadata.Visualization.Type):
                    Type of the image visualization. Only applicable to
                    [Integrated Gradients
                    attribution][google.cloud.aiplatform.v1beta1.ExplanationParameters.integrated_gradients_attribution].
                    OUTLINES shows regions of attribution, while PIXELS shows
                    per-pixel attribution. Defaults to OUTLINES.
                polarity (google.cloud.aiplatform_v1beta1.types.ExplanationMetadata.InputMetadata.Visualization.Polarity):
                    Whether to only highlight pixels with
                    positive contributions, negative or both.
                    Defaults to POSITIVE.
                color_map (google.cloud.aiplatform_v1beta1.types.ExplanationMetadata.InputMetadata.Visualization.ColorMap):
                    The color scheme used for the highlighted areas.

                    Defaults to PINK_GREEN for [Integrated Gradients
                    attribution][google.cloud.aiplatform.v1beta1.ExplanationParameters.integrated_gradients_attribution],
                    which shows positive attributions in green and negative in
                    pink.

                    Defaults to VIRIDIS for [XRAI
                    attribution][google.cloud.aiplatform.v1beta1.ExplanationParameters.xrai_attribution],
                    which highlights the most influential regions in yellow and
                    the least influential in blue.
                clip_percent_upperbound (float):
                    Excludes attributions above the specified percentile from
                    the highlighted areas. Using the clip_percent_upperbound and
                    clip_percent_lowerbound together can be useful for filtering
                    out noise and making it easier to see areas of strong
                    attribution. Defaults to 99.9.
                clip_percent_lowerbound (float):
                    Excludes attributions below the specified
                    percentile, from the highlighted areas. Defaults
                    to 62.
                overlay_type (google.cloud.aiplatform_v1beta1.types.ExplanationMetadata.InputMetadata.Visualization.OverlayType):
                    How the original image is displayed in the
                    visualization. Adjusting the overlay can help
                    increase visual clarity if the original image
                    makes it difficult to view the visualization.
                    Defaults to NONE.
            """

            class Type(proto.Enum):
                r"""Type of the image visualization. Only applicable to [Integrated
                Gradients
                attribution][google.cloud.aiplatform.v1beta1.ExplanationParameters.integrated_gradients_attribution].

                Values:
                    TYPE_UNSPECIFIED (0):
                        Should not be used.
                    PIXELS (1):
                        Shows which pixel contributed to the image
                        prediction.
                    OUTLINES (2):
                        Shows which region contributed to the image
                        prediction by outlining the region.
                """
                TYPE_UNSPECIFIED = 0
                PIXELS = 1
                OUTLINES = 2

            class Polarity(proto.Enum):
                r"""Whether to only highlight pixels with positive contributions,
                negative or both. Defaults to POSITIVE.

                Values:
                    POLARITY_UNSPECIFIED (0):
                        Default value. This is the same as POSITIVE.
                    POSITIVE (1):
                        Highlights the pixels/outlines that were most
                        influential to the model's prediction.
                    NEGATIVE (2):
                        Setting polarity to negative highlights areas
                        that does not lead to the models's current
                        prediction.
                    BOTH (3):
                        Shows both positive and negative
                        attributions.
                """
                POLARITY_UNSPECIFIED = 0
                POSITIVE = 1
                NEGATIVE = 2
                BOTH = 3

            class ColorMap(proto.Enum):
                r"""The color scheme used for highlighting areas.

                Values:
                    COLOR_MAP_UNSPECIFIED (0):
                        Should not be used.
                    PINK_GREEN (1):
                        Positive: green. Negative: pink.
                    VIRIDIS (2):
                        Viridis color map: A perceptually uniform
                        color mapping which is easier to see by those
                        with colorblindness and progresses from yellow
                        to green to blue. Positive: yellow. Negative:
                        blue.
                    RED (3):
                        Positive: red. Negative: red.
                    GREEN (4):
                        Positive: green. Negative: green.
                    RED_GREEN (6):
                        Positive: green. Negative: red.
                    PINK_WHITE_GREEN (5):
                        PiYG palette.
                """
                COLOR_MAP_UNSPECIFIED = 0
                PINK_GREEN = 1
                VIRIDIS = 2
                RED = 3
                GREEN = 4
                RED_GREEN = 6
                PINK_WHITE_GREEN = 5

            class OverlayType(proto.Enum):
                r"""How the original image is displayed in the visualization.

                Values:
                    OVERLAY_TYPE_UNSPECIFIED (0):
                        Default value. This is the same as NONE.
                    NONE (1):
                        No overlay.
                    ORIGINAL (2):
                        The attributions are shown on top of the
                        original image.
                    GRAYSCALE (3):
                        The attributions are shown on top of
                        grayscaled version of the original image.
                    MASK_BLACK (4):
                        The attributions are used as a mask to reveal
                        predictive parts of the image and hide the
                        un-predictive parts.
                """
                OVERLAY_TYPE_UNSPECIFIED = 0
                NONE = 1
                ORIGINAL = 2
                GRAYSCALE = 3
                MASK_BLACK = 4

            type_: "ExplanationMetadata.InputMetadata.Visualization.Type" = proto.Field(
                proto.ENUM,
                number=1,
                enum="ExplanationMetadata.InputMetadata.Visualization.Type",
            )
            polarity: "ExplanationMetadata.InputMetadata.Visualization.Polarity" = (
                proto.Field(
                    proto.ENUM,
                    number=2,
                    enum="ExplanationMetadata.InputMetadata.Visualization.Polarity",
                )
            )
            color_map: "ExplanationMetadata.InputMetadata.Visualization.ColorMap" = (
                proto.Field(
                    proto.ENUM,
                    number=3,
                    enum="ExplanationMetadata.InputMetadata.Visualization.ColorMap",
                )
            )
            clip_percent_upperbound: float = proto.Field(
                proto.FLOAT,
                number=4,
            )
            clip_percent_lowerbound: float = proto.Field(
                proto.FLOAT,
                number=5,
            )
            overlay_type: "ExplanationMetadata.InputMetadata.Visualization.OverlayType" = proto.Field(
                proto.ENUM,
                number=6,
                enum="ExplanationMetadata.InputMetadata.Visualization.OverlayType",
            )

        input_baselines: MutableSequence[struct_pb2.Value] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=struct_pb2.Value,
        )
        input_tensor_name: str = proto.Field(
            proto.STRING,
            number=2,
        )
        encoding: "ExplanationMetadata.InputMetadata.Encoding" = proto.Field(
            proto.ENUM,
            number=3,
            enum="ExplanationMetadata.InputMetadata.Encoding",
        )
        modality: str = proto.Field(
            proto.STRING,
            number=4,
        )
        feature_value_domain: "ExplanationMetadata.InputMetadata.FeatureValueDomain" = (
            proto.Field(
                proto.MESSAGE,
                number=5,
                message="ExplanationMetadata.InputMetadata.FeatureValueDomain",
            )
        )
        indices_tensor_name: str = proto.Field(
            proto.STRING,
            number=6,
        )
        dense_shape_tensor_name: str = proto.Field(
            proto.STRING,
            number=7,
        )
        index_feature_mapping: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=8,
        )
        encoded_tensor_name: str = proto.Field(
            proto.STRING,
            number=9,
        )
        encoded_baselines: MutableSequence[struct_pb2.Value] = proto.RepeatedField(
            proto.MESSAGE,
            number=10,
            message=struct_pb2.Value,
        )
        visualization: "ExplanationMetadata.InputMetadata.Visualization" = proto.Field(
            proto.MESSAGE,
            number=11,
            message="ExplanationMetadata.InputMetadata.Visualization",
        )
        group_name: str = proto.Field(
            proto.STRING,
            number=12,
        )

    class OutputMetadata(proto.Message):
        r"""Metadata of the prediction output to be explained.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            index_display_name_mapping (google.protobuf.struct_pb2.Value):
                Static mapping between the index and display name.

                Use this if the outputs are a deterministic n-dimensional
                array, e.g. a list of scores of all the classes in a
                pre-defined order for a multi-classification Model. It's not
                feasible if the outputs are non-deterministic, e.g. the
                Model produces top-k classes or sort the outputs by their
                values.

                The shape of the value must be an n-dimensional array of
                strings. The number of dimensions must match that of the
                outputs to be explained. The
                [Attribution.output_display_name][google.cloud.aiplatform.v1beta1.Attribution.output_display_name]
                is populated by locating in the mapping with
                [Attribution.output_index][google.cloud.aiplatform.v1beta1.Attribution.output_index].

                This field is a member of `oneof`_ ``display_name_mapping``.
            display_name_mapping_key (str):
                Specify a field name in the prediction to look for the
                display name.

                Use this if the prediction contains the display names for
                the outputs.

                The display names in the prediction must have the same shape
                of the outputs, so that it can be located by
                [Attribution.output_index][google.cloud.aiplatform.v1beta1.Attribution.output_index]
                for a specific output.

                This field is a member of `oneof`_ ``display_name_mapping``.
            output_tensor_name (str):
                Name of the output tensor. Required and is
                only applicable to Vertex AI provided images for
                Tensorflow.
        """

        index_display_name_mapping: struct_pb2.Value = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="display_name_mapping",
            message=struct_pb2.Value,
        )
        display_name_mapping_key: str = proto.Field(
            proto.STRING,
            number=2,
            oneof="display_name_mapping",
        )
        output_tensor_name: str = proto.Field(
            proto.STRING,
            number=3,
        )

    inputs: MutableMapping[str, InputMetadata] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=1,
        message=InputMetadata,
    )
    outputs: MutableMapping[str, OutputMetadata] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=2,
        message=OutputMetadata,
    )
    feature_attributions_schema_uri: str = proto.Field(
        proto.STRING,
        number=3,
    )
    latent_space_source: str = proto.Field(
        proto.STRING,
        number=5,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
