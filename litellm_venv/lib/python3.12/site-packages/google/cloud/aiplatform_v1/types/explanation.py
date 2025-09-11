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

from google.cloud.aiplatform_v1.types import explanation_metadata
from google.cloud.aiplatform_v1.types import io
from google.protobuf import struct_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "Explanation",
        "ModelExplanation",
        "Attribution",
        "Neighbor",
        "ExplanationSpec",
        "ExplanationParameters",
        "SampledShapleyAttribution",
        "IntegratedGradientsAttribution",
        "XraiAttribution",
        "SmoothGradConfig",
        "FeatureNoiseSigma",
        "BlurBaselineConfig",
        "Examples",
        "Presets",
        "ExplanationSpecOverride",
        "ExplanationMetadataOverride",
        "ExamplesOverride",
        "ExamplesRestrictionsNamespace",
    },
)


class Explanation(proto.Message):
    r"""Explanation of a prediction (provided in
    [PredictResponse.predictions][google.cloud.aiplatform.v1.PredictResponse.predictions])
    produced by the Model on a given
    [instance][google.cloud.aiplatform.v1.ExplainRequest.instances].

    Attributes:
        attributions (MutableSequence[google.cloud.aiplatform_v1.types.Attribution]):
            Output only. Feature attributions grouped by predicted
            outputs.

            For Models that predict only one output, such as regression
            Models that predict only one score, there is only one
            attibution that explains the predicted output. For Models
            that predict multiple outputs, such as multiclass Models
            that predict multiple classes, each element explains one
            specific item.
            [Attribution.output_index][google.cloud.aiplatform.v1.Attribution.output_index]
            can be used to identify which output this attribution is
            explaining.

            By default, we provide Shapley values for the predicted
            class. However, you can configure the explanation request to
            generate Shapley values for any other classes too. For
            example, if a model predicts a probability of ``0.4`` for
            approving a loan application, the model's decision is to
            reject the application since
            ``p(reject) = 0.6 > p(approve) = 0.4``, and the default
            Shapley values would be computed for rejection decision and
            not approval, even though the latter might be the positive
            class.

            If users set
            [ExplanationParameters.top_k][google.cloud.aiplatform.v1.ExplanationParameters.top_k],
            the attributions are sorted by
            [instance_output_value][Attributions.instance_output_value]
            in descending order. If
            [ExplanationParameters.output_indices][google.cloud.aiplatform.v1.ExplanationParameters.output_indices]
            is specified, the attributions are stored by
            [Attribution.output_index][google.cloud.aiplatform.v1.Attribution.output_index]
            in the same order as they appear in the output_indices.
        neighbors (MutableSequence[google.cloud.aiplatform_v1.types.Neighbor]):
            Output only. List of the nearest neighbors
            for example-based explanations.
            For models deployed with the examples
            explanations feature enabled, the attributions
            field is empty and instead the neighbors field
            is populated.
    """

    attributions: MutableSequence["Attribution"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Attribution",
    )
    neighbors: MutableSequence["Neighbor"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="Neighbor",
    )


class ModelExplanation(proto.Message):
    r"""Aggregated explanation metrics for a Model over a set of
    instances.

    Attributes:
        mean_attributions (MutableSequence[google.cloud.aiplatform_v1.types.Attribution]):
            Output only. Aggregated attributions explaining the Model's
            prediction outputs over the set of instances. The
            attributions are grouped by outputs.

            For Models that predict only one output, such as regression
            Models that predict only one score, there is only one
            attibution that explains the predicted output. For Models
            that predict multiple outputs, such as multiclass Models
            that predict multiple classes, each element explains one
            specific item.
            [Attribution.output_index][google.cloud.aiplatform.v1.Attribution.output_index]
            can be used to identify which output this attribution is
            explaining.

            The
            [baselineOutputValue][google.cloud.aiplatform.v1.Attribution.baseline_output_value],
            [instanceOutputValue][google.cloud.aiplatform.v1.Attribution.instance_output_value]
            and
            [featureAttributions][google.cloud.aiplatform.v1.Attribution.feature_attributions]
            fields are averaged over the test data.

            NOTE: Currently AutoML tabular classification Models produce
            only one attribution, which averages attributions over all
            the classes it predicts.
            [Attribution.approximation_error][google.cloud.aiplatform.v1.Attribution.approximation_error]
            is not populated.
    """

    mean_attributions: MutableSequence["Attribution"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="Attribution",
    )


class Attribution(proto.Message):
    r"""Attribution that explains a particular prediction output.

    Attributes:
        baseline_output_value (float):
            Output only. Model predicted output if the input instance is
            constructed from the baselines of all the features defined
            in
            [ExplanationMetadata.inputs][google.cloud.aiplatform.v1.ExplanationMetadata.inputs].
            The field name of the output is determined by the key in
            [ExplanationMetadata.outputs][google.cloud.aiplatform.v1.ExplanationMetadata.outputs].

            If the Model's predicted output has multiple dimensions
            (rank > 1), this is the value in the output located by
            [output_index][google.cloud.aiplatform.v1.Attribution.output_index].

            If there are multiple baselines, their output values are
            averaged.
        instance_output_value (float):
            Output only. Model predicted output on the corresponding
            [explanation instance][ExplainRequest.instances]. The field
            name of the output is determined by the key in
            [ExplanationMetadata.outputs][google.cloud.aiplatform.v1.ExplanationMetadata.outputs].

            If the Model predicted output has multiple dimensions, this
            is the value in the output located by
            [output_index][google.cloud.aiplatform.v1.Attribution.output_index].
        feature_attributions (google.protobuf.struct_pb2.Value):
            Output only. Attributions of each explained feature.
            Features are extracted from the [prediction
            instances][google.cloud.aiplatform.v1.ExplainRequest.instances]
            according to [explanation metadata for
            inputs][google.cloud.aiplatform.v1.ExplanationMetadata.inputs].

            The value is a struct, whose keys are the name of the
            feature. The values are how much the feature in the
            [instance][google.cloud.aiplatform.v1.ExplainRequest.instances]
            contributed to the predicted result.

            The format of the value is determined by the feature's input
            format:

            -  If the feature is a scalar value, the attribution value
               is a [floating
               number][google.protobuf.Value.number_value].

            -  If the feature is an array of scalar values, the
               attribution value is an
               [array][google.protobuf.Value.list_value].

            -  If the feature is a struct, the attribution value is a
               [struct][google.protobuf.Value.struct_value]. The keys in
               the attribution value struct are the same as the keys in
               the feature struct. The formats of the values in the
               attribution struct are determined by the formats of the
               values in the feature struct.

            The
            [ExplanationMetadata.feature_attributions_schema_uri][google.cloud.aiplatform.v1.ExplanationMetadata.feature_attributions_schema_uri]
            field, pointed to by the
            [ExplanationSpec][google.cloud.aiplatform.v1.ExplanationSpec]
            field of the
            [Endpoint.deployed_models][google.cloud.aiplatform.v1.Endpoint.deployed_models]
            object, points to the schema file that describes the
            features and their attribution values (if it is populated).
        output_index (MutableSequence[int]):
            Output only. The index that locates the explained prediction
            output.

            If the prediction output is a scalar value, output_index is
            not populated. If the prediction output has multiple
            dimensions, the length of the output_index list is the same
            as the number of dimensions of the output. The i-th element
            in output_index is the element index of the i-th dimension
            of the output vector. Indices start from 0.
        output_display_name (str):
            Output only. The display name of the output identified by
            [output_index][google.cloud.aiplatform.v1.Attribution.output_index].
            For example, the predicted class name by a
            multi-classification Model.

            This field is only populated iff the Model predicts display
            names as a separate field along with the explained output.
            The predicted display name must has the same shape of the
            explained output, and can be located using output_index.
        approximation_error (float):
            Output only. Error of
            [feature_attributions][google.cloud.aiplatform.v1.Attribution.feature_attributions]
            caused by approximation used in the explanation method.
            Lower value means more precise attributions.

            -  For Sampled Shapley
               [attribution][google.cloud.aiplatform.v1.ExplanationParameters.sampled_shapley_attribution],
               increasing
               [path_count][google.cloud.aiplatform.v1.SampledShapleyAttribution.path_count]
               might reduce the error.
            -  For Integrated Gradients
               [attribution][google.cloud.aiplatform.v1.ExplanationParameters.integrated_gradients_attribution],
               increasing
               [step_count][google.cloud.aiplatform.v1.IntegratedGradientsAttribution.step_count]
               might reduce the error.
            -  For [XRAI
               attribution][google.cloud.aiplatform.v1.ExplanationParameters.xrai_attribution],
               increasing
               [step_count][google.cloud.aiplatform.v1.XraiAttribution.step_count]
               might reduce the error.

            See `this
            introduction </vertex-ai/docs/explainable-ai/overview>`__
            for more information.
        output_name (str):
            Output only. Name of the explain output. Specified as the
            key in
            [ExplanationMetadata.outputs][google.cloud.aiplatform.v1.ExplanationMetadata.outputs].
    """

    baseline_output_value: float = proto.Field(
        proto.DOUBLE,
        number=1,
    )
    instance_output_value: float = proto.Field(
        proto.DOUBLE,
        number=2,
    )
    feature_attributions: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=3,
        message=struct_pb2.Value,
    )
    output_index: MutableSequence[int] = proto.RepeatedField(
        proto.INT32,
        number=4,
    )
    output_display_name: str = proto.Field(
        proto.STRING,
        number=5,
    )
    approximation_error: float = proto.Field(
        proto.DOUBLE,
        number=6,
    )
    output_name: str = proto.Field(
        proto.STRING,
        number=7,
    )


class Neighbor(proto.Message):
    r"""Neighbors for example-based explanations.

    Attributes:
        neighbor_id (str):
            Output only. The neighbor id.
        neighbor_distance (float):
            Output only. The neighbor distance.
    """

    neighbor_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    neighbor_distance: float = proto.Field(
        proto.DOUBLE,
        number=2,
    )


class ExplanationSpec(proto.Message):
    r"""Specification of Model explanation.

    Attributes:
        parameters (google.cloud.aiplatform_v1.types.ExplanationParameters):
            Required. Parameters that configure
            explaining of the Model's predictions.
        metadata (google.cloud.aiplatform_v1.types.ExplanationMetadata):
            Optional. Metadata describing the Model's
            input and output for explanation.
    """

    parameters: "ExplanationParameters" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ExplanationParameters",
    )
    metadata: explanation_metadata.ExplanationMetadata = proto.Field(
        proto.MESSAGE,
        number=2,
        message=explanation_metadata.ExplanationMetadata,
    )


class ExplanationParameters(proto.Message):
    r"""Parameters to configure explaining for Model's predictions.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        sampled_shapley_attribution (google.cloud.aiplatform_v1.types.SampledShapleyAttribution):
            An attribution method that approximates
            Shapley values for features that contribute to
            the label being predicted. A sampling strategy
            is used to approximate the value rather than
            considering all subsets of features. Refer to
            this paper for model details:
            https://arxiv.org/abs/1306.4265.

            This field is a member of `oneof`_ ``method``.
        integrated_gradients_attribution (google.cloud.aiplatform_v1.types.IntegratedGradientsAttribution):
            An attribution method that computes
            Aumann-Shapley values taking advantage of the
            model's fully differentiable structure. Refer to
            this paper for more details:
            https://arxiv.org/abs/1703.01365

            This field is a member of `oneof`_ ``method``.
        xrai_attribution (google.cloud.aiplatform_v1.types.XraiAttribution):
            An attribution method that redistributes
            Integrated Gradients attribution to segmented
            regions, taking advantage of the model's fully
            differentiable structure. Refer to this paper
            for more details:
            https://arxiv.org/abs/1906.02825

            XRAI currently performs better on natural
            images, like a picture of a house or an animal.
            If the images are taken in artificial
            environments, like a lab or manufacturing line,
            or from diagnostic equipment, like x-rays or
            quality-control cameras, use Integrated
            Gradients instead.

            This field is a member of `oneof`_ ``method``.
        examples (google.cloud.aiplatform_v1.types.Examples):
            Example-based explanations that returns the
            nearest neighbors from the provided dataset.

            This field is a member of `oneof`_ ``method``.
        top_k (int):
            If populated, returns attributions for top K
            indices of outputs (defaults to 1). Only applies
            to Models that predicts more than one outputs
            (e,g, multi-class Models). When set to -1,
            returns explanations for all outputs.
        output_indices (google.protobuf.struct_pb2.ListValue):
            If populated, only returns attributions that have
            [output_index][google.cloud.aiplatform.v1.Attribution.output_index]
            contained in output_indices. It must be an ndarray of
            integers, with the same shape of the output it's explaining.

            If not populated, returns attributions for
            [top_k][google.cloud.aiplatform.v1.ExplanationParameters.top_k]
            indices of outputs. If neither top_k nor output_indices is
            populated, returns the argmax index of the outputs.

            Only applicable to Models that predict multiple outputs
            (e,g, multi-class Models that predict multiple classes).
    """

    sampled_shapley_attribution: "SampledShapleyAttribution" = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="method",
        message="SampledShapleyAttribution",
    )
    integrated_gradients_attribution: "IntegratedGradientsAttribution" = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="method",
        message="IntegratedGradientsAttribution",
    )
    xrai_attribution: "XraiAttribution" = proto.Field(
        proto.MESSAGE,
        number=3,
        oneof="method",
        message="XraiAttribution",
    )
    examples: "Examples" = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="method",
        message="Examples",
    )
    top_k: int = proto.Field(
        proto.INT32,
        number=4,
    )
    output_indices: struct_pb2.ListValue = proto.Field(
        proto.MESSAGE,
        number=5,
        message=struct_pb2.ListValue,
    )


class SampledShapleyAttribution(proto.Message):
    r"""An attribution method that approximates Shapley values for
    features that contribute to the label being predicted. A
    sampling strategy is used to approximate the value rather than
    considering all subsets of features.

    Attributes:
        path_count (int):
            Required. The number of feature permutations to consider
            when approximating the Shapley values.

            Valid range of its value is [1, 50], inclusively.
    """

    path_count: int = proto.Field(
        proto.INT32,
        number=1,
    )


class IntegratedGradientsAttribution(proto.Message):
    r"""An attribution method that computes the Aumann-Shapley value
    taking advantage of the model's fully differentiable structure.
    Refer to this paper for more details:
    https://arxiv.org/abs/1703.01365

    Attributes:
        step_count (int):
            Required. The number of steps for approximating the path
            integral. A good value to start is 50 and gradually increase
            until the sum to diff property is within the desired error
            range.

            Valid range of its value is [1, 100], inclusively.
        smooth_grad_config (google.cloud.aiplatform_v1.types.SmoothGradConfig):
            Config for SmoothGrad approximation of
            gradients.
            When enabled, the gradients are approximated by
            averaging the gradients from noisy samples in
            the vicinity of the inputs. Adding noise can
            help improve the computed gradients. Refer to
            this paper for more details:
            https://arxiv.org/pdf/1706.03825.pdf
        blur_baseline_config (google.cloud.aiplatform_v1.types.BlurBaselineConfig):
            Config for IG with blur baseline.

            When enabled, a linear path from the maximally
            blurred image to the input image is created.
            Using a blurred baseline instead of zero (black
            image) is motivated by the BlurIG approach
            explained here:

            https://arxiv.org/abs/2004.03383
    """

    step_count: int = proto.Field(
        proto.INT32,
        number=1,
    )
    smooth_grad_config: "SmoothGradConfig" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="SmoothGradConfig",
    )
    blur_baseline_config: "BlurBaselineConfig" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="BlurBaselineConfig",
    )


class XraiAttribution(proto.Message):
    r"""An explanation method that redistributes Integrated Gradients
    attributions to segmented regions, taking advantage of the
    model's fully differentiable structure. Refer to this paper for
    more details:

    https://arxiv.org/abs/1906.02825

    Supported only by image Models.

    Attributes:
        step_count (int):
            Required. The number of steps for approximating the path
            integral. A good value to start is 50 and gradually increase
            until the sum to diff property is met within the desired
            error range.

            Valid range of its value is [1, 100], inclusively.
        smooth_grad_config (google.cloud.aiplatform_v1.types.SmoothGradConfig):
            Config for SmoothGrad approximation of
            gradients.
            When enabled, the gradients are approximated by
            averaging the gradients from noisy samples in
            the vicinity of the inputs. Adding noise can
            help improve the computed gradients. Refer to
            this paper for more details:
            https://arxiv.org/pdf/1706.03825.pdf
        blur_baseline_config (google.cloud.aiplatform_v1.types.BlurBaselineConfig):
            Config for XRAI with blur baseline.

            When enabled, a linear path from the maximally
            blurred image to the input image is created.
            Using a blurred baseline instead of zero (black
            image) is motivated by the BlurIG approach
            explained here:

            https://arxiv.org/abs/2004.03383
    """

    step_count: int = proto.Field(
        proto.INT32,
        number=1,
    )
    smooth_grad_config: "SmoothGradConfig" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="SmoothGradConfig",
    )
    blur_baseline_config: "BlurBaselineConfig" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="BlurBaselineConfig",
    )


class SmoothGradConfig(proto.Message):
    r"""Config for SmoothGrad approximation of gradients.

    When enabled, the gradients are approximated by averaging the
    gradients from noisy samples in the vicinity of the inputs.
    Adding noise can help improve the computed gradients. Refer to
    this paper for more details:

    https://arxiv.org/pdf/1706.03825.pdf

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        noise_sigma (float):
            This is a single float value and will be used to add noise
            to all the features. Use this field when all features are
            normalized to have the same distribution: scale to range [0,
            1], [-1, 1] or z-scoring, where features are normalized to
            have 0-mean and 1-variance. Learn more about
            `normalization <https://developers.google.com/machine-learning/data-prep/transform/normalization>`__.

            For best results the recommended value is about 10% - 20% of
            the standard deviation of the input feature. Refer to
            section 3.2 of the SmoothGrad paper:
            https://arxiv.org/pdf/1706.03825.pdf. Defaults to 0.1.

            If the distribution is different per feature, set
            [feature_noise_sigma][google.cloud.aiplatform.v1.SmoothGradConfig.feature_noise_sigma]
            instead for each feature.

            This field is a member of `oneof`_ ``GradientNoiseSigma``.
        feature_noise_sigma (google.cloud.aiplatform_v1.types.FeatureNoiseSigma):
            This is similar to
            [noise_sigma][google.cloud.aiplatform.v1.SmoothGradConfig.noise_sigma],
            but provides additional flexibility. A separate noise sigma
            can be provided for each feature, which is useful if their
            distributions are different. No noise is added to features
            that are not set. If this field is unset,
            [noise_sigma][google.cloud.aiplatform.v1.SmoothGradConfig.noise_sigma]
            will be used for all features.

            This field is a member of `oneof`_ ``GradientNoiseSigma``.
        noisy_sample_count (int):
            The number of gradient samples to use for approximation. The
            higher this number, the more accurate the gradient is, but
            the runtime complexity increases by this factor as well.
            Valid range of its value is [1, 50]. Defaults to 3.
    """

    noise_sigma: float = proto.Field(
        proto.FLOAT,
        number=1,
        oneof="GradientNoiseSigma",
    )
    feature_noise_sigma: "FeatureNoiseSigma" = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="GradientNoiseSigma",
        message="FeatureNoiseSigma",
    )
    noisy_sample_count: int = proto.Field(
        proto.INT32,
        number=3,
    )


class FeatureNoiseSigma(proto.Message):
    r"""Noise sigma by features. Noise sigma represents the standard
    deviation of the gaussian kernel that will be used to add noise
    to interpolated inputs prior to computing gradients.

    Attributes:
        noise_sigma (MutableSequence[google.cloud.aiplatform_v1.types.FeatureNoiseSigma.NoiseSigmaForFeature]):
            Noise sigma per feature. No noise is added to
            features that are not set.
    """

    class NoiseSigmaForFeature(proto.Message):
        r"""Noise sigma for a single feature.

        Attributes:
            name (str):
                The name of the input feature for which noise sigma is
                provided. The features are defined in [explanation metadata
                inputs][google.cloud.aiplatform.v1.ExplanationMetadata.inputs].
            sigma (float):
                This represents the standard deviation of the Gaussian
                kernel that will be used to add noise to the feature prior
                to computing gradients. Similar to
                [noise_sigma][google.cloud.aiplatform.v1.SmoothGradConfig.noise_sigma]
                but represents the noise added to the current feature.
                Defaults to 0.1.
        """

        name: str = proto.Field(
            proto.STRING,
            number=1,
        )
        sigma: float = proto.Field(
            proto.FLOAT,
            number=2,
        )

    noise_sigma: MutableSequence[NoiseSigmaForFeature] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=NoiseSigmaForFeature,
    )


class BlurBaselineConfig(proto.Message):
    r"""Config for blur baseline.

    When enabled, a linear path from the maximally blurred image to
    the input image is created. Using a blurred baseline instead of
    zero (black image) is motivated by the BlurIG approach explained
    here:

    https://arxiv.org/abs/2004.03383

    Attributes:
        max_blur_sigma (float):
            The standard deviation of the blur kernel for
            the blurred baseline. The same blurring
            parameter is used for both the height and the
            width dimension. If not set, the method defaults
            to the zero (i.e. black for images) baseline.
    """

    max_blur_sigma: float = proto.Field(
        proto.FLOAT,
        number=1,
    )


class Examples(proto.Message):
    r"""Example-based explainability that returns the nearest
    neighbors from the provided dataset.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        example_gcs_source (google.cloud.aiplatform_v1.types.Examples.ExampleGcsSource):
            The Cloud Storage input instances.

            This field is a member of `oneof`_ ``source``.
        nearest_neighbor_search_config (google.protobuf.struct_pb2.Value):
            The full configuration for the generated index, the
            semantics are the same as
            [metadata][google.cloud.aiplatform.v1.Index.metadata] and
            should match
            `NearestNeighborSearchConfig <https://cloud.google.com/vertex-ai/docs/explainable-ai/configuring-explanations-example-based#nearest-neighbor-search-config>`__.

            This field is a member of `oneof`_ ``config``.
        presets (google.cloud.aiplatform_v1.types.Presets):
            Simplified preset configuration, which
            automatically sets configuration values based on
            the desired query speed-precision trade-off and
            modality.

            This field is a member of `oneof`_ ``config``.
        neighbor_count (int):
            The number of neighbors to return when
            querying for examples.
    """

    class ExampleGcsSource(proto.Message):
        r"""The Cloud Storage input instances.

        Attributes:
            data_format (google.cloud.aiplatform_v1.types.Examples.ExampleGcsSource.DataFormat):
                The format in which instances are given, if
                not specified, assume it's JSONL format.
                Currently only JSONL format is supported.
            gcs_source (google.cloud.aiplatform_v1.types.GcsSource):
                The Cloud Storage location for the input
                instances.
        """

        class DataFormat(proto.Enum):
            r"""The format of the input example instances.

            Values:
                DATA_FORMAT_UNSPECIFIED (0):
                    Format unspecified, used when unset.
                JSONL (1):
                    Examples are stored in JSONL files.
            """
            DATA_FORMAT_UNSPECIFIED = 0
            JSONL = 1

        data_format: "Examples.ExampleGcsSource.DataFormat" = proto.Field(
            proto.ENUM,
            number=1,
            enum="Examples.ExampleGcsSource.DataFormat",
        )
        gcs_source: io.GcsSource = proto.Field(
            proto.MESSAGE,
            number=2,
            message=io.GcsSource,
        )

    example_gcs_source: ExampleGcsSource = proto.Field(
        proto.MESSAGE,
        number=5,
        oneof="source",
        message=ExampleGcsSource,
    )
    nearest_neighbor_search_config: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="config",
        message=struct_pb2.Value,
    )
    presets: "Presets" = proto.Field(
        proto.MESSAGE,
        number=4,
        oneof="config",
        message="Presets",
    )
    neighbor_count: int = proto.Field(
        proto.INT32,
        number=3,
    )


class Presets(proto.Message):
    r"""Preset configuration for example-based explanations

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        query (google.cloud.aiplatform_v1.types.Presets.Query):
            Preset option controlling parameters for speed-precision
            trade-off when querying for examples. If omitted, defaults
            to ``PRECISE``.

            This field is a member of `oneof`_ ``_query``.
        modality (google.cloud.aiplatform_v1.types.Presets.Modality):
            The modality of the uploaded model, which
            automatically configures the distance
            measurement and feature normalization for the
            underlying example index and queries. If your
            model does not precisely fit one of these types,
            it is okay to choose the closest type.
    """

    class Query(proto.Enum):
        r"""Preset option controlling parameters for query
        speed-precision trade-off

        Values:
            PRECISE (0):
                More precise neighbors as a trade-off against
                slower response.
            FAST (1):
                Faster response as a trade-off against less
                precise neighbors.
        """
        PRECISE = 0
        FAST = 1

    class Modality(proto.Enum):
        r"""Preset option controlling parameters for different modalities

        Values:
            MODALITY_UNSPECIFIED (0):
                Should not be set. Added as a recommended
                best practice for enums
            IMAGE (1):
                IMAGE modality
            TEXT (2):
                TEXT modality
            TABULAR (3):
                TABULAR modality
        """
        MODALITY_UNSPECIFIED = 0
        IMAGE = 1
        TEXT = 2
        TABULAR = 3

    query: Query = proto.Field(
        proto.ENUM,
        number=1,
        optional=True,
        enum=Query,
    )
    modality: Modality = proto.Field(
        proto.ENUM,
        number=2,
        enum=Modality,
    )


class ExplanationSpecOverride(proto.Message):
    r"""The [ExplanationSpec][google.cloud.aiplatform.v1.ExplanationSpec]
    entries that can be overridden at [online
    explanation][google.cloud.aiplatform.v1.PredictionService.Explain]
    time.

    Attributes:
        parameters (google.cloud.aiplatform_v1.types.ExplanationParameters):
            The parameters to be overridden. Note that
            the attribution method cannot be changed. If not
            specified, no parameter is overridden.
        metadata (google.cloud.aiplatform_v1.types.ExplanationMetadataOverride):
            The metadata to be overridden. If not
            specified, no metadata is overridden.
        examples_override (google.cloud.aiplatform_v1.types.ExamplesOverride):
            The example-based explanations parameter
            overrides.
    """

    parameters: "ExplanationParameters" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="ExplanationParameters",
    )
    metadata: "ExplanationMetadataOverride" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="ExplanationMetadataOverride",
    )
    examples_override: "ExamplesOverride" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="ExamplesOverride",
    )


class ExplanationMetadataOverride(proto.Message):
    r"""The
    [ExplanationMetadata][google.cloud.aiplatform.v1.ExplanationMetadata]
    entries that can be overridden at [online
    explanation][google.cloud.aiplatform.v1.PredictionService.Explain]
    time.

    Attributes:
        inputs (MutableMapping[str, google.cloud.aiplatform_v1.types.ExplanationMetadataOverride.InputMetadataOverride]):
            Required. Overrides the [input
            metadata][google.cloud.aiplatform.v1.ExplanationMetadata.inputs]
            of the features. The key is the name of the feature to be
            overridden. The keys specified here must exist in the input
            metadata to be overridden. If a feature is not specified
            here, the corresponding feature's input metadata is not
            overridden.
    """

    class InputMetadataOverride(proto.Message):
        r"""The [input
        metadata][google.cloud.aiplatform.v1.ExplanationMetadata.InputMetadata]
        entries to be overridden.

        Attributes:
            input_baselines (MutableSequence[google.protobuf.struct_pb2.Value]):
                Baseline inputs for this feature.

                This overrides the ``input_baseline`` field of the
                [ExplanationMetadata.InputMetadata][google.cloud.aiplatform.v1.ExplanationMetadata.InputMetadata]
                object of the corresponding feature's input metadata. If
                it's not specified, the original baselines are not
                overridden.
        """

        input_baselines: MutableSequence[struct_pb2.Value] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=struct_pb2.Value,
        )

    inputs: MutableMapping[str, InputMetadataOverride] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=1,
        message=InputMetadataOverride,
    )


class ExamplesOverride(proto.Message):
    r"""Overrides for example-based explanations.

    Attributes:
        neighbor_count (int):
            The number of neighbors to return.
        crowding_count (int):
            The number of neighbors to return that have
            the same crowding tag.
        restrictions (MutableSequence[google.cloud.aiplatform_v1.types.ExamplesRestrictionsNamespace]):
            Restrict the resulting nearest neighbors to
            respect these constraints.
        return_embeddings (bool):
            If true, return the embeddings instead of
            neighbors.
        data_format (google.cloud.aiplatform_v1.types.ExamplesOverride.DataFormat):
            The format of the data being provided with
            each call.
    """

    class DataFormat(proto.Enum):
        r"""Data format enum.

        Values:
            DATA_FORMAT_UNSPECIFIED (0):
                Unspecified format. Must not be used.
            INSTANCES (1):
                Provided data is a set of model inputs.
            EMBEDDINGS (2):
                Provided data is a set of embeddings.
        """
        DATA_FORMAT_UNSPECIFIED = 0
        INSTANCES = 1
        EMBEDDINGS = 2

    neighbor_count: int = proto.Field(
        proto.INT32,
        number=1,
    )
    crowding_count: int = proto.Field(
        proto.INT32,
        number=2,
    )
    restrictions: MutableSequence[
        "ExamplesRestrictionsNamespace"
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="ExamplesRestrictionsNamespace",
    )
    return_embeddings: bool = proto.Field(
        proto.BOOL,
        number=4,
    )
    data_format: DataFormat = proto.Field(
        proto.ENUM,
        number=5,
        enum=DataFormat,
    )


class ExamplesRestrictionsNamespace(proto.Message):
    r"""Restrictions namespace for example-based explanations
    overrides.

    Attributes:
        namespace_name (str):
            The namespace name.
        allow (MutableSequence[str]):
            The list of allowed tags.
        deny (MutableSequence[str]):
            The list of deny tags.
    """

    namespace_name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    allow: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )
    deny: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
