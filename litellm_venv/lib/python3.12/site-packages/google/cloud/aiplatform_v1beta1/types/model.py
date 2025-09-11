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

from google.cloud.aiplatform_v1beta1.types import deployed_model_ref
from google.cloud.aiplatform_v1beta1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1beta1.types import env_var
from google.cloud.aiplatform_v1beta1.types import explanation
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "Model",
        "LargeModelReference",
        "ModelGardenSource",
        "GenieSource",
        "PredictSchemata",
        "ModelContainerSpec",
        "Port",
        "ModelSourceInfo",
        "Probe",
    },
)


class Model(proto.Message):
    r"""A trained machine learning Model.

    Attributes:
        name (str):
            The resource name of the Model.
        version_id (str):
            Output only. Immutable. The version ID of the
            model. A new version is committed when a new
            model version is uploaded or trained under an
            existing model id. It is an auto-incrementing
            decimal number in string representation.
        version_aliases (MutableSequence[str]):
            User provided version aliases so that a model version can be
            referenced via alias (i.e.
            ``projects/{project}/locations/{location}/models/{model_id}@{version_alias}``
            instead of auto-generated version id (i.e.
            ``projects/{project}/locations/{location}/models/{model_id}@{version_id})``.
            The format is [a-z][a-zA-Z0-9-]{0,126}[a-z0-9] to
            distinguish from version_id. A default version alias will be
            created for the first version of the model, and there must
            be exactly one default version alias for a model.
        version_create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this version was
            created.
        version_update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this version was
            most recently updated.
        display_name (str):
            Required. The display name of the Model.
            The name can be up to 128 characters long and
            can consist of any UTF-8 characters.
        description (str):
            The description of the Model.
        version_description (str):
            The description of this version.
        predict_schemata (google.cloud.aiplatform_v1beta1.types.PredictSchemata):
            The schemata that describe formats of the Model's
            predictions and explanations as given and returned via
            [PredictionService.Predict][google.cloud.aiplatform.v1beta1.PredictionService.Predict]
            and
            [PredictionService.Explain][google.cloud.aiplatform.v1beta1.PredictionService.Explain].
        metadata_schema_uri (str):
            Immutable. Points to a YAML file stored on Google Cloud
            Storage describing additional information about the Model,
            that is specific to it. Unset if the Model does not have any
            additional information. The schema is defined as an OpenAPI
            3.0.2 `Schema
            Object <https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#schemaObject>`__.
            AutoML Models always have this field populated by Vertex AI,
            if no additional metadata is needed, this field is set to an
            empty string. Note: The URI given on output will be
            immutable and probably different, including the URI scheme,
            than the one given on input. The output URI will point to a
            location where the user only has a read access.
        metadata (google.protobuf.struct_pb2.Value):
            Immutable. An additional information about the Model; the
            schema of the metadata can be found in
            [metadata_schema][google.cloud.aiplatform.v1beta1.Model.metadata_schema_uri].
            Unset if the Model does not have any additional information.
        supported_export_formats (MutableSequence[google.cloud.aiplatform_v1beta1.types.Model.ExportFormat]):
            Output only. The formats in which this Model
            may be exported. If empty, this Model is not
            available for export.
        training_pipeline (str):
            Output only. The resource name of the
            TrainingPipeline that uploaded this Model, if
            any.
        container_spec (google.cloud.aiplatform_v1beta1.types.ModelContainerSpec):
            Input only. The specification of the container that is to be
            used when deploying this Model. The specification is
            ingested upon
            [ModelService.UploadModel][google.cloud.aiplatform.v1beta1.ModelService.UploadModel],
            and all binaries it contains are copied and stored
            internally by Vertex AI. Not required for AutoML Models.
        artifact_uri (str):
            Immutable. The path to the directory
            containing the Model artifact and any of its
            supporting files. Not required for AutoML
            Models.
        supported_deployment_resources_types (MutableSequence[google.cloud.aiplatform_v1beta1.types.Model.DeploymentResourcesType]):
            Output only. When this Model is deployed, its prediction
            resources are described by the ``prediction_resources``
            field of the
            [Endpoint.deployed_models][google.cloud.aiplatform.v1beta1.Endpoint.deployed_models]
            object. Because not all Models support all resource
            configuration types, the configuration types this Model
            supports are listed here. If no configuration types are
            listed, the Model cannot be deployed to an
            [Endpoint][google.cloud.aiplatform.v1beta1.Endpoint] and
            does not support online predictions
            ([PredictionService.Predict][google.cloud.aiplatform.v1beta1.PredictionService.Predict]
            or
            [PredictionService.Explain][google.cloud.aiplatform.v1beta1.PredictionService.Explain]).
            Such a Model can serve predictions by using a
            [BatchPredictionJob][google.cloud.aiplatform.v1beta1.BatchPredictionJob],
            if it has at least one entry each in
            [supported_input_storage_formats][google.cloud.aiplatform.v1beta1.Model.supported_input_storage_formats]
            and
            [supported_output_storage_formats][google.cloud.aiplatform.v1beta1.Model.supported_output_storage_formats].
        supported_input_storage_formats (MutableSequence[str]):
            Output only. The formats this Model supports in
            [BatchPredictionJob.input_config][google.cloud.aiplatform.v1beta1.BatchPredictionJob.input_config].
            If
            [PredictSchemata.instance_schema_uri][google.cloud.aiplatform.v1beta1.PredictSchemata.instance_schema_uri]
            exists, the instances should be given as per that schema.

            The possible formats are:

            -  ``jsonl`` The JSON Lines format, where each instance is a
               single line. Uses
               [GcsSource][google.cloud.aiplatform.v1beta1.BatchPredictionJob.InputConfig.gcs_source].

            -  ``csv`` The CSV format, where each instance is a single
               comma-separated line. The first line in the file is the
               header, containing comma-separated field names. Uses
               [GcsSource][google.cloud.aiplatform.v1beta1.BatchPredictionJob.InputConfig.gcs_source].

            -  ``tf-record`` The TFRecord format, where each instance is
               a single record in tfrecord syntax. Uses
               [GcsSource][google.cloud.aiplatform.v1beta1.BatchPredictionJob.InputConfig.gcs_source].

            -  ``tf-record-gzip`` Similar to ``tf-record``, but the file
               is gzipped. Uses
               [GcsSource][google.cloud.aiplatform.v1beta1.BatchPredictionJob.InputConfig.gcs_source].

            -  ``bigquery`` Each instance is a single row in BigQuery.
               Uses
               [BigQuerySource][google.cloud.aiplatform.v1beta1.BatchPredictionJob.InputConfig.bigquery_source].

            -  ``file-list`` Each line of the file is the location of an
               instance to process, uses ``gcs_source`` field of the
               [InputConfig][google.cloud.aiplatform.v1beta1.BatchPredictionJob.InputConfig]
               object.

            If this Model doesn't support any of these formats it means
            it cannot be used with a
            [BatchPredictionJob][google.cloud.aiplatform.v1beta1.BatchPredictionJob].
            However, if it has
            [supported_deployment_resources_types][google.cloud.aiplatform.v1beta1.Model.supported_deployment_resources_types],
            it could serve online predictions by using
            [PredictionService.Predict][google.cloud.aiplatform.v1beta1.PredictionService.Predict]
            or
            [PredictionService.Explain][google.cloud.aiplatform.v1beta1.PredictionService.Explain].
        supported_output_storage_formats (MutableSequence[str]):
            Output only. The formats this Model supports in
            [BatchPredictionJob.output_config][google.cloud.aiplatform.v1beta1.BatchPredictionJob.output_config].
            If both
            [PredictSchemata.instance_schema_uri][google.cloud.aiplatform.v1beta1.PredictSchemata.instance_schema_uri]
            and
            [PredictSchemata.prediction_schema_uri][google.cloud.aiplatform.v1beta1.PredictSchemata.prediction_schema_uri]
            exist, the predictions are returned together with their
            instances. In other words, the prediction has the original
            instance data first, followed by the actual prediction
            content (as per the schema).

            The possible formats are:

            -  ``jsonl`` The JSON Lines format, where each prediction is
               a single line. Uses
               [GcsDestination][google.cloud.aiplatform.v1beta1.BatchPredictionJob.OutputConfig.gcs_destination].

            -  ``csv`` The CSV format, where each prediction is a single
               comma-separated line. The first line in the file is the
               header, containing comma-separated field names. Uses
               [GcsDestination][google.cloud.aiplatform.v1beta1.BatchPredictionJob.OutputConfig.gcs_destination].

            -  ``bigquery`` Each prediction is a single row in a
               BigQuery table, uses
               [BigQueryDestination][google.cloud.aiplatform.v1beta1.BatchPredictionJob.OutputConfig.bigquery_destination]
               .

            If this Model doesn't support any of these formats it means
            it cannot be used with a
            [BatchPredictionJob][google.cloud.aiplatform.v1beta1.BatchPredictionJob].
            However, if it has
            [supported_deployment_resources_types][google.cloud.aiplatform.v1beta1.Model.supported_deployment_resources_types],
            it could serve online predictions by using
            [PredictionService.Predict][google.cloud.aiplatform.v1beta1.PredictionService.Predict]
            or
            [PredictionService.Explain][google.cloud.aiplatform.v1beta1.PredictionService.Explain].
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Model was
            uploaded into Vertex AI.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Model was
            most recently updated.
        deployed_models (MutableSequence[google.cloud.aiplatform_v1beta1.types.DeployedModelRef]):
            Output only. The pointers to DeployedModels
            created from this Model. Note that Model could
            have been deployed to Endpoints in different
            Locations.
        explanation_spec (google.cloud.aiplatform_v1beta1.types.ExplanationSpec):
            The default explanation specification for this Model.

            The Model can be used for [requesting
            explanation][google.cloud.aiplatform.v1beta1.PredictionService.Explain]
            after being
            [deployed][google.cloud.aiplatform.v1beta1.EndpointService.DeployModel]
            if it is populated. The Model can be used for [batch
            explanation][google.cloud.aiplatform.v1beta1.BatchPredictionJob.generate_explanation]
            if it is populated.

            All fields of the explanation_spec can be overridden by
            [explanation_spec][google.cloud.aiplatform.v1beta1.DeployedModel.explanation_spec]
            of
            [DeployModelRequest.deployed_model][google.cloud.aiplatform.v1beta1.DeployModelRequest.deployed_model],
            or
            [explanation_spec][google.cloud.aiplatform.v1beta1.BatchPredictionJob.explanation_spec]
            of
            [BatchPredictionJob][google.cloud.aiplatform.v1beta1.BatchPredictionJob].

            If the default explanation specification is not set for this
            Model, this Model can still be used for [requesting
            explanation][google.cloud.aiplatform.v1beta1.PredictionService.Explain]
            by setting
            [explanation_spec][google.cloud.aiplatform.v1beta1.DeployedModel.explanation_spec]
            of
            [DeployModelRequest.deployed_model][google.cloud.aiplatform.v1beta1.DeployModelRequest.deployed_model]
            and for [batch
            explanation][google.cloud.aiplatform.v1beta1.BatchPredictionJob.generate_explanation]
            by setting
            [explanation_spec][google.cloud.aiplatform.v1beta1.BatchPredictionJob.explanation_spec]
            of
            [BatchPredictionJob][google.cloud.aiplatform.v1beta1.BatchPredictionJob].
        etag (str):
            Used to perform consistent read-modify-write
            updates. If not set, a blind "overwrite" update
            happens.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to
            organize your Models.
            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            and examples of labels.
        encryption_spec (google.cloud.aiplatform_v1beta1.types.EncryptionSpec):
            Customer-managed encryption key spec for a
            Model. If set, this Model and all sub-resources
            of this Model will be secured by this key.
        model_source_info (google.cloud.aiplatform_v1beta1.types.ModelSourceInfo):
            Output only. Source of a model. It can either
            be automl training pipeline, custom training
            pipeline, BigQuery ML, or saved and tuned from
            Genie or Model Garden.
        original_model_info (google.cloud.aiplatform_v1beta1.types.Model.OriginalModelInfo):
            Output only. If this Model is a copy of
            another Model, this contains info about the
            original.
        metadata_artifact (str):
            Output only. The resource name of the Artifact that was
            created in MetadataStore when creating the Model. The
            Artifact resource name pattern is
            ``projects/{project}/locations/{location}/metadataStores/{metadata_store}/artifacts/{artifact}``.
        base_model_source (google.cloud.aiplatform_v1beta1.types.Model.BaseModelSource):
            Optional. User input field to specify the
            base model source. Currently it only supports
            specifing the Model Garden models and Genie
            models.
    """

    class DeploymentResourcesType(proto.Enum):
        r"""Identifies a type of Model's prediction resources.

        Values:
            DEPLOYMENT_RESOURCES_TYPE_UNSPECIFIED (0):
                Should not be used.
            DEDICATED_RESOURCES (1):
                Resources that are dedicated to the
                [DeployedModel][google.cloud.aiplatform.v1beta1.DeployedModel],
                and that need a higher degree of manual configuration.
            AUTOMATIC_RESOURCES (2):
                Resources that to large degree are decided by
                Vertex AI, and require only a modest additional
                configuration.
            SHARED_RESOURCES (3):
                Resources that can be shared by multiple
                [DeployedModels][google.cloud.aiplatform.v1beta1.DeployedModel].
                A pre-configured
                [DeploymentResourcePool][google.cloud.aiplatform.v1beta1.DeploymentResourcePool]
                is required.
        """
        DEPLOYMENT_RESOURCES_TYPE_UNSPECIFIED = 0
        DEDICATED_RESOURCES = 1
        AUTOMATIC_RESOURCES = 2
        SHARED_RESOURCES = 3

    class ExportFormat(proto.Message):
        r"""Represents export format supported by the Model.
        All formats export to Google Cloud Storage.

        Attributes:
            id (str):
                Output only. The ID of the export format. The possible
                format IDs are:

                -  ``tflite`` Used for Android mobile devices.

                -  ``edgetpu-tflite`` Used for `Edge
                   TPU <https://cloud.google.com/edge-tpu/>`__ devices.

                -  ``tf-saved-model`` A tensorflow model in SavedModel
                   format.

                -  ``tf-js`` A
                   `TensorFlow.js <https://www.tensorflow.org/js>`__ model
                   that can be used in the browser and in Node.js using
                   JavaScript.

                -  ``core-ml`` Used for iOS mobile devices.

                -  ``custom-trained`` A Model that was uploaded or trained
                   by custom code.
            exportable_contents (MutableSequence[google.cloud.aiplatform_v1beta1.types.Model.ExportFormat.ExportableContent]):
                Output only. The content of this Model that
                may be exported.
        """

        class ExportableContent(proto.Enum):
            r"""The Model content that can be exported.

            Values:
                EXPORTABLE_CONTENT_UNSPECIFIED (0):
                    Should not be used.
                ARTIFACT (1):
                    Model artifact and any of its supported files. Will be
                    exported to the location specified by the
                    ``artifactDestination`` field of the
                    [ExportModelRequest.output_config][google.cloud.aiplatform.v1beta1.ExportModelRequest.output_config]
                    object.
                IMAGE (2):
                    The container image that is to be used when deploying this
                    Model. Will be exported to the location specified by the
                    ``imageDestination`` field of the
                    [ExportModelRequest.output_config][google.cloud.aiplatform.v1beta1.ExportModelRequest.output_config]
                    object.
            """
            EXPORTABLE_CONTENT_UNSPECIFIED = 0
            ARTIFACT = 1
            IMAGE = 2

        id: str = proto.Field(
            proto.STRING,
            number=1,
        )
        exportable_contents: MutableSequence[
            "Model.ExportFormat.ExportableContent"
        ] = proto.RepeatedField(
            proto.ENUM,
            number=2,
            enum="Model.ExportFormat.ExportableContent",
        )

    class OriginalModelInfo(proto.Message):
        r"""Contains information about the original Model if this Model
        is a copy.

        Attributes:
            model (str):
                Output only. The resource name of the Model this Model is a
                copy of, including the revision. Format:
                ``projects/{project}/locations/{location}/models/{model_id}@{version_id}``
        """

        model: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class BaseModelSource(proto.Message):
        r"""User input field to specify the base model source. Currently
        it only supports specifing the Model Garden models and Genie
        models.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            model_garden_source (google.cloud.aiplatform_v1beta1.types.ModelGardenSource):
                Source information of Model Garden models.

                This field is a member of `oneof`_ ``source``.
            genie_source (google.cloud.aiplatform_v1beta1.types.GenieSource):
                Information about the base model of Genie
                models.

                This field is a member of `oneof`_ ``source``.
        """

        model_garden_source: "ModelGardenSource" = proto.Field(
            proto.MESSAGE,
            number=1,
            oneof="source",
            message="ModelGardenSource",
        )
        genie_source: "GenieSource" = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="source",
            message="GenieSource",
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    version_id: str = proto.Field(
        proto.STRING,
        number=28,
    )
    version_aliases: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=29,
    )
    version_create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=31,
        message=timestamp_pb2.Timestamp,
    )
    version_update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=32,
        message=timestamp_pb2.Timestamp,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=3,
    )
    version_description: str = proto.Field(
        proto.STRING,
        number=30,
    )
    predict_schemata: "PredictSchemata" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="PredictSchemata",
    )
    metadata_schema_uri: str = proto.Field(
        proto.STRING,
        number=5,
    )
    metadata: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=6,
        message=struct_pb2.Value,
    )
    supported_export_formats: MutableSequence[ExportFormat] = proto.RepeatedField(
        proto.MESSAGE,
        number=20,
        message=ExportFormat,
    )
    training_pipeline: str = proto.Field(
        proto.STRING,
        number=7,
    )
    container_spec: "ModelContainerSpec" = proto.Field(
        proto.MESSAGE,
        number=9,
        message="ModelContainerSpec",
    )
    artifact_uri: str = proto.Field(
        proto.STRING,
        number=26,
    )
    supported_deployment_resources_types: MutableSequence[
        DeploymentResourcesType
    ] = proto.RepeatedField(
        proto.ENUM,
        number=10,
        enum=DeploymentResourcesType,
    )
    supported_input_storage_formats: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=11,
    )
    supported_output_storage_formats: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=12,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=13,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=14,
        message=timestamp_pb2.Timestamp,
    )
    deployed_models: MutableSequence[
        deployed_model_ref.DeployedModelRef
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=15,
        message=deployed_model_ref.DeployedModelRef,
    )
    explanation_spec: explanation.ExplanationSpec = proto.Field(
        proto.MESSAGE,
        number=23,
        message=explanation.ExplanationSpec,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=16,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=17,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=24,
        message=gca_encryption_spec.EncryptionSpec,
    )
    model_source_info: "ModelSourceInfo" = proto.Field(
        proto.MESSAGE,
        number=38,
        message="ModelSourceInfo",
    )
    original_model_info: OriginalModelInfo = proto.Field(
        proto.MESSAGE,
        number=34,
        message=OriginalModelInfo,
    )
    metadata_artifact: str = proto.Field(
        proto.STRING,
        number=44,
    )
    base_model_source: BaseModelSource = proto.Field(
        proto.MESSAGE,
        number=50,
        message=BaseModelSource,
    )


class LargeModelReference(proto.Message):
    r"""Contains information about the Large Model.

    Attributes:
        name (str):
            Required. The unique name of the large
            Foundation or pre-built model. Like
            "chat-bison", "text-bison". Or model name with
            version ID, like "chat-bison@001",
            "text-bison@005", etc.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ModelGardenSource(proto.Message):
    r"""Contains information about the source of the models generated
    from Model Garden.

    Attributes:
        public_model_name (str):
            Required. The model garden source model
            resource name.
    """

    public_model_name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class GenieSource(proto.Message):
    r"""Contains information about the source of the models generated
    from Generative AI Studio.

    Attributes:
        base_model_uri (str):
            Required. The public base model URI.
    """

    base_model_uri: str = proto.Field(
        proto.STRING,
        number=1,
    )


class PredictSchemata(proto.Message):
    r"""Contains the schemata used in Model's predictions and explanations
    via
    [PredictionService.Predict][google.cloud.aiplatform.v1beta1.PredictionService.Predict],
    [PredictionService.Explain][google.cloud.aiplatform.v1beta1.PredictionService.Explain]
    and
    [BatchPredictionJob][google.cloud.aiplatform.v1beta1.BatchPredictionJob].

    Attributes:
        instance_schema_uri (str):
            Immutable. Points to a YAML file stored on Google Cloud
            Storage describing the format of a single instance, which
            are used in
            [PredictRequest.instances][google.cloud.aiplatform.v1beta1.PredictRequest.instances],
            [ExplainRequest.instances][google.cloud.aiplatform.v1beta1.ExplainRequest.instances]
            and
            [BatchPredictionJob.input_config][google.cloud.aiplatform.v1beta1.BatchPredictionJob.input_config].
            The schema is defined as an OpenAPI 3.0.2 `Schema
            Object <https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#schemaObject>`__.
            AutoML Models always have this field populated by Vertex AI.
            Note: The URI given on output will be immutable and probably
            different, including the URI scheme, than the one given on
            input. The output URI will point to a location where the
            user only has a read access.
        parameters_schema_uri (str):
            Immutable. Points to a YAML file stored on Google Cloud
            Storage describing the parameters of prediction and
            explanation via
            [PredictRequest.parameters][google.cloud.aiplatform.v1beta1.PredictRequest.parameters],
            [ExplainRequest.parameters][google.cloud.aiplatform.v1beta1.ExplainRequest.parameters]
            and
            [BatchPredictionJob.model_parameters][google.cloud.aiplatform.v1beta1.BatchPredictionJob.model_parameters].
            The schema is defined as an OpenAPI 3.0.2 `Schema
            Object <https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#schemaObject>`__.
            AutoML Models always have this field populated by Vertex AI,
            if no parameters are supported, then it is set to an empty
            string. Note: The URI given on output will be immutable and
            probably different, including the URI scheme, than the one
            given on input. The output URI will point to a location
            where the user only has a read access.
        prediction_schema_uri (str):
            Immutable. Points to a YAML file stored on Google Cloud
            Storage describing the format of a single prediction
            produced by this Model, which are returned via
            [PredictResponse.predictions][google.cloud.aiplatform.v1beta1.PredictResponse.predictions],
            [ExplainResponse.explanations][google.cloud.aiplatform.v1beta1.ExplainResponse.explanations],
            and
            [BatchPredictionJob.output_config][google.cloud.aiplatform.v1beta1.BatchPredictionJob.output_config].
            The schema is defined as an OpenAPI 3.0.2 `Schema
            Object <https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#schemaObject>`__.
            AutoML Models always have this field populated by Vertex AI.
            Note: The URI given on output will be immutable and probably
            different, including the URI scheme, than the one given on
            input. The output URI will point to a location where the
            user only has a read access.
    """

    instance_schema_uri: str = proto.Field(
        proto.STRING,
        number=1,
    )
    parameters_schema_uri: str = proto.Field(
        proto.STRING,
        number=2,
    )
    prediction_schema_uri: str = proto.Field(
        proto.STRING,
        number=3,
    )


class ModelContainerSpec(proto.Message):
    r"""Specification of a container for serving predictions. Some fields in
    this message correspond to fields in the `Kubernetes Container v1
    core
    specification <https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.23/#container-v1-core>`__.

    Attributes:
        image_uri (str):
            Required. Immutable. URI of the Docker image to be used as
            the custom container for serving predictions. This URI must
            identify an image in Artifact Registry or Container
            Registry. Learn more about the `container publishing
            requirements <https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#publishing>`__,
            including permissions requirements for the Vertex AI Service
            Agent.

            The container image is ingested upon
            [ModelService.UploadModel][google.cloud.aiplatform.v1beta1.ModelService.UploadModel],
            stored internally, and this original path is afterwards not
            used.

            To learn about the requirements for the Docker image itself,
            see `Custom container
            requirements <https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#>`__.

            You can use the URI to one of Vertex AI's `pre-built
            container images for
            prediction <https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers>`__
            in this field.
        command (MutableSequence[str]):
            Immutable. Specifies the command that runs when the
            container starts. This overrides the container's
            `ENTRYPOINT <https://docs.docker.com/engine/reference/builder/#entrypoint>`__.
            Specify this field as an array of executable and arguments,
            similar to a Docker ``ENTRYPOINT``'s "exec" form, not its
            "shell" form.

            If you do not specify this field, then the container's
            ``ENTRYPOINT`` runs, in conjunction with the
            [args][google.cloud.aiplatform.v1beta1.ModelContainerSpec.args]
            field or the container's
            ```CMD`` <https://docs.docker.com/engine/reference/builder/#cmd>`__,
            if either exists. If this field is not specified and the
            container does not have an ``ENTRYPOINT``, then refer to the
            Docker documentation about `how ``CMD`` and ``ENTRYPOINT``
            interact <https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact>`__.

            If you specify this field, then you can also specify the
            ``args`` field to provide additional arguments for this
            command. However, if you specify this field, then the
            container's ``CMD`` is ignored. See the `Kubernetes
            documentation about how the ``command`` and ``args`` fields
            interact with a container's ``ENTRYPOINT`` and
            ``CMD`` <https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/#notes>`__.

            In this field, you can reference `environment variables set
            by Vertex
            AI <https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables>`__
            and environment variables set in the
            [env][google.cloud.aiplatform.v1beta1.ModelContainerSpec.env]
            field. You cannot reference environment variables set in the
            Docker image. In order for environment variables to be
            expanded, reference them by using the following syntax:
            $(VARIABLE_NAME) Note that this differs from Bash variable
            expansion, which does not use parentheses. If a variable
            cannot be resolved, the reference in the input string is
            used unchanged. To avoid variable expansion, you can escape
            this syntax with ``$$``; for example: $$(VARIABLE_NAME) This
            field corresponds to the ``command`` field of the Kubernetes
            Containers `v1 core
            API <https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.23/#container-v1-core>`__.
        args (MutableSequence[str]):
            Immutable. Specifies arguments for the command that runs
            when the container starts. This overrides the container's
            ```CMD`` <https://docs.docker.com/engine/reference/builder/#cmd>`__.
            Specify this field as an array of executable and arguments,
            similar to a Docker ``CMD``'s "default parameters" form.

            If you don't specify this field but do specify the
            [command][google.cloud.aiplatform.v1beta1.ModelContainerSpec.command]
            field, then the command from the ``command`` field runs
            without any additional arguments. See the `Kubernetes
            documentation about how the ``command`` and ``args`` fields
            interact with a container's ``ENTRYPOINT`` and
            ``CMD`` <https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/#notes>`__.

            If you don't specify this field and don't specify the
            ``command`` field, then the container's
            ```ENTRYPOINT`` <https://docs.docker.com/engine/reference/builder/#cmd>`__
            and ``CMD`` determine what runs based on their default
            behavior. See the Docker documentation about `how ``CMD``
            and ``ENTRYPOINT``
            interact <https://docs.docker.com/engine/reference/builder/#understand-how-cmd-and-entrypoint-interact>`__.

            In this field, you can reference `environment variables set
            by Vertex
            AI <https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables>`__
            and environment variables set in the
            [env][google.cloud.aiplatform.v1beta1.ModelContainerSpec.env]
            field. You cannot reference environment variables set in the
            Docker image. In order for environment variables to be
            expanded, reference them by using the following syntax:
            $(VARIABLE_NAME) Note that this differs from Bash variable
            expansion, which does not use parentheses. If a variable
            cannot be resolved, the reference in the input string is
            used unchanged. To avoid variable expansion, you can escape
            this syntax with ``$$``; for example: $$(VARIABLE_NAME) This
            field corresponds to the ``args`` field of the Kubernetes
            Containers `v1 core
            API <https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.23/#container-v1-core>`__.
        env (MutableSequence[google.cloud.aiplatform_v1beta1.types.EnvVar]):
            Immutable. List of environment variables to set in the
            container. After the container starts running, code running
            in the container can read these environment variables.

            Additionally, the
            [command][google.cloud.aiplatform.v1beta1.ModelContainerSpec.command]
            and
            [args][google.cloud.aiplatform.v1beta1.ModelContainerSpec.args]
            fields can reference these variables. Later entries in this
            list can also reference earlier entries. For example, the
            following example sets the variable ``VAR_2`` to have the
            value ``foo bar``:

            .. code:: json

               [
                 {
                   "name": "VAR_1",
                   "value": "foo"
                 },
                 {
                   "name": "VAR_2",
                   "value": "$(VAR_1) bar"
                 }
               ]

            If you switch the order of the variables in the example,
            then the expansion does not occur.

            This field corresponds to the ``env`` field of the
            Kubernetes Containers `v1 core
            API <https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.23/#container-v1-core>`__.
        ports (MutableSequence[google.cloud.aiplatform_v1beta1.types.Port]):
            Immutable. List of ports to expose from the container.
            Vertex AI sends any prediction requests that it receives to
            the first port on this list. Vertex AI also sends `liveness
            and health
            checks <https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#liveness>`__
            to this port.

            If you do not specify this field, it defaults to following
            value:

            .. code:: json

               [
                 {
                   "containerPort": 8080
                 }
               ]

            Vertex AI does not use ports other than the first one
            listed. This field corresponds to the ``ports`` field of the
            Kubernetes Containers `v1 core
            API <https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.23/#container-v1-core>`__.
        predict_route (str):
            Immutable. HTTP path on the container to send prediction
            requests to. Vertex AI forwards requests sent using
            [projects.locations.endpoints.predict][google.cloud.aiplatform.v1beta1.PredictionService.Predict]
            to this path on the container's IP address and port. Vertex
            AI then returns the container's response in the API
            response.

            For example, if you set this field to ``/foo``, then when
            Vertex AI receives a prediction request, it forwards the
            request body in a POST request to the ``/foo`` path on the
            port of your container specified by the first value of this
            ``ModelContainerSpec``'s
            [ports][google.cloud.aiplatform.v1beta1.ModelContainerSpec.ports]
            field.

            If you don't specify this field, it defaults to the
            following value when you [deploy this Model to an
            Endpoint][google.cloud.aiplatform.v1beta1.EndpointService.DeployModel]:
            /v1/endpoints/ENDPOINT/deployedModels/DEPLOYED_MODEL:predict
            The placeholders in this value are replaced as follows:

            -  ENDPOINT: The last segment (following ``endpoints/``)of
               the Endpoint.name][] field of the Endpoint where this
               Model has been deployed. (Vertex AI makes this value
               available to your container code as the
               ```AIP_ENDPOINT_ID`` environment
               variable <https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables>`__.)

            -  DEPLOYED_MODEL:
               [DeployedModel.id][google.cloud.aiplatform.v1beta1.DeployedModel.id]
               of the ``DeployedModel``. (Vertex AI makes this value
               available to your container code as the
               ```AIP_DEPLOYED_MODEL_ID`` environment
               variable <https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables>`__.)
        health_route (str):
            Immutable. HTTP path on the container to send health checks
            to. Vertex AI intermittently sends GET requests to this path
            on the container's IP address and port to check that the
            container is healthy. Read more about `health
            checks <https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#health>`__.

            For example, if you set this field to ``/bar``, then Vertex
            AI intermittently sends a GET request to the ``/bar`` path
            on the port of your container specified by the first value
            of this ``ModelContainerSpec``'s
            [ports][google.cloud.aiplatform.v1beta1.ModelContainerSpec.ports]
            field.

            If you don't specify this field, it defaults to the
            following value when you [deploy this Model to an
            Endpoint][google.cloud.aiplatform.v1beta1.EndpointService.DeployModel]:
            /v1/endpoints/ENDPOINT/deployedModels/DEPLOYED_MODEL:predict
            The placeholders in this value are replaced as follows:

            -  ENDPOINT: The last segment (following ``endpoints/``)of
               the Endpoint.name][] field of the Endpoint where this
               Model has been deployed. (Vertex AI makes this value
               available to your container code as the
               ```AIP_ENDPOINT_ID`` environment
               variable <https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables>`__.)

            -  DEPLOYED_MODEL:
               [DeployedModel.id][google.cloud.aiplatform.v1beta1.DeployedModel.id]
               of the ``DeployedModel``. (Vertex AI makes this value
               available to your container code as the
               ```AIP_DEPLOYED_MODEL_ID`` environment
               variable <https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables>`__.)
        grpc_ports (MutableSequence[google.cloud.aiplatform_v1beta1.types.Port]):
            Immutable. List of ports to expose from the container.
            Vertex AI sends gRPC prediction requests that it receives to
            the first port on this list. Vertex AI also sends liveness
            and health checks to this port.

            If you do not specify this field, gRPC requests to the
            container will be disabled.

            Vertex AI does not use ports other than the first one
            listed. This field corresponds to the ``ports`` field of the
            Kubernetes Containers v1 core API.
        deployment_timeout (google.protobuf.duration_pb2.Duration):
            Immutable. Deployment timeout.
            Limit for deployment timeout is 2 hours.
        shared_memory_size_mb (int):
            Immutable. The amount of the VM memory to
            reserve as the shared memory for the model in
            megabytes.
        startup_probe (google.cloud.aiplatform_v1beta1.types.Probe):
            Immutable. Specification for Kubernetes
            startup probe.
        health_probe (google.cloud.aiplatform_v1beta1.types.Probe):
            Immutable. Specification for Kubernetes
            readiness probe.
    """

    image_uri: str = proto.Field(
        proto.STRING,
        number=1,
    )
    command: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )
    args: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=3,
    )
    env: MutableSequence[env_var.EnvVar] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=env_var.EnvVar,
    )
    ports: MutableSequence["Port"] = proto.RepeatedField(
        proto.MESSAGE,
        number=5,
        message="Port",
    )
    predict_route: str = proto.Field(
        proto.STRING,
        number=6,
    )
    health_route: str = proto.Field(
        proto.STRING,
        number=7,
    )
    grpc_ports: MutableSequence["Port"] = proto.RepeatedField(
        proto.MESSAGE,
        number=9,
        message="Port",
    )
    deployment_timeout: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=10,
        message=duration_pb2.Duration,
    )
    shared_memory_size_mb: int = proto.Field(
        proto.INT64,
        number=11,
    )
    startup_probe: "Probe" = proto.Field(
        proto.MESSAGE,
        number=12,
        message="Probe",
    )
    health_probe: "Probe" = proto.Field(
        proto.MESSAGE,
        number=13,
        message="Probe",
    )


class Port(proto.Message):
    r"""Represents a network port in a container.

    Attributes:
        container_port (int):
            The number of the port to expose on the pod's
            IP address. Must be a valid port number, between
            1 and 65535 inclusive.
    """

    container_port: int = proto.Field(
        proto.INT32,
        number=3,
    )


class ModelSourceInfo(proto.Message):
    r"""Detail description of the source information of the model.

    Attributes:
        source_type (google.cloud.aiplatform_v1beta1.types.ModelSourceInfo.ModelSourceType):
            Type of the model source.
        copy (bool):
            If this Model is copy of another Model. If true then
            [source_type][google.cloud.aiplatform.v1beta1.ModelSourceInfo.source_type]
            pertains to the original.
    """

    class ModelSourceType(proto.Enum):
        r"""Source of the model. Different from ``objective`` field, this
        ``ModelSourceType`` enum indicates the source from which the model
        was accessed or obtained, whereas the ``objective`` indicates the
        overall aim or function of this model.

        Values:
            MODEL_SOURCE_TYPE_UNSPECIFIED (0):
                Should not be used.
            AUTOML (1):
                The Model is uploaded by automl training
                pipeline.
            CUSTOM (2):
                The Model is uploaded by user or custom
                training pipeline.
            BQML (3):
                The Model is registered and sync'ed from
                BigQuery ML.
            MODEL_GARDEN (4):
                The Model is saved or tuned from Model
                Garden.
            GENIE (5):
                The Model is saved or tuned from Genie.
            CUSTOM_TEXT_EMBEDDING (6):
                The Model is uploaded by text embedding
                finetuning pipeline.
            MARKETPLACE (7):
                The Model is saved or tuned from Marketplace.
        """
        MODEL_SOURCE_TYPE_UNSPECIFIED = 0
        AUTOML = 1
        CUSTOM = 2
        BQML = 3
        MODEL_GARDEN = 4
        GENIE = 5
        CUSTOM_TEXT_EMBEDDING = 6
        MARKETPLACE = 7

    source_type: ModelSourceType = proto.Field(
        proto.ENUM,
        number=1,
        enum=ModelSourceType,
    )
    copy: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class Probe(proto.Message):
    r"""Probe describes a health check to be performed against a
    container to determine whether it is alive or ready to receive
    traffic.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        exec_ (google.cloud.aiplatform_v1beta1.types.Probe.ExecAction):
            Exec specifies the action to take.

            This field is a member of `oneof`_ ``probe_type``.
        period_seconds (int):
            How often (in seconds) to perform the probe. Default to 10
            seconds. Minimum value is 1. Must be less than
            timeout_seconds.

            Maps to Kubernetes probe argument 'periodSeconds'.
        timeout_seconds (int):
            Number of seconds after which the probe times out. Defaults
            to 1 second. Minimum value is 1. Must be greater or equal to
            period_seconds.

            Maps to Kubernetes probe argument 'timeoutSeconds'.
    """

    class ExecAction(proto.Message):
        r"""ExecAction specifies a command to execute.

        Attributes:
            command (MutableSequence[str]):
                Command is the command line to execute inside the container,
                the working directory for the command is root ('/') in the
                container's filesystem. The command is simply exec'd, it is
                not run inside a shell, so traditional shell instructions
                ('|', etc) won't work. To use a shell, you need to
                explicitly call out to that shell. Exit status of 0 is
                treated as live/healthy and non-zero is unhealthy.
        """

        command: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=1,
        )

    exec_: ExecAction = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="probe_type",
        message=ExecAction,
    )
    period_seconds: int = proto.Field(
        proto.INT32,
        number=2,
    )
    timeout_seconds: int = proto.Field(
        proto.INT32,
        number=3,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
