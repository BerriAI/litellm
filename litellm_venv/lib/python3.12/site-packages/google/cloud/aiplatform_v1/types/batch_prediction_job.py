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

from google.cloud.aiplatform_v1.types import completion_stats as gca_completion_stats
from google.cloud.aiplatform_v1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1.types import explanation
from google.cloud.aiplatform_v1.types import io
from google.cloud.aiplatform_v1.types import job_state
from google.cloud.aiplatform_v1.types import machine_resources
from google.cloud.aiplatform_v1.types import (
    manual_batch_tuning_parameters as gca_manual_batch_tuning_parameters,
)
from google.cloud.aiplatform_v1.types import (
    unmanaged_container_model as gca_unmanaged_container_model,
)
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "BatchPredictionJob",
    },
)


class BatchPredictionJob(proto.Message):
    r"""A job that uses a
    [Model][google.cloud.aiplatform.v1.BatchPredictionJob.model] to
    produce predictions on multiple [input
    instances][google.cloud.aiplatform.v1.BatchPredictionJob.input_config].
    If predictions for significant portion of the instances fail, the
    job may finish without attempting predictions for all remaining
    instances.

    Attributes:
        name (str):
            Output only. Resource name of the
            BatchPredictionJob.
        display_name (str):
            Required. The user-defined name of this
            BatchPredictionJob.
        model (str):
            The name of the Model resource that produces the predictions
            via this job, must share the same ancestor Location.
            Starting this job has no impact on any existing deployments
            of the Model and their resources. Exactly one of model and
            unmanaged_container_model must be set.

            The model resource name may contain version id or version
            alias to specify the version. Example:
            ``projects/{project}/locations/{location}/models/{model}@2``
            or
            ``projects/{project}/locations/{location}/models/{model}@golden``
            if no version is specified, the default version will be
            deployed.

            The model resource could also be a publisher model. Example:
            ``publishers/{publisher}/models/{model}`` or
            ``projects/{project}/locations/{location}/publishers/{publisher}/models/{model}``
        model_version_id (str):
            Output only. The version ID of the Model that
            produces the predictions via this job.
        unmanaged_container_model (google.cloud.aiplatform_v1.types.UnmanagedContainerModel):
            Contains model information necessary to perform batch
            prediction without requiring uploading to model registry.
            Exactly one of model and unmanaged_container_model must be
            set.
        input_config (google.cloud.aiplatform_v1.types.BatchPredictionJob.InputConfig):
            Required. Input configuration of the instances on which
            predictions are performed. The schema of any single instance
            may be specified via the
            [Model's][google.cloud.aiplatform.v1.BatchPredictionJob.model]
            [PredictSchemata's][google.cloud.aiplatform.v1.Model.predict_schemata]
            [instance_schema_uri][google.cloud.aiplatform.v1.PredictSchemata.instance_schema_uri].
        instance_config (google.cloud.aiplatform_v1.types.BatchPredictionJob.InstanceConfig):
            Configuration for how to convert batch
            prediction input instances to the prediction
            instances that are sent to the Model.
        model_parameters (google.protobuf.struct_pb2.Value):
            The parameters that govern the predictions. The schema of
            the parameters may be specified via the
            [Model's][google.cloud.aiplatform.v1.BatchPredictionJob.model]
            [PredictSchemata's][google.cloud.aiplatform.v1.Model.predict_schemata]
            [parameters_schema_uri][google.cloud.aiplatform.v1.PredictSchemata.parameters_schema_uri].
        output_config (google.cloud.aiplatform_v1.types.BatchPredictionJob.OutputConfig):
            Required. The Configuration specifying where output
            predictions should be written. The schema of any single
            prediction may be specified as a concatenation of
            [Model's][google.cloud.aiplatform.v1.BatchPredictionJob.model]
            [PredictSchemata's][google.cloud.aiplatform.v1.Model.predict_schemata]
            [instance_schema_uri][google.cloud.aiplatform.v1.PredictSchemata.instance_schema_uri]
            and
            [prediction_schema_uri][google.cloud.aiplatform.v1.PredictSchemata.prediction_schema_uri].
        dedicated_resources (google.cloud.aiplatform_v1.types.BatchDedicatedResources):
            The config of resources used by the Model during the batch
            prediction. If the Model
            [supports][google.cloud.aiplatform.v1.Model.supported_deployment_resources_types]
            DEDICATED_RESOURCES this config may be provided (and the job
            will use these resources), if the Model doesn't support
            AUTOMATIC_RESOURCES, this config must be provided.
        service_account (str):
            The service account that the DeployedModel's container runs
            as. If not specified, a system generated one will be used,
            which has minimal permissions and the custom container, if
            used, may not have enough permission to access other Google
            Cloud resources.

            Users deploying the Model must have the
            ``iam.serviceAccounts.actAs`` permission on this service
            account.
        manual_batch_tuning_parameters (google.cloud.aiplatform_v1.types.ManualBatchTuningParameters):
            Immutable. Parameters configuring the batch behavior.
            Currently only applicable when
            [dedicated_resources][google.cloud.aiplatform.v1.BatchPredictionJob.dedicated_resources]
            are used (in other cases Vertex AI does the tuning itself).
        generate_explanation (bool):
            Generate explanation with the batch prediction results.

            When set to ``true``, the batch prediction output changes
            based on the ``predictions_format`` field of the
            [BatchPredictionJob.output_config][google.cloud.aiplatform.v1.BatchPredictionJob.output_config]
            object:

            -  ``bigquery``: output includes a column named
               ``explanation``. The value is a struct that conforms to
               the [Explanation][google.cloud.aiplatform.v1.Explanation]
               object.
            -  ``jsonl``: The JSON objects on each line include an
               additional entry keyed ``explanation``. The value of the
               entry is a JSON object that conforms to the
               [Explanation][google.cloud.aiplatform.v1.Explanation]
               object.
            -  ``csv``: Generating explanations for CSV format is not
               supported.

            If this field is set to true, either the
            [Model.explanation_spec][google.cloud.aiplatform.v1.Model.explanation_spec]
            or
            [explanation_spec][google.cloud.aiplatform.v1.BatchPredictionJob.explanation_spec]
            must be populated.
        explanation_spec (google.cloud.aiplatform_v1.types.ExplanationSpec):
            Explanation configuration for this BatchPredictionJob. Can
            be specified only if
            [generate_explanation][google.cloud.aiplatform.v1.BatchPredictionJob.generate_explanation]
            is set to ``true``.

            This value overrides the value of
            [Model.explanation_spec][google.cloud.aiplatform.v1.Model.explanation_spec].
            All fields of
            [explanation_spec][google.cloud.aiplatform.v1.BatchPredictionJob.explanation_spec]
            are optional in the request. If a field of the
            [explanation_spec][google.cloud.aiplatform.v1.BatchPredictionJob.explanation_spec]
            object is not populated, the corresponding field of the
            [Model.explanation_spec][google.cloud.aiplatform.v1.Model.explanation_spec]
            object is inherited.
        output_info (google.cloud.aiplatform_v1.types.BatchPredictionJob.OutputInfo):
            Output only. Information further describing
            the output of this job.
        state (google.cloud.aiplatform_v1.types.JobState):
            Output only. The detailed state of the job.
        error (google.rpc.status_pb2.Status):
            Output only. Only populated when the job's state is
            JOB_STATE_FAILED or JOB_STATE_CANCELLED.
        partial_failures (MutableSequence[google.rpc.status_pb2.Status]):
            Output only. Partial failures encountered.
            For example, single files that can't be read.
            This field never exceeds 20 entries.
            Status details fields contain standard Google
            Cloud error details.
        resources_consumed (google.cloud.aiplatform_v1.types.ResourcesConsumed):
            Output only. Information about resources that
            had been consumed by this job. Provided in real
            time at best effort basis, as well as a final
            value once the job completes.

            Note: This field currently may be not populated
            for batch predictions that use AutoML Models.
        completion_stats (google.cloud.aiplatform_v1.types.CompletionStats):
            Output only. Statistics on completed and
            failed prediction instances.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the BatchPredictionJob
            was created.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the BatchPredictionJob for the first
            time entered the ``JOB_STATE_RUNNING`` state.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the BatchPredictionJob entered any of
            the following states: ``JOB_STATE_SUCCEEDED``,
            ``JOB_STATE_FAILED``, ``JOB_STATE_CANCELLED``.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the BatchPredictionJob
            was most recently updated.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to
            organize BatchPredictionJobs.
            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            and examples of labels.
        encryption_spec (google.cloud.aiplatform_v1.types.EncryptionSpec):
            Customer-managed encryption key options for a
            BatchPredictionJob. If this is set, then all
            resources created by the BatchPredictionJob will
            be encrypted with the provided encryption key.
        disable_container_logging (bool):
            For custom-trained Models and AutoML Tabular Models, the
            container of the DeployedModel instances will send
            ``stderr`` and ``stdout`` streams to Cloud Logging by
            default. Please note that the logs incur cost, which are
            subject to `Cloud Logging
            pricing <https://cloud.google.com/logging/pricing>`__.

            User can disable container logging by setting this flag to
            true.
    """

    class InputConfig(proto.Message):
        r"""Configures the input to
        [BatchPredictionJob][google.cloud.aiplatform.v1.BatchPredictionJob].
        See
        [Model.supported_input_storage_formats][google.cloud.aiplatform.v1.Model.supported_input_storage_formats]
        for Model's supported input formats, and how instances should be
        expressed via any of them.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            gcs_source (google.cloud.aiplatform_v1.types.GcsSource):
                The Cloud Storage location for the input
                instances.

                This field is a member of `oneof`_ ``source``.
            bigquery_source (google.cloud.aiplatform_v1.types.BigQuerySource):
                The BigQuery location of the input table.
                The schema of the table should be in the format
                described by the given context OpenAPI Schema,
                if one is provided. The table may contain
                additional columns that are not described by the
                schema, and they will be ignored.

                This field is a member of `oneof`_ ``source``.
            instances_format (str):
                Required. The format in which instances are given, must be
                one of the
                [Model's][google.cloud.aiplatform.v1.BatchPredictionJob.model]
                [supported_input_storage_formats][google.cloud.aiplatform.v1.Model.supported_input_storage_formats].
        """

        gcs_source: io.GcsSource = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="source",
            message=io.GcsSource,
        )
        bigquery_source: io.BigQuerySource = proto.Field(
            proto.MESSAGE,
            number=3,
            oneof="source",
            message=io.BigQuerySource,
        )
        instances_format: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class InstanceConfig(proto.Message):
        r"""Configuration defining how to transform batch prediction
        input instances to the instances that the Model accepts.

        Attributes:
            instance_type (str):
                The format of the instance that the Model accepts. Vertex AI
                will convert compatible [batch prediction input instance
                formats][google.cloud.aiplatform.v1.BatchPredictionJob.InputConfig.instances_format]
                to the specified format.

                Supported values are:

                -  ``object``: Each input is converted to JSON object
                   format.

                   -  For ``bigquery``, each row is converted to an object.
                   -  For ``jsonl``, each line of the JSONL input must be an
                      object.
                   -  Does not apply to ``csv``, ``file-list``,
                      ``tf-record``, or ``tf-record-gzip``.

                -  ``array``: Each input is converted to JSON array format.

                   -  For ``bigquery``, each row is converted to an array.
                      The order of columns is determined by the BigQuery
                      column order, unless
                      [included_fields][google.cloud.aiplatform.v1.BatchPredictionJob.InstanceConfig.included_fields]
                      is populated.
                      [included_fields][google.cloud.aiplatform.v1.BatchPredictionJob.InstanceConfig.included_fields]
                      must be populated for specifying field orders.
                   -  For ``jsonl``, if each line of the JSONL input is an
                      object,
                      [included_fields][google.cloud.aiplatform.v1.BatchPredictionJob.InstanceConfig.included_fields]
                      must be populated for specifying field orders.
                   -  Does not apply to ``csv``, ``file-list``,
                      ``tf-record``, or ``tf-record-gzip``.

                If not specified, Vertex AI converts the batch prediction
                input as follows:

                -  For ``bigquery`` and ``csv``, the behavior is the same as
                   ``array``. The order of columns is the same as defined in
                   the file or table, unless
                   [included_fields][google.cloud.aiplatform.v1.BatchPredictionJob.InstanceConfig.included_fields]
                   is populated.
                -  For ``jsonl``, the prediction instance format is
                   determined by each line of the input.
                -  For ``tf-record``/``tf-record-gzip``, each record will be
                   converted to an object in the format of
                   ``{"b64": <value>}``, where ``<value>`` is the
                   Base64-encoded string of the content of the record.
                -  For ``file-list``, each file in the list will be
                   converted to an object in the format of
                   ``{"b64": <value>}``, where ``<value>`` is the
                   Base64-encoded string of the content of the file.
            key_field (str):
                The name of the field that is considered as a key.

                The values identified by the key field is not included in
                the transformed instances that is sent to the Model. This is
                similar to specifying this name of the field in
                [excluded_fields][google.cloud.aiplatform.v1.BatchPredictionJob.InstanceConfig.excluded_fields].
                In addition, the batch prediction output will not include
                the instances. Instead the output will only include the
                value of the key field, in a field named ``key`` in the
                output:

                -  For ``jsonl`` output format, the output will have a
                   ``key`` field instead of the ``instance`` field.
                -  For ``csv``/``bigquery`` output format, the output will
                   have have a ``key`` column instead of the instance
                   feature columns.

                The input must be JSONL with objects at each line, CSV,
                BigQuery or TfRecord.
            included_fields (MutableSequence[str]):
                Fields that will be included in the prediction instance that
                is sent to the Model.

                If
                [instance_type][google.cloud.aiplatform.v1.BatchPredictionJob.InstanceConfig.instance_type]
                is ``array``, the order of field names in included_fields
                also determines the order of the values in the array.

                When included_fields is populated,
                [excluded_fields][google.cloud.aiplatform.v1.BatchPredictionJob.InstanceConfig.excluded_fields]
                must be empty.

                The input must be JSONL with objects at each line, BigQuery
                or TfRecord.
            excluded_fields (MutableSequence[str]):
                Fields that will be excluded in the prediction instance that
                is sent to the Model.

                Excluded will be attached to the batch prediction output if
                [key_field][google.cloud.aiplatform.v1.BatchPredictionJob.InstanceConfig.key_field]
                is not specified.

                When excluded_fields is populated,
                [included_fields][google.cloud.aiplatform.v1.BatchPredictionJob.InstanceConfig.included_fields]
                must be empty.

                The input must be JSONL with objects at each line, BigQuery
                or TfRecord.
        """

        instance_type: str = proto.Field(
            proto.STRING,
            number=1,
        )
        key_field: str = proto.Field(
            proto.STRING,
            number=2,
        )
        included_fields: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=3,
        )
        excluded_fields: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=4,
        )

    class OutputConfig(proto.Message):
        r"""Configures the output of
        [BatchPredictionJob][google.cloud.aiplatform.v1.BatchPredictionJob].
        See
        [Model.supported_output_storage_formats][google.cloud.aiplatform.v1.Model.supported_output_storage_formats]
        for supported output formats, and how predictions are expressed via
        any of them.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            gcs_destination (google.cloud.aiplatform_v1.types.GcsDestination):
                The Cloud Storage location of the directory where the output
                is to be written to. In the given directory a new directory
                is created. Its name is
                ``prediction-<model-display-name>-<job-create-time>``, where
                timestamp is in YYYY-MM-DDThh:mm:ss.sssZ ISO-8601 format.
                Inside of it files ``predictions_0001.<extension>``,
                ``predictions_0002.<extension>``, ...,
                ``predictions_N.<extension>`` are created where
                ``<extension>`` depends on chosen
                [predictions_format][google.cloud.aiplatform.v1.BatchPredictionJob.OutputConfig.predictions_format],
                and N may equal 0001 and depends on the total number of
                successfully predicted instances. If the Model has both
                [instance][google.cloud.aiplatform.v1.PredictSchemata.instance_schema_uri]
                and
                [prediction][google.cloud.aiplatform.v1.PredictSchemata.parameters_schema_uri]
                schemata defined then each such file contains predictions as
                per the
                [predictions_format][google.cloud.aiplatform.v1.BatchPredictionJob.OutputConfig.predictions_format].
                If prediction for any instance failed (partially or
                completely), then an additional ``errors_0001.<extension>``,
                ``errors_0002.<extension>``,..., ``errors_N.<extension>``
                files are created (N depends on total number of failed
                predictions). These files contain the failed instances, as
                per their schema, followed by an additional ``error`` field
                which as value has [google.rpc.Status][google.rpc.Status]
                containing only ``code`` and ``message`` fields.

                This field is a member of `oneof`_ ``destination``.
            bigquery_destination (google.cloud.aiplatform_v1.types.BigQueryDestination):
                The BigQuery project or dataset location where the output is
                to be written to. If project is provided, a new dataset is
                created with name
                ``prediction_<model-display-name>_<job-create-time>`` where
                is made BigQuery-dataset-name compatible (for example, most
                special characters become underscores), and timestamp is in
                YYYY_MM_DDThh_mm_ss_sssZ "based on ISO-8601" format. In the
                dataset two tables will be created, ``predictions``, and
                ``errors``. If the Model has both
                [instance][google.cloud.aiplatform.v1.PredictSchemata.instance_schema_uri]
                and
                [prediction][google.cloud.aiplatform.v1.PredictSchemata.parameters_schema_uri]
                schemata defined then the tables have columns as follows:
                The ``predictions`` table contains instances for which the
                prediction succeeded, it has columns as per a concatenation
                of the Model's instance and prediction schemata. The
                ``errors`` table contains rows for which the prediction has
                failed, it has instance columns, as per the instance schema,
                followed by a single "errors" column, which as values has
                [google.rpc.Status][google.rpc.Status] represented as a
                STRUCT, and containing only ``code`` and ``message``.

                This field is a member of `oneof`_ ``destination``.
            predictions_format (str):
                Required. The format in which Vertex AI gives the
                predictions, must be one of the
                [Model's][google.cloud.aiplatform.v1.BatchPredictionJob.model]
                [supported_output_storage_formats][google.cloud.aiplatform.v1.Model.supported_output_storage_formats].
        """

        gcs_destination: io.GcsDestination = proto.Field(
            proto.MESSAGE,
            number=2,
            oneof="destination",
            message=io.GcsDestination,
        )
        bigquery_destination: io.BigQueryDestination = proto.Field(
            proto.MESSAGE,
            number=3,
            oneof="destination",
            message=io.BigQueryDestination,
        )
        predictions_format: str = proto.Field(
            proto.STRING,
            number=1,
        )

    class OutputInfo(proto.Message):
        r"""Further describes this job's output. Supplements
        [output_config][google.cloud.aiplatform.v1.BatchPredictionJob.output_config].

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            gcs_output_directory (str):
                Output only. The full path of the Cloud
                Storage directory created, into which the
                prediction output is written.

                This field is a member of `oneof`_ ``output_location``.
            bigquery_output_dataset (str):
                Output only. The path of the BigQuery dataset created, in
                ``bq://projectId.bqDatasetId`` format, into which the
                prediction output is written.

                This field is a member of `oneof`_ ``output_location``.
            bigquery_output_table (str):
                Output only. The name of the BigQuery table created, in
                ``predictions_<timestamp>`` format, into which the
                prediction output is written. Can be used by UI to generate
                the BigQuery output path, for example.
        """

        gcs_output_directory: str = proto.Field(
            proto.STRING,
            number=1,
            oneof="output_location",
        )
        bigquery_output_dataset: str = proto.Field(
            proto.STRING,
            number=2,
            oneof="output_location",
        )
        bigquery_output_table: str = proto.Field(
            proto.STRING,
            number=4,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    model: str = proto.Field(
        proto.STRING,
        number=3,
    )
    model_version_id: str = proto.Field(
        proto.STRING,
        number=30,
    )
    unmanaged_container_model: gca_unmanaged_container_model.UnmanagedContainerModel = (
        proto.Field(
            proto.MESSAGE,
            number=28,
            message=gca_unmanaged_container_model.UnmanagedContainerModel,
        )
    )
    input_config: InputConfig = proto.Field(
        proto.MESSAGE,
        number=4,
        message=InputConfig,
    )
    instance_config: InstanceConfig = proto.Field(
        proto.MESSAGE,
        number=27,
        message=InstanceConfig,
    )
    model_parameters: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=5,
        message=struct_pb2.Value,
    )
    output_config: OutputConfig = proto.Field(
        proto.MESSAGE,
        number=6,
        message=OutputConfig,
    )
    dedicated_resources: machine_resources.BatchDedicatedResources = proto.Field(
        proto.MESSAGE,
        number=7,
        message=machine_resources.BatchDedicatedResources,
    )
    service_account: str = proto.Field(
        proto.STRING,
        number=29,
    )
    manual_batch_tuning_parameters: gca_manual_batch_tuning_parameters.ManualBatchTuningParameters = proto.Field(
        proto.MESSAGE,
        number=8,
        message=gca_manual_batch_tuning_parameters.ManualBatchTuningParameters,
    )
    generate_explanation: bool = proto.Field(
        proto.BOOL,
        number=23,
    )
    explanation_spec: explanation.ExplanationSpec = proto.Field(
        proto.MESSAGE,
        number=25,
        message=explanation.ExplanationSpec,
    )
    output_info: OutputInfo = proto.Field(
        proto.MESSAGE,
        number=9,
        message=OutputInfo,
    )
    state: job_state.JobState = proto.Field(
        proto.ENUM,
        number=10,
        enum=job_state.JobState,
    )
    error: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=11,
        message=status_pb2.Status,
    )
    partial_failures: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=12,
        message=status_pb2.Status,
    )
    resources_consumed: machine_resources.ResourcesConsumed = proto.Field(
        proto.MESSAGE,
        number=13,
        message=machine_resources.ResourcesConsumed,
    )
    completion_stats: gca_completion_stats.CompletionStats = proto.Field(
        proto.MESSAGE,
        number=14,
        message=gca_completion_stats.CompletionStats,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=15,
        message=timestamp_pb2.Timestamp,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=16,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=17,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=18,
        message=timestamp_pb2.Timestamp,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=19,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=24,
        message=gca_encryption_spec.EncryptionSpec,
    )
    disable_container_logging: bool = proto.Field(
        proto.BOOL,
        number=34,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
