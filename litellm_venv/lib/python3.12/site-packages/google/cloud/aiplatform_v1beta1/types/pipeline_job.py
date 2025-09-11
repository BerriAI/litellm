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

from google.cloud.aiplatform_v1beta1.types import artifact
from google.cloud.aiplatform_v1beta1.types import context
from google.cloud.aiplatform_v1beta1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1beta1.types import execution as gca_execution
from google.cloud.aiplatform_v1beta1.types import pipeline_failure_policy
from google.cloud.aiplatform_v1beta1.types import pipeline_state
from google.cloud.aiplatform_v1beta1.types import value as gca_value
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "PipelineJob",
        "PipelineTemplateMetadata",
        "PipelineJobDetail",
        "PipelineTaskDetail",
        "PipelineTaskExecutorDetail",
    },
)


class PipelineJob(proto.Message):
    r"""An instance of a machine learning PipelineJob.

    Attributes:
        name (str):
            Output only. The resource name of the
            PipelineJob.
        display_name (str):
            The display name of the Pipeline.
            The name can be up to 128 characters long and
            can consist of any UTF-8 characters.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Pipeline creation time.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Pipeline start time.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Pipeline end time.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this PipelineJob
            was most recently updated.
        pipeline_spec (google.protobuf.struct_pb2.Struct):
            The spec of the pipeline.
        state (google.cloud.aiplatform_v1beta1.types.PipelineState):
            Output only. The detailed state of the job.
        job_detail (google.cloud.aiplatform_v1beta1.types.PipelineJobDetail):
            Output only. The details of pipeline run. Not
            available in the list view.
        error (google.rpc.status_pb2.Status):
            Output only. The error that occurred during
            pipeline execution. Only populated when the
            pipeline's state is FAILED or CANCELLED.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to organize
            PipelineJob.

            Label keys and values can be no longer than 64 characters
            (Unicode codepoints), can only contain lowercase letters,
            numeric characters, underscores and dashes. International
            characters are allowed.

            See https://goo.gl/xmQnxf for more information and examples
            of labels.

            Note there is some reserved label key for Vertex AI
            Pipelines.

            -  ``vertex-ai-pipelines-run-billing-id``, user set value
               will get overrided.
        runtime_config (google.cloud.aiplatform_v1beta1.types.PipelineJob.RuntimeConfig):
            Runtime config of the pipeline.
        encryption_spec (google.cloud.aiplatform_v1beta1.types.EncryptionSpec):
            Customer-managed encryption key spec for a
            pipelineJob. If set, this PipelineJob and all of
            its sub-resources will be secured by this key.
        service_account (str):
            The service account that the pipeline workload runs as. If
            not specified, the Compute Engine default service account in
            the project will be used. See
            https://cloud.google.com/compute/docs/access/service-accounts#default_service_account

            Users starting the pipeline must have the
            ``iam.serviceAccounts.actAs`` permission on this service
            account.
        network (str):
            The full name of the Compute Engine
            `network </compute/docs/networks-and-firewalls#networks>`__
            to which the Pipeline Job's workload should be peered. For
            example, ``projects/12345/global/networks/myVPC``.
            `Format </compute/docs/reference/rest/v1/networks/insert>`__
            is of the form
            ``projects/{project}/global/networks/{network}``. Where
            {project} is a project number, as in ``12345``, and
            {network} is a network name.

            Private services access must already be configured for the
            network. Pipeline job will apply the network configuration
            to the Google Cloud resources being launched, if applied,
            such as Vertex AI Training or Dataflow job. If left
            unspecified, the workload is not peered with any network.
        reserved_ip_ranges (MutableSequence[str]):
            A list of names for the reserved ip ranges under the VPC
            network that can be used for this Pipeline Job's workload.

            If set, we will deploy the Pipeline Job's workload within
            the provided ip ranges. Otherwise, the job will be deployed
            to any ip ranges under the provided VPC network.

            Example: ['vertex-ai-ip-range'].
        template_uri (str):
            A template uri from where the
            [PipelineJob.pipeline_spec][google.cloud.aiplatform.v1beta1.PipelineJob.pipeline_spec],
            if empty, will be downloaded. Currently, only uri from
            Vertex Template Registry & Gallery is supported. Reference
            to
            https://cloud.google.com/vertex-ai/docs/pipelines/create-pipeline-template.
        template_metadata (google.cloud.aiplatform_v1beta1.types.PipelineTemplateMetadata):
            Output only. Pipeline template metadata. Will fill up fields
            if
            [PipelineJob.template_uri][google.cloud.aiplatform.v1beta1.PipelineJob.template_uri]
            is from supported template registry.
        schedule_name (str):
            Output only. The schedule resource name.
            Only returned if the Pipeline is created by
            Schedule API.
        preflight_validations (bool):
            Optional. Whether to do component level
            validations before job creation.
    """

    class RuntimeConfig(proto.Message):
        r"""The runtime config of a PipelineJob.

        Attributes:
            parameters (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.Value]):
                Deprecated. Use
                [RuntimeConfig.parameter_values][google.cloud.aiplatform.v1beta1.PipelineJob.RuntimeConfig.parameter_values]
                instead. The runtime parameters of the PipelineJob. The
                parameters will be passed into
                [PipelineJob.pipeline_spec][google.cloud.aiplatform.v1beta1.PipelineJob.pipeline_spec]
                to replace the placeholders at runtime. This field is used
                by pipelines built using
                ``PipelineJob.pipeline_spec.schema_version`` 2.0.0 or lower,
                such as pipelines built using Kubeflow Pipelines SDK 1.8 or
                lower.
            gcs_output_directory (str):
                Required. A path in a Cloud Storage bucket, which will be
                treated as the root output directory of the pipeline. It is
                used by the system to generate the paths of output
                artifacts. The artifact paths are generated with a sub-path
                pattern ``{job_id}/{task_id}/{output_key}`` under the
                specified output directory. The service account specified in
                this pipeline must have the ``storage.objects.get`` and
                ``storage.objects.create`` permissions for this bucket.
            parameter_values (MutableMapping[str, google.protobuf.struct_pb2.Value]):
                The runtime parameters of the PipelineJob. The parameters
                will be passed into
                [PipelineJob.pipeline_spec][google.cloud.aiplatform.v1beta1.PipelineJob.pipeline_spec]
                to replace the placeholders at runtime. This field is used
                by pipelines built using
                ``PipelineJob.pipeline_spec.schema_version`` 2.1.0, such as
                pipelines built using Kubeflow Pipelines SDK 1.9 or higher
                and the v2 DSL.
            failure_policy (google.cloud.aiplatform_v1beta1.types.PipelineFailurePolicy):
                Represents the failure policy of a pipeline. Currently, the
                default of a pipeline is that the pipeline will continue to
                run until no more tasks can be executed, also known as
                PIPELINE_FAILURE_POLICY_FAIL_SLOW. However, if a pipeline is
                set to PIPELINE_FAILURE_POLICY_FAIL_FAST, it will stop
                scheduling any new tasks when a task has failed. Any
                scheduled tasks will continue to completion.
            input_artifacts (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.PipelineJob.RuntimeConfig.InputArtifact]):
                The runtime artifacts of the PipelineJob. The
                key will be the input artifact name and the
                value would be one of the InputArtifact.
        """

        class InputArtifact(proto.Message):
            r"""The type of an input artifact.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                artifact_id (str):
                    Artifact resource id from MLMD. Which is the last portion of
                    an artifact resource name:
                    ``projects/{project}/locations/{location}/metadataStores/default/artifacts/{artifact_id}``.
                    The artifact must stay within the same project, location and
                    default metadatastore as the pipeline.

                    This field is a member of `oneof`_ ``kind``.
            """

            artifact_id: str = proto.Field(
                proto.STRING,
                number=1,
                oneof="kind",
            )

        parameters: MutableMapping[str, gca_value.Value] = proto.MapField(
            proto.STRING,
            proto.MESSAGE,
            number=1,
            message=gca_value.Value,
        )
        gcs_output_directory: str = proto.Field(
            proto.STRING,
            number=2,
        )
        parameter_values: MutableMapping[str, struct_pb2.Value] = proto.MapField(
            proto.STRING,
            proto.MESSAGE,
            number=3,
            message=struct_pb2.Value,
        )
        failure_policy: pipeline_failure_policy.PipelineFailurePolicy = proto.Field(
            proto.ENUM,
            number=4,
            enum=pipeline_failure_policy.PipelineFailurePolicy,
        )
        input_artifacts: MutableMapping[
            str, "PipelineJob.RuntimeConfig.InputArtifact"
        ] = proto.MapField(
            proto.STRING,
            proto.MESSAGE,
            number=5,
            message="PipelineJob.RuntimeConfig.InputArtifact",
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    pipeline_spec: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=7,
        message=struct_pb2.Struct,
    )
    state: pipeline_state.PipelineState = proto.Field(
        proto.ENUM,
        number=8,
        enum=pipeline_state.PipelineState,
    )
    job_detail: "PipelineJobDetail" = proto.Field(
        proto.MESSAGE,
        number=9,
        message="PipelineJobDetail",
    )
    error: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=10,
        message=status_pb2.Status,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=11,
    )
    runtime_config: RuntimeConfig = proto.Field(
        proto.MESSAGE,
        number=12,
        message=RuntimeConfig,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=16,
        message=gca_encryption_spec.EncryptionSpec,
    )
    service_account: str = proto.Field(
        proto.STRING,
        number=17,
    )
    network: str = proto.Field(
        proto.STRING,
        number=18,
    )
    reserved_ip_ranges: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=25,
    )
    template_uri: str = proto.Field(
        proto.STRING,
        number=19,
    )
    template_metadata: "PipelineTemplateMetadata" = proto.Field(
        proto.MESSAGE,
        number=20,
        message="PipelineTemplateMetadata",
    )
    schedule_name: str = proto.Field(
        proto.STRING,
        number=22,
    )
    preflight_validations: bool = proto.Field(
        proto.BOOL,
        number=26,
    )


class PipelineTemplateMetadata(proto.Message):
    r"""Pipeline template metadata if
    [PipelineJob.template_uri][google.cloud.aiplatform.v1beta1.PipelineJob.template_uri]
    is from supported template registry. Currently, the only supported
    registry is Artifact Registry.

    Attributes:
        version (str):
            The version_name in artifact registry.

            Will always be presented in output if the
            [PipelineJob.template_uri][google.cloud.aiplatform.v1beta1.PipelineJob.template_uri]
            is from supported template registry.

            Format is "sha256:abcdef123456...".
    """

    version: str = proto.Field(
        proto.STRING,
        number=3,
    )


class PipelineJobDetail(proto.Message):
    r"""The runtime detail of PipelineJob.

    Attributes:
        pipeline_context (google.cloud.aiplatform_v1beta1.types.Context):
            Output only. The context of the pipeline.
        pipeline_run_context (google.cloud.aiplatform_v1beta1.types.Context):
            Output only. The context of the current
            pipeline run.
        task_details (MutableSequence[google.cloud.aiplatform_v1beta1.types.PipelineTaskDetail]):
            Output only. The runtime details of the tasks
            under the pipeline.
    """

    pipeline_context: context.Context = proto.Field(
        proto.MESSAGE,
        number=1,
        message=context.Context,
    )
    pipeline_run_context: context.Context = proto.Field(
        proto.MESSAGE,
        number=2,
        message=context.Context,
    )
    task_details: MutableSequence["PipelineTaskDetail"] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="PipelineTaskDetail",
    )


class PipelineTaskDetail(proto.Message):
    r"""The runtime detail of a task execution.

    Attributes:
        task_id (int):
            Output only. The system generated ID of the
            task.
        parent_task_id (int):
            Output only. The id of the parent task if the
            task is within a component scope. Empty if the
            task is at the root level.
        task_name (str):
            Output only. The user specified name of the task that is
            defined in
            [pipeline_spec][google.cloud.aiplatform.v1beta1.PipelineJob.pipeline_spec].
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Task create time.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Task start time.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Task end time.
        executor_detail (google.cloud.aiplatform_v1beta1.types.PipelineTaskExecutorDetail):
            Output only. The detailed execution info.
        state (google.cloud.aiplatform_v1beta1.types.PipelineTaskDetail.State):
            Output only. State of the task.
        execution (google.cloud.aiplatform_v1beta1.types.Execution):
            Output only. The execution metadata of the
            task.
        error (google.rpc.status_pb2.Status):
            Output only. The error that occurred during
            task execution. Only populated when the task's
            state is FAILED or CANCELLED.
        pipeline_task_status (MutableSequence[google.cloud.aiplatform_v1beta1.types.PipelineTaskDetail.PipelineTaskStatus]):
            Output only. A list of task status. This
            field keeps a record of task status evolving
            over time.
        inputs (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.PipelineTaskDetail.ArtifactList]):
            Output only. The runtime input artifacts of
            the task.
        outputs (MutableMapping[str, google.cloud.aiplatform_v1beta1.types.PipelineTaskDetail.ArtifactList]):
            Output only. The runtime output artifacts of
            the task.
    """

    class State(proto.Enum):
        r"""Specifies state of TaskExecution

        Values:
            STATE_UNSPECIFIED (0):
                Unspecified.
            PENDING (1):
                Specifies pending state for the task.
            RUNNING (2):
                Specifies task is being executed.
            SUCCEEDED (3):
                Specifies task completed successfully.
            CANCEL_PENDING (4):
                Specifies Task cancel is in pending state.
            CANCELLING (5):
                Specifies task is being cancelled.
            CANCELLED (6):
                Specifies task was cancelled.
            FAILED (7):
                Specifies task failed.
            SKIPPED (8):
                Specifies task was skipped due to cache hit.
            NOT_TRIGGERED (9):
                Specifies that the task was not triggered because the task's
                trigger policy is not satisfied. The trigger policy is
                specified in the ``condition`` field of
                [PipelineJob.pipeline_spec][google.cloud.aiplatform.v1beta1.PipelineJob.pipeline_spec].
        """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        SUCCEEDED = 3
        CANCEL_PENDING = 4
        CANCELLING = 5
        CANCELLED = 6
        FAILED = 7
        SKIPPED = 8
        NOT_TRIGGERED = 9

    class PipelineTaskStatus(proto.Message):
        r"""A single record of the task status.

        Attributes:
            update_time (google.protobuf.timestamp_pb2.Timestamp):
                Output only. Update time of this status.
            state (google.cloud.aiplatform_v1beta1.types.PipelineTaskDetail.State):
                Output only. The state of the task.
            error (google.rpc.status_pb2.Status):
                Output only. The error that occurred during
                the state. May be set when the state is any of
                the non-final state (PENDING/RUNNING/CANCELLING)
                or FAILED state. If the state is FAILED, the
                error here is final and not going to be retried.
                If the state is a non-final state, the error
                indicates a system-error being retried.
        """

        update_time: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=1,
            message=timestamp_pb2.Timestamp,
        )
        state: "PipelineTaskDetail.State" = proto.Field(
            proto.ENUM,
            number=2,
            enum="PipelineTaskDetail.State",
        )
        error: status_pb2.Status = proto.Field(
            proto.MESSAGE,
            number=3,
            message=status_pb2.Status,
        )

    class ArtifactList(proto.Message):
        r"""A list of artifact metadata.

        Attributes:
            artifacts (MutableSequence[google.cloud.aiplatform_v1beta1.types.Artifact]):
                Output only. A list of artifact metadata.
        """

        artifacts: MutableSequence[artifact.Artifact] = proto.RepeatedField(
            proto.MESSAGE,
            number=1,
            message=artifact.Artifact,
        )

    task_id: int = proto.Field(
        proto.INT64,
        number=1,
    )
    parent_task_id: int = proto.Field(
        proto.INT64,
        number=12,
    )
    task_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    executor_detail: "PipelineTaskExecutorDetail" = proto.Field(
        proto.MESSAGE,
        number=6,
        message="PipelineTaskExecutorDetail",
    )
    state: State = proto.Field(
        proto.ENUM,
        number=7,
        enum=State,
    )
    execution: gca_execution.Execution = proto.Field(
        proto.MESSAGE,
        number=8,
        message=gca_execution.Execution,
    )
    error: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=9,
        message=status_pb2.Status,
    )
    pipeline_task_status: MutableSequence[PipelineTaskStatus] = proto.RepeatedField(
        proto.MESSAGE,
        number=13,
        message=PipelineTaskStatus,
    )
    inputs: MutableMapping[str, ArtifactList] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=10,
        message=ArtifactList,
    )
    outputs: MutableMapping[str, ArtifactList] = proto.MapField(
        proto.STRING,
        proto.MESSAGE,
        number=11,
        message=ArtifactList,
    )


class PipelineTaskExecutorDetail(proto.Message):
    r"""The runtime detail of a pipeline executor.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        container_detail (google.cloud.aiplatform_v1beta1.types.PipelineTaskExecutorDetail.ContainerDetail):
            Output only. The detailed info for a
            container executor.

            This field is a member of `oneof`_ ``details``.
        custom_job_detail (google.cloud.aiplatform_v1beta1.types.PipelineTaskExecutorDetail.CustomJobDetail):
            Output only. The detailed info for a custom
            job executor.

            This field is a member of `oneof`_ ``details``.
    """

    class ContainerDetail(proto.Message):
        r"""The detail of a container execution. It contains the job
        names of the lifecycle of a container execution.

        Attributes:
            main_job (str):
                Output only. The name of the
                [CustomJob][google.cloud.aiplatform.v1beta1.CustomJob] for
                the main container execution.
            pre_caching_check_job (str):
                Output only. The name of the
                [CustomJob][google.cloud.aiplatform.v1beta1.CustomJob] for
                the pre-caching-check container execution. This job will be
                available if the
                [PipelineJob.pipeline_spec][google.cloud.aiplatform.v1beta1.PipelineJob.pipeline_spec]
                specifies the ``pre_caching_check`` hook in the lifecycle
                events.
            failed_main_jobs (MutableSequence[str]):
                Output only. The names of the previously failed
                [CustomJob][google.cloud.aiplatform.v1beta1.CustomJob] for
                the main container executions. The list includes the all
                attempts in chronological order.
            failed_pre_caching_check_jobs (MutableSequence[str]):
                Output only. The names of the previously failed
                [CustomJob][google.cloud.aiplatform.v1beta1.CustomJob] for
                the pre-caching-check container executions. This job will be
                available if the
                [PipelineJob.pipeline_spec][google.cloud.aiplatform.v1beta1.PipelineJob.pipeline_spec]
                specifies the ``pre_caching_check`` hook in the lifecycle
                events. The list includes the all attempts in chronological
                order.
        """

        main_job: str = proto.Field(
            proto.STRING,
            number=1,
        )
        pre_caching_check_job: str = proto.Field(
            proto.STRING,
            number=2,
        )
        failed_main_jobs: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=3,
        )
        failed_pre_caching_check_jobs: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=4,
        )

    class CustomJobDetail(proto.Message):
        r"""The detailed info for a custom job executor.

        Attributes:
            job (str):
                Output only. The name of the
                [CustomJob][google.cloud.aiplatform.v1beta1.CustomJob].
            failed_jobs (MutableSequence[str]):
                Output only. The names of the previously failed
                [CustomJob][google.cloud.aiplatform.v1beta1.CustomJob]. The
                list includes the all attempts in chronological order.
        """

        job: str = proto.Field(
            proto.STRING,
            number=1,
        )
        failed_jobs: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=3,
        )

    container_detail: ContainerDetail = proto.Field(
        proto.MESSAGE,
        number=1,
        oneof="details",
        message=ContainerDetail,
    )
    custom_job_detail: CustomJobDetail = proto.Field(
        proto.MESSAGE,
        number=2,
        oneof="details",
        message=CustomJobDetail,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
