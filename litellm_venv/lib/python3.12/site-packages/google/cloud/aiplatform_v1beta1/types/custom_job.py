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

from google.cloud.aiplatform_v1beta1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1beta1.types import env_var
from google.cloud.aiplatform_v1beta1.types import io
from google.cloud.aiplatform_v1beta1.types import job_state
from google.cloud.aiplatform_v1beta1.types import machine_resources
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CustomJob",
        "CustomJobSpec",
        "WorkerPoolSpec",
        "ContainerSpec",
        "PythonPackageSpec",
        "Scheduling",
    },
)


class CustomJob(proto.Message):
    r"""Represents a job that runs custom workloads such as a Docker
    container or a Python package. A CustomJob can have multiple
    worker pools and each worker pool can have its own machine and
    input spec. A CustomJob will be cleaned up once the job enters
    terminal state (failed or succeeded).

    Attributes:
        name (str):
            Output only. Resource name of a CustomJob.
        display_name (str):
            Required. The display name of the CustomJob.
            The name can be up to 128 characters long and
            can consist of any UTF-8 characters.
        job_spec (google.cloud.aiplatform_v1beta1.types.CustomJobSpec):
            Required. Job spec.
        state (google.cloud.aiplatform_v1beta1.types.JobState):
            Output only. The detailed state of the job.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the CustomJob was
            created.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the CustomJob for the first time
            entered the ``JOB_STATE_RUNNING`` state.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the CustomJob entered any of the
            following states: ``JOB_STATE_SUCCEEDED``,
            ``JOB_STATE_FAILED``, ``JOB_STATE_CANCELLED``.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the CustomJob was most
            recently updated.
        error (google.rpc.status_pb2.Status):
            Output only. Only populated when job's state is
            ``JOB_STATE_FAILED`` or ``JOB_STATE_CANCELLED``.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to
            organize CustomJobs.
            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            and examples of labels.
        encryption_spec (google.cloud.aiplatform_v1beta1.types.EncryptionSpec):
            Customer-managed encryption key options for a
            CustomJob. If this is set, then all resources
            created by the CustomJob will be encrypted with
            the provided encryption key.
        web_access_uris (MutableMapping[str, str]):
            Output only. URIs for accessing `interactive
            shells <https://cloud.google.com/vertex-ai/docs/training/monitor-debug-interactive-shell>`__
            (one URI for each training node). Only available if
            [job_spec.enable_web_access][google.cloud.aiplatform.v1beta1.CustomJobSpec.enable_web_access]
            is ``true``.

            The keys are names of each node in the training job; for
            example, ``workerpool0-0`` for the primary node,
            ``workerpool1-0`` for the first node in the second worker
            pool, and ``workerpool1-1`` for the second node in the
            second worker pool.

            The values are the URIs for each node's interactive shell.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    job_spec: "CustomJobSpec" = proto.Field(
        proto.MESSAGE,
        number=4,
        message="CustomJobSpec",
    )
    state: job_state.JobState = proto.Field(
        proto.ENUM,
        number=5,
        enum=job_state.JobState,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=9,
        message=timestamp_pb2.Timestamp,
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
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=12,
        message=gca_encryption_spec.EncryptionSpec,
    )
    web_access_uris: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=16,
    )


class CustomJobSpec(proto.Message):
    r"""Represents the spec of a CustomJob.

    Attributes:
        persistent_resource_id (str):
            Optional. The ID of the PersistentResource in
            the same Project and Location which to run

            If this is specified, the job will be run on
            existing machines held by the PersistentResource
            instead of on-demand short-live machines. The
            network and CMEK configs on the job should be
            consistent with those on the PersistentResource,
            otherwise, the job will be rejected.
        worker_pool_specs (MutableSequence[google.cloud.aiplatform_v1beta1.types.WorkerPoolSpec]):
            Required. The spec of the worker pools
            including machine type and Docker image. All
            worker pools except the first one are optional
            and can be skipped by providing an empty value.
        scheduling (google.cloud.aiplatform_v1beta1.types.Scheduling):
            Scheduling options for a CustomJob.
        service_account (str):
            Specifies the service account for workload run-as account.
            Users submitting jobs must have act-as permission on this
            run-as account. If unspecified, the `Vertex AI Custom Code
            Service
            Agent <https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents>`__
            for the CustomJob's project is used.
        network (str):
            Optional. The full name of the Compute Engine
            `network </compute/docs/networks-and-firewalls#networks>`__
            to which the Job should be peered. For example,
            ``projects/12345/global/networks/myVPC``.
            `Format </compute/docs/reference/rest/v1/networks/insert>`__
            is of the form
            ``projects/{project}/global/networks/{network}``. Where
            {project} is a project number, as in ``12345``, and
            {network} is a network name.

            To specify this field, you must have already `configured VPC
            Network Peering for Vertex
            AI <https://cloud.google.com/vertex-ai/docs/general/vpc-peering>`__.

            If this field is left unspecified, the job is not peered
            with any network.
        reserved_ip_ranges (MutableSequence[str]):
            Optional. A list of names for the reserved ip ranges under
            the VPC network that can be used for this job.

            If set, we will deploy the job within the provided ip
            ranges. Otherwise, the job will be deployed to any ip ranges
            under the provided VPC network.

            Example: ['vertex-ai-ip-range'].
        base_output_directory (google.cloud.aiplatform_v1beta1.types.GcsDestination):
            The Cloud Storage location to store the output of this
            CustomJob or HyperparameterTuningJob. For
            HyperparameterTuningJob, the baseOutputDirectory of each
            child CustomJob backing a Trial is set to a subdirectory of
            name [id][google.cloud.aiplatform.v1beta1.Trial.id] under
            its parent HyperparameterTuningJob's baseOutputDirectory.

            The following Vertex AI environment variables will be passed
            to containers or python modules when this field is set:

            For CustomJob:

            -  AIP_MODEL_DIR = ``<base_output_directory>/model/``
            -  AIP_CHECKPOINT_DIR =
               ``<base_output_directory>/checkpoints/``
            -  AIP_TENSORBOARD_LOG_DIR =
               ``<base_output_directory>/logs/``

            For CustomJob backing a Trial of HyperparameterTuningJob:

            -  AIP_MODEL_DIR =
               ``<base_output_directory>/<trial_id>/model/``
            -  AIP_CHECKPOINT_DIR =
               ``<base_output_directory>/<trial_id>/checkpoints/``
            -  AIP_TENSORBOARD_LOG_DIR =
               ``<base_output_directory>/<trial_id>/logs/``
        protected_artifact_location_id (str):
            The ID of the location to store protected
            artifacts. e.g. us-central1. Populate only when
            the location is different than CustomJob
            location. List of supported locations:

            https://cloud.google.com/vertex-ai/docs/general/locations
        tensorboard (str):
            Optional. The name of a Vertex AI
            [Tensorboard][google.cloud.aiplatform.v1beta1.Tensorboard]
            resource to which this CustomJob will upload Tensorboard
            logs. Format:
            ``projects/{project}/locations/{location}/tensorboards/{tensorboard}``
        enable_web_access (bool):
            Optional. Whether you want Vertex AI to enable `interactive
            shell
            access <https://cloud.google.com/vertex-ai/docs/training/monitor-debug-interactive-shell>`__
            to training containers.

            If set to ``true``, you can access interactive shells at the
            URIs given by
            [CustomJob.web_access_uris][google.cloud.aiplatform.v1beta1.CustomJob.web_access_uris]
            or
            [Trial.web_access_uris][google.cloud.aiplatform.v1beta1.Trial.web_access_uris]
            (within
            [HyperparameterTuningJob.trials][google.cloud.aiplatform.v1beta1.HyperparameterTuningJob.trials]).
        enable_dashboard_access (bool):
            Optional. Whether you want Vertex AI to enable access to the
            customized dashboard in training chief container.

            If set to ``true``, you can access the dashboard at the URIs
            given by
            [CustomJob.web_access_uris][google.cloud.aiplatform.v1beta1.CustomJob.web_access_uris]
            or
            [Trial.web_access_uris][google.cloud.aiplatform.v1beta1.Trial.web_access_uris]
            (within
            [HyperparameterTuningJob.trials][google.cloud.aiplatform.v1beta1.HyperparameterTuningJob.trials]).
        experiment (str):
            Optional. The Experiment associated with this job. Format:
            ``projects/{project}/locations/{location}/metadataStores/{metadataStores}/contexts/{experiment-name}``
        experiment_run (str):
            Optional. The Experiment Run associated with this job.
            Format:
            ``projects/{project}/locations/{location}/metadataStores/{metadataStores}/contexts/{experiment-name}-{experiment-run-name}``
        models (MutableSequence[str]):
            Optional. The name of the Model resources for which to
            generate a mapping to artifact URIs. Applicable only to some
            of the Google-provided custom jobs. Format:
            ``projects/{project}/locations/{location}/models/{model}``

            In order to retrieve a specific version of the model, also
            provide the version ID or version alias. Example:
            ``projects/{project}/locations/{location}/models/{model}@2``
            or
            ``projects/{project}/locations/{location}/models/{model}@golden``
            If no version ID or alias is specified, the "default"
            version will be returned. The "default" version alias is
            created for the first version of the model, and can be moved
            to other versions later on. There will be exactly one
            default version.
    """

    persistent_resource_id: str = proto.Field(
        proto.STRING,
        number=14,
    )
    worker_pool_specs: MutableSequence["WorkerPoolSpec"] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message="WorkerPoolSpec",
    )
    scheduling: "Scheduling" = proto.Field(
        proto.MESSAGE,
        number=3,
        message="Scheduling",
    )
    service_account: str = proto.Field(
        proto.STRING,
        number=4,
    )
    network: str = proto.Field(
        proto.STRING,
        number=5,
    )
    reserved_ip_ranges: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=13,
    )
    base_output_directory: io.GcsDestination = proto.Field(
        proto.MESSAGE,
        number=6,
        message=io.GcsDestination,
    )
    protected_artifact_location_id: str = proto.Field(
        proto.STRING,
        number=19,
    )
    tensorboard: str = proto.Field(
        proto.STRING,
        number=7,
    )
    enable_web_access: bool = proto.Field(
        proto.BOOL,
        number=10,
    )
    enable_dashboard_access: bool = proto.Field(
        proto.BOOL,
        number=16,
    )
    experiment: str = proto.Field(
        proto.STRING,
        number=17,
    )
    experiment_run: str = proto.Field(
        proto.STRING,
        number=18,
    )
    models: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=20,
    )


class WorkerPoolSpec(proto.Message):
    r"""Represents the spec of a worker pool in a job.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        container_spec (google.cloud.aiplatform_v1beta1.types.ContainerSpec):
            The custom container task.

            This field is a member of `oneof`_ ``task``.
        python_package_spec (google.cloud.aiplatform_v1beta1.types.PythonPackageSpec):
            The Python packaged task.

            This field is a member of `oneof`_ ``task``.
        machine_spec (google.cloud.aiplatform_v1beta1.types.MachineSpec):
            Optional. Immutable. The specification of a
            single machine.
        replica_count (int):
            Optional. The number of worker replicas to
            use for this worker pool.
        nfs_mounts (MutableSequence[google.cloud.aiplatform_v1beta1.types.NfsMount]):
            Optional. List of NFS mount spec.
        disk_spec (google.cloud.aiplatform_v1beta1.types.DiskSpec):
            Disk spec.
    """

    container_spec: "ContainerSpec" = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="task",
        message="ContainerSpec",
    )
    python_package_spec: "PythonPackageSpec" = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="task",
        message="PythonPackageSpec",
    )
    machine_spec: machine_resources.MachineSpec = proto.Field(
        proto.MESSAGE,
        number=1,
        message=machine_resources.MachineSpec,
    )
    replica_count: int = proto.Field(
        proto.INT64,
        number=2,
    )
    nfs_mounts: MutableSequence[machine_resources.NfsMount] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=machine_resources.NfsMount,
    )
    disk_spec: machine_resources.DiskSpec = proto.Field(
        proto.MESSAGE,
        number=5,
        message=machine_resources.DiskSpec,
    )


class ContainerSpec(proto.Message):
    r"""The spec of a Container.

    Attributes:
        image_uri (str):
            Required. The URI of a container image in the
            Container Registry that is to be run on each
            worker replica.
        command (MutableSequence[str]):
            The command to be invoked when the container
            is started. It overrides the entrypoint
            instruction in Dockerfile when provided.
        args (MutableSequence[str]):
            The arguments to be passed when starting the
            container.
        env (MutableSequence[google.cloud.aiplatform_v1beta1.types.EnvVar]):
            Environment variables to be passed to the
            container. Maximum limit is 100.
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


class PythonPackageSpec(proto.Message):
    r"""The spec of a Python packaged code.

    Attributes:
        executor_image_uri (str):
            Required. The URI of a container image in Artifact Registry
            that will run the provided Python package. Vertex AI
            provides a wide range of executor images with pre-installed
            packages to meet users' various use cases. See the list of
            `pre-built containers for
            training <https://cloud.google.com/vertex-ai/docs/training/pre-built-containers>`__.
            You must use an image from this list.
        package_uris (MutableSequence[str]):
            Required. The Google Cloud Storage location
            of the Python package files which are the
            training program and its dependent packages. The
            maximum number of package URIs is 100.
        python_module (str):
            Required. The Python module name to run after
            installing the packages.
        args (MutableSequence[str]):
            Command line arguments to be passed to the
            Python task.
        env (MutableSequence[google.cloud.aiplatform_v1beta1.types.EnvVar]):
            Environment variables to be passed to the
            python module. Maximum limit is 100.
    """

    executor_image_uri: str = proto.Field(
        proto.STRING,
        number=1,
    )
    package_uris: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=2,
    )
    python_module: str = proto.Field(
        proto.STRING,
        number=3,
    )
    args: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=4,
    )
    env: MutableSequence[env_var.EnvVar] = proto.RepeatedField(
        proto.MESSAGE,
        number=5,
        message=env_var.EnvVar,
    )


class Scheduling(proto.Message):
    r"""All parameters related to queuing and scheduling of custom
    jobs.

    Attributes:
        timeout (google.protobuf.duration_pb2.Duration):
            The maximum job running time. The default is
            7 days.
        restart_job_on_worker_restart (bool):
            Restarts the entire CustomJob if a worker
            gets restarted. This feature can be used by
            distributed training jobs that are not resilient
            to workers leaving and joining a job.
        disable_retries (bool):
            Optional. Indicates if the job should retry for internal
            errors after the job starts running. If true, overrides
            ``Scheduling.restart_job_on_worker_restart`` to false.
    """

    timeout: duration_pb2.Duration = proto.Field(
        proto.MESSAGE,
        number=1,
        message=duration_pb2.Duration,
    )
    restart_job_on_worker_restart: bool = proto.Field(
        proto.BOOL,
        number=3,
    )
    disable_retries: bool = proto.Field(
        proto.BOOL,
        number=5,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
