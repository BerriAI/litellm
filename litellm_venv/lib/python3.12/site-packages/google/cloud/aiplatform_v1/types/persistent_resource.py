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

from google.cloud.aiplatform_v1.types import encryption_spec as gca_encryption_spec
from google.cloud.aiplatform_v1.types import machine_resources
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "PersistentResource",
        "ResourcePool",
        "ResourceRuntimeSpec",
        "RaySpec",
        "ResourceRuntime",
        "ServiceAccountSpec",
    },
)


class PersistentResource(proto.Message):
    r"""Represents long-lasting resources that are dedicated to users
    to runs custom workloads.
    A PersistentResource can have multiple node pools and each node
    pool can have its own machine spec.

    Attributes:
        name (str):
            Immutable. Resource name of a
            PersistentResource.
        display_name (str):
            Optional. The display name of the
            PersistentResource. The name can be up to 128
            characters long and can consist of any UTF-8
            characters.
        resource_pools (MutableSequence[google.cloud.aiplatform_v1.types.ResourcePool]):
            Required. The spec of the pools of different
            resources.
        state (google.cloud.aiplatform_v1.types.PersistentResource.State):
            Output only. The detailed state of a Study.
        error (google.rpc.status_pb2.Status):
            Output only. Only populated when persistent resource's state
            is ``STOPPING`` or ``ERROR``.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the PersistentResource
            was created.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the PersistentResource for the first
            time entered the ``RUNNING`` state.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the PersistentResource
            was most recently updated.
        labels (MutableMapping[str, str]):
            Optional. The labels with user-defined
            metadata to organize PersistentResource.

            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            and examples of labels.
        network (str):
            Optional. The full name of the Compute Engine
            `network </compute/docs/networks-and-firewalls#networks>`__
            to peered with Vertex AI to host the persistent resources.
            For example, ``projects/12345/global/networks/myVPC``.
            `Format </compute/docs/reference/rest/v1/networks/insert>`__
            is of the form
            ``projects/{project}/global/networks/{network}``. Where
            {project} is a project number, as in ``12345``, and
            {network} is a network name.

            To specify this field, you must have already `configured VPC
            Network Peering for Vertex
            AI <https://cloud.google.com/vertex-ai/docs/general/vpc-peering>`__.

            If this field is left unspecified, the resources aren't
            peered with any network.
        encryption_spec (google.cloud.aiplatform_v1.types.EncryptionSpec):
            Optional. Customer-managed encryption key
            spec for a PersistentResource. If set, this
            PersistentResource and all sub-resources of this
            PersistentResource will be secured by this key.
        resource_runtime_spec (google.cloud.aiplatform_v1.types.ResourceRuntimeSpec):
            Optional. Persistent Resource runtime spec.
            For example, used for Ray cluster configuration.
        resource_runtime (google.cloud.aiplatform_v1.types.ResourceRuntime):
            Output only. Runtime information of the
            Persistent Resource.
        reserved_ip_ranges (MutableSequence[str]):
            Optional. A list of names for the reserved IP ranges under
            the VPC network that can be used for this persistent
            resource.

            If set, we will deploy the persistent resource within the
            provided IP ranges. Otherwise, the persistent resource is
            deployed to any IP ranges under the provided VPC network.

            Example: ['vertex-ai-ip-range'].
    """

    class State(proto.Enum):
        r"""Describes the PersistentResource state.

        Values:
            STATE_UNSPECIFIED (0):
                Not set.
            PROVISIONING (1):
                The PROVISIONING state indicates the
                persistent resources is being created.
            RUNNING (3):
                The RUNNING state indicates the persistent
                resource is healthy and fully usable.
            STOPPING (4):
                The STOPPING state indicates the persistent
                resource is being deleted.
            ERROR (5):
                The ERROR state indicates the persistent resource may be
                unusable. Details can be found in the ``error`` field.
            REBOOTING (6):
                The REBOOTING state indicates the persistent
                resource is being rebooted (PR is not available
                right now but is expected to be ready again
                later).
            UPDATING (7):
                The UPDATING state indicates the persistent
                resource is being updated.
        """
        STATE_UNSPECIFIED = 0
        PROVISIONING = 1
        RUNNING = 3
        STOPPING = 4
        ERROR = 5
        REBOOTING = 6
        UPDATING = 7

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    resource_pools: MutableSequence["ResourcePool"] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message="ResourcePool",
    )
    state: State = proto.Field(
        proto.ENUM,
        number=5,
        enum=State,
    )
    error: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=6,
        message=status_pb2.Status,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=9,
        message=timestamp_pb2.Timestamp,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=10,
    )
    network: str = proto.Field(
        proto.STRING,
        number=11,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=12,
        message=gca_encryption_spec.EncryptionSpec,
    )
    resource_runtime_spec: "ResourceRuntimeSpec" = proto.Field(
        proto.MESSAGE,
        number=13,
        message="ResourceRuntimeSpec",
    )
    resource_runtime: "ResourceRuntime" = proto.Field(
        proto.MESSAGE,
        number=14,
        message="ResourceRuntime",
    )
    reserved_ip_ranges: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=15,
    )


class ResourcePool(proto.Message):
    r"""Represents the spec of a group of resources of the same type,
    for example machine type, disk, and accelerators, in a
    PersistentResource.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        id (str):
            Immutable. The unique ID in a
            PersistentResource for referring to this
            resource pool. User can specify it if necessary.
            Otherwise, it's generated automatically.
        machine_spec (google.cloud.aiplatform_v1.types.MachineSpec):
            Required. Immutable. The specification of a
            single machine.
        replica_count (int):
            Optional. The total number of machines to use
            for this resource pool.

            This field is a member of `oneof`_ ``_replica_count``.
        disk_spec (google.cloud.aiplatform_v1.types.DiskSpec):
            Optional. Disk spec for the machine in this
            node pool.
        used_replica_count (int):
            Output only. The number of machines currently in use by
            training jobs for this resource pool. Will replace
            idle_replica_count.
        autoscaling_spec (google.cloud.aiplatform_v1.types.ResourcePool.AutoscalingSpec):
            Optional. Optional spec to configure GKE
            autoscaling
    """

    class AutoscalingSpec(proto.Message):
        r"""The min/max number of replicas allowed if enabling
        autoscaling


        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            min_replica_count (int):
                Optional. min replicas in the node pool, must be ≤
                replica_count and < max_replica_count or will throw error

                This field is a member of `oneof`_ ``_min_replica_count``.
            max_replica_count (int):
                Optional. max replicas in the node pool, must be ≥
                replica_count and > min_replica_count or will throw error

                This field is a member of `oneof`_ ``_max_replica_count``.
        """

        min_replica_count: int = proto.Field(
            proto.INT64,
            number=1,
            optional=True,
        )
        max_replica_count: int = proto.Field(
            proto.INT64,
            number=2,
            optional=True,
        )

    id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    machine_spec: machine_resources.MachineSpec = proto.Field(
        proto.MESSAGE,
        number=2,
        message=machine_resources.MachineSpec,
    )
    replica_count: int = proto.Field(
        proto.INT64,
        number=3,
        optional=True,
    )
    disk_spec: machine_resources.DiskSpec = proto.Field(
        proto.MESSAGE,
        number=4,
        message=machine_resources.DiskSpec,
    )
    used_replica_count: int = proto.Field(
        proto.INT64,
        number=6,
    )
    autoscaling_spec: AutoscalingSpec = proto.Field(
        proto.MESSAGE,
        number=7,
        message=AutoscalingSpec,
    )


class ResourceRuntimeSpec(proto.Message):
    r"""Configuration for the runtime on a PersistentResource instance,
    including but not limited to:

    -  Service accounts used to run the workloads.
    -  Whether to make it a dedicated Ray Cluster.

    Attributes:
        service_account_spec (google.cloud.aiplatform_v1.types.ServiceAccountSpec):
            Optional. Configure the use of workload
            identity on the PersistentResource
        ray_spec (google.cloud.aiplatform_v1.types.RaySpec):
            Optional. Ray cluster configuration.
            Required when creating a dedicated RayCluster on
            the PersistentResource.
    """

    service_account_spec: "ServiceAccountSpec" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="ServiceAccountSpec",
    )
    ray_spec: "RaySpec" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="RaySpec",
    )


class RaySpec(proto.Message):
    r"""Configuration information for the Ray cluster.
    For experimental launch, Ray cluster creation and Persistent
    cluster creation are 1:1 mapping: We will provision all the
    nodes within the Persistent cluster as Ray nodes.

    """


class ResourceRuntime(proto.Message):
    r"""Persistent Cluster runtime information as output"""


class ServiceAccountSpec(proto.Message):
    r"""Configuration for the use of custom service account to run
    the workloads.

    Attributes:
        enable_custom_service_account (bool):
            Required. If true, custom user-managed service account is
            enforced to run any workloads (for example, Vertex Jobs) on
            the resource. Otherwise, uses the `Vertex AI Custom Code
            Service
            Agent <https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents>`__.
        service_account (str):
            Optional. Required when all below conditions are met

            -  ``enable_custom_service_account`` is true;
            -  any runtime is specified via ``ResourceRuntimeSpec`` on
               creation time, for example, Ray

            The users must have ``iam.serviceAccounts.actAs`` permission
            on this service account and then the specified runtime
            containers will run as it.

            Do not set this field if you want to submit jobs using
            custom service account to this PersistentResource after
            creation, but only specify the ``service_account`` inside
            the job.
    """

    enable_custom_service_account: bool = proto.Field(
        proto.BOOL,
        number=1,
    )
    service_account: str = proto.Field(
        proto.STRING,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
