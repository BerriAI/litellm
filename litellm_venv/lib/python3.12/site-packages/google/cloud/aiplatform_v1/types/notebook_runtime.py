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

from google.cloud.aiplatform_v1.types import machine_resources
from google.cloud.aiplatform_v1.types import network_spec as gca_network_spec
from google.cloud.aiplatform_v1.types import notebook_euc_config
from google.cloud.aiplatform_v1.types import notebook_idle_shutdown_config
from google.cloud.aiplatform_v1.types import (
    notebook_runtime_template_ref as gca_notebook_runtime_template_ref,
)
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "NotebookRuntimeType",
        "NotebookRuntimeTemplate",
        "NotebookRuntime",
    },
)


class NotebookRuntimeType(proto.Enum):
    r"""Represents a notebook runtime type.

    Values:
        NOTEBOOK_RUNTIME_TYPE_UNSPECIFIED (0):
            Unspecified notebook runtime type, NotebookRuntimeType will
            default to USER_DEFINED.
        USER_DEFINED (1):
            runtime or template with coustomized
            configurations from user.
        ONE_CLICK (2):
            runtime or template with system defined
            configurations.
    """
    NOTEBOOK_RUNTIME_TYPE_UNSPECIFIED = 0
    USER_DEFINED = 1
    ONE_CLICK = 2


class NotebookRuntimeTemplate(proto.Message):
    r"""A template that specifies runtime configurations such as
    machine type, runtime version, network configurations, etc.
    Multiple runtimes can be created from a runtime template.

    Attributes:
        name (str):
            Output only. The resource name of the
            NotebookRuntimeTemplate.
        display_name (str):
            Required. The display name of the
            NotebookRuntimeTemplate. The name can be up to
            128 characters long and can consist of any UTF-8
            characters.
        description (str):
            The description of the
            NotebookRuntimeTemplate.
        is_default (bool):
            Output only. The default template to use if
            not specified.
        machine_spec (google.cloud.aiplatform_v1.types.MachineSpec):
            Optional. Immutable. The specification of a
            single machine for the template.
        data_persistent_disk_spec (google.cloud.aiplatform_v1.types.PersistentDiskSpec):
            Optional. The specification of [persistent
            disk][https://cloud.google.com/compute/docs/disks/persistent-disks]
            attached to the runtime as data disk storage.
        network_spec (google.cloud.aiplatform_v1.types.NetworkSpec):
            Optional. Network spec.
        service_account (str):
            The service account that the runtime workload runs as. You
            can use any service account within the same project, but you
            must have the service account user permission to use the
            instance.

            If not specified, the `Compute Engine default service
            account <https://cloud.google.com/compute/docs/access/service-accounts#default_service_account>`__
            is used.
        etag (str):
            Used to perform consistent read-modify-write
            updates. If not set, a blind "overwrite" update
            happens.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to
            organize the NotebookRuntimeTemplates.

            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            and examples of labels.
        idle_shutdown_config (google.cloud.aiplatform_v1.types.NotebookIdleShutdownConfig):
            The idle shutdown configuration of
            NotebookRuntimeTemplate. This config will only
            be set when idle shutdown is enabled.
        euc_config (google.cloud.aiplatform_v1.types.NotebookEucConfig):
            EUC configuration of the
            NotebookRuntimeTemplate.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            NotebookRuntimeTemplate was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            NotebookRuntimeTemplate was most recently
            updated.
        notebook_runtime_type (google.cloud.aiplatform_v1.types.NotebookRuntimeType):
            Optional. Immutable. The type of the notebook
            runtime template.
        shielded_vm_config (google.cloud.aiplatform_v1.types.ShieldedVmConfig):
            Optional. Immutable. Runtime Shielded VM
            spec.
        network_tags (MutableSequence[str]):
            Optional. The Compute Engine tags to add to runtime (see
            `Tagging
            instances <https://cloud.google.com/vpc/docs/add-remove-network-tags>`__).
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    description: str = proto.Field(
        proto.STRING,
        number=3,
    )
    is_default: bool = proto.Field(
        proto.BOOL,
        number=4,
    )
    machine_spec: machine_resources.MachineSpec = proto.Field(
        proto.MESSAGE,
        number=5,
        message=machine_resources.MachineSpec,
    )
    data_persistent_disk_spec: machine_resources.PersistentDiskSpec = proto.Field(
        proto.MESSAGE,
        number=8,
        message=machine_resources.PersistentDiskSpec,
    )
    network_spec: gca_network_spec.NetworkSpec = proto.Field(
        proto.MESSAGE,
        number=12,
        message=gca_network_spec.NetworkSpec,
    )
    service_account: str = proto.Field(
        proto.STRING,
        number=13,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=14,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=15,
    )
    idle_shutdown_config: notebook_idle_shutdown_config.NotebookIdleShutdownConfig = (
        proto.Field(
            proto.MESSAGE,
            number=17,
            message=notebook_idle_shutdown_config.NotebookIdleShutdownConfig,
        )
    )
    euc_config: notebook_euc_config.NotebookEucConfig = proto.Field(
        proto.MESSAGE,
        number=18,
        message=notebook_euc_config.NotebookEucConfig,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=10,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=11,
        message=timestamp_pb2.Timestamp,
    )
    notebook_runtime_type: "NotebookRuntimeType" = proto.Field(
        proto.ENUM,
        number=19,
        enum="NotebookRuntimeType",
    )
    shielded_vm_config: machine_resources.ShieldedVmConfig = proto.Field(
        proto.MESSAGE,
        number=20,
        message=machine_resources.ShieldedVmConfig,
    )
    network_tags: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=21,
    )


class NotebookRuntime(proto.Message):
    r"""A runtime is a virtual machine allocated to a particular user
    for a particular Notebook file on temporary basis with lifetime
    limited to 24 hours.

    Attributes:
        name (str):
            Output only. The resource name of the
            NotebookRuntime.
        runtime_user (str):
            Required. The user email of the
            NotebookRuntime.
        notebook_runtime_template_ref (google.cloud.aiplatform_v1.types.NotebookRuntimeTemplateRef):
            Output only. The pointer to
            NotebookRuntimeTemplate this NotebookRuntime is
            created from.
        proxy_uri (str):
            Output only. The proxy endpoint used to
            access the NotebookRuntime.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            NotebookRuntime was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            NotebookRuntime was most recently updated.
        health_state (google.cloud.aiplatform_v1.types.NotebookRuntime.HealthState):
            Output only. The health state of the
            NotebookRuntime.
        display_name (str):
            Required. The display name of the
            NotebookRuntime. The name can be up to 128
            characters long and can consist of any UTF-8
            characters.
        description (str):
            The description of the NotebookRuntime.
        service_account (str):
            Output only. The service account that the
            NotebookRuntime workload runs as.
        runtime_state (google.cloud.aiplatform_v1.types.NotebookRuntime.RuntimeState):
            Output only. The runtime (instance) state of
            the NotebookRuntime.
        is_upgradable (bool):
            Output only. Whether NotebookRuntime is
            upgradable.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to organize your
            NotebookRuntime.

            Label keys and values can be no longer than 64 characters
            (Unicode codepoints), can only contain lowercase letters,
            numeric characters, underscores and dashes. International
            characters are allowed. No more than 64 user labels can be
            associated with one NotebookRuntime (System labels are
            excluded).

            See https://goo.gl/xmQnxf for more information and examples
            of labels. System reserved label keys are prefixed with
            "aiplatform.googleapis.com/" and are immutable. Following
            system labels exist for NotebookRuntime:

            -  "aiplatform.googleapis.com/notebook_runtime_gce_instance_id":
               output only, its value is the Compute Engine instance id.
            -  "aiplatform.googleapis.com/colab_enterprise_entry_service":
               its value is either "bigquery" or "vertex"; if absent, it
               should be "vertex". This is to describe the entry
               service, either BigQuery or Vertex.
        expiration_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            NotebookRuntime will be expired:

            1. System Predefined NotebookRuntime: 24 hours
                after creation. After expiration, system
                predifined runtime will be deleted.
            2. User created NotebookRuntime: 6 months after
                last upgrade. After expiration, user created
                runtime will be stopped and allowed for
                upgrade.
        version (str):
            Output only. The VM os image version of
            NotebookRuntime.
        notebook_runtime_type (google.cloud.aiplatform_v1.types.NotebookRuntimeType):
            Output only. The type of the notebook
            runtime.
        network_tags (MutableSequence[str]):
            Optional. The Compute Engine tags to add to runtime (see
            `Tagging
            instances <https://cloud.google.com/vpc/docs/add-remove-network-tags>`__).
    """

    class HealthState(proto.Enum):
        r"""The substate of the NotebookRuntime to display health
        information.

        Values:
            HEALTH_STATE_UNSPECIFIED (0):
                Unspecified health state.
            HEALTHY (1):
                NotebookRuntime is in healthy state. Applies
                to ACTIVE state.
            UNHEALTHY (2):
                NotebookRuntime is in unhealthy state.
                Applies to ACTIVE state.
        """
        HEALTH_STATE_UNSPECIFIED = 0
        HEALTHY = 1
        UNHEALTHY = 2

    class RuntimeState(proto.Enum):
        r"""The substate of the NotebookRuntime to display state of
        runtime. The resource of NotebookRuntime is in ACTIVE state for
        these sub state.

        Values:
            RUNTIME_STATE_UNSPECIFIED (0):
                Unspecified runtime state.
            RUNNING (1):
                NotebookRuntime is in running state.
            BEING_STARTED (2):
                NotebookRuntime is in starting state.
            BEING_STOPPED (3):
                NotebookRuntime is in stopping state.
            STOPPED (4):
                NotebookRuntime is in stopped state.
            BEING_UPGRADED (5):
                NotebookRuntime is in upgrading state. It is
                in the middle of upgrading process.
            ERROR (100):
                NotebookRuntime was unable to start/stop
                properly.
            INVALID (101):
                NotebookRuntime is in invalid state. Cannot
                be recovered.
        """
        RUNTIME_STATE_UNSPECIFIED = 0
        RUNNING = 1
        BEING_STARTED = 2
        BEING_STOPPED = 3
        STOPPED = 4
        BEING_UPGRADED = 5
        ERROR = 100
        INVALID = 101

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    runtime_user: str = proto.Field(
        proto.STRING,
        number=2,
    )
    notebook_runtime_template_ref: gca_notebook_runtime_template_ref.NotebookRuntimeTemplateRef = proto.Field(
        proto.MESSAGE,
        number=3,
        message=gca_notebook_runtime_template_ref.NotebookRuntimeTemplateRef,
    )
    proxy_uri: str = proto.Field(
        proto.STRING,
        number=5,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    health_state: HealthState = proto.Field(
        proto.ENUM,
        number=8,
        enum=HealthState,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=10,
    )
    description: str = proto.Field(
        proto.STRING,
        number=11,
    )
    service_account: str = proto.Field(
        proto.STRING,
        number=13,
    )
    runtime_state: RuntimeState = proto.Field(
        proto.ENUM,
        number=14,
        enum=RuntimeState,
    )
    is_upgradable: bool = proto.Field(
        proto.BOOL,
        number=15,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=16,
    )
    expiration_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=17,
        message=timestamp_pb2.Timestamp,
    )
    version: str = proto.Field(
        proto.STRING,
        number=18,
    )
    notebook_runtime_type: "NotebookRuntimeType" = proto.Field(
        proto.ENUM,
        number=19,
        enum="NotebookRuntimeType",
    )
    network_tags: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=25,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
