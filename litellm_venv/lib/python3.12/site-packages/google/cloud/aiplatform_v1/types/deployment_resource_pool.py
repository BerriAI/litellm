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
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "DeploymentResourcePool",
    },
)


class DeploymentResourcePool(proto.Message):
    r"""A description of resources that can be shared by multiple
    DeployedModels, whose underlying specification consists of a
    DedicatedResources.

    Attributes:
        name (str):
            Immutable. The resource name of the DeploymentResourcePool.
            Format:
            ``projects/{project}/locations/{location}/deploymentResourcePools/{deployment_resource_pool}``
        dedicated_resources (google.cloud.aiplatform_v1.types.DedicatedResources):
            Required. The underlying DedicatedResources
            that the DeploymentResourcePool uses.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            DeploymentResourcePool was created.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    dedicated_resources: machine_resources.DedicatedResources = proto.Field(
        proto.MESSAGE,
        number=2,
        message=machine_resources.DedicatedResources,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
