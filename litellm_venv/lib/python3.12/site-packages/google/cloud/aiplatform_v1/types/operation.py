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

from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "GenericOperationMetadata",
        "DeleteOperationMetadata",
    },
)


class GenericOperationMetadata(proto.Message):
    r"""Generic Metadata shared by all operations.

    Attributes:
        partial_failures (MutableSequence[google.rpc.status_pb2.Status]):
            Output only. Partial failures encountered.
            E.g. single files that couldn't be read.
            This field should never exceed 20 entries.
            Status details field will contain standard
            Google Cloud error details.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the operation was
            created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when the operation was
            updated for the last time. If the operation has
            finished (successfully or not), this is the
            finish time.
    """

    partial_failures: MutableSequence[status_pb2.Status] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=status_pb2.Status,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )


class DeleteOperationMetadata(proto.Message):
    r"""Details of operations that perform deletes of any entities.

    Attributes:
        generic_metadata (google.cloud.aiplatform_v1.types.GenericOperationMetadata):
            The common part of the operation metadata.
    """

    generic_metadata: "GenericOperationMetadata" = proto.Field(
        proto.MESSAGE,
        number=1,
        message="GenericOperationMetadata",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
