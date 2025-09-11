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
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "MetadataStore",
    },
)


class MetadataStore(proto.Message):
    r"""Instance of a metadata store. Contains a set of metadata that
    can be queried.

    Attributes:
        name (str):
            Output only. The resource name of the
            MetadataStore instance.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            MetadataStore was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            MetadataStore was last updated.
        encryption_spec (google.cloud.aiplatform_v1.types.EncryptionSpec):
            Customer-managed encryption key spec for a
            Metadata Store. If set, this Metadata Store and
            all sub-resources of this Metadata Store are
            secured using this key.
        description (str):
            Description of the MetadataStore.
        state (google.cloud.aiplatform_v1.types.MetadataStore.MetadataStoreState):
            Output only. State information of the
            MetadataStore.
    """

    class MetadataStoreState(proto.Message):
        r"""Represents state information for a MetadataStore.

        Attributes:
            disk_utilization_bytes (int):
                The disk utilization of the MetadataStore in
                bytes.
        """

        disk_utilization_bytes: int = proto.Field(
            proto.INT64,
            number=1,
        )

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=5,
        message=gca_encryption_spec.EncryptionSpec,
    )
    description: str = proto.Field(
        proto.STRING,
        number=6,
    )
    state: MetadataStoreState = proto.Field(
        proto.MESSAGE,
        number=7,
        message=MetadataStoreState,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
