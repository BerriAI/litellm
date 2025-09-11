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

from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "DatasetVersion",
    },
)


class DatasetVersion(proto.Message):
    r"""Describes the dataset version.

    Attributes:
        name (str):
            Output only. The resource name of the
            DatasetVersion.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            DatasetVersion was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            DatasetVersion was last updated.
        etag (str):
            Used to perform consistent read-modify-write
            updates. If not set, a blind "overwrite" update
            happens.
        big_query_dataset_name (str):
            Output only. Name of the associated BigQuery
            dataset.
        display_name (str):
            The user-defined name of the DatasetVersion.
            The name can be up to 128 characters long and
            can consist of any UTF-8 characters.
        metadata (google.protobuf.struct_pb2.Value):
            Required. Output only. Additional information
            about the DatasetVersion.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=2,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=3,
    )
    big_query_dataset_name: str = proto.Field(
        proto.STRING,
        number=4,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=7,
    )
    metadata: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=8,
        message=struct_pb2.Value,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
