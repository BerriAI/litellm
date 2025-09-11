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
        "DataItem",
    },
)


class DataItem(proto.Message):
    r"""A piece of data in a Dataset. Could be an image, a video, a
    document or plain text.

    Attributes:
        name (str):
            Output only. The resource name of the
            DataItem.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this DataItem was
            created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this DataItem was
            last updated.
        labels (MutableMapping[str, str]):
            Optional. The labels with user-defined
            metadata to organize your DataItems.
            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed. No more than 64 user labels can be
            associated with one DataItem(System labels are
            excluded).

            See https://goo.gl/xmQnxf for more information
            and examples of labels. System reserved label
            keys are prefixed with
            "aiplatform.googleapis.com/" and are immutable.
        payload (google.protobuf.struct_pb2.Value):
            Required. The data that the DataItem represents (for
            example, an image or a text snippet). The schema of the
            payload is stored in the parent Dataset's [metadata
            schema's][google.cloud.aiplatform.v1beta1.Dataset.metadata_schema_uri]
            dataItemSchemaUri field.
        etag (str):
            Optional. Used to perform consistent
            read-modify-write updates. If not set, a blind
            "overwrite" update happens.
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
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=3,
    )
    payload: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=4,
        message=struct_pb2.Value,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=7,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
