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
        "Context",
    },
)


class Context(proto.Message):
    r"""Instance of a general context.

    Attributes:
        name (str):
            Immutable. The resource name of the Context.
        display_name (str):
            User provided display name of the Context.
            May be up to 128 Unicode characters.
        etag (str):
            An eTag used to perform consistent
            read-modify-write updates. If not set, a blind
            "overwrite" update happens.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to
            organize your Contexts.
            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed. No more than 64 user labels can be
            associated with one Context (System labels are
            excluded).
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Context was
            created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Context was
            last updated.
        parent_contexts (MutableSequence[str]):
            Output only. A list of resource names of Contexts that are
            parents of this Context. A Context may have at most 10
            parent_contexts.
        schema_title (str):
            The title of the schema describing the
            metadata.
            Schema title and version is expected to be
            registered in earlier Create Schema calls. And
            both are used together as unique identifiers to
            identify schemas within the local metadata
            store.
        schema_version (str):
            The version of the schema in schema_name to use.

            Schema title and version is expected to be registered in
            earlier Create Schema calls. And both are used together as
            unique identifiers to identify schemas within the local
            metadata store.
        metadata (google.protobuf.struct_pb2.Struct):
            Properties of the Context.
            Top level metadata keys' heading and trailing
            spaces will be trimmed. The size of this field
            should not exceed 200KB.
        description (str):
            Description of the Context
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=8,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=9,
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
    parent_contexts: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=12,
    )
    schema_title: str = proto.Field(
        proto.STRING,
        number=13,
    )
    schema_version: str = proto.Field(
        proto.STRING,
        number=14,
    )
    metadata: struct_pb2.Struct = proto.Field(
        proto.MESSAGE,
        number=15,
        message=struct_pb2.Struct,
    )
    description: str = proto.Field(
        proto.STRING,
        number=16,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
