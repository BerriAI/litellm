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
        "Execution",
    },
)


class Execution(proto.Message):
    r"""Instance of a general execution.

    Attributes:
        name (str):
            Output only. The resource name of the
            Execution.
        display_name (str):
            User provided display name of the Execution.
            May be up to 128 Unicode characters.
        state (google.cloud.aiplatform_v1beta1.types.Execution.State):
            The state of this Execution. This is a
            property of the Execution, and does not imply or
            capture any ongoing process. This property is
            managed by clients (such as Vertex AI Pipelines)
            and the system does not prescribe or check the
            validity of state transitions.
        etag (str):
            An eTag used to perform consistent
            read-modify-write updates. If not set, a blind
            "overwrite" update happens.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to
            organize your Executions.
            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed. No more than 64 user labels can be
            associated with one Execution (System labels are
            excluded).
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Execution
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Execution
            was last updated.
        schema_title (str):
            The title of the schema describing the
            metadata.
            Schema title and version is expected to be
            registered in earlier Create Schema calls. And
            both are used together as unique identifiers to
            identify schemas within the local metadata
            store.
        schema_version (str):
            The version of the schema in ``schema_title`` to use.

            Schema title and version is expected to be registered in
            earlier Create Schema calls. And both are used together as
            unique identifiers to identify schemas within the local
            metadata store.
        metadata (google.protobuf.struct_pb2.Struct):
            Properties of the Execution.
            Top level metadata keys' heading and trailing
            spaces will be trimmed. The size of this field
            should not exceed 200KB.
        description (str):
            Description of the Execution
    """

    class State(proto.Enum):
        r"""Describes the state of the Execution.

        Values:
            STATE_UNSPECIFIED (0):
                Unspecified Execution state
            NEW (1):
                The Execution is new
            RUNNING (2):
                The Execution is running
            COMPLETE (3):
                The Execution has finished running
            FAILED (4):
                The Execution has failed
            CACHED (5):
                The Execution completed through Cache hit.
            CANCELLED (6):
                The Execution was cancelled.
        """
        STATE_UNSPECIFIED = 0
        NEW = 1
        RUNNING = 2
        COMPLETE = 3
        FAILED = 4
        CACHED = 5
        CANCELLED = 6

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=6,
        enum=State,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=9,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=10,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=11,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=12,
        message=timestamp_pb2.Timestamp,
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
