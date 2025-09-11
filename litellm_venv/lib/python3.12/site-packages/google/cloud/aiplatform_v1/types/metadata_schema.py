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


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "MetadataSchema",
    },
)


class MetadataSchema(proto.Message):
    r"""Instance of a general MetadataSchema.

    Attributes:
        name (str):
            Output only. The resource name of the
            MetadataSchema.
        schema_version (str):
            The version of the MetadataSchema. The version's format must
            match the following regular expression:
            ``^[0-9]+[.][0-9]+[.][0-9]+$``, which would allow to
            order/compare different versions. Example: 1.0.0, 1.0.1,
            etc.
        schema (str):
            Required. The raw YAML string representation of the
            MetadataSchema. The combination of [MetadataSchema.version]
            and the schema name given by ``title`` in
            [MetadataSchema.schema] must be unique within a
            MetadataStore.

            The schema is defined as an OpenAPI 3.0.2 `MetadataSchema
            Object <https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.2.md#schemaObject>`__
        schema_type (google.cloud.aiplatform_v1.types.MetadataSchema.MetadataSchemaType):
            The type of the MetadataSchema. This is a
            property that identifies which metadata types
            will use the MetadataSchema.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            MetadataSchema was created.
        description (str):
            Description of the Metadata Schema
    """

    class MetadataSchemaType(proto.Enum):
        r"""Describes the type of the MetadataSchema.

        Values:
            METADATA_SCHEMA_TYPE_UNSPECIFIED (0):
                Unspecified type for the MetadataSchema.
            ARTIFACT_TYPE (1):
                A type indicating that the MetadataSchema
                will be used by Artifacts.
            EXECUTION_TYPE (2):
                A typee indicating that the MetadataSchema
                will be used by Executions.
            CONTEXT_TYPE (3):
                A state indicating that the MetadataSchema
                will be used by Contexts.
        """
        METADATA_SCHEMA_TYPE_UNSPECIFIED = 0
        ARTIFACT_TYPE = 1
        EXECUTION_TYPE = 2
        CONTEXT_TYPE = 3

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    schema_version: str = proto.Field(
        proto.STRING,
        number=2,
    )
    schema: str = proto.Field(
        proto.STRING,
        number=3,
    )
    schema_type: MetadataSchemaType = proto.Field(
        proto.ENUM,
        number=4,
        enum=MetadataSchemaType,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    description: str = proto.Field(
        proto.STRING,
        number=6,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
