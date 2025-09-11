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

from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore

__protobuf__ = proto.module(
    package="google.ai.generativelanguage.v1beta",
    manifest={
        "Corpus",
        "Document",
        "StringList",
        "CustomMetadata",
        "MetadataFilter",
        "Condition",
        "Chunk",
        "ChunkData",
    },
)


class Corpus(proto.Message):
    r"""A ``Corpus`` is a collection of ``Document``\ s. A project can
    create up to 5 corpora.

    Attributes:
        name (str):
            Immutable. Identifier. The ``Corpus`` resource name. The ID
            (name excluding the "corpora/" prefix) can contain up to 40
            characters that are lowercase alphanumeric or dashes (-).
            The ID cannot start or end with a dash. If the name is empty
            on create, a unique name will be derived from
            ``display_name`` along with a 12 character random suffix.
            Example: ``corpora/my-awesome-corpora-123a456b789c``
        display_name (str):
            Optional. The human-readable display name for the
            ``Corpus``. The display name must be no more than 512
            characters in length, including spaces. Example: "Docs on
            Semantic Retriever".
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The Timestamp of when the ``Corpus`` was
            created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The Timestamp of when the ``Corpus`` was last
            updated.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
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


class Document(proto.Message):
    r"""A ``Document`` is a collection of ``Chunk``\ s. A ``Corpus`` can
    have a maximum of 10,000 ``Document``\ s.

    Attributes:
        name (str):
            Immutable. Identifier. The ``Document`` resource name. The
            ID (name excluding the `corpora/*/documents/` prefix) can
            contain up to 40 characters that are lowercase alphanumeric
            or dashes (-). The ID cannot start or end with a dash. If
            the name is empty on create, a unique name will be derived
            from ``display_name`` along with a 12 character random
            suffix. Example:
            ``corpora/{corpus_id}/documents/my-awesome-doc-123a456b789c``
        display_name (str):
            Optional. The human-readable display name for the
            ``Document``. The display name must be no more than 512
            characters in length, including spaces. Example: "Semantic
            Retriever Documentation".
        custom_metadata (MutableSequence[google.ai.generativelanguage_v1beta.types.CustomMetadata]):
            Optional. User provided custom metadata stored as key-value
            pairs used for querying. A ``Document`` can have a maximum
            of 20 ``CustomMetadata``.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The Timestamp of when the ``Document`` was last
            updated.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The Timestamp of when the ``Document`` was
            created.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    custom_metadata: MutableSequence["CustomMetadata"] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="CustomMetadata",
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )


class StringList(proto.Message):
    r"""User provided string values assigned to a single metadata
    key.

    Attributes:
        values (MutableSequence[str]):
            The string values of the metadata to store.
    """

    values: MutableSequence[str] = proto.RepeatedField(
        proto.STRING,
        number=1,
    )


class CustomMetadata(proto.Message):
    r"""User provided metadata stored as key-value pairs.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        string_value (str):
            The string value of the metadata to store.

            This field is a member of `oneof`_ ``value``.
        string_list_value (google.ai.generativelanguage_v1beta.types.StringList):
            The StringList value of the metadata to
            store.

            This field is a member of `oneof`_ ``value``.
        numeric_value (float):
            The numeric value of the metadata to store.

            This field is a member of `oneof`_ ``value``.
        key (str):
            Required. The key of the metadata to store.
    """

    string_value: str = proto.Field(
        proto.STRING,
        number=2,
        oneof="value",
    )
    string_list_value: "StringList" = proto.Field(
        proto.MESSAGE,
        number=6,
        oneof="value",
        message="StringList",
    )
    numeric_value: float = proto.Field(
        proto.FLOAT,
        number=7,
        oneof="value",
    )
    key: str = proto.Field(
        proto.STRING,
        number=1,
    )


class MetadataFilter(proto.Message):
    r"""User provided filter to limit retrieval based on ``Chunk`` or
    ``Document`` level metadata values. Example (genre = drama OR genre
    = action): key = "document.custom_metadata.genre" conditions =
    [{string_value = "drama", operation = EQUAL}, {string_value =
    "action", operation = EQUAL}]

    Attributes:
        key (str):
            Required. The key of the metadata to filter
            on.
        conditions (MutableSequence[google.ai.generativelanguage_v1beta.types.Condition]):
            Required. The ``Condition``\ s for the given key that will
            trigger this filter. Multiple ``Condition``\ s are joined by
            logical ORs.
    """

    key: str = proto.Field(
        proto.STRING,
        number=1,
    )
    conditions: MutableSequence["Condition"] = proto.RepeatedField(
        proto.MESSAGE,
        number=2,
        message="Condition",
    )


class Condition(proto.Message):
    r"""Filter condition applicable to a single key.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        string_value (str):
            The string value to filter the metadata on.

            This field is a member of `oneof`_ ``value``.
        numeric_value (float):
            The numeric value to filter the metadata on.

            This field is a member of `oneof`_ ``value``.
        operation (google.ai.generativelanguage_v1beta.types.Condition.Operator):
            Required. Operator applied to the given
            key-value pair to trigger the condition.
    """

    class Operator(proto.Enum):
        r"""Defines the valid operators that can be applied to a
        key-value pair.

        Values:
            OPERATOR_UNSPECIFIED (0):
                The default value. This value is unused.
            LESS (1):
                Supported by numeric.
            LESS_EQUAL (2):
                Supported by numeric.
            EQUAL (3):
                Supported by numeric & string.
            GREATER_EQUAL (4):
                Supported by numeric.
            GREATER (5):
                Supported by numeric.
            NOT_EQUAL (6):
                Supported by numeric & string.
            INCLUDES (7):
                Supported by string only when ``CustomMetadata`` value type
                for the given key has a ``string_list_value``.
            EXCLUDES (8):
                Supported by string only when ``CustomMetadata`` value type
                for the given key has a ``string_list_value``.
        """
        OPERATOR_UNSPECIFIED = 0
        LESS = 1
        LESS_EQUAL = 2
        EQUAL = 3
        GREATER_EQUAL = 4
        GREATER = 5
        NOT_EQUAL = 6
        INCLUDES = 7
        EXCLUDES = 8

    string_value: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="value",
    )
    numeric_value: float = proto.Field(
        proto.FLOAT,
        number=6,
        oneof="value",
    )
    operation: Operator = proto.Field(
        proto.ENUM,
        number=5,
        enum=Operator,
    )


class Chunk(proto.Message):
    r"""A ``Chunk`` is a subpart of a ``Document`` that is treated as an
    independent unit for the purposes of vector representation and
    storage. A ``Corpus`` can have a maximum of 1 million ``Chunk``\ s.

    Attributes:
        name (str):
            Immutable. Identifier. The ``Chunk`` resource name. The ID
            (name excluding the `corpora/*/documents/*/chunks/` prefix)
            can contain up to 40 characters that are lowercase
            alphanumeric or dashes (-). The ID cannot start or end with
            a dash. If the name is empty on create, a random
            12-character unique ID will be generated. Example:
            ``corpora/{corpus_id}/documents/{document_id}/chunks/123a456b789c``
        data (google.ai.generativelanguage_v1beta.types.ChunkData):
            Required. The content for the ``Chunk``, such as the text
            string. The maximum number of tokens per chunk is 2043.
        custom_metadata (MutableSequence[google.ai.generativelanguage_v1beta.types.CustomMetadata]):
            Optional. User provided custom metadata stored as key-value
            pairs. The maximum number of ``CustomMetadata`` per chunk is
            20.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The Timestamp of when the ``Chunk`` was
            created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. The Timestamp of when the ``Chunk`` was last
            updated.
        state (google.ai.generativelanguage_v1beta.types.Chunk.State):
            Output only. Current state of the ``Chunk``.
    """

    class State(proto.Enum):
        r"""States for the lifecycle of a ``Chunk``.

        Values:
            STATE_UNSPECIFIED (0):
                The default value. This value is used if the
                state is omitted.
            STATE_PENDING_PROCESSING (1):
                ``Chunk`` is being processed (embedding and vector storage).
            STATE_ACTIVE (2):
                ``Chunk`` is processed and available for querying.
            STATE_FAILED (10):
                ``Chunk`` failed processing.
        """
        STATE_UNSPECIFIED = 0
        STATE_PENDING_PROCESSING = 1
        STATE_ACTIVE = 2
        STATE_FAILED = 10

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    data: "ChunkData" = proto.Field(
        proto.MESSAGE,
        number=2,
        message="ChunkData",
    )
    custom_metadata: MutableSequence["CustomMetadata"] = proto.RepeatedField(
        proto.MESSAGE,
        number=3,
        message="CustomMetadata",
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=5,
        message=timestamp_pb2.Timestamp,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=6,
        enum=State,
    )


class ChunkData(proto.Message):
    r"""Extracted data that represents the ``Chunk`` content.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        string_value (str):
            The ``Chunk`` content as a string. The maximum number of
            tokens per chunk is 2043.

            This field is a member of `oneof`_ ``data``.
    """

    string_value: str = proto.Field(
        proto.STRING,
        number=1,
        oneof="data",
    )


__all__ = tuple(sorted(__protobuf__.manifest))
