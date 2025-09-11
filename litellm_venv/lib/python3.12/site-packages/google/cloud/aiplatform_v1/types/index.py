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

from google.cloud.aiplatform_v1.types import deployed_index_ref
from google.cloud.aiplatform_v1.types import encryption_spec as gca_encryption_spec
from google.protobuf import struct_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "Index",
        "IndexDatapoint",
        "IndexStats",
    },
)


class Index(proto.Message):
    r"""A representation of a collection of database items organized
    in a way that allows for approximate nearest neighbor (a.k.a
    ANN) algorithms search.

    Attributes:
        name (str):
            Output only. The resource name of the Index.
        display_name (str):
            Required. The display name of the Index.
            The name can be up to 128 characters long and
            can consist of any UTF-8 characters.
        description (str):
            The description of the Index.
        metadata_schema_uri (str):
            Immutable. Points to a YAML file stored on Google Cloud
            Storage describing additional information about the Index,
            that is specific to it. Unset if the Index does not have any
            additional information. The schema is defined as an OpenAPI
            3.0.2 `Schema
            Object <https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#schemaObject>`__.
            Note: The URI given on output will be immutable and probably
            different, including the URI scheme, than the one given on
            input. The output URI will point to a location where the
            user only has a read access.
        metadata (google.protobuf.struct_pb2.Value):
            An additional information about the Index; the schema of the
            metadata can be found in
            [metadata_schema][google.cloud.aiplatform.v1.Index.metadata_schema_uri].
        deployed_indexes (MutableSequence[google.cloud.aiplatform_v1.types.DeployedIndexRef]):
            Output only. The pointers to DeployedIndexes
            created from this Index. An Index can be only
            deleted if all its DeployedIndexes had been
            undeployed first.
        etag (str):
            Used to perform consistent read-modify-write
            updates. If not set, a blind "overwrite" update
            happens.
        labels (MutableMapping[str, str]):
            The labels with user-defined metadata to
            organize your Indexes.
            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            and examples of labels.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Index was
            created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Index was most recently
            updated. This also includes any update to the contents of
            the Index. Note that Operations working on this Index may
            have their
            [Operations.metadata.generic_metadata.update_time]
            [google.cloud.aiplatform.v1.GenericOperationMetadata.update_time]
            a little after the value of this timestamp, yet that does
            not mean their results are not already reflected in the
            Index. Result of any successfully completed Operation on the
            Index is reflected in it.
        index_stats (google.cloud.aiplatform_v1.types.IndexStats):
            Output only. Stats of the index resource.
        index_update_method (google.cloud.aiplatform_v1.types.Index.IndexUpdateMethod):
            Immutable. The update method to use with this Index. If not
            set, BATCH_UPDATE will be used by default.
        encryption_spec (google.cloud.aiplatform_v1.types.EncryptionSpec):
            Immutable. Customer-managed encryption key
            spec for an Index. If set, this Index and all
            sub-resources of this Index will be secured by
            this key.
    """

    class IndexUpdateMethod(proto.Enum):
        r"""The update method of an Index.

        Values:
            INDEX_UPDATE_METHOD_UNSPECIFIED (0):
                Should not be used.
            BATCH_UPDATE (1):
                BatchUpdate: user can call UpdateIndex with
                files on Cloud Storage of Datapoints to update.
            STREAM_UPDATE (2):
                StreamUpdate: user can call
                UpsertDatapoints/DeleteDatapoints to update the
                Index and the updates will be applied in
                corresponding DeployedIndexes in nearly
                real-time.
        """
        INDEX_UPDATE_METHOD_UNSPECIFIED = 0
        BATCH_UPDATE = 1
        STREAM_UPDATE = 2

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
    metadata_schema_uri: str = proto.Field(
        proto.STRING,
        number=4,
    )
    metadata: struct_pb2.Value = proto.Field(
        proto.MESSAGE,
        number=6,
        message=struct_pb2.Value,
    )
    deployed_indexes: MutableSequence[
        deployed_index_ref.DeployedIndexRef
    ] = proto.RepeatedField(
        proto.MESSAGE,
        number=7,
        message=deployed_index_ref.DeployedIndexRef,
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
    index_stats: "IndexStats" = proto.Field(
        proto.MESSAGE,
        number=14,
        message="IndexStats",
    )
    index_update_method: IndexUpdateMethod = proto.Field(
        proto.ENUM,
        number=16,
        enum=IndexUpdateMethod,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=17,
        message=gca_encryption_spec.EncryptionSpec,
    )


class IndexDatapoint(proto.Message):
    r"""A datapoint of Index.

    Attributes:
        datapoint_id (str):
            Required. Unique identifier of the datapoint.
        feature_vector (MutableSequence[float]):
            Required. Feature embedding vector. An array of numbers with
            the length of [NearestNeighborSearchConfig.dimensions].
        restricts (MutableSequence[google.cloud.aiplatform_v1.types.IndexDatapoint.Restriction]):
            Optional. List of Restrict of the datapoint,
            used to perform "restricted searches" where
            boolean rule are used to filter the subset of
            the database eligible for matching. This uses
            categorical tokens. See:

            https://cloud.google.com/vertex-ai/docs/matching-engine/filtering
        numeric_restricts (MutableSequence[google.cloud.aiplatform_v1.types.IndexDatapoint.NumericRestriction]):
            Optional. List of Restrict of the datapoint,
            used to perform "restricted searches" where
            boolean rule are used to filter the subset of
            the database eligible for matching. This uses
            numeric comparisons.
        crowding_tag (google.cloud.aiplatform_v1.types.IndexDatapoint.CrowdingTag):
            Optional. CrowdingTag of the datapoint, the
            number of neighbors to return in each crowding
            can be configured during query.
    """

    class Restriction(proto.Message):
        r"""Restriction of a datapoint which describe its
        attributes(tokens) from each of several attribute
        categories(namespaces).

        Attributes:
            namespace (str):
                The namespace of this restriction. e.g.:
                color.
            allow_list (MutableSequence[str]):
                The attributes to allow in this namespace.
                e.g.: 'red'
            deny_list (MutableSequence[str]):
                The attributes to deny in this namespace.
                e.g.: 'blue'
        """

        namespace: str = proto.Field(
            proto.STRING,
            number=1,
        )
        allow_list: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=2,
        )
        deny_list: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=3,
        )

    class NumericRestriction(proto.Message):
        r"""This field allows restricts to be based on numeric
        comparisons rather than categorical tokens.

        This message has `oneof`_ fields (mutually exclusive fields).
        For each oneof, at most one member field can be set at the same time.
        Setting any member of the oneof automatically clears all other
        members.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            value_int (int):
                Represents 64 bit integer.

                This field is a member of `oneof`_ ``Value``.
            value_float (float):
                Represents 32 bit float.

                This field is a member of `oneof`_ ``Value``.
            value_double (float):
                Represents 64 bit float.

                This field is a member of `oneof`_ ``Value``.
            namespace (str):
                The namespace of this restriction. e.g.:
                cost.
            op (google.cloud.aiplatform_v1.types.IndexDatapoint.NumericRestriction.Operator):
                This MUST be specified for queries and must
                NOT be specified for datapoints.
        """

        class Operator(proto.Enum):
            r"""Which comparison operator to use.  Should be specified for
            queries only; specifying this for a datapoint is an error.

            Datapoints for which Operator is true relative to the query's
            Value field will be allowlisted.

            Values:
                OPERATOR_UNSPECIFIED (0):
                    Default value of the enum.
                LESS (1):
                    Datapoints are eligible iff their value is <
                    the query's.
                LESS_EQUAL (2):
                    Datapoints are eligible iff their value is <=
                    the query's.
                EQUAL (3):
                    Datapoints are eligible iff their value is ==
                    the query's.
                GREATER_EQUAL (4):
                    Datapoints are eligible iff their value is >=
                    the query's.
                GREATER (5):
                    Datapoints are eligible iff their value is >
                    the query's.
                NOT_EQUAL (6):
                    Datapoints are eligible iff their value is !=
                    the query's.
            """
            OPERATOR_UNSPECIFIED = 0
            LESS = 1
            LESS_EQUAL = 2
            EQUAL = 3
            GREATER_EQUAL = 4
            GREATER = 5
            NOT_EQUAL = 6

        value_int: int = proto.Field(
            proto.INT64,
            number=2,
            oneof="Value",
        )
        value_float: float = proto.Field(
            proto.FLOAT,
            number=3,
            oneof="Value",
        )
        value_double: float = proto.Field(
            proto.DOUBLE,
            number=4,
            oneof="Value",
        )
        namespace: str = proto.Field(
            proto.STRING,
            number=1,
        )
        op: "IndexDatapoint.NumericRestriction.Operator" = proto.Field(
            proto.ENUM,
            number=5,
            enum="IndexDatapoint.NumericRestriction.Operator",
        )

    class CrowdingTag(proto.Message):
        r"""Crowding tag is a constraint on a neighbor list produced by nearest
        neighbor search requiring that no more than some value k' of the k
        neighbors returned have the same value of crowding_attribute.

        Attributes:
            crowding_attribute (str):
                The attribute value used for crowding. The maximum number of
                neighbors to return per crowding attribute value
                (per_crowding_attribute_num_neighbors) is configured
                per-query. This field is ignored if
                per_crowding_attribute_num_neighbors is larger than the
                total number of neighbors to return for a given query.
        """

        crowding_attribute: str = proto.Field(
            proto.STRING,
            number=1,
        )

    datapoint_id: str = proto.Field(
        proto.STRING,
        number=1,
    )
    feature_vector: MutableSequence[float] = proto.RepeatedField(
        proto.FLOAT,
        number=2,
    )
    restricts: MutableSequence[Restriction] = proto.RepeatedField(
        proto.MESSAGE,
        number=4,
        message=Restriction,
    )
    numeric_restricts: MutableSequence[NumericRestriction] = proto.RepeatedField(
        proto.MESSAGE,
        number=6,
        message=NumericRestriction,
    )
    crowding_tag: CrowdingTag = proto.Field(
        proto.MESSAGE,
        number=5,
        message=CrowdingTag,
    )


class IndexStats(proto.Message):
    r"""Stats of the Index.

    Attributes:
        vectors_count (int):
            Output only. The number of vectors in the
            Index.
        shards_count (int):
            Output only. The number of shards in the
            Index.
    """

    vectors_count: int = proto.Field(
        proto.INT64,
        number=1,
    )
    shards_count: int = proto.Field(
        proto.INT32,
        number=2,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
