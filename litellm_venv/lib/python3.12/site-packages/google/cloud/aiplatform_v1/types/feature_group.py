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

from google.cloud.aiplatform_v1.types import io
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "FeatureGroup",
    },
)


class FeatureGroup(proto.Message):
    r"""Vertex AI Feature Group.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        big_query (google.cloud.aiplatform_v1.types.FeatureGroup.BigQuery):
            Indicates that features for this group come from BigQuery
            Table/View. By default treats the source as a sparse time
            series source, which is required to have an entity_id and a
            feature_timestamp column in the source.

            This field is a member of `oneof`_ ``source``.
        name (str):
            Identifier. Name of the FeatureGroup. Format:
            ``projects/{project}/locations/{location}/featureGroups/{featureGroup}``
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this FeatureGroup
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this FeatureGroup
            was last updated.
        etag (str):
            Optional. Used to perform consistent
            read-modify-write updates. If not set, a blind
            "overwrite" update happens.
        labels (MutableMapping[str, str]):
            Optional. The labels with user-defined
            metadata to organize your FeatureGroup.

            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            on and examples of labels. No more than 64 user
            labels can be associated with one
            FeatureGroup(System labels are excluded)."
            System reserved label keys are prefixed with
            "aiplatform.googleapis.com/" and are immutable.
        description (str):
            Optional. Description of the FeatureGroup.
    """

    class BigQuery(proto.Message):
        r"""Input source type for BigQuery Tables and Views.

        Attributes:
            big_query_source (google.cloud.aiplatform_v1.types.BigQuerySource):
                Required. Immutable. The BigQuery source URI
                that points to either a BigQuery Table or View.
            entity_id_columns (MutableSequence[str]):
                Optional. Columns to construct entity_id / row keys. If not
                provided defaults to ``entity_id``.
        """

        big_query_source: io.BigQuerySource = proto.Field(
            proto.MESSAGE,
            number=1,
            message=io.BigQuerySource,
        )
        entity_id_columns: MutableSequence[str] = proto.RepeatedField(
            proto.STRING,
            number=2,
        )

    big_query: BigQuery = proto.Field(
        proto.MESSAGE,
        number=7,
        oneof="source",
        message=BigQuery,
    )
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
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=4,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=5,
    )
    description: str = proto.Field(
        proto.STRING,
        number=6,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
