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

from google.cloud.aiplatform_v1.types import featurestore_monitoring
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "EntityType",
    },
)


class EntityType(proto.Message):
    r"""An entity type is a type of object in a system that needs to
    be modeled and have stored information about. For example,
    driver is an entity type, and driver0 is an instance of an
    entity type driver.

    Attributes:
        name (str):
            Immutable. Name of the EntityType. Format:
            ``projects/{project}/locations/{location}/featurestores/{featurestore}/entityTypes/{entity_type}``

            The last part entity_type is assigned by the client. The
            entity_type can be up to 64 characters long and can consist
            only of ASCII Latin letters A-Z and a-z and underscore(_),
            and ASCII digits 0-9 starting with a letter. The value will
            be unique given a featurestore.
        description (str):
            Optional. Description of the EntityType.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this EntityType
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this EntityType
            was most recently updated.
        labels (MutableMapping[str, str]):
            Optional. The labels with user-defined
            metadata to organize your EntityTypes.

            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            on and examples of labels. No more than 64 user
            labels can be associated with one EntityType
            (System labels are excluded)."
            System reserved label keys are prefixed with
            "aiplatform.googleapis.com/" and are immutable.
        etag (str):
            Optional. Used to perform a consistent
            read-modify-write updates. If not set, a blind
            "overwrite" update happens.
        monitoring_config (google.cloud.aiplatform_v1.types.FeaturestoreMonitoringConfig):
            Optional. The default monitoring configuration for all
            Features with value type
            ([Feature.ValueType][google.cloud.aiplatform.v1.Feature.ValueType])
            BOOL, STRING, DOUBLE or INT64 under this EntityType.

            If this is populated with
            [FeaturestoreMonitoringConfig.monitoring_interval]
            specified, snapshot analysis monitoring is enabled.
            Otherwise, snapshot analysis monitoring is disabled.
        offline_storage_ttl_days (int):
            Optional. Config for data retention policy in offline
            storage. TTL in days for feature values that will be stored
            in offline storage. The Feature Store offline storage
            periodically removes obsolete feature values older than
            ``offline_storage_ttl_days`` since the feature generation
            time. If unset (or explicitly set to 0), default to 4000
            days TTL.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    description: str = proto.Field(
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
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=6,
    )
    etag: str = proto.Field(
        proto.STRING,
        number=7,
    )
    monitoring_config: featurestore_monitoring.FeaturestoreMonitoringConfig = (
        proto.Field(
            proto.MESSAGE,
            number=8,
            message=featurestore_monitoring.FeaturestoreMonitoringConfig,
        )
    )
    offline_storage_ttl_days: int = proto.Field(
        proto.INT32,
        number=10,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
