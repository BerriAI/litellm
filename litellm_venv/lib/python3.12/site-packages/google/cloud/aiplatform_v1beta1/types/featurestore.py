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

from google.cloud.aiplatform_v1beta1.types import encryption_spec as gca_encryption_spec
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "Featurestore",
    },
)


class Featurestore(proto.Message):
    r"""Vertex AI Feature Store provides a centralized repository for
    organizing, storing, and serving ML features. The Featurestore
    is a top-level container for your features and their values.

    Attributes:
        name (str):
            Output only. Name of the Featurestore. Format:
            ``projects/{project}/locations/{location}/featurestores/{featurestore}``
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Featurestore
            was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Featurestore
            was last updated.
        etag (str):
            Optional. Used to perform consistent
            read-modify-write updates. If not set, a blind
            "overwrite" update happens.
        labels (MutableMapping[str, str]):
            Optional. The labels with user-defined
            metadata to organize your Featurestore.

            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            on and examples of labels. No more than 64 user
            labels can be associated with one
            Featurestore(System labels are excluded)."
            System reserved label keys are prefixed with
            "aiplatform.googleapis.com/" and are immutable.
        online_serving_config (google.cloud.aiplatform_v1beta1.types.Featurestore.OnlineServingConfig):
            Optional. Config for online storage resources. The field
            should not co-exist with the field of
            ``OnlineStoreReplicationConfig``. If both of it and
            OnlineStoreReplicationConfig are unset, the feature store
            will not have an online store and cannot be used for online
            serving.
        state (google.cloud.aiplatform_v1beta1.types.Featurestore.State):
            Output only. State of the featurestore.
        online_storage_ttl_days (int):
            Optional. TTL in days for feature values that will be stored
            in online serving storage. The Feature Store online storage
            periodically removes obsolete feature values older than
            ``online_storage_ttl_days`` since the feature generation
            time. Note that ``online_storage_ttl_days`` should be less
            than or equal to ``offline_storage_ttl_days`` for each
            EntityType under a featurestore. If not set, default to 4000
            days
        encryption_spec (google.cloud.aiplatform_v1beta1.types.EncryptionSpec):
            Optional. Customer-managed encryption key
            spec for data storage. If set, both of the
            online and offline data storage will be secured
            by this key.
    """

    class State(proto.Enum):
        r"""Possible states a featurestore can have.

        Values:
            STATE_UNSPECIFIED (0):
                Default value. This value is unused.
            STABLE (1):
                State when the featurestore configuration is
                not being updated and the fields reflect the
                current configuration of the featurestore. The
                featurestore is usable in this state.
            UPDATING (2):
                The state of the featurestore configuration when it is being
                updated. During an update, the fields reflect either the
                original configuration or the updated configuration of the
                featurestore. For example,
                ``online_serving_config.fixed_node_count`` can take minutes
                to update. While the update is in progress, the featurestore
                is in the UPDATING state, and the value of
                ``fixed_node_count`` can be the original value or the
                updated value, depending on the progress of the operation.
                Until the update completes, the actual number of nodes can
                still be the original value of ``fixed_node_count``. The
                featurestore is still usable in this state.
        """
        STATE_UNSPECIFIED = 0
        STABLE = 1
        UPDATING = 2

    class OnlineServingConfig(proto.Message):
        r"""OnlineServingConfig specifies the details for provisioning
        online serving resources.

        Attributes:
            fixed_node_count (int):
                The number of nodes for the online store. The
                number of nodes doesn't scale automatically, but
                you can manually update the number of nodes. If
                set to 0, the featurestore will not have an
                online store and cannot be used for online
                serving.
            scaling (google.cloud.aiplatform_v1beta1.types.Featurestore.OnlineServingConfig.Scaling):
                Online serving scaling configuration. Only one of
                ``fixed_node_count`` and ``scaling`` can be set. Setting one
                will reset the other.
        """

        class Scaling(proto.Message):
            r"""Online serving scaling configuration. If min_node_count and
            max_node_count are set to the same value, the cluster will be
            configured with the fixed number of node (no auto-scaling).

            Attributes:
                min_node_count (int):
                    Required. The minimum number of nodes to
                    scale down to. Must be greater than or equal to
                    1.
                max_node_count (int):
                    The maximum number of nodes to scale up to. Must be greater
                    than min_node_count, and less than or equal to 10 times of
                    'min_node_count'.
                cpu_utilization_target (int):
                    Optional. The cpu utilization that the
                    Autoscaler should be trying to achieve. This
                    number is on a scale from 0 (no utilization) to
                    100 (total utilization), and is limited between
                    10 and 80. When a cluster's CPU utilization
                    exceeds the target that you have set, Bigtable
                    immediately adds nodes to the cluster. When CPU
                    utilization is substantially lower than the
                    target, Bigtable removes nodes. If not set or
                    set to 0, default to 50.
            """

            min_node_count: int = proto.Field(
                proto.INT32,
                number=1,
            )
            max_node_count: int = proto.Field(
                proto.INT32,
                number=2,
            )
            cpu_utilization_target: int = proto.Field(
                proto.INT32,
                number=3,
            )

        fixed_node_count: int = proto.Field(
            proto.INT32,
            number=2,
        )
        scaling: "Featurestore.OnlineServingConfig.Scaling" = proto.Field(
            proto.MESSAGE,
            number=4,
            message="Featurestore.OnlineServingConfig.Scaling",
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
    etag: str = proto.Field(
        proto.STRING,
        number=5,
    )
    labels: MutableMapping[str, str] = proto.MapField(
        proto.STRING,
        proto.STRING,
        number=6,
    )
    online_serving_config: OnlineServingConfig = proto.Field(
        proto.MESSAGE,
        number=7,
        message=OnlineServingConfig,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=8,
        enum=State,
    )
    online_storage_ttl_days: int = proto.Field(
        proto.INT32,
        number=13,
    )
    encryption_spec: gca_encryption_spec.EncryptionSpec = proto.Field(
        proto.MESSAGE,
        number=10,
        message=gca_encryption_spec.EncryptionSpec,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
