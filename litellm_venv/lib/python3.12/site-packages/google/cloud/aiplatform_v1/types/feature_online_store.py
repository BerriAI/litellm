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
        "FeatureOnlineStore",
    },
)


class FeatureOnlineStore(proto.Message):
    r"""Vertex AI Feature Online Store provides a centralized
    repository for serving ML features and embedding indexes at low
    latency. The Feature Online Store is a top-level container.

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        bigtable (google.cloud.aiplatform_v1.types.FeatureOnlineStore.Bigtable):
            Contains settings for the Cloud Bigtable
            instance that will be created to serve
            featureValues for all FeatureViews under this
            FeatureOnlineStore.

            This field is a member of `oneof`_ ``storage_type``.
        optimized (google.cloud.aiplatform_v1.types.FeatureOnlineStore.Optimized):
            Contains settings for the Optimized store that will be
            created to serve featureValues for all FeatureViews under
            this FeatureOnlineStore. When choose Optimized storage type,
            need to set
            [PrivateServiceConnectConfig.enable_private_service_connect][google.cloud.aiplatform.v1.PrivateServiceConnectConfig.enable_private_service_connect]
            to use private endpoint. Otherwise will use public endpoint
            by default.

            This field is a member of `oneof`_ ``storage_type``.
        name (str):
            Identifier. Name of the FeatureOnlineStore. Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/{featureOnlineStore}``
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            FeatureOnlineStore was created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this
            FeatureOnlineStore was last updated.
        etag (str):
            Optional. Used to perform consistent
            read-modify-write updates. If not set, a blind
            "overwrite" update happens.
        labels (MutableMapping[str, str]):
            Optional. The labels with user-defined
            metadata to organize your FeatureOnlineStore.

            Label keys and values can be no longer than 64
            characters (Unicode codepoints), can only
            contain lowercase letters, numeric characters,
            underscores and dashes. International characters
            are allowed.

            See https://goo.gl/xmQnxf for more information
            on and examples of labels. No more than 64 user
            labels can be associated with one
            FeatureOnlineStore(System labels are excluded)."
            System reserved label keys are prefixed with
            "aiplatform.googleapis.com/" and are immutable.
        state (google.cloud.aiplatform_v1.types.FeatureOnlineStore.State):
            Output only. State of the featureOnlineStore.
        dedicated_serving_endpoint (google.cloud.aiplatform_v1.types.FeatureOnlineStore.DedicatedServingEndpoint):
            Optional. The dedicated serving endpoint for
            this FeatureOnlineStore, which is different from
            common Vertex service endpoint.
    """

    class State(proto.Enum):
        r"""Possible states a featureOnlineStore can have.

        Values:
            STATE_UNSPECIFIED (0):
                Default value. This value is unused.
            STABLE (1):
                State when the featureOnlineStore
                configuration is not being updated and the
                fields reflect the current configuration of the
                featureOnlineStore. The featureOnlineStore is
                usable in this state.
            UPDATING (2):
                The state of the featureOnlineStore
                configuration when it is being updated. During
                an update, the fields reflect either the
                original configuration or the updated
                configuration of the featureOnlineStore. The
                featureOnlineStore is still usable in this
                state.
        """
        STATE_UNSPECIFIED = 0
        STABLE = 1
        UPDATING = 2

    class Bigtable(proto.Message):
        r"""

        Attributes:
            auto_scaling (google.cloud.aiplatform_v1.types.FeatureOnlineStore.Bigtable.AutoScaling):
                Required. Autoscaling config applied to
                Bigtable Instance.
        """

        class AutoScaling(proto.Message):
            r"""

            Attributes:
                min_node_count (int):
                    Required. The minimum number of nodes to
                    scale down to. Must be greater than or equal to
                    1.
                max_node_count (int):
                    Required. The maximum number of nodes to scale up to. Must
                    be greater than or equal to min_node_count, and less than or
                    equal to 10 times of 'min_node_count'.
                cpu_utilization_target (int):
                    Optional. A percentage of the cluster's CPU
                    capacity. Can be from 10% to 80%. When a
                    cluster's CPU utilization exceeds the target
                    that you have set, Bigtable immediately adds
                    nodes to the cluster. When CPU utilization is
                    substantially lower than the target, Bigtable
                    removes nodes. If not set will default to 50%.
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

        auto_scaling: "FeatureOnlineStore.Bigtable.AutoScaling" = proto.Field(
            proto.MESSAGE,
            number=1,
            message="FeatureOnlineStore.Bigtable.AutoScaling",
        )

    class Optimized(proto.Message):
        r"""Optimized storage type"""

    class DedicatedServingEndpoint(proto.Message):
        r"""The dedicated serving endpoint for this FeatureOnlineStore.
        Only need to set when you choose Optimized storage type. Public
        endpoint is provisioned by default.

        Attributes:
            public_endpoint_domain_name (str):
                Output only. This field will be populated
                with the domain name to use for this
                FeatureOnlineStore
        """

        public_endpoint_domain_name: str = proto.Field(
            proto.STRING,
            number=2,
        )

    bigtable: Bigtable = proto.Field(
        proto.MESSAGE,
        number=8,
        oneof="storage_type",
        message=Bigtable,
    )
    optimized: Optimized = proto.Field(
        proto.MESSAGE,
        number=12,
        oneof="storage_type",
        message=Optimized,
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
    state: State = proto.Field(
        proto.ENUM,
        number=7,
        enum=State,
    )
    dedicated_serving_endpoint: DedicatedServingEndpoint = proto.Field(
        proto.MESSAGE,
        number=10,
        message=DedicatedServingEndpoint,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
