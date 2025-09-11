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
        "FeatureStatsAnomaly",
    },
)


class FeatureStatsAnomaly(proto.Message):
    r"""Stats and Anomaly generated at specific timestamp for specific
    Feature. The start_time and end_time are used to define the time
    range of the dataset that current stats belongs to, e.g. prediction
    traffic is bucketed into prediction datasets by time window. If the
    Dataset is not defined by time window, start_time = end_time.
    Timestamp of the stats and anomalies always refers to end_time. Raw
    stats and anomalies are stored in stats_uri or anomaly_uri in the
    tensorflow defined protos. Field data_stats contains almost
    identical information with the raw stats in Vertex AI defined proto,
    for UI to display.

    Attributes:
        score (float):
            Feature importance score, only populated when cross-feature
            monitoring is enabled. For now only used to represent
            feature attribution score within range [0, 1] for
            [ModelDeploymentMonitoringObjectiveType.FEATURE_ATTRIBUTION_SKEW][google.cloud.aiplatform.v1.ModelDeploymentMonitoringObjectiveType.FEATURE_ATTRIBUTION_SKEW]
            and
            [ModelDeploymentMonitoringObjectiveType.FEATURE_ATTRIBUTION_DRIFT][google.cloud.aiplatform.v1.ModelDeploymentMonitoringObjectiveType.FEATURE_ATTRIBUTION_DRIFT].
        stats_uri (str):
            Path of the stats file for current feature values in Cloud
            Storage bucket. Format:
            gs://<bucket_name>/<object_name>/stats. Example:
            gs://monitoring_bucket/feature_name/stats. Stats are stored
            as binary format with Protobuf message
            `tensorflow.metadata.v0.FeatureNameStatistics <https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/statistics.proto>`__.
        anomaly_uri (str):
            Path of the anomaly file for current feature values in Cloud
            Storage bucket. Format:
            gs://<bucket_name>/<object_name>/anomalies. Example:
            gs://monitoring_bucket/feature_name/anomalies. Stats are
            stored as binary format with Protobuf message Anoamlies are
            stored as binary format with Protobuf message
            [tensorflow.metadata.v0.AnomalyInfo]
            (https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/anomalies.proto).
        distribution_deviation (float):
            Deviation from the current stats to baseline
            stats.
              1. For categorical feature, the distribution
                distance is calculated by      L-inifinity
                norm.
              2. For numerical feature, the distribution
                distance is calculated by
                Jensen–Shannon divergence.
        anomaly_detection_threshold (float):
            This is the threshold used when detecting anomalies. The
            threshold can be changed by user, so this one might be
            different from
            [ThresholdConfig.value][google.cloud.aiplatform.v1.ThresholdConfig.value].
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            The start timestamp of window where stats were generated.
            For objectives where time window doesn't make sense (e.g.
            Featurestore Snapshot Monitoring), start_time is only used
            to indicate the monitoring intervals, so it always equals to
            (end_time - monitoring_interval).
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            The end timestamp of window where stats were generated. For
            objectives where time window doesn't make sense (e.g.
            Featurestore Snapshot Monitoring), end_time indicates the
            timestamp of the data used to generate stats (e.g. timestamp
            we take snapshots for feature values).
    """

    score: float = proto.Field(
        proto.DOUBLE,
        number=1,
    )
    stats_uri: str = proto.Field(
        proto.STRING,
        number=3,
    )
    anomaly_uri: str = proto.Field(
        proto.STRING,
        number=4,
    )
    distribution_deviation: float = proto.Field(
        proto.DOUBLE,
        number=5,
    )
    anomaly_detection_threshold: float = proto.Field(
        proto.DOUBLE,
        number=9,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
