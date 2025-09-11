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
from google.rpc import status_pb2  # type: ignore
from google.type import interval_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "FeatureViewSync",
    },
)


class FeatureViewSync(proto.Message):
    r"""FeatureViewSync is a representation of sync operation which
    copies data from data source to Feature View in Online Store.

    Attributes:
        name (str):
            Identifier. Name of the FeatureViewSync. Format:
            ``projects/{project}/locations/{location}/featureOnlineStores/{feature_online_store}/featureViews/{feature_view}/featureViewSyncs/{feature_view_sync}``
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Time when this FeatureViewSync
            is created. Creation of a FeatureViewSync means
            that the job is pending / waiting for sufficient
            resources but may not have started the actual
            data transfer yet.
        run_time (google.type.interval_pb2.Interval):
            Output only. Time when this FeatureViewSync
            is finished.
        final_status (google.rpc.status_pb2.Status):
            Output only. Final status of the
            FeatureViewSync.
        sync_summary (google.cloud.aiplatform_v1.types.FeatureViewSync.SyncSummary):
            Output only. Summary of the sync job.
    """

    class SyncSummary(proto.Message):
        r"""Summary from the Sync job. For continuous syncs, the summary
        is updated periodically. For batch syncs, it gets updated on
        completion of the sync.

        Attributes:
            row_synced (int):
                Output only. Total number of rows synced.
            total_slot (int):
                Output only. BigQuery slot milliseconds
                consumed for the sync job.
        """

        row_synced: int = proto.Field(
            proto.INT64,
            number=1,
        )
        total_slot: int = proto.Field(
            proto.INT64,
            number=2,
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
    run_time: interval_pb2.Interval = proto.Field(
        proto.MESSAGE,
        number=5,
        message=interval_pb2.Interval,
    )
    final_status: status_pb2.Status = proto.Field(
        proto.MESSAGE,
        number=4,
        message=status_pb2.Status,
    )
    sync_summary: SyncSummary = proto.Field(
        proto.MESSAGE,
        number=6,
        message=SyncSummary,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
