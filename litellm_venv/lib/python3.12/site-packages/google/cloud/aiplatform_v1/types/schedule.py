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

from google.cloud.aiplatform_v1.types import pipeline_service
from google.protobuf import timestamp_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1",
    manifest={
        "Schedule",
    },
)


class Schedule(proto.Message):
    r"""An instance of a Schedule periodically schedules runs to make
    API calls based on user specified time specification and API
    request type.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        cron (str):
            Cron schedule (https://en.wikipedia.org/wiki/Cron) to launch
            scheduled runs. To explicitly set a timezone to the cron
            tab, apply a prefix in the cron tab:
            "CRON_TZ=${IANA_TIME_ZONE}" or "TZ=${IANA_TIME_ZONE}". The
            ${IANA_TIME_ZONE} may only be a valid string from IANA time
            zone database. For example, "CRON_TZ=America/New_York 1 \*
            \* \* \*", or "TZ=America/New_York 1 \* \* \* \*".

            This field is a member of `oneof`_ ``time_specification``.
        create_pipeline_job_request (google.cloud.aiplatform_v1.types.CreatePipelineJobRequest):
            Request for
            [PipelineService.CreatePipelineJob][google.cloud.aiplatform.v1.PipelineService.CreatePipelineJob].
            CreatePipelineJobRequest.parent field is required (format:
            projects/{project}/locations/{location}).

            This field is a member of `oneof`_ ``request``.
        name (str):
            Immutable. The resource name of the Schedule.
        display_name (str):
            Required. User provided name of the Schedule.
            The name can be up to 128 characters long and
            can consist of any UTF-8 characters.
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            Optional. Timestamp after which the first run
            can be scheduled. Default to Schedule create
            time if not specified.
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            Optional. Timestamp after which no new runs can be
            scheduled. If specified, The schedule will be completed when
            either end_time is reached or when scheduled_run_count >=
            max_run_count. If not specified, new runs will keep getting
            scheduled until this Schedule is paused or deleted. Already
            scheduled runs will be allowed to complete. Unset if not
            specified.
        max_run_count (int):
            Optional. Maximum run count of the schedule. If specified,
            The schedule will be completed when either started_run_count
            >= max_run_count or when end_time is reached. If not
            specified, new runs will keep getting scheduled until this
            Schedule is paused or deleted. Already scheduled runs will
            be allowed to complete. Unset if not specified.
        started_run_count (int):
            Output only. The number of runs started by
            this schedule.
        state (google.cloud.aiplatform_v1.types.Schedule.State):
            Output only. The state of this Schedule.
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Schedule was
            created.
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Schedule was
            updated.
        next_run_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Schedule should schedule
            the next run. Having a next_run_time in the past means the
            runs are being started behind schedule.
        last_pause_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Schedule was
            last paused. Unset if never paused.
        last_resume_time (google.protobuf.timestamp_pb2.Timestamp):
            Output only. Timestamp when this Schedule was
            last resumed. Unset if never resumed from pause.
        max_concurrent_run_count (int):
            Required. Maximum number of runs that can be
            started concurrently for this Schedule. This is
            the limit for starting the scheduled requests
            and not the execution of the operations/jobs
            created by the requests (if applicable).
        allow_queueing (bool):
            Optional. Whether new scheduled runs can be queued when
            max_concurrent_runs limit is reached. If set to true, new
            runs will be queued instead of skipped. Default to false.
        catch_up (bool):
            Output only. Whether to backfill missed runs
            when the schedule is resumed from PAUSED state.
            If set to true, all missed runs will be
            scheduled. New runs will be scheduled after the
            backfill is complete. Default to false.
        last_scheduled_run_response (google.cloud.aiplatform_v1.types.Schedule.RunResponse):
            Output only. Response of the last scheduled
            run. This is the response for starting the
            scheduled requests and not the execution of the
            operations/jobs created by the requests (if
            applicable). Unset if no run has been scheduled
            yet.
    """

    class State(proto.Enum):
        r"""Possible state of the schedule.

        Values:
            STATE_UNSPECIFIED (0):
                Unspecified.
            ACTIVE (1):
                The Schedule is active. Runs are being
                scheduled on the user-specified timespec.
            PAUSED (2):
                The schedule is paused. No new runs will be
                created until the schedule is resumed. Already
                started runs will be allowed to complete.
            COMPLETED (3):
                The Schedule is completed. No new runs will
                be scheduled. Already started runs will be
                allowed to complete. Schedules in completed
                state cannot be paused or resumed.
        """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        PAUSED = 2
        COMPLETED = 3

    class RunResponse(proto.Message):
        r"""Status of a scheduled run.

        Attributes:
            scheduled_run_time (google.protobuf.timestamp_pb2.Timestamp):
                The scheduled run time based on the
                user-specified schedule.
            run_response (str):
                The response of the scheduled run.
        """

        scheduled_run_time: timestamp_pb2.Timestamp = proto.Field(
            proto.MESSAGE,
            number=1,
            message=timestamp_pb2.Timestamp,
        )
        run_response: str = proto.Field(
            proto.STRING,
            number=2,
        )

    cron: str = proto.Field(
        proto.STRING,
        number=10,
        oneof="time_specification",
    )
    create_pipeline_job_request: pipeline_service.CreatePipelineJobRequest = (
        proto.Field(
            proto.MESSAGE,
            number=14,
            oneof="request",
            message=pipeline_service.CreatePipelineJobRequest,
        )
    )
    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    display_name: str = proto.Field(
        proto.STRING,
        number=2,
    )
    start_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=3,
        message=timestamp_pb2.Timestamp,
    )
    end_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=4,
        message=timestamp_pb2.Timestamp,
    )
    max_run_count: int = proto.Field(
        proto.INT64,
        number=16,
    )
    started_run_count: int = proto.Field(
        proto.INT64,
        number=17,
    )
    state: State = proto.Field(
        proto.ENUM,
        number=5,
        enum=State,
    )
    create_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=6,
        message=timestamp_pb2.Timestamp,
    )
    update_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=19,
        message=timestamp_pb2.Timestamp,
    )
    next_run_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=7,
        message=timestamp_pb2.Timestamp,
    )
    last_pause_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=8,
        message=timestamp_pb2.Timestamp,
    )
    last_resume_time: timestamp_pb2.Timestamp = proto.Field(
        proto.MESSAGE,
        number=9,
        message=timestamp_pb2.Timestamp,
    )
    max_concurrent_run_count: int = proto.Field(
        proto.INT64,
        number=11,
    )
    allow_queueing: bool = proto.Field(
        proto.BOOL,
        number=12,
    )
    catch_up: bool = proto.Field(
        proto.BOOL,
        number=13,
    )
    last_scheduled_run_response: RunResponse = proto.Field(
        proto.MESSAGE,
        number=18,
        message=RunResponse,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
