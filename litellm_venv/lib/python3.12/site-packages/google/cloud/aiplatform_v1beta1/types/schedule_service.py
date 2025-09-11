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

from google.cloud.aiplatform_v1beta1.types import schedule as gca_schedule
from google.protobuf import field_mask_pb2  # type: ignore


__protobuf__ = proto.module(
    package="google.cloud.aiplatform.v1beta1",
    manifest={
        "CreateScheduleRequest",
        "GetScheduleRequest",
        "ListSchedulesRequest",
        "ListSchedulesResponse",
        "DeleteScheduleRequest",
        "PauseScheduleRequest",
        "ResumeScheduleRequest",
        "UpdateScheduleRequest",
    },
)


class CreateScheduleRequest(proto.Message):
    r"""Request message for
    [ScheduleService.CreateSchedule][google.cloud.aiplatform.v1beta1.ScheduleService.CreateSchedule].

    Attributes:
        parent (str):
            Required. The resource name of the Location to create the
            Schedule in. Format:
            ``projects/{project}/locations/{location}``
        schedule (google.cloud.aiplatform_v1beta1.types.Schedule):
            Required. The Schedule to create.
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    schedule: gca_schedule.Schedule = proto.Field(
        proto.MESSAGE,
        number=2,
        message=gca_schedule.Schedule,
    )


class GetScheduleRequest(proto.Message):
    r"""Request message for
    [ScheduleService.GetSchedule][google.cloud.aiplatform.v1beta1.ScheduleService.GetSchedule].

    Attributes:
        name (str):
            Required. The name of the Schedule resource. Format:
            ``projects/{project}/locations/{location}/schedules/{schedule}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ListSchedulesRequest(proto.Message):
    r"""Request message for
    [ScheduleService.ListSchedules][google.cloud.aiplatform.v1beta1.ScheduleService.ListSchedules].

    Attributes:
        parent (str):
            Required. The resource name of the Location to list the
            Schedules from. Format:
            ``projects/{project}/locations/{location}``
        filter (str):
            Lists the Schedules that match the filter expression. The
            following fields are supported:

            -  ``display_name``: Supports ``=``, ``!=`` comparisons, and
               ``:`` wildcard.
            -  ``state``: Supports ``=`` and ``!=`` comparisons.
            -  ``request``: Supports existence of the <request_type>
               check. (e.g. ``create_pipeline_job_request:*`` -->
               Schedule has create_pipeline_job_request).
            -  ``create_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``<=``, and ``>=`` comparisons. Values must be in RFC
               3339 format.
            -  ``start_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``<=``, and ``>=`` comparisons. Values must be in RFC
               3339 format.
            -  ``end_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``<=``, ``>=`` comparisons and ``:*`` existence check.
               Values must be in RFC 3339 format.
            -  ``next_run_time``: Supports ``=``, ``!=``, ``<``, ``>``,
               ``<=``, and ``>=`` comparisons. Values must be in RFC
               3339 format.

            Filter expressions can be combined together using logical
            operators (``NOT``, ``AND`` & ``OR``). The syntax to define
            filter expression is based on https://google.aip.dev/160.

            Examples:

            -  ``state="ACTIVE" AND display_name:"my_schedule_*"``
            -  ``NOT display_name="my_schedule"``
            -  ``create_time>"2021-05-18T00:00:00Z"``
            -  ``end_time>"2021-05-18T00:00:00Z" OR NOT end_time:*``
            -  ``create_pipeline_job_request:*``
        page_size (int):
            The standard list page size.
            Default to 100 if not specified.
        page_token (str):
            The standard list page token. Typically obtained via
            [ListSchedulesResponse.next_page_token][google.cloud.aiplatform.v1beta1.ListSchedulesResponse.next_page_token]
            of the previous
            [ScheduleService.ListSchedules][google.cloud.aiplatform.v1beta1.ScheduleService.ListSchedules]
            call.
        order_by (str):
            A comma-separated list of fields to order by. The default
            sort order is in ascending order. Use "desc" after a field
            name for descending. You can have multiple order_by fields
            provided.

            For example, using "create_time desc, end_time" will order
            results by create time in descending order, and if there are
            multiple schedules having the same create time, order them
            by the end time in ascending order.

            If order_by is not specified, it will order by default with
            create_time in descending order.

            Supported fields:

            -  ``create_time``
            -  ``start_time``
            -  ``end_time``
            -  ``next_run_time``
    """

    parent: str = proto.Field(
        proto.STRING,
        number=1,
    )
    filter: str = proto.Field(
        proto.STRING,
        number=2,
    )
    page_size: int = proto.Field(
        proto.INT32,
        number=3,
    )
    page_token: str = proto.Field(
        proto.STRING,
        number=4,
    )
    order_by: str = proto.Field(
        proto.STRING,
        number=5,
    )


class ListSchedulesResponse(proto.Message):
    r"""Response message for
    [ScheduleService.ListSchedules][google.cloud.aiplatform.v1beta1.ScheduleService.ListSchedules]

    Attributes:
        schedules (MutableSequence[google.cloud.aiplatform_v1beta1.types.Schedule]):
            List of Schedules in the requested page.
        next_page_token (str):
            A token to retrieve the next page of results. Pass to
            [ListSchedulesRequest.page_token][google.cloud.aiplatform.v1beta1.ListSchedulesRequest.page_token]
            to obtain that page.
    """

    @property
    def raw_page(self):
        return self

    schedules: MutableSequence[gca_schedule.Schedule] = proto.RepeatedField(
        proto.MESSAGE,
        number=1,
        message=gca_schedule.Schedule,
    )
    next_page_token: str = proto.Field(
        proto.STRING,
        number=2,
    )


class DeleteScheduleRequest(proto.Message):
    r"""Request message for
    [ScheduleService.DeleteSchedule][google.cloud.aiplatform.v1beta1.ScheduleService.DeleteSchedule].

    Attributes:
        name (str):
            Required. The name of the Schedule resource to be deleted.
            Format:
            ``projects/{project}/locations/{location}/schedules/{schedule}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class PauseScheduleRequest(proto.Message):
    r"""Request message for
    [ScheduleService.PauseSchedule][google.cloud.aiplatform.v1beta1.ScheduleService.PauseSchedule].

    Attributes:
        name (str):
            Required. The name of the Schedule resource to be paused.
            Format:
            ``projects/{project}/locations/{location}/schedules/{schedule}``
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )


class ResumeScheduleRequest(proto.Message):
    r"""Request message for
    [ScheduleService.ResumeSchedule][google.cloud.aiplatform.v1beta1.ScheduleService.ResumeSchedule].

    Attributes:
        name (str):
            Required. The name of the Schedule resource to be resumed.
            Format:
            ``projects/{project}/locations/{location}/schedules/{schedule}``
        catch_up (bool):
            Optional. Whether to backfill missed runs when the schedule
            is resumed from PAUSED state. If set to true, all missed
            runs will be scheduled. New runs will be scheduled after the
            backfill is complete. This will also update
            [Schedule.catch_up][google.cloud.aiplatform.v1beta1.Schedule.catch_up]
            field. Default to false.
    """

    name: str = proto.Field(
        proto.STRING,
        number=1,
    )
    catch_up: bool = proto.Field(
        proto.BOOL,
        number=2,
    )


class UpdateScheduleRequest(proto.Message):
    r"""Request message for
    [ScheduleService.UpdateSchedule][google.cloud.aiplatform.v1beta1.ScheduleService.UpdateSchedule].

    Attributes:
        schedule (google.cloud.aiplatform_v1beta1.types.Schedule):
            Required. The Schedule which replaces the resource on the
            server. The following restrictions will be applied:

            -  The scheduled request type cannot be changed.
            -  The non-empty fields cannot be unset.
            -  The output_only fields will be ignored if specified.
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            Required. The update mask applies to the resource. See
            [google.protobuf.FieldMask][google.protobuf.FieldMask].
    """

    schedule: gca_schedule.Schedule = proto.Field(
        proto.MESSAGE,
        number=1,
        message=gca_schedule.Schedule,
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE,
        number=2,
        message=field_mask_pb2.FieldMask,
    )


__all__ = tuple(sorted(__protobuf__.manifest))
