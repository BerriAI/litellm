# -*- coding: utf-8 -*-

# Copyright 2023 Google LLC
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

from typing import List, Optional

from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import (
    PipelineJob,
)
from google.cloud.aiplatform.pipeline_job_schedules import (
    PipelineJobSchedule as PipelineJobScheduleGa,
)
from google.cloud.aiplatform.preview.schedule.schedules import (
    _Schedule as _SchedulePreview,
)


class PipelineJobSchedule(
    PipelineJobScheduleGa,
    _SchedulePreview,
):
    def __init__(
        self,
        pipeline_job: PipelineJob,
        display_name: str,
        credentials: Optional[auth_credentials.Credentials] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        """Retrieves a PipelineJobSchedule resource and instantiates its
        representation.

        Args:
            pipeline_job (PipelineJob):
                Required. PipelineJob used to init the schedule.
            display_name (str):
                Required. The user-defined name of this PipelineJobSchedule.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create this PipelineJobSchedule.
                Overrides credentials set in aiplatform.init.
            project (str):
                Optional. The project that you want to run this PipelineJobSchedule in.
                If not set, the project set in aiplatform.init will be used.
            location (str):
                Optional. Location to create PipelineJobSchedule. If not set,
                location set in aiplatform.init will be used.
        """
        super().__init__(
            pipeline_job=pipeline_job,
            display_name=display_name,
            credentials=credentials,
            project=project,
            location=location,
        )

    def create(
        self,
        cron_expression: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        allow_queueing: bool = False,
        max_run_count: Optional[int] = None,
        max_concurrent_run_count: int = 1,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        create_request_timeout: Optional[float] = None,
    ) -> None:
        """Create a PipelineJobSchedule.

        Args:
            cron_expression (str):
                Required. Time specification (cron schedule expression) to launch scheduled runs.
                To explicitly set a timezone to the cron tab, apply a prefix: "CRON_TZ=${IANA_TIME_ZONE}" or "TZ=${IANA_TIME_ZONE}".
                The ${IANA_TIME_ZONE} may only be a valid string from IANA time zone database.
                For example, "CRON_TZ=America/New_York 1 * * * *", or "TZ=America/New_York 1 * * * *".
            start_time (str):
                Optional. Timestamp after which the first run can be scheduled.
                If unspecified, it defaults to the schedule creation timestamp.
            end_time (str):
                Optional. Timestamp after which no more runs will be scheduled.
                If unspecified, then runs will be scheduled indefinitely.
            allow_queueing (bool):
                Optional. Whether new scheduled runs can be queued when max_concurrent_runs limit is reached.
            max_run_count (int):
                Optional. Maximum run count of the schedule.
                If specified, The schedule will be completed when either started_run_count >= max_run_count or when end_time is reached.
                Must be positive and <= 2^63-1.
            max_concurrent_run_count (int):
                Optional. Maximum number of runs that can be started concurrently for this PipelineJobSchedule.
            service_account (str):
                Optional. Specifies the service account for workload run-as account.
                Users submitting jobs must have act-as permission on this run-as account.
            network (str):
                Optional. The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.
                Private services access must already be configured for the network.
                If left unspecified, the network set in aiplatform.init will be used.
                Otherwise, the job is not peered with any network.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
        """
        super().create(
            cron=cron_expression,
            start_time=start_time,
            end_time=end_time,
            allow_queueing=allow_queueing,
            max_run_count=max_run_count,
            max_concurrent_run_count=max_concurrent_run_count,
            service_account=service_account,
            network=network,
            create_request_timeout=create_request_timeout,
        )

    def list_jobs(
        self,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        enable_simple_view: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List[PipelineJob]:
        """List all PipelineJob 's created by this PipelineJobSchedule.

        Example usage:

        pipeline_job_schedule.list_jobs(order_by='create_time_desc')

        Args:
            filter (str):
                Optional. An expression for filtering the results of the request.
                For field names both snake_case and camelCase are supported.
            order_by (str):
                Optional. A comma-separated list of fields to order by, sorted in
                ascending order. Use "desc" after a field name for descending.
                Supported fields: `display_name`, `create_time`, `update_time`
            enable_simple_view (bool):
                Optional. Whether to pass the `read_mask` parameter to the list call.
                Defaults to False if not provided. This will improve the performance of calling
                list(). However, the returned PipelineJob list will not include all fields for
                each PipelineJob. Setting this to True will exclude the following fields in your
                response: `runtime_config`, `service_account`, `network`, and some subfields of
                `pipeline_spec` and `job_detail`. The following fields will be included in
                each PipelineJob resource in your response: `state`, `display_name`,
                `pipeline_spec.pipeline_info`, `create_time`, `start_time`, `end_time`,
                `update_time`, `labels`, `template_uri`, `template_metadata.version`,
                `job_detail.pipeline_run_context`, `job_detail.pipeline_context`.
            project (str):
                Optional. Project to retrieve list from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve list from. If not set, location
                set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to retrieve list. Overrides
                credentials set in aiplatform.init.

        Returns:
            List[PipelineJob] - A list of PipelineJob resource objects.
        """
        return super().list_jobs(
            filter=filter,
            order_by=order_by,
            enable_simple_view=enable_simple_view,
            project=project,
            location=location,
            credentials=credentials,
        )

    def update(
        self,
        display_name: Optional[str] = None,
        cron_expression: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        allow_queueing: Optional[bool] = None,
        max_run_count: Optional[int] = None,
        max_concurrent_run_count: Optional[int] = None,
    ) -> None:
        """Update an existing PipelineJobSchedule.

        Example usage:

        pipeline_job_schedule.update(
            display_name='updated-display-name',
            cron_expression='* * * * *',
        )

        Args:
            display_name (str):
                Optional. The user-defined name of this PipelineJobSchedule.
            cron_expression (str):
                Optional. Time specification (cron schedule expression) to launch scheduled runs.
                To explicitly set a timezone to the cron tab, apply a prefix: "CRON_TZ=${IANA_TIME_ZONE}" or "TZ=${IANA_TIME_ZONE}".
                The ${IANA_TIME_ZONE} may only be a valid string from IANA time zone database.
                For example, "CRON_TZ=America/New_York 1 * * * *", or "TZ=America/New_York 1 * * * *".
            start_time (str):
                Optional. Timestamp after which the first run can be scheduled.
                If unspecified, it defaults to the schedule creation timestamp.
            end_time (str):
                Optional. Timestamp after which no more runs will be scheduled.
                If unspecified, then runs will be scheduled indefinitely.
            allow_queueing (bool):
                Optional. Whether new scheduled runs can be queued when max_concurrent_runs limit is reached.
            max_run_count (int):
                Optional. Maximum run count of the schedule.
                If specified, The schedule will be completed when either started_run_count >= max_run_count or when end_time is reached.
                Must be positive and <= 2^63-1.
            max_concurrent_run_count (int):
                Optional. Maximum number of runs that can be started concurrently for this PipelineJobSchedule.

        Raises:
            RuntimeError: User tried to call update() before create().
        """
        super().update(
            display_name=display_name,
            cron=cron_expression,
            start_time=start_time,
            end_time=end_time,
            allow_queueing=allow_queueing,
            max_run_count=max_run_count,
            max_concurrent_run_count=max_concurrent_run_count,
        )
