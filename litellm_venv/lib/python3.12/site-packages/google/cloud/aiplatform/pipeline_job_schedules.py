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
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import (
    PipelineJob,
)
from google.cloud.aiplatform.schedules import (
    _Schedule,
)
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.compat.types import (
    schedule as gca_schedule,
)
from google.cloud.aiplatform.constants import (
    schedule as schedule_constants,
)

from google.protobuf import field_mask_pb2 as field_mask

_LOGGER = base.Logger(__name__)

# Pattern for valid names used as a Vertex resource name.
_VALID_NAME_PATTERN = schedule_constants._VALID_NAME_PATTERN

# Pattern for an Artifact Registry URL.
_VALID_AR_URL = schedule_constants._VALID_AR_URL

# Pattern for any JSON or YAML file over HTTPS.
_VALID_HTTPS_URL = schedule_constants._VALID_HTTPS_URL

_SCHEDULE_ERROR_STATES = schedule_constants._SCHEDULE_ERROR_STATES


class PipelineJobSchedule(
    _Schedule,
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
                If not set, the project used for the PipelineJob will be used.
            location (str):
                Optional. Location to create PipelineJobSchedule. If not set,
                location used for the PipelineJob will be used.
        """
        if not display_name:
            display_name = self.__class__._generate_display_name()
        utils.validate_display_name(display_name)

        project = project or pipeline_job.project
        location = location or pipeline_job.location
        super().__init__(credentials=credentials, project=project, location=location)

        self._parent = initializer.global_config.common_location_path(
            project=project, location=location
        )

        create_pipeline_job_request = {
            "parent": self._parent,
            "pipeline_job": {
                "runtime_config": pipeline_job.runtime_config,
                "pipeline_spec": pipeline_job.pipeline_spec,
            },
        }
        if "template_uri" in pipeline_job._gca_resource:
            create_pipeline_job_request["pipeline_job"][
                "template_uri"
            ] = pipeline_job._gca_resource.template_uri
        if "labels" in pipeline_job._gca_resource:
            create_pipeline_job_request["pipeline_job"][
                "labels"
            ] = pipeline_job._gca_resource.labels
        if "encryption_spec" in pipeline_job._gca_resource:
            create_pipeline_job_request["pipeline_job"][
                "encryption_spec"
            ] = pipeline_job._gca_resource.encryption_spec
        if "reserved_ip_ranges" in pipeline_job._gca_resource:
            create_pipeline_job_request["pipeline_job"][
                "reserved_ip_ranges"
            ] = pipeline_job._gca_resource.reserved_ip_ranges
        pipeline_job_schedule_args = {
            "display_name": display_name,
            "create_pipeline_job_request": create_pipeline_job_request,
        }

        self._gca_resource = gca_schedule.Schedule(**pipeline_job_schedule_args)

    def create(
        self,
        cron: str,
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
            cron (str):
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
        network = network or initializer.global_config.network

        self._create(
            cron=cron,
            start_time=start_time,
            end_time=end_time,
            allow_queueing=allow_queueing,
            max_run_count=max_run_count,
            max_concurrent_run_count=max_concurrent_run_count,
            service_account=service_account,
            network=network,
            create_request_timeout=create_request_timeout,
        )

    def _create(
        self,
        cron: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        allow_queueing: bool = False,
        max_run_count: Optional[int] = None,
        max_concurrent_run_count: int = 1,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        create_request_timeout: Optional[float] = None,
    ) -> None:
        """Helper method to create the PipelineJobSchedule.

        Args:
            cron (str):
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
        if cron:
            self._gca_resource.cron = cron
        if start_time:
            self._gca_resource.start_time = start_time
        if end_time:
            self._gca_resource.end_time = end_time
        if allow_queueing:
            self._gca_resource.allow_queueing = allow_queueing
        if max_run_count:
            self._gca_resource.max_run_count = max_run_count
        if max_concurrent_run_count:
            self._gca_resource.max_concurrent_run_count = max_concurrent_run_count

        service_account = service_account or initializer.global_config.service_account
        network = network or initializer.global_config.network

        if service_account:
            self._gca_resource.create_pipeline_job_request.pipeline_job.service_account = (
                service_account
            )

        if network:
            self._gca_resource.create_pipeline_job_request.pipeline_job.network = (
                network
            )

        _LOGGER.log_create_with_lro(self.__class__)

        self._gca_resource = self.api_client.create_schedule(
            parent=self._parent,
            schedule=self._gca_resource,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_complete_with_getter(
            self.__class__, self._gca_resource, "schedule"
        )

        _LOGGER.info("View Schedule:\n%s" % self._dashboard_uri())

    @classmethod
    def list(
        cls,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["PipelineJobSchedule"]:
        """List all instances of this PipelineJobSchedule resource.

        Example Usage:

        aiplatform.PipelineJobSchedule.list(
            filter='display_name="experiment_a27"',
            order_by='create_time desc'
        )

        Args:
            filter (str):
                Optional. An expression for filtering the results of the request.
                For field names both snake_case and camelCase are supported.
            order_by (str):
                Optional. A comma-separated list of fields to order by, sorted in
                ascending order. Use "desc" after a field name for descending.
                Supported fields: `display_name`, `create_time`, `update_time`
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
            List[PipelineJobSchedule] - A list of PipelineJobSchedule resource objects.
        """
        return cls._list_with_local_order(
            filter=filter,
            order_by=order_by,
            project=project,
            location=location,
            credentials=credentials,
        )

    def list_jobs(
        self,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        enable_simple_view: bool = True,
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
                Defaults to True if not provided. This will improve the performance of calling
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
        list_filter = f"schedule_name={self._gca_resource.name}"
        if filter:
            list_filter = list_filter + f" AND {filter}"

        return PipelineJob.list(
            filter=list_filter,
            order_by=order_by,
            enable_simple_view=enable_simple_view,
            project=project,
            location=location,
            credentials=credentials,
        )

    def update(
        self,
        display_name: Optional[str] = None,
        cron: Optional[str] = None,
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
            cron='* * * * *',
        )

        Args:
            display_name (str):
                Optional. The user-defined name of this PipelineJobSchedule.
            cron (str):
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
        pipeline_job_schedule = self._gca_resource
        if pipeline_job_schedule.state in _SCHEDULE_ERROR_STATES:
            raise RuntimeError(
                "Not updating PipelineJobSchedule: PipelineJobSchedule must be active or completed."
            )

        updated_fields = []
        if display_name is not None:
            updated_fields.append("display_name")
            setattr(pipeline_job_schedule, "display_name", display_name)
        if cron is not None:
            updated_fields.append("cron")
            setattr(pipeline_job_schedule, "cron", cron)
        if start_time is not None:
            updated_fields.append("start_time")
            setattr(pipeline_job_schedule, "start_time", start_time)
        if end_time is not None:
            updated_fields.append("end_time")
            setattr(pipeline_job_schedule, "end_time", end_time)
        if allow_queueing is not None:
            updated_fields.append("allow_queueing")
            setattr(pipeline_job_schedule, "allow_queueing", allow_queueing)
        if max_run_count is not None:
            updated_fields.append("max_run_count")
            setattr(pipeline_job_schedule, "max_run_count", max_run_count)
        if max_concurrent_run_count is not None:
            updated_fields.append("max_concurrent_run_count")
            setattr(
                pipeline_job_schedule,
                "max_concurrent_run_count",
                max_concurrent_run_count,
            )

        update_mask = field_mask.FieldMask(paths=updated_fields)
        self.api_client.update_schedule(
            schedule=pipeline_job_schedule,
            update_mask=update_mask,
        )
