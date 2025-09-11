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

from google.cloud.aiplatform.pipeline_jobs import (
    PipelineJob as PipelineJobGa,
)
from google.cloud.aiplatform_v1.services.pipeline_service import (
    PipelineServiceClient as PipelineServiceClientGa,
)
from google.cloud import aiplatform_v1beta1
from google.cloud.aiplatform import compat, pipeline_job_schedules
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils

from google.cloud.aiplatform.metadata import constants as metadata_constants
from google.cloud.aiplatform.metadata import experiment_resources


class _PipelineJob(
    PipelineJobGa,
    experiment_loggable_schemas=(
        experiment_resources._ExperimentLoggableSchema(
            title=metadata_constants.SYSTEM_PIPELINE_RUN
        ),
    ),
):
    """Preview PipelineJob resource for Vertex AI."""

    def create_schedule(
        self,
        cron_expression: str,
        display_name: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        allow_queueing: bool = False,
        max_run_count: Optional[int] = None,
        max_concurrent_run_count: int = 1,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        create_request_timeout: Optional[float] = None,
    ) -> "pipeline_job_schedules.PipelineJobSchedule":  # noqa: F821
        """Creates a PipelineJobSchedule directly from a PipelineJob.

        Example Usage:

        pipeline_job = aiplatform.PipelineJob(
            display_name='job_display_name',
            template_path='your_pipeline.yaml',
        )
        pipeline_job.run()
        pipeline_job_schedule = pipeline_job.create_schedule(
            cron_expression='* * * * *',
            display_name='schedule_display_name',
        )

        Args:
            cron_expression (str):
                Required. Time specification (cron schedule expression) to launch scheduled runs.
                To explicitly set a timezone to the cron tab, apply a prefix: "CRON_TZ=${IANA_TIME_ZONE}" or "TZ=${IANA_TIME_ZONE}".
                The ${IANA_TIME_ZONE} may only be a valid string from IANA time zone database.
                For example, "CRON_TZ=America/New_York 1 * * * *", or "TZ=America/New_York 1 * * * *".
            display_name (str):
                Required. The user-defined name of this PipelineJobSchedule.
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

        Returns:
            A Vertex AI PipelineJobSchedule.
        """
        return super().create_schedule(
            cron=cron_expression,
            display_name=display_name,
            start_time=start_time,
            end_time=end_time,
            allow_queueing=allow_queueing,
            max_run_count=max_run_count,
            max_concurrent_run_count=max_concurrent_run_count,
            service_account=service_account,
            network=network,
            create_request_timeout=create_request_timeout,
        )

    @classmethod
    def batch_delete(
        cls,
        names: List[str],
        project: Optional[str] = None,
        location: Optional[str] = None,
    ) -> aiplatform_v1beta1.BatchDeletePipelineJobsResponse:
        """
        Example Usage:
          aiplatform.init(
            project='your_project_name',
            location='your_location',
          )
          aiplatform.PipelineJob.batch_delete(
            names=['pipeline_job_name', 'pipeline_job_name2']
          )

        Args:
            names (List[str]):
                Required. The fully-qualified resource name or ID of the
                Pipeline Jobs to batch delete. Example:
                "projects/123/locations/us-central1/pipelineJobs/456"
                or "456" when project and location are initialized or passed.
            project (str):
                Optional. Project containing the Pipeline Jobs to
                batch delete. If not set, the project given to `aiplatform.init`
                will be used.
            location (str):
                Optional. Location containing the Pipeline Jobs to
                batch delete. If not set, the location given to `aiplatform.init`
                will be used.

        Returns:
          BatchDeletePipelineJobsResponse contains PipelineJobs deleted.
        """
        user_project = project or initializer.global_config.project
        user_location = location or initializer.global_config.location
        parent = initializer.global_config.common_location_path(
            project=user_project, location=user_location
        )
        pipeline_jobs_names = [
            utils.full_resource_name(
                resource_name=name,
                resource_noun="pipelineJobs",
                parse_resource_name_method=PipelineServiceClientGa.parse_pipeline_job_path,
                format_resource_name_method=PipelineServiceClientGa.pipeline_job_path,
                project=user_project,
                location=user_location,
            )
            for name in names
        ]
        request = aiplatform_v1beta1.BatchDeletePipelineJobsRequest(
            parent=parent, names=pipeline_jobs_names
        )
        client = cls._instantiate_client(
            location=user_location,
            appended_user_agent=["preview-pipeline-jobs-batch-delete"],
        )
        v1beta1_client = client.select_version(compat.V1BETA1)
        operation = v1beta1_client.batch_delete_pipeline_jobs(request)
        return operation.result()
