# -*- coding: utf-8 -*-

# Copyright 2021 Google LLC
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

import datetime
import logging
import re
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Union
from google.api_core import operation

from google.auth import credentials as auth_credentials
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.constants import pipeline as pipeline_constants
from google.cloud.aiplatform.metadata import artifact
from google.cloud.aiplatform.metadata import constants as metadata_constants
from google.cloud.aiplatform.metadata import context
from google.cloud.aiplatform.metadata import execution
from google.cloud.aiplatform.metadata import experiment_resources
from google.cloud.aiplatform.metadata import utils as metadata_utils
from google.cloud.aiplatform.utils import gcs_utils
from google.cloud.aiplatform.utils import pipeline_utils
from google.cloud.aiplatform.utils import yaml_utils
from google.protobuf import field_mask_pb2 as field_mask
from google.protobuf import json_format

from google.cloud.aiplatform.compat.types import (
    pipeline_job as gca_pipeline_job,
    pipeline_state as gca_pipeline_state,
)
from google.cloud.aiplatform_v1.types import (
    pipeline_service as PipelineServiceV1,
)
from google.cloud.aiplatform_v1.services.pipeline_service import (
    PipelineServiceClient as PipelineServiceClientGa,
)

_LOGGER = base.Logger(__name__)

_PIPELINE_COMPLETE_STATES = pipeline_constants._PIPELINE_COMPLETE_STATES

_PIPELINE_ERROR_STATES = pipeline_constants._PIPELINE_ERROR_STATES

# Pattern for valid names used as a Vertex resource name.
_VALID_NAME_PATTERN = pipeline_constants._VALID_NAME_PATTERN

# Pattern for an Artifact Registry URL.
_VALID_AR_URL = pipeline_constants._VALID_AR_URL

# Pattern for any JSON or YAML file over HTTPS.
_VALID_HTTPS_URL = pipeline_constants._VALID_HTTPS_URL

_READ_MASK_FIELDS = pipeline_constants._READ_MASK_FIELDS

# _block_until_complete wait times
_JOB_WAIT_TIME = 5  # start at five seconds
_LOG_WAIT_TIME = 5
_MAX_WAIT_TIME = 60 * 5  # 5 minute wait
_WAIT_TIME_MULTIPLIER = 2  # scale wait by 2 every iteration


def _get_current_time() -> datetime.datetime:
    """Gets the current timestamp."""
    return datetime.datetime.now()


def _set_enable_caching_value(
    pipeline_spec: Dict[str, Any], enable_caching: bool
) -> None:
    """Sets pipeline tasks caching options.

    Args:
     pipeline_spec (Dict[str, Any]):
          Required. The dictionary of pipeline spec.
     enable_caching (bool):
          Required. Whether to enable caching.
    """
    for component in [pipeline_spec["root"]] + list(
        pipeline_spec["components"].values()
    ):
        if "dag" in component:
            for task in component["dag"]["tasks"].values():
                task["cachingOptions"] = {"enableCache": enable_caching}


class PipelineJob(
    base.VertexAiStatefulResource,
    experiment_resources._ExperimentLoggable,
    experiment_loggable_schemas=(
        experiment_resources._ExperimentLoggableSchema(
            title=metadata_constants.SYSTEM_PIPELINE_RUN
        ),
    ),
):
    client_class = utils.PipelineJobClientWithOverride
    _resource_noun = "pipelineJobs"
    _delete_method = "delete_pipeline_job"
    _getter_method = "get_pipeline_job"
    _list_method = "list_pipeline_jobs"
    _parse_resource_name_method = "parse_pipeline_job_path"
    _format_resource_name_method = "pipeline_job_path"

    # Required by the done() method
    _valid_done_states = _PIPELINE_COMPLETE_STATES

    def __init__(
        self,
        # TODO(b/223262536): Make the display_name parameter optional in the next major release
        display_name: str,
        template_path: str,
        job_id: Optional[str] = None,
        pipeline_root: Optional[str] = None,
        parameter_values: Optional[Dict[str, Any]] = None,
        input_artifacts: Optional[Dict[str, str]] = None,
        enable_caching: Optional[bool] = None,
        encryption_spec_key_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        failure_policy: Optional[str] = None,
    ):
        """Retrieves a PipelineJob resource and instantiates its
        representation.

        Args:
            display_name (str):
                Required. The user-defined name of this Pipeline.
            template_path (str):
                Required. The path of PipelineJob or PipelineSpec JSON or YAML file. It
                can be a local path, a Google Cloud Storage URI (e.g. "gs://project.name"),
                an Artifact Registry URI (e.g.
                "https://us-central1-kfp.pkg.dev/proj/repo/pack/latest"), or an HTTPS URI.
            job_id (str):
                Optional. The unique ID of the job run.
                If not specified, pipeline name + timestamp will be used.
            pipeline_root (str):
                Optional. The root of the pipeline outputs. If not set, the staging bucket
                set in aiplatform.init will be used. If that's not set a pipeline-specific
                artifacts bucket will be used.
            parameter_values (Dict[str, Any]):
                Optional. The mapping from runtime parameter names to its values that
                control the pipeline run.
            input_artifacts (Dict[str, str]):
                Optional. The mapping from the runtime parameter name for this artifact to its resource id.
                For example: "vertex_model":"456". Note: full resource name ("projects/123/locations/us-central1/metadataStores/default/artifacts/456") cannot be used.
            enable_caching (bool):
                Optional. Whether to turn on caching for the run.

                If this is not set, defaults to the compile time settings, which
                are True for all tasks by default, while users may specify
                different caching options for individual tasks.

                If this is set, the setting applies to all tasks in the pipeline.

                Overrides the compile time settings.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the job. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If this is set, then all
                resources created by the PipelineJob will
                be encrypted with the provided encryption key.

                Overrides encryption_spec_key_name set in aiplatform.init.
            labels (Dict[str, str]):
                Optional. The user defined metadata to organize PipelineJob.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create this PipelineJob.
                Overrides credentials set in aiplatform.init.
            project (str):
                Optional. The project that you want to run this PipelineJob in. If not set,
                the project set in aiplatform.init will be used.
            location (str):
                Optional. Location to create PipelineJob. If not set,
                location set in aiplatform.init will be used.
            failure_policy (str):
                Optional. The failure policy - "slow" or "fast".
                Currently, the default of a pipeline is that the pipeline will continue to
                run until no more tasks can be executed, also known as
                PIPELINE_FAILURE_POLICY_FAIL_SLOW (corresponds to "slow").
                However, if a pipeline is set to
                PIPELINE_FAILURE_POLICY_FAIL_FAST (corresponds to "fast"),
                it will stop scheduling any new tasks when a task has failed. Any
                scheduled tasks will continue to completion.

        Raises:
            ValueError: If job_id or labels have incorrect format.
        """
        if not display_name:
            display_name = self.__class__._generate_display_name()
        utils.validate_display_name(display_name)

        if labels:
            utils.validate_labels(labels)

        super().__init__(project=project, location=location, credentials=credentials)

        self._parent = initializer.global_config.common_location_path(
            project=project, location=location
        )

        # this loads both .yaml and .json files because YAML is a superset of JSON
        pipeline_json = yaml_utils.load_yaml(
            template_path, self.project, self.credentials
        )

        # Pipeline_json can be either PipelineJob or PipelineSpec.
        if pipeline_json.get("pipelineSpec") is not None:
            pipeline_job = pipeline_json
            pipeline_root = (
                pipeline_root
                or pipeline_job["pipelineSpec"].get("defaultPipelineRoot")
                or pipeline_job["runtimeConfig"].get("gcsOutputDirectory")
                or initializer.global_config.staging_bucket
            )
        else:
            pipeline_job = {
                "pipelineSpec": pipeline_json,
                "runtimeConfig": {},
            }
            pipeline_root = (
                pipeline_root
                or pipeline_job["pipelineSpec"].get("defaultPipelineRoot")
                or initializer.global_config.staging_bucket
            )
        pipeline_root = (
            pipeline_root
            or gcs_utils.generate_gcs_directory_for_pipeline_artifacts(
                project=project,
                location=location,
            )
        )
        builder = pipeline_utils.PipelineRuntimeConfigBuilder.from_job_spec_json(
            pipeline_job
        )
        builder.update_pipeline_root(pipeline_root)
        builder.update_runtime_parameters(parameter_values)
        builder.update_input_artifacts(input_artifacts)

        builder.update_failure_policy(failure_policy)
        runtime_config_dict = builder.build()

        runtime_config = gca_pipeline_job.PipelineJob.RuntimeConfig()._pb
        json_format.ParseDict(runtime_config_dict, runtime_config)

        pipeline_name = pipeline_job["pipelineSpec"]["pipelineInfo"]["name"]
        self.job_id = job_id or "{pipeline_name}-{timestamp}".format(
            pipeline_name=re.sub("[^-0-9a-z]+", "-", pipeline_name.lower())
            .lstrip("-")
            .rstrip("-"),
            timestamp=_get_current_time().strftime("%Y%m%d%H%M%S"),
        )
        if not _VALID_NAME_PATTERN.match(self.job_id):
            raise ValueError(
                f"Generated job ID: {self.job_id} is illegal as a Vertex pipelines job ID. "
                "Expecting an ID following the regex pattern "
                f'"{_VALID_NAME_PATTERN.pattern[1:-1]}"'
            )

        if enable_caching is not None:
            _set_enable_caching_value(pipeline_job["pipelineSpec"], enable_caching)

        pipeline_job_args = {
            "display_name": display_name,
            "pipeline_spec": pipeline_job["pipelineSpec"],
            "labels": labels,
            "runtime_config": runtime_config,
            "encryption_spec": initializer.global_config.get_encryption_spec(
                encryption_spec_key_name=encryption_spec_key_name
            ),
        }

        if _VALID_AR_URL.match(template_path) or _VALID_HTTPS_URL.match(template_path):
            pipeline_job_args["template_uri"] = template_path

        self._gca_resource = gca_pipeline_job.PipelineJob(**pipeline_job_args)

    def run(
        self,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        reserved_ip_ranges: Optional[List[str]] = None,
        sync: Optional[bool] = True,
        create_request_timeout: Optional[float] = None,
    ) -> None:
        """Run this configured PipelineJob and monitor the job until completion.

        Args:
            service_account (str):
                Optional. Specifies the service account for workload run-as account.
                Users submitting jobs must have act-as permission on this run-as account.
            network (str):
                Optional. The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.
                Private services access must already be configured for the network.
                If left unspecified, the network set in aiplatform.init will be used.
                Otherwise, the job is not peered with any network.
            reserved_ip_ranges (List[str]):
                Optional. A list of names for the reserved IP ranges under the VPC network that can be used for this PipelineJob's workload.
                For example: ['vertex-ai-ip-range'].
            sync (bool):
                Optional. Whether to execute this method synchronously. If False, this method will unblock and it will be executed in a concurrent Future.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
        """
        network = network or initializer.global_config.network

        self._run(
            service_account=service_account,
            network=network,
            reserved_ip_ranges=reserved_ip_ranges,
            sync=sync,
            create_request_timeout=create_request_timeout,
        )

    @base.optional_sync()
    def _run(
        self,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        reserved_ip_ranges: Optional[List[str]] = None,
        sync: Optional[bool] = True,
        create_request_timeout: Optional[float] = None,
    ) -> None:
        """Helper method to ensure network synchronization and to run
        the configured PipelineJob and monitor the job until completion.

        Args:
            service_account (str):
                Optional. Specifies the service account for workload run-as account.
                Users submitting jobs must have act-as permission on this run-as account.
            network (str):
                Optional. The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.
                Private services access must already be configured for the network.
            reserved_ip_ranges (List[str]):
                Optional. A list of names for the reserved IP ranges under the VPC network that can be used for this PipelineJob's workload.
                For example: ['vertex-ai-ip-range'].
            sync (bool):
                Optional. Whether to execute this method synchronously. If False, this method will unblock and it will be executed in a concurrent Future.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
        """
        self.submit(
            service_account=service_account,
            network=network,
            reserved_ip_ranges=reserved_ip_ranges,
            create_request_timeout=create_request_timeout,
        )

        self._block_until_complete()

    def submit(
        self,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        reserved_ip_ranges: Optional[List[str]] = None,
        create_request_timeout: Optional[float] = None,
        *,
        experiment: Optional[Union[str, experiment_resources.Experiment]] = None,
    ) -> None:
        """Run this configured PipelineJob.

        Args:
            service_account (str):
                Optional. Specifies the service account for workload run-as account.
                Users submitting jobs must have act-as permission on this run-as account.
            network (str):
                Optional. The full name of the Compute Engine network to which the job
                should be peered. For example, projects/12345/global/networks/myVPC.

                Private services access must already be configured for the network.
                If left unspecified, the network set in aiplatform.init will be used.
                Otherwise, the job is not peered with any network.
            reserved_ip_ranges (List[str]):
                Optional. A list of names for the reserved IP ranges under the VPC
                network that can be used for this PipelineJob's workload. For example: ['vertex-ai-ip-range'].

                If left unspecified, the job will be deployed to any IP ranges under
                the provided VPC network.
            create_request_timeout (float):
                Optional. The timeout for the create request in seconds.
            experiment (Union[str, experiments_resource.Experiment]):
                Optional. The Vertex AI experiment name or instance to associate to this PipelineJob.

                Metrics produced by the PipelineJob as system.Metric Artifacts
                will be associated as metrics to the current Experiment Run.

                Pipeline parameters will be associated as parameters to the
                current Experiment Run.
        """
        network = network or initializer.global_config.network
        service_account = service_account or initializer.global_config.service_account

        if service_account:
            self._gca_resource.service_account = service_account

        if network:
            self._gca_resource.network = network

        if reserved_ip_ranges:
            self._gca_resource.reserved_ip_ranges = reserved_ip_ranges

        try:
            output_artifacts_gcs_dir = (
                self._gca_resource.runtime_config.gcs_output_directory
            )
            assert output_artifacts_gcs_dir
            gcs_utils.create_gcs_bucket_for_pipeline_artifacts_if_it_does_not_exist(
                output_artifacts_gcs_dir=output_artifacts_gcs_dir,
                service_account=self._gca_resource.service_account,
                project=self.project,
                location=self.location,
                credentials=self.credentials,
            )
        except:  # noqa: E722
            _LOGGER.exception(
                "Error when trying to get or create a GCS bucket for the pipeline output artifacts"
            )

        # Prevents logs from being supressed on TFX pipelines
        if self._gca_resource.pipeline_spec.get("sdkVersion", "").startswith("tfx"):
            _LOGGER.setLevel(logging.INFO)

        if experiment:
            self._validate_experiment(experiment)

        _LOGGER.log_create_with_lro(self.__class__)

        self._gca_resource = self.api_client.create_pipeline_job(
            parent=self._parent,
            pipeline_job=self._gca_resource,
            pipeline_job_id=self.job_id,
            timeout=create_request_timeout,
        )

        _LOGGER.log_create_complete_with_getter(
            self.__class__, self._gca_resource, "pipeline_job"
        )

        _LOGGER.info("View Pipeline Job:\n%s" % self._dashboard_uri())

        if experiment:
            self._associate_to_experiment(experiment)

    def create_schedule(
        self,
        cron: str,
        display_name: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        allow_queueing: bool = False,
        max_run_count: Optional[int] = None,
        max_concurrent_run_count: int = 1,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        create_request_timeout: Optional[float] = None,
    ) -> "aiplatform.PipelineJobSchedule":
        """Creates a PipelineJobSchedule directly from a PipelineJob.

        Example Usage:

        pipeline_job = aiplatform.PipelineJob(
            display_name='job_display_name',
            template_path='your_pipeline.yaml',
        )
        pipeline_job.run()
        pipeline_job_schedule = pipeline_job.create_schedule(
            cron='* * * * *',
            display_name='schedule_display_name',
        )

        Args:
            cron (str):
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
        if not display_name:
            display_name = self._generate_display_name(prefix="PipelineJobSchedule")
        utils.validate_display_name(display_name)

        pipeline_job_schedule = aiplatform.PipelineJobSchedule(
            pipeline_job=self,
            display_name=display_name,
        )

        pipeline_job_schedule.create(
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
        return pipeline_job_schedule

    def wait(self):
        """Wait for this PipelineJob to complete."""
        if self._latest_future is None:
            self._block_until_complete()
        else:
            super().wait()

    def batch_delete(
        self,
        project: str,
        location: str,
        names: List[str],
    ) -> PipelineServiceV1.BatchDeletePipelineJobsResponse:
        """
        Example Usage:
          pipeline_job = aiplatform.PipelineJob(
            display_name='job_display_name',
            template_path='your_pipeline.yaml',
         )
          pipeline_job.batch_delete(
            project='your_project_id',
            location='your_location',
            names=['pipeline_job_name',
          'pipeline_job_name2']
          )

        Args:
          project: Required. The project id of the PipelineJobs to batch delete.
          location: Required. The location of the PipelineJobs to batch delete.
          names: Required. The names of the PipelineJobs to delete. A
               maximum of 32 PipelineJobs can be deleted in a batch.

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
        request = aiplatform_v1.BatchDeletePipelineJobsRequest(
            parent=parent, names=pipeline_jobs_names
        )
        operation = self.api_client.batch_delete_pipeline_jobs(request)
        return operation.result()

    def batch_cancel(
        self,
        project: str,
        location: str,
        names: List[str],
    ) -> operation.Operation:
        """
        Example Usage:
          pipeline_job = aiplatform.PipelineJob(
            display_name='job_display_name',
            template_path='your_pipeline.yaml',
         )
          pipeline_job.batch_cancel(
            project='your_project_id',
            location='your_location',
            names=['pipeline_job_name',
          'pipeline_job_name2']
          )

        Args:
          project: Required. The project id of the PipelineJobs to batch delete.
          location: Required. The location of the PipelineJobs to batch delete.
          names: Required. The names of the PipelineJobs to cancel. A
               maximum of 32 PipelineJobs can be cancelled in a batch.

        Returns:
          operation (Operation):
                An object representing a long-running operation.
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
        request = aiplatform_v1.BatchCancelPipelineJobsRequest(
            parent=parent, names=pipeline_jobs_names
        )
        return self.api_client.batch_cancel_pipeline_jobs(request)

    @property
    def pipeline_spec(self):
        return self._gca_resource.pipeline_spec

    @property
    def runtime_config(self) -> gca_pipeline_job.PipelineJob.RuntimeConfig:
        return self._gca_resource.runtime_config

    @property
    def state(self) -> Optional[gca_pipeline_state.PipelineState]:
        """Current pipeline state."""
        self._sync_gca_resource()
        return self._gca_resource.state

    @property
    def task_details(self) -> List[gca_pipeline_job.PipelineTaskDetail]:
        self._sync_gca_resource()
        return list(self._gca_resource.job_detail.task_details)

    @property
    def has_failed(self) -> bool:
        """Returns True if pipeline has failed.

        False otherwise.
        """
        return self.state == gca_pipeline_state.PipelineState.PIPELINE_STATE_FAILED

    def _dashboard_uri(self) -> str:
        """Helper method to compose the dashboard uri where pipeline can be
        viewed."""
        fields = self._parse_resource_name(self.resource_name)
        url = f"https://console.cloud.google.com/vertex-ai/locations/{fields['location']}/pipelines/runs/{fields['pipeline_job']}?project={fields['project']}"
        return url

    def _block_until_complete(self):
        """Helper method to block and check on job until complete."""
        # Used these numbers so failures surface fast
        wait = _JOB_WAIT_TIME  # start at five seconds
        log_wait = _LOG_WAIT_TIME
        max_wait = _MAX_WAIT_TIME  # 5 minute wait
        multiplier = _WAIT_TIME_MULTIPLIER  # scale wait by 2 every iteration

        previous_time = time.time()
        while self.state not in _PIPELINE_COMPLETE_STATES:
            current_time = time.time()
            if current_time - previous_time >= log_wait:
                _LOGGER.info(
                    "%s %s current state:\n%s"
                    % (
                        self.__class__.__name__,
                        self._gca_resource.name,
                        self._gca_resource.state,
                    )
                )
                log_wait = min(log_wait * multiplier, max_wait)
                previous_time = current_time
            time.sleep(wait)

        # Error is only populated when the job state is
        # JOB_STATE_FAILED or JOB_STATE_CANCELLED.
        if self._gca_resource.state in _PIPELINE_ERROR_STATES:
            raise RuntimeError("Job failed with:\n%s" % self._gca_resource.error)
        else:
            _LOGGER.log_action_completed_against_resource("run", "completed", self)

    @classmethod
    def get(
        cls,
        resource_name: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "PipelineJob":
        """Get a Vertex AI Pipeline Job for the given resource_name.

        Args:
            resource_name (str):
                Required. A fully-qualified resource name or ID.
            project (str):
                Optional. Project to retrieve dataset from. If not set, project
                set in aiplatform.init will be used.
            location (str):
                Optional. Location to retrieve dataset from. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to upload this model.
                Overrides credentials set in aiplatform.init.

        Returns:
            A Vertex AI PipelineJob.
        """
        self = cls._empty_constructor(
            project=project,
            location=location,
            credentials=credentials,
            resource_name=resource_name,
        )

        self._gca_resource = self._get_gca_resource(resource_name=resource_name)

        return self

    def cancel(self) -> None:
        """Starts asynchronous cancellation on the PipelineJob. The server
        makes a best effort to cancel the job, but success is not guaranteed.
        On successful cancellation, the PipelineJob is not deleted; instead it
        becomes a job with state set to `CANCELLED`.
        """
        self.api_client.cancel_pipeline_job(name=self.resource_name)

    @classmethod
    def list(
        cls,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        enable_simple_view: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["PipelineJob"]:
        """List all instances of this PipelineJob resource.

        Example Usage:

        aiplatform.PipelineJob.list(
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
            List[PipelineJob] - A list of PipelineJob resource objects
        """

        read_mask_fields = None

        if enable_simple_view:
            read_mask_fields = field_mask.FieldMask(paths=_READ_MASK_FIELDS)
            _LOGGER.warn(
                "By enabling simple view, the PipelineJob resources returned from this method will not contain all fields."
            )

        return cls._list_with_local_order(
            filter=filter,
            order_by=order_by,
            read_mask=read_mask_fields,
            project=project,
            location=location,
            credentials=credentials,
        )

    def wait_for_resource_creation(self) -> None:
        """Waits until resource has been created."""
        self._wait_for_resource_creation()

    def done(self) -> bool:
        """Helper method that return True is PipelineJob is done. False otherwise."""
        if not self._gca_resource:
            return False

        return self.state in _PIPELINE_COMPLETE_STATES

    def _get_context(self) -> context.Context:
        """Returns the PipelineRun Context for this PipelineJob in the MetadataStore.

        Returns:
            System.PipelineRUn Context instance that represents this PipelineJob.

        Raises:
            RuntimeError if Pipeline has failed or system.PipelineRun context is not found.
        """
        self.wait_for_resource_creation()
        pipeline_run_context = self._gca_resource.job_detail.pipeline_run_context

        # PipelineJob context is created asynchronously so we need to poll until it exists.
        while not self.done():
            pipeline_run_context = self._gca_resource.job_detail.pipeline_run_context
            if pipeline_run_context:
                break
            time.sleep(1)

        if not pipeline_run_context:
            if self.has_failed:
                raise RuntimeError(
                    f"Cannot associate PipelineJob to Experiment: {self.gca_resource.error}"
                )
            else:
                raise RuntimeError(
                    "Cannot associate PipelineJob to Experiment because PipelineJob context could not be found."
                )

        return context.Context(
            resource=pipeline_run_context,
            project=self.project,
            location=self.location,
            credentials=self.credentials,
        )

    @classmethod
    def _query_experiment_row(
        cls, node: context.Context
    ) -> experiment_resources._ExperimentRow:
        """Queries the PipelineJob metadata as an experiment run parameter and metric row.

        Parameters are retrieved from the system.Run Execution.metadata of the PipelineJob.

        Metrics are retrieved from the system.Metric Artifacts.metadata produced by this PipelineJob.

        Args:
            node (context._Context):
                Required. System.PipelineRun context that represents a PipelineJob Run.
        Returns:
            Experiment run row representing this PipelineJob.
        """

        system_run_executions = execution.Execution.list(
            project=node.project,
            location=node.location,
            credentials=node.credentials,
            filter=metadata_utils._make_filter_string(
                in_context=[node.resource_name],
                schema_title=metadata_constants.SYSTEM_RUN,
            ),
        )

        metric_artifacts = artifact.Artifact.list(
            project=node.project,
            location=node.location,
            credentials=node.credentials,
            filter=metadata_utils._make_filter_string(
                in_context=[node.resource_name],
                schema_title=[
                    metadata_constants.SYSTEM_METRICS,
                    metadata_constants.GOOGLE_CLASSIFICATION_METRICS,
                    metadata_constants.GOOGLE_REGRESSION_METRICS,
                    metadata_constants.GOOGLE_FORECASTING_METRICS,
                ],
            ),
        )
        row = experiment_resources._ExperimentRow(
            experiment_run_type=node.schema_title, name=node.display_name
        )

        if system_run_executions:
            row.params = {}
            for key, value in system_run_executions[0].metadata.items():
                if key.startswith(metadata_constants.PIPELINE_PARAM_PREFIX):
                    row.params[
                        key[len(metadata_constants.PIPELINE_PARAM_PREFIX) :]
                    ] = value
            row.state = system_run_executions[0].state.name

        for metric_artifact in metric_artifacts:
            if row.metrics:
                row.metrics.update(metric_artifact.metadata)
            else:
                row.metrics = metric_artifact.metadata

        return row

    def clone(
        self,
        display_name: Optional[str] = None,
        job_id: Optional[str] = None,
        pipeline_root: Optional[str] = None,
        parameter_values: Optional[Dict[str, Any]] = None,
        input_artifacts: Optional[Dict[str, str]] = None,
        enable_caching: Optional[bool] = None,
        encryption_spec_key_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ) -> "PipelineJob":
        """Returns a new PipelineJob object with the same settings as the original one.

        Args:
            display_name (str):
                Optional. The user-defined name of this cloned Pipeline.
                If not specified, original pipeline display name will be used.
            job_id (str):
                Optional. The unique ID of the job run.
                If not specified, "cloned" + pipeline name + timestamp will be used.
            pipeline_root (str):
                Optional. The root of the pipeline outputs. Default to be the same
                staging bucket as original pipeline.
            parameter_values (Dict[str, Any]):
                Optional. The mapping from runtime parameter names to its values that
                control the pipeline run. Defaults to be the same values as original
                PipelineJob.
            input_artifacts (Dict[str, str]):
                Optional. The mapping from the runtime parameter name for this artifact to its resource id. Defaults to be the same values as original
                PipelineJob. For example: "vertex_model":"456". Note: full resource name ("projects/123/locations/us-central1/metadataStores/default/artifacts/456") cannot be used.
            enable_caching (bool):
                Optional. Whether to turn on caching for the run.
                If this is not set, defaults to be the same as original pipeline.
                If this is set, the setting applies to all tasks in the pipeline.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the job. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute resource is created.
                If this is set, then all
                resources created by the PipelineJob will
                be encrypted with the provided encryption key.
                If not specified, encryption_spec of original PipelineJob will be used.
            labels (Dict[str, str]):
                Optional. The user defined metadata to organize PipelineJob.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create this PipelineJob.
                Overrides credentials set in aiplatform.init.
            project (str):
                Optional. The project that you want to run this PipelineJob in.
                If not set, the project set in original PipelineJob will be used.
            location (str):
                Optional. Location to create PipelineJob.
                If not set, location set in original PipelineJob will be used.

        Returns:
            A Vertex AI PipelineJob.

        Raises:
            ValueError: If job_id or labels have incorrect format.
        """
        ## Initialize an empty PipelineJob
        if not project:
            project = self.project
        if not location:
            location = self.location
        if not credentials:
            credentials = self.credentials

        cloned = self.__class__._empty_constructor(
            project=project,
            location=location,
            credentials=credentials,
        )
        cloned._parent = initializer.global_config.common_location_path(
            project=project, location=location
        )

        ## Get gca_resource from original PipelineJob
        pipeline_job = json_format.MessageToDict(self._gca_resource._pb)

        ## Set pipeline_spec
        pipeline_spec = pipeline_job["pipelineSpec"]
        if "deploymentConfig" in pipeline_spec:
            del pipeline_spec["deploymentConfig"]

        ## Set caching
        if enable_caching is not None:
            _set_enable_caching_value(pipeline_spec, enable_caching)

        ## Set job_id
        pipeline_name = pipeline_spec["pipelineInfo"]["name"]
        cloned.job_id = job_id or "cloned-{pipeline_name}-{timestamp}".format(
            pipeline_name=re.sub("[^-0-9a-z]+", "-", pipeline_name.lower())
            .lstrip("-")
            .rstrip("-"),
            timestamp=_get_current_time().strftime("%Y%m%d%H%M%S"),
        )
        if not _VALID_NAME_PATTERN.match(cloned.job_id):
            raise ValueError(
                f"Generated job ID: {cloned.job_id} is illegal as a Vertex pipelines job ID. "
                "Expecting an ID following the regex pattern "
                f'"{_VALID_NAME_PATTERN.pattern[1:-1]}"'
            )

        ## Set display_name, labels and encryption_spec
        if display_name:
            utils.validate_display_name(display_name)
        elif not display_name and "displayName" in pipeline_job:
            display_name = pipeline_job["displayName"]

        if labels:
            utils.validate_labels(labels)
        elif not labels and "labels" in pipeline_job:
            labels = pipeline_job["labels"]

        if encryption_spec_key_name or "encryptionSpec" not in pipeline_job:
            encryption_spec = initializer.global_config.get_encryption_spec(
                encryption_spec_key_name=encryption_spec_key_name
            )
        else:
            encryption_spec = pipeline_job["encryptionSpec"]

        ## Set runtime_config
        builder = pipeline_utils.PipelineRuntimeConfigBuilder.from_job_spec_json(
            pipeline_job
        )
        builder.update_pipeline_root(pipeline_root)
        builder.update_runtime_parameters(parameter_values)
        builder.update_input_artifacts(input_artifacts)
        runtime_config_dict = builder.build()
        runtime_config = gca_pipeline_job.PipelineJob.RuntimeConfig()._pb
        json_format.ParseDict(runtime_config_dict, runtime_config)

        ## Create gca_resource for cloned PipelineJob
        cloned._gca_resource = gca_pipeline_job.PipelineJob(
            display_name=display_name,
            pipeline_spec=pipeline_spec,
            labels=labels,
            runtime_config=runtime_config,
            encryption_spec=encryption_spec,
        )

        return cloned

    @staticmethod
    def from_pipeline_func(
        # Parameters for the PipelineJob constructor
        pipeline_func: Callable,
        parameter_values: Optional[Dict[str, Any]] = None,
        input_artifacts: Optional[Dict[str, str]] = None,
        output_artifacts_gcs_dir: Optional[str] = None,
        enable_caching: Optional[bool] = None,
        context_name: Optional[str] = "pipeline",
        display_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        job_id: Optional[str] = None,
        # Parameters for the Vertex SDK
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
        encryption_spec_key_name: Optional[str] = None,
    ) -> "PipelineJob":
        """Creates PipelineJob by compiling a pipeline function.

        Args:
            pipeline_func (Callable):
                Required. A pipeline function to compile.
                A pipeline function creates instances of components and connects
                component inputs to outputs.
            parameter_values (Dict[str, Any]):
                Optional. The mapping from runtime parameter names to its values that
                control the pipeline run.
            input_artifacts (Dict[str, str]):
                Optional. The mapping from the runtime parameter name for this artifact to its resource id. For example: "vertex_model":"456". Note: full resource name ("projects/123/locations/us-central1/metadataStores/default/artifacts/456") cannot be used.
            output_artifacts_gcs_dir (str):
                Optional. The GCS location of the pipeline outputs.
                A GCS bucket for artifacts will be created if not specified.
            enable_caching (bool):
                Optional. Whether to turn on caching for the run.

                If this is not set, defaults to the compile time settings, which
                are True for all tasks by default, while users may specify
                different caching options for individual tasks.

                If this is set, the setting applies to all tasks in the pipeline.

                Overrides the compile time settings.
            context_name (str):
                Optional. The name of metadata context. Used for cached execution reuse.
            display_name (str):
                Optional. The user-defined name of this Pipeline.
            labels (Dict[str, str]):
                Optional. The user defined metadata to organize PipelineJob.
            job_id (str):
                Optional. The unique ID of the job run.
                If not specified, pipeline name + timestamp will be used.

            project (str):
                Optional. The project that you want to run this PipelineJob in. If not set,
                the project set in aiplatform.init will be used.
            location (str):
                Optional. Location to create PipelineJob. If not set,
                location set in aiplatform.init will be used.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to use to create this PipelineJob.
                Overrides credentials set in aiplatform.init.
            encryption_spec_key_name (str):
                Optional. The Cloud KMS resource identifier of the customer
                managed encryption key used to protect the job. Has the
                form:
                ``projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key``.
                The key needs to be in the same region as where the compute
                resource is created.

                If this is set, then all
                resources created by the PipelineJob will
                be encrypted with the provided encryption key.

                Overrides encryption_spec_key_name set in aiplatform.init.

        Returns:
            A Vertex AI PipelineJob.

        Raises:
            ValueError: If job_id or labels have incorrect format.
        """

        # Importing the KFP module here to prevent import errors when the kfp package is not installed.
        try:
            from kfp.v2 import compiler as compiler_v2
        except ImportError as err:
            raise RuntimeError(
                "Cannot import the kfp.v2.compiler module. Please install or update the kfp package."
            ) from err

        automatic_display_name = " ".join(
            [
                pipeline_func.__name__.replace("_", " "),
                datetime.datetime.now().isoformat(sep=" "),
            ]
        )
        display_name = display_name or automatic_display_name
        job_id = job_id or re.sub(
            r"[^-a-z0-9]", "-", automatic_display_name.lower()
        ).strip("-")
        pipeline_file = tempfile.mktemp(suffix=".json")
        compiler_v2.Compiler().compile(
            pipeline_func=pipeline_func,
            pipeline_name=context_name,
            package_path=pipeline_file,
        )
        pipeline_job = PipelineJob(
            template_path=pipeline_file,
            parameter_values=parameter_values,
            input_artifacts=input_artifacts,
            pipeline_root=output_artifacts_gcs_dir,
            enable_caching=enable_caching,
            display_name=display_name,
            job_id=job_id,
            labels=labels,
            project=project,
            location=location,
            credentials=credentials,
            encryption_spec_key_name=encryption_spec_key_name,
        )
        return pipeline_job

    def get_associated_experiment(self) -> Optional["aiplatform.Experiment"]:
        """Gets the aiplatform.Experiment associated with this PipelineJob,
        or None if this PipelineJob is not associated with an experiment.

        Returns:
            An aiplatform.Experiment resource or None if this PipelineJob is
            not associated with an experiment..

        """

        pipeline_parent_contexts = (
            self._gca_resource.job_detail.pipeline_run_context.parent_contexts
        )

        pipeline_experiment_resources = [
            context.Context(resource_name=c)._gca_resource
            for c in pipeline_parent_contexts
            if c != self._gca_resource.job_detail.pipeline_context.name
        ]

        pipeline_experiment_resource_names = []

        for c in pipeline_experiment_resources:
            if c.schema_title == metadata_constants.SYSTEM_EXPERIMENT:
                pipeline_experiment_resource_names.append(c.name)

        if len(pipeline_experiment_resource_names) > 1:
            _LOGGER.warning(
                f"There is more than one Experiment is associated with this pipeline."
                f"The following experiments were found: {pipeline_experiment_resource_names.join(', ')}\n"
                f"Returning only the following experiment: {pipeline_experiment_resource_names[0]}"
            )

        if len(pipeline_experiment_resource_names) >= 1:
            return experiment_resources.Experiment(
                pipeline_experiment_resource_names[0],
                project=self.project,
                location=self.location,
                credentials=self.credentials,
            )
