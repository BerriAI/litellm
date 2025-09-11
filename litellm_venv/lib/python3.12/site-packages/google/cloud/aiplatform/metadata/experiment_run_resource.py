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
"""Vertex Experiment Run class."""

from collections import abc
import concurrent.futures
import functools
from typing import Any, Callable, Dict, List, Optional, Set, Union

from google.api_core import exceptions
from google.auth import credentials as auth_credentials
from google.cloud.aiplatform import base
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import pipeline_jobs
from google.cloud.aiplatform import jobs
from google.cloud.aiplatform.compat.types import artifact as gca_artifact
from google.cloud.aiplatform.compat.types import execution as gca_execution
from google.cloud.aiplatform.compat.types import (
    tensorboard_time_series as gca_tensorboard_time_series,
)
from google.cloud.aiplatform.metadata import artifact
from google.cloud.aiplatform.metadata import constants
from google.cloud.aiplatform.metadata import context
from google.cloud.aiplatform.metadata import execution
from google.cloud.aiplatform.metadata import experiment_resources
from google.cloud.aiplatform.metadata import metadata
from google.cloud.aiplatform.metadata import _models
from google.cloud.aiplatform.metadata import resource
from google.cloud.aiplatform.metadata import utils as metadata_utils
from google.cloud.aiplatform.metadata.schema import utils as schema_utils
from google.cloud.aiplatform.metadata.schema.google import (
    artifact_schema as google_artifact_schema,
)
from google.cloud.aiplatform.tensorboard import tensorboard_resource
from google.cloud.aiplatform.utils import rest_utils

from google.protobuf import timestamp_pb2


_LOGGER = base.Logger(__name__)


def _format_experiment_run_resource_id(experiment_name: str, run_name: str) -> str:
    """Formats the the experiment run resource id.

    It is a concatenation of experiment name and run name.

    Args:
        experiment_name (str): Name of the experiment which is it's resource id.
        run_name (str): Name of the run.
    Returns:
        The resource id to be used with this run.
    """
    return f"{experiment_name}-{run_name}"


def _v1_not_supported(method: Callable) -> Callable:
    """Helpers wrapper for backward compatibility. Raises when using an API not support for legacy runs."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if isinstance(self._metadata_node, execution.Execution):
            raise NotImplementedError(
                f"{self._run_name} is an Execution run created during Vertex Experiment Preview and does not support"
                f" {method.__name__}. Please create a new Experiment run to use this method."
            )
        else:
            return method(self, *args, **kwargs)

    return wrapper


class ExperimentRun(
    experiment_resources._ExperimentLoggable,
    experiment_loggable_schemas=(
        experiment_resources._ExperimentLoggableSchema(
            title=constants.SYSTEM_EXPERIMENT_RUN, type=context.Context
        ),
        # backwards compatibility with Preview Experiment runs
        experiment_resources._ExperimentLoggableSchema(
            title=constants.SYSTEM_RUN, type=execution.Execution
        ),
    ),
):
    """A Vertex AI Experiment run."""

    def __init__(
        self,
        run_name: str,
        experiment: Union[experiment_resources.Experiment, str],
        *,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """

        ```py
        my_run = aiplatform.ExperimentRun('my-run', experiment='my-experiment')
        ```

        Args:
            run_name (str):
                Required. The name of this run.
            experiment (Union[experiment_resources.Experiment, str]):
                Required. The name or instance of this experiment.
            project (str):
                Optional. Project where this experiment run is located. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location where this experiment run is located. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to retrieve this experiment run. Overrides
                credentials set in aiplatform.init.
        """

        self._experiment = self._get_experiment(
            experiment=experiment,
            project=project,
            location=location,
            credentials=credentials,
        )
        self._run_name = run_name

        run_id = _format_experiment_run_resource_id(
            experiment_name=self._experiment.name, run_name=run_name
        )

        metadata_args = dict(
            project=project,
            location=location,
            credentials=credentials,
        )

        def _get_context() -> context.Context:
            with experiment_resources._SetLoggerLevel(resource):
                run_context = context.Context(
                    **{**metadata_args, "resource_name": run_id}
                )
                if run_context.schema_title != constants.SYSTEM_EXPERIMENT_RUN:
                    raise ValueError(
                        f"Run {run_name} must be of type {constants.SYSTEM_EXPERIMENT_RUN}"
                        f" but is of type {run_context.schema_title}"
                    )
                return run_context

        try:
            self._metadata_node = _get_context()
        except exceptions.NotFound as context_not_found:
            try:
                # backward compatibility
                self._v1_resolve_experiment_run(
                    {
                        **metadata_args,
                        "execution_name": run_id,
                    }
                )
            except exceptions.NotFound:
                raise context_not_found
        else:
            self._backing_tensorboard_run = self._lookup_tensorboard_run_artifact()

            # initially set to None. Will initially update from resource then track locally.
            self._largest_step: Optional[int] = None

    def _v1_resolve_experiment_run(self, metadata_args: Dict[str, Any]):
        """Resolves preview Experiment.

        Args:
            metadata_args (Dict[str, Any): Arguments to pass to Execution constructor.
        """

        def _get_execution():
            with experiment_resources._SetLoggerLevel(resource):
                run_execution = execution.Execution(**metadata_args)
                if run_execution.schema_title != constants.SYSTEM_RUN:
                    # note this will raise the context not found exception in the constructor
                    raise exceptions.NotFound("Experiment run not found.")
                return run_execution

        self._metadata_node = _get_execution()
        self._metadata_metric_artifact = self._v1_get_metric_artifact()

    def _v1_get_metric_artifact(self) -> artifact.Artifact:
        """Resolves metric artifact for backward compatibility.

        Returns:
            Instance of Artifact that represents this run's metric artifact.
        """
        metadata_args = dict(
            artifact_name=self._v1_format_artifact_name(self._metadata_node.name),
            project=self.project,
            location=self.location,
            credentials=self.credentials,
        )

        with experiment_resources._SetLoggerLevel(resource):
            metric_artifact = artifact.Artifact(**metadata_args)

        if metric_artifact.schema_title != constants.SYSTEM_METRICS:
            # note this will raise the context not found exception in the constructor
            raise exceptions.NotFound("Experiment run not found.")

        return metric_artifact

    @staticmethod
    def _v1_format_artifact_name(run_id: str) -> str:
        """Formats resource id of legacy metric artifact for this run."""
        return f"{run_id}-metrics"

    def _get_context(self) -> context.Context:
        """Returns this metadata context that represents this run.

        Returns:
            Context instance of this run.
        """
        return self._metadata_node

    @property
    def resource_id(self) -> str:
        """The resource ID of this experiment run's Metadata context.

        The resource ID is the final part of the resource name:
        ``projects/{project}/locations/{location}/metadataStores/{metadatastore}/contexts/{resource ID}``
        """
        return self._metadata_node.name

    @property
    def name(self) -> str:
        """This run's name used to identify this run within it's Experiment."""
        return self._run_name

    @property
    def resource_name(self) -> str:
        """This run's Metadata context resource name.

        In the format: ``projects/{project}/locations/{location}/metadataStores/{metadatastore}/contexts/{context}``
        """
        return self._metadata_node.resource_name

    @property
    def project(self) -> str:
        """The project that this experiment run is located in."""
        return self._metadata_node.project

    @property
    def location(self) -> str:
        """The location that this experiment is located in."""
        return self._metadata_node.location

    @property
    def credentials(self) -> auth_credentials.Credentials:
        """The credentials used to access this experiment run."""
        return self._metadata_node.credentials

    @property
    def state(self) -> gca_execution.Execution.State:
        """The state of this run."""
        if self._is_legacy_experiment_run():
            return self._metadata_node.state
        else:
            return getattr(
                gca_execution.Execution.State,
                self._metadata_node.metadata[constants._STATE_KEY],
            )

    @staticmethod
    def _get_experiment(
        experiment: Optional[Union[experiment_resources.Experiment, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> experiment_resources.Experiment:
        """Helper method ot get the experiment by name(str) or instance.

        Args:
            experiment(str):
                Optional. The name of this experiment. Defaults to experiment set in aiplatform.init if not provided.
            project (str):
                Optional. Project where this experiment is located. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location where this experiment is located. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to retrieve this experiment. Overrides
                credentials set in aiplatform.init.
        Raises:
            ValueError if experiment is None and experiment has not been set using aiplatform.init.
        """

        experiment = experiment or initializer.global_config.experiment

        if not experiment:
            raise ValueError(
                "experiment must be provided or experiment should be set using aiplatform.init"
            )

        if not isinstance(experiment, experiment_resources.Experiment):
            experiment = experiment_resources.Experiment(
                experiment_name=experiment,
                project=project,
                location=location,
                credentials=credentials,
            )
        return experiment

    def _is_backing_tensorboard_run_artifact(self, artifact: artifact.Artifact) -> bool:
        """Helper method to confirm tensorboard run metadata artifact is this run's tensorboard artifact.

        Args:
            artifact (artifact.Artifact): Required. Instance of metadata Artifact.
        Returns:
            bool whether the provided artifact is this run's TensorboardRun's artifact.
        """
        return all(
            [
                artifact.metadata.get(constants._VERTEX_EXPERIMENT_TRACKING_LABEL),
                artifact.name == self._tensorboard_run_id(self._metadata_node.name),
                artifact.schema_title
                == constants._TENSORBOARD_RUN_REFERENCE_ARTIFACT.schema_title,
            ]
        )

    def _is_legacy_experiment_run(self) -> bool:
        """Helper method that return True if this is a legacy experiment run."""
        return isinstance(self._metadata_node, execution.Execution)

    def update_state(self, state: gca_execution.Execution.State):
        """Update the state of this experiment run.

        ```py
        my_run = aiplatform.ExperimentRun('my-run', experiment='my-experiment')
        my_run.update_state(state=aiplatform.gapic.Execution.State.COMPLETE)
        ```

        Args:
            state (aiplatform.gapic.Execution.State): State of this run.
        """
        if self._is_legacy_experiment_run():
            self._metadata_node.update(state=state)
        else:
            self._metadata_node.update(metadata={constants._STATE_KEY: state.name})

    def _lookup_tensorboard_run_artifact(
        self,
    ) -> Optional[experiment_resources._VertexResourceWithMetadata]:
        """Helpers method to resolve this run's TensorboardRun Artifact if it exists.

        Returns:
            Tuple of Tensorboard Run Artifact and TensorboardRun is it exists.
        """
        with experiment_resources._SetLoggerLevel(resource):
            try:
                tensorboard_run_artifact = artifact.Artifact(
                    artifact_name=self._tensorboard_run_id(self._metadata_node.name),
                    project=self._metadata_node.project,
                    location=self._metadata_node.location,
                    credentials=self._metadata_node.credentials,
                )
            except exceptions.NotFound:
                tensorboard_run_artifact = None

        if tensorboard_run_artifact and self._is_backing_tensorboard_run_artifact(
            tensorboard_run_artifact
        ):
            return experiment_resources._VertexResourceWithMetadata(
                resource=tensorboard_resource.TensorboardRun(
                    tensorboard_run_artifact.metadata[
                        constants.GCP_ARTIFACT_RESOURCE_NAME_KEY
                    ]
                ),
                metadata=tensorboard_run_artifact,
            )

    @classmethod
    def get(
        cls,
        run_name: str,
        *,
        experiment: Optional[Union[experiment_resources.Experiment, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> Optional["ExperimentRun"]:
        """Gets experiment run if one exists with this run_name.

        Args:
            run_name (str):
                Required. The name of this run.
            experiment (Union[experiment_resources.Experiment, str]):
                Optional. The name or instance of this experiment.
                If not set, use the default experiment in `aiplatform.init`
            project (str):
                Optional. Project where this experiment run is located.
                Overrides project set in aiplatform.init.
            location (str):
                Optional. Location where this experiment run is located.
                Overrides location set in aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to retrieve this experiment run.
                Overrides credentials set in aiplatform.init.

        Returns:
            Vertex AI experimentRun or None if no resource was found.
        """
        experiment = experiment or metadata._experiment_tracker.experiment

        if not experiment:
            raise ValueError(
                "experiment must be provided or "
                "experiment should be set using aiplatform.init"
            )

        try:
            return cls(
                run_name=run_name,
                experiment=experiment,
                project=project,
                location=location,
                credentials=credentials,
            )
        except exceptions.NotFound:
            return None

    def _initialize_experiment_run(
        self,
        node: Union[context.Context, execution.Execution],
        experiment: Optional[experiment_resources.Experiment] = None,
    ):
        self._experiment = experiment
        self._run_name = node.display_name
        self._metadata_node = node
        self._largest_step = None

        if self._is_legacy_experiment_run():
            self._metadata_metric_artifact = self._v1_get_metric_artifact()
            self._backing_tensorboard_run = None
        else:
            self._metadata_metric_artifact = None
            self._backing_tensorboard_run = self._lookup_tensorboard_run_artifact()

    @classmethod
    def list(
        cls,
        *,
        experiment: Optional[Union[experiment_resources.Experiment, str]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["ExperimentRun"]:
        """List the experiment runs for a given aiplatform.Experiment.

        ```py
        my_runs = aiplatform.ExperimentRun.list(experiment='my-experiment')
        ```

        Args:
            experiment (Union[aiplatform.Experiment, str]):
                Optional. The experiment name or instance to list the experiment run from. If not provided,
                will use the experiment set in aiplatform.init.
            project (str):
                Optional. Project where this experiment is located. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location where this experiment is located. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to retrieve this experiment. Overrides
                credentials set in aiplatform.init.
        Returns:
            List of experiment runs.
        """

        experiment = cls._get_experiment(
            experiment=experiment,
            project=project,
            location=location,
            credentials=credentials,
        )

        metadata_args = dict(
            project=experiment._metadata_context.project,
            location=experiment._metadata_context.location,
            credentials=experiment._metadata_context.credentials,
        )

        filter_str = metadata_utils._make_filter_string(
            schema_title=constants.SYSTEM_EXPERIMENT_RUN,
            parent_contexts=[experiment.resource_name],
        )

        run_contexts = context.Context.list(filter=filter_str, **metadata_args)

        filter_str = metadata_utils._make_filter_string(
            schema_title=constants.SYSTEM_RUN, in_context=[experiment.resource_name]
        )

        run_executions = execution.Execution.list(filter=filter_str, **metadata_args)

        def _create_experiment_run(context: context.Context) -> ExperimentRun:
            this_experiment_run = cls.__new__(cls)
            this_experiment_run._initialize_experiment_run(context, experiment)

            return this_experiment_run

        def _create_v1_experiment_run(
            execution: execution.Execution,
        ) -> ExperimentRun:
            this_experiment_run = cls.__new__(cls)
            this_experiment_run._initialize_experiment_run(execution, experiment)

            return this_experiment_run

        if run_contexts or run_executions:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max([len(run_contexts), len(run_executions)])
            ) as executor:
                submissions = [
                    executor.submit(_create_experiment_run, context)
                    for context in run_contexts
                ]
                experiment_runs = [submission.result() for submission in submissions]

                submissions = [
                    executor.submit(_create_v1_experiment_run, execution)
                    for execution in run_executions
                ]

                for submission in submissions:
                    experiment_runs.append(submission.result())

            return experiment_runs
        else:
            return []

    @classmethod
    def _query_experiment_row(
        cls, node: Union[context.Context, execution.Execution]
    ) -> experiment_resources._ExperimentRow:
        """Retrieves the runs metric and parameters into an experiment run row.

        Args:
            node (Union[context._Context, execution.Execution]):
                Required. Metadata node instance that represents this run.
        Returns:
            Experiment run row that represents this run.
        """
        this_experiment_run = cls.__new__(cls)
        this_experiment_run._initialize_experiment_run(node)

        row = experiment_resources._ExperimentRow(
            experiment_run_type=node.schema_title,
            name=node.display_name,
        )

        row.params = this_experiment_run.get_params()
        row.metrics = this_experiment_run.get_metrics()
        row.state = this_experiment_run.get_state()
        row.time_series_metrics = (
            this_experiment_run._get_latest_time_series_metric_columns()
        )

        return row

    def _get_logged_pipeline_runs(self) -> List[context.Context]:
        """Returns Pipeline Run contexts logged to this Experiment Run.

        Returns:
            List of Pipeline system.PipelineRun contexts.
        """

        service_request_args = dict(
            project=self._metadata_node.project,
            location=self._metadata_node.location,
            credentials=self._metadata_node.credentials,
        )

        filter_str = metadata_utils._make_filter_string(
            schema_title=constants.SYSTEM_PIPELINE_RUN,
            parent_contexts=[self._metadata_node.resource_name],
        )

        return context.Context.list(filter=filter_str, **service_request_args)

    def _get_latest_time_series_metric_columns(self) -> Dict[str, Union[float, int]]:
        """Determines the latest step for each time series metric.

        Returns:
            Dictionary mapping time series metric key to the latest step of that metric.
        """
        if self._backing_tensorboard_run:
            time_series_metrics = (
                self._backing_tensorboard_run.resource.read_time_series_data()
            )

            return {
                display_name: data.values[-1].scalar.value
                for display_name, data in time_series_metrics.items()
                if data.value_type
                == gca_tensorboard_time_series.TensorboardTimeSeries.ValueType.SCALAR
            }
        return {}

    def _log_pipeline_job(self, pipeline_job: pipeline_jobs.PipelineJob):
        """Associate this PipelineJob's Context to the current ExperimentRun Context as a child context.

        Args:
            pipeline_job (pipeline_jobs.PipelineJob):
                Required. The PipelineJob to associate.
        """

        pipeline_job_context = pipeline_job._get_context()
        self._metadata_node.add_context_children([pipeline_job_context])

    @_v1_not_supported
    def log(
        self,
        *,
        pipeline_job: Optional[pipeline_jobs.PipelineJob] = None,
    ):
        """Log a Vertex Resource to this experiment run.

        ```py
        my_run = aiplatform.ExperimentRun('my-run', experiment='my-experiment')
        my_job = aiplatform.PipelineJob(...)
        my_job.submit()
        my_run.log(my_job)
        ```

        Args:
            pipeline_job (aiplatform.PipelineJob): Optional. A Vertex PipelineJob.
        """
        if pipeline_job:
            self._log_pipeline_job(pipeline_job=pipeline_job)

    @staticmethod
    def _validate_run_id(run_id: str):
        """Validates the run id.

        Args:
            run_id(str): Required. The run id to validate.
        Raises:
            ValueError if run id is too long.
        """

        if len(run_id) > constants._EXPERIMENT_RUN_MAX_LENGTH:
            raise ValueError(
                f"Length of Experiment ID and Run ID cannot be greater than {constants._EXPERIMENT_RUN_MAX_LENGTH}. "
                f"{run_id} is of length {len(run_id)}"
            )

    @classmethod
    def create(
        cls,
        run_name: str,
        *,
        experiment: Optional[Union[experiment_resources.Experiment, str]] = None,
        tensorboard: Optional[Union[tensorboard_resource.Tensorboard, str]] = None,
        state: gca_execution.Execution.State = gca_execution.Execution.State.RUNNING,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "ExperimentRun":
        """Creates a new experiment run in Vertex AI Experiments.

        ```py
        my_run = aiplatform.ExperimentRun.create('my-run', experiment='my-experiment')
        ```

        Args:
            run_name (str): Required. The name of this run.
            experiment (Union[aiplatform.Experiment, str]):
                Optional. The name or instance of the experiment to create this run under.
                If not provided, will default to the experiment set in `aiplatform.init`.
            tensorboard (Union[aiplatform.Tensorboard, str]):
                Optional. The resource name or instance of Vertex Tensorboard to use as the backing
                Tensorboard for time series metric logging. If not provided, will default to the
                the backing tensorboard of parent experiment if set. Must be in same project and location
                as this experiment run.
            state (aiplatform.gapic.Execution.State):
                Optional. The state of this run. Defaults to RUNNING.
            project (str):
                Optional. Project where this experiment will be created. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location where this experiment will be created. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to create this experiment. Overrides
                credentials set in aiplatform.init.
        Returns:
            The newly created experiment run.
        """

        experiment = cls._get_experiment(experiment)

        run_id = _format_experiment_run_resource_id(
            experiment_name=experiment.name, run_name=run_name
        )

        cls._validate_run_id(run_id)

        def _create_context():
            with experiment_resources._SetLoggerLevel(resource):
                return context.Context._create(
                    resource_id=run_id,
                    display_name=run_name,
                    schema_title=constants.SYSTEM_EXPERIMENT_RUN,
                    schema_version=constants.SCHEMA_VERSIONS[
                        constants.SYSTEM_EXPERIMENT_RUN
                    ],
                    metadata={
                        constants._PARAM_KEY: {},
                        constants._METRIC_KEY: {},
                        constants._STATE_KEY: state.name,
                    },
                    project=project,
                    location=location,
                    credentials=credentials,
                )

        metadata_context = _create_context()

        if metadata_context is None:
            raise RuntimeError(
                f"Experiment Run with name {run_name} in {experiment.name} already exists."
            )

        experiment_run = cls.__new__(cls)
        experiment_run._experiment = experiment
        experiment_run._run_name = metadata_context.display_name
        experiment_run._metadata_node = metadata_context
        experiment_run._backing_tensorboard_run = None
        experiment_run._largest_step = None

        try:
            if tensorboard:
                cls._assign_backing_tensorboard(
                    self=experiment_run, tensorboard=tensorboard
                )
            else:
                cls._assign_to_experiment_backing_tensorboard(self=experiment_run)
        except Exception as e:
            metadata_context.delete()
            raise e

        experiment_run._associate_to_experiment(experiment)
        return experiment_run

    def _assign_to_experiment_backing_tensorboard(self):
        """Assigns parent Experiment backing tensorboard resource to this Experiment Run."""
        backing_tensorboard_resource = (
            self._experiment.get_backing_tensorboard_resource()
        )

        if backing_tensorboard_resource:
            self.assign_backing_tensorboard(tensorboard=backing_tensorboard_resource)

    @staticmethod
    def _format_tensorboard_experiment_display_name(experiment_name: str) -> str:
        """Formats Tensorboard experiment name that backs this run.
        Args:
            experiment_name (str): Required. The name of the experiment.
        Returns:
            Formatted Tensorboard Experiment name
        """
        # post fix helps distinguish from the Vertex Experiment in console
        return f"{experiment_name} Backing Tensorboard Experiment"

    def _assign_backing_tensorboard(
        self, tensorboard: Union[tensorboard_resource.Tensorboard, str]
    ):
        """Assign tensorboard as the backing tensorboard to this run.

        Args:
            tensorboard (Union[tensorboard_resource.Tensorboard, str]):
                Required. Tensorboard instance or resource name.
        """
        if isinstance(tensorboard, str):
            tensorboard = tensorboard_resource.Tensorboard(
                tensorboard, credentials=self._metadata_node.credentials
            )

        tensorboard_resource_name_parts = tensorboard._parse_resource_name(
            tensorboard.resource_name
        )
        tensorboard_experiment_resource_name = (
            tensorboard_resource.TensorboardExperiment._format_resource_name(
                experiment=self._experiment.name, **tensorboard_resource_name_parts
            )
        )
        try:
            tensorboard_experiment = tensorboard_resource.TensorboardExperiment(
                tensorboard_experiment_resource_name,
                credentials=tensorboard.credentials,
            )
        except exceptions.NotFound:
            with experiment_resources._SetLoggerLevel(tensorboard_resource):
                tensorboard_experiment = (
                    tensorboard_resource.TensorboardExperiment.create(
                        tensorboard_experiment_id=self._experiment.name,
                        display_name=self._format_tensorboard_experiment_display_name(
                            self._experiment.name
                        ),
                        tensorboard_name=tensorboard.resource_name,
                        credentials=tensorboard.credentials,
                        labels=constants._VERTEX_EXPERIMENT_TB_EXPERIMENT_LABEL,
                    )
                )

        tensorboard_experiment_name_parts = tensorboard_experiment._parse_resource_name(
            tensorboard_experiment.resource_name
        )
        tensorboard_run_resource_name = (
            tensorboard_resource.TensorboardRun._format_resource_name(
                run=self._run_name, **tensorboard_experiment_name_parts
            )
        )
        try:
            tensorboard_run = tensorboard_resource.TensorboardRun(
                tensorboard_run_resource_name
            )
        except exceptions.NotFound:
            with experiment_resources._SetLoggerLevel(tensorboard_resource):
                tensorboard_run = tensorboard_resource.TensorboardRun.create(
                    tensorboard_run_id=self._run_name,
                    tensorboard_experiment_name=tensorboard_experiment.resource_name,
                    credentials=tensorboard.credentials,
                )

        gcp_resource_url = rest_utils.make_gcp_resource_rest_url(tensorboard_run)

        with experiment_resources._SetLoggerLevel(resource):
            tensorboard_run_metadata_artifact = artifact.Artifact._create(
                uri=gcp_resource_url,
                resource_id=self._tensorboard_run_id(self._metadata_node.name),
                metadata={
                    "resourceName": tensorboard_run.resource_name,
                    constants._VERTEX_EXPERIMENT_TRACKING_LABEL: True,
                },
                schema_title=constants._TENSORBOARD_RUN_REFERENCE_ARTIFACT.schema_title,
                schema_version=constants._TENSORBOARD_RUN_REFERENCE_ARTIFACT.schema_version,
                state=gca_artifact.Artifact.State.LIVE,
            )

        self._metadata_node.add_artifacts_and_executions(
            artifact_resource_names=[tensorboard_run_metadata_artifact.resource_name]
        )

        self._backing_tensorboard_run = (
            experiment_resources._VertexResourceWithMetadata(
                resource=tensorboard_run, metadata=tensorboard_run_metadata_artifact
            )
        )

    @staticmethod
    def _tensorboard_run_id(run_id: str) -> str:
        """Helper method to format the tensorboard run artifact resource id for a run.

        Args:
            run_id: The resource id of the experiment run.

        Returns:
            Resource id for the associated tensorboard run artifact.
        """
        return f"{run_id}{constants._TB_RUN_ARTIFACT_POST_FIX_ID}"

    @_v1_not_supported
    def assign_backing_tensorboard(
        self, tensorboard: Union[tensorboard_resource.Tensorboard, str]
    ):
        """Assigns tensorboard as backing tensorboard to support timeseries metrics logging for this run.

        Args:
            tensorboard (Union[aiplatform.Tensorboard, str]):
                Required. Tensorboard instance or resource name.
        """

        backing_tensorboard = self._lookup_tensorboard_run_artifact()
        if backing_tensorboard:
            raise ValueError(
                f"Experiment run {self._run_name} already associated to tensorboard resource {backing_tensorboard.resource.resource_name}.\n"
                f"To delete backing tensorboard run, execute the following:\n"
                f'tensorboard_run_artifact = aiplatform.metadata.artifact.Artifact(artifact_name=f"{self._tensorboard_run_id(self._metadata_node.name)}")\n'
                f'tensorboard_run_resource = aiplatform.TensorboardRun(tensorboard_run_artifact.metadata["resourceName"])\n'
                f"tensorboard_run_resource.delete()\n"
                f"tensorboard_run_artifact.delete()"
            )

        self._assign_backing_tensorboard(tensorboard=tensorboard)

    def _get_latest_time_series_step(self) -> int:
        """Gets latest time series step of all time series from Tensorboard resource.

        Returns:
            Latest step of all time series metrics.
        """
        data = self._backing_tensorboard_run.resource.read_time_series_data()
        return max(ts.values[-1].step if ts.values else 0 for ts in data.values())

    @_v1_not_supported
    def log_time_series_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        wall_time: Optional[timestamp_pb2.Timestamp] = None,
    ):
        """Logs time series metrics to backing TensorboardRun of this Experiment Run.

        ```py
        run.log_time_series_metrics({'accuracy': 0.9}, step=10)
        ```

        Args:
            metrics (Dict[str, Union[str, float]]):
                Required. Dictionary of where keys are metric names and values are metric values.
            step (int):
                Optional. Step index of this data point within the run.

                If not provided, the latest
                step amongst all time series metrics already logged will be used.
            wall_time (timestamp_pb2.Timestamp):
                Optional. Wall clock timestamp when this data point is
                generated by the end user.

                If not provided, this will be generated based on the value from time.time()
        Raises:
            RuntimeError: If current experiment run doesn't have a backing Tensorboard resource.
        """

        if not self._backing_tensorboard_run:
            self._assign_to_experiment_backing_tensorboard()
            if not self._backing_tensorboard_run:
                raise RuntimeError(
                    "Please set this experiment run with backing tensorboard resource to use log_time_series_metrics."
                )

        self._soft_create_time_series(metric_keys=set(metrics.keys()))

        if step is None:
            step = self._largest_step or self._get_latest_time_series_step()
            step += 1
            self._largest_step = step

        self._backing_tensorboard_run.resource.write_tensorboard_scalar_data(
            time_series_data=metrics, step=step, wall_time=wall_time
        )

    def _soft_create_time_series(self, metric_keys: Set[str]):
        """Creates TensorboardTimeSeries for the metric keys if one currently does not exist.

        Args:
            metric_keys (Set[str]): Keys of the metrics.
        """

        if any(
            key
            not in self._backing_tensorboard_run.resource._time_series_display_name_to_id_mapping
            for key in metric_keys
        ):
            self._backing_tensorboard_run.resource._sync_time_series_display_name_to_id_mapping()

        for key in metric_keys:
            if (
                key
                not in self._backing_tensorboard_run.resource._time_series_display_name_to_id_mapping
            ):
                with experiment_resources._SetLoggerLevel(tensorboard_resource):
                    self._backing_tensorboard_run.resource.create_tensorboard_time_series(
                        display_name=key
                    )

    def log_params(self, params: Dict[str, Union[float, int, str]]):
        """Log single or multiple parameters with specified key value pairs.

        Parameters with the same key will be overwritten.

        ```py
        my_run = aiplatform.ExperimentRun('my-run', experiment='my-experiment')
        my_run.log_params({'learning_rate': 0.1, 'dropout_rate': 0.2})
        ```

        Args:
            params (Dict[str, Union[float, int, str]]):
                Required. Parameter key/value pairs.

        Raises:
            TypeError: If key is not str or value is not float, int, str.
        """
        # query the latest run execution resource before logging.
        for key, value in params.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{key} is of type {type(key).__name__} must of type str"
                )
            if not isinstance(value, (float, int, str)):
                raise TypeError(
                    f"Value for key {key} is of type {type(value).__name__} but must be one of float, int, str"
                )

        if self._is_legacy_experiment_run():
            self._metadata_node.update(metadata=params)
        else:
            self._metadata_node.update(metadata={constants._PARAM_KEY: params})

    def log_metrics(self, metrics: Dict[str, Union[float, int, str]]):
        """Log single or multiple Metrics with specified key and value pairs.

        Metrics with the same key will be overwritten.

        ```py
        my_run = aiplatform.ExperimentRun('my-run', experiment='my-experiment')
        my_run.log_metrics({'accuracy': 0.9, 'recall': 0.8})
        ```

        Args:
            metrics (Dict[str, Union[float, int, str]]):
                Required. Metrics key/value pairs.
        Raises:
            TypeError: If keys are not str or values are not float, int, or str.
        """
        for key, value in metrics.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{key} is of type {type(key).__name__} must of type str"
                )
            if not isinstance(value, (float, int, str)):
                raise TypeError(
                    f"Value for key {key} is of type {type(value).__name__} but must be one of float, int, str"
                )

        if self._is_legacy_experiment_run():
            self._metadata_metric_artifact.update(metadata=metrics)
        else:
            # TODO: query the latest metrics artifact resource before logging.
            self._metadata_node.update(metadata={constants._METRIC_KEY: metrics})

    @_v1_not_supported
    def log_classification_metrics(
        self,
        *,
        labels: Optional[List[str]] = None,
        matrix: Optional[List[List[int]]] = None,
        fpr: Optional[List[float]] = None,
        tpr: Optional[List[float]] = None,
        threshold: Optional[List[float]] = None,
        display_name: Optional[str] = None,
    ) -> google_artifact_schema.ClassificationMetrics:
        """Create an artifact for classification metrics and log to ExperimentRun. Currently supports confusion matrix and ROC curve.

        ```py
        my_run = aiplatform.ExperimentRun('my-run', experiment='my-experiment')
        classification_metrics = my_run.log_classification_metrics(
            display_name='my-classification-metrics',
            labels=['cat', 'dog'],
            matrix=[[9, 1], [1, 9]],
            fpr=[0.1, 0.5, 0.9],
            tpr=[0.1, 0.7, 0.9],
            threshold=[0.9, 0.5, 0.1],
        )
        ```

        Args:
            labels (List[str]):
                Optional. List of label names for the confusion matrix. Must be set if 'matrix' is set.
            matrix (List[List[int]):
                Optional. Values for the confusion matrix. Must be set if 'labels' is set.
            fpr (List[float]):
                Optional. List of false positive rates for the ROC curve. Must be set if 'tpr' or 'thresholds' is set.
            tpr (List[float]):
                Optional. List of true positive rates for the ROC curve. Must be set if 'fpr' or 'thresholds' is set.
            threshold (List[float]):
                Optional. List of thresholds for the ROC curve. Must be set if 'fpr' or 'tpr' is set.
            display_name (str):
                Optional. The user-defined name for the classification metric artifact.

        Returns:
            ClassificationMetrics artifact.

        Raises:
            ValueError: if 'labels' and 'matrix' are not set together
                        or if 'labels' and 'matrix' are not in the same length
                        or if 'fpr' and 'tpr' and 'threshold' are not set together
                        or if 'fpr' and 'tpr' and 'threshold' are not in the same length
        """
        if (labels or matrix) and not (labels and matrix):
            raise ValueError("labels and matrix must be set together.")

        if (fpr or tpr or threshold) and not (fpr and tpr and threshold):
            raise ValueError("fpr, tpr, and thresholds must be set together.")

        confusion_matrix = confidence_metrics = None

        if labels and matrix:
            if len(matrix) != len(labels):
                raise ValueError(
                    "Length of labels and matrix must be the same. "
                    "Got lengths {} and {} respectively.".format(
                        len(labels), len(matrix)
                    )
                )
            annotation_specs = [
                schema_utils.AnnotationSpec(display_name=label) for label in labels
            ]
            confusion_matrix = schema_utils.ConfusionMatrix(
                annotation_specs=annotation_specs,
                matrix=matrix,
            )

        if fpr and tpr and threshold:
            if (
                len(fpr) != len(tpr)
                or len(fpr) != len(threshold)
                or len(tpr) != len(threshold)
            ):
                raise ValueError(
                    "Length of fpr, tpr and threshold must be the same. "
                    "Got lengths {}, {} and {} respectively.".format(
                        len(fpr), len(tpr), len(threshold)
                    )
                )

            confidence_metrics = [
                schema_utils.ConfidenceMetric(
                    confidence_threshold=confidence_threshold,
                    false_positive_rate=false_positive_rate,
                    recall=recall,
                )
                for confidence_threshold, false_positive_rate, recall in zip(
                    threshold, fpr, tpr
                )
            ]

        classification_metrics = google_artifact_schema.ClassificationMetrics(
            display_name=display_name,
            confusion_matrix=confusion_matrix,
            confidence_metrics=confidence_metrics,
        )

        classfication_metrics = classification_metrics.create()
        self._metadata_node.add_artifacts_and_executions(
            artifact_resource_names=[classfication_metrics.resource_name]
        )
        return classification_metrics

    @_v1_not_supported
    def log_model(
        self,
        model: Union[
            "sklearn.base.BaseEstimator", "xgb.Booster", "tf.Module"  # noqa: F821
        ],
        artifact_id: Optional[str] = None,
        *,
        uri: Optional[str] = None,
        input_example: Union[
            "list", dict, "pd.DataFrame", "np.ndarray"  # noqa: F821
        ] = None,
        display_name: Optional[str] = None,
        metadata_store_id: Optional[str] = "default",
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> google_artifact_schema.ExperimentModel:
        """Saves a ML model into a MLMD artifact and log it to this ExperimentRun.

        Supported model frameworks: sklearn, xgboost, tensorflow.

        Example usage:
            model = LinearRegression()
            model.fit(X, y)
            aiplatform.init(
                project="my-project",
                location="my-location",
                staging_bucket="gs://my-bucket",
                experiment="my-exp"
            )
            with aiplatform.start_run("my-run"):
                aiplatform.log_model(model, "my-sklearn-model")

        Args:
            model (Union["sklearn.base.BaseEstimator", "xgb.Booster", "tf.Module"]):
                Required. A machine learning model.
            artifact_id (str):
                Optional. The resource id of the artifact. This id must be globally unique
                in a metadataStore. It may be up to 63 characters, and valid characters
                are `[a-z0-9_-]`. The first character cannot be a number or hyphen.
            uri (str):
                Optional. A gcs directory to save the model file. If not provided,
                `gs://default-bucket/timestamp-uuid-frameworkName-model` will be used.
                If default staging bucket is not set, a new bucket will be created.
            input_example (Union[list, dict, pd.DataFrame, np.ndarray]):
                Optional. An example of a valid model input. Will be stored as a yaml file
                in the gcs uri. Accepts list, dict, pd.DataFrame, and np.ndarray
                The value inside a list must be a scalar or list. The value inside
                a dict must be a scalar, list, or np.ndarray.
            display_name (str):
                Optional. The display name of the artifact.
            metadata_store_id (str):
                Optional. The <metadata_store_id> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/artifacts/<resource_id>
                If not provided, the MetadataStore's ID will be set to "default".
            project (str):
                Optional. Project used to create this Artifact. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location used to create this Artifact. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to create this Artifact. Overrides
                credentials set in aiplatform.init.

        Returns:
            An ExperimentModel instance.

        Raises:
            ValueError: if model type is not supported.
        """
        experiment_model = _models.save_model(
            model=model,
            artifact_id=artifact_id,
            uri=uri,
            input_example=input_example,
            display_name=display_name,
            metadata_store_id=metadata_store_id,
            project=project,
            location=location,
            credentials=credentials,
        )

        self._metadata_node.add_artifacts_and_executions(
            artifact_resource_names=[experiment_model.resource_name]
        )
        return experiment_model

    @_v1_not_supported
    def get_time_series_data_frame(self) -> "pd.DataFrame":  # noqa: F821
        """Returns all time series in this Run as a DataFrame.

        Returns:
            pd.DataFrame: Time series metrics in this Run as a Dataframe.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is not installed and is required to get dataframe as the return format. "
                'Please install the SDK using "pip install google-cloud-aiplatform[metadata]"'
            )

        if not self._backing_tensorboard_run:
            return pd.DataFrame({})
        data = self._backing_tensorboard_run.resource.read_time_series_data()

        if not data:
            return pd.DataFrame({})

        return (
            pd.DataFrame(
                {
                    name: entry.scalar.value,
                    "step": entry.step,
                    "wall_time": entry.wall_time,
                }
                for name, ts in data.items()
                for entry in ts.values
            )
            .groupby(["step", "wall_time"])
            .first()
            .reset_index()
        )

    @_v1_not_supported
    def get_logged_pipeline_jobs(self) -> List[pipeline_jobs.PipelineJob]:
        """Get all PipelineJobs associated to this experiment run.

        Returns:
            List of PipelineJobs associated this run.
        """

        pipeline_job_contexts = self._get_logged_pipeline_runs()

        return [
            pipeline_jobs.PipelineJob.get(
                c.display_name,
                project=c.project,
                location=c.location,
                credentials=c.credentials,
            )
            for c in pipeline_job_contexts
        ]

    @_v1_not_supported
    def get_logged_custom_jobs(self) -> List[jobs.CustomJob]:
        """Get all CustomJobs associated to this experiment run.

        Returns:
            List of CustomJobs associated this run.
        """

        custom_jobs = self._metadata_node.metadata.get(constants._CUSTOM_JOB_KEY)

        return [
            jobs.CustomJob.get(
                resource_name=custom_job.get(constants._CUSTOM_JOB_RESOURCE_NAME),
                credentials=self.credentials,
            )
            for custom_job in custom_jobs
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        state = (
            gca_execution.Execution.State.FAILED
            if exc_type
            else gca_execution.Execution.State.COMPLETE
        )

        if metadata._experiment_tracker.experiment_run is self:
            metadata._experiment_tracker.end_run(state=state)
        else:
            self.end_run(state=state)

    def end_run(
        self,
        *,
        state: gca_execution.Execution.State = gca_execution.Execution.State.COMPLETE,
    ):
        """Ends this experiment run and sets state to COMPLETE.

        Args:
            state (aiplatform.gapic.Execution.State):
                Optional. Override the state at the end of run. Defaults to COMPLETE.
        """
        self.update_state(state)

    def delete(self, *, delete_backing_tensorboard_run: bool = False):
        """Deletes this experiment run.

        Does not delete the executions, artifacts, or resources logged to this run.

        Args:
            delete_backing_tensorboard_run (bool):
                Optional. Whether to delete the backing tensorboard run that stores time series metrics for this run.
        """
        if delete_backing_tensorboard_run:
            if not self._is_legacy_experiment_run():
                if not self._backing_tensorboard_run:
                    self._backing_tensorboard_run = (
                        self._lookup_tensorboard_run_artifact()
                    )
                if self._backing_tensorboard_run:
                    self._backing_tensorboard_run.resource.delete()
                    self._backing_tensorboard_run.metadata.delete()
                else:
                    _LOGGER.warning(
                        f"Experiment run {self.name} does not have a backing tensorboard run."
                        " Skipping deletion."
                    )
            else:
                _LOGGER.warning(
                    f"Experiment run {self.name} does not have a backing tensorboard run."
                    " Skipping deletion."
                )
        else:
            _LOGGER.warning(
                f"Experiment run {self.name} skipped backing tensorboard run deletion.\n"
                f"To delete backing tensorboard run, execute the following:\n"
                f'tensorboard_run_artifact = aiplatform.metadata.artifact.Artifact(artifact_name=f"{self._tensorboard_run_id(self._metadata_node.name)}")\n'
                f'tensorboard_run_resource = aiplatform.TensorboardRun(tensorboard_run_artifact.metadata["resourceName"])\n'
                f"tensorboard_run_resource.delete()\n"
                f"tensorboard_run_artifact.delete()"
            )

        try:
            self._metadata_node.delete()
        except exceptions.NotFound:
            _LOGGER.warning(
                f"Experiment run {self.name} metadata node not found."
                " Skipping deletion."
            )

        if self._is_legacy_experiment_run():
            try:
                self._metadata_metric_artifact.delete()
            except exceptions.NotFound:
                _LOGGER.warning(
                    f"Experiment run {self.name} metadata node not found."
                    " Skipping deletion."
                )

    @_v1_not_supported
    def get_artifacts(self) -> List[artifact.Artifact]:
        """Get the list of artifacts associated to this run.

        Returns:
            List of artifacts associated to this run.
        """
        return self._metadata_node.get_artifacts()

    @_v1_not_supported
    def get_executions(self) -> List[execution.Execution]:
        """Get the List of Executions associated to this run

        Returns:
            List of executions associated to this run.
        """
        return self._metadata_node.get_executions()

    def get_params(self) -> Dict[str, Union[int, float, str]]:
        """Get the parameters logged to this run.

        Returns:
            Parameters logged to this experiment run.
        """
        if self._is_legacy_experiment_run():
            return self._metadata_node.metadata
        else:
            return self._metadata_node.metadata[constants._PARAM_KEY]

    def get_metrics(self) -> Dict[str, Union[float, int, str]]:
        """Get the summary metrics logged to this run.

        Returns:
            Summary metrics logged to this experiment run.
        """
        if self._is_legacy_experiment_run():
            return self._metadata_metric_artifact.metadata
        else:
            return self._metadata_node.metadata[constants._METRIC_KEY]

    def get_state(self) -> gca_execution.Execution.State:
        """The state of this run."""
        if self._is_legacy_experiment_run():
            return self._metadata_node.state.name
        else:
            return self._metadata_node.metadata[constants._STATE_KEY]

    @_v1_not_supported
    def get_classification_metrics(self) -> List[Dict[str, Union[str, List]]]:
        """Get all the classification metrics logged to this run.

        ```py
        my_run = aiplatform.ExperimentRun('my-run', experiment='my-experiment')
        metric = my_run.get_classification_metrics()[0]
        print(metric)
        ## print result:
            {
                "id": "e6c893a4-222e-4c60-a028-6a3b95dfc109",
                "display_name": "my-classification-metrics",
                "labels": ["cat", "dog"],
                "matrix": [[9,1], [1,9]],
                "fpr": [0.1, 0.5, 0.9],
                "tpr": [0.1, 0.7, 0.9],
                "thresholds": [0.9, 0.5, 0.1]
            }
        ```

        Returns:
            List of classification metrics logged to this experiment run.
        """

        artifact_list = artifact.Artifact.list(
            filter=metadata_utils._make_filter_string(
                in_context=[self.resource_name],
                schema_title=google_artifact_schema.ClassificationMetrics.schema_title,
            ),
            project=self.project,
            location=self.location,
            credentials=self.credentials,
        )

        metrics = []
        for metric_artifact in artifact_list:
            metric = {}
            metric["id"] = metric_artifact.name
            metric["display_name"] = metric_artifact.display_name
            metadata = metric_artifact.metadata
            if "confusionMatrix" in metadata:
                metric["labels"] = [
                    d["displayName"]
                    for d in metadata["confusionMatrix"]["annotationSpecs"]
                ]
                metric["matrix"] = metadata["confusionMatrix"]["rows"]

            if "confidenceMetrics" in metadata:
                metric["fpr"] = [
                    d["falsePositiveRate"] for d in metadata["confidenceMetrics"]
                ]
                metric["tpr"] = [d["recall"] for d in metadata["confidenceMetrics"]]
                metric["threshold"] = [
                    d["confidenceThreshold"] for d in metadata["confidenceMetrics"]
                ]
            metrics.append(metric)

        return metrics

    @_v1_not_supported
    def get_experiment_models(self) -> List[google_artifact_schema.ExperimentModel]:
        """Get all ExperimentModel associated to this experiment run.

        Returns:
            List of ExperimentModel instances associated this run.
        """
        experiment_model_list = google_artifact_schema.ExperimentModel.list(
            filter=metadata_utils._make_filter_string(in_context=[self.resource_name]),
            project=self.project,
            location=self.location,
            credentials=self.credentials,
        )

        return experiment_model_list

    @_v1_not_supported
    def associate_execution(self, execution: execution.Execution):
        """Associate an execution to this experiment run.

        Args:
            execution (aiplatform.Execution): Execution to associate to this run.
        """
        self._metadata_node.add_artifacts_and_executions(
            execution_resource_names=[execution.resource_name]
        )

    def _association_wrapper(self, f: Callable[..., Any]) -> Callable[..., Any]:
        """Wraps methods and automatically associates all passed in Artifacts or Executions to this ExperimentRun.

        This is used to wrap artifact passing methods of Executions so they get associated to this run.
        """

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            artifacts = []
            executions = []
            for value in [*args, *kwargs.values()]:
                value = value if isinstance(value, abc.Iterable) else [value]
                for item in value:
                    if isinstance(item, execution.Execution):
                        executions.append(item)
                    elif isinstance(item, artifact.Artifact):
                        artifacts.append(item)
                    elif artifact._VertexResourceArtifactResolver.supports_metadata(
                        item
                    ):
                        artifacts.append(
                            artifact._VertexResourceArtifactResolver.resolve_or_create_resource_artifact(
                                item
                            )
                        )

            if artifacts or executions:
                self._metadata_node.add_artifacts_and_executions(
                    artifact_resource_names=[a.resource_name for a in artifacts],
                    execution_resource_names=[e.resource_name for e in executions],
                )

            result = f(*args, **kwargs)
            return result

        return wrapper
