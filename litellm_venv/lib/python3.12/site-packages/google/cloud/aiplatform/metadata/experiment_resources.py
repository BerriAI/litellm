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

import abc
import concurrent.futures
from dataclasses import dataclass
import logging
from typing import Dict, List, NamedTuple, Optional, Tuple, Type, Union

from google.api_core import exceptions
from google.auth import credentials as auth_credentials

from google.cloud.aiplatform import base
from google.cloud.aiplatform.metadata import artifact
from google.cloud.aiplatform.metadata import constants
from google.cloud.aiplatform.metadata import context
from google.cloud.aiplatform.metadata import execution
from google.cloud.aiplatform.metadata import metadata
from google.cloud.aiplatform.metadata import metadata_store
from google.cloud.aiplatform.metadata import resource
from google.cloud.aiplatform.metadata import utils as metadata_utils
from google.cloud.aiplatform.tensorboard import tensorboard_resource

_LOGGER = base.Logger(__name__)


@dataclass
class _ExperimentRow:
    """Class for representing a run row in an Experiments Dataframe.

    Attributes:
        params (Dict[str, Union[float, int, str]]): Optional. The parameters of this run.
        metrics (Dict[str, Union[float, int, str]]): Optional. The metrics of this run.
        time_series_metrics (Dict[str, float]): Optional. The latest time series metrics of this run.
        experiment_run_type (Optional[str]): Optional. The type of this run.
        name (str): Optional. The name of this run.
        state (str): Optional. The state of this run.
    """

    params: Optional[Dict[str, Union[float, int, str]]] = None
    metrics: Optional[Dict[str, Union[float, int, str]]] = None
    time_series_metrics: Optional[Dict[str, float]] = None
    experiment_run_type: Optional[str] = None
    name: Optional[str] = None
    state: Optional[str] = None

    def to_dict(self) -> Dict[str, Union[float, int, str]]:
        """Converts this experiment row into a dictionary.

        Returns:
            Row as a dictionary.
        """
        result = {
            "run_type": self.experiment_run_type,
            "run_name": self.name,
            "state": self.state,
        }
        for prefix, field in [
            (constants._PARAM_PREFIX, self.params),
            (constants._METRIC_PREFIX, self.metrics),
            (constants._TIME_SERIES_METRIC_PREFIX, self.time_series_metrics),
        ]:
            if field:
                result.update(
                    {f"{prefix}.{key}": value for key, value in field.items()}
                )
        return result


class Experiment:
    """Represents a Vertex AI Experiment resource."""

    def __init__(
        self,
        experiment_name: str,
        *,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """

        ```py
        my_experiment = aiplatform.Experiment('my-experiment')
        ```

        Args:
            experiment_name (str):
                Required. The name or resource name of this experiment.

                Resource name is of the format:
                `projects/123/locations/us-central1/metadataStores/default/contexts/my-experiment`
            project (str):
                Optional. Project where this experiment is located. Overrides
                project set in aiplatform.init.
            location (str):
                Optional. Location where this experiment is located. Overrides
                location set in aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to retrieve this experiment.
                Overrides credentials set in aiplatform.init.
        """

        metadata_args = dict(
            resource_name=experiment_name,
            project=project,
            location=location,
            credentials=credentials,
        )

        with _SetLoggerLevel(resource):
            experiment_context = context.Context(**metadata_args)
        self._validate_experiment_context(experiment_context)

        self._metadata_context = experiment_context

    @staticmethod
    def _validate_experiment_context(experiment_context: context.Context):
        """Validates this context is an experiment context.

        Args:
            experiment_context (context._Context): Metadata context.
        Raises:
            ValueError: If Metadata context is not an experiment context or a TensorboardExperiment.
        """
        if experiment_context.schema_title != constants.SYSTEM_EXPERIMENT:
            raise ValueError(
                f"Experiment name {experiment_context.name} is of type "
                f"({experiment_context.schema_title}) in this MetadataStore. "
                f"It must of type {constants.SYSTEM_EXPERIMENT}."
            )
        if Experiment._is_tensorboard_experiment(experiment_context):
            raise ValueError(
                f"Experiment name {experiment_context.name} is a TensorboardExperiment context "
                f"and cannot be used as a Vertex AI Experiment."
            )

    @staticmethod
    def _is_tensorboard_experiment(context: context.Context) -> bool:
        """Returns True if Experiment is a Tensorboard Experiment created by CustomJob."""
        return constants.TENSORBOARD_CUSTOM_JOB_EXPERIMENT_FIELD in context.metadata

    @property
    def name(self) -> str:
        """The name of this experiment."""
        return self._metadata_context.name

    @classmethod
    def create(
        cls,
        experiment_name: str,
        *,
        description: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "Experiment":
        """Creates a new experiment in Vertex AI Experiments.

        ```py
        my_experiment = aiplatform.Experiment.create('my-experiment', description='my description')
        ```

        Args:
            experiment_name (str): Required. The name of this experiment.
            description (str): Optional. Describes this experiment's purpose.
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
            The newly created experiment.
        """

        metadata_store._MetadataStore.ensure_default_metadata_store_exists(
            project=project, location=location, credentials=credentials
        )

        with _SetLoggerLevel(resource):
            experiment_context = context.Context._create(
                resource_id=experiment_name,
                display_name=experiment_name,
                description=description,
                schema_title=constants.SYSTEM_EXPERIMENT,
                schema_version=metadata._get_experiment_schema_version(),
                metadata=constants.EXPERIMENT_METADATA,
                project=project,
                location=location,
                credentials=credentials,
            )

        self = cls.__new__(cls)
        self._metadata_context = experiment_context

        return self

    @classmethod
    def get(
        cls,
        experiment_name: str,
        *,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> Optional["Experiment"]:
        """Gets experiment if one exists with this experiment_name in Vertex AI Experiments.

        Args:
            experiment_name (str):
                Required. The name of this experiment.
            project (str):
                Optional. Project used to retrieve this resource.
                Overrides project set in aiplatform.init.
            location (str):
                Optional. Location used to retrieve this resource.
                Overrides location set in aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to retrieve this resource.
                Overrides credentials set in aiplatform.init.

        Returns:
            Vertex AI experiment or None if no resource was found.
        """
        try:
            return cls(
                experiment_name=experiment_name,
                project=project,
                location=location,
                credentials=credentials,
            )
        except exceptions.NotFound:
            return None

    @classmethod
    def get_or_create(
        cls,
        experiment_name: str,
        *,
        description: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> "Experiment":
        """Gets experiment if one exists with this experiment_name in Vertex AI Experiments.

        Otherwise creates this experiment.

        ```py
        my_experiment = aiplatform.Experiment.get_or_create('my-experiment', description='my description')
        ```

        Args:
            experiment_name (str): Required. The name of this experiment.
            description (str): Optional. Describes this experiment's purpose.
            project (str):
                Optional. Project where this experiment will be retrieved from or created. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location where this experiment will be retrieved from or created. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to retrieve or create this experiment. Overrides
                credentials set in aiplatform.init.
        Returns:
            Vertex AI experiment.
        """

        metadata_store._MetadataStore.ensure_default_metadata_store_exists(
            project=project, location=location, credentials=credentials
        )

        with _SetLoggerLevel(resource):
            experiment_context = context.Context.get_or_create(
                resource_id=experiment_name,
                display_name=experiment_name,
                description=description,
                schema_title=constants.SYSTEM_EXPERIMENT,
                schema_version=metadata._get_experiment_schema_version(),
                metadata=constants.EXPERIMENT_METADATA,
                project=project,
                location=location,
                credentials=credentials,
            )

        cls._validate_experiment_context(experiment_context)

        if description and description != experiment_context.description:
            experiment_context.update(description=description)

        self = cls.__new__(cls)
        self._metadata_context = experiment_context

        return self

    @classmethod
    def list(
        cls,
        *,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> List["Experiment"]:
        """List all Vertex AI Experiments in the given project.

        ```py
        my_experiments = aiplatform.Experiment.list()
        ```

        Args:
            project (str):
                Optional. Project to list these experiments from. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location to list these experiments from. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials to list these experiments. Overrides
                credentials set in aiplatform.init.
        Returns:
            List of Vertex AI experiments.
        """

        filter_str = metadata_utils._make_filter_string(
            schema_title=constants.SYSTEM_EXPERIMENT
        )

        with _SetLoggerLevel(resource):
            experiment_contexts = context.Context.list(
                filter=filter_str,
                project=project,
                location=location,
                credentials=credentials,
            )

        experiments = []
        for experiment_context in experiment_contexts:
            # Filters Tensorboard Experiments
            if not cls._is_tensorboard_experiment(experiment_context):
                experiment = cls.__new__(cls)
                experiment._metadata_context = experiment_context
                experiments.append(experiment)
        return experiments

    @property
    def resource_name(self) -> str:
        """The Metadata context resource name of this experiment."""
        return self._metadata_context.resource_name

    @property
    def backing_tensorboard_resource_name(self) -> Optional[str]:
        """The Tensorboard resource associated with this Experiment if there is one."""
        return self._metadata_context.metadata.get(
            constants._BACKING_TENSORBOARD_RESOURCE_KEY
        )

    def delete(self, *, delete_backing_tensorboard_runs: bool = False):
        """Deletes this experiment all the experiment runs under this experiment

        Does not delete Pipeline runs, Artifacts, or Executions associated to this experiment
        or experiment runs in this experiment.

        ```py
        my_experiment = aiplatform.Experiment('my-experiment')
        my_experiment.delete(delete_backing_tensorboard_runs=True)
        ```

        Args:
            delete_backing_tensorboard_runs (bool):
                Optional. If True will also delete the Tensorboard Runs associated to the experiment
                runs under this experiment that we used to store time series metrics.
        """

        experiment_runs = _SUPPORTED_LOGGABLE_RESOURCES[context.Context][
            constants.SYSTEM_EXPERIMENT_RUN
        ].list(experiment=self)
        for experiment_run in experiment_runs:
            experiment_run.delete(
                delete_backing_tensorboard_run=delete_backing_tensorboard_runs
            )
        try:
            self._metadata_context.delete()
        except exceptions.NotFound:
            _LOGGER.warning(
                f"Experiment {self.name} metadata node not found. Skipping deletion."
            )

    def get_data_frame(self) -> "pd.DataFrame":  # noqa: F821
        """Get parameters, metrics, and time series metrics of all runs in this experiment as Dataframe.

        ```py
        my_experiment = aiplatform.Experiment('my-experiment')
        df = my_experiment.get_data_frame()
        ```

        Returns:
            pd.DataFrame: Pandas Dataframe of Experiment Runs.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is not installed and is required to get dataframe as the return format. "
                'Please install the SDK using "pip install google-cloud-aiplatform[metadata]"'
            )

        service_request_args = dict(
            project=self._metadata_context.project,
            location=self._metadata_context.location,
            credentials=self._metadata_context.credentials,
        )

        filter_str = metadata_utils._make_filter_string(
            schema_title=sorted(
                list(_SUPPORTED_LOGGABLE_RESOURCES[context.Context].keys())
            ),
            parent_contexts=[self._metadata_context.resource_name],
        )
        contexts = context.Context.list(filter_str, **service_request_args)

        filter_str = metadata_utils._make_filter_string(
            schema_title=list(
                _SUPPORTED_LOGGABLE_RESOURCES[execution.Execution].keys()
            ),
            in_context=[self._metadata_context.resource_name],
        )

        executions = execution.Execution.list(filter_str, **service_request_args)

        rows = []
        if contexts or executions:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max([len(contexts), len(executions)])
            ) as executor:
                futures = [
                    executor.submit(
                        _SUPPORTED_LOGGABLE_RESOURCES[context.Context][
                            metadata_context.schema_title
                        ]._query_experiment_row,
                        metadata_context,
                    )
                    for metadata_context in contexts
                ]

                # backward compatibility
                futures.extend(
                    executor.submit(
                        _SUPPORTED_LOGGABLE_RESOURCES[execution.Execution][
                            metadata_execution.schema_title
                        ]._query_experiment_row,
                        metadata_execution,
                    )
                    for metadata_execution in executions
                )

                for future in futures:
                    try:
                        row_dict = future.result().to_dict()
                    except Exception as exc:
                        raise ValueError(
                            f"Failed to get experiment row for {self.name}"
                        ) from exc
                    else:
                        row_dict.update({"experiment_name": self.name})
                        rows.append(row_dict)

        df = pd.DataFrame(rows)

        column_name_sort_map = {
            "experiment_name": -1,
            "run_name": 1,
            "run_type": 2,
            "state": 3,
        }

        def column_sort_key(key: str) -> int:
            """Helper method to reorder columns."""
            order = column_name_sort_map.get(key)
            if order:
                return order
            elif key.startswith("param"):
                return 5
            elif key.startswith("metric"):
                return 6
            else:
                return 7

        columns = df.columns
        columns = sorted(columns, key=column_sort_key)
        df = df.reindex(columns, axis=1)

        return df

    def _lookup_backing_tensorboard(self) -> Optional[tensorboard_resource.Tensorboard]:
        """Returns backing tensorboard if one is set.

        Returns:
            Tensorboard resource if one exists, otherwise returns None.
        """
        tensorboard_resource_name = self._metadata_context.metadata.get(
            constants._BACKING_TENSORBOARD_RESOURCE_KEY
        )

        if not tensorboard_resource_name:
            with _SetLoggerLevel(resource):
                self._metadata_context.sync_resource()
            tensorboard_resource_name = self._metadata_context.metadata.get(
                constants._BACKING_TENSORBOARD_RESOURCE_KEY
            )

        if tensorboard_resource_name:
            try:
                return tensorboard_resource.Tensorboard(
                    tensorboard_resource_name,
                    credentials=self._metadata_context.credentials,
                )
            except exceptions.NotFound:
                self._metadata_context.update(
                    metadata={constants._BACKING_TENSORBOARD_RESOURCE_KEY: None}
                )
        return None

    def get_backing_tensorboard_resource(
        self,
    ) -> Optional[tensorboard_resource.Tensorboard]:
        """Get the backing tensorboard for this experiment if one exists.

        ```py
        my_experiment = aiplatform.Experiment('my-experiment')
        tb = my_experiment.get_backing_tensorboard_resource()
        ```

        Returns:
            Backing Tensorboard resource for this experiment if one exists.
        """
        return self._lookup_backing_tensorboard()

    def assign_backing_tensorboard(
        self, tensorboard: Union[tensorboard_resource.Tensorboard, str]
    ):
        """Assigns tensorboard as backing tensorboard to support time series metrics logging.

        ```py
        tb = aiplatform.Tensorboard('tensorboard-resource-id')
        my_experiment = aiplatform.Experiment('my-experiment')
        my_experiment.assign_backing_tensorboard(tb)
        ```

        Args:
            tensorboard (Union[aiplatform.Tensorboard, str]):
                Required. Tensorboard resource or resource name to associate to this experiment.

        Raises:
            ValueError: If this experiment already has a previously set backing tensorboard resource.
            ValueError: If Tensorboard is not in same project and location as this experiment.
        """

        backing_tensorboard = self._lookup_backing_tensorboard()
        if backing_tensorboard:
            tensorboard_resource_name = (
                tensorboard
                if isinstance(tensorboard, str)
                else tensorboard.resource_name
            )
            if tensorboard_resource_name != backing_tensorboard.resource_name:
                raise ValueError(
                    f"Experiment {self._metadata_context.name} already associated '"
                    f"to tensorboard resource {backing_tensorboard.resource_name}"
                )

        if isinstance(tensorboard, str):
            tensorboard = tensorboard_resource.Tensorboard(
                tensorboard,
                project=self._metadata_context.project,
                location=self._metadata_context.location,
                credentials=self._metadata_context.credentials,
            )

        if tensorboard.project not in self._metadata_context._project_tuple:
            raise ValueError(
                f"Tensorboard is in project {tensorboard.project} but must be in project {self._metadata_context.project}"
            )
        if tensorboard.location != self._metadata_context.location:
            raise ValueError(
                f"Tensorboard is in location {tensorboard.location} but must be in location {self._metadata_context.location}"
            )

        self._metadata_context.update(
            metadata={
                constants._BACKING_TENSORBOARD_RESOURCE_KEY: tensorboard.resource_name
            }
        )

    def _log_experiment_loggable(self, experiment_loggable: "_ExperimentLoggable"):
        """Associates a Vertex resource that can be logged to an Experiment as run of this experiment.

        Args:
            experiment_loggable (_ExperimentLoggable):
                A Vertex Resource that can be logged to an Experiment directly.
        """
        context = experiment_loggable._get_context()
        self._metadata_context.add_context_children([context])

    @property
    def dashboard_url(self) -> Optional[str]:
        """Cloud console URL for this resource."""
        url = f"https://console.cloud.google.com/vertex-ai/experiments/locations/{self._metadata_context.location}/experiments/{self._metadata_context.name}?project={self._metadata_context.project}"
        return url


class _SetLoggerLevel:
    """Helper method to suppress logging."""

    def __init__(self, module):
        self._module = module

    def __enter__(self):
        logging.getLogger(self._module.__name__).setLevel(logging.WARNING)

    def __exit__(self, exc_type, exc_value, traceback):
        logging.getLogger(self._module.__name__).setLevel(logging.INFO)


class _VertexResourceWithMetadata(NamedTuple):
    """Represents a resource coupled with it's metadata representation"""

    resource: base.VertexAiResourceNoun
    metadata: Union[artifact.Artifact, execution.Execution, context.Context]


class _ExperimentLoggableSchema(NamedTuple):
    """Used with _ExperimentLoggable to capture Metadata representation information about resoure.

    For example:
    _ExperimentLoggableSchema(title='system.PipelineRun', type=context._Context)

    Defines the schema and metadata type to lookup PipelineJobs.
    """

    title: str
    type: Union[Type[context.Context], Type[execution.Execution]] = context.Context


class _ExperimentLoggable(abc.ABC):
    """Abstract base class to define a Vertex Resource as loggable against an Experiment.

    For example:
    class PipelineJob(..., experiment_loggable_schemas=
        (_ExperimentLoggableSchema(title='system.PipelineRun'), )

    """

    def __init_subclass__(
        cls, *, experiment_loggable_schemas: Tuple[_ExperimentLoggableSchema], **kwargs
    ):
        """Register the metadata_schema for the subclass so Experiment can use it to retrieve the associated types.

        usage:

        class PipelineJob(..., experiment_loggable_schemas=
            (_ExperimentLoggableSchema(title='system.PipelineRun'), )

        Args:
            experiment_loggable_schemas:
                Tuple of the schema_title and type pairs that represent this resource. Note that a single item in the
                tuple will be most common. Currently only experiment run has multiple representation for backwards
                compatibility. Almost all schemas should be Contexts and Execution is currently only supported
                for backwards compatibility of experiment runs.

        """
        super().__init_subclass__(**kwargs)

        # register the type when module is loaded
        for schema in experiment_loggable_schemas:
            _SUPPORTED_LOGGABLE_RESOURCES[schema.type][schema.title] = cls

    @abc.abstractmethod
    def _get_context(self) -> context.Context:
        """Should return the  metadata context that represents this resource.

        The subclass should enforce this context exists.

        Returns:
            Context that represents this resource.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def _query_experiment_row(
        cls, node: Union[context.Context, execution.Execution]
    ) -> _ExperimentRow:
        """Should return parameters and metrics for this resource as a run row.

        Args:
            node: The metadata node that represents this resource.
        Returns:
            A populated run row for this resource.
        """
        pass

    def _validate_experiment(self, experiment: Union[str, Experiment]):
        """Validates experiment is accessible. Can be used by subclass to throw before creating the intended resource.

        Args:
            experiment (Union[str, Experiment]): The experiment that this resource will be associated to.

        Raises:
            RuntimeError: If service raises any exception when trying to access this experiment.
            ValueError: If resource project or location do not match experiment project or location.
        """

        if isinstance(experiment, str):
            try:
                experiment = Experiment.get_or_create(
                    experiment,
                    project=self.project,
                    location=self.location,
                    credentials=self.credentials,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Experiment {experiment} could not be found or created. {self.__class__.__name__} not created"
                ) from e

        if self.project not in experiment._metadata_context._project_tuple:
            raise ValueError(
                f"{self.__class__.__name__} project {self.project} does not match experiment "
                f"{experiment.name} project {experiment.project}"
            )

        if experiment._metadata_context.location != self.location:
            raise ValueError(
                f"{self.__class__.__name__} location {self.location} does not match experiment "
                f"{experiment.name} location {experiment.location}"
            )

    def _associate_to_experiment(self, experiment: Union[str, Experiment]):
        """Associates this resource to the provided Experiment.

        Args:
            experiment (Union[str, Experiment]): Required. Experiment name or experiment instance.

        Raises:
            RuntimeError: If Metadata service cannot associate resource to Experiment.
        """
        experiment_name = experiment if isinstance(experiment, str) else experiment.name
        _LOGGER.info(
            "Associating %s to Experiment: %s" % (self.resource_name, experiment_name)
        )

        try:
            if isinstance(experiment, str):
                experiment = Experiment.get_or_create(
                    experiment,
                    project=self.project,
                    location=self.location,
                    credentials=self.credentials,
                )
            experiment._log_experiment_loggable(self)
        except Exception as e:
            raise RuntimeError(
                f"{self.resource_name} could not be associated with Experiment {experiment.name}"
            ) from e


# maps context names to their resources classes
# used by the Experiment implementation to filter for representations in the metadata store
# populated at module import time from class that inherit _ExperimentLoggable
# example mapping:
# {Metadata Type} -> {schema title} -> {vertex sdk class}
# Context -> 'system.PipelineRun' -> aiplatform.PipelineJob
# Context -> 'system.ExperimentRun' -> aiplatform.ExperimentRun
# Execution -> 'system.Run' -> aiplatform.ExperimentRun
_SUPPORTED_LOGGABLE_RESOURCES: Dict[
    Union[Type[context.Context], Type[execution.Execution]],
    Dict[str, _ExperimentLoggable],
] = {execution.Execution: dict(), context.Context: dict()}
