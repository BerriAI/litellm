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

import datetime
import logging
import os
from typing import Dict, Union, Optional, Any, List

from google.api_core import exceptions
import google.auth
from google.auth import credentials as auth_credentials
from google.protobuf import timestamp_pb2

from google.cloud.aiplatform import base
from google.cloud.aiplatform import pipeline_jobs
from google.cloud.aiplatform.compat.types import execution as gca_execution
from google.cloud.aiplatform.metadata import constants
from google.cloud.aiplatform.metadata import context
from google.cloud.aiplatform.metadata import execution
from google.cloud.aiplatform.metadata import experiment_resources
from google.cloud.aiplatform.metadata import experiment_run_resource
from google.cloud.aiplatform.metadata.schema.google import (
    artifact_schema as google_artifact_schema,
)
from google.cloud.aiplatform.tensorboard import tensorboard_resource
from google.cloud.aiplatform.utils import autologging_utils
from google.cloud.aiplatform.utils import _ipython_utils

from google.cloud.aiplatform_v1.types import execution as execution_v1

_LOGGER = base.Logger(__name__)


class _MLFlowLogFilter(logging.Filter):
    """Log filter to only show MLFlow logs for unsupported framework versions."""

    def filter(self, record) -> bool:
        if record.msg.startswith("You are using an unsupported version"):
            return True
        else:
            return False


def _get_experiment_schema_version() -> str:
    """Helper method to get experiment schema version

    Returns:
        str: schema version of the currently set experiment tracking version
    """
    return constants.SCHEMA_VERSIONS[constants.SYSTEM_EXPERIMENT]


def _get_or_create_default_tensorboard() -> tensorboard_resource.Tensorboard:
    """Helper method to get the default TensorBoard instance if already exists, or create a default TensorBoard instance.

    Returns:
         tensorboard_resource.Tensorboard: the default TensorBoard instance.
    """
    tensorboards = tensorboard_resource.Tensorboard.list(filter="is_default=true")
    if tensorboards:
        return tensorboards[0]
    else:
        default_tensorboard = tensorboard_resource.Tensorboard.create(
            display_name="Default Tensorboard "
            + datetime.datetime.now().isoformat(sep=" "),
            is_default=True,
        )
        return default_tensorboard


# Legacy Experiment tracking
# Maintaining creation APIs for backwards compatibility testing
class _LegacyExperimentService:
    """Contains the exposed APIs to interact with the Managed Metadata Service."""

    @staticmethod
    def get_pipeline_df(pipeline: str) -> "pd.DataFrame":  # noqa: F821
        """Returns a Pandas DataFrame of the parameters and metrics associated with one pipeline.

        Args:
            pipeline: Name of the Pipeline to filter results.

        Returns:
            Pandas Dataframe of Pipeline with metrics and parameters.
        """

        source = "pipeline"
        pipeline_resource_name = (
            _LegacyExperimentService._get_experiment_or_pipeline_resource_name(
                name=pipeline, source=source, expected_schema=constants.SYSTEM_PIPELINE
            )
        )

        return _LegacyExperimentService._query_runs_to_data_frame(
            context_id=pipeline,
            context_resource_name=pipeline_resource_name,
            source=source,
        )

    @staticmethod
    def _get_experiment_or_pipeline_resource_name(
        name: str, source: str, expected_schema: str
    ) -> str:
        """Get the full resource name of the Context representing an Experiment or Pipeline.

        Args:
            name (str):
                Name of the Experiment or Pipeline.
            source (str):
                Identify whether the this is an Experiment or a Pipeline.
            expected_schema (str):
                expected_schema identifies the expected schema used for Experiment or Pipeline.

        Returns:
            The full resource name of the Experiment or Pipeline Context.

        Raise:
            NotFound exception if experiment or pipeline does not exist.
        """

        this_context = context.Context(resource_name=name)

        if this_context.schema_title != expected_schema:
            raise ValueError(
                f"Please provide a valid {source} name. {name} is not a {source}."
            )
        return this_context.resource_name

    @staticmethod
    def _query_runs_to_data_frame(
        context_id: str, context_resource_name: str, source: str
    ) -> "pd.DataFrame":  # noqa: F821
        """Get metrics and parameters associated with a given Context into a Dataframe.

        Args:
            context_id (str):
                Name of the Experiment or Pipeline.
            context_resource_name (str):
                Full resource name of the Context associated with an Experiment or Pipeline.
            source (str):
                Identify whether the this is an Experiment or a Pipeline.

        Returns:
            The full resource name of the Experiment or Pipeline Context.
        """

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is not installed and is required to get dataframe as the return format. "
                'Please install the SDK using "pip install google-cloud-aiplatform[metadata]"'
            )

        filter = f'schema_title="{constants.SYSTEM_RUN}" AND in_context("{context_resource_name}")'
        run_executions = execution.Execution.list(filter=filter)

        context_summary = []
        for run_execution in run_executions:
            run_dict = {
                f"{source}_name": context_id,
                "run_name": run_execution.display_name,
            }
            run_dict.update(
                _LegacyExperimentService._execution_to_column_named_metadata(
                    "param", run_execution.metadata
                )
            )

            for metric_artifact in run_execution.get_output_artifacts():
                run_dict.update(
                    _LegacyExperimentService._execution_to_column_named_metadata(
                        "metric", metric_artifact.metadata
                    )
                )

            context_summary.append(run_dict)

        return pd.DataFrame(context_summary)

    @staticmethod
    def _execution_to_column_named_metadata(
        metadata_type: str, metadata: Dict, filter_prefix: Optional[str] = None
    ) -> Dict[str, Union[int, float, str]]:
        """Returns a dict of the Execution/Artifact metadata with column names.

        Args:
          metadata_type: The type of this execution properties (param, metric).
          metadata: Either an Execution or Artifact metadata field.
          filter_prefix:
            Remove this prefix from the key of metadata field. Mainly used for removing
            "input:" from PipelineJob parameter keys

        Returns:
          Dict of custom properties with keys mapped to column names
        """
        column_key_to_value = {}
        for key, value in metadata.items():
            if filter_prefix and key.startswith(filter_prefix):
                key = key[len(filter_prefix) :]
            column_key_to_value[".".join([metadata_type, key])] = value

        return column_key_to_value


class _ExperimentTracker:
    """Tracks Experiments and Experiment Runs with high level APIs."""

    def __init__(self):
        self._experiment: Optional[experiment_resources.Experiment] = None
        self._experiment_run: Optional[experiment_run_resource.ExperimentRun] = None
        self._global_tensorboard: Optional[tensorboard_resource.Tensorboard] = None
        self._existing_tracking_uri: Optional[str] = None

    def reset(self):
        """Resets this experiment tracker, clearing the current experiment and run."""
        self._experiment = None
        self._experiment_run = None

    def _get_global_tensorboard(self) -> Optional[tensorboard_resource.Tensorboard]:
        """Helper method to get the global TensorBoard instance.

        Returns:
            tensorboard_resource.Tensorboard: the global TensorBoard instance.
        """
        if self._global_tensorboard:
            credentials, _ = google.auth.default()
            if self.experiment and self.experiment._metadata_context.credentials:
                credentials = self.experiment._metadata_context.credentials
            try:
                return tensorboard_resource.Tensorboard(
                    self._global_tensorboard.resource_name,
                    project=self._global_tensorboard.project,
                    location=self._global_tensorboard.location,
                    credentials=credentials,
                )
            except exceptions.NotFound:
                self._global_tensorboard = None
        return None

    @property
    def experiment_name(self) -> Optional[str]:
        """Return the currently set experiment name, if experiment is not set, return None"""
        if self.experiment:
            return self.experiment.name
        return None

    @property
    def experiment(self) -> Optional[experiment_resources.Experiment]:
        """Returns the currently set Experiment or Experiment set via env variable AIP_EXPERIMENT_NAME."""
        if self._experiment:
            return self._experiment
        if os.getenv(constants.ENV_EXPERIMENT_KEY):
            self._experiment = experiment_resources.Experiment.get(
                os.getenv(constants.ENV_EXPERIMENT_KEY)
            )
            return self._experiment
        return None

    @property
    def experiment_run(self) -> Optional[experiment_run_resource.ExperimentRun]:
        """Returns the currently set experiment run or experiment run set via env variable AIP_EXPERIMENT_RUN_NAME."""
        if self._experiment_run:
            return self._experiment_run
        if os.getenv(constants.ENV_EXPERIMENT_RUN_KEY):
            self._experiment_run = experiment_run_resource.ExperimentRun.get(
                os.getenv(constants.ENV_EXPERIMENT_RUN_KEY),
                experiment=self.experiment,
            )
            return self._experiment_run
        return None

    def set_experiment(
        self,
        experiment: str,
        *,
        description: Optional[str] = None,
        backing_tensorboard: Optional[
            Union[str, tensorboard_resource.Tensorboard, bool]
        ] = None,
    ):
        """Set the experiment. Will retrieve the Experiment if it exists or create one with the provided name.

        Args:
            experiment (str):
                Required. Name of the experiment to set.
            description (str):
                Optional. Description of an experiment.
            backing_tensorboard Union[str, aiplatform.Tensorboard, bool]:
                Optional. If provided, assigns tensorboard as backing tensorboard to support time series metrics
                logging.

                If ommitted, or set to `True` or `None`, the global tensorboard is used.
                If no global tensorboard is set, the default tensorboard will be used, and created if it does not exist.

                To disable using a backing tensorboard, set `backing_tensorboard` to `False`.
                To maintain this behavior, set `experiment_tensorboard` to `False` in subsequent calls to aiplatform.init().
        """
        self.reset()

        experiment = experiment_resources.Experiment.get_or_create(
            experiment_name=experiment, description=description
        )

        if backing_tensorboard and not isinstance(backing_tensorboard, bool):
            backing_tb = backing_tensorboard
        elif isinstance(backing_tensorboard, bool) and not backing_tensorboard:
            backing_tb = None
        else:
            backing_tb = (
                self._get_global_tensorboard() or _get_or_create_default_tensorboard()
            )

        current_backing_tb = experiment.backing_tensorboard_resource_name

        if not current_backing_tb and backing_tb:
            experiment.assign_backing_tensorboard(tensorboard=backing_tb)

        _ipython_utils.display_experiment_button(experiment)

        self._experiment = experiment

    def set_tensorboard(
        self,
        tensorboard: Union[
            tensorboard_resource.Tensorboard,
            str,
        ],
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ):
        """Sets the global Tensorboard resource for this session.

        Args:
            tensorboard (Union[str, aiplatform.Tensorboard]):
                Required. The Tensorboard resource to set as the global Tensorboard.
            project (str):
                Optional. Project associated with this Tensorboard resource.
            location (str):
                Optional. Location associated with this Tensorboard resource.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to set this Tensorboard resource.
        """
        if tensorboard and isinstance(tensorboard, str):
            tensorboard = tensorboard_resource.Tensorboard(
                tensorboard,
                project=project,
                location=location,
                credentials=credentials,
            )

        self._global_tensorboard = tensorboard

    def _initialize_mlflow_plugin():
        """Invokes the Vertex MLFlow plugin.

        Adding our log filter to MLFlow before calling mlflow.autolog() with
        silent=False will only surface warning logs when the installed ML
        framework version used for autologging is not supported by MLFlow.
        """

        import mlflow
        from mlflow.tracking._tracking_service import utils as mlflow_tracking_utils
        from google.cloud.aiplatform._mlflow_plugin._vertex_mlflow_tracking import (
            _VertexMlflowTracking,
        )

        # Only show MLFlow warning logs for ML framework version mismatches
        logging.getLogger("mlflow").setLevel(logging.WARNING)
        logging.getLogger("mlflow.tracking.fluent").disabled = True
        logging.getLogger("mlflow.utils.autologging_utils").addFilter(
            _MLFlowLogFilter()
        )

        mlflow_tracking_utils._tracking_store_registry.register(
            "vertex-mlflow-plugin", _VertexMlflowTracking
        )

        mlflow.set_tracking_uri("vertex-mlflow-plugin://")

        mlflow.autolog(
            log_input_examples=False,
            log_model_signatures=False,
            log_models=False,
            silent=False,  # using False to show unsupported framework version warnings with _MLFlowLogFilter
        )

    def start_run(
        self,
        run: str,
        *,
        tensorboard: Union[tensorboard_resource.Tensorboard, str, None] = None,
        resume=False,
    ) -> experiment_run_resource.ExperimentRun:
        """Start a run to current session.

        ```py
        aiplatform.init(experiment='my-experiment')
        aiplatform.start_run('my-run')
        aiplatform.log_params({'learning_rate':0.1})
        ```

        Use as context manager. Run will be ended on context exit:
        ```py
        aiplatform.init(experiment='my-experiment')
        with aiplatform.start_run('my-run') as my_run:
            my_run.log_params({'learning_rate':0.1})
        ```

        Resume a previously started run:
        ```py
        aiplatform.init(experiment='my-experiment')
        with aiplatform.start_run('my-run', resume=True) as my_run:
            my_run.log_params({'learning_rate':0.1})
        ```


        Args:
            run(str):
                Required. Name of the run to assign current session with.
            tensorboard Union[str, tensorboard_resource.Tensorboard]:
                Optional. Backing Tensorboard Resource to enable and store time series metrics
                logged to this Experiment Run using `log_time_series_metrics`.

                If not provided will the the default backing tensorboard of the currently
                set experiment.
            resume (bool):
                Whether to resume this run. If False a new run will be created.
        Raises:
            ValueError:
                if experiment is not set. Or if run execution or metrics artifact is already created
                but with a different schema.
        """

        if not self.experiment:
            raise ValueError(
                "No experiment set for this run. Make sure to call aiplatform.init(experiment='my-experiment') "
                "before invoking start_run. "
            )

        if self.experiment_run:
            self.end_run()

        if resume:
            self._experiment_run = experiment_run_resource.ExperimentRun(
                run_name=run, experiment=self.experiment
            )
            if tensorboard:
                self._experiment_run.assign_backing_tensorboard(tensorboard=tensorboard)

            self._experiment_run.update_state(
                state=execution_v1.Execution.State.RUNNING
            )

        else:
            self._experiment_run = experiment_run_resource.ExperimentRun.create(
                run_name=run, experiment=self.experiment, tensorboard=tensorboard
            )

        return self._experiment_run

    def end_run(
        self,
        state: execution_v1.Execution.State = execution_v1.Execution.State.COMPLETE,
    ):
        """Ends the the current experiment run.

        ```py
        aiplatform.start_run('my-run')
        ...
        aiplatform.end_run()
        ```

        """
        self._validate_experiment_and_run(method_name="end_run")
        try:
            self.experiment_run.end_run(state=state)
        except exceptions.NotFound:
            _LOGGER.warning(
                f"Experiment run {self.experiment_run.name} was not found."
                "It may have been deleted"
            )
        finally:
            self._experiment_run = None

    def autolog(self, disable=False):
        """Enables autologging of parameters and metrics to Vertex Experiments.

        After calling `aiplatform.autolog()`, any metrics and parameters from
        model training calls with supported ML frameworks will be automatically
        logged to Vertex Experiments.

        Using autologging requires setting an experiment and experiment_tensorboard.

        Args:
            disable (bool):
                Optional. Whether to disable autologging. Defaults to False.
                If set to True, this resets the MLFlow tracking URI to its
                previous state before autologging was called and remove logging
                filters.
        Raises:
            ImportError:
                If MLFlow is not installed. MLFlow is required to use
                autologging in Vertex.
            ValueError:
                If experiment or experiment_tensorboard is not set.
                If `disable` is passed and autologging hasn't been enbaled.
        """

        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "MLFlow is not installed. Please install MLFlow using pip install google-cloud-aiplatform[autologging] to use autologging in the Vertex SDK."
            )

        if disable:
            if not autologging_utils._is_autologging_enabled():
                raise ValueError(
                    "Autologging is not enabled. Enable autologging by calling aiplatform.autolog()."
                )
            if self._existing_tracking_uri:
                mlflow.set_tracking_uri(self._existing_tracking_uri)
            mlflow.autolog(disable=True)

            # Remove the log filters we applied in the plugin
            logging.getLogger("mlflow").setLevel(logging.INFO)
            logging.getLogger("mlflow.tracking.fluent").disabled = False
            logging.getLogger("mlflow.utils.autologging_utils").removeFilter(
                _MLFlowLogFilter()
            )
        elif not self.experiment:
            raise ValueError(
                "No experiment set. Make sure to call aiplatform.init(experiment='my-experiment') "
                "before calling aiplatform.autolog()."
            )
        elif not self.experiment._metadata_context.metadata.get(
            constants._BACKING_TENSORBOARD_RESOURCE_KEY
        ):
            raise ValueError(
                "Setting an experiment tensorboard is required to use autologging. "
                "Please set a backing tensorboard resource by calling "
                "aiplatform.init(experiment_tensorboard=aiplatform.Tensorboard(...))."
            )
        else:
            self._existing_tracking_uri = mlflow.get_tracking_uri()

            _ExperimentTracker._initialize_mlflow_plugin()

    def log_params(self, params: Dict[str, Union[float, int, str]]):
        """Log single or multiple parameters with specified key and value pairs.

        Parameters with the same key will be overwritten.

        ```py
        aiplatform.start_run('my-run')
        aiplatform.log_params({'learning_rate': 0.1, 'dropout_rate': 0.2})
        ```

        Args:
            params (Dict[str, Union[float, int, str]]):
                Required. Parameter key/value pairs.
        """

        self._validate_experiment_and_run(method_name="log_params")
        # query the latest run execution resource before logging.
        self.experiment_run.log_params(params=params)

    def log_metrics(self, metrics: Dict[str, Union[float, int, str]]):
        """Log single or multiple Metrics with specified key and value pairs.

        Metrics with the same key will be overwritten.

        ```py
        aiplatform.start_run('my-run', experiment='my-experiment')
        aiplatform.log_metrics({'accuracy': 0.9, 'recall': 0.8})
        ```

        Args:
            metrics (Dict[str, Union[float, int, str]]):
                Required. Metrics key/value pairs.
        """

        self._validate_experiment_and_run(method_name="log_metrics")
        # query the latest metrics artifact resource before logging.
        self.experiment_run.log_metrics(metrics=metrics)

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
        """Create an artifact for classification metrics and log to ExperimentRun. Currently support confusion matrix and ROC curve.

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

        Raises:
            ValueError: if 'labels' and 'matrix' are not set together
                        or if 'labels' and 'matrix' are not in the same length
                        or if 'fpr' and 'tpr' and 'threshold' are not set together
                        or if 'fpr' and 'tpr' and 'threshold' are not in the same length
        """

        self._validate_experiment_and_run(method_name="log_classification_metrics")
        # query the latest metrics artifact resource before logging.
        return self.experiment_run.log_classification_metrics(
            display_name=display_name,
            labels=labels,
            matrix=matrix,
            fpr=fpr,
            tpr=tpr,
            threshold=threshold,
        )

    def log_model(
        self,
        model: Union[
            "sklearn.base.BaseEstimator", "xgb.Booster", "tf.Module"  # noqa: F821
        ],
        artifact_id: Optional[str] = None,
        *,
        uri: Optional[str] = None,
        input_example: Union[
            list, dict, "pd.DataFrame", "np.ndarray"  # noqa: F821
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
        ```py
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
        ```

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
        self._validate_experiment_and_run(method_name="log_model")
        self.experiment_run.log_model(
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

    def _validate_experiment_and_run(self, method_name: str):
        """Validates Experiment and Run are set and raises informative error message.

        Args:
            method_name: The name of th method to raise from.

        Raises:
            ValueError: If Experiment or Run are not set.
        """

        if not self.experiment:
            raise ValueError(
                f"No experiment set. Make sure to call aiplatform.init(experiment='my-experiment') "
                f"before trying to {method_name}. "
            )
        if not self.experiment_run:
            raise ValueError(
                f"No run set. Make sure to call aiplatform.start_run('my-run') before trying to {method_name}. "
            )

    def get_experiment_df(
        self, experiment: Optional[str] = None
    ) -> "pd.DataFrame":  # noqa: F821
        """Returns a Pandas DataFrame of the parameters and metrics associated with one experiment.

        Example:

        ```py
        aiplatform.init(experiment='exp-1')
        aiplatform.start_run(run='run-1')
        aiplatform.log_params({'learning_rate': 0.1})
        aiplatform.log_metrics({'accuracy': 0.9})

        aiplatform.start_run(run='run-2')
        aiplatform.log_params({'learning_rate': 0.2})
        aiplatform.log_metrics({'accuracy': 0.95})

        aiplatform.get_experiment_df()
        ```

        Will result in the following DataFrame:

        ```
        experiment_name | run_name      | param.learning_rate | metric.accuracy
        exp-1           | run-1         | 0.1                 | 0.9
        exp-1           | run-2         | 0.2                 | 0.95
        ```

        Args:
            experiment (str):
                Name of the Experiment to filter results. If not set, return results
                of current active experiment.

        Returns:
            Pandas Dataframe of Experiment with metrics and parameters.

        Raise:
            NotFound exception if experiment does not exist.
            ValueError if given experiment is not associated with a wrong schema.
        """

        if not experiment:
            experiment = self.experiment
        else:
            experiment = experiment_resources.Experiment(experiment)

        return experiment.get_data_frame()

    def log(
        self,
        *,
        pipeline_job: Optional[pipeline_jobs.PipelineJob] = None,
    ):
        """Log Vertex AI Resources to the current experiment run.

        ```py
        aiplatform.start_run('my-run')
        my_job = aiplatform.PipelineJob(...)
        my_job.submit()
        aiplatform.log(my_job)
        ```

        Args:
            pipeline_job (pipeline_jobs.PipelineJob):
                Optional. Vertex PipelineJob to associate to this Experiment Run.
        """
        self._validate_experiment_and_run(method_name="log")
        self.experiment_run.log(pipeline_job=pipeline_job)

    def log_time_series_metrics(
        self,
        metrics: Dict[str, Union[float]],
        step: Optional[int] = None,
        wall_time: Optional[timestamp_pb2.Timestamp] = None,
    ):
        """Logs time series metrics to to this Experiment Run.

        Requires the experiment or experiment run has a backing Vertex Tensorboard resource.

        ```py
        my_tensorboard = aiplatform.Tensorboard(...)
        aiplatform.init(experiment='my-experiment', experiment_tensorboard=my_tensorboard)
        aiplatform.start_run('my-run')

        # increments steps as logged
        for i in range(10):
            aiplatform.log_time_series_metrics({'loss': loss})

        # explicitly log steps
        for i in range(10):
            aiplatform.log_time_series_metrics({'loss': loss}, step=i)
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
        self._validate_experiment_and_run(method_name="log_time_series_metrics")
        self.experiment_run.log_time_series_metrics(
            metrics=metrics, step=step, wall_time=wall_time
        )

    def start_execution(
        self,
        *,
        schema_title: Optional[str] = None,
        display_name: Optional[str] = None,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        schema_version: Optional[str] = None,
        description: Optional[str] = None,
        resume: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[auth_credentials.Credentials] = None,
    ) -> execution.Execution:
        """
        Create and starts a new Metadata Execution or resumes a previously created Execution.

        To start a new execution:

        ```py
        with aiplatform.start_execution(schema_title='system.ContainerExecution', display_name='trainer) as exc:
          exc.assign_input_artifacts([my_artifact])
          model = aiplatform.Artifact.create(uri='gs://my-uri', schema_title='system.Model')
          exc.assign_output_artifacts([model])
        ```

        To continue a previously created execution:
        ```py
        with aiplatform.start_execution(resource_id='my-exc', resume=True) as exc:
            ...
        ```
        Args:
            schema_title (str):
                Optional. schema_title identifies the schema title used by the Execution. Required if starting
                a new Execution.
            resource_id (str):
                Optional. The <resource_id> portion of the Execution name with
                the format. This is globally unique in a metadataStore:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/executions/<resource_id>.
            display_name (str):
                Optional. The user-defined name of the Execution.
            schema_version (str):
                Optional. schema_version specifies the version used by the Execution.
                If not set, defaults to use the latest version.
            metadata (Dict):
                Optional. Contains the metadata information that will be stored in the Execution.
            description (str):
                Optional. Describes the purpose of the Execution to be created.
            metadata_store_id (str):
                Optional. The <metadata_store_id> portion of the resource name with
                the format:
                projects/123/locations/us-central1/metadataStores/<metadata_store_id>/artifacts/<resource_id>
                If not provided, the MetadataStore's ID will be set to "default".
            project (str):
                Optional. Project used to create this Execution. Overrides project set in
                aiplatform.init.
            location (str):
                Optional. Location used to create this Execution. Overrides location set in
                aiplatform.init.
            credentials (auth_credentials.Credentials):
                Optional. Custom credentials used to create this Execution. Overrides
                credentials set in aiplatform.init.

        Returns:
            Execution: Instantiated representation of the managed Metadata Execution.

        Raises:
            ValueError: If experiment run is set and project or location do not match experiment run.
            ValueError: If resume set to `True` and resource_id is not provided.
            ValueError: If creating a new executin and schema_title is not provided.
        """

        if self.experiment_run and not self.experiment_run._is_legacy_experiment_run():
            if project and project != self.experiment_run.project:
                raise ValueError(
                    f"Currently set Experiment run project {self.experiment_run.project} must"
                    f"match provided project {project}"
                )
            if location and location != self.experiment_run.location:
                raise ValueError(
                    f"Currently set Experiment run location {self.experiment_run.location} must"
                    f"match provided location {project}"
                )

        if resume:
            if not resource_id:
                raise ValueError("resource_id is required when resume=True")

            run_execution = execution.Execution(
                execution_name=resource_id,
                project=project,
                location=location,
                credentials=credentials,
            )

            # TODO(handle updates if resuming)

            run_execution.update(state=gca_execution.Execution.State.RUNNING)
        else:
            if not schema_title:
                raise ValueError(
                    "schema_title must be provided when starting a new Execution"
                )

            run_execution = execution.Execution.create(
                display_name=display_name,
                schema_title=schema_title,
                schema_version=schema_version,
                metadata=metadata,
                description=description,
                resource_id=resource_id,
                project=project,
                location=location,
                credentials=credentials,
            )

        if self.experiment_run:
            if self.experiment_run._is_legacy_experiment_run():
                _LOGGER.warning(
                    f"{self.experiment_run._run_name} is an Experiment run created in Vertex Experiment Preview",
                    " and does not support tracking Executions."
                    " Please create a new Experiment run to track executions against an Experiment run.",
                )
            else:
                self.experiment_run.associate_execution(run_execution)
                run_execution.assign_input_artifacts = (
                    self.experiment_run._association_wrapper(
                        run_execution.assign_input_artifacts
                    )
                )
                run_execution.assign_output_artifacts = (
                    self.experiment_run._association_wrapper(
                        run_execution.assign_output_artifacts
                    )
                )

        return run_execution


_experiment_tracker = _ExperimentTracker()
