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

import concurrent
import functools
import inspect
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

from google.cloud.aiplatform import base
from google.cloud.aiplatform_v1.services.vizier_service import (
    VizierServiceClient,
)
from google.cloud.aiplatform_v1.types import study as gca_study
import vertexai
from vertexai.preview._workflow.driver import remote
from vertexai.preview._workflow.driver import (
    VertexRemoteFunctor,
)
from vertexai.preview._workflow.executor import (
    remote_container_training,
)
from vertexai.preview._workflow.executor import (
    training,
)
from vertexai.preview._workflow.shared import configs
from vertexai.preview._workflow.shared import (
    supported_frameworks,
)


try:
    import pandas as pd

    PandasData = pd.DataFrame

except ImportError:
    PandasData = Any

_LOGGER = base.Logger(__name__)

# Metric id constants
_CUSTOM_METRIC_ID = "custom"
_ROC_AUC_METRIC_ID = "roc_auc"
_F1_METRIC_ID = "f1"
_PRECISION_METRIC_ID = "precision"
_RECALL_METRIC_ID = "recall"
_ACCURACY_METRIC_ID = "accuracy"
_MAE_METRIC_ID = "mae"
_MAPE_METRIC_ID = "mape"
_R2_METRIC_ID = "r2"
_RMSE_METRIC_ID = "rmse"
_RMSLE_METRIC_ID = "rmsle"
_MSE_METRIC_ID = "mse"

_SUPPORTED_METRIC_IDS = frozenset(
    [
        _CUSTOM_METRIC_ID,
        _ROC_AUC_METRIC_ID,
        _F1_METRIC_ID,
        _PRECISION_METRIC_ID,
        _RECALL_METRIC_ID,
        _ACCURACY_METRIC_ID,
        _MAE_METRIC_ID,
        _MAPE_METRIC_ID,
        _R2_METRIC_ID,
        _RMSE_METRIC_ID,
        _RMSLE_METRIC_ID,
        _MSE_METRIC_ID,
    ]
)
_SUPPORTED_CLASSIFICATION_METRIC_IDS = frozenset(
    [
        _ROC_AUC_METRIC_ID,
        _F1_METRIC_ID,
        _PRECISION_METRIC_ID,
        _RECALL_METRIC_ID,
        _ACCURACY_METRIC_ID,
    ]
)


# Vizier client constnats
_STUDY_NAME_PREFIX = "vizier_hyperparameter_tuner_study"
_CLIENT_ID = "client"

# Train and test split constants
_DEFAULT_TEST_FRACTION = 0.25

# Parameter constants
_TRAINING_X_PARAMS = ["X", "x", "X_train", "x_train"]
_TRAINING_DATA_PARAMS = ["X", "x", "X_train", "x_train", "training_data"]
_OSS_TRAINING_DATA_PARAMS = ["X", "x"]
_TRAINING_TARGET_VALUE_PARAMS = ["y", "y_train"]
_Y_DATA_PARAM = "y"
_X_TEST_PARAMS = ["X_test", "x_test"]
_Y_TEST = "y_test"
_VALIDATION_DATA = "validation_data"


class VizierHyperparameterTuner:
    """The Vizier hyperparameter tuner for local and remote tuning."""

    def __init__(
        self,
        get_model_func: Callable[..., Any],
        max_trial_count: int,
        parallel_trial_count: int,
        hparam_space: List[Dict[str, Any]],
        metric_id: str = _ACCURACY_METRIC_ID,
        metric_goal: str = "MAXIMIZE",
        max_failed_trial_count: int = 0,
        search_algorithm: str = "ALGORITHM_UNSPECIFIED",
        project: Optional[str] = None,
        location: Optional[str] = None,
        study_display_name_prefix: str = _STUDY_NAME_PREFIX,
    ):
        """Initializes a VizierHyperparameterTuner instance.

        VizierHyperparameterTuner provides support for local and remote Vizier
        hyperparameter tuning. For information on Vertex AI Vizier, refer to
        https://cloud.google.com/vertex-ai/docs/vizier/overview.

        Args:
            get_model_func (Callable[..., Any]):
                Required. A function that returns a model to be tuned. Non-tunable
                parameters should be preset by get_model_func, and tunable
                parameters will be set byVizierHyperparameterTuner.

                Example:
                # parameter_a and parameter_b are tunable.
                def get_model_func(parameter_a, parameter_b):
                    # parameter_c is non-tunable
                    parameter_c = 10
                    return ExampleModel(parameter_a, parameter_b, parameter_c)

                For lightning models, get_model_func should return a dictionary
                containing the following keys: 'model', 'trainer',
                'train_dataloaders'; each representing the lightning model, the
                trainer and the training dataloader(s) respectively.

            max_trial_count (int):
                Required. The desired total number of trials.
            parallel_trial_count (int):
                Required. The desired number of trials to run in parallel. For
                pytorch lightning, currently we only support parallel_trial_count=1.
            hparam_space (List[Dict[str, Any]]):
                Required. A list of parameter specs each representing a single
                tunable parameter. For parameter specs, refer to
                https://cloud.google.com/vertex-ai/docs/reference/rest/v1/StudySpec#parameterspec
            metric_id (str):
                Optional. The ID of the metric. Must be one of 'roc_auc', 'f1',
                'precision', 'recall', 'accuracy', 'mae', 'mape', 'r2', 'rmse',
                'rmsle', 'mse' or 'custom'. Only 'accuracy' supports multi-class
                classification. Set to 'custom' to use a custom score function.
                Default is 'accuracy'.
            metric_goal (str):
                Optional. The optimization goal of the metric. Must be one of
                'GOAL_TYPE_UNSPECIFIED', 'MAXIMIZE' and 'MINIMIZE'.
                'GOAL_TYPE_UNSPECIFIED' defaults to maximize. Default is
                'MAXIMIZE'. Refer to
                https://cloud.google.com/vertex-ai/docs/reference/rest/v1/StudySpec#goaltype
                for details on goal types.
            max_failed_trial_count (int):
                Optional. The number of failed trials that need to be seen before
                failing the tuning process. If 0, the tuning process only fails
                when all trials have failed. Default is 0.
            search_algorithm (str):
                Optional. The search algorithm specified for the study. Must be
                one of 'ALGORITHM_UNSPECIFIED', 'GRID_SEARCH' and 'RANDOM_SEARCH'.
                Default is 'ALGORITHM_UNSPECIFIED'. Refer to
                https://cloud.google.com/vertex-ai/docs/reference/rest/v1/StudySpec#algorithm
                for details on the study algorithms.
            project (str):
                Optional. Project for the study. If not set, project set in
                vertexai.init will be used.
            location (str):
                Optional. Location for the study. If not set, location set in
                vertexai.init will be used.
            study_display_name_prefix (str):
                Optional. Prefix of the study display name. Default is
                'vizier-hyperparameter-tuner-study'.
        """
        self.get_model_func = get_model_func
        self.max_trial_count = max_trial_count
        self.parallel_trial_count = parallel_trial_count
        self.hparam_space = hparam_space

        if metric_id not in _SUPPORTED_METRIC_IDS:
            raise ValueError(
                f"Unsupported metric_id {metric_id}. Supported metric_ids: {_SUPPORTED_METRIC_IDS}"
            )
        self.metric_id = metric_id

        self.metric_goal = metric_goal
        self.max_failed_trial_count = max_failed_trial_count
        self.search_algorithm = search_algorithm

        # Initializes Vertex config
        self.vertex = configs.VertexConfig()

        # Creates Vizier client, study and trials
        project = project or vertexai.preview.global_config.project
        location = location or vertexai.preview.global_config.location
        self.vizier_client, self.study = self._create_study(
            project, location, study_display_name_prefix
        )

        # self.models should be a mapping from trial names to trained models.
        self.models = {}

    def _create_study(
        self,
        project: str,
        location: str,
        study_display_name_prefix: str = _STUDY_NAME_PREFIX,
    ) -> Tuple[VizierServiceClient, gca_study.Study]:
        """Creates a Vizier study config.

        Args:
            project (str):
                Project for the study.
            location (str):
                Location for the study.
            study_display_name_prefix (str):
                Prefix for the study display name. Default is
                'vizier-hyperparameter-tuner-study'.
        Returns:
            A Vizier client and the created study.
        """
        vizier_client = VizierServiceClient(
            client_options=dict(api_endpoint=f"{location}-aiplatform.googleapis.com")
        )
        study_config = {
            "display_name": f"{study_display_name_prefix}_{uuid.uuid4()}".replace(
                "-", "_"
            ),
            "study_spec": {
                "algorithm": self.search_algorithm,
                "parameters": self.hparam_space,
                "metrics": [{"metric_id": self.metric_id, "goal": self.metric_goal}],
            },
        }
        parent = f"projects/{project}/locations/{location}"
        study = vizier_client.create_study(parent=parent, study=study_config)
        return vizier_client, study

    def _suggest_trials(self, num_trials: int) -> List[gca_study.Trial]:
        """Suggests trials using the Vizier client.

        During each round of tuning, num_trials number of trials will
        be suggested. For each trial, training will be performed locally or
        remotely. After training finishes, we use the trained model to measure
        the metrics and report the metrics to the trial before marking it as
        completed. At the next round of tuning, another parallel_trial_count
        of trials will be suggested based on previous measurements.

        Args:
            num_trials (int): Required. Number of trials to suggest.
        Returns:
            A list of suggested trials.
        """
        return (
            self.vizier_client.suggest_trials(
                {
                    "parent": self.study.name,
                    "suggestion_count": num_trials,
                    "client_id": _CLIENT_ID,
                }
            )
            .result()
            .trials
        )

    def get_best_models(self, num_models: int = 1) -> List[Any]:
        """Gets the best models from completed trials.

        Args:
            num_models (int):
                Optional. The number of best models to return. Default is 1.

        Returns:
            A list of best models.
        """
        trials = []
        for trial in self.vizier_client.list_trials({"parent": self.study.name}).trials:
            if (
                trial.state == gca_study.Trial.State.SUCCEEDED
                and trial.name in self.models
            ):
                trials.append((trial.name, trial.final_measurement.metrics[0].value))

        maximize = True if self.metric_goal == "MAXIMIZE" else False
        trials.sort(reverse=maximize, key=lambda x: x[1])

        return [self.models[trial[0]] for trial in trials[:num_models]]

    def _create_train_and_test_splits(
        self,
        x: PandasData,
        y: Union[PandasData, str],
        test_fraction: float = _DEFAULT_TEST_FRACTION,
    ) -> Tuple[PandasData, PandasData, Optional[PandasData], PandasData]:
        """Creates train and test splits if no manual test splits provided.

        Depending on the model to be tuned, the training step may take in either
        one or two DataFrames for training data and target values.

        1. Two pandas DataFrames:
            - One contains training data and the other contains target values.
            - Four DataFrames will be returned, ie. X_train, X_test, y_train,
            y_test.
        2. One pandas DataFrame:
            - Contains both training data and target values.
            - Only three DataFrames will be returned, ie. X_train, X_test,
            y_test. X_train contains both training data and target values. The
            testing splits need to be separated into data and values to make
            predictions.

        Args:
            x (pandas.DataFrame):
                Required. A pandas DataFrame for the dataset. If it contains the
                target column, y must be a string specifying the target column
                name.
            y (Union[pandas.DataFrame, str]):
                Required. A pandas DataFrame containing target values for the
                dataset or a string specifying the target column name.
            test_fraction (float):
                Optional. The proportion of the dataset to include in the test
                split. eg. test_fraction=0.25 for a pandas Dataframe with 100
                rows would result in 75 rows for training and 25 rows for
                testing. Default is 0.25.
        Returns:
            A tuple containing training data, testing data, training target
            values, testing target values. Training target values may be None if
            training data contrains training target.
        """
        if test_fraction <= 0 or test_fraction >= 1:
            raise ValueError(
                "test_fraction must be greater than 0 and less than 1 but was "
                f"{test_fraction}."
            )
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError(
                "scikit-learn must be installed to create train and test splits. "
                "Please call `pip install scikit-learn>=0.24`"
            ) from None

        if isinstance(y, str):
            try:
                import pandas as pd
            except ImportError:
                raise ImportError(
                    "pandas must be installed to create train and test splits "
                    "with a target column name."
                ) from None
            x_train, x_test = train_test_split(x, test_size=test_fraction)
            y_test = pd.DataFrame(x_test.pop(y))
            return x_train, x_test, None, y_test
        else:
            return train_test_split(x, y, test_size=test_fraction)

    def _evaluate_model(
        self, model: Any, x_test: PandasData, y_test: PandasData
    ) -> Tuple[Any, float]:
        """Evaluates a model.

        Metrics are calculated based on the metric_id set by the user. After
        reporting the metrics, mark the trial as complete. Only completed trials
        can be listed as optimal trials.

        Supported metric_id: 'roc_auc', 'f1', 'precision', 'recall', 'accuracy',
        'mae', 'mape', 'r2', 'rmse', 'rmsle', 'mse' or 'custom'. Only 'accuracy'
        supports multi-class classification.

        When metric_id is 'custom', the model must provide a score() function to
        provide a metric value. Otherwise, the model must provide a predict()
        function that returns array-like prediction results.

        e.g.
        class ExampleModel:
            def score(x_test, y_test):
                # Code to make predictions and calculate metrics
                return custom_metric(y_true=y_test, y_pred=self.predict(x_test))

        Args:
            model (Any):
                Required. The model trained during the trial.
            x_test (pandas.DataFrame):
                Required. The testing data.
            y_test (pandas.DataFrame):
                Required. The testing values.
        Returns:
            A tuple containing the model and the corresponding metric value.
        """
        try:  # Only used by local tuning loop
            import sklearn.metrics

            _SUPPORTED_METRIC_FUNCTIONS = {
                _ROC_AUC_METRIC_ID: sklearn.metrics.roc_auc_score,
                _F1_METRIC_ID: sklearn.metrics.f1_score,
                _PRECISION_METRIC_ID: sklearn.metrics.precision_score,
                _RECALL_METRIC_ID: sklearn.metrics.recall_score,
                _ACCURACY_METRIC_ID: sklearn.metrics.accuracy_score,
                _MAE_METRIC_ID: sklearn.metrics.mean_absolute_error,
                _MAPE_METRIC_ID: sklearn.metrics.mean_absolute_percentage_error,
                _R2_METRIC_ID: sklearn.metrics.r2_score,
                _RMSE_METRIC_ID: functools.partial(
                    sklearn.metrics.mean_squared_error, squared=False
                ),
                _RMSLE_METRIC_ID: functools.partial(
                    sklearn.metrics.mean_squared_log_error, squared=False
                ),
                _MSE_METRIC_ID: sklearn.metrics.mean_squared_error,
            }
        except Exception as e:
            raise ImportError(
                "scikit-learn must be installed to evaluate models. "
                "Please call `pip install scikit-learn>=0.24`"
            ) from e

        if self.metric_id == _CUSTOM_METRIC_ID:
            metric_value = model.score(x_test, y_test)
        else:
            if self.metric_id in _SUPPORTED_METRIC_IDS:
                predictions = model.predict(x_test)
                # Keras outputs probabilities. Must convert to output label.
                if (
                    supported_frameworks._is_keras(model)
                    and self.metric_id in _SUPPORTED_CLASSIFICATION_METRIC_IDS
                ):
                    if isinstance(predictions, pd.DataFrame):
                        predictions = predictions.to_numpy()
                    predictions = (
                        predictions.argmax(axis=-1)
                        if predictions.shape[-1] > 1
                        else (predictions > 0.5).astype("int32")
                    )
                metric_value = _SUPPORTED_METRIC_FUNCTIONS[self.metric_id](
                    y_test, predictions
                )
            else:
                raise ValueError(
                    f"Unsupported metric_id {self.metric_id}. Supported metric_ids: {_SUPPORTED_METRIC_IDS}"
                )
        return (model, metric_value)

    def _add_model_and_report_trial_metrics(
        self, trial_name: str, trial_output: Optional[Tuple[Any, float]]
    ) -> None:
        """Adds a model to the dictionary of trained models and report metrics.

        If trial_output is None, it means that the trial has failed and should
        be marked as infeasible.

        Args:
            trial_name (str):
                Required. The trial name.
            trial_output (Optional[Tuple[Any, float]]):
                Required. A tuple containing the model and the metric value, or
                None if the trial has failed.
        """
        if trial_output is not None:
            model, metric_value = trial_output
            self.vizier_client.complete_trial(
                {
                    "name": trial_name,
                    "final_measurement": {
                        "metrics": [
                            {"metric_id": self.metric_id, "value": metric_value}
                        ]
                    },
                }
            )
            self.models[trial_name] = model
        else:
            self.vizier_client.complete_trial(
                {"name": trial_name, "trial_infeasible": True}
            )

    def _get_model_param_type_mapping(self):
        """Gets a mapping from parameter_id to its type.

        Returns:
            A mapping from parameter id to its type.
        """
        model_param_type_mapping = {}
        for param in self.hparam_space:
            param_id = param["parameter_id"]
            if "double_value_spec" in param:
                param_type = float
            elif "integer_value_spec" in param:
                param_type = int
            elif "categorical_value_spec" in param:
                param_type = str
            elif "discrete_value_spec" in param:
                param_type = type(param["discrete_value_spec"]["values"][0])
            else:
                raise ValueError(
                    f"Invalid hparam_space configuration for parameter {param_id}"
                )
            model_param_type_mapping[param_id] = param_type

        return model_param_type_mapping

    def _set_model_parameters(
        self,
        trial: gca_study.Trial,
        fixed_init_params: Optional[Dict[Any, Any]] = None,
        fixed_runtime_params: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[Any, Dict[Any, Any]]:
        """Returns a model intialized with trial parameters and a dictionary of runtime parameters.

        Initialization parameters are passed to the get_model_func. Runtime parameters
        will be passed to the model's fit() or @developer.mark.train()-decorated
        method outside of this function.

        Args:
            trial (gca_study.Trial): Required. A trial suggested by Vizier.
            fixed_init_params (Dict[Any, Any]): Optional. A dictionary of fixed
                parameters to be passed to get_model_func.
            fixed_runtime_params (Dict[Any, Any]): Optional. A dictionary of fixed
                runtime parameters.

        Returns:
            A model initialized using parameters from the specified trial and
            a dictionary of runtime parameters.
        """
        model_init_params = {}
        model_runtime_params = {}
        get_model_func_binding = inspect.signature(self.get_model_func).parameters

        model_param_type_mapping = self._get_model_param_type_mapping()

        for param in trial.parameters:
            param_id = param.parameter_id
            param_value = (
                model_param_type_mapping[param_id](param.value)
                if param_id in model_param_type_mapping
                else param.value
            )
            if param_id in get_model_func_binding:
                model_init_params[param_id] = param_value
            else:
                model_runtime_params[param_id] = param_value

        if fixed_init_params:
            model_init_params.update(fixed_init_params)
        if fixed_runtime_params:
            model_runtime_params.update(fixed_runtime_params)

        return self.get_model_func(**model_init_params), model_runtime_params

    def _is_remote(self, train_method: VertexRemoteFunctor) -> bool:
        """Checks if a train method will be executed locally or remotely.

        The train method will be executed remotely if:
            - The train method's vertex config sets remote to True (eg.
              train.vertex.remote=True)
            - Or, .vertex.remote is not set but the global config defaults
              remote to True. (eg. vertexai.preview.init(remote=True, ...))

        Otherwise, the train method will be executed locally.

        Args:
            train_method (VertexRemoteFunctor):
                Required. The train method.
        Returns:
            Whether the train method will be executed locally or remotely.
        """
        return train_method.vertex.remote or (
            train_method.vertex.remote is None and vertexai.preview.global_config.remote
        )

    def _override_staging_bucket(
        self, train_method: VertexRemoteFunctor, trial_name: str
    ) -> None:
        """Overrides the staging bucket for a train method.

        A staging bucket must be specified by:
            - The train method's training config.
              eg. train.vertex.remote_config.staging_bucket = ...
            - Or, .vertex.remote_config.staging_bucket is not set, but a
              default staging bucket is specified in the global config.
              eg. vertexai.init(staging_bucket=...)

        The staging bucket for each trial is overriden so that each trial uses
        its own directory.

        Args:
            train_method (VertexRemoteFunctor):
                Required. The train method.
            trial_name (str): Required. The trial name.
        Raises:
            ValueError if no staging bucket specified and no default staging
            bucket set.
        """
        staging_bucket = (
            train_method.vertex.remote_config.staging_bucket
            or vertexai.preview.global_config.staging_bucket
        )
        if not staging_bucket:
            raise ValueError(
                "No default staging bucket set. "
                "Please call `vertexai.init(staging_bucket='gs://my-bucket')."
            )
        train_method.vertex.remote_config.staging_bucket = os.path.join(
            staging_bucket,
            "-".join(trial_name.split("/")[:-1]),
            trial_name.split("/")[-1],
        )

    def _get_vertex_model_train_method_and_params(
        self,
        model: remote.VertexModel,
        x_train: PandasData,
        y_train: Optional[PandasData],
        x_test: PandasData,
        y_test: PandasData,
        trial_name: str,
    ) -> Tuple[VertexRemoteFunctor, Dict[str, Any]]:
        """Gets the train method for a VertexModel model and data parameters.

        Supported parameter names:
        - Training data: ['X', 'X_train', 'x', 'x_train', 'training_data'].
        - Training target values: ['y', 'y_train']. If not provided, training
        data should contain target values.
        - Testing data: ['X_test', 'x_test', 'validation_data'].
        - Testing target values: ['y_test']. If not provided, testing data
        should contain target values.

        If remote mode is turned on, overrides the training staging bucket for
        each trial.

        Args:
            model (remote.VertexModel):
                Required. An instance of VertexModel.
            x_train (pandas.DataFrame):
                Required. Training data.
            y_train (Optional[pandas.DataFrame]):
                Required. Training target values. If None, x_train should
                include training target values.
            x_test (pandas.DataFrame):
                Required. Testing data.
            y_test (pandas.DataFrame):
                Required. Testing target values.
            trial_name (str):
                Required. The trial name.
        Returns:
            The train method for the Vertex model and data params.
        Raises:
            ValueError if there is no remote executable train method.
        """
        data_params = {}
        for _, attr_value in inspect.getmembers(model):
            if isinstance(attr_value, VertexRemoteFunctor) and (
                attr_value._remote_executor == training.remote_training
                or attr_value._remote_executor == remote_container_training.train
            ):
                params = inspect.signature(attr_value).parameters
                for param in params:
                    if param in _TRAINING_DATA_PARAMS:
                        data_params[param] = x_train
                    elif param in _TRAINING_TARGET_VALUE_PARAMS:
                        data_params[param] = y_train
                    elif param in _X_TEST_PARAMS:
                        data_params[param] = x_test
                    elif param == _Y_TEST:
                        data_params[_Y_TEST] = y_test
                    elif param == _VALIDATION_DATA:
                        data_params[_VALIDATION_DATA] = pd.concat(
                            [x_test, y_test], axis=1
                        )
                if self._is_remote(attr_value):
                    self._override_staging_bucket(attr_value, trial_name)
                return (attr_value, data_params)
        raise ValueError("No remote executable train method.")

    def _get_lightning_train_method_and_params(
        self,
        model: Dict[str, Any],
        trial_name: str,
    ):
        """Gets the train method and parameters for a Lightning model.

        Given the lightning model, the trainer and the training dataloader(s),
        returns trainer.fit and the parameters containing the model and the
        training dataloader(s). If the trainer is enabled to run remotely and
        remote mode is turned on, overrides the training staging bucket for
        each trial.

        Training data and target values have already been passed into the
        training dataloader(s), so no additional runtime parameters need to be
        set.

        Args:
            model (Dict[str, Any]):
                Required. A dictionary containing the following keys: 'model',
                'trainer', 'train_dataloaders'; each representing the lightning
                model, the trainer and the training dataloader(s) respectively.
            trial_name (str):
                Required. The trial name.
        Returns:
            The train method and its parameters for the lightning model.
        """
        trainer = model["trainer"]
        if isinstance(trainer.fit, VertexRemoteFunctor) and self._is_remote(
            trainer.fit
        ):
            self._override_staging_bucket(trainer.fit, trial_name)
        return trainer.fit, {
            "model": model["model"],
            "train_dataloaders": model["train_dataloaders"],
        }

    def _run_trial(
        self,
        x_train: PandasData,
        y_train: Optional[PandasData],
        x_test: PandasData,
        y_test: PandasData,
        trial: gca_study.Trial,
        fixed_init_params: Optional[Dict[Any, Any]] = None,
        fixed_runtime_params: Optional[Dict[Any, Any]] = None,
    ) -> Optional[Tuple[Any, float]]:
        """Runs a trial.

        This function sets model parameters and train method parameters,
        launches either local or remote training, and evaluates the model. With
        parallel tuning, this function can be the target function that would be
        executed in parallel.

        Args:
            x_train (pandas.DataFrame):
                Required. Training data.
            y_train (Optional[pandas.DataFrame]):
                Required. Training target values. If None, x_train should
                include training target values.
            x_test (pandas.DataFrame):
                Required. Testing data.
            y_test (pandas.DataFrame):
                Required. Testing target values.
            trial (gca_study.Trial): Required. A trial suggested by Vizier.
            fixed_init_params (Dict[Any, Any]): Optional. A dictionary of fixed
                parameters to be passed to get_model_func.
            fixed_runtime_params (Dict[Any, Any]): Optional. A dictionary of
                fixed runtime parameters.
        Returns:
            If the trial is feasible, returns a tuple of the trained model and
            its corresponding metric value. If the trial is infeasible, returns
            None.
        """
        model, model_runtime_params = self._set_model_parameters(
            trial, fixed_init_params, fixed_runtime_params
        )

        if isinstance(model, remote.VertexModel):
            train_method, params = self._get_vertex_model_train_method_and_params(
                model,
                x_train,
                y_train,
                x_test,
                y_test,
                trial.name,
            )
        elif isinstance(model, dict):
            train_method, params = self._get_lightning_train_method_and_params(
                model,
                trial.name,
            )
        elif supported_frameworks._is_keras(model):
            train_method, params = self._get_train_method_and_params(
                model, x_train, y_train, trial.name, params=["x", "y"]
            )
        elif supported_frameworks._is_sklearn(model):
            train_method, params = self._get_train_method_and_params(
                model, x_train, y_train, trial.name, params=["X", "y"]
            )
        else:
            raise ValueError(f"Unsupported model type {type(model)}")

        model_runtime_params.update(params)

        try:
            train_method(**model_runtime_params)
        except Exception as e:
            _LOGGER.warning(f"Trial {trial.name} failed: {e}.")
            return None

        if isinstance(model, dict):
            # For lightning, evaluate the model and keep track of the dictionary
            # containing the model, the trainer, and the training dataloader(s).
            _, metric_value = self._evaluate_model(model["model"], x_test, y_test)
            return model, metric_value

        return self._evaluate_model(model, x_test, y_test)

    def _get_train_method_and_params(
        self,
        model: Any,
        x_train: PandasData,
        y_train: Optional[PandasData],
        trial_name: str,
        params: List[str],
    ) -> Tuple[VertexRemoteFunctor, Dict[str, Any]]:
        """Gets the train method for an Sklearn or Keras model and data parameters.

        Args:
            model (Any):
                Required. An instance of an Sklearn or Keras model.
            x_train (pandas.DataFrame):
                Required. Training data.
            y_train (Optional[pandas.DataFrame]):
                Required. Training target values.
            trial_name (str):
                Required. The trial name.
            params (str):
                Required. The list of data parameters.
        Returns:
            The train method for the model and data params.
        Raises:
            ValueError if there is no remote executable train method.
        """
        data_params = {}
        if isinstance(model.fit, VertexRemoteFunctor) and self._is_remote(model.fit):
            self._override_staging_bucket(model.fit, trial_name)
        attr_params = inspect.signature(model.fit).parameters
        for param in params:
            if param not in attr_params:
                raise ValueError(f"Invalid data parameter {param}.")
            if param in _OSS_TRAINING_DATA_PARAMS:
                data_params[param] = x_train
            elif param == _Y_DATA_PARAM:
                data_params[param] = y_train
        return (model.fit, data_params)

    def fit(
        self,
        x: PandasData,
        y: Union[PandasData, str],
        x_test: Optional[PandasData] = None,
        y_test: Optional[PandasData] = None,
        test_fraction: Optional[float] = _DEFAULT_TEST_FRACTION,
        **kwargs,
    ):
        """Runs Vizier-backed hyperparameter tuning for a model.

        Extra runtime arguments will be forwarded to a model's fit() or
        @vertexai.preview.developer.mark.train()-decorated method.

        Example Usage:
        ```
        def get_model_func(parameter_a, parameter_b):
            # parameter_c is non-tunable
            parameter_c = 10
            return ExampleModel(parameter_a, parameter_b, parameter_c)

        x, y = pd.DataFrame(...), pd.DataFrame(...)
        tuner = VizierHyperparameterTuner(get_model_func, ...)
        # num_epochs will be passed to ExampleModel.fit()
        # (ex: ExampleModel.fit(x, y, num_epochs=5))
        tuner.fit(x, y, num_epochs=5)
        ```

        Args:
            x (pandas.DataFrame):
                Required. A pandas DataFrame for the dataset. If it contains the
                target column, y must be a string specifying the target column
                name.
            y (Union[pandas.DataFrame, str]):
                Required. A pandas DataFrame containing target values for the
                dataset or a string specifying the target column name.
            x_test (pandas.DataFrame):
                Optional. A pandas DataFrame for the test dataset. If not provided,
                X will be split into X_train and X_test based on test_fraction.
            y_test (pandas.DataFrame):
                Optional. A pandas DataFrame containing target values for the test
                dataset. If not provided, y will be split into y_train and t_test
                based on test_fraction.
            test_fraction (float):
                Optional. The proportion of the dataset to include in the test
                split. eg. test_fraction=0.25 for a pandas Dataframe with 100
                rows would result in 75 rows for training and 25 rows for
                testing. Default is 0.25.
            **kwargs (Any):
                Optional. Keyword arguments to pass to the model's fit(),
                or @vertexai.preview.developer.mark.train()-decorated method.

        Returns:
            A model initialized using parameters from the specified trial and
            a dictionary of runtime parameters.
        """
        if x_test is None or y_test is None or x_test.empty or y_test.empty:
            x, x_test, y, y_test = self._create_train_and_test_splits(
                x, y, test_fraction
            )

        # Fixed params that are passed to get_model_func.
        # Lightning, for example, requires X and y to be passed to get_model_func.
        fixed_init_params = {}
        get_model_func_binding = inspect.signature(self.get_model_func).parameters
        for x_param_name in _TRAINING_X_PARAMS:
            if x_param_name in get_model_func_binding:
                # Temporary solution for b/295191253
                # TODO(b/295191253)
                if self.parallel_trial_count > 1:
                    raise ValueError(
                        "Currently pytorch lightning only supports `parallel_trial_count = 1`. "
                        f"In {self} it was set to {self.parallel_trial_count}."
                    )
                fixed_init_params[x_param_name] = x
                break
        for y_param_name in _TRAINING_TARGET_VALUE_PARAMS:
            if y_param_name in get_model_func_binding:
                fixed_init_params[y_param_name] = y
                break

        # Disable remote job logs when running trials.
        logging.getLogger("vertexai.remote_execution").disabled = True
        try:
            num_completed_trials = 0
            num_failed_trials = 0
            while num_completed_trials < self.max_trial_count:
                num_new_trials = min(
                    (self.max_trial_count - num_completed_trials),
                    self.parallel_trial_count,
                )
                suggested_trials = self._suggest_trials(num_new_trials)
                inputs = [
                    (x, y, x_test, y_test, trial, fixed_init_params, kwargs)
                    for trial in suggested_trials
                ]
                _LOGGER.info(
                    f"Number of completed trials: {num_completed_trials}, "
                    f"Number of new trials: {num_new_trials}."
                )

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_new_trials
                ) as executor:
                    trial_outputs = list(
                        executor.map(lambda t: self._run_trial(*t), inputs)
                    )

                for i in range(num_new_trials):
                    trial_output = trial_outputs[i]
                    self._add_model_and_report_trial_metrics(
                        suggested_trials[i].name, trial_output
                    )
                    if not trial_output:
                        num_failed_trials += 1
                        if num_failed_trials == self.max_failed_trial_count:
                            raise ValueError("Maximum number of failed trials reached.")
                num_completed_trials += num_new_trials
        except Exception as e:
            raise e
        finally:
            # Enable remote job logs after trials are complete.
            logging.getLogger("vertexai.remote_execution").disabled = False

        if num_failed_trials == num_completed_trials:
            raise ValueError("All trials failed.")

        _LOGGER.info(
            f"Number of completed trials: {num_completed_trials}. Tuning complete."
        )
