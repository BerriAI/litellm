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

import inspect
from typing import Any

from google.cloud.aiplatform import base
from google.cloud.aiplatform.utils import gcs_utils
from vertexai.preview import developer
from vertexai.preview._workflow.driver import remote
from vertexai.preview._workflow.shared import configs
from vertexai.preview.developer import remote_specs


try:
    import pandas as pd

    PandasData = pd.DataFrame

except ImportError:
    PandasData = Any


_LOGGER = base.Logger(__name__)

# Constants for TabNetTrainer
_TABNET_TRAINING_IMAGE = "us-docker.pkg.dev/vertex-ai-restricted/automl-tabular/tabnet-training:20230605_1325"

_TABNET_FIT_DISPLAY_NAME = "fit"
_TABNET_MACHINE_TYPE = "c2-standard-16"
_TABNET_BOOT_DISK_TYPE = "pd-ssd"
_TABNET_BOOT_DISK_SIZE_GB = 100

_CLASSIFICATION = "classification"
_REGRESSION = "regression"


class TabNetTrainer(remote.VertexModel):
    """The TabNet trainer for remote training and prediction."""

    def __init__(
        self,
        model_type: str,
        target_column: str,
        learning_rate: float,
        job_dir: str = "",
        enable_profiler: bool = False,
        cache_data: str = "auto",
        seed: int = 1,
        large_category_dim: int = 1,
        large_category_thresh: int = 300,
        yeo_johnson_transform: bool = False,
        weight_column: str = "",
        max_steps: int = -1,
        max_train_secs: int = -1,
        measurement_selection_type: str = "BEST_MEASUREMENT",
        optimization_metric: str = "",
        eval_steps: int = 0,
        batch_size: int = 100,
        eval_frequency_secs: int = 600,
        feature_dim: int = 64,
        feature_dim_ratio: float = 0.5,
        num_decision_steps: int = 6,
        relaxation_factor: float = 1.5,
        decay_every: float = 100.0,
        decay_rate: float = 0.95,
        gradient_thresh: float = 2000.0,
        sparsity_loss_weight: float = 0.00001,
        batch_momentum: float = 0.95,
        batch_size_ratio: float = 0.25,
        num_transformer_layers: int = 4,
        num_transformer_layers_ratio: float = 0.25,
        class_weight: float = 1.0,
        loss_function_type: str = "default",
        alpha_focal_loss: float = 0.25,
        gamma_focal_loss: float = 2.0,
    ):
        """Initializes a TabNetTrainer instance.

        is_remote_trainer is always set to True because TabNetTrainer only
        supports remote training.

        Args:
            model_type (str):
                Required. The type of prediction the model is to produce.
                'classification' or 'regression'.
            target_column (str):
                Required. The target column name.
            learning_rate (float):
                Required. The learning rate used by the linear optimizer.
            job_dir (str):
                Optional. The GCS directory for reading and writing inside the
                the custom job. If provided, must start with 'gs://'. Default is
                ''.
            enable_profiler (bool):
                Optional. Enables profiling and saves a trace during evaluation.
                Default is False.
            cache_data (str):
                Optional. Whether to cache data or not. If set to 'auto',
                caching is determined based on the dataset size. Default is
                'auto'.
            seed (int):
                Optional. Seed to be used for this run. Default is 1.
            large_category_dim (int):
                Optional. Embedding dimension for categorical feature with
                large number of categories. Default is 1.
            large_category_thresh (int):
                Optional. Threshold for number of categories to apply
                large_category_dim embedding dimension to. Default is 300.
            yeo_johnson_transform (bool):
                Optional. Enables trainable Yeo-Johnson power transform. Default
                is False.
            weight_column (str):
                Optional. The weight column name. ''(empty string) for no
                weight column. Default is ''(empty string).
            max_steps (int):
                Optional. Number of steps to run the trainer for. -1 for no
                maximum steps. Default is -1.
            max_train_seconds (int):
                Optional. Amount of time in seconds to run the trainer for. -1
                for no maximum train seconds. Default is -1.
            measurement_selection_type (str):
                Optional. Which measurement to use if/when the service
                automatically selects the final measurement from previously
                reported intermediate measurements. One of 'BEST_MEASUREMENT'
                or 'LAST_MEASUREMENT'. Default is 'BEST_MEASUREMENT'.
            optimization_metric (str):
                Optional. Optimization metric used for
                `measurement_selection_type`. ''(empty string) for using the
                default value: 'rmse' for regression and 'auc' for
                classification. Default is ''(empty string).
            eval_steps (int):
                Optional. Number of steps to run evaluation for. If not
                specified or negative, it means run evaluation on the whole
                validation dataset. If set to 0, it means run evaluation for a
                fixed number of samples. Default is 0.
            batch_size (int):
                Optional. Batch size for training. Default is 100.
            eval_frequency_secs (int):
                Optional. Frequency at which evaluation and checkpointing will
                take place. Default is 600.
            feature_dim (int):
                Optional. Dimensionality of the hidden representation in feature
                transformation block. Default is 64.
            feature_dim_ratio (float):
                Optional. The ratio of output dimension (dimensionality of the
                outputs of each decision step) to feature dimension. Default is
                0.5.
            num_decision_steps (int):
                Optional. Number of sequential decision steps. Default is 6.
            relaxation_factor (float):
                Optional. Relaxation factor that promotes the reuse of each
                feature at different decision steps. When it is 1, a feature is
                enforced to be used only at one decision step and as it
                increases, more flexibility is provided to use a feature at
                multiple decision steps. Default is 1.5.
            decay_every (float):
                Optional. Number of iterations for periodically applying
                learning rate decaying. Default is 100.0.
            decay_rate (float):
                Optional. Learning rate decaying. Default is 0.95.
            gradient_thresh (float):
                Optional. Threshold for the norm of gradients for clipping.
                Default is 2000.0.
            sparsity_loss_weight (float):
                Optional. Weight of the loss for sparsity regularization
                (increasing it will yield more sparse feature selection).
                Default is 0.00001.
            batch_momentum (float):
                Optional. Momentum in ghost batch normalization. Default is
                0.95.
            batch_size_ratio (float):
                Optional. The ratio of virtual batch size (size of the ghost
                batch normalization) to batch size. Default is 0.25.
            num_transformer_layers (int):
                Optional. The number of transformer layers for each decision
                step. used only at one decision step and as it increases, more
                flexibility is provided to use a feature at multiple decision
                steps. Default is 4.
            num_transformer_layers_ratio (float):
                Optional. The ratio of shared transformer layer to transformer
                layers. Default is 0.25.
            class_weight (float):
                Optional. The class weight is used to compute a weighted cross
                entropy which is helpful in classifying imbalanced dataset. Only
                used for classification. Default is 1.0.
            loss_function_type (str):
                Optional. Loss function type. Loss function in classification
                [cross_entropy, weighted_cross_entropy, focal_loss], default is
                cross_entropy. Loss function in regression: [rmse, mae, mse],
                default is mse. "default" for default values. Default is
                "default".
            alpha_focal_loss (float):
                Optional. Alpha value (balancing factor) in focal_loss function.
                Only used for classification. Default is 0.25.
            gamma_focal_loss (float):
                Optional. Gamma value (modulating factor) for focal loss for
                focal loss. Only used for classification. Default is 2.0.
        Raises:
            ValueError if job_dir is set to an invalid GCS path.
        """
        super().__init__()
        if job_dir:
            gcs_utils.validate_gcs_path(job_dir)
        sig = inspect.signature(self.__init__)
        self._binding = sig.bind(
            model_type,
            target_column,
            learning_rate,
            job_dir,
            enable_profiler,
            cache_data,
            seed,
            large_category_dim,
            large_category_thresh,
            yeo_johnson_transform,
            weight_column,
            max_steps,
            max_train_secs,
            measurement_selection_type,
            optimization_metric,
            eval_steps,
            batch_size,
            eval_frequency_secs,
            feature_dim,
            feature_dim_ratio,
            num_decision_steps,
            relaxation_factor,
            decay_every,
            decay_rate,
            gradient_thresh,
            sparsity_loss_weight,
            batch_momentum,
            batch_size_ratio,
            num_transformer_layers,
            num_transformer_layers_ratio,
            class_weight,
            loss_function_type,
            alpha_focal_loss,
            gamma_focal_loss,
        ).arguments
        self._binding["is_remote_trainer"] = True
        self.model = None

    @developer.mark._remote_container_train(
        image_uri=_TABNET_TRAINING_IMAGE,
        additional_data=[
            remote_specs._InputParameterSpec(
                "training_data",
                argument_name="training_data_path",
                serializer="parquet",
            ),
            remote_specs._InputParameterSpec(
                "validation_data",
                argument_name="validation_data_path",
                serializer="parquet",
            ),
            remote_specs._InputParameterSpec("model_type"),
            remote_specs._InputParameterSpec("target_column"),
            remote_specs._InputParameterSpec("learning_rate"),
            remote_specs._InputParameterSpec("job_dir"),
            remote_specs._InputParameterSpec("enable_profiler"),
            remote_specs._InputParameterSpec("cache_data"),
            remote_specs._InputParameterSpec("seed"),
            remote_specs._InputParameterSpec("large_category_dim"),
            remote_specs._InputParameterSpec("large_category_thresh"),
            remote_specs._InputParameterSpec("yeo_johnson_transform"),
            remote_specs._InputParameterSpec("weight_column"),
            remote_specs._InputParameterSpec("max_steps"),
            remote_specs._InputParameterSpec("max_train_secs"),
            remote_specs._InputParameterSpec("measurement_selection_type"),
            remote_specs._InputParameterSpec("optimization_metric"),
            remote_specs._InputParameterSpec("eval_steps"),
            remote_specs._InputParameterSpec("batch_size"),
            remote_specs._InputParameterSpec("eval_frequency_secs"),
            remote_specs._InputParameterSpec("feature_dim"),
            remote_specs._InputParameterSpec("feature_dim_ratio"),
            remote_specs._InputParameterSpec("num_decision_steps"),
            remote_specs._InputParameterSpec("relaxation_factor"),
            remote_specs._InputParameterSpec("decay_every"),
            remote_specs._InputParameterSpec("decay_rate"),
            remote_specs._InputParameterSpec("gradient_thresh"),
            remote_specs._InputParameterSpec("sparsity_loss_weight"),
            remote_specs._InputParameterSpec("batch_momentum"),
            remote_specs._InputParameterSpec("batch_size_ratio"),
            remote_specs._InputParameterSpec("num_transformer_layers"),
            remote_specs._InputParameterSpec("num_transformer_layers_ratio"),
            remote_specs._InputParameterSpec("class_weight"),
            remote_specs._InputParameterSpec("loss_function_type"),
            remote_specs._InputParameterSpec("alpha_focal_loss"),
            remote_specs._InputParameterSpec("gamma_focal_loss"),
            remote_specs._InputParameterSpec("is_remote_trainer"),
            remote_specs._OutputParameterSpec("output_model_path"),
        ],
        remote_config=configs.DistributedTrainingConfig(
            display_name=_TABNET_FIT_DISPLAY_NAME,
            machine_type=_TABNET_MACHINE_TYPE,
            boot_disk_type=_TABNET_BOOT_DISK_TYPE,
            boot_disk_size_gb=_TABNET_BOOT_DISK_SIZE_GB,
        ),
    )
    def fit(self, training_data: PandasData, validation_data: PandasData) -> None:
        """Trains a tabnet model in a custom job.

        After the custom job successfully finishes, load the model and set it to
        self.model to enable prediction. If TensorFlow is not installed, the
        model will not be loaded.

        Training config can be overriden by setting the training config.

        Example Usage:
        `
        tabnet_trainer = TabNetTrainer(...)
        tabnet_trainer.fit.vertex.remote_config.staging_bucket = 'gs://...'
        tabnet_trainer.fit.vertex.remote_config.display_name = 'example'
        tabnet_trainer.fit(...)
        `

        PandasData refers to a pandas DataFrame. Each data frame should meet the
        following requirements:
            1. All entries should be numerical (no string, array or object).
            2. For categorical columns, the entries should be integers. In
            addition, the column type should be set to 'category'. Otherwise, it
            will be treated as numerical columns.
            3. The column names should be string.

        Args:
            training_data (pandas.DataFrame):
                Required. A pandas DataFrame for training.
            validation_data (pandas.DataFrame):
                Required. A pandas DataFrame for validation.
        """
        try:
            import tensorflow.saved_model as tf_saved_model

            self.model = tf_saved_model.load(self.output_model_path)
        except ImportError:
            _LOGGER.warning(
                "TensorFlow must be installed to load the trained model. The model is stored at %s",
                self.output_model_path,
            )

    def predict(self, input_data: PandasData) -> PandasData:
        """Makes prediction on input data through a trained model.

        Unlike in training and validation data, the categorical columns in
        prediction input data can have dtypes either 'category' or 'int', with
        'int' being numpy.int64 in pandas DataFrame.


        Args:
            input_data (pandas.DataFrame):
                Required. An input Pandas DataFrame containing data for
                prediction. It will be preprocessed into a dictionary as the
                input for to the trained model.
        Returns:
            Prediction results in the format of pandas DataFrame.
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow must be installed to make predictions."
            ) from None

        if self.model is None:
            if not hasattr(self, "output_model_path") or self.output_model_path is None:
                raise ValueError("No trained model. Please call .fit first.")
            self.model = tf.saved_model.load(self.output_model_path)

        prediction_inputs = {}
        for col in input_data.columns:
            if input_data[col].dtypes == "category":
                dtype = tf.int64
            else:
                dtype = tf.dtypes.as_dtype(input_data[col].dtypes)
            prediction_inputs[col] = tf.constant(input_data[col].to_list(), dtype=dtype)
        prediction_outputs = self.model.signatures["serving_default"](
            **prediction_inputs
        )
        if self._binding["model_type"] == _CLASSIFICATION:
            predicted_labels = []
            for score, labels in zip(
                prediction_outputs["scores"].numpy(),
                prediction_outputs["classes"].numpy().astype(int),
            ):
                predicted_labels.append(labels[score.argmax()])
            return pd.DataFrame({self._binding["target_column"]: predicted_labels})
        elif self._binding["model_type"] == _REGRESSION:
            return pd.DataFrame(
                {
                    self._binding["target_column"]: prediction_outputs["value"]
                    .numpy()
                    .reshape(-1)
                }
            )
        else:
            raise ValueError(f"Unsupported model type: {self._binding['model_type']}.")
