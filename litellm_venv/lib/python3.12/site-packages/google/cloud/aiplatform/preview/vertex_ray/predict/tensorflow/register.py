"""Regsiter Tensorflow for Ray on Vertex AI."""

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
import os
import logging
import ray
from typing import Callable, Optional, Union, TYPE_CHECKING

from google.cloud import aiplatform
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform import utils
from google.cloud.aiplatform.preview.vertex_ray.predict.util import (
    predict_utils,
)


try:
    from ray.train import tensorflow as ray_tensorflow

    if TYPE_CHECKING:
        import tensorflow as tf

except ModuleNotFoundError as mnfe:
    raise ModuleNotFoundError("Tensorflow isn't installed.") from mnfe


def register_tensorflow(
    checkpoint: ray_tensorflow.TensorflowCheckpoint,
    artifact_uri: Optional[str] = None,
    _model: Optional[Union["tf.keras.Model", Callable[[], "tf.keras.Model"]]] = None,
    display_name: Optional[str] = None,
    **kwargs,
) -> aiplatform.Model:
    """Uploads a Ray Tensorflow Checkpoint as Tensorflow Model to Model Registry.

    Example usage:
        from vertex_ray.predict import tensorflow

        def create_model():
            model = tf.keras.Sequential(...)
            ...
            return model

        result = trainer.fit()
        my_model = tensorflow.register_tensorflow(
            checkpoint=result.checkpoint,
            _model=create_model,
            artifact_uri="gs://{gcs-bucket-name}/path/to/store",
            use_gpu=True
        )

        1. `use_gpu` will be passed to aiplatform.Model.upload_tensorflow_saved_model()
        2. The `create_model` provides the model_definition which is required if
        you create the TensorflowCheckpoint using `from_model` method.
        More here, https://docs.ray.io/en/latest/train/api/doc/ray.train.tensorflow.TensorflowCheckpoint.get_model.html#ray.train.tensorflow.TensorflowCheckpoint.get_model

    Args:
        checkpoint: TensorflowCheckpoint instance.
        artifact_uri (str):
            Optional. The path to the directory where Model Artifacts will be saved. If
            not set, will use staging bucket set in aiplatform.init().
        _model: Tensorflow Model Definition. Refer
            https://docs.ray.io/en/latest/train/api/doc/ray.train.tensorflow.TensorflowCheckpoint.get_model.html#ray.train.tensorflow.TensorflowCheckpoint.get_model
        display_name (str):
            Optional. The display name of the Model. The name can be up to 128
            characters long and can be consist of any UTF-8 characters.
        **kwargs:
            Any kwargs will be passed to aiplatform.Model registration.

    Returns:
        model (aiplatform.Model):
                Instantiated representation of the uploaded model resource.

    Raises:
        ValueError: Invalid Argument.
    """
    artifact_uri = artifact_uri or initializer.global_config.staging_bucket
    predict_utils.validate_artifact_uri(artifact_uri)
    prefix = "ray-on-vertex-registered-tensorflow-model"
    display_model_name = (
        (f"{prefix}-{utils.timestamped_unique_name()}")
        if display_name is None
        else display_name
    )
    tf_model = _get_tensorflow_model_from(checkpoint, model=_model)
    model_dir = os.path.join(artifact_uri, prefix)
    tf_model.save(model_dir)
    return aiplatform.Model.upload_tensorflow_saved_model(
        saved_model_dir=model_dir,
        display_name=display_model_name,
        **kwargs,
    )


def _get_tensorflow_model_from(
    checkpoint: ray_tensorflow.TensorflowCheckpoint,
    model: Optional[Union["tf.keras.Model", Callable[[], "tf.keras.Model"]]] = None,
) -> "tf.keras.Model":
    """Converts a TensorflowCheckpoint to Tensorflow Model.

    Args:
        checkpoint: TensorflowCheckpoint instance.
        model: Tensorflow Model Defination.

    Returns:
        A Tensorflow Native Framework Model.

    Raises:
        ValueError: Invalid Argument.
    """
    ray_version = ray.__version__
    if ray_version == "2.4.0":
        if not isinstance(checkpoint, ray_tensorflow.TensorflowCheckpoint):
            raise ValueError(
                "[Ray on Vertex AI]: arg checkpoint should be a"
                " ray.train.tensorflow.TensorflowCheckpoint instance"
            )
        if checkpoint.get_preprocessor() is not None:
            logging.warning(
                "Checkpoint contains preprocessor. However, converting from a Ray"
                " Checkpoint to framework specific model does NOT support"
                " preprocessing. The model will be exported without preprocessors."
            )

        return checkpoint.get_model(model)

    # get_model() signature changed in future versions
    try:
        from tensorflow import keras

        try:
            return keras.models.load_model(checkpoint.path)
        except OSError:
            return keras.models.load_model("gs://" + checkpoint.path)
    except ImportError:
        logging.warning("TensorFlow must be installed to load the trained model.")
