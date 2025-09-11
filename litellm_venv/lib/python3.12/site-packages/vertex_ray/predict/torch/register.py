"""Regsiter Torch for Ray on Vertex AI."""
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

import logging
import os
import ray
from ray.air._internal.torch_utils import load_torch_model
import tempfile
from google.cloud.aiplatform.utils import gcs_utils
from typing import Optional


try:
    from ray.train import torch as ray_torch
    import torch
except ModuleNotFoundError as mnfe:
    raise ModuleNotFoundError("Torch isn't installed.") from mnfe


def get_pytorch_model_from(
    checkpoint: ray_torch.TorchCheckpoint,
    model: Optional[torch.nn.Module] = None,
) -> torch.nn.Module:
    """Converts a TorchCheckpoint to Pytorch Model.

    Example:
        from vertex_ray.predict import torch
        result = TorchTrainer.fit(...)

        pytorch_model = torch.get_pytorch_model_from(
            checkpoint=result.checkpoint
        )

    Args:
        checkpoint: TorchCheckpoint instance.
        model: If the checkpoint contains a model state dict, and not the model
          itself, then the state dict will be loaded to this `model`. Otherwise,
          the model will be discarded.

    Returns:
        A Pytorch Native Framework Model.

    Raises:
        ValueError: Invalid Argument.
        ModuleNotFoundError: PyTorch isn't installed.
        RuntimeError: Model not found.
    """
    ray_version = ray.__version__
    if ray_version == "2.4.0":
        if not isinstance(checkpoint, ray_torch.TorchCheckpoint):
            raise ValueError(
                "[Ray on Vertex AI]: arg checkpoint should be a"
                " ray.train.torch.TorchCheckpoint instance"
            )
        if checkpoint.get_preprocessor() is not None:
            logging.warning(
                "Checkpoint contains preprocessor. However, converting from a Ray"
                " Checkpoint to framework specific model does NOT support"
                " preprocessing. The model will be exported without preprocessors."
            )
        return checkpoint.get_model(model=model)

    try:
        return checkpoint.get_model()
    except AttributeError:
        model_file_name = ray.train.torch.TorchCheckpoint.MODEL_FILENAME

    model_path = os.path.join(checkpoint.path, model_file_name)

    try:
        import torch

    except ModuleNotFoundError as mnfe:
        raise ModuleNotFoundError("PyTorch isn't installed.") from mnfe

    if os.path.exists(model_path):
        model_or_state_dict = torch.load(model_path, map_location="cpu")
    else:
        try:
            # Download from GCS to temp and then load_model
            with tempfile.TemporaryDirectory() as temp_dir:
                gcs_utils.download_from_gcs("gs://" + checkpoint.path, temp_dir)
                model_or_state_dict = torch.load(
                    f"{temp_dir}/{model_file_name}", map_location="cpu"
                )
        except Exception as e:
            raise RuntimeError(
                f"{model_file_name} not found in this checkpoint due to: {e}."
            )

    model = load_torch_model(saved_model=model_or_state_dict, model_definition=model)
    return model
