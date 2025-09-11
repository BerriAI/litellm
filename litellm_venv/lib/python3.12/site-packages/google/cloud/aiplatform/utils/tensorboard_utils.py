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

from typing import Sequence, Dict
from google.cloud.aiplatform_v1beta1.services.tensorboard_service.client import (
    TensorboardServiceClient,
)

_SERVING_DOMAIN = "tensorboard.googleusercontent.com"


def _parse_experiment_name(experiment_name: str) -> Dict[str, str]:
    """Parses an experiment_name into its component segments.

    Args:
      experiment_name: Resource name of the TensorboardExperiment. E.g.
        "projects/123/locations/asia-east1/tensorboards/456/experiments/exp1"

    Returns:
      Components of the experiment name.

    Raises:
      ValueError: If the experiment_name is invalid.
    """
    matched = TensorboardServiceClient.parse_tensorboard_experiment_path(
        experiment_name
    )
    if not matched:
        raise ValueError(f"Invalid experiment name: {experiment_name}.")
    return matched


def get_experiment_url(experiment_name: str) -> str:
    """Get URL for comparing experiments.

    Args:
      experiment_name: Resource name of the TensorboardExperiment. E.g.
        "projects/123/locations/asia-east1/tensorboards/456/experiments/exp1"

    Returns:
      URL for the tensorboard web app.
    """
    location = _parse_experiment_name(experiment_name)["location"]
    name_for_url = experiment_name.replace("/", "+")
    return f"https://{location}.{_SERVING_DOMAIN}/experiment/{name_for_url}"


def get_experiments_compare_url(experiment_names: Sequence[str]) -> str:
    """Get URL for comparing experiments.

    Args:
      experiment_names: Resource names of the TensorboardExperiments that needs to
        be compared.

    Returns:
      URL for the tensorboard web app.
    """
    if len(experiment_names) < 2:
        raise ValueError("At least two experiment_names are required.")

    locations = {
        _parse_experiment_name(experiment_name)["location"]
        for experiment_name in experiment_names
    }
    if len(locations) != 1:
        raise ValueError(
            f"Got experiments from different locations: {', '.join(locations)}."
        )
    location = locations.pop()

    experiment_url_segments = []
    for idx, experiment_name in enumerate(experiment_names):
        name_segments = _parse_experiment_name(experiment_name)
        experiment_url_segments.append(
            "{cnt}-{experiment}:{project}+{location}+{tensorboard}+{experiment}".format(
                cnt=idx + 1, **name_segments
            )
        )
    encoded_names = ",".join(experiment_url_segments)
    return f"https://{location}.{_SERVING_DOMAIN}/compare/{encoded_names}"
