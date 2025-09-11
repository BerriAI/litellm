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

from google.cloud.aiplatform.metadata import metadata

from vertexai.preview import developer
from vertexai.preview import hyperparameter_tuning
from vertexai.preview import initializer
from vertexai.preview import tabular_models
from vertexai.preview._workflow.driver import (
    remote as remote_decorator,
)
from vertexai.preview._workflow.shared import (
    model_utils,
)


global_config = initializer.global_config
init = global_config.init
remote = remote_decorator.remote
VertexModel = remote_decorator.VertexModel
register = model_utils.register
from_pretrained = model_utils.from_pretrained

# For Vertex AI Experiment.

# ExperimentRun manipulation.
start_run = metadata._experiment_tracker.start_run
end_run = metadata._experiment_tracker.end_run
get_experiment_df = metadata._experiment_tracker.get_experiment_df

# Experiment logging.
log_params = metadata._experiment_tracker.log_params
log_metrics = metadata._experiment_tracker.log_metrics
log_time_series_metrics = metadata._experiment_tracker.log_time_series_metrics
log_classification_metrics = metadata._experiment_tracker.log_classification_metrics


__all__ = (
    "init",
    "remote",
    "VertexModel",
    "register",
    "from_pretrained",
    "start_run",
    "end_run",
    "get_experiment_df",
    "log_params",
    "log_metrics",
    "log_time_series_metrics",
    "log_classification_metrics",
    "developer",
    "hyperparameter_tuning",
    "tabular_models",
)
