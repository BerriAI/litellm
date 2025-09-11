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
"""Constants used by Metadata and Vertex Experiments."""

from google.cloud.aiplatform.compat.types import artifact

SYSTEM_RUN = "system.Run"
SYSTEM_EXPERIMENT = "system.Experiment"
SYSTEM_EXPERIMENT_RUN = "system.ExperimentRun"
SYSTEM_PIPELINE = "system.Pipeline"
SYSTEM_PIPELINE_RUN = "system.PipelineRun"
SYSTEM_METRICS = "system.Metrics"
GOOGLE_CLASSIFICATION_METRICS = "google.ClassificationMetrics"
GOOGLE_REGRESSION_METRICS = "google.RegressionMetrics"
GOOGLE_FORECASTING_METRICS = "google.ForecastingMetrics"
GOOGLE_EXPERIMENT_MODEL = "google.ExperimentModel"
_EXPERIMENTS_V2_TENSORBOARD_RUN = "google.VertexTensorboardRun"

_DEFAULT_SCHEMA_VERSION = "0.0.1"

SCHEMA_VERSIONS = {
    SYSTEM_RUN: _DEFAULT_SCHEMA_VERSION,
    SYSTEM_EXPERIMENT: _DEFAULT_SCHEMA_VERSION,
    SYSTEM_EXPERIMENT_RUN: _DEFAULT_SCHEMA_VERSION,
    SYSTEM_PIPELINE: _DEFAULT_SCHEMA_VERSION,
    SYSTEM_METRICS: _DEFAULT_SCHEMA_VERSION,
}

_BACKING_TENSORBOARD_RESOURCE_KEY = "backing_tensorboard_resource"

_CUSTOM_JOB_KEY = "_custom_jobs"
_CUSTOM_JOB_RESOURCE_NAME = "custom_job_resource_name"
_CUSTOM_JOB_CONSOLE_URI = "custom_job_console_uri"

_PARAM_KEY = "_params"
_METRIC_KEY = "_metrics"
_STATE_KEY = "_state"

_PARAM_PREFIX = "param"
_METRIC_PREFIX = "metric"
_TIME_SERIES_METRIC_PREFIX = "time_series_metric"

# This is currently used to filter in the Console.
EXPERIMENT_METADATA = {"experiment_deleted": False}

PIPELINE_PARAM_PREFIX = "input:"

TENSORBOARD_CUSTOM_JOB_EXPERIMENT_FIELD = "tensorboard_link"

GCP_ARTIFACT_RESOURCE_NAME_KEY = "resourceName"

# constant to mark an Experiment context as originating from the SDK
# TODO(b/235593750) Remove this field
_VERTEX_EXPERIMENT_TRACKING_LABEL = "vertex_experiment_tracking"

_TENSORBOARD_RUN_REFERENCE_ARTIFACT = artifact.Artifact(
    name="google-vertex-tensorboard-run-v0-0-1",
    schema_title=_EXPERIMENTS_V2_TENSORBOARD_RUN,
    schema_version="0.0.1",
    metadata={_VERTEX_EXPERIMENT_TRACKING_LABEL: True},
)

_TB_RUN_ARTIFACT_POST_FIX_ID = "-tb-run"
_EXPERIMENT_RUN_MAX_LENGTH = 128 - len(_TB_RUN_ARTIFACT_POST_FIX_ID)

# Label used to identify TensorboardExperiment as created from Vertex
# Experiments
_VERTEX_EXPERIMENT_TB_EXPERIMENT_LABEL = {
    "vertex_tensorboard_experiment_source": "vertex_experiment"
}

ENV_EXPERIMENT_KEY = "AIP_EXPERIMENT_NAME"
ENV_EXPERIMENT_RUN_KEY = "AIP_EXPERIMENT_RUN_NAME"
