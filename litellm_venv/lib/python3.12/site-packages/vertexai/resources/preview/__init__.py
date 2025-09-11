# -*- coding: utf-8 -*-

# Copyright 2024 Google LLC
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
"""The vertexai resources preview module."""

from google.cloud.aiplatform.preview.jobs import (
    CustomJob,
    HyperparameterTuningJob,
)
from google.cloud.aiplatform.preview.models import (
    Prediction,
    DeploymentResourcePool,
    Endpoint,
    Model,
)
from google.cloud.aiplatform.preview.featurestore.entity_type import (
    EntityType,
)

from google.cloud.aiplatform.preview.pipelinejobschedule.pipeline_job_schedules import (
    PipelineJobSchedule,
)

from vertexai.resources.preview.feature_store import (
    FeatureOnlineStore,
    FeatureOnlineStoreType,
    FeatureView,
    FeatureViewReadResponse,
    IndexConfig,
    TreeAhConfig,
    BruteForceConfig,
    DistanceMeasureType,
    AlgorithmConfig,
)


__all__ = (
    "CustomJob",
    "HyperparameterTuningJob",
    "Prediction",
    "DeploymentResourcePool",
    "Endpoint",
    "Model",
    "PersistentResource",
    "EntityType",
    "PipelineJobSchedule",
    "FeatureOnlineStoreType",
    "FeatureOnlineStore",
    "FeatureView",
    "FeatureViewReadResponse",
    "IndexConfig",
    "TreeAhConfig",
    "BruteForceConfig",
    "DistanceMeasureType",
    "AlgorithmConfig",
)
