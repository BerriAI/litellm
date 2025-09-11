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

from google.cloud.aiplatform.compat.types import (
    explanation as explanation_compat,
    explanation_metadata as explanation_metadata_compat,
)

ExplanationMetadata = explanation_metadata_compat.ExplanationMetadata

# ExplanationMetadata subclasses
InputMetadata = ExplanationMetadata.InputMetadata
OutputMetadata = ExplanationMetadata.OutputMetadata

# InputMetadata subclasses
Encoding = InputMetadata.Encoding
FeatureValueDomain = InputMetadata.FeatureValueDomain
Visualization = InputMetadata.Visualization


ExplanationParameters = explanation_compat.ExplanationParameters
FeatureNoiseSigma = explanation_compat.FeatureNoiseSigma

ExplanationSpec = explanation_compat.ExplanationSpec

# Classes used by ExplanationParameters
IntegratedGradientsAttribution = explanation_compat.IntegratedGradientsAttribution
SampledShapleyAttribution = explanation_compat.SampledShapleyAttribution
SmoothGradConfig = explanation_compat.SmoothGradConfig
XraiAttribution = explanation_compat.XraiAttribution
Presets = explanation_compat.Presets
Examples = explanation_compat.Examples


__all__ = (
    "Encoding",
    "ExplanationSpec",
    "ExplanationMetadata",
    "ExplanationParameters",
    "FeatureNoiseSigma",
    "FeatureValueDomain",
    "InputMetadata",
    "IntegratedGradientsAttribution",
    "OutputMetadata",
    "SampledShapleyAttribution",
    "SmoothGradConfig",
    "Visualization",
    "XraiAttribution",
    "Presets",
    "Examples",
)
