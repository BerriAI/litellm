# -*- coding: utf-8 -*-

# Copyright 2022 Google LLC
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

from typing import Optional

from google.cloud.aiplatform import explain
from google.cloud.aiplatform.compat.types import (
    endpoint as gca_endpoint_compat,
)


def create_and_validate_explanation_spec(
    explanation_metadata: Optional[explain.ExplanationMetadata] = None,
    explanation_parameters: Optional[explain.ExplanationParameters] = None,
) -> Optional[explain.ExplanationSpec]:
    """Validates the parameters needed to create explanation_spec and creates it.

    Args:
        explanation_metadata (explain.ExplanationMetadata):
            Optional. Metadata describing the Model's input and output for
            explanation. `explanation_metadata` is optional while
            `explanation_parameters` must be specified when used.
            For more details, see `Ref docs <http://tinyurl.com/1igh60kt>`
        explanation_parameters (explain.ExplanationParameters):
            Optional. Parameters to configure explaining for Model's
            predictions.
            For more details, see `Ref docs <http://tinyurl.com/1an4zake>`

    Returns:
        explanation_spec: Specification of Model explanation.

    Raises:
        ValueError: If `explanation_metadata` is given, but
        `explanation_parameters` is omitted. `explanation_metadata` is optional
        while `explanation_parameters` must be specified when used.
    """
    if bool(explanation_metadata) and not bool(explanation_parameters):
        raise ValueError(
            "To get model explanation, `explanation_parameters` must be specified."
        )

    if explanation_parameters:
        explanation_spec = gca_endpoint_compat.explanation.ExplanationSpec()
        explanation_spec.parameters = explanation_parameters
        if explanation_metadata:
            explanation_spec.metadata = explanation_metadata
        return explanation_spec

    return None
