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

from abc import ABC, abstractmethod
from typing import Any


class Predictor(ABC):
    """Interface of the Predictor class for Custom Prediction Routines.

    The Predictor is responsible for the ML logic for processing a prediction request.
    Specifically, the Predictor must define:
    (1) How to load all model artifacts used during prediction into memory.
    (2) The logic that should be executed at predict time.

    When using the default ``PredictionHandler``, the ``Predictor`` will be invoked as
    follows:

    .. code-block:: python

        predictor.postprocess(predictor.predict(predictor.preprocess(prediction_input)))

    """

    def __init__(self):
        return

    @abstractmethod
    def load(self, artifacts_uri: str) -> None:
        """Loads the model artifact.

        Args:
            artifacts_uri (str):
                Required. The value of the environment variable AIP_STORAGE_URI.
        """
        pass

    def preprocess(self, prediction_input: Any) -> Any:
        """Preprocesses the prediction input before doing the prediction.

        Args:
            prediction_input (Any):
                Required. The prediction input that needs to be preprocessed.

        Returns:
            The preprocessed prediction input.
        """
        return prediction_input

    @abstractmethod
    def predict(self, instances: Any) -> Any:
        """Performs prediction.

        Args:
            instances (Any):
                Required. The instance(s) used for performing prediction.

        Returns:
            Prediction results.
        """
        pass

    def postprocess(self, prediction_results: Any) -> Any:
        """Postprocesses the prediction results.

        Args:
            prediction_results (Any):
                Required. The prediction results.

        Returns:
            The postprocessed prediction results.
        """
        return prediction_results
