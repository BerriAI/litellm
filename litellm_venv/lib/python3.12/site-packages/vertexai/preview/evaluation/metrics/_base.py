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

from typing import Any, Callable, Dict


class CustomMetric:
    """The custom evaluation metric.

    Attributes:
      name: The name of the metric.
      metric_function: The evaluation function. Must use the dataset row/instance
       as the metric_function input. Returns per-instance metric result as a
       dictionary. The metric score must mapped to the CustomMetric.name as key.
    """

    def __init__(
        self,
        name: str,
        metric_function: Callable[
            [Dict[str, Any]],
            Dict[str, Any],
        ],
    ):
        """Initializes the evaluation metric."""
        self.name = name
        self.metric_function = metric_function

    def __str__(self):
        return self.name


def make_metric(
    name: str, metric_function: Callable[[Dict[str, Any]], Dict[str, Any]]
) -> CustomMetric:
    """Makes a custom metric.

    Args:
      name: The name of the metric
      metric_function: The evaluation function. Must use the dataset row/instance
        as the metric_function input. Returns per-instance metric result as a
        dictionary. The metric score must mapped to the CustomMetric.name as key.

    Returns:
      A CustomMetric instance, can be passed to evaluate() function.
    """
    return CustomMetric(name, metric_function)
