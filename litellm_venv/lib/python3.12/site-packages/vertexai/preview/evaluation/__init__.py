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
"""Rapid GenAI Evaluation Module."""

from vertexai.preview.evaluation import _base
from vertexai.preview.evaluation import _eval_tasks
from vertexai.preview.evaluation import metrics
from vertexai.preview.evaluation import prompt_template


EvalResult = _base.EvalResult
EvalTask = _eval_tasks.EvalTask
CustomMetric = metrics.CustomMetric
make_metric = metrics.make_metric
PromptTemplate = prompt_template.PromptTemplate

__all__ = [
    "CustomMetric",
    "EvalResult",
    "EvalTask",
    "make_metric",
    "PromptTemplate",
]
