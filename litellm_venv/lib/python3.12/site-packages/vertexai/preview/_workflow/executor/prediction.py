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
from typing import Any

from vertexai.preview._workflow import (
    shared,
)
from vertexai.preview._workflow.executor import (
    training,
)


def remote_prediction(invokable: shared._Invokable, rewrapper: Any):
    """Wrapper function that makes a method executable by Vertex CustomJob."""
    predictions = training.remote_training(invokable=invokable, rewrapper=rewrapper)
    return predictions


def _online_prediction(invokable: shared._Invokable):
    # TODO(b/283292903) Implement online prediction method
    raise ValueError("Online prediction is not currently supported.")


def _batch_prediction(invokable: shared._Invokable):
    # TODO(b/283289019) Implement batch prediction method
    raise ValueError("Batch prediction is not currently supported.")
