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

from typing import Any

from vertexai.preview._workflow import shared
from vertexai.preview._workflow.executor import (
    remote_container_training,
    training,
    prediction,
)


class _WorkflowExecutor:
    """Executes an invokable either locally or remotely."""

    def local_execute(self, invokable: shared._Invokable) -> Any:
        if invokable.remote_executor is remote_container_training.train:
            raise ValueError(
                "Remote container train is only supported for remote mode."
            )
        return invokable.method(
            *invokable.bound_arguments.args, **invokable.bound_arguments.kwargs
        )

    def remote_execute(self, invokable: shared._Invokable, rewrapper: Any) -> Any:
        if invokable.remote_executor not in (
            remote_container_training.train,
            training.remote_training,
            prediction.remote_prediction,
        ):
            raise ValueError(f"{invokable.remote_executor} is not supported.")

        if invokable.remote_executor == remote_container_training.train:
            invokable.remote_executor(invokable)
        else:
            return invokable.remote_executor(invokable, rewrapper=rewrapper)


_workflow_executor = _WorkflowExecutor()
