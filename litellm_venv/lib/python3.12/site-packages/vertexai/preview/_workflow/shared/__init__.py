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


import dataclasses
import inspect
from typing import Any, Callable, Dict, Optional

from vertexai.preview._workflow.shared import configs


@dataclasses.dataclass(frozen=True)
class _Invokable:
    """Represents a single invokable method.

    method: The method to invoke.
    bound_arguments: The arguments to use to invoke the method.
    vertex_config: User-specified configs for Vertex services.
    remote_executor: The executor that execute the method remotely.
    remote_executor_kwargs: kwargs used in the remote executor.
    instance: The instance the method is bound.
    """

    method: Callable[..., Any]
    bound_arguments: inspect.BoundArguments
    vertex_config: configs.VertexConfig
    remote_executor: Callable[..., Any]
    remote_executor_kwargs: Optional[Dict[str, Any]] = None
    instance: Optional[Any] = None
