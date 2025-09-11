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

import abc
import inspect
from typing import Any, Callable, Dict, Optional, Type

from vertexai.preview._workflow import driver
from vertexai.preview._workflow.executor import (
    training,
)
from vertexai.preview._workflow.serialization_engine import (
    any_serializer,
)
from vertexai.preview._workflow.shared import (
    supported_frameworks,
)
from vertexai.preview.developer import remote_specs


def remote_method_decorator(
    method: Callable[..., Any],
    remote_executor: Callable[..., Any],
    remote_executor_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable[..., Any]:
    """Wraps methods as Functor object to support configuration on method."""
    return driver.VertexRemoteFunctor(method, remote_executor, remote_executor_kwargs)


def remote_class_decorator(cls: Type) -> Type:
    """Add Vertex attributes to a class object."""

    if not supported_frameworks._is_oss(cls):
        raise ValueError(
            f"Class {cls.__name__} not supported. "
            "Currently support remote execution on "
            f"{supported_frameworks.REMOTE_FRAMEWORKS} classes."
        )

    return driver._patch_class(cls)


def remote(cls_or_method: Any) -> Any:
    """Takes a class or method and add Vertex remote execution support.

    ex:
    ```

    LogisticRegression = vertexai.preview.remote(LogisticRegression)
    model = LogisticRegression()
    model.fit.vertex.remote_config.staging_bucket = REMOTE_JOB_BUCKET
    model.fit.vertex.remote=True
    model.fit(X_train, y_train)
    ```

    Args:
        cls_or_method (Any):
            Required. A class or method that will be added Vertex remote
            execution support.

    Returns:
        A class or method that can be executed remotely.
    """
    # Make sure AnySerializer has been instantiated before wrapping any classes.
    if any_serializer.AnySerializer not in any_serializer.AnySerializer._instances:
        any_serializer.AnySerializer()

    if inspect.isclass(cls_or_method):
        return remote_class_decorator(cls_or_method)
    else:
        return remote_method_decorator(cls_or_method, training.remote_training)


class VertexModel(metaclass=abc.ABCMeta):
    """mixin class that can be used to add Vertex AI remote execution to a custom model."""

    def __init__(self):
        vertex_wrapper = False
        for _, attr_value in inspect.getmembers(self):
            if isinstance(attr_value, driver.VertexRemoteFunctor):
                vertex_wrapper = True
                break
        # TODO(b/279631878) Remove this check once we support more decorators.
        if not vertex_wrapper:
            raise ValueError(
                "No method is enabled for Vertex remote training. Please decorator "
                "your training methods with `@vertexai.preview.developer.mark.train`."
            )
        self._cluster_spec = None

    @property
    def cluster_spec(self):
        return self._cluster_spec

    @cluster_spec.setter
    def cluster_spec(self, cluster_spec: remote_specs._ClusterSpec):
        self._cluster_spec = cluster_spec
