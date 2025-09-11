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

import functools
import inspect
from typing import Any, Callable, List, Optional, Union

from vertexai.preview._workflow.driver import remote
from vertexai.preview._workflow.executor import (
    remote_container_training,
)
from vertexai.preview._workflow.executor import (
    training,
    prediction,
)
from vertexai.preview._workflow.shared import configs
from vertexai.preview.developer import remote_specs


def train(
    remote_config: Optional[configs.RemoteConfig] = None,
) -> Callable[..., Any]:
    """Decorator to enable Vertex remote training on a method.

    Example Usage:
        ```
        vertexai.init(
            project="my-project",
            location="my-location",
            staging_bucket="gs://my-bucket",
        )
        vertexai.preview.init(remote=True)

        class MyModel(vertexai.preview.VertexModel):
            ...

            @vertexai.preview.developer.mark.train()
            def my_train_method(...):
                ...

        model = MyModel(...)

        # This train method will be executed remotely
        model.my_train_method(...)
        ```

    Args:
        remote_config (config.RemoteConfig):
            Optional. A class that holds the configuration for the remote job.

    Returns:
        A wrapped method with its original signature.
    """

    def remote_training_wrapper(method: Callable[..., Any]) -> Callable[..., Any]:
        functor = remote.remote_method_decorator(method, training.remote_training)
        if remote_config is not None:
            if inspect.ismethod(method):
                functor.vertex.remote_config = remote_config
            else:
                functor.vertex = functools.partial(
                    configs.VertexConfig, remote_config=remote_config
                )

        return functor

    return remote_training_wrapper


# pylint: disable=protected-access
def _remote_container_train(
    image_uri: str,
    additional_data: List[
        Union[remote_specs._InputParameterSpec, remote_specs._OutputParameterSpec]
    ],
    remote_config: Optional[configs.DistributedTrainingConfig] = None,
) -> Callable[..., Any]:
    """Decorator to enable remote training with a container image.

    This decorator takes the parameters from the __init__ function (requires
    setting up binding outside of the decorator) and the function that it
    decorates, preprocesses the arguments, and launches a custom job for
    training.

    As the custom job is running, the inputs are read and parsed according to
    the container code, and the outputs are written to the GCS paths specified
    for each output field.

    If the custom job succeeds, the decorator deserializes the outputs from the
    custom job and sets them as instance attributes. Each output will be either
    a string or bytes, and the function this decorator decorates may
    additionally post-process the outputs to their corresponding types.

    Args:
        image_uri (str):
            Required. The pre-built docker image uri for CustomJob.
        additional_data (List):
            Required. A list of input and output parameter specs.
        remote_config (config.DistributedTrainingConfig):
            Optional. A class that holds the configuration for the distributed
            training remote job.

    Returns:
        An inner decorator that returns the decorated remote container training
        function.

    Raises:
        ValueError if the decorated function has a duplicate argument name as
        the parameters in existing binding, or if an additional data is neither
        an input parameter spec or an output parameter spec.
    """

    def remote_training_wrapper(method: Callable[..., Any]) -> Callable[..., Any]:
        functor = remote.remote_method_decorator(
            method,
            remote_container_training.train,
            remote_executor_kwargs={
                "image_uri": image_uri,
                "additional_data": additional_data,
            },
        )
        config = remote_config or configs.DistributedTrainingConfig()
        if inspect.ismethod(method):
            functor.vertex.remote_config = config
            functor.vertex.remote = True
        else:
            functor.vertex = functools.partial(
                configs.VertexConfig, remote=True, remote_config=config
            )

        return functor

    return remote_training_wrapper


def predict(
    remote_config: Optional[configs.RemoteConfig] = None,
) -> Callable[..., Any]:
    """Decorator to enable Vertex remote prediction on a method.

    Example Usage:
        ```
        vertexai.init(
            project="my-project",
            location="my-location",
            staging_bucket="gs://my-bucket",
        )
        vertexai.preview.init(remote=True)

        class MyModel(vertexai.preview.VertexModel):
            ...

            @vertexai.preview.developer.mark.predict()
            def my_predict_method(...):
                ...

        model = MyModel(...)

        # This train method will be executed remotely
        model.my_predict_method(...)
        ```

    Args:
        remote_config (config.RemoteConfig):
            Optional. A class that holds the configuration for the remote job.

    Returns:
        A wrapped method with its original signature.
    """

    def remote_prediction_wrapper(method: Callable[..., Any]) -> Callable[..., Any]:
        functor = remote.remote_method_decorator(method, prediction.remote_prediction)
        if remote_config is not None:
            if inspect.ismethod(method):
                functor.vertex.remote_config = remote_config
            else:
                functor.vertex = functools.partial(
                    configs.VertexConfig, remote_config=remote_config
                )

        return functor

    return remote_prediction_wrapper
