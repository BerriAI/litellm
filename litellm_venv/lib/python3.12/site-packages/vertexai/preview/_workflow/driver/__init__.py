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
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type, TypeVar

from google.cloud.aiplatform import jobs
import vertexai
from vertexai.preview._workflow import launcher
from vertexai.preview._workflow import shared
from vertexai.preview._workflow.executor import (
    training,
    prediction,
)
from vertexai.preview._workflow.executor import (
    remote_container_training,
)

ModelBase = TypeVar("ModelBase")
ModelVertexSubclass = TypeVar("ModelVertexSubclass", bound=ModelBase)

_WRAPPED_CLASS_PREFIX = "_Vertex"


class VertexRemoteFunctor:
    """Functor to be used to wrap methods for remote execution."""

    def __init__(
        self,
        method: Callable[..., Any],
        remote_executor: Callable[..., Any],
        remote_executor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Wraps a method into VertexRemoteFunctor so that the method is remotely executable.

        Example Usage:
        ```
        functor = VertexRemoteFunctor(LogisticRegression.fit, training.remote_training)
        setattr(LogisticRegression, "fit", functor)

        model = LogisticRegression()
        model.fit.vertex.remote_config.staging_bucket = REMOTE_JOB_BUCKET
        model.fit.vertex.remote=True
        model.fit(X_train, y_train)
        ```

        Args:
            method (Callable[..., Any]):
                Required. The method to be wrapped.
            remote_executor (Callable[..., Any]):
                Required. The remote executor for the method.
            remote_executor_kwargs (Dict[str, Any]):
                Optional. kwargs used in remote executor.
        """
        self._method = method
        # TODO(b/278074360) Consider multiple levels of configurations.
        if inspect.ismethod(method):
            # For instance method, instantiate vertex config directly.
            self.vertex = shared.configs.VertexConfig()
        else:
            # For function, instantiate vertex config later, when the method is
            # bounded to an instance.
            self.vertex = shared.configs.VertexConfig
        self._remote_executor = remote_executor
        self._remote_executor_kwargs = remote_executor_kwargs or {}
        functools.update_wrapper(self, method)

    def __get__(self, instance, owner) -> Any:
        # For class and instance method that already instantiate a new functor,
        # return self directly
        if (instance is None) or isinstance(self.vertex, shared.configs.VertexConfig):
            return self

        # Instantiate a new functor for the instance method
        functor_with_instance_bound_method = self.__class__(
            self._method.__get__(instance, owner),
            self._remote_executor,
            self._remote_executor_kwargs,
        )
        functor_with_instance_bound_method.vertex = self.vertex()
        setattr(instance, self._method.__name__, functor_with_instance_bound_method)
        return functor_with_instance_bound_method

    def __call__(self, *args, **kwargs) -> Any:
        bound_args = inspect.signature(self._method).bind(*args, **kwargs)

        # NOTE: may also need to handle the case of
        # bound_args.arguments.get("self"),

        invokable = shared._Invokable(
            instance=getattr(self._method, "__self__"),
            method=self._method,
            bound_arguments=bound_args,
            remote_executor=self._remote_executor,
            remote_executor_kwargs=self._remote_executor_kwargs,
            vertex_config=self.vertex,
        )

        return _workflow_driver.invoke(invokable)


def _supported_member_iter(instance: Any) -> Iterator[Tuple[str, Callable[..., Any]]]:
    """Iterates through known method names and returns matching methods."""
    for attr_name in shared.supported_frameworks.REMOTE_TRAINING_OVERRIDE_LIST:
        attr_value = getattr(instance, attr_name, None)
        if attr_value:
            yield attr_name, attr_value, training.remote_training, None

    for attr_name in shared.supported_frameworks.REMOTE_PREDICTION_OVERRIDE_LIST:
        attr_value = getattr(instance, attr_name, None)
        if attr_value:
            yield attr_name, attr_value, prediction.remote_prediction, None


def _patch_class(cls: Type[ModelBase]) -> Type[ModelVertexSubclass]:
    """Creates a new class that inherited from original class and add Vertex remote execution support."""

    if hasattr(cls, "_wrapped_by_vertex"):
        return cls

    new_cls = type(
        f"{_WRAPPED_CLASS_PREFIX}{cls.__name__}", (cls,), {"_wrapped_by_vertex": True}
    )
    for (
        attr_name,
        attr_value,
        remote_executor,
        remote_executor_kwargs,
    ) in _supported_member_iter(cls):
        setattr(
            new_cls,
            attr_name,
            VertexRemoteFunctor(attr_value, remote_executor, remote_executor_kwargs),
        )

    return new_cls


def _rewrapper(
    instance: Any,
    wrapped_class: Any,
    config_map: Dict[str, shared.configs.VertexConfig],
):
    """Rewraps in place instances after remote execution has completed.

    Args:
        instance (Any):
            Required. Instance to rewrap.
        wrapped_class (Any):
            Required. The class type that the instance will be wrapped into.
        config_map (Dict[str, shared.configs.VertexConfig]):
            Required. Instance of config before unwrapping. Maintains
            the config after wrapping.
    """
    instance.__class__ = wrapped_class
    for attr_name, (
        vertex_config,
        remote_executor,
        remote_executor_kwargs,
    ) in config_map.items():
        method = getattr(instance, attr_name)
        if isinstance(method, VertexRemoteFunctor):
            method.vertex = vertex_config
            setattr(instance, attr_name, method)
        else:
            functor = VertexRemoteFunctor(
                method, remote_executor, remote_executor_kwargs
            )
            functor.vertex = vertex_config
            setattr(instance, attr_name, functor)


def _unwrapper(instance: Any) -> Callable[..., Any]:
    """Unwraps all Vertex functor method.

    This should be done before locally executing or remotely executing.
    """
    current_class = instance.__class__
    super_class = current_class.__mro__[1]
    wrapped_in_place = (
        current_class.__name__ != f"{_WRAPPED_CLASS_PREFIX}{super_class.__name__}"
    )

    config_map = dict()

    if not wrapped_in_place:
        for (
            attr_name,
            attr_value,
            remote_executor,
            remote_executor_kwargs,
        ) in _supported_member_iter(instance):
            if isinstance(attr_value, VertexRemoteFunctor):
                config_map[attr_name] = (
                    attr_value.vertex,
                    remote_executor,
                    remote_executor_kwargs,
                )
                setattr(instance, attr_name, attr_value._method)

        instance.__class__ = super_class

    else:
        for attr_name, attr_value in inspect.getmembers(instance):
            if isinstance(attr_value, VertexRemoteFunctor):
                config_map[attr_name] = (
                    attr_value.vertex,
                    attr_value._remote_executor,
                    attr_value._remote_executor_kwargs,
                )
                setattr(instance, attr_name, attr_value._method)

    return functools.partial(
        _rewrapper, wrapped_class=current_class, config_map=config_map
    )


class _WorkFlowDriver:
    def __init__(self):
        self._launcher = launcher._WorkflowLauncher()

    def invoke(self, invokable: shared._Invokable) -> Any:
        """
        Wrapper should forward implementation to this method.

        NOTE: Not threadsafe w.r.t the instance.
        """

        rewrapper = None
        # unwrap
        if (
            invokable.instance is not None
            and invokable.remote_executor is not remote_container_training.train
        ):
            rewrapper = _unwrapper(invokable.instance)

        result = self._launch(invokable, rewrapper)

        # rewrap the original instance
        if rewrapper and invokable.instance is not None:
            rewrapper(invokable.instance)
        # also rewrap the result if the result is an estimator not a dataset
        if rewrapper and isinstance(result, type(invokable.instance)):
            rewrapper(result)

        if hasattr(result, "state") and result.state in jobs._JOB_ERROR_STATES:
            raise RuntimeError("Remote job failed with:\n%s" % result.error)

        return result

    def _launch(self, invokable: shared._Invokable, rewrapper: Any) -> Any:
        """
        Launches an invokable.
        """
        return self._launcher.launch(
            invokable=invokable,
            global_remote=vertexai.preview.global_config.remote,
            rewrapper=rewrapper,
        )


_workflow_driver = _WorkFlowDriver()
