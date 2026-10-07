import inspect
from typing import TYPE_CHECKING, cast

from litellm.router_backends.python_router import PythonRouter
from litellm.types.router import (
    AlertingConfig,
    AllowedFailsPolicy,
    AssistantsTypedDict,
    CustomRoutingStrategyBase,
    Deployment,
    LiteLLM_Params,
    ModelGroupInfo,
    RetryPolicy,
)

__all__ = (
    "AlertingConfig",
    "AllowedFailsPolicy",
    "AssistantsTypedDict",
    "CustomRoutingStrategyBase",
    "Deployment",
    "LiteLLM_Params",
    "ModelGroupInfo",
    "RetryPolicy",
    "Router",
)

# Provisional (router-poc.md, "Facade surface"): Router exposes the backend's full surface,
# raw internals included, until callers move to a typed protocol.
if TYPE_CHECKING:
    _RouterSurface = PythonRouter
else:
    _RouterSurface = object


class _Forwarded:
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __get__(self, instance: "Router | None", owner: type) -> object:
        return cast(object, getattr(PythonRouter if instance is None else instance.backend, self._name))  # cast-ok: forwarded attribute

    def __set__(self, instance: "Router", value: object) -> None:
        setattr(instance.backend, self._name, value)

    def __delete__(self, instance: "Router") -> None:
        delattr(instance.backend, self._name)


class Router(_RouterSurface):
    backend: PythonRouter

    if not TYPE_CHECKING:

        def __init__(self, *args, **kwargs):
            object.__setattr__(self, "backend", PythonRouter(*args, **kwargs))

        __init__.__signature__ = inspect.signature(PythonRouter.__init__)

        def __getattr__(self, name):
            if name == "backend":
                raise AttributeError(name)
            return getattr(self.backend, name)

        def __setattr__(self, name, value):
            setattr(self.backend, name, value)

        def __delattr__(self, name):
            delattr(self.backend, name)


for _name in [name for name in vars(PythonRouter) if not name.startswith("__")]:
    setattr(Router, _name, _Forwarded(_name))
