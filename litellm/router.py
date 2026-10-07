import inspect
from typing import TYPE_CHECKING, Final, TypeAlias, cast

from litellm.router_backends.python_router import PythonRouter
from litellm.router_backends.rust_router import RustRouter
from litellm.router_backends.selection import select_backend
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
    _RouterSurface: TypeAlias = PythonRouter
else:
    _RouterSurface: TypeAlias = object


class _Forwarded:
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __get__(self, instance: "Router | None", owner: type) -> object:
        target: Final = PythonRouter if instance is None else instance.backend
        return cast(object, getattr(target, self._name))  # cast-ok: forwarded attribute

    def __set__(self, instance: "Router", value: object) -> None:
        setattr(instance.backend, self._name, value)

    def __delete__(self, instance: "Router") -> None:
        delattr(instance.backend, self._name)


class Router(_RouterSurface):
    backend: PythonRouter | RustRouter

    if not TYPE_CHECKING:

        def __init__(self, *args, **kwargs) -> None:  # kwargs-ok: forwards PythonRouter.__init__'s signature
            object.__setattr__(self, "backend", select_backend(args, kwargs))

        __init__.__signature__ = inspect.signature(PythonRouter.__init__)

        def __getattr__(self, name: str) -> object:
            if name == "backend":
                raise AttributeError(name)
            return getattr(self.backend, name)

        def __setattr__(self, name: str, value: object) -> None:
            setattr(self.backend, name, value)

        def __delattr__(self, name: str) -> None:
            delattr(self.backend, name)


for _name in [name for name in vars(PythonRouter) if not name.startswith("__")]:
    setattr(Router, _name, _Forwarded(_name))
