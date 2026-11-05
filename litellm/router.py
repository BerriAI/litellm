"""
litellm.Router: load balancing, retries, fallbacks and cooldowns across deployments.

``Router`` holds no routing logic. It builds its configuration (``litellm/router_config.py``), picks
one ``RouterCore`` at construction (``litellm/router_core.py``) and delegates every member to it:
the endpoint methods (acompletion, aembedding, aresponses, aanthropic_messages, ...) and the entry
points (registry, settings, health, select, apply, observe, register_plugin, retire). The members
outside code used before the entry points live in ``DeprecatedRouterMembers`` until they are removed.
"""

import inspect
import weakref
from collections.abc import Mapping
from types import FunctionType, MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm import router_python_core
from litellm.router_config import RouterConfig, build_router_config
from litellm.router_core import RouterCore, make_core
from litellm.router_deprecated import DeprecatedRouterMembers
from litellm.router_python_core import (
    PythonCore,
    as_output_cap,
    model_info_is_active_for_environment,
    stamped_client_ceiling,
)
from litellm.router_python_host import PythonHost, python_host
from litellm.types.router import (
    AllowedFailsPolicy,
    AssistantsTypedDict,
    CustomRoutingStrategyBase,
    Deployment,
    LiteLLM_Params,
    ModelGroupInfo,
    RetryPolicy,
)

_CORE_SIGNATURE: Final = inspect.signature(PythonCore.__init__)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # reflection reads only the parameters' names


def router_arguments(
    *args: object,
    **kwargs: object,  # kwargs-ok: Router's constructor arguments, as passed
) -> Mapping[str, object]:
    """The constructor arguments as passed, by name (positional ones bound to their parameter)."""
    try:
        bound: Final = _CORE_SIGNATURE.bind_partial(None, *args, **kwargs).arguments
    except TypeError as error:
        raise TypeError(f"Router.__init__() {error}") from None
    return MappingProxyType({name: value for name, value in bound.items() if name != "self"})


def _class_descriptor(cls: type, name: str) -> bool:
    return any(hasattr(vars(base).get(name), "__set__") for base in cls.__mro__ if name in vars(base))


if TYPE_CHECKING:

    class Router(PythonCore, DeprecatedRouterMembers):
        @classmethod
        def from_config(cls, config: RouterConfig, host: PythonHost = python_host) -> "Router": ...

else:
    _MISSING: Final = object()
    _replaced_core_members: Final[  # mutable-ok: originals of patched core members, restored on delete
        dict[tuple[type, str], list[object]]
    ] = {}

    class _RouterType(type):
        """
        Class-level reads of core members (``Router._helper(...)``) go to ``PythonCore``, and so do
        class-level patches, which put the original back when the patch is removed.
        ``isinstance`` and ``dir`` see ``PythonCore`` too.
        """

        def __getattr__(cls, name: str) -> object:
            return getattr(PythonCore, name)

        def __setattr__(cls, name: str, value: object) -> None:
            if any(name in vars(base) for base in cls.__mro__) or not hasattr(PythonCore, name):
                super().__setattr__(name, value)
                return
            _replaced_core_members.setdefault((PythonCore, name), []).append(vars(PythonCore).get(name, _MISSING))
            setattr(PythonCore, name, value)

        def __delattr__(cls, name: str) -> None:
            replaced: Final = _replaced_core_members.get((PythonCore, name))
            if not replaced or any(name in vars(base) for base in cls.__mro__):
                super().__delattr__(name)
                return
            original: Final = replaced.pop()
            if original is _MISSING:
                delattr(PythonCore, name)
                return
            setattr(PythonCore, name, original)

        def __instancecheck__(cls, instance: object) -> bool:
            return type.__instancecheck__(cls, instance) or isinstance(instance, PythonCore)

        def __dir__(cls) -> list[str]:  # mutable-ok: dir() returns a list
            return sorted({*type.__dir__(cls), *dir(PythonCore)})

    def _patchable_method(core: object, name: str) -> bool:
        """A method of the core: an instance-level replacement of it is also kept on the facade, so
        ``mock.patch.object(router, name)`` finds it there and nested patches unwind in order. A core
        that serves its members from another core (``__getattr__``) is asked for the bound method."""
        static: Final = inspect.getattr_static(type(core), name, None)
        if static is not None:
            return isinstance(static, FunctionType | staticmethod | classmethod)
        return hasattr(type(core), "__getattr__") and inspect.ismethod(getattr(core, name, None))

    class Router(DeprecatedRouterMembers, metaclass=_RouterType):
        def __init_subclass__(cls, **kwargs: object) -> None:  # kwargs-ok: class keywords, refused with the class
            raise TypeError(
                "litellm.Router cannot be subclassed. Use the router's entry points instead: "
                "register_plugin() for custom routing, apply() for settings, observe() for outcomes. "
                "See https://docs.litellm.ai/docs/routing_plugins."
            )

        def __init__(self, *args: object, **kwargs: object) -> None:  # kwargs-ok: Router's constructor arguments
            self._attach(make_core(build_router_config(router_arguments(*args, **kwargs)), python_host))

        @classmethod
        def from_config(cls, config: RouterConfig, host: PythonHost = python_host) -> "Router":
            router: Final = cls.__new__(cls)
            router._attach(make_core(config, host))
            return router

        def _attach(self, core: RouterCore) -> None:
            pending: Final = self.__dict__.pop("_pending", {})
            self.__dict__["_core"] = core
            core.__dict__["_core_owner"] = weakref.ref(self)  # rebind-ok: the facade records itself on the core it owns
            for name, value in pending.items():
                setattr(core, name, value)

        @property
        def _core(self) -> RouterCore:
            core: Final = self.__dict__.get("_core")
            if core is None:
                raise AttributeError("_core")
            return core

        def __getattribute__(self, name: str) -> object:
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                pass
            facade: Final = object.__getattribute__(self, "__dict__")
            core: Final = facade.get("_core")
            if core is None:
                pending: Final = facade.get("_pending", {})
                if name in pending:
                    return pending[name]
                raise AttributeError(name)
            try:
                return getattr(core, name)
            except AttributeError as error:
                if error.obj is not core or error.name != name:
                    raise
                # What reading an attribute the router lacks raised before: it names the Router.
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'", name=name, obj=self
                ) from None

        def __setattr__(self, name: str, value: object) -> None:
            if _class_descriptor(type(self), name):
                object.__setattr__(self, name, value)
                return
            if "_core" not in self.__dict__:
                self.__dict__.setdefault("_pending", {})[name] = value
                return
            core: Final = self._core
            mirrored: Final = name in self.__dict__ or _patchable_method(core, name)
            setattr(core, name, value)
            if mirrored:
                self.__dict__[name] = value

        def __delattr__(self, name: str) -> None:
            if _class_descriptor(type(self), name):
                object.__delattr__(self, name)
                return
            self.__dict__.pop(name, None)
            delattr(self._core, name)

        def __dir__(self) -> list[str]:  # mutable-ok: dir() returns a list
            return sorted({*super().__dir__(), *dir(self.__dict__.get("_core", PythonCore))})

    Router.__init__.__signature__ = _CORE_SIGNATURE.replace(  # pyright: ignore[reportFunctionMemberAccess]  # inspect reads it
        parameters=[parameter for parameter in _CORE_SIGNATURE.parameters.values() if parameter.name != "_host"]
    )
    Router.__init__.__doc__ = PythonCore.__init__.__doc__


def __getattr__(name: str) -> object:
    """Module-level helpers that lived in this module before the core moved out."""
    return getattr(router_python_core, name)


__all__ = [
    "AllowedFailsPolicy",
    "AssistantsTypedDict",
    "CustomRoutingStrategyBase",
    "Deployment",
    "LiteLLM_Params",
    "ModelGroupInfo",
    "RetryPolicy",
    "Router",
    "as_output_cap",
    "model_info_is_active_for_environment",
    "router_arguments",
    "stamped_client_ceiling",
]
