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
from types import CellType, FunctionType, MappingProxyType
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

_CORE_SIGNATURE: Final = inspect.signature(PythonCore.__init__)


def router_arguments(
    *args: object, **kwargs: object
) -> Mapping[str, object]:  # kwargs-ok: Router's constructor arguments, as passed
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
    _replaced_core_members: Final[
        dict[tuple[type, str], list[object]]
    ] = {}  # mutable-ok: originals of patched core members, restored on delete
    _FACADE_ONLY: Final = frozenset(
        {
            "__module__",
            "__qualname__",
            "__doc__",
            "__dict__",
            "__weakref__",
            "__init__",
            "__init_subclass__",
            "__annotations__",
            "__firstlineno__",
            "__static_attributes__",
            "__classcell__",
            "_core_class",
        }
    )

    def _with_class_cell(value: object, cell: "CellType") -> object:
        """``value`` with its ``__class__`` cell (what zero-argument ``super()`` reads) replaced by ``cell``."""
        if isinstance(value, staticmethod | classmethod):
            return type(value)(_with_class_cell(value.__func__, cell))
        if isinstance(value, property):
            return property(
                *(None if f is None else _with_class_cell(f, cell) for f in (value.fget, value.fset, value.fdel)),
                value.__doc__,
            )
        if not isinstance(value, FunctionType) or "__class__" not in value.__code__.co_freevars:
            return value
        closure: Final = tuple(
            cell if name == "__class__" else original
            for name, original in zip(value.__code__.co_freevars, value.__closure__ or (), strict=True)
        )
        rebound: Final = FunctionType(value.__code__, value.__globals__, value.__name__, value.__defaults__, closure)
        rebound.__kwdefaults__ = value.__kwdefaults__
        rebound.__qualname__ = value.__qualname__
        rebound.__dict__.update(value.__dict__)
        return rebound

    class _Absent:
        """A facade-only member, hidden from the core class: reading it raises the usual AttributeError."""

        def __set_name__(self, owner: type, name: str) -> None:
            self._name = name

        def __get__(self, instance: object, owner: type | None = None) -> object:
            if instance is None:
                raise AttributeError(
                    f"type object {getattr(owner, '__name__', owner)!r} has no attribute {self._name!r}"
                )
            raise AttributeError(f"{type(instance).__name__!r} object has no attribute {self._name!r}")

    def _core_class_for(facade: type, parent: type) -> type:
        """
        A core class carrying a Router subclass's own members, so the core runs its overrides. It
        subclasses the facade too, so an override's ``super(Sub, self)`` finds the core it runs on.
        """
        cell: Final = CellType()
        members: Final = {
            name: _with_class_cell(value, cell) for name, value in vars(facade).items() if name not in _FACADE_ONLY
        }
        namespace: Final = {
            # The facade's own plumbing sits in the core class's MRO ahead of the parent core: put
            # the parent core's versions back, and hide the facade-only members.
            "__init__": parent.__init__,
            "__getattribute__": object.__getattribute__,
            "__setattr__": parent.__setattr__,
            "__delattr__": parent.__delattr__,
            "__dir__": parent.__dir__,
            **{name: _Absent() for name in ("_attach", "_core", "from_config", "_core_class")},
            **members,
            "__module__": facade.__module__,
        }
        core_class: Final = _CoreType(f"{facade.__name__}Core", (facade, parent), namespace)
        cell.cell_contents = core_class
        for name in members:
            type.__delattr__(facade, name)
        return core_class

    class _RouterType(type):
        """
        Class-level reads of core members (``Router._helper(...)``) go to the router's core class, and
        so do class-level patches, which put the original back when the patch is removed.
        ``isinstance`` and ``dir`` see the core class too.
        """

        def __getattr__(cls, name: str) -> object:
            if name == "_core_class":
                raise AttributeError(name)
            return getattr(cls._core_class, name)

        def __setattr__(cls, name: str, value: object) -> None:
            core_class: Final = vars(cls).get("_core_class") or getattr(cls, "_core_class", None)
            if core_class is None or any(name in vars(base) for base in cls.__mro__) or not hasattr(core_class, name):
                super().__setattr__(name, value)
                return
            _replaced_core_members.setdefault((core_class, name), []).append(vars(core_class).get(name, _MISSING))
            setattr(core_class, name, value)

        def __delattr__(cls, name: str) -> None:
            core_class: Final = getattr(cls, "_core_class", None)
            replaced: Final = _replaced_core_members.get((core_class, name)) if core_class is not None else None
            if not replaced or any(name in vars(base) for base in cls.__mro__):
                super().__delattr__(name)
                return
            original: Final = replaced.pop()
            if original is _MISSING:
                delattr(core_class, name)
                return
            setattr(core_class, name, original)

        def __instancecheck__(cls, instance: object) -> bool:
            return type.__instancecheck__(cls, instance) or isinstance(instance, cls._core_class)

        def __dir__(cls) -> list[str]:  # mutable-ok: dir() returns a list
            return sorted({*type.__dir__(cls), *dir(cls._core_class)})

    class _CoreType(_RouterType):
        """The metaclass of a Router subclass's core class: a plain class, none of the facade's redirects."""

        def __getattr__(cls, name: str) -> object:
            raise AttributeError(f"type object {cls.__name__!r} has no attribute {name!r}")

        __setattr__ = type.__setattr__
        __delattr__ = type.__delattr__
        __instancecheck__ = type.__instancecheck__
        __dir__ = type.__dir__

    def _is_core(instance: object) -> bool:
        return isinstance(type(instance), _CoreType)

    def _patchable_method(core: object, name: str) -> bool:
        """A method of the core: an instance-level replacement of it is also kept on the facade, so
        ``mock.patch.object(router, name)`` finds it there and nested patches unwind in order. A core
        that serves its members from another core (``__getattr__``) is asked for the bound method."""
        static: Final = inspect.getattr_static(type(core), name, None)
        if static is not None:
            return isinstance(static, FunctionType | staticmethod | classmethod)
        return hasattr(type(core), "__getattr__") and inspect.ismethod(getattr(core, name, None))

    class Router(DeprecatedRouterMembers, metaclass=_RouterType):
        _core_class: type[PythonCore] = PythonCore

        def __init_subclass__(cls, **kwargs: object) -> None:  # kwargs-ok: class keywords, passed on to super()
            if isinstance(cls, _CoreType):
                # A core class built below: the parent core's hooks wire it, as for any core class.
                super().__init_subclass__(**kwargs)
                return
            # DeprecatedRouterMembers wires a subclass's overrides of old members; here that runs
            # on the core class built from it, which is where the router's own calls land.
            super(DeprecatedRouterMembers, cls).__init_subclass__(**kwargs)
            type.__setattr__(cls, "_core_class", _core_class_for(cls, cls._core_class))

        def __init__(self, *args: object, **kwargs: object) -> None:  # kwargs-ok: Router's constructor arguments
            self._attach(
                make_core(build_router_config(router_arguments(*args, **kwargs)), python_host, type(self)._core_class)
            )

        @classmethod
        def from_config(cls, config: RouterConfig, host: PythonHost = python_host) -> "Router":
            router: Final = cls.__new__(cls)
            router._attach(make_core(config, host, cls._core_class))
            return router

        def _attach(self, core: RouterCore) -> None:
            pending: Final = self.__dict__.pop("_pending", {})
            self.__dict__["_core"] = core
            core.__dict__["_core_owner"] = weakref.ref(self)
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
                if _is_core(self):
                    raise
            facade: Final = object.__getattribute__(self, "__dict__")
            core: Final = facade.get("_core")
            if core is None:
                pending: Final = facade.get("_pending", {})
                if name in pending:
                    return pending[name]
                raise AttributeError(name)
            return getattr(core, name)

        def __setattr__(self, name: str, value: object) -> None:
            if _is_core(self):
                super().__setattr__(name, value)
                return
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
            if _is_core(self):
                super().__delattr__(name)
                return
            if _class_descriptor(type(self), name):
                object.__delattr__(self, name)
                return
            self.__dict__.pop(name, None)
            delattr(self._core, name)

        def __dir__(self) -> list[str]:  # mutable-ok: dir() returns a list
            if _is_core(self):
                return super().__dir__()
            return sorted({*super().__dir__(), *dir(self.__dict__.get("_core", type(self)._core_class))})

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
