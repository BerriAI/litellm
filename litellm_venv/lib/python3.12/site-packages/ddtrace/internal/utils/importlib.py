from __future__ import absolute_import

from importlib import import_module
from types import TracebackType
from typing import Any  # noqa:F401
from typing import Callable  # noqa:F401
from typing import List  # noqa:F401
from typing import Optional
from typing import Type


class require_modules(object):
    """Context manager to check the availability of required modules."""

    def __init__(self, modules):
        # type: (List[str]) -> None
        self._missing_modules = []
        for module in modules:
            try:
                import_module(module)
            except ImportError:
                self._missing_modules.append(module)

    def __enter__(self):
        # type: () -> List[str]
        return self._missing_modules

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        return


def func_name(f):
    # type: (Callable[..., Any]) -> str
    """Return a human readable version of the function's name."""
    if hasattr(f, "__module__"):
        return "%s.%s" % (f.__module__, getattr(f, "__name__", f.__class__.__name__))
    return getattr(f, "__name__", f.__class__.__name__)


def module_name(instance):
    # type: (Any) -> str
    """Return the instance module name."""
    return instance.__class__.__module__.split(".")[0]
