import sys

if sys.version_info < (3, 6):  # noqa: UP036
    msg = "pybind11 does not support Python < 3.6. 2.9 was the last release supporting Python 2.7 and 3.5."
    raise ImportError(msg)


from ._version import __version__, version_info
from .commands import get_cmake_dir, get_include, get_pkgconfig_dir

__all__ = (
    "version_info",
    "__version__",
    "get_include",
    "get_cmake_dir",
    "get_pkgconfig_dir",
)
