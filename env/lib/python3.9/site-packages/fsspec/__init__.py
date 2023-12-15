from importlib.metadata import entry_points

from . import _version, caching
from .callbacks import Callback
from .compression import available_compressions
from .core import get_fs_token_paths, open, open_files, open_local
from .exceptions import FSTimeoutError
from .mapping import FSMap, get_mapper
from .registry import (
    available_protocols,
    filesystem,
    get_filesystem_class,
    register_implementation,
    registry,
)
from .spec import AbstractFileSystem

__version__ = _version.get_versions()["version"]

__all__ = [
    "AbstractFileSystem",
    "FSTimeoutError",
    "FSMap",
    "filesystem",
    "register_implementation",
    "get_filesystem_class",
    "get_fs_token_paths",
    "get_mapper",
    "open",
    "open_files",
    "open_local",
    "registry",
    "caching",
    "Callback",
    "available_protocols",
    "available_compressions",
]


def process_entries():
    if entry_points is not None:
        try:
            eps = entry_points()
        except TypeError:
            pass  # importlib-metadata < 0.8
        else:
            if hasattr(eps, "select"):  # Python 3.10+ / importlib_metadata >= 3.9.0
                specs = eps.select(group="fsspec.specs")
            else:
                specs = eps.get("fsspec.specs", [])
            for spec in specs:
                err_msg = f"Unable to load filesystem from {spec}"
                register_implementation(
                    spec.name,
                    spec.value.replace(":", "."),
                    errtxt=err_msg,
                    # We take our implementations as the ones to overload with if
                    # for some reason we encounter some, may be the same, already
                    # registered
                    clobber=True,
                )


process_entries()
