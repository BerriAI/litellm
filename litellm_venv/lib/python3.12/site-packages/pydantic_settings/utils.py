import sys
import types
from pathlib import Path
from typing import Any, _GenericAlias  # type: ignore [attr-defined]

from typing_extensions import get_origin

_PATH_TYPE_LABELS = {
    Path.is_dir: "directory",
    Path.is_file: "file",
    Path.is_mount: "mount point",
    Path.is_symlink: "symlink",
    Path.is_block_device: "block device",
    Path.is_char_device: "char device",
    Path.is_fifo: "FIFO",
    Path.is_socket: "socket",
}


def path_type_label(p: Path) -> str:
    """
    Find out what sort of thing a path is.
    """
    assert p.exists(), "path does not exist"
    for method, name in _PATH_TYPE_LABELS.items():
        if method(p):
            return name

    return "unknown"  # pragma: no cover


# TODO remove and replace usage by `isinstance(cls, type) and issubclass(cls, class_or_tuple)`
# once we drop support for Python 3.10.
def _lenient_issubclass(cls: Any, class_or_tuple: Any) -> bool:  # pragma: no cover
    try:
        return isinstance(cls, type) and issubclass(cls, class_or_tuple)
    except TypeError:
        if get_origin(cls) is not None:
            # Up until Python 3.10, isinstance(<generic_alias>, type) is True
            # (e.g. list[int])
            return False
        raise


if sys.version_info < (3, 10):
    _WithArgsTypes = tuple()
else:
    _WithArgsTypes = (_GenericAlias, types.GenericAlias, types.UnionType)
