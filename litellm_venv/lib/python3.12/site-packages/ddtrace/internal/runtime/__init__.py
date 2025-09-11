import typing as t
import uuid

from .. import forksafe


__all__ = [
    "get_runtime_id",
]


def _generate_runtime_id() -> str:
    return uuid.uuid4().hex


_RUNTIME_ID: str = _generate_runtime_id()
_ANCESTOR_RUNTIME_ID: t.Optional[str] = None
_ON_RUNTIME_ID_CHANGE: t.Set[t.Callable[[str], None]] = set()


def on_runtime_id_change(cb: t.Callable[[str], None]) -> None:
    """Register a callback to be called when the runtime ID changes.

    This can happen after a fork.
    """
    global _ON_RUNTIME_ID_CHANGE
    _ON_RUNTIME_ID_CHANGE.add(cb)


@forksafe.register
def _set_runtime_id():
    global _RUNTIME_ID, _ANCESTOR_RUNTIME_ID

    # Save the runtime ID of the common ancestor of all processes.
    if _ANCESTOR_RUNTIME_ID is None:
        _ANCESTOR_RUNTIME_ID = _RUNTIME_ID

    _RUNTIME_ID = _generate_runtime_id()
    for cb in _ON_RUNTIME_ID_CHANGE:
        cb(_RUNTIME_ID)


def get_runtime_id():
    """Return a unique string identifier for this runtime.

    Do not store this identifier as it can change when, e.g., the process forks.
    """
    return _RUNTIME_ID


def get_ancestor_runtime_id() -> t.Optional[str]:
    """Return the runtime ID of the common ancestor of this process.

    Once this value is set (this will happen after a fork) it will not change
    for the lifetime of the process. This function returns ``None`` for the
    ancestor process.
    """
    return _ANCESTOR_RUNTIME_ID
