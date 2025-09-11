from __future__ import annotations

from ._compat import get_origin


def is_list_type(typ: type | None) -> bool:
    if typ is None:
        return False

    return (get_origin(typ) or typ) == list
