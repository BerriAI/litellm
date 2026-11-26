from typing import TypeVar

from typing_extensions import TypeIs  # noqa: TID251  # narrows a duck-typed value to the Protocol its caller reads

_Shape = TypeVar("_Shape")


def has_attributes(
    value: object,
    shape: type[_Shape],
    *names: str,
) -> TypeIs[_Shape]:  # guard-ok: hasattr sees what __getattr__ serves, which Protocol isinstance skips on 3.12+
    return all(hasattr(value, name) for name in names)


def require_attributes(value: object, shape: type[_Shape], *names: str) -> _Shape:
    if has_attributes(value, shape, *names):
        return value
    for name in names:
        getattr(value, name)
    raise AttributeError(f"{type(value).__name__!r} object lacks {names!r}")
