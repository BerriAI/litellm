from typing import Final, Protocol, cast

_HIDDEN_PARAMS_ATTR: Final = "_hidden_params"
HIDDEN_PARAMS_ATTR: Final = _HIDDEN_PARAMS_ATTR


class _SupportsItemAssignment(Protocol):
    def __setitem__(self, key: str, value: object) -> None: ...


def _get_hidden_params_storage(obj: object) -> object | None:
    return obj.get(_HIDDEN_PARAMS_ATTR) if isinstance(obj, dict) else getattr(obj, _HIDDEN_PARAMS_ATTR, None)


def get_hidden_params(obj: object) -> dict[str, object] | None:
    hidden_params: Final[object | None] = _get_hidden_params_storage(obj)
    if isinstance(hidden_params, dict):
        return cast(  # cast-ok: runtime dict validation preserves dynamically typed legacy storage
            dict[str, object], hidden_params
        )
    return None


def set_hidden_params(obj: object, hidden_params: dict[str, object]) -> None:
    if isinstance(obj, dict):
        obj[_HIDDEN_PARAMS_ATTR] = hidden_params
    else:
        setattr(obj, _HIDDEN_PARAMS_ATTR, hidden_params)


def set_hidden_param(obj: object, key: str, value: object) -> None:
    hidden_params: Final[object | None] = _get_hidden_params_storage(obj)
    if hidden_params is None:
        set_hidden_params(obj, {key: value})
        return
    if isinstance(hidden_params, dict):
        cast(  # cast-ok: runtime dict validation preserves dynamically typed legacy storage
            dict[str, object], hidden_params
        )[key] = value
        return
    if hasattr(hidden_params, "__setitem__"):
        cast(  # cast-ok: hasattr validates the legacy storage assignment protocol
            _SupportsItemAssignment, hidden_params
        )[key] = value
        return
    raise TypeError(f"unsupported hidden params storage: {type(hidden_params).__name__}")


def get_or_create_hidden_params(obj: object) -> dict[str, object]:
    hidden_params: Final[object | None] = _get_hidden_params_storage(obj)
    if hidden_params is None:
        created_hidden_params: Final[dict[str, object]] = {}
        set_hidden_params(obj, created_hidden_params)
        return created_hidden_params
    if isinstance(hidden_params, dict):
        return cast(  # cast-ok: runtime dict validation preserves dynamically typed legacy storage
            dict[str, object], hidden_params
        )
    raise TypeError(f"unsupported hidden params storage: {type(hidden_params).__name__}")
