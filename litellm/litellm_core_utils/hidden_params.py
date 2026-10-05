from typing import Final

_HIDDEN_PARAMS_ATTR: Final = "_hidden_params"
HIDDEN_PARAMS_ATTR: Final = _HIDDEN_PARAMS_ATTR


def get_hidden_params(obj: object) -> dict[str, object] | None:
    hidden_params: Final = (
        obj.get(_HIDDEN_PARAMS_ATTR) if isinstance(obj, dict) else getattr(obj, _HIDDEN_PARAMS_ATTR, None)
    )
    return hidden_params if isinstance(hidden_params, dict) else None


def set_hidden_params(obj: object, hidden_params: dict[str, object]) -> None:
    if isinstance(obj, dict):
        obj[_HIDDEN_PARAMS_ATTR] = hidden_params
    else:
        setattr(obj, _HIDDEN_PARAMS_ATTR, hidden_params)


def get_or_create_hidden_params(obj: object) -> dict[str, object]:
    hidden_params: Final = get_hidden_params(obj)
    if hidden_params is not None:
        return hidden_params
    created_hidden_params: Final[dict[str, object]] = {}
    set_hidden_params(obj, created_hidden_params)
    return created_hidden_params
