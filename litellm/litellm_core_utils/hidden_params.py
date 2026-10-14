from collections.abc import Iterator, MutableMapping
from typing import Final, Protocol, cast

from litellm.types.llms.base import HiddenParams

_HIDDEN_PARAMS_ATTR: Final = "_hidden_params"
HIDDEN_PARAMS_ATTR: Final = _HIDDEN_PARAMS_ATTR


class _SupportsItemAssignment(Protocol):
    def __setitem__(self, key: str, value: object) -> None: ...


class HiddenParamsModelView(MutableMapping[str, object]):
    def __init__(self, hidden_params: HiddenParams) -> None:
        self._hidden_params = hidden_params

    def __getitem__(self, key: str) -> object:
        fields: Final = self._keys()
        if key not in fields:
            raise KeyError(key)
        try:
            return cast(object, getattr(self._hidden_params, key))
        except AttributeError as error:
            raise KeyError(key) from error

    def __setitem__(self, key: str, value: object) -> None:
        setattr(self._hidden_params, key, value)

    def __delitem__(self, key: str) -> None:
        fields: Final = self._keys()
        if key not in fields:
            raise KeyError(key)
        try:
            delattr(self._hidden_params, key)
        except AttributeError as error:
            raise KeyError(key) from error
        self._hidden_params.model_fields_set.discard(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys())

    def __len__(self) -> int:
        return len(self._keys())

    def _keys(self) -> frozenset[str]:
        return frozenset(self._hidden_params.model_fields_set) | frozenset(self._hidden_params.model_extra or {})


def _get_hidden_params_storage(obj: object) -> object | None:
    return obj.get(_HIDDEN_PARAMS_ATTR) if isinstance(obj, dict) else getattr(obj, _HIDDEN_PARAMS_ATTR, None)


def get_hidden_params_storage(obj: object) -> dict[str, object] | HiddenParams | None:
    hidden_params: Final[object | None] = _get_hidden_params_storage(obj)
    if isinstance(hidden_params, dict):
        return cast(  # cast-ok: runtime dict validation preserves dynamically typed legacy storage
            dict[str, object], hidden_params
        )
    if isinstance(hidden_params, HiddenParams):
        return hidden_params
    return None


def _as_hidden_params_mapping(hidden_params: object | None) -> MutableMapping[str, object] | None:
    if isinstance(hidden_params, dict):
        return cast(  # cast-ok: runtime dict validation preserves dynamically typed legacy storage
            dict[str, object], hidden_params
        )
    if isinstance(hidden_params, HiddenParams):
        return HiddenParamsModelView(hidden_params)
    return None


def get_hidden_params(obj: object) -> MutableMapping[str, object] | None:
    hidden_params: Final[object | None] = _get_hidden_params_storage(obj)
    return _as_hidden_params_mapping(hidden_params)


def set_hidden_params(obj: object, hidden_params: dict[str, object] | HiddenParams) -> None:
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


def get_or_create_hidden_params(obj: object) -> MutableMapping[str, object]:
    hidden_params: Final[object | None] = _get_hidden_params_storage(obj)
    if hidden_params is None:
        created_hidden_params: Final[dict[str, object]] = {}
        set_hidden_params(obj, created_hidden_params)
        return created_hidden_params
    hidden_params_mapping: Final[MutableMapping[str, object] | None] = _as_hidden_params_mapping(hidden_params)
    if hidden_params_mapping is not None:
        return hidden_params_mapping
    raise TypeError(f"unsupported hidden params storage: {type(hidden_params).__name__}")
