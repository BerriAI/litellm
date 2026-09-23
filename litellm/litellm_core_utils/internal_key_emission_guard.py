import threading
from collections.abc import Mapping
from typing import Final, cast

from litellm._logging import verbose_logger
from litellm.types.utils import all_litellm_params

OWNED_KEY_PREFIX: Final = "_litellm_"
OWNED_KEYS: Final = frozenset(key for key in ("litellm_params", *all_litellm_params) if key != "metadata")
OWNED_KEY_SCOPES: Final = frozenset(
    ("", "metadata", "additionalModelRequestFields", "additionalModelRequestFields.extra_body")
)
_CONTAINER_TYPES: Final[tuple[type[object], ...]] = (dict, list, tuple)


class LeakCounter:
    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self._value = 0

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def increment(self) -> None:
        with self._lock:
            self._value += 1


internal_key_leak_counter: Final = LeakCounter()


def is_litellm_owned_key(key: str, scope: str) -> bool:
    return key.startswith(OWNED_KEY_PREFIX) or (scope in OWNED_KEY_SCOPES and key in OWNED_KEYS)


def _walk_mapping(value: Mapping[object, object], prefix: str, found: list[str]) -> None:
    scoped: Final = prefix in OWNED_KEY_SCOPES
    for key, child in value.items():
        if not isinstance(key, str):
            continue
        if key.startswith(OWNED_KEY_PREFIX) or (scoped and key in OWNED_KEYS):
            found.append(f"{prefix}.{key}" if prefix else key)
        if isinstance(child, _CONTAINER_TYPES):
            _walk(cast(object, child), f"{prefix}.{key}" if prefix else key, found)  # cast-ok: undo Unknown narrowing


def _walk(value: object, prefix: str, found: list[str]) -> None:
    if isinstance(value, dict):
        _walk_mapping(cast("dict[object, object]", value), prefix, found)  # cast-ok: isinstance leaves Unknown params
        return
    if not isinstance(value, (list, tuple)):
        return
    for index, child in enumerate(cast("list[object] | tuple[object, ...]", value)):  # cast-ok: same as above
        if isinstance(child, _CONTAINER_TYPES):
            _walk(cast(object, child), f"{prefix}[{index}]", found)  # cast-ok: undo Unknown narrowing


def litellm_owned_key_paths(body: Mapping[str, object]) -> tuple[str, ...]:
    found: Final[list[str]] = []  # mutable-ok: single accumulator for the hot-path walk, sealed into a tuple below
    _walk_mapping(cast("Mapping[object, object]", body), "", found)  # cast-ok: widen invariant key type
    return tuple(found)


def observe_internal_keys(body: Mapping[str, object], provider: str) -> tuple[str, ...]:
    paths: Final = litellm_owned_key_paths(body)
    if paths:
        internal_key_leak_counter.increment()
        verbose_logger.warning(
            "LiteLLM internal keys reached the %s provider request body: %s", provider, ", ".join(paths)
        )
    return paths
