import threading
from collections.abc import Mapping, Sequence
from typing import Final, cast

from litellm._logging import verbose_logger
from litellm.types.utils import all_litellm_params

OWNED_KEY_PREFIX: Final = "_litellm_"
OWNED_KEYS: Final = frozenset(key for key in ("litellm_params", *all_litellm_params) if key != "metadata")
OWNED_KEY_SCOPES: Final = frozenset(
    ("", "metadata", "additionalModelRequestFields", "additionalModelRequestFields.extra_body")
)


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


def _as_mapping(value: object) -> Mapping[object, object] | None:
    if not isinstance(value, Mapping):
        return None
    return cast("Mapping[object, object]", value)  # cast-ok: isinstance narrows only to Mapping[Unknown, Unknown]


def _as_sequence(value: object) -> Sequence[object] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    return cast("Sequence[object]", value)  # cast-ok: isinstance narrows only to Sequence[Unknown]


def _child_paths(value: object, prefix: str) -> tuple[str, ...]:
    mapping: Final = _as_mapping(value)
    if mapping is not None:
        return tuple(path for key, child in mapping.items() for path in _entry_paths(key, child, prefix))
    sequence: Final = _as_sequence(value)
    if sequence is not None:
        return tuple(path for index, child in enumerate(sequence) for path in _child_paths(child, f"{prefix}[{index}]"))
    return ()


def _entry_paths(key: object, child: object, prefix: str) -> tuple[str, ...]:
    name: Final = str(key)
    path: Final = f"{prefix}.{name}" if prefix else name
    own: Final = (path,) if is_litellm_owned_key(name, prefix) else ()
    return (*own, *_child_paths(child, path))


def litellm_owned_key_paths(body: Mapping[str, object]) -> tuple[str, ...]:
    return _child_paths(body, "")


def observe_internal_keys(body: Mapping[str, object], provider: str) -> tuple[str, ...]:
    paths: Final = litellm_owned_key_paths(body)
    if paths:
        internal_key_leak_counter.increment()
        verbose_logger.warning(
            "LiteLLM internal keys reached the %s provider request body: %s", provider, ", ".join(paths)
        )
    return paths
