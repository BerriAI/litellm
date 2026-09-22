from collections.abc import Mapping, Sequence
from typing import Final

from litellm._logging import verbose_logger
from litellm.types.utils import all_litellm_params

OWNED_KEY_PREFIX: Final = "_litellm_"
OWNED_KEYS: Final = frozenset(key for key in ("litellm_params", *all_litellm_params) if key != "metadata")


class LeakCounter:
    def __init__(self) -> None:
        self._value = 0

    @property
    def value(self) -> int:
        return self._value

    def increment(self) -> None:
        self._value += 1


internal_key_leak_counter: Final = LeakCounter()


def is_litellm_owned_key(key: str) -> bool:
    return key in OWNED_KEYS or key.startswith(OWNED_KEY_PREFIX)


def _child_paths(value: object, prefix: str) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(path for key, child in value.items() for path in _entry_paths(key, child, prefix))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(path for index, child in enumerate(value) for path in _child_paths(child, f"{prefix}[{index}]"))
    return ()


def _entry_paths(key: object, child: object, prefix: str) -> tuple[str, ...]:
    name: Final = str(key)
    path: Final = f"{prefix}.{name}" if prefix else name
    own: Final = (path,) if is_litellm_owned_key(name) else ()
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
