from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


_SENTINEL = object()


class LRUCache:
    def __init__(self, max_size):
        # type: (int) -> None
        if max_size <= 0:
            raise AssertionError(f"invalid max_size: {max_size}")
        self.max_size = max_size
        self._data = {}  # type: dict[Any, Any]
        self.hits = self.misses = 0
        self.full = False

    def set(self, key, value):
        # type: (Any, Any) -> None
        current = self._data.pop(key, _SENTINEL)
        if current is not _SENTINEL:
            self._data[key] = value
        elif self.full:
            self._data.pop(next(iter(self._data)))
            self._data[key] = value
        else:
            self._data[key] = value
        self.full = len(self._data) >= self.max_size

    def get(self, key, default=None):
        # type: (Any, Any) -> Any
        try:
            ret = self._data.pop(key)
        except KeyError:
            self.misses += 1
            ret = default
        else:
            self.hits += 1
            self._data[key] = ret

        return ret

    def get_all(self):
        # type: () -> list[tuple[Any, Any]]
        return list(self._data.items())
