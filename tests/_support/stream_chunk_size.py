from collections.abc import Mapping


def keys_at_every_depth(value: object) -> frozenset[str]:
    if isinstance(value, Mapping):
        return frozenset(value) | frozenset().union(*(keys_at_every_depth(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return frozenset().union(*(keys_at_every_depth(item) for item in value))
    return frozenset()
