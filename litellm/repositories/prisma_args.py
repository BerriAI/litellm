"""Prisma's query builder serializes plain ``dict``/``list`` values only, so read-only
mappings and sequences are converted here instead of at every call site."""

from collections.abc import Mapping, Sequence


def prisma_args(fields: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: prisma requires a plain dict
    return dict(fields)  # mutable-ok: prisma's query builder rejects read-only mappings


def prisma_str_list(values: Sequence[str]) -> list[str]:  # mutable-ok: prisma requires a plain list
    return list(values)  # mutable-ok: prisma serializes String[] columns from plain lists
