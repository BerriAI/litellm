"""Shared primitive for resolving a settings value from its sources.

A ``FieldDescriptor`` names, for one setting, where it lives in the stored DB
row (``db_key``), which process env var carries it (``env_var``), whether it is
a secret, and its effective default. ``resolve_fields`` reconciles a set of
descriptors against a decrypted DB row and the process environment with a fixed
precedence, returning the resolved values plus per-field provenance so a caller
can tell whether a value came from the database, the environment, a default, or
is unset.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

FieldSource = Literal["db", "env", "default", "unset"]


@dataclass(frozen=True, slots=True)
class FieldDescriptor:
    field_name: str
    db_key: str
    env_var: str
    is_secret: bool = False
    default: str | None = None


def _db_is_set(db_value: object, empty_db_is_set: bool) -> bool:
    if empty_db_is_set:
        # A stored key that is present, even as "", is an explicit admin choice
        # (e.g. clearing an alerting webhook) and must win over a stale env var.
        return db_value is not None
    # A blank stored value is treated as absent, so it falls through to env. This
    # fits settings whose clear path also unsets the env var (e.g. SSO).
    return isinstance(db_value, str) and bool(db_value.strip())


def _resolve_one(
    descriptor: FieldDescriptor,
    db_values: Mapping[str, object],
    env: Mapping[str, str],
    empty_db_is_set: bool,
) -> tuple[str, str | None, FieldSource]:
    db_value = db_values.get(descriptor.db_key)
    if _db_is_set(db_value, empty_db_is_set):
        return descriptor.field_name, db_value if isinstance(db_value, str) else str(db_value), "db"
    env_value = env.get(descriptor.env_var)
    if isinstance(env_value, str) and env_value.strip():
        return descriptor.field_name, env_value, "env"
    if descriptor.default is not None:
        return descriptor.field_name, descriptor.default, "default"
    return descriptor.field_name, None, "unset"


def resolve_fields(
    descriptors: Sequence[FieldDescriptor],
    db_values: Mapping[str, object],
    env: Mapping[str, str],
    empty_db_is_set: bool = False,
) -> tuple[dict[str, str | None], dict[str, FieldSource]]:
    """Resolve every descriptor to (values, provenance).

    Precedence per field: a set stored value wins, else a non-blank process env
    var, else the descriptor default, else unset. ``empty_db_is_set`` selects
    how a present-but-empty stored value is read: ``False`` treats it as absent
    so it falls back to env (SSO, whose clear path also unsets the env var);
    ``True`` treats it as an explicit clear that wins over env (alerting, whose
    clear path stores "" without unsetting the env var).
    """
    resolved = tuple(_resolve_one(descriptor, db_values, env, empty_db_is_set) for descriptor in descriptors)
    values = {field_name: value for field_name, value, _ in resolved}
    provenance = {field_name: source for field_name, _, source in resolved}
    return values, provenance
