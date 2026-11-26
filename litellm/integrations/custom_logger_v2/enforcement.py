from collections.abc import Mapping
from typing import Final, Protocol

from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.hooks import HOOK_SPECS, Subscription
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch, Reroute, Respond


class ContractViolation(Exception):
    pass


class UndeclaredReadError(ContractViolation, AttributeError):
    pass


def _field_named(name: str) -> Field | None:
    member: Final = Field._value2member_map_.get(name)
    return member if isinstance(member, Field) else None


class FieldView:
    __slots__ = ("_fields",)

    def __init__(self, fields: Mapping[Field, object]) -> None:
        self._fields = fields

    def __getattr__(self, name: str) -> object:
        field: Final = _field_named(name)
        if field is None or field not in self._fields:
            raise AttributeError(name)
        return self._fields[field]


class DeclaredView:
    __slots__ = ("_fields", "_reads")

    def __init__(self, fields: Mapping[Field, object], reads: frozenset[Field]) -> None:
        self._fields = fields
        self._reads = reads

    def __getattr__(self, name: str) -> object:
        field: Final = _field_named(name)
        if field is None or field not in self._reads:
            raise UndeclaredReadError(f"{name} is not among the declared reads")
        if field not in self._fields:
            raise AttributeError(name)
        return self._fields[field]


class Policy(Protocol):
    def view(self, subscription: Subscription, fields: Mapping[Field, object]) -> object: ...

    def check(self, subscription: Subscription, result: object) -> None: ...


class Unchecked:
    def view(self, subscription: Subscription, fields: Mapping[Field, object]) -> object:
        return FieldView(fields)

    def check(self, subscription: Subscription, result: object) -> None:
        return None


def _outcome_problem(hook: Hook, subscription: Subscription, outcome: Abort | Respond | Reroute) -> str | None:
    if not subscription.may_abort:
        return f"returned {type(outcome).__name__} without may_abort"
    if type(outcome) not in HOOK_SPECS[hook].outcomes:
        return f"returned {type(outcome).__name__}, which {hook.value} does not allow"
    return None


def _result_problem(subscription: Subscription, result: object) -> str | None:
    match result:
        case None:
            return None
        case Patch():
            undeclared: Final = result.fields - subscription.writes
            if not undeclared:
                return None
            return f"patched {', '.join(sorted(field.value for field in undeclared))} outside its declared writes"
        case Abort() | Respond() | Reroute():
            return _outcome_problem(subscription.hook, subscription, result)
        case _:
            return f"returned {type(result).__name__}, which is neither None, a Patch nor an outcome"


class Enforced:
    def view(self, subscription: Subscription, fields: Mapping[Field, object]) -> object:
        return DeclaredView(fields, subscription.reads)

    def check(self, subscription: Subscription, result: object) -> None:
        problem: Final = _result_problem(subscription, result)
        if problem is not None:
            raise ContractViolation(f"{subscription.handler} on {subscription.hook.value} {problem}")
