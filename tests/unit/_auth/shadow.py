import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
from functools import reduce
from typing import Final

from litellm._auth import decision
from tests.unit._auth.reference_core import PythonCore
from litellm._auth.rust_core import (
    AUTH_STEP,
    MEMBER_STEP,
    OBJECT_MAPPING,
    OBJECT_TUPLE,
    CoreStep,
    RustCore,
    field_value,
)


class ShadowMismatch(RuntimeError):
    pass


def snapshot(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return (float, "nan")
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple(
                (field.name, snapshot(field_value(value, field.name)))
                for field in fields(value)
                if field.name != "resume"
            ),
        )
    if isinstance(value, Mapping):
        return tuple((key, snapshot(item)) for key, item in OBJECT_MAPPING.validate_python(value).items())
    if isinstance(value, (tuple, list)):
        return tuple(snapshot(item) for item in OBJECT_TUPLE.validate_python(value))
    return value


def compare(primary: object, reference: object) -> None:
    if snapshot(primary) != snapshot(reference):
        raise ShadowMismatch(f"auth shadow mismatch at {type(primary).__name__}")


@dataclass(frozen=True, slots=True)
class _RecordedEffect:
    signature: object
    answers: tuple[object, ...]


def _replay(reference: CoreStep, record: _RecordedEffect) -> CoreStep:
    if snapshot(reference) != record.signature or isinstance(reference, (decision.Allow, decision.Deny)):
        raise ShadowMismatch(f"auth shadow effect mismatch at {type(reference).__name__}")
    return _resume(reference.resume, record.answers)


def _resume(continuation: Callable[..., CoreStep], answers: tuple[object, ...]) -> CoreStep:
    return continuation(*answers)


def _shadow(
    primary: CoreStep, reference: Callable[[], CoreStep], records: tuple[_RecordedEffect, ...] = ()
) -> CoreStep:
    if isinstance(primary, (decision.Allow, decision.Deny)):
        compare(primary, reduce(_replay, records, reference()))
        return primary
    primary_resume: Final[Callable[..., CoreStep]] = primary.resume
    signature: Final = snapshot(primary)

    def resume(*recorded_answers: object) -> CoreStep:
        recorded: Final = _RecordedEffect(signature, recorded_answers)
        next_primary: Final = _resume(primary_resume, recorded_answers)
        return _shadow(next_primary, reference, (*records, recorded))

    return replace(primary, resume=resume)


class ShadowCore:
    def __init__(self, primary: decision.AuthCore | None = None, reference: decision.AuthCore | None = None) -> None:
        self.primary: Final = primary if primary is not None else RustCore()
        self.reference: Final = reference if reference is not None else PythonCore()

    def start(self, facts: decision.RequestFacts, settings: decision.Settings, now: datetime) -> decision.Step:
        return AUTH_STEP.validate_python(
            _shadow(self.primary.start(facts, settings, now), lambda: self.reference.start(facts, settings, now)),
        )

    def start_resolved_key(
        self, facts: decision.RequestFacts, settings: decision.Settings, now: datetime
    ) -> decision.Step:
        return AUTH_STEP.validate_python(
            _shadow(
                self.primary.start_resolved_key(facts, settings, now),
                lambda: self.reference.start_resolved_key(facts, settings, now),
            ),
        )

    def check_model_budgets(self) -> decision.Step:
        return AUTH_STEP.validate_python(
            _shadow(self.primary.check_model_budgets(), self.reference.check_model_budgets)
        )

    def check_user_model_budgets(self) -> decision.Step:
        return AUTH_STEP.validate_python(
            _shadow(self.primary.check_user_model_budgets(), self.reference.check_user_model_budgets)
        )

    def check_team_member_model_budgets(self) -> decision.TeamMemberStep:
        return MEMBER_STEP.validate_python(
            _shadow(self.primary.check_team_member_model_budgets(), self.reference.check_team_member_model_budgets),
        )
