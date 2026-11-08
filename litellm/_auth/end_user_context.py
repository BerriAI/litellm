from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class IdFacts:
    value: str | None


@dataclass(frozen=True, slots=True)
class DefaultFacts:
    prisma_present: bool
    explicit_id: str | None
    global_id: str | None
    global_read: bool = False


@dataclass(frozen=True, slots=True)
class DefaultDecision:
    read_global: bool
    budget_id: str | None


@dataclass(frozen=True, slots=True)
class ApplyFacts:
    own_budget_id: str | None
    own_budget_present: bool
    key_id: str | None
    global_read: bool = False
    global_id: str | None = None
    default_read: bool = False
    default_present: bool = False


ApplyDecision: TypeAlias = Literal["Keep", "ReadGlobal", "ReadDefault", "Copy"]


@dataclass(frozen=True, slots=True)
class RegistryFacts:
    end_user_id: str
    global_id: str | None
    key_id: str | None
    validate_id: bool
    token_max_present: bool
    registry_read: bool = False
    registry: tuple[str, ...] | None = None


RegistryDecision: TypeAlias = Literal["Read", "Restricted", "Unrestricted"]
Field: TypeAlias = Literal["EndUserId", "Tpm", "Rpm", "Tpd", "MaxBudget", "Models", "Region"]


@dataclass(frozen=True, slots=True)
class Fields:
    tpm: bool = False
    rpm: bool = False
    tpd: bool = False
    max_budget: bool = False
    models: bool = False
    region: bool = False


@dataclass(frozen=True, slots=True)
class AssignmentFacts:
    mode: Literal["Params", "Token"]
    fields: Fields


@dataclass(frozen=True, slots=True)
class LookupFacts:
    key_id: str | None
    own_budget_present: bool = False
    end_user_read: bool = False
    end_user_present: bool = False
    global_read: bool = False
    global_id: str | None = None
    default_read: bool = False
    default_present: bool = False
    token_max_present: bool = False
    applied: bool = False
    read_failed: bool = False


@dataclass(frozen=True, slots=True)
class LookupDecision:
    action: Literal["ReadEndUser", "ReadGlobal", "ReadDefault", "ApplyOwn", "ApplyDefault", "Done"]
    set_max_budget: bool
    project_budget: bool


@dataclass(frozen=True, slots=True)
class ValidationFacts:
    mode: Literal["Resolve", "Exists"]
    end_user_id: str | None
    enabled: bool = False
    prisma_present: bool = False
    global_id: str | None = None
    key_id: str | None = None
    cache_read: bool = False
    cached: str | None = None
    end_user_read: bool = False
    end_user_present: bool = False
    user_read: bool = False
    user_present: bool = False
    cache_written: bool = False


@dataclass(frozen=True, slots=True)
class ValidationDecision:
    read: Literal["Cache", "EndUser", "User", "WriteCache"] | None
    end_user_id: str | None
    valid: bool
    cache_value: str | None
    cache_ttl: int
    user_email: str | None


_ID: Final = TypeAdapter(IdFacts)
_STRING: Final = TypeAdapter[str | None](str | None)
_DEFAULT: Final = TypeAdapter(DefaultFacts)
_DEFAULT_RESULT: Final = TypeAdapter(DefaultDecision)
_APPLY: Final = TypeAdapter(ApplyFacts)
_APPLY_RESULT: Final = TypeAdapter[ApplyDecision](ApplyDecision)
_REGISTRY: Final = TypeAdapter(RegistryFacts)
_REGISTRY_RESULT: Final = TypeAdapter[RegistryDecision](RegistryDecision)
_ASSIGNMENTS: Final = TypeAdapter(AssignmentFacts)
_FIELDS: Final = TypeAdapter(tuple[Field, ...])
_LOOKUP: Final = TypeAdapter(LookupFacts)
_LOOKUP_RESULT: Final = TypeAdapter(LookupDecision)
_VALIDATION: Final = TypeAdapter(ValidationFacts)
_VALIDATION_RESULT: Final = TypeAdapter(ValidationDecision)


@native_boundary
def normalize_id(facts: IdFacts) -> str | None:
    return _STRING.validate_python(_native.request_auth_end_user_normalize_id(_ID.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen typed facts


@native_boundary
def default_id(facts: DefaultFacts) -> DefaultDecision:
    return _DEFAULT_RESULT.validate_python(_native.request_auth_end_user_default_id(_DEFAULT.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen typed facts


@native_boundary
def apply_default(facts: ApplyFacts) -> ApplyDecision:
    return _APPLY_RESULT.validate_python(_native.request_auth_end_user_apply(_APPLY.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen typed facts


@native_boundary
def registry(facts: RegistryFacts) -> RegistryDecision:
    return _REGISTRY_RESULT.validate_python(_native.request_auth_end_user_registry(_REGISTRY.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen typed facts


@native_boundary
def assignments(facts: AssignmentFacts) -> tuple[Field, ...]:
    return _FIELDS.validate_python(_native.request_auth_end_user_assignments(_ASSIGNMENTS.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen typed facts


@native_boundary
def lookup(facts: LookupFacts) -> LookupDecision:
    return _LOOKUP_RESULT.validate_python(_native.request_auth_end_user_lookup(_LOOKUP.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen typed facts


@native_boundary
def validation(facts: ValidationFacts) -> ValidationDecision:
    return _VALIDATION_RESULT.validate_python(_native.request_auth_end_user_validation(_VALIDATION.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen typed facts
