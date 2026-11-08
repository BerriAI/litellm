from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class ModelCheckFacts:
    scope: Literal["Team", "EndUser"]
    entity_id: str | None
    budget_count: int
    model_count: int


@dataclass(frozen=True, slots=True)
class BudgetFields:
    max_budget: float | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    tpd_limit: int | None = None
    model_budget_count: int | None = None


@dataclass(frozen=True, slots=True)
class EndUserFacts:
    mode: Literal["Models", "Token", "Default"]
    object_present: bool
    end_user_id: str | None
    prisma_present: bool
    key_budget_id: str | None
    key_read: bool = False
    key_budget: BudgetFields | None = None
    global_config_read: bool = False
    global_budget_id: str | None = None
    global_read: bool = False
    global_budget: BudgetFields | None = None


EndUserSource: TypeAlias = Literal["Own", "ReadKey", "Key", "ReadGlobalConfig", "ReadGlobal", "Global", "None"]


@dataclass(frozen=True, slots=True)
class TokenDefaultsFacts:
    current: BudgetFields
    keep_token_limits: bool
    default_present: bool


TokenField: TypeAlias = Literal["MaxBudget", "Tpm", "Rpm", "Tpd", "Models"]


@dataclass(frozen=True, slots=True)
class MembershipSnapshot:
    budget_id: str | None
    model_budget_count: int | None


@dataclass(frozen=True, slots=True)
class MemberFacts:
    user_id: str | None
    team_id: str | None
    prisma_present: bool
    default_id: str | None
    skip_budget_checks: bool
    membership_read: bool = False
    membership: MembershipSnapshot | None = None
    default_read: bool = False
    default_budget: BudgetFields | None = None
    lookup_failure: Literal["Membership", "Default"] | None = None


MemberSource: TypeAlias = Literal[
    "Done", "ReadMembership", "ReadDefault", "Member", "Default", "Clear", "RaiseLookup", "SkipLookup"
]


@dataclass(frozen=True, slots=True)
class IncreaseFacts:
    number: float | None
    truthy: bool
    type_name: str


@dataclass(frozen=True, slots=True)
class TemporaryFacts:
    max_budget: float | None
    metadata_read: bool = False
    has_increase: bool = False
    has_expiry: bool = False
    expires_at: int | None = None
    now: int | None = None
    increase: IncreaseFacts | None = None


@dataclass(frozen=True, slots=True)
class TemporaryDecision:
    read: Literal["Metadata", "Expiry", "Now", "Increase"] | None = None
    copy: bool = False
    max_budget: float | None = None
    invalid_increase_type: str | None = None


@dataclass(frozen=True, slots=True)
class NumericFacts:
    number: float | None
    boolean: bool


@dataclass(frozen=True, slots=True)
class ThrottleFacts:
    opt_in: bool
    tpm_limit: int | None
    rpm_limit: int | None
    percentage: NumericFacts | None = None


@dataclass(frozen=True, slots=True)
class ThrottleDecision:
    read_percentage: bool = False
    percentage: float | None = None


_MODELS: Final = TypeAdapter(ModelCheckFacts)
_INDICES: Final = TypeAdapter(tuple[int, ...])
_END: Final = TypeAdapter(EndUserFacts)
_END_SOURCE: Final = TypeAdapter[EndUserSource](EndUserSource)
_TOKEN: Final = TypeAdapter(TokenDefaultsFacts)
_FIELDS: Final = TypeAdapter(tuple[TokenField, ...])
_MEMBER: Final = TypeAdapter(MemberFacts)
_MEMBER_SOURCE: Final = TypeAdapter[MemberSource](MemberSource)
_TEMP: Final = TypeAdapter(TemporaryFacts)
_TEMP_DECISION: Final = TypeAdapter(TemporaryDecision)
_THROTTLE: Final = TypeAdapter(ThrottleFacts)
_THROTTLE_DECISION: Final = TypeAdapter(ThrottleDecision)
_NUMBER: Final = TypeAdapter(NumericFacts)
_PERCENTAGE: Final = TypeAdapter[float | None](float | None)


@native_boundary
def model_checks(facts: ModelCheckFacts) -> tuple[int, ...]:
    return _INDICES.validate_python(_native.request_auth_context_models(_MODELS.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts


@native_boundary
def end_user_source(facts: EndUserFacts) -> EndUserSource:
    return _END_SOURCE.validate_python(_native.request_auth_context_end_user(_END.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts


@native_boundary
def token_assignments(facts: TokenDefaultsFacts) -> tuple[TokenField, ...]:
    return _FIELDS.validate_python(_native.request_auth_context_token(_TOKEN.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts


@native_boundary
def member_source(facts: MemberFacts) -> MemberSource:
    return _MEMBER_SOURCE.validate_python(_native.request_auth_context_member(_MEMBER.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts


@native_boundary
def temporary(facts: TemporaryFacts) -> TemporaryDecision:
    return _TEMP_DECISION.validate_python(_native.request_auth_context_temporary(_TEMP.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts


@native_boundary
def percentage(facts: NumericFacts) -> float | None:
    return _PERCENTAGE.validate_python(_native.request_auth_context_percentage(_NUMBER.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts


@native_boundary
def throttle(facts: ThrottleFacts) -> ThrottleDecision:
    return _THROTTLE_DECISION.validate_python(_native.request_auth_context_throttle(_THROTTLE.dump_python(facts)))  # pyright: ignore[reportAny]  # Pydantic serializes frozen facts
