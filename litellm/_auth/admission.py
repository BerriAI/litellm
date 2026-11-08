from dataclasses import dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.rust_core import encode_facts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class CommonGate:
    force_virtual_key_checks: bool
    public_route: bool
    additional_public_route: bool
    unauthenticated_passthrough: bool
    master_key_present: bool
    jwt_enabled: bool
    oauth_enabled: bool
    oauth_proxy_enabled: bool
    custom_auth: bool
    custom_auth_common_checks: bool


@dataclass(frozen=True, slots=True)
class Reservation:
    skip_budget_checks: bool
    disabled: bool
    apply_user_budget_to_team_keys: bool
    fail_closed: bool
    billing_agent: bool


@dataclass(frozen=True, slots=True)
class Ownership:
    virtual_key: bool
    custom_auth: bool
    jwt: bool
    oauth: bool


@dataclass(frozen=True, slots=True)
class CommonPlan:
    skip: Literal["PublicRoute", "UnauthenticatedPassthrough", "NoAuthDevelopment", "CustomAuth"] | None


@dataclass(frozen=True, slots=True)
class ReservationPlan:
    reserve: bool
    apply_user_budget_to_team_keys: bool
    fail_closed: bool


@dataclass(frozen=True, slots=True)
class OwnershipPlan:
    owner: Literal["Python"]
    candidate: Literal["VirtualKey", "CustomAuth", "Validator", "Other"]


_COMMON: Final = TypeAdapter(CommonPlan)
_RESERVATION: Final = TypeAdapter(ReservationPlan)
_OWNERSHIP: Final = TypeAdapter(OwnershipPlan)


def common_plan(facts: CommonGate) -> CommonPlan:
    return _COMMON.validate_python(_native.request_auth_admission(encode_facts(facts)))


def reservation_plan(facts: Reservation) -> ReservationPlan:
    return _RESERVATION.validate_python(_native.request_auth_admission(encode_facts(facts)))


def ownership_plan(facts: Ownership) -> OwnershipPlan:
    return _OWNERSHIP.validate_python(_native.request_auth_admission(encode_facts(facts)))
