from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import (
    Final,
    TypeAlias,
    cast,  # noqa: TID251  # dataclass and enum reflection erases field types
)

from pydantic import TypeAdapter

from litellm._auth import decision
from litellm._auth.credential import KeySource, PresentedKeys, SelectedKey, WebSocketDenial, WebSocketKeys
from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native

Wire: TypeAlias = None | bool | int | float | str | list["Wire"] | dict[str, "Wire"]
CoreStep: TypeAlias = decision.Step | decision.TeamMemberStep

OBJECT_MAPPING: Final = TypeAdapter(dict[str, object])
OBJECT_TUPLE: Final = TypeAdapter(tuple[object, ...])
AUTH_STEP: Final = TypeAdapter[decision.Step](decision.Step)
MEMBER_STEP: Final = TypeAdapter[decision.TeamMemberStep](decision.TeamMemberStep)
CORE_STEP: Final = TypeAdapter[CoreStep](CoreStep)
TEAM_BUDGET: Final = TypeAdapter(decision.TeamMemberBudget)
FALLBACK_TARGET: Final = TypeAdapter(decision.FallbackTarget)
TEAM_ALERT: Final = TypeAdapter(decision.AlertTeamMemberBudget)
MEMBER_FALLBACK: Final = TypeAdapter(decision.VetTeamMemberFallback)
MODEL_DENIAL: Final = TypeAdapter(decision.ModelAccessDenied)

_PUBLIC_TYPES: Final[Mapping[str, Callable[..., object]]] = {
    cls.__name__: cls
    for cls in (
        decision.NoMasterKey,
        decision.UserAuthRoute,
        decision.MasterKey,
        decision.CachedAdminKey,
        decision.VirtualKey,
        decision.WithinModelBudgets,
        decision.MissingKey,
        decision.MalformedKey,
        decision.UserAuthDisabled,
        decision.ExpiredKey,
        decision.MasterKeyOnlyRoute,
        decision.NoDatabase,
        decision.NotAVirtualKey,
        decision.KeyBlocked,
        decision.OwnerDeactivated,
        decision.TeamMemberBudgetExceeded,
        decision.KeyBudgetExceeded,
        decision.KeyWithoutToken,
        decision.ModelBudgetExceeded,
        decision.ModelAccessDenied,
        decision.Allow,
        decision.Deny,
        decision.ResolveEndUser,
        decision.ResolvePresentedKey,
        decision.EvictPresentedKey,
        decision.LoadKey,
        decision.PrepareKey,
        decision.InspectKey,
        decision.EnforceModelAccess,
        decision.PrefetchAuthObjects,
        decision.LoadKeyOwner,
        decision.CheckBudgetExemption,
        decision.ReadTeamMemberBudget,
        decision.AlertTeamMemberBudget,
        decision.SendKeyBudgetAlerts,
        decision.ReadKeySpend,
        decision.AlertKeyBudget,
        decision.AlertKeySoftBudget,
        decision.LoadKeyContext,
        decision.ReadProxySpend,
        decision.AlertProxyBudget,
        decision.ReadModelBudgetScope,
        decision.CheckKeyModelBudget,
        decision.AuthorizeBudgetFallback,
        decision.RewriteRequestModel,
        decision.CheckUserModelBudget,
        decision.ReadTeamMemberModelBudgetScope,
        decision.CheckTeamMemberModelBudget,
        decision.ReadTeamMemberFallbacks,
        decision.VetTeamMemberFallback,
        decision.RerouteTeamMemberRequest,
    )
}
_ANSWER_TYPES: Final = {
    "EnforceModelAccess": "ModelAccess",
    "ResolvePresentedKey": "Presented",
    "InspectKey": "Key",
    "LoadKeyOwner": "Owner",
    "CheckBudgetExemption": "Exemption",
    "ReadTeamMemberBudget": "TeamBudget",
    "ReadKeySpend": "Spend",
    "ReadProxySpend": "ProxySpend",
    "ReadModelBudgetScope": "ModelScope",
    "CheckKeyModelBudget": "ModelBudgetData",
    "AuthorizeBudgetFallback": "FallbackAccess",
    "CheckUserModelBudget": "ModelBudgetData",
    "ReadTeamMemberModelBudgetScope": "TeamScope",
    "CheckTeamMemberModelBudget": "ModelBudgetData",
    "ReadTeamMemberFallbacks": "Targets",
    "VetTeamMemberFallback": "Bool",
}


class _Projection:
    def __init__(self) -> None:
        self.datetimes: tuple[datetime, ...] = ()

    @native_boundary
    def encode(self, value: object) -> Wire:
        if isinstance(value, datetime):
            offset: Final = value.utcoffset()
            reference: Final = len(self.datetimes)
            self.datetimes = (*self.datetimes, value)
            return {
                "wall_microseconds": (
                    (value.toordinal() * 86400 + value.hour * 3600 + value.minute * 60 + value.second) * 1000000
                    + value.microsecond
                ),
                "offset_microseconds": None
                if offset is None
                else (offset.days * 86400 + offset.seconds) * 1000000 + offset.microseconds,
                "timezone_id": id(value.tzinfo),
                "reference": reference,
            }
        if isinstance(value, Enum):
            return self.encode(cast(object, value.value))  # cast-ok: Enum.value is dynamically typed
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "kind": type(value).__name__,
                **{field.name: self.encode(field_value(value, field.name)) for field in fields(value)},
            }
        if isinstance(value, Mapping):
            return {key: self.encode(item) for key, item in OBJECT_MAPPING.validate_python(value).items()}
        if isinstance(value, (tuple, list, frozenset)):
            return [self.encode(item) for item in OBJECT_TUPLE.validate_python(value)]
        raise TypeError("unsupported auth protocol value")

    def decode(self, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data: Final = OBJECT_MAPPING.validate_python(value)
        if "wall_microseconds" in data:
            reference: Final = data["reference"]
            if not isinstance(reference, int):
                raise TypeError("invalid auth datetime reference")
            return self.datetimes[reference]
        kind: Final = data.get("kind")
        if kind == "ModelAccessDenied":
            return MODEL_DENIAL.validate_python({"decision": data["decision"]})
        if kind is None:
            return {key: self.decode(item) for key, item in data.items()}
        if not isinstance(kind, str):
            raise TypeError("invalid auth protocol tag")
        constructor: Final = public_type(kind)
        return constructor(**{key: self.decode(item) for key, item in data.items() if key != "kind"})

    @native_boundary
    def step(self, result: tuple[Mapping[str, object], _native.NativeAuthContinuation | None]) -> CoreStep:
        data, continuation = result
        kind: Final = data["kind"]
        if not isinstance(kind, str):
            raise TypeError("invalid auth protocol tag")
        if continuation is None:
            return CORE_STEP.validate_python(self.decode(data))

        @native_boundary
        def resume(*answers: object) -> CoreStep:
            answer: Final = {
                "kind": _ANSWER_TYPES.get(kind, "Unit"),
                "value": self.encode(answers[0]) if answers else None,
            }
            return self.step(continuation.resume(answer))

        payload: Final = {key: self.decode(value) for key, value in data.items() if key != "kind"}
        if kind == "AlertTeamMemberBudget":
            return TEAM_ALERT.validate_python(
                {**payload, "budget": TEAM_BUDGET.validate_python(payload["budget"]), "resume": resume}
            )
        if kind == "VetTeamMemberFallback":
            return MEMBER_FALLBACK.validate_python(
                {**payload, "target": FALLBACK_TARGET.validate_python(payload["target"]), "resume": resume}
            )
        constructor: Final = public_type(kind)
        return CORE_STEP.validate_python(constructor(**payload, resume=resume))


def field_value(value: object, name: str) -> object:
    return cast(object, getattr(value, name))  # cast-ok: dataclass fields are discovered dynamically


@native_boundary
def encode_facts(value: object) -> Wire:
    return _Projection().encode(value)


def public_type(kind: str) -> Callable[..., object]:
    return _PUBLIC_TYPES[kind]


class RustCore:
    @native_boundary
    def start(self, facts: decision.RequestFacts, settings: decision.Settings, now: datetime) -> decision.Step:
        projection: Final = _Projection()
        return AUTH_STEP.validate_python(
            projection.step(
                _native.request_auth_start(
                    projection.encode(facts), projection.encode(settings), projection.encode(now), False
                )
            ),
        )

    @native_boundary
    def start_resolved_key(
        self, facts: decision.RequestFacts, settings: decision.Settings, now: datetime
    ) -> decision.Step:
        projection: Final = _Projection()
        return AUTH_STEP.validate_python(
            projection.step(
                _native.request_auth_start(
                    projection.encode(facts), projection.encode(settings), projection.encode(now), True
                )
            ),
        )

    @native_boundary
    def check_model_budgets(self) -> decision.Step:
        return AUTH_STEP.validate_python(_Projection().step(_native.request_auth_budgets("Key")))

    @native_boundary
    def check_user_model_budgets(self) -> decision.Step:
        return AUTH_STEP.validate_python(_Projection().step(_native.request_auth_budgets("User")))

    @native_boundary
    def check_team_member_model_budgets(self) -> decision.TeamMemberStep:
        return MEMBER_STEP.validate_python(_Projection().step(_native.request_auth_budgets("TeamMember")))


@native_boundary
def select_api_key(keys: PresentedKeys) -> SelectedKey | None:
    selected: Final = _native.request_auth_select_api_key(_Projection().encode(keys))
    if selected is None:
        return None
    return SelectedKey(source=KeySource(selected["source"]), presented=selected["presented"], token=selected["token"])


@native_boundary
def select_websocket_api_key(keys: WebSocketKeys) -> str | WebSocketDenial:
    selected: Final = _native.request_auth_select_websocket_api_key(_Projection().encode(keys))
    return WebSocketDenial(selected["value"]) if selected["kind"] == "Denial" else selected["value"]


@native_boundary
def scheme_credential(value: str) -> str | None:
    return _native.request_auth_scheme_credential(value)


@native_boundary
def litellm_header_token(value: str) -> str:
    return _native.request_auth_litellm_header_token(value)
