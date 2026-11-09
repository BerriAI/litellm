from collections.abc import Mapping
from dataclasses import replace
from typing import Final

from pydantic import TypeAdapter

from litellm._auth.failure import NativeAuthFailure
from litellm._auth.org_routes import (
    AdminFacts,
    Identifier,
    OrganizationMembership,
    OrganizationRouteFacts,
    admin,
    organization_route,
)
from litellm.proxy._types import LiteLLM_UserTable, LiteLLMRoutes, ProxyErrorTypes, ProxyException

_OBJECTS: Final = TypeAdapter(tuple[object, ...])


def memberships(user: LiteLLM_UserTable | None) -> tuple[OrganizationMembership, ...] | None:
    return (
        tuple(OrganizationMembership(item.organization_id, item.user_role) for item in user.organization_memberships)
        if user is not None and user.organization_memberships is not None
        else None
    )


def user_is_org_admin(request_data: Mapping[str, object], user_object: LiteLLM_UserTable | None = None) -> bool:
    facts = AdminFacts(memberships(user_object))  # rebind-ok: native-directed read stage
    singular: object = None  # rebind-ok: raw request field loaded only after the native directive
    plural: tuple[object, ...] = ()  # rebind-ok: raw request field loaded only after the native directive
    while True:
        decision = admin(facts)
        match decision.action:
            case "ReadRequest":
                singular = request_data.get("organization_id")
                raw_plural = request_data.get("organizations")
                plural = _OBJECTS.validate_python(raw_plural) if isinstance(raw_plural, list) else ()
                facts = replace(
                    facts, request_read=True, singular_present=singular is not None, plural_count=len(plural)
                )
            case "ReadSingular" | "ReadPlural":
                raw = singular if decision.action == "ReadSingular" else plural[decision.plural_index]
                facts = replace(facts, candidate_index=decision.candidate_index, candidate=_identifier(raw))
            case "Allow":
                return True
            case "Deny":
                return False
            case "HashFailure":
                raise TypeError(decision.hash_failure)


def _identifier(value: object) -> Identifier:
    try:
        hash(value)
    except NativeAuthFailure:
        raise
    except TypeError as error:
        return Identifier(None, str(error))
    return Identifier(value if isinstance(value, str) else None)


def organization_role_based_access_check(
    request_body: Mapping[str, object], user_object: LiteLLM_UserTable | None, route: str
) -> None:
    facts = OrganizationRouteFacts(  # rebind-ok: native-directed read driver
        route,
        user_object is not None,
        None,
        None,
        None,
        tuple(LiteLLMRoutes.org_admin_only_routes.value),
        request_pending=True,
    )
    passed: object = None  # rebind-ok: raw field is read only at the native-selected stage
    while True:
        decision = organization_route(facts)
        if decision.read_request:
            assert user_object is not None
            passed = request_body.get("organization_id")
            facts = replace(
                facts,
                user_role=user_object.user_role,
                organization_id=passed if isinstance(passed, str) else None,
                organization_id_present=passed is not None,
                request_pending=False,
                memberships_pending=True,
                identifier_pending=True,
            )
        elif decision.read_memberships:
            facts = replace(facts, memberships=memberships(user_object), memberships_pending=False)
        elif decision.read_identifier:
            facts = replace(facts, identifier_pending=False, hash_failure=_identifier(passed).hash_failure)
        else:
            break
    if decision.hash_failure is not None:
        raise TypeError(decision.hash_failure)
    organization_id: Final = passed
    role: Final = facts.user_role
    messages: Final = {
        "CreateOrganization": (f"Only proxy admins can create new organizations. You are {role}", "user_role"),
        "NoMemberships": (
            f"Tried to access route={route} but you are not a member of any organization. Please contact the proxy admin to request access.",
            "organization_id",
        ),
        "MissingOrganization": (
            "Passed organization_id is None, please pass an organization_id in your request",
            "organization_id",
        ),
        "MissingRole": (
            f"You do not have a role within the selected organization. Passed organization_id: {organization_id}. Please contact the organization admin to request access.",
            "organization_id",
        ),
        "WrongRole": (
            f"You do not have the required role to perform {route} in Organization {organization_id}. Your role is {decision.organization_role} in Organization {organization_id}",
            "user_role",
        ),
        "TeamMissingOrganization": (
            f"Passed organization_id is None, please specify the organization_id in your request. You are part of multiple organizations: {list(decision.organization_ids)}",
            "organization_id",
        ),
        "TeamWrongRole": (
            f"You do not have the required role to call {route}. Your role is {decision.organization_role} in Organization {organization_id}",
            "user_role",
        ),
    }
    if decision.denial is not None:
        message, param = messages[decision.denial]
        raise ProxyException(message=message, type=ProxyErrorTypes.auth_error.value, param=param, code=401)
