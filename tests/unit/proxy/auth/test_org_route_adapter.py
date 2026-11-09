from collections.abc import Iterator, Mapping
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.proxy._types import LiteLLM_UserTable, ProxyException
from litellm.proxy.auth import org_route_adapter as native

from .test_route_role_adapter import observed

_OBJECTS: Final = TypeAdapter(tuple[object, ...])


def reference_admin(body: Mapping[str, object], user: LiteLLM_UserTable | None) -> bool:
    if user is None or user.organization_memberships is None:
        return False
    singular: Final = body.get("organization_id")
    plural: Final = body.get("organizations")
    candidates: Final = (
        *((singular,) if singular is not None else ()),
        *(_OBJECTS.validate_python(plural) if isinstance(plural, list) else ()),
    )
    admins: Final = frozenset(
        item.organization_id for item in user.organization_memberships
        if item.user_role == "org_admin" and item.organization_id is not None  # pyright: ignore[reportUnnecessaryComparison]  # preserve Phase 1 constructed membership handling
    )
    return bool(candidates) and all(candidate in admins for candidate in candidates)


def user(kind: str) -> LiteLLM_UserTable | None:
    if kind == "absent":
        return None
    return LiteLLM_UserTable.model_validate({
        "user_id": "u",
        "user_role": "proxy_admin" if kind == "proxy" else "internal_user",
        "organization_memberships": None if kind == "none" else (
            [] if kind == "empty" else [{
                "organization_id": "org", "user_role": "org_admin", "user_id": "u",
                "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z",
            }]
        ),
    })


@pytest.mark.parametrize("kind", ("absent", "none", "empty", "member", "proxy"))
@pytest.mark.parametrize("singular", (None, "org", "other", "", 0, False, 1.5, [], {}, ()))
@pytest.mark.parametrize("plural", (None, [], ["org"], ["other", []], ["org", []], [0], "org"))
def test_raw_singular_and_plural_keep_membership_failure_and_short_circuit(
    kind: str, singular: object, plural: object
) -> None:
    member: Final = user(kind)
    body: Final = {"organization_id": singular, "organizations": plural}
    assert observed(lambda: native.user_is_org_admin(body, member)) == observed(lambda: reference_admin(body, member))


@pytest.mark.parametrize("identifier", (0, False, 1.5, (), ""))
@pytest.mark.parametrize("route", ("/organization/delete", "/team/new"))
def test_nonstring_present_identifier_is_not_changed_to_missing(identifier: object, route: str) -> None:
    with pytest.raises(ProxyException) as captured:
        native.organization_role_based_access_check({"organization_id": identifier}, user("member"), route)
    expected: Final = (
        f"You do not have a role within the selected organization. Passed organization_id: {identifier}. Please contact the organization admin to request access."
        if route == "/organization/delete" else
        f"You do not have the required role to call {route}. Your role is None in Organization {identifier}"
    )
    assert captured.value.message == expected
    assert captured.value.code == "401"


@pytest.mark.parametrize("identifier", ([], {}))
@pytest.mark.parametrize("route", ("/organization/delete", "/team/new"))
def test_unhashable_identifier_raises_only_when_membership_lookup_is_selected(identifier: object, route: str) -> None:
    with pytest.raises(TypeError, match="unhashable type"):
        native.organization_role_based_access_check({"organization_id": identifier}, user("member"), route)
    assert native.organization_role_based_access_check({"organization_id": identifier}, user("proxy"), route) is None
    assert native.organization_role_based_access_check({"organization_id": identifier}, user("absent"), route) is None
    with pytest.raises(ProxyException, match="not a member of any organization"):
        native.organization_role_based_access_check({"organization_id": identifier}, user("none"), "/organization/delete")


class UnreadableBody(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError(f"unexpected body read: {key}")

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


class UnreadableHash:
    def __hash__(self) -> int:
        raise AssertionError("unexpected candidate hash")


def test_no_user_never_reads_request_and_denied_candidate_never_hashes_later_candidate() -> None:
    assert native.user_is_org_admin(UnreadableBody(), None) is False
    assert native.organization_role_based_access_check(UnreadableBody(), None, "/organization/delete") is None
    assert native.user_is_org_admin({"organizations": ["not-allowed", UnreadableHash()]}, user("member")) is False
