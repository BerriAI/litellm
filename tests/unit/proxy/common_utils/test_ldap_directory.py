from collections.abc import Mapping, Sequence
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.common_utils.ldap_directory import DirectoryError, DirectoryGroup, read_directory
from litellm.types.proxy.ldap_sync import LDAPSyncSettings

SETTINGS: Final = LDAPSyncSettings(
    enabled=True,
    server_url="ldap://directory.example.org",
    group_search_base="ou=groups,dc=example,dc=org",
    group_filter="(cn=project-*)",
    user_search_base="ou=people,dc=example,dc=org",
)
ALICE: Final = "uid=alice,ou=people,dc=example,dc=org"
BOB: Final = "uid=bob,ou=people,dc=example,dc=org"


def entry(dn: str, **attributes: object) -> Mapping[str, object]:
    return {"type": "searchResEntry", "dn": dn, "attributes": attributes}


def group(name: str = "project-a", **attributes: object) -> Mapping[str, object]:
    return entry(f"cn={name},ou=groups,dc=example,dc=org", cn=[name], **attributes)


def user(mail: str, **attributes: object) -> Mapping[str, object]:
    return entry(ALICE, mail=[mail], objectClass=["person"], **attributes)


class Connection:
    def __init__(
        self,
        groups: Sequence[Mapping[str, object]],
        users: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
        result_code: int = 0,
        tls_ok: bool = True,
    ) -> None:
        self.groups = groups
        self.users = users or {}
        self.result_code = result_code
        self.result: Mapping[str, object] = {"result": result_code}
        self.response: object = ()
        self.extend = self
        self.standard = self
        self.searches: list[tuple[str, str, str]] = []
        self.closed = False
        self.tls_ok = tls_ok

    def open(self) -> None:
        pass

    def bind(self) -> bool:
        return True

    def unbind(self) -> bool:
        self.closed = True
        return True

    def start_tls(self) -> bool:
        return self.tls_ok

    def paged_search(
        self,
        *,
        search_base: str,
        search_filter: str,
        search_scope: str,
        attributes: Sequence[str],
        paged_size: int,
        generator: bool,
    ) -> object:
        return self.groups

    def search(self, *, search_base: str, search_filter: str, search_scope: str, attributes: Sequence[str]) -> bool:
        self.searches.append((search_base, search_filter, search_scope))
        self.response = self.users.get(search_base if search_scope == "BASE" else search_filter, ())
        self.result = {"result": self.result_code}
        return self.result_code == 0


def test_direct_members_resolve_to_user_identities_and_connection_is_closed():
    connection = Connection(
        [group(member=[ALICE, BOB])], {ALICE: [user("alice@example.org")], BOB: [user("bob@example.org")]}
    )
    result = read_directory(SETTINGS, connection)
    assert result == (DirectoryGroup("project-a", "project-a", ("alice@example.org", "bob@example.org")),)
    assert connection.closed


@pytest.mark.parametrize("membership", [[], ()])
def test_successfully_read_empty_group_has_empty_roster(membership: Sequence[str]):
    assert read_directory(SETTINGS, Connection([group(member=membership)])) == (
        DirectoryGroup("project-a", "project-a", ()),
    )


def test_omitted_membership_attribute_is_an_empty_roster_with_complete_read_permissions():
    assert read_directory(SETTINGS, Connection([group()])) == (DirectoryGroup("project-a", "project-a", ()),)


def test_ldap_attribute_names_are_case_insensitive():
    connection = Connection(
        [entry("group", CN=["project-a"], MEMBER=[ALICE])], {ALICE: [entry(ALICE, MAIL=["a@example.org"])]}
    )
    assert read_directory(SETTINGS, connection) == (DirectoryGroup("project-a", "project-a", ("a@example.org",)),)


@pytest.mark.parametrize(
    "members",
    [
        (),
        (user("one@example.org"), user("two@example.org")),
        (entry(ALICE, mail=["one@example.org"], objectClass=["groupOfNames"]),),
        (entry(ALICE, mail=[], objectClass=["person"]),),
    ],
)
def test_incomplete_or_nested_member_does_not_become_empty_group(members: Sequence[Mapping[str, object]]):
    result = read_directory(SETTINGS, Connection([group(member=[ALICE])], {ALICE: members}))
    assert isinstance(result, tuple) and isinstance(result[0], DirectoryError)


def test_duplicate_ids_are_rejected_even_when_one_group_cannot_resolve_members():
    result = read_directory(SETTINGS, Connection([group(member=[]), group(member=[ALICE])]))
    assert result == DirectoryError("LDAP group IDs are not unique within the selected groups")


def test_ranged_membership_is_rejected_instead_of_removing_unread_members():
    result = read_directory(SETTINGS, Connection([group(**{"member;range=0-1499": [ALICE]})]))
    assert isinstance(result, tuple) and isinstance(result[0], DirectoryError)
    assert "ranged" in result[0].reason


@pytest.mark.parametrize("code", [4, 10, 50])
def test_partial_directory_results_are_rejected(code: int):
    result = read_directory(SETTINGS, Connection([group(member=[])], result_code=code))
    assert result == DirectoryError("LDAP group search did not complete successfully")


def test_referrals_are_not_treated_as_groups():
    result = read_directory(SETTINGS, Connection([{"type": "searchResRef", "uri": ["ldap://other.example.org"]}]))
    assert isinstance(result, DirectoryError)


def test_missing_group_preserves_absence_in_snapshot():
    assert read_directory(SETTINGS, Connection([])) == ()


def test_binary_group_identifiers_are_not_implicitly_decoded():
    result = read_directory(SETTINGS, Connection([entry("group", cn=[b"project-a"], member=[])]))
    assert isinstance(result, DirectoryError)


def test_member_uid_lookup_escapes_filter_values():
    settings = SETTINGS.model_copy(update={"member_attribute": "memberUid"})
    expected_filter = r"(uid=alice\2a\29\28uid=\2a)"
    connection = Connection([group(memberUid=["alice*)(uid=*"])], {expected_filter: [user("alice@example.org")]})
    assert read_directory(settings, connection) == (DirectoryGroup("project-a", "project-a", ("alice@example.org",)),)
    assert connection.searches == [(SETTINGS.user_search_base, expected_filter, "SUBTREE")]


def test_start_tls_failure_prevents_search_and_closes_connection():
    connection = Connection([group(member=[])], tls_ok=False)
    assert read_directory(SETTINGS.model_copy(update={"start_tls": True}), connection) == DirectoryError(
        "LDAP TLS negotiation failed"
    )
    assert connection.closed


def test_disabled_configuration_needs_no_connection_details():
    assert not LDAPSyncSettings(enabled=False).enabled


@pytest.mark.parametrize(
    "updates",
    [
        {"server_url": "https://example.org"},
        {"server_url": "ldap://user:password@example.org"},
        {"bind_dn": "cn=admin", "bind_password": "password"},
        {"group_id_attribute": "cn)(uid=*"},
        {"litellm_match_field": "user_role"},
        {"interval_minutes": 0},
    ],
)
def test_invalid_configuration_is_rejected(updates: Mapping[str, object]):
    with pytest.raises(ValidationError):
        LDAPSyncSettings.model_validate({**SETTINGS.model_dump(), **updates})
