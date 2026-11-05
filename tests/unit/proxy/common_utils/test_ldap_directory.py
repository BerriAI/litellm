import ssl
from collections.abc import Iterable, Mapping, Sequence
from threading import Event
from typing import Final

import pytest
from ldap3.core.exceptions import LDAPInsufficientAccessRightsResult, LDAPNoSuchObjectResult, LDAPSocketReceiveError
from pydantic import ValidationError

from litellm.proxy.common_utils.ldap_directory import (
    DirectoryError,
    DirectoryGroup,
    open_ldap_connection,
    read_directory,
)
from litellm.types.proxy.ldap_sync import LDAPSyncSettings

SETTINGS: Final = LDAPSyncSettings(
    enabled=True,
    server_url="ldap://directory.example.org",
    start_tls=True,
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
        groups: Iterable[Mapping[str, object]],
        users: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
        result_code: int = 0,
        tls_ok: bool = True,
        stop_after_search: Event | None = None,
        raising: Mapping[str, tuple[int | None, Exception]] | None = None,
    ) -> None:
        self.groups = groups
        self.users = users or {}
        self.raising = raising or {}
        self.result_code = result_code
        self.result: Mapping[str, object] = {"result": result_code}
        self.response: object = ()
        self.extend = self
        self.standard = self
        self.searches: list[tuple[str, str, str]] = []
        self.closed = False
        self.tls_ok = tls_ok
        self.stop_after_search = stop_after_search

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
        time_limit: int,
        generator: bool,
    ) -> object:
        return self.groups

    def search(
        self,
        *,
        search_base: str,
        search_filter: str,
        search_scope: str,
        attributes: Sequence[str],
        time_limit: int,
    ) -> bool:
        self.searches.append((search_base, search_filter, search_scope))
        if search_base in self.raising:
            result_code, exception = self.raising[search_base]
            if result_code is not None:
                self.result = {"result": result_code}
            raise exception
        self.response = self.users.get(search_base if search_scope == "BASE" else search_filter, ())
        self.result = {"result": self.result_code}
        if self.stop_after_search is not None:
            self.stop_after_search.set()
        return self.result_code == 0


def test_direct_members_resolve_to_user_identities_and_connection_is_closed():
    connection = Connection(
        [group(member=[ALICE, BOB])], {ALICE: [user("alice@example.org")], BOB: [user("bob@example.org")]}
    )
    result = read_directory(SETTINGS, connection)
    assert result == (DirectoryGroup("project-a", "project-a", ("alice@example.org", "bob@example.org")),)
    assert connection.closed


def test_deleted_member_entry_skips_only_its_group():
    connection = Connection(
        [group("project-a", member=[ALICE, BOB]), group("project-b", member=[ALICE])],
        {ALICE: [user("alice@example.org")]},
        raising={BOB: (32, LDAPNoSuchObjectResult(result=32, description="noSuchObject"))},
    )
    assert read_directory(SETTINGS, connection) == (
        DirectoryError("LDAP member entry does not exist", "cn=project-a,ou=groups,dc=example,dc=org"),
        DirectoryGroup("project-b", "project-b", ("alice@example.org",)),
    )
    assert connection.closed


def test_unreadable_member_entry_still_aborts_snapshot():
    connection = Connection(
        [group("project-a", member=[ALICE, BOB]), group("project-b", member=[ALICE])],
        {ALICE: [user("alice@example.org")]},
        raising={BOB: (50, LDAPInsufficientAccessRightsResult(result=50, description="insufficientAccessRights"))},
    )
    assert read_directory(SETTINGS, connection) == DirectoryError(
        "LDAP read failed (LDAPInsufficientAccessRightsResult)"
    )


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
        {"bind_dn": "cn=admin", "bind_password": "password", "start_tls": False},
        {"group_id_attribute": "cn)(uid=*"},
        {"litellm_match_field": "user_role"},
        {"interval_minutes": 0},
    ],
)
def test_invalid_configuration_is_rejected(updates: Mapping[str, object]):
    with pytest.raises(ValidationError):
        LDAPSyncSettings.model_validate({**SETTINGS.model_dump(), **updates})


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"bind_dn": "cn=admin", "start_tls": False}, "require TLS"),
        ({}, "bind_dn and bind_password must be configured together"),
    ],
)
def test_invalid_configuration_errors_do_not_expose_bind_password(updates: Mapping[str, object], message: str):
    password: Final = "qa-secret-42"
    with pytest.raises(ValidationError, match=message) as error:
        LDAPSyncSettings.model_validate(
            {**SETTINGS.model_dump(exclude_unset=True), **updates, "bind_password": password}
        )
    assert password not in str(error.value)


def test_cancelled_reader_stops_before_opening_ldap():
    cancellation = Event()
    cancellation.set()
    connection = Connection([group(member=[])])
    result = read_directory(SETTINGS, connection, cancellation)
    assert isinstance(result, DirectoryError)
    assert not connection.closed
    assert connection.searches == []


def test_cancellation_after_member_read_stops_remaining_lookups_and_closes_connection():
    cancellation = Event()
    connection = Connection(
        [group(member=[ALICE, BOB])],
        {ALICE: [user("alice@example.org")], BOB: [user("bob@example.org")]},
        stop_after_search=cancellation,
    )
    result = read_directory(SETTINGS, connection, cancellation)
    assert isinstance(result, DirectoryError)
    assert len(connection.searches) == 1
    assert connection.closed


def test_cancellation_during_paged_snapshot_discards_results_before_membership_reads():
    cancellation = Event()

    def pages() -> Iterable[Mapping[str, object]]:
        cancellation.set()
        yield group(member=[ALICE])
        pytest.fail("LDAP snapshot should stop consuming pages after cancellation")

    connection = Connection(pages())
    result = read_directory(SETTINGS, connection, cancellation)
    assert isinstance(result, DirectoryError)
    assert connection.searches == []
    assert connection.closed


@pytest.mark.parametrize(
    "url, ssl_enabled, port",
    [("ldaps://directory.example.org", True, 636), ("ldap://directory.example.org:1389", False, 1389)],
)
def test_real_ldap_client_constructor_verifies_tls_and_disables_referrals(url: str, ssl_enabled: bool, port: int):
    pytest.importorskip("ldap3")
    client = open_ldap_connection(SETTINGS.model_copy(update={"server_url": url}))
    assert client.server.tls.validate == ssl.CERT_REQUIRED
    assert client.server.ssl is ssl_enabled
    assert client.server.port == port
    assert client.auto_referrals is False


@pytest.mark.parametrize("invalid_id", [(), ("",), ("one", "two")])
def test_multiple_malformed_group_ids_do_not_block_healthy_groups(invalid_id: tuple[str, ...]):
    connection: Final = Connection(
        [
            entry("cn=broken-a", cn=invalid_id, member=[ALICE]),
            entry("cn=broken-b", cn=invalid_id, member=[ALICE]),
            group("healthy", member=[ALICE]),
        ],
        {ALICE: [user("alice@example.org")]},
    )
    result: Final = read_directory(SETTINGS, connection)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(item, DirectoryError) for item in result[:2])
    assert result[2] == DirectoryGroup("healthy", "healthy", ("alice@example.org",))
    assert connection.closed


def test_connection_failure_after_deleted_member_aborts_snapshot_instead_of_reusing_old_result():
    connection: Final = Connection(
        [group("deleted", member=[ALICE]), group("unreadable", member=[BOB]), group("healthy", member=[])],
        raising={
            ALICE: (32, LDAPNoSuchObjectResult(result=32, description="noSuchObject")),
            BOB: (None, LDAPSocketReceiveError("connection lost")),
        },
    )
    assert read_directory(SETTINGS, connection) == DirectoryError("LDAP read failed (LDAPSocketReceiveError)")
    assert connection.closed


@pytest.mark.parametrize("dry_run", [False, True])
def test_anonymous_sync_rejects_unprotected_directory_responses(dry_run: bool):
    with pytest.raises(ValidationError, match="TLS"):
        LDAPSyncSettings.model_validate({**SETTINGS.model_dump(), "start_tls": False, "dry_run": dry_run})


@pytest.mark.parametrize(
    "url,start_tls", [("ldaps://directory.example.org", False), ("ldap://directory.example.org", True)]
)
def test_anonymous_sync_accepts_verified_encrypted_transport(url: str, start_tls: bool):
    settings: Final = LDAPSyncSettings.model_validate(
        {**SETTINGS.model_dump(), "server_url": url, "start_tls": start_tls}
    )
    client: Final = open_ldap_connection(settings)
    assert client.server.tls.validate == ssl.CERT_REQUIRED
    assert client.server.ssl or settings.start_tls


def test_real_client_leaves_ranged_memberships_for_our_completeness_guard():
    client: Final = open_ldap_connection(SETTINGS)
    assert client.auto_range is False
    connection: Final = Connection([group(**{"member;range=0-1499": [ALICE]})], {ALICE: [user("alice@example.org")]})
    result: Final = read_directory(SETTINGS, connection)
    assert isinstance(result, tuple) and isinstance(result[0], DirectoryError)
    assert connection.searches == []
