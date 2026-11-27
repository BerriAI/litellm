from types import MappingProxyType
from typing import Final

import pytest

from litellm.proxy.management_helpers.mcp_auth_profile_grants import (
    AmbiguousProfile,
    MalformedGrants,
    TooManyGrants,
    UnknownProfile,
    UnknownServer,
    UnreachableServer,
    canonicalize_grants,
    parse_grants,
)
from litellm.types.mcp_server.auth_profile import StaticAuthProfile


@pytest.mark.parametrize("raw", [None, [], [("server", "Build")], {1: "Build"}, {"server": None}, {"server": b"Build"}])
def test_grants_must_be_a_mapping_of_strings(raw: object) -> None:
    assert parse_grants(raw) == MalformedGrants()


def test_no_grants_parse_to_empty() -> None:
    assert parse_grants({}) == {}


def test_grants_are_limited_in_count() -> None:
    allowed: Final = {str(i): "default" for i in range(256)}

    assert parse_grants(MappingProxyType(allowed)) == allowed
    assert parse_grants({**allowed, "extra": "default"}) == TooManyGrants(257)


def test_grants_preserve_server_specific_profile_spelling_and_explicit_default() -> None:
    first: Final = StaticAuthProfile.model_validate({"id": "Build", "auth_type": "basic"})
    second: Final = StaticAuthProfile.model_validate({"id": "BUILD", "auth_type": "basic"})

    assert canonicalize_grants(
        {"s1": "build", "s2": "build", "s3": "DeFaUlT"},
        {"s1": [first], "s2": [second], "s3": []},
        reachable={"s1", "s2", "s3"},
    ) == {"s1": "Build", "s2": "BUILD", "s3": "default"}


def test_parsing_preserves_the_named_grant_before_catalog_resolution() -> None:
    assert parse_grants({"tickets": "bUiLd"}) == {"tickets": "bUiLd"}


def test_one_unknown_profile_refuses_the_entire_grant_edit() -> None:
    assert canonicalize_grants(
        {"tickets": "default", "wiki": "Build"}, {"tickets": (), "wiki": ()}, reachable={"tickets", "wiki"}
    ) == UnknownProfile("wiki", "Build")


def test_the_largest_allowed_grant_set_resolves_completely() -> None:
    grants: Final = {str(i): "default" for i in range(256)}
    assert canonicalize_grants(grants, dict.fromkeys(grants, ()), reachable=grants.keys()) == grants


@pytest.mark.parametrize("names", [("Build", "BUILD"), ("BUILD", "Build"), ("Build", "Build")])
def test_a_grant_requires_one_matching_profile_on_its_server(names: tuple[str, str]) -> None:
    profiles: Final = tuple(StaticAuthProfile.model_validate({"id": name, "auth_type": "basic"}) for name in names)

    assert canonicalize_grants({"s1": "BUILD"}, {"s1": profiles}, reachable={"s1"}) == AmbiguousProfile("s1", "BUILD")


def test_default_grant_requires_a_registered_server() -> None:
    assert canonicalize_grants({"unknown": "DEFAULT"}, {}, reachable={"unknown"}) == UnknownServer("unknown")


@pytest.mark.parametrize("requested", ["absent", "\u212aey"])
def test_grants_refuse_an_unknown_or_non_ascii_profile_id(requested: str) -> None:
    profile: Final = StaticAuthProfile.model_validate({"id": "Key", "auth_type": "basic"})

    assert canonicalize_grants({"s1": requested}, {"s1": [profile]}, reachable={"s1"}) == UnknownProfile(
        "s1", requested
    )


def test_reach_is_checked_before_catalog_membership_for_every_server() -> None:
    assert canonicalize_grants({"unknown": "default", "outside": "Build"}, {}, reachable={"unknown"}) == (
        UnreachableServer("outside")
    )


def test_preparsed_grants_still_observe_the_size_limit() -> None:
    grants: Final = {str(i): "default" for i in range(257)}
    assert canonicalize_grants(grants, {}, reachable=frozenset()) == TooManyGrants(257)


def test_the_size_limit_is_checked_before_entry_types() -> None:
    grants: Final = {**{str(i): "default" for i in range(256)}, "extra": None}
    assert parse_grants(grants) == TooManyGrants(257)


def test_profile_refusals_follow_request_order() -> None:
    profile: Final = StaticAuthProfile.model_validate({"id": "Build", "auth_type": "basic"})

    assert canonicalize_grants(
        {"wiki": "Deploy", "tickets": "default"}, {"wiki": [profile]}, reachable={"wiki", "tickets"}
    ) == UnknownProfile("wiki", "Deploy")
