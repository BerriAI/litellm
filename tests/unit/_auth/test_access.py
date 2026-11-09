from collections.abc import Callable
from itertools import product
from typing import Final, cast

import pytest

from litellm._auth import access
from tests.unit._auth.reference_policy import access_decision


@pytest.fixture(params=("reference", "native"))
def decide(request: pytest.FixtureRequest) -> Callable[[access.AccessFacts], access.AccessDecision]:
    return access_decision if cast(str, request.param) == "reference" else access.decide


@pytest.mark.parametrize(
    "facts",
    tuple(
        access.UserParam(*values)
        for values in product((False, True), (None, "POST", "post", "GET"), (False, True), (False, True), (False, True))
    ),
)
def test_user_param_gate_preserves_method_route_and_presence(
    decide: Callable[[access.AccessFacts], access.AccessDecision], facts: access.UserParam
) -> None:
    assert decide(facts) == access_decision(facts)


@pytest.mark.parametrize("facts", tuple(access.GrantLayers(*values) for values in product((False, True), repeat=7)))
def test_default_deny_layers_preserve_master_dashboard_and_session_exceptions(
    decide: Callable[[access.AccessFacts], access.AccessDecision], facts: access.GrantLayers
) -> None:
    assert decide(facts) == access_decision(facts)


@pytest.mark.parametrize(
    "key,team,expected",
    ((None, None, ()), ((), ("a",), ("b", "b")), (("b",), ("a",), ("a",)), (("a", "b"), (), ()), ((), (), ())),
)
def test_key_tools_override_team_and_keep_duplicate_denials(
    decide: Callable[[access.AccessFacts], access.AccessDecision],
    key: tuple[str, ...] | None,
    team: tuple[str, ...] | None,
    expected: tuple[str, ...],
) -> None:
    assert decide(access.Tools(("a", "b", "b"), key, team)) == access.AccessDecision(
        denial="Tools" if expected else None, rejected=expected
    )


@pytest.mark.parametrize(
    "strict,allowed,expected",
    (
        (False, None, None),
        (False, (), None),
        (True, None, "MissingVectorGrant"),
        (True, (), "MissingVectorGrant"),
        (True, ("a",), "VectorStores"),
        (False, ("a",), "VectorStores"),
    ),
)
def test_vector_grants_distinguish_missing_from_denied_and_keep_first_index(
    decide: Callable[[access.AccessFacts], access.AccessDecision],
    strict: bool,
    allowed: tuple[str, ...] | None,
    expected: access.AccessDenial | None,
) -> None:
    assert decide(access.VectorStores(("a", "b", "a"), allowed, strict)) == access.AccessDecision(
        denial=expected, rejected_index=1 if expected == "VectorStores" else None
    )


@pytest.mark.parametrize(
    "facts",
    tuple(
        access.TraceStep(source, value, present, headers_only)
        for source, value, present, headers_only in product(
            ("Header", "Body", "Traceparent", "Selected", "Other"),
            (None, "", " ", "trace"),
            (False, True),
            (False, True),
        )
    ),
)
def test_trace_source_precedence_and_read_short_circuits(
    decide: Callable[[access.AccessFacts], access.AccessDecision], facts: access.TraceStep
) -> None:
    assert decide(facts) == access_decision(facts)


@pytest.mark.parametrize(
    "key",
    ("guardrails", "disable_global_guardrails", "disable_global_guardrail", "opted_out_global_guardrails", "unrelated"),
)
@pytest.mark.parametrize("modify", (None, False, True))
def test_guardrail_permission_requires_literal_false(
    decide: Callable[[access.AccessFacts], access.AccessDecision], key: str, modify: bool | None
) -> None:
    facts: Final = access.Guardrails((key,), modify)
    assert decide(facts) == access_decision(facts)


@pytest.mark.parametrize(
    "enabled,route,read",
    ((False, None, None), (True, None, "ClientTagRoute"), (True, False, None), (True, True, "ClientTags")),
)
def test_client_tags_only_read_route_and_request_state_when_enabled(
    decide: Callable[[access.AccessFacts], access.AccessDecision], enabled: bool, route: bool | None, read: str | None
) -> None:
    assert decide(access.ClientTagRequest(enabled, route)).read == read
