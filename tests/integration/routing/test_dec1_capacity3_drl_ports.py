"""Model capacity behaviours from the dynamic limiter changes ported into the Rust callback (#51540).

#51952. Below ``saturation_threshold`` the priority counters are tracked, not enforced. Since #51952 that tracking
also runs when the caller holds a class share slice (``priority_class_share_config.split_equally_among_active``):
before, a slice's requests were never counted while the model was idle, so a team that used the idle model heavily
started saturation with an empty slice and could starve a teammate. Tracking now adds one request and no tokens per
admission (``_track_priority_usage``, dynamic_rate_limiter_v3.py:1419) and the answer's tokens land on the same
counter at settlement (:1865-1884).

#52081. The tracking call is atomic and writes nothing once any of its counters is full, so a slice or pool whose
token counter filled while the model was idle used to stop counting requests too. The tracking now reports the full
counter as it stands and counts the rest without it (``_without_counter``, :381; ``_track_priority_usage``, :1419).
Only requests are added (``{"requests": 1, "tokens": 0}``, :1426).

#51705. A request takes its class share slice as its team first, then customer member, then customer organization,
then key (``_class_share_holder``, :679), so customer headers a caller picks cannot move a team's key off the team's
slice. A team id that spells a key or customer holder (``key\\x1f...``) falls back to the class pool (:277, :689).
A customer member slice has no metric label of its own and its refusal is labelled ``Customer member`` (:996).

Counters are read from Redis, ``{<scope>:<value>}:requests|tokens``; every provider answer reports 18 tokens.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.wire import Wire, wire_server
from tests.integration.authorization.dec1_limcap2_support import (
    MEMBER_HEADER,
    ORG_ALIAS_HEADER,
    create_customer,
    metric_total,
)
from tests.integration.authorization.dec1_limcap4_support import PROVIDER_TOTAL, counter, counter_settles
from tests.integration.routing.dec1_capacity3_support import (
    CLASS_SHARE_SCOPE,
    add_member,
    class_share_value,
    holder,
    interactive_key,
)
from tests.integration.routing.dec1_capacity_support import (
    PRIORITY_SCOPE,
    capacity_proxy,
    chat,
    entry,
    error_message,
    group_name,
    hashed,
    provider,
    scope_headers,
    statuses,
)

THROTTLED: Final = "litellm_dynamic_limiter_throttled_total"


@contextmanager
def _rig(
    gateway: Gateway,
    directory: Path,
    *,
    entities: Sequence[str],
    reservation: Mapping[str, JsonValue],
    threshold: float,
    rpm: int | None = 10,
    tpm: int | None = None,
    status: int = 200,
    callbacks: Sequence[str] = ("dynamic_rate_limiter_v3",),
) -> Iterator[tuple[Gateway, Wire, str, Scenario]]:
    group: Final = group_name("c3ports")
    limits: Final[dict[str, JsonValue]] = {
        **({"rpm": rpm} if rpm is not None else {}),
        **({"tpm": tpm} if tpm is not None else {}),
    }
    with (
        wire_server(provider(status=status)) as wire,
        capacity_proxy(
            gateway,
            directory,
            model_list=[entry(group, wire.url, **limits)],
            priority_reservation=reservation,
            priority_settings={
                "saturation_threshold": threshold,
                "saturation_check_cache_ttl": 0,
                "priority_class_share_config": {"split_equally_among_active": list(entities)},
            },
            callbacks=callbacks,
            shared=True,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        yield candidate, wire, group, scenario


def test_an_idle_models_class_share_slice_counts_every_admitted_request(gateway: Gateway, tmp_path: Path) -> None:
    """#51952. Teams split, ``saturation_threshold: 0.9`` keeps the ``rpm: 10`` model idle, and one ``interactive``
    team sends two requests. Both are served and ``{priority_class_share_saturated:<len>:<m>:<team_id>}:requests``
    holds 2 (dynamic_rate_limiter_v3.py:1408-1415). Before #51952 the slice counter was never written below
    saturation, so the slice started empty when the model saturated."""
    with _rig(gateway, tmp_path, entities=["team"], reservation={"interactive": 0.6, "batch": 0.4}, threshold=0.9) as (
        candidate,
        _wire,
        group,
        scenario,
    ):
        team: Final = scenario.team(metadata={"priority": "interactive"})
        key: Final = scenario.key(team_id=team)
        assert statuses([chat(candidate, group, key) for _ in range(2)]) == [200, 200]
        assert counter_settles(CLASS_SHARE_SCOPE, class_share_value(group, team), "requests", 2) == 2


def test_an_idle_models_class_share_slice_settles_the_answers_tokens(gateway: Gateway, tmp_path: Path) -> None:
    """#51952 at settlement. Same idle split, a model with ``tpm: 10000``. After two answers of 18 tokens the team's
    slice counter ``{priority_class_share_saturated:<len>:<m>:<team_id>}:tokens`` holds exactly 36: settlement charges
    usage to the slice (:1859-1884) and admission added no tokens (:1426)."""
    with _rig(
        gateway,
        tmp_path,
        entities=["team"],
        reservation={"interactive": 0.6, "batch": 0.4},
        threshold=0.9,
        rpm=None,
        tpm=10_000,
    ) as (candidate, _wire, group, scenario):
        team: Final = scenario.team(metadata={"priority": "interactive"})
        key: Final = scenario.key(team_id=team)
        assert statuses([chat(candidate, group, key) for _ in range(2)]) == [200, 200]
        expected: Final = 2 * PROVIDER_TOTAL
        assert counter_settles(CLASS_SHARE_SCOPE, class_share_value(group, team), "tokens", expected) == expected


def test_an_idle_slice_reports_its_own_class_share_headers_on_the_served_response(
    gateway: Gateway, tmp_path: Path
) -> None:
    """#51952 makes the idle tracking statuses the response headers. Two ``interactive`` teams on an idle ``rpm: 10``
    model: the first team's first response reports ``x-ratelimit-priority_class_share_saturated-limit-requests: 10``
    (one active team takes the whole class, and only active classes count, priority_class_share.py:70) and
    ``remaining-requests: 9``; the second team's first response reports ``limit-requests: 5`` and
    ``remaining-requests: 4``, and neither carries a ``priority_model`` pool header."""
    with _rig(gateway, tmp_path, entities=["team"], reservation={"interactive": 0.6, "batch": 0.4}, threshold=0.9) as (
        candidate,
        _wire,
        group,
        scenario,
    ):
        first_team: Final = scenario.team(metadata={"priority": "interactive"})
        second_team: Final = scenario.team(metadata={"priority": "interactive"})
        first: Final = chat(candidate, group, scenario.key(team_id=first_team))
        second: Final = chat(candidate, group, scenario.key(team_id=second_team))
        assert scope_headers(first, CLASS_SHARE_SCOPE) == {"limit-requests": "10", "remaining-requests": "9"}
        assert scope_headers(second, CLASS_SHARE_SCOPE) == {"limit-requests": "5", "remaining-requests": "4"}
        assert scope_headers(first, PRIORITY_SCOPE) == {}


def test_an_idle_slice_whose_tokens_ran_out_keeps_counting_requests(gateway: Gateway, tmp_path: Path) -> None:
    """#51952 with #52081. ``{"interactive": 0.1, "batch": 0.9}`` on ``rpm: 1000, tpm: 1000`` with a ``batch`` team
    active gives an ``interactive`` team a slice of 100 tokens and 100 requests. Six answers settle 108 tokens, which
    fills the slice while the model is about 13% saturated. Two more requests are still served, the slice's
    ``requests`` counter reaches 8, and the last response reports ``remaining-tokens: 0`` and ``remaining-requests:
    92`` for ``priority_class_share_saturated``."""
    with _rig(
        gateway,
        tmp_path,
        entities=["team"],
        reservation={"interactive": 0.1, "batch": 0.9},
        threshold=0.5,
        rpm=1000,
        tpm=1000,
    ) as (candidate, _wire, group, scenario):
        batch: Final = scenario.key(team_id=scenario.team(metadata={"priority": "batch"}))
        team: Final = scenario.team(metadata={"priority": "interactive"})
        key: Final = scenario.key(team_id=team)
        value: Final = class_share_value(group, team)
        assert chat(candidate, group, batch).status_code == 200
        assert statuses([chat(candidate, group, key) for _ in range(6)]) == [200] * 6
        assert counter_settles(CLASS_SHARE_SCOPE, value, "tokens", 6 * PROVIDER_TOTAL) == 6 * PROVIDER_TOTAL
        later: Final = [chat(candidate, group, key) for _ in range(2)]
        assert statuses(later) == [200, 200], [response.text for response in later]
        assert counter(CLASS_SHARE_SCOPE, value, "requests") == 8
        assert scope_headers(later[-1], CLASS_SHARE_SCOPE) == {
            "limit-requests": "100",
            "remaining-requests": "92",
            "limit-tokens": "100",
            "remaining-tokens": "0",
        }


def test_tracking_a_priority_pool_adds_requests_but_never_tokens(gateway: Gateway, tmp_path: Path) -> None:
    """#52081 tracks ``{"requests": 1, "tokens": 0}`` on a pool (:1426). A ``premium`` key on an idle ``tpm: 10000,
    rpm: 100`` model gets a provider 500, so nothing settles: ``{priority_model:<m>:premium}:requests`` holds 1 and
    ``{priority_model:<m>:premium}:tokens`` is never written. A port that reserved an estimate on the pool at
    admission would leave tokens behind for a failed call."""
    with _rig(
        gateway, tmp_path, entities=[], reservation={"premium": 0.5}, threshold=0.9, rpm=100, tpm=10_000, status=500
    ) as (candidate, _wire, group, scenario):
        key: Final = scenario.key(metadata={"priority": "premium"})
        assert chat(candidate, group, key).status_code >= 500
        assert counter_settles(PRIORITY_SCOPE, f"{group}:premium", "requests", 1) == 1
        assert counter(PRIORITY_SCOPE, f"{group}:premium", "tokens") is None


def test_a_team_key_stays_on_its_teams_slice_when_it_names_a_customer_organization(
    gateway: Gateway, tmp_path: Path
) -> None:
    """#51705. ``team`` and ``customer_organization`` both split. A key of an ``interactive`` team sends one request
    with ``x-litellm-customer-org-alias`` naming a customer organization. The request is counted on the team's slice,
    ``{priority_class_share_saturated:<len>:<m>:<team_id>}:requests`` = 1, and the organization's slice
    ``...:customer_org\\x1f<org_id>`` is never written, because the team comes first (:686-691)."""
    with _rig(
        gateway, tmp_path, entities=["team", "customer_organization"], reservation={"interactive": 1.0}, threshold=0.9
    ) as (candidate, _wire, group, scenario):
        org, alias, _team, _member = create_customer(candidate, scenario)
        team: Final = scenario.team(metadata={"priority": "interactive"})
        key: Final = scenario.key(team_id=team)
        assert chat(candidate, group, key, headers={ORG_ALIAS_HEADER: alias}).status_code == 200
        assert counter_settles(CLASS_SHARE_SCOPE, class_share_value(group, team), "requests", 1) == 1
        assert counter(CLASS_SHARE_SCOPE, class_share_value(group, holder("customer_org", org)), "requests") is None


def test_a_customer_member_takes_its_own_slice_ahead_of_its_organization(gateway: Gateway, tmp_path: Path) -> None:
    """#51705. ``customer_member`` and ``customer_organization`` split on an idle model. A teamless key sends one
    request with ``x-litellm-customer-member-id``: it counts on ``...:customer_member\\x1f<member_id>`` (member before
    organization, :692-699) and the organization's slice stays unwritten."""
    with _rig(
        gateway,
        tmp_path,
        entities=["customer_member", "customer_organization"],
        reservation={"interactive": 1.0},
        threshold=0.9,
    ) as (candidate, _wire, group, scenario):
        org, _alias, _team, member = create_customer(candidate, scenario)
        key: Final = interactive_key(scenario)
        assert chat(candidate, group, key, headers={MEMBER_HEADER: member}).status_code == 200
        member_value: Final = class_share_value(group, holder("customer_member", member))
        assert counter_settles(CLASS_SHARE_SCOPE, member_value, "requests", 1) == 1
        assert counter(CLASS_SHARE_SCOPE, class_share_value(group, holder("customer_org", org)), "requests") is None


def test_a_customer_members_class_share_refusal_is_labelled_customer_member(gateway: Gateway, tmp_path: Path) -> None:
    """#51705. Members split on a saturated (``saturation_threshold: 0.0``) ``rpm: 10`` model. Two members of one
    customer team are active through one ``interactive`` key, so each slice is ``int(10 x 1.0 / 2) = 5``. The second
    member's sixth request is refused with ``rate_limited_by: priority_class_share_saturated`` and a text starting
    ``Priority class share reached for <m> and the model is saturated. Customer member: <member_id>, Priority:
    interactive, Active callers in class: 2, Active classes: 1, Rate limit type: requests, Share: 5, Remaining: 0``
    (``_class_share_error``, :996-1009: a member slice has no holder kind, so the label is ``Customer member``)."""
    with _rig(gateway, tmp_path, entities=["customer_member"], reservation={"interactive": 1.0}, threshold=0.0) as (
        candidate,
        _wire,
        group,
        scenario,
    ):
        _org, _alias, customer_team, first_member = create_customer(candidate, scenario)
        second_member: Final = add_member(candidate, customer_team)
        key: Final = interactive_key(scenario)
        assert chat(candidate, group, key, headers={MEMBER_HEADER: first_member}).status_code == 200
        served: Final = [chat(candidate, group, key, headers={MEMBER_HEADER: second_member}) for _ in range(5)]
        assert statuses(served) == [200] * 5, [response.text for response in served]
        refused: Final = chat(candidate, group, key, headers={MEMBER_HEADER: second_member})
        assert refused.status_code == 429, refused.text
        assert refused.headers["rate_limited_by"] == "priority_class_share_saturated"
        assert error_message(refused).startswith(
            f"Priority class share reached for {group} and the model is saturated. Customer member: {second_member}, "
            "Priority: interactive, Active callers in class: 2, Active classes: 1, Rate limit type: requests, "
            "Share: 5, Remaining: 0, Model saturation: "
        ), error_message(refused)


def test_a_customer_organizations_class_share_refusal_is_counted_under_its_alias(
    gateway: Gateway, tmp_path: Path
) -> None:
    """#51705 with metrics. Customer organizations split on a saturated ``rpm: 10`` model and two organizations are
    active, so each has 5. The first organization's sixth request is refused for ``priority_class_share_saturated``
    and ``litellm_dynamic_limiter_throttled_total`` for ``customer_org_id=<org_id>``, ``customer_org_alias=<alias>``
    and that reason reaches 1: the alias comes from the request's customer hierarchy (:826-833)."""
    with _rig(
        gateway,
        tmp_path,
        entities=["customer_organization"],
        reservation={"interactive": 1.0},
        threshold=0.0,
        callbacks=("dynamic_rate_limiter_v3", "prometheus"),
    ) as (candidate, _wire, group, scenario):
        org_a, alias_a, _team_a, _member_a = create_customer(candidate, scenario)
        _org_b, alias_b, _team_b, _member_b = create_customer(candidate, scenario)
        key: Final = interactive_key(scenario)
        assert chat(candidate, group, key, headers={ORG_ALIAS_HEADER: alias_b}).status_code == 200
        served: Final = [chat(candidate, group, key, headers={ORG_ALIAS_HEADER: alias_a}) for _ in range(5)]
        assert statuses(served) == [200] * 5, [response.text for response in served]
        refused: Final = chat(candidate, group, key, headers={ORG_ALIAS_HEADER: alias_a})
        assert refused.status_code == 429, refused.text
        total: Final = metric_total(
            candidate,
            THROTTLED,
            1.0,
            model_group=group,
            customer_org_id=org_a,
            customer_org_alias=alias_a,
            reason="priority_class_share_saturated",
        )
        assert total == 1.0


def test_an_idle_customer_org_borrowers_share_gauge_carries_the_org_alias(gateway: Gateway, tmp_path: Path) -> None:
    """#51705 with metrics. Customer organizations borrow (``borrow_idle_capacity: ["customer_organization"]``) on an
    idle ``tpm: 10000`` model and the organization has ``model_tpm_limit: {<m>: 100}``. After one served request
    ``litellm_dynamic_limiter_share_tokens`` for ``customer_org_id=<org_id>`` and ``customer_org_alias=<alias>`` is 100:
    an idle borrower's gauge falls back to its share on the serving model (:889-897) and the alias comes from the
    request's ``user_api_key_customer_org_alias`` (litellm_pre_call_utils.py:2274)."""
    group: Final = group_name("c3gauge")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, tpm=10_000)],
            priority_settings={
                "saturation_threshold": 0.9,
                "model_limit_borrowing_config": {"borrow_idle_capacity": ["customer_organization"]},
            },
            callbacks=("dynamic_rate_limiter_v3", "prometheus"),
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        org, alias, _team, _member = create_customer(candidate, scenario, org_budget={"model_tpm_limit": {group: 100}})
        assert chat(candidate, group, scenario.key(), headers={ORG_ALIAS_HEADER: alias}).status_code == 200
        share: Final = metric_total(
            candidate,
            "litellm_dynamic_limiter_share_tokens",
            100.0,
            model_group=group,
            customer_org_id=org,
            customer_org_alias=alias,
        )
        assert share == 100.0


def test_a_team_id_spelled_like_a_key_holder_is_held_to_the_class_pool(gateway: Gateway, tmp_path: Path) -> None:
    """#51705 guard. Teams split and a team is created with id ``key\\x1f<random>``, which spells a key holder.
    ``_is_team_class_share_holder`` rejects it (:277, :689), so the request keeps the class pool: the pool counter
    ``{priority_model:<m>:interactive}:requests`` holds 1 and no class share counter is written under that id.
    UNVERIFIED: that ``/team/new`` accepts a ``\\x1f`` in ``team_id``; if it refuses, the guard is unreachable from the
    API and this test should assert the refusal instead."""
    with _rig(gateway, tmp_path, entities=["team"], reservation={"interactive": 1.0}, threshold=0.9) as (
        candidate,
        _wire,
        group,
        scenario,
    ):
        lookalike: Final = holder("key", f"capacity3-{group[-10:]}")
        team: Final = scenario.team(team_id=lookalike, metadata={"priority": "interactive"})
        assert chat(candidate, group, scenario.key(team_id=team)).status_code == 200
        assert counter_settles(PRIORITY_SCOPE, f"{group}:interactive", "requests", 1) == 1
        assert counter(CLASS_SHARE_SCOPE, class_share_value(group, team), "requests") is None


def test_a_key_class_share_slice_is_counted_while_the_model_is_idle(gateway: Gateway, tmp_path: Path) -> None:
    """#51952 for key holders. Keys split on an idle model, and a teamless ``interactive`` key sends three requests.
    ``{priority_class_share_saturated:<len>:<m>:key\\x1f<hash>}:requests`` reaches 3, and the pool counter
    ``{priority_model:<m>:interactive}:requests`` is never written, since a slice replaces the pool descriptor
    (:759-762)."""
    with _rig(gateway, tmp_path, entities=["key"], reservation={"interactive": 1.0}, threshold=0.9) as (
        candidate,
        _wire,
        group,
        scenario,
    ):
        key: Final = interactive_key(scenario)
        assert statuses([chat(candidate, group, key) for _ in range(3)]) == [200] * 3
        value: Final = class_share_value(group, holder("key", hashed(key)))
        assert counter_settles(CLASS_SHARE_SCOPE, value, "requests", 3) == 3
        assert counter(PRIORITY_SCOPE, f"{group}:interactive", "requests") is None


def test_a_slice_tracked_while_idle_is_enforced_from_its_first_saturated_request(
    gateway: Gateway, tmp_path: Path
) -> None:
    """#51952 end to end. ``saturation_threshold: 0.5`` on ``rpm: 10`` with teams splitting ``{"interactive": 1.0}``
    and two active teams, so each slice is 5. The greedy team sends four requests while the model is idle (one more
    from the light team makes saturation 50%), then its fifth is served and its sixth is refused for
    ``priority_class_share_saturated``: the four idle requests counted toward the slice. Before #51952 the slice was
    empty at saturation and the greedy team got five more. Saturation before each greedy request is
    ``model requests / 10``, so its fifth request is the first one enforced (50%) and finds 4 on the slice."""
    with _rig(gateway, tmp_path, entities=["team"], reservation={"interactive": 1.0}, threshold=0.5) as (
        candidate,
        _wire,
        group,
        scenario,
    ):
        light: Final = scenario.key(team_id=scenario.team(metadata={"priority": "interactive"}))
        greedy: Final = scenario.key(team_id=scenario.team(metadata={"priority": "interactive"}))
        assert chat(candidate, group, light).status_code == 200
        responses: Final = [chat(candidate, group, greedy) for _ in range(6)]
        assert statuses(responses) == [200] * 5 + [429], [response.text for response in responses]
        assert responses[-1].headers["rate_limited_by"] == "priority_class_share_saturated"
