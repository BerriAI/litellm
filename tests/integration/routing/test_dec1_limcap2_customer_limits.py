"""Customer hierarchy per-model limits, and the exact refusals of keys and customer organizations that borrow
idle model capacity (#50857, #51489, #51705).

Without borrowing, each level of the customer hierarchy a request runs under (customer organization, customer team,
customer member) whose budget sets ``model_rpm_limit`` or ``model_tpm_limit`` for the model group gets its own v3
descriptor ``model_per_<level>`` with value ``<entity_id>:<group>``
(``_customer_hierarchy_model_rate_limit_descriptors``, parallel_request_limiter_v3.py:2837). A request names its
customer with ``x-litellm-customer-member-id`` (the member's team and organization follow) or with
``x-litellm-customer-org-alias`` alone (organization level only, litellm/proxy/auth/customer_hierarchy_auth.py:60).

With ``priority_reservation_settings.model_limit_borrowing_config.borrow_idle_capacity`` listing ``key`` or
``customer_organization``, those limits become shares of the dynamic limiter instead
(``model_limit_borrowers``, parallel_request_limiter_v3.py:340). The refusal texts are built by ``_team_limit_error``
and ``_team_borrow_ceiling_error`` (dynamic_rate_limiter_v3.py:918 and 953) with the labels ``Key`` and ``Customer
organization``. ``routing/test_customer_org_and_key_model_limits_borrow_idle_capacity.py`` pins only headers and
fragments; these tests pin whole texts on rpm limits, which are deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.wire import Wire, wire_server
from tests.integration.authorization.dec1_limcap2_support import MEMBER_HEADER, ORG_ALIAS_HEADER, create_customer
from tests.integration.authorization.dec1_limits_support import chat, error_message, hashed, posts, provider, statuses
from tests.integration.authorization.dec1_ratelim_support import assert_refused_by
from tests.integration.routing.dec1_capacity_support import capacity_proxy, entry, group_name

MODEL_RPM: Final = 1000
SHARE_RPM: Final = 2


def test_a_customer_org_model_rpm_limit_refuses_with_model_per_customer_org(gateway: Gateway) -> None:
    """A customer organization's budget has ``model_rpm_limit: {<model>: 2}``. Three requests from a plain key with
    ``x-litellm-customer-member-id`` of a member of that organization: two are served, the third is the v3 429 for
    ``model_per_customer_org: <org_id>:<model>``, requests, limit 2, with ``rate_limited_by: model_per_customer_org``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        org, _alias, _team, member = create_customer(gateway, scenario, org_budget={"model_rpm_limit": {model: 2}})
        key: Final = scenario.key(models=[model])
        responses: Final = [chat(gateway, model, key, headers={MEMBER_HEADER: member}) for _ in range(3)]
        assert statuses(responses[:2]) == [200, 200], [response.text for response in responses]
        assert_refused_by(responses[2], "model_per_customer_org", f"{org}:{model}", "requests", 2)
        assert len(posts(wire)) == 2


def test_a_customer_team_model_rpm_limit_refuses_with_model_per_customer_team(gateway: Gateway) -> None:
    """Only the customer team's budget has ``model_rpm_limit: {<model>: 1}``. The second request of the member is the
    429 for ``model_per_customer_team: <customer_team_id>:<model>``, requests, limit 1."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        _org, _alias, team, member = create_customer(gateway, scenario, team_budget={"model_rpm_limit": {model: 1}})
        key: Final = scenario.key(models=[model])
        assert chat(gateway, model, key, headers={MEMBER_HEADER: member}).status_code == 200
        refused: Final = chat(gateway, model, key, headers={MEMBER_HEADER: member})
        assert_refused_by(refused, "model_per_customer_team", f"{team}:{model}", "requests", 1)


def test_a_customer_member_model_rpm_limit_from_the_org_template_refuses_with_model_per_customer_member(
    gateway: Gateway,
) -> None:
    """The organization's default member budget template has ``model_rpm_limit: {<model>: 1}``, so the member gets it.
    The member's second request is the 429 for ``model_per_customer_member: <member_id>:<model>``, limit 1. UNVERIFIED:
    that the template's per-model map is copied onto the member's budget row at creation."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        _org, _alias, _team, member = create_customer(gateway, scenario, member_budget={"model_rpm_limit": {model: 1}})
        key: Final = scenario.key(models=[model])
        assert chat(gateway, model, key, headers={MEMBER_HEADER: member}).status_code == 200
        refused: Final = chat(gateway, model, key, headers={MEMBER_HEADER: member})
        assert_refused_by(refused, "model_per_customer_member", f"{member}:{model}", "requests", 1)


def test_an_org_alias_header_alone_charges_only_the_org_level(gateway: Gateway) -> None:
    """The customer team has ``model_rpm_limit: {<model>: 1}`` and the organization none. Requests that send only
    ``x-litellm-customer-org-alias`` run at the organization level with no team or member
    (``CustomerHierarchyAuth`` docstring), so three are all served: the team's limit does not apply to them."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        _org, alias, _team, _member = create_customer(gateway, scenario, team_budget={"model_rpm_limit": {model: 1}})
        key: Final = scenario.key(models=[model])
        responses: Final = [chat(gateway, model, key, headers={ORG_ALIAS_HEADER: alias}) for _ in range(3)]
        assert statuses(responses) == [200, 200, 200], [response.text for response in responses]


def test_customer_model_limits_count_one_window_across_every_key_of_the_customer(gateway: Gateway) -> None:
    """The organization has ``model_rpm_limit: {<model>: 2}``. The member calls once through each of two different keys
    and a third time through the first: the third is refused for ``model_per_customer_org``, because the counter is
    keyed by the organization id, not by the key."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        org, _alias, _team, member = create_customer(gateway, scenario, org_budget={"model_rpm_limit": {model: 2}})
        first, second = scenario.key(models=[model]), scenario.key(models=[model])
        headers: Final = {MEMBER_HEADER: member}
        assert statuses(
            [chat(gateway, model, first, headers=headers), chat(gateway, model, second, headers=headers)]
        ) == [
            200,
            200,
        ]
        assert_refused_by(
            chat(gateway, model, first, headers=headers), "model_per_customer_org", f"{org}:{model}", "requests", 2
        )


@contextmanager
def _borrowing(
    gateway: Gateway, directory: Path, entities: list[str], threshold: float, max_borrow_multiple: int | None = None
) -> Iterator[tuple[Gateway, Wire, str, Scenario]]:
    group: Final = group_name("limcap2-borrow")
    config: Final[dict[str, JsonValue]] = {
        "borrow_idle_capacity": list(entities),
        **({"max_borrow_multiple": max_borrow_multiple} if max_borrow_multiple is not None else {}),
    }
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway,
            directory,
            model_list=[entry(group, wire.url, rpm=MODEL_RPM)],
            priority_settings={"saturation_threshold": threshold, "model_limit_borrowing_config": config},
            shared=True,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        yield candidate, wire, group, scenario


def _key_with_share(scenario: Scenario, group: str, metadata: Mapping[str, JsonValue] | None = None) -> str:
    return scenario.key(model_rpm_limit={group: SHARE_RPM}, metadata=dict(metadata or {}))


def test_a_saturated_borrowing_key_gets_the_full_key_no_idle_capacity_text(gateway: Gateway, tmp_path: Path) -> None:
    """Keys borrow (``borrow_idle_capacity: ["key"]``) and ``saturation_threshold: 0.0`` keeps the model saturated, so
    a key with ``model_rpm_limit: {<m>: 2}`` is held to 2. The third request is refused with exactly ``Key
    model_rpm_limit reached for <m> and the model is saturated, so there is no idle capacity to borrow. Key: <key
    hash>, Rate limit type: requests, Key model_rpm_limit: 2, Remaining: 0, Model saturation: 0.2%`` and the headers
    ``rate_limited_by: model_per_key_saturated``, ``x-litellm-saturation: 0.20%``. The entity is the hashed token
    (``ModelLimitBorrower.entity_id = str(api_key)``), never the raw key."""
    with _borrowing(gateway, tmp_path, ["key"], 0.0) as (candidate, wire, group, scenario):
        key: Final = _key_with_share(scenario, group)
        responses: Final = [chat(candidate, group, key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
        assert error_message(responses[2]) == (
            f"Key model_rpm_limit reached for {group} and the model is saturated, so there is no idle capacity to "
            f"borrow. Key: {hashed(key)}, Rate limit type: requests, Key model_rpm_limit: {SHARE_RPM}, Remaining: 0, "
            "Model saturation: 0.2%"
        )
        assert responses[2].headers["rate_limited_by"] == "model_per_key_saturated", dict(responses[2].headers)
        assert responses[2].headers["x-litellm-saturation"] == "0.20%", dict(responses[2].headers)
        assert key not in responses[2].text
        assert len(posts(wire)) == 2


def test_an_idle_borrowing_key_stops_at_its_own_borrow_multiple_with_the_full_cap_text(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Keys borrow, the model stays idle (``saturation_threshold: 1.0``) and the key's own metadata has
    ``model_limit_max_borrow_multiple: 2``. Four requests are served (2 x 2), the fifth is refused with exactly ``Key
    model_rpm_limit borrow cap reached for <m>: the key may use at most 2.0x its model_rpm_limit while borrowing idle
    capacity. Key: <hash>, Rate limit type: requests, Key model_rpm_limit: 2, Borrow cap: 4, Remaining: 0, Model
    saturation: 0.4%`` and ``rate_limited_by: model_per_key_max_borrow_multiple``. The multiple is a float, hence
    ``2.0x`` (``_MAX_BORROW_MULTIPLE``, parallel_request_limiter_v3.py:207)."""
    with _borrowing(gateway, tmp_path, ["key"], 1.0) as (candidate, _wire, group, scenario):
        key: Final = _key_with_share(scenario, group, {"model_limit_max_borrow_multiple": 2})
        responses: Final = [chat(candidate, group, key) for _ in range(5)]
        assert statuses(responses) == [200, 200, 200, 200, 429], [response.text for response in responses]
        assert error_message(responses[4]) == (
            f"Key model_rpm_limit borrow cap reached for {group}: the key may use at most 2.0x its model_rpm_limit "
            f"while borrowing idle capacity. Key: {hashed(key)}, Rate limit type: requests, Key model_rpm_limit: "
            f"{SHARE_RPM}, Borrow cap: 4, Remaining: 0, Model saturation: 0.4%"
        )
        assert responses[4].headers["rate_limited_by"] == "model_per_key_max_borrow_multiple", dict(
            responses[4].headers
        )


def test_an_out_of_range_key_borrow_multiple_is_ignored_and_the_global_one_applies(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The key's metadata has ``model_limit_max_borrow_multiple: 0.5`` (below the minimum of 1), the proxy's
    ``max_borrow_multiple`` is 3. The key value is ignored with a warning (``_max_borrow_multiple``), so the cap is
    2 x 3 = 6: six requests are served and the seventh names ``Borrow cap: 6``."""
    with _borrowing(gateway, tmp_path, ["key"], 1.0, max_borrow_multiple=3) as (candidate, _wire, group, scenario):
        key: Final = _key_with_share(scenario, group, {"model_limit_max_borrow_multiple": 0.5})
        responses: Final = [chat(candidate, group, key) for _ in range(7)]
        assert statuses(responses) == [200] * 6 + [429], [response.text for response in responses]
        assert "Borrow cap: 6, " in error_message(responses[6]), responses[6].text


def test_a_saturated_customer_org_gets_the_full_customer_organization_text(gateway: Gateway, tmp_path: Path) -> None:
    """Customer organizations borrow and the model is saturated. The organization's budget has ``model_rpm_limit:
    {<m>: 2}``; requests carry ``x-litellm-customer-org-alias``. The third is refused with exactly ``Customer
    organization model_rpm_limit reached for <m> and the model is saturated, so there is no idle capacity to borrow.
    Customer organization: <org_id>, Rate limit type: requests, Customer organization model_rpm_limit: 2, Remaining:
    0, Model saturation: 0.2%`` and ``rate_limited_by: model_per_customer_org_saturated``. The org's v3
    ``model_per_customer_org`` descriptor is dropped while it borrows (parallel_request_limiter_v3.py:2846), so the
    text is the dynamic limiter's, not the v3 one."""
    with _borrowing(gateway, tmp_path, ["customer_organization"], 0.0) as (candidate, _wire, group, scenario):
        org, alias, _team, _member = create_customer(
            candidate, scenario, org_budget={"model_rpm_limit": {group: SHARE_RPM}}
        )
        key: Final = scenario.key()
        responses: Final = [chat(candidate, group, key, headers={ORG_ALIAS_HEADER: alias}) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
        assert error_message(responses[2]) == (
            f"Customer organization model_rpm_limit reached for {group} and the model is saturated, so there is no "
            f"idle capacity to borrow. Customer organization: {org}, Rate limit type: requests, Customer "
            f"organization model_rpm_limit: {SHARE_RPM}, Remaining: 0, Model saturation: 0.2%"
        )
        assert responses[2].headers["rate_limited_by"] == "model_per_customer_org_saturated"


def test_an_idle_customer_org_stops_at_the_global_borrow_multiple_with_the_full_cap_text(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Customer organizations borrow, the model is idle and ``max_borrow_multiple: 2`` is set globally (an
    organization has no per-entity multiple, ``borrowing_customer_org_model_limits``). With ``model_rpm_limit: 2`` the
    fifth request is refused with a text that starts ``Customer organization model_rpm_limit borrow cap reached for
    <m>: the customer organization may use at most 2`` and contains ``Customer organization: <org_id>``, ``Borrow
    cap: 4`` and ``Model saturation: 0.4%``. The multiple is rendered as given by the settings model (``2`` or
    ``2.0``), so that part is matched loosely."""
    with _borrowing(gateway, tmp_path, ["customer_organization"], 1.0, max_borrow_multiple=2) as (
        candidate,
        _wire,
        group,
        scenario,
    ):
        org, alias, _team, _member = create_customer(
            candidate, scenario, org_budget={"model_rpm_limit": {group: SHARE_RPM}}
        )
        key: Final = scenario.key()
        responses: Final = [chat(candidate, group, key, headers={ORG_ALIAS_HEADER: alias}) for _ in range(5)]
        assert statuses(responses) == [200] * 4 + [429], [response.text for response in responses]
        assert re.fullmatch(
            rf"Customer organization model_rpm_limit borrow cap reached for {re.escape(group)}: the customer "
            r"organization may use at most 2(\.0)?x its model_rpm_limit while borrowing idle capacity\. Customer "
            rf"organization: {re.escape(org)}, Rate limit type: requests, Customer organization model_rpm_limit: "
            rf"{SHARE_RPM}, Borrow cap: 4, Remaining: 0, Model saturation: 0\.4%",
            error_message(responses[4]),
        ), responses[4].text
        assert responses[4].headers["rate_limited_by"] == "model_per_customer_org_max_borrow_multiple"
