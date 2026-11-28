"""Priority class share splits by key and customer organization, and the dynamic limiter's Prometheus metrics for
holders other than teams.

``priority_reservation_settings.priority_class_share_config.split_equally_among_active`` lists the caller types that
take an equal slice of their priority class's share of a model once it saturates (#51366, #51705). A request counts
under its team first, then customer member, then customer organization, then key (``_class_share_holder``,
dynamic_rate_limiter_v3.py:679). The slice is ``class weight / active classes / active callers in the class`` of the
model's limit, floored at 1 (``team_share``, priority_class_share.py:70; ``max(floor, int(limit * weight))``,
dynamic_rate_limiter_v3.py:745). The refusal is ``_class_share_error`` (dynamic_rate_limiter_v3.py:985):
``Priority class share reached for <m> and the model is saturated. <Label>: <id>, Priority: <p>, Active callers in
class: <n>, Active classes: <k>, Rate limit type: <t>, Share: <s>, Remaining: <r>, Model saturation: <x.x%>`` with
``rate_limited_by: priority_class_share_saturated``. The existing ``test_priority_class_share_*`` files cover teams.

Metrics (#51417): ``litellm_dynamic_limiter_throttled_total`` counts every refusal whose ``rate_limited_by`` names a
holder (``_throttled``, dynamic_rate_limiter_v3.py:837), labelled by the holder's own label pair: ``team`` /
``team_alias``, ``hashed_api_key`` / ``api_key_alias``, or ``customer_org_id`` / ``customer_org_alias``
(``_dynamic_limiter_holder_labels``, litellm/integrations/prometheus.py:184). A model capacity refusal names no holder
and is not counted. ``routing/test_dynamic_limiter_prometheus_metrics.py`` covers the team holder.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.wire import Wire, wire_server
from tests.integration.authorization.dec1_limcap2_support import ORG_ALIAS_HEADER, create_customer, metric_total
from tests.integration.authorization.dec1_limits_support import chat, error_message, hashed, provider, statuses
from tests.integration.routing.dec1_capacity_support import capacity_proxy, entry, group_name

MODEL_RPM: Final = 10
THROTTLED: Final = "litellm_dynamic_limiter_throttled_total"


@contextmanager
def _split(
    gateway: Gateway,
    directory: Path,
    entities: Sequence[str],
    *,
    borrow: Sequence[str] = (),
    callbacks: Sequence[str] = ("dynamic_rate_limiter_v3",),
) -> Iterator[tuple[Gateway, Wire, str, Scenario]]:
    """A saturated model (``saturation_threshold: 0.0``) of ``rpm: 10`` with one class ``interactive`` at weight 1.0,
    the given split entities and, optionally, borrowing entities and extra callbacks."""
    group: Final = group_name("split")
    settings: Final[dict[str, JsonValue]] = {
        "saturation_threshold": 0.0,
        "saturation_check_cache_ttl": 0,
        "priority_class_share_config": {"split_equally_among_active": list(entities)},
        **({"model_limit_borrowing_config": {"borrow_idle_capacity": list(borrow)}} if borrow else {}),
    }
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway,
            directory,
            model_list=[entry(group, wire.url, rpm=MODEL_RPM)],
            priority_reservation={"interactive": 1.0},
            priority_settings=settings,
            callbacks=callbacks,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        yield candidate, wire, group, scenario


def _interactive_key(scenario: Scenario, **fields: JsonValue) -> str:
    metadata: Final[Mapping[str, JsonValue]] = {"priority": "interactive"}
    return scenario.key(metadata=dict(metadata), **fields)


def test_two_active_keys_split_their_class_share_and_the_refusal_names_the_key(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Keys split (``split_equally_among_active: ["key"]``). Two teamless ``interactive`` keys each send one request,
    which makes both active, so each slice is ``int(10 x 1.0 / 2) = 5``. The first key then sends four more (5 in
    all) and its next is refused with exactly ``Priority class share reached for <m> and the model is saturated. Key:
    <hash>, Priority: interactive, Active callers in class: 2, Active classes: 1, Rate limit type: requests, Share:
    5, Remaining: 0, Model saturation: 60.0%`` (6 of 10 requests used) and ``rate_limited_by:
    priority_class_share_saturated``."""
    with _split(gateway, tmp_path, ["key"]) as (candidate, _wire, group, scenario):
        first, second = _interactive_key(scenario), _interactive_key(scenario)
        assert statuses([chat(candidate, group, first), chat(candidate, group, second)]) == [200, 200]
        more: Final = [chat(candidate, group, first) for _ in range(5)]
        assert statuses(more) == [200, 200, 200, 200, 429], [response.text for response in more]
        assert error_message(more[4]) == (
            f"Priority class share reached for {group} and the model is saturated. Key: {hashed(first)}, Priority: "
            "interactive, Active callers in class: 2, Active classes: 1, Rate limit type: requests, Share: 5, "
            "Remaining: 0, Model saturation: 60.0%"
        )
        assert more[4].headers["rate_limited_by"] == "priority_class_share_saturated", dict(more[4].headers)


def test_a_team_key_counts_under_its_team_even_when_keys_also_split(gateway: Gateway, tmp_path: Path) -> None:
    """Both ``team`` and ``key`` are listed. Two keys of one team are one caller (the team comes first in
    ``_class_share_holder``), so a teamless key and the team are the two active callers with 5 each, and the team's
    sixth request across its two keys is refused with a text that names ``Team: <team_id>``, not a key hash. A port
    that lets a key pick its own slice would let a team multiply its share by minting keys."""
    with _split(gateway, tmp_path, ["team", "key"]) as (candidate, _wire, group, scenario):
        team: Final = scenario.team(metadata={"priority": "interactive"})
        team_keys: Final = (_interactive_key(scenario, team_id=team), _interactive_key(scenario, team_id=team))
        loner: Final = _interactive_key(scenario)
        assert statuses([chat(candidate, group, team_keys[0]), chat(candidate, group, loner)]) == [200, 200]
        more: Final = [chat(candidate, group, team_keys[index % 2]) for index in range(5)]
        assert statuses(more) == [200, 200, 200, 200, 429], [response.text for response in more]
        assert f" Team: {team}, Priority: interactive, Active callers in class: 2, " in error_message(more[4]), more[
            4
        ].text


def test_customer_organizations_split_their_class_share_across_their_keys(gateway: Gateway, tmp_path: Path) -> None:
    """``customer_organization`` splits. Two customer organizations (by ``x-litellm-customer-org-alias``) send one
    request each through one shared service key, so each has 5. The first organization's sixth request is refused with
    ``Customer organization: <org_id>`` in the class share text: the slice follows the organization, not the shared
    key. UNVERIFIED: the key's ``priority`` metadata is what puts the organizations' requests in ``interactive``."""
    with _split(gateway, tmp_path, ["customer_organization"]) as (candidate, _wire, group, scenario):
        org_a, alias_a, _team_a, _member_a = create_customer(candidate, scenario)
        _org_b, alias_b, _team_b, _member_b = create_customer(candidate, scenario)
        key: Final = _interactive_key(scenario)
        assert statuses(
            [
                chat(candidate, group, key, headers={ORG_ALIAS_HEADER: alias_a}),
                chat(candidate, group, key, headers={ORG_ALIAS_HEADER: alias_b}),
            ]
        ) == [200, 200]
        more: Final = [chat(candidate, group, key, headers={ORG_ALIAS_HEADER: alias_a}) for _ in range(5)]
        assert statuses(more) == [200, 200, 200, 200, 429], [response.text for response in more]
        assert f" Customer organization: {org_a}, Priority: interactive, Active callers in class: 2, " in error_message(
            more[4]
        ), more[4].text


def test_a_key_class_share_refusal_is_counted_under_the_hashed_api_key_label(gateway: Gateway, tmp_path: Path) -> None:
    """With ``prometheus`` on, the key class share refusal of the first test increments
    ``litellm_dynamic_limiter_throttled_total`` once for ``hashed_api_key=<hash>``, ``model_group=<m>``,
    ``priority=interactive``, ``rate_limit_type=requests``, ``reason=priority_class_share_saturated``. The holder is
    labelled by its own key label, not by ``team``."""
    with _split(gateway, tmp_path, ["key"], callbacks=("dynamic_rate_limiter_v3", "prometheus")) as (
        candidate,
        _wire,
        group,
        scenario,
    ):
        first, second = _interactive_key(scenario), _interactive_key(scenario)
        assert statuses([chat(candidate, group, first), chat(candidate, group, second)]) == [200, 200]
        assert [chat(candidate, group, first) for _ in range(5)][4].status_code == 429
        total: Final = metric_total(
            candidate,
            THROTTLED,
            1.0,
            hashed_api_key=hashed(first),
            model_group=group,
            priority="interactive",
            rate_limit_type="requests",
            reason="priority_class_share_saturated",
        )
        assert total == 1.0


def test_a_borrowing_key_refusal_is_counted_with_the_key_saturated_reason(gateway: Gateway, tmp_path: Path) -> None:
    """Keys borrow on the saturated model and a key has ``model_rpm_limit: {<m>: 1}``. Its second request is refused
    with ``rate_limited_by: model_per_key_saturated`` and the throttle counter for ``hashed_api_key=<hash>``,
    ``reason=model_per_key_saturated``, ``priority=default`` reaches 1."""
    with _split(gateway, tmp_path, [], borrow=["key"], callbacks=("dynamic_rate_limiter_v3", "prometheus")) as (
        candidate,
        _wire,
        group,
        scenario,
    ):
        key: Final = scenario.key(model_rpm_limit={group: 1})
        responses: Final = [chat(candidate, group, key) for _ in range(2)]
        assert statuses(responses) == [200, 429], [response.text for response in responses]
        assert responses[1].headers["rate_limited_by"] == "model_per_key_saturated"
        total: Final = metric_total(
            candidate, THROTTLED, 1.0, hashed_api_key=hashed(key), reason="model_per_key_saturated", priority="default"
        )
        assert total == 1.0


def test_a_borrowing_customer_org_refusal_is_counted_under_customer_org_labels(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Customer organizations borrow and the organization has ``model_rpm_limit: {<m>: 1}``. Its second request is
    refused for ``model_per_customer_org_saturated`` and the throttle counter for ``customer_org_id=<org_id>`` and
    ``customer_org_alias=<alias>`` with that reason reaches 1."""
    with _split(
        gateway, tmp_path, [], borrow=["customer_organization"], callbacks=("dynamic_rate_limiter_v3", "prometheus")
    ) as (candidate, _wire, group, scenario):
        org, alias, _team, _member = create_customer(candidate, scenario, org_budget={"model_rpm_limit": {group: 1}})
        key: Final = scenario.key()
        responses: Final = [chat(candidate, group, key, headers={ORG_ALIAS_HEADER: alias}) for _ in range(2)]
        assert statuses(responses) == [200, 429], [response.text for response in responses]
        total: Final = metric_total(
            candidate,
            THROTTLED,
            1.0,
            customer_org_id=org,
            customer_org_alias=alias,
            reason="model_per_customer_org_saturated",
        )
        assert total == 1.0


def test_a_model_capacity_refusal_is_not_counted_as_a_throttle(gateway: Gateway, tmp_path: Path) -> None:
    """The plain ``Model capacity reached`` 429 carries no ``rate_limited_by`` header (dynamic_rate_limiter_v3.py:1316
    sets only ``retry-after``, ``rate_limit_type`` and ``x-litellm-priority``), so ``_throttled`` finds no holder and
    ``litellm_dynamic_limiter_throttled_total`` has no sample for the model. Pinned so a port does not start counting
    capacity refusals against whichever caller happened to arrive last."""
    group: Final = group_name("cap-metric")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, rpm=1)],
            callbacks=("dynamic_rate_limiter_v3", "prometheus"),
        ) as candidate,
    ):
        responses: Final = [chat(candidate, group, candidate.key) for _ in range(2)]
        assert statuses(responses) == [200, 429], [response.text for response in responses]
        assert "rate_limited_by" not in responses[1].headers, dict(responses[1].headers)
        assert metric_total(candidate, THROTTLED, 0.0, seconds=5, model_group=group) == 0.0
