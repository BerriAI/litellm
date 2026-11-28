"""Model capacity: the exact refusals of a team that borrows idle capacity (#51328, #51359, #51705).

With ``priority_reservation_settings.model_limit_borrowing_config.borrow_idle_capacity: ["team"]`` a team's
``model_rpm_limit``/``model_tpm_limit`` for a model stop being ceilings while the model is below
``saturation_threshold``, and become its share once the model saturates. ``model_limit_max_borrow_multiple`` in the
team's metadata caps how far it borrows while idle. ``routing/test_team_model_limit_borrows_idle_capacity.py`` pins
that the team is held and capped, and checks only fragments of the text. These tests pin the full refusal texts and
headers, which a client or dashboard reads and which the Rust port (#51540) re-implements by hand:

At saturation: ``Team model_rpm_limit reached for <m> and the model is saturated, so there is no idle capacity to
borrow. Team: <team_id>, Rate limit type: requests, Team model_rpm_limit: <n>, Remaining: 0, Model saturation:
<x.x%>`` with ``rate_limited_by: model_per_team_saturated``.

At the borrow cap: ``Team model_rpm_limit borrow cap reached for <m>: the team may use at most <k>x its
model_rpm_limit while borrowing idle capacity. Team: <team_id>, ..., Borrow cap: <n*k>, ...`` with
``rate_limited_by: model_per_team_max_borrow_multiple``.

Read from ``litellm/proxy/hooks/dynamic_rate_limiter_v3.py`` (``_team_limit_error``, ``_team_borrow_ceiling_error``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from tests.integration._support.client import Gateway
from tests.integration._support.wire import wire_server
from tests.integration.routing.dec1_capacity_support import (
    WINDOW_SECONDS,
    capacity_proxy,
    chat,
    entry,
    error_message,
    group_name,
    provider,
    statuses,
)

MODEL_RPM: Final = 1000
TEAM_RPM: Final = 2


@contextmanager
def _borrowing(gateway: Gateway, directory: Path, threshold: float) -> Iterator[tuple[Gateway, str, str, str]]:
    """A proxy where teams borrow, one model of ``rpm: 1000``, and one key of a team with ``model_rpm_limit: 2``
    and ``model_limit_max_borrow_multiple: 2`` on it. Yields the proxy, the model, the team id and the key."""
    group: Final = group_name("borrow")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway,
            directory,
            model_list=[entry(group, wire.url, rpm=MODEL_RPM)],
            priority_settings={
                "saturation_threshold": threshold,
                "model_limit_borrowing_config": {"borrow_idle_capacity": ["team"]},
            },
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        team: Final = scenario.team(
            metadata={"model_rpm_limit": {group: TEAM_RPM}, "model_limit_max_borrow_multiple": 2}
        )
        yield candidate, group, team, scenario.key(team_id=team)


def test_a_saturated_team_gets_the_full_no_idle_capacity_text(gateway: Gateway, tmp_path: Path) -> None:
    """``saturation_threshold: 0.0`` keeps the model saturated, so the team is held to ``model_rpm_limit: 2``. The
    third request is refused with exactly ``Team model_rpm_limit reached for <m> and the model is saturated, so there
    is no idle capacity to borrow. Team: <team_id>, Rate limit type: requests, Team model_rpm_limit: 2, Remaining:
    0, Model saturation: 0.2%`` (2 of 1000 requests)."""
    with _borrowing(gateway, tmp_path, 0.0) as (candidate, group, team, key):
        responses: Final = [chat(candidate, group, key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
        assert error_message(responses[2]) == (
            f"Team model_rpm_limit reached for {group} and the model is saturated, so there is no idle capacity to "
            f"borrow. Team: {team}, Rate limit type: requests, Team model_rpm_limit: {TEAM_RPM}, Remaining: 0, "
            "Model saturation: 0.2%"
        )


def test_a_saturated_teams_refusal_names_its_reason_priority_and_saturation_in_headers(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The same saturated team refusal carries ``rate_limited_by: model_per_team_saturated``, ``retry-after: 60``,
    ``rate_limit_type: requests``, ``x-litellm-priority: default`` and ``x-litellm-saturation: 0.20%``."""
    with _borrowing(gateway, tmp_path, 0.0) as (candidate, group, _team, key):
        responses: Final = [chat(candidate, group, key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429]
        names: Final = (
            "rate_limited_by",
            "retry-after",
            "rate_limit_type",
            "x-litellm-priority",
            "x-litellm-saturation",
        )
        assert {name: responses[2].headers.get(name) for name in names} == {
            "rate_limited_by": "model_per_team_saturated",
            "retry-after": str(WINDOW_SECONDS),
            "rate_limit_type": "requests",
            "x-litellm-priority": "default",
            "x-litellm-saturation": "0.20%",
        }


def test_an_idle_team_stops_at_its_borrow_cap_with_the_borrow_cap_text(gateway: Gateway, tmp_path: Path) -> None:
    """The default threshold keeps the model idle, so the team borrows past ``model_rpm_limit: 2`` up to its cap of
    ``2 x 2 = 4`` requests. The fifth request is refused with ``rate_limited_by: model_per_team_max_borrow_multiple``
    and a text that starts ``Team model_rpm_limit borrow cap reached for <m>: the team may use at most`` and contains
    ``Team: <team_id>``, ``Team model_rpm_limit: 2``, ``Borrow cap: 4`` and ``Model saturation: 0.4%``. UNVERIFIED:
    whether the multiple renders as ``2x`` or ``2.0x`` (``ModelLimitBorrower.limits.max_borrow_multiple``)."""
    with _borrowing(gateway, tmp_path, 0.5) as (candidate, group, team, key):
        responses: Final = [chat(candidate, group, key) for _ in range(5)]
        assert statuses(responses) == [200] * 4 + [429], [response.text for response in responses]
        message: Final = error_message(responses[4])
        assert message.startswith(f"Team model_rpm_limit borrow cap reached for {group}: the team may use at most "), (
            message
        )
        assert all(
            part in message
            for part in (
                f"Team: {team}, Rate limit type: requests, Team model_rpm_limit: {TEAM_RPM}, Borrow cap: 4, ",
                "Remaining: 0, Model saturation: 0.4%",
            )
        ), message
        assert responses[4].headers.get("rate_limited_by") == "model_per_team_max_borrow_multiple"
