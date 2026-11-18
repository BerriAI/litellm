import uuid
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value

COST_PER_REQUEST: Final = 20 * 0.001 + 20 * 0.002
FIRST_BURST: Final = 6
SECOND_BURST: Final = 4


@dataclass(frozen=True, slots=True)
class Owners:
    key: str
    team_id: str
    user_id: str
    organization_id: str


def _reported(gateway: Gateway, owners: Owners) -> tuple[float, float, float, float]:
    key_info: Final = object_value(gateway.get("/key/info", {"key": owners.key})["info"])
    team_info: Final = object_value(gateway.get("/team/info", {"team_id": owners.team_id})["team_info"])
    user_info: Final = object_value(gateway.get("/user/info", {"user_id": owners.user_id})["user_info"])
    organization: Final = gateway.get("/organization/info", {"organization_id": owners.organization_id})
    return (
        float(str(key_info["spend"])),
        float(str(team_info["spend"])),
        float(str(user_info["spend"])),
        float(str(organization["spend"])),
    )


def _matches(observed: tuple[float, float, float, float], expected: float) -> bool:
    return all(value == pytest.approx(expected, rel=1e-9) for value in observed)


def _burst(
    gateway: Gateway, model: str, owners: Owners, requests: int, total_requests: int
) -> tuple[float, float, float, float]:
    usage: Final = tuple(
        object_value(gateway.chat(model, key=owners.key, text=f"burst {uuid.uuid4().hex}")["usage"])
        for _ in range(requests)
    )
    assert [(entry["prompt_tokens"], entry["completion_tokens"]) for entry in usage] == [(20, 20)] * requests
    return eventually(
        lambda: _reported(gateway, owners),
        lambda observed: _matches(observed, total_requests * COST_PER_REQUEST),
        seconds=70,
        return_last_on_timeout=True,
    )


def test_every_burst_rolls_up_exactly_to_key_team_user_and_organization(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        organization_id: Final = scenario.organization()
        team_id: Final = scenario.team(organization_id=organization_id, models=[model])
        user_id: Final = scenario.user(user_role="internal_user")
        owners: Final = Owners(
            key=scenario.key(user_id=user_id, team_id=team_id, models=[model]),
            team_id=team_id,
            user_id=user_id,
            organization_id=organization_id,
        )
        first: Final = _burst(gateway, model, owners, FIRST_BURST, FIRST_BURST)
        assert first == pytest.approx((FIRST_BURST * COST_PER_REQUEST,) * 4, rel=1e-9), first
        both: Final = _burst(gateway, model, owners, SECOND_BURST, FIRST_BURST + SECOND_BURST)
        assert both == pytest.approx(((FIRST_BURST + SECOND_BURST) * COST_PER_REQUEST,) * 4, rel=1e-9), both
