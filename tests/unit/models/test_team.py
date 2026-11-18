from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy._types import NewTeamRequest, UpdateTeamRequest


@pytest.mark.parametrize("request_type", [NewTeamRequest, UpdateTeamRequest])
@pytest.mark.parametrize("percent", [-1, 1001, float("inf"), float("nan")])
def test_team_margin_rejects_invalid_chargeback_percentages(
    request_type: type[NewTeamRequest] | type[UpdateTeamRequest], percent: float
) -> None:
    with pytest.raises(ValidationError, match="cost_margin_percent"):
        request_type(team_id="team", cost_margin_percent=percent)


@pytest.mark.parametrize("percent", [None, 0.0, 2.0, 1000.0])
def test_team_update_preserves_an_explicit_margin_or_clear(percent: float | None) -> None:
    request: Final = UpdateTeamRequest(team_id="team", cost_margin_percent=percent)
    assert request.model_dump(exclude_unset=True) == {"team_id": "team", "cost_margin_percent": percent}


def test_team_update_does_not_clear_an_omitted_margin() -> None:
    assert UpdateTeamRequest(team_id="team").model_dump(exclude_unset=True) == {"team_id": "team"}
