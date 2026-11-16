import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
import yaml

from tests.integration._support.client import Gateway, gateway_from_environment, object_value, string_value
from tests.integration._support.process import owned_proxy

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

_MAX_BUDGET_IN_TEAM: Final = 7.5


@contextmanager
def _gateway_with_default_team(directory: Path, team_id: str) -> Generator[Gateway]:
    config: Final = directory / "default_team_proxy_config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_list": [],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                },
                "litellm_settings": {
                    "default_internal_user_params": {
                        "teams": [{"team_id": team_id, "max_budget_in_team": _MAX_BUDGET_IN_TEAM}]
                    }
                },
            }
        )
    )
    with gateway_from_environment() as rig, owned_proxy(rig, directory, {}, config=config) as gateway:
        yield gateway


def test_a_new_user_keeps_the_default_team_cap_when_the_team_member_budget_row_was_deleted(tmp_path: Path) -> None:
    team_id: Final = f"default-team-{uuid.uuid4().hex[:12]}"
    with _gateway_with_default_team(tmp_path, team_id) as gateway, gateway.scenario() as scenario:
        team: Final = scenario.team(team_id=team_id, team_member_budget=25)
        team_info: Final = object_value(gateway.get(f"/team/info?team_id={team}")["team_info"])
        team_member_budget_id: Final = string_value(object_value(team_info["metadata"])["team_member_budget_id"])
        deleted: Final = gateway.request("POST", "/budget/delete", {"id": team_member_budget_id})
        assert deleted.status_code == 200, deleted.text

        user: Final = scenario.user()

        memberships: Final = gateway.get(f"/team/info?team_id={team}")["team_memberships"]
        assert isinstance(memberships, list), memberships
        caps: Final = [
            object_value(object_value(membership)["litellm_budget_table"])["max_budget"]
            for membership in memberships
            if object_value(membership)["user_id"] == user
        ]
        assert caps == [_MAX_BUDGET_IN_TEAM], memberships
