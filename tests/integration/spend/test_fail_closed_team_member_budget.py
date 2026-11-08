import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process

_MEMBERSHIP_ROWS: Final = 'SELECT spend FROM "LiteLLM_TeamMembership" WHERE user_id = %s AND team_id = %s'


def _fail_closed_config(directory: Path) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "general_settings": {**object_value(base["general_settings"]), "fail_closed_budget_enforcement": True},
    }
    path: Final = directory / f"fail-closed-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def fail_closed_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("fail-closed-team-member")
    with ExitStack() as stack:
        rig_gateway: Final = stack.enter_context(gateway_from_environment())
        owned: Final = stack.enter_context(
            owned_proxy_process(rig_gateway, directory, {}, config=_fail_closed_config(directory))
        )
        yield owned.gateway


def test_team_member_without_membership_row_is_admitted_under_default_member_budget(
    fail_closed_gateway: Gateway,
) -> None:
    with fail_closed_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_id: Final = scenario.team(models=[model], team_member_budget=100.0)
        user_id: Final = scenario.user(max_budget=100.0)
        key: Final = scenario.key(team_id=team_id, user_id=user_id, models=[model])
        assert read_rows(_MEMBERSHIP_ROWS, (user_id, team_id)) == []

        response: Final = fail_closed_gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "integration control"}]},
            key=key,
        )

        assert response.status_code == 200, response.text
        assert object_value(JSON_OBJECT.validate_json(response.content)["usage"])["total_tokens"]
