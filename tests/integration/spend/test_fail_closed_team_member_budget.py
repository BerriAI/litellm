import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, object_value, string_value
from integration._support.process import owned_proxy_process
from pydantic import JsonValue


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


def _chat(gateway: Gateway, model: str, key: str) -> tuple[int, dict[str, JsonValue]]:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "integration control"}]},
        key=key,
    )
    return response.status_code, JSON_OBJECT.validate_json(response.content)


def test_team_member_without_membership_row_is_told_to_be_added_then_admitted(fail_closed_gateway: Gateway) -> None:
    with fail_closed_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_id: Final = scenario.team(models=[model], team_member_budget=100.0)
        user_id: Final = scenario.user(max_budget=100.0)
        key: Final = scenario.key(team_id=team_id, user_id=user_id, models=[model])

        status, body = _chat(fail_closed_gateway, model, key)

        assert status == 403, body
        message: Final = string_value(object_value(body["error"])["message"])
        assert user_id in message and team_id in message and "/team/member_add" in message

        fail_closed_gateway.post(
            "/team/member_add", {"team_id": team_id, "member": {"role": "user", "user_id": user_id}}
        )

        admitted_status, admitted = _chat(fail_closed_gateway, model, key)

        assert admitted_status == 200, admitted
        assert object_value(admitted["usage"])["total_tokens"]
