"""A proxy booting against keys and teams that still rely on the deprecated empty models defaults logs how many.

The seeding proxy owns a fresh database, creates keys and teams through the management API in every shape the
deprecation covers plus controls that must not count, and a second proxy booted against that database is the boot
whose log an admin reads before upgrading.
"""

import uuid
from pathlib import Path
from typing import Final

import psycopg

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, string_value
from tests.integration._support.process import OwnedState, owned_proxy_process

LISTED_MODEL: Final = "openai/gpt-5.4-mini"
WARNING_HEADER: Final = "Deprecated model access defaults are still in use."


def _team(gateway: Gateway, models: list[str], access_group_ids: tuple[str, ...] = ()) -> str:
    created: Final = gateway.post(
        "/team/new",
        {
            "team_alias": f"empty-models-{uuid.uuid4().hex}",
            "models": models,
            "access_group_ids": list(access_group_ids),
        },
    )
    return string_value(created["team_id"])


def _access_group(gateway: Gateway) -> str:
    created: Final = gateway.request(
        "POST",
        "/v1/access_group",
        {"access_group_name": f"empty-models-{uuid.uuid4().hex}", "access_model_names": [LISTED_MODEL]},
    )
    assert created.status_code == 201, created.text
    return string_value(JSON_OBJECT.validate_json(created.content)["access_group_id"])


def _seed(gateway: Gateway, database_url: str) -> None:
    empty_team: Final = _team(gateway, [])
    listed_team: Final = _team(gateway, [LISTED_MODEL])
    access_group: Final = _access_group(gateway)
    _team(gateway, [], access_group_ids=(access_group,))
    keys: Final = (
        {"models": []},
        {"models": []},
        {"models": ["all-team-models"]},
        {"models": [], "team_id": empty_team},
        {"models": [], "team_id": listed_team},
        {"models": [], "team_id": listed_team},
        {"models": ["all-team-models"], "team_id": listed_team},
        {"models": [LISTED_MODEL]},
        {"models": ["all-proxy-models"]},
        {"models": [], "access_group_ids": [access_group]},
        {"models": ["all-team-models"], "access_group_ids": [access_group]},
        {"models": [], "team_id": listed_team, "access_group_ids": [access_group]},
        {"models": [], "key_type": "management"},
        {"models": [], "key_type": "read_only"},
        {"models": [], "team_id": listed_team, "key_type": "management"},
    )
    for body in keys:
        gateway.post("/key/generate", body)
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(
            'INSERT INTO "LiteLLM_VerificationToken" (token, team_id, models) VALUES (%s, %s, %s)',
            (f"ui-session-{uuid.uuid4().hex}", "litellm-dashboard", []),
        )


def test_boot_logs_the_keys_and_teams_on_the_deprecated_empty_models_defaults(
    gateway: Gateway, tmp_path: Path
) -> None:
    with owned_proxy_process(gateway, tmp_path / "seed", {}, state=OwnedState.PRIVATE) as seeding:
        database_url: Final = seeding.gateway.database_url
        assert database_url is not None
        _seed(seeding.gateway, database_url)
        with owned_proxy_process(
            gateway, tmp_path / "reboot", {"DATABASE_URL": database_url}, database_setup=()
        ) as rebooted:
            log: Final = eventually(lambda: rebooted.log.read_text(), lambda text: WARNING_HEADER in text, seconds=60)

    assert "2 keys without a team have an empty models list, which means every proxy model." in log
    assert "1 key without a team lists all-team-models, which means every proxy model." in log
    assert "3 team keys have an empty models list, which means all of the team's models." in log
    assert "1 team has an empty models list, which means every proxy model." in log
    assert "2 keys with access groups have an empty models list." in log
    assert "1 team with access groups has an empty models list." in log
    assert "Keys without a team that list all-team-models have no team to inherit from" in log
