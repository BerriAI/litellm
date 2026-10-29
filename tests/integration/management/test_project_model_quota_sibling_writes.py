import uuid
from pathlib import Path
from typing import Final

import httpx
import yaml

from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration._support.database import scratch_database
from tests.integration._support.process import owned_proxy

STRICT_ROUTE: Final = "vertex_ai/gemini-3.8-flash"
EXEMPT_ROUTE: Final = "runwayml/gen4_turbo"
TEAM_MODEL: Final = "clip-studio"


def _policy_config(directory: Path) -> Path:
    path: Final = directory / "project-quota-policy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": [],
                "general_settings": {
                    "store_model_in_db": True,
                    "enforce_project_model_quota": {"providers": {"vertex_ai": ["rpm", "tpm"], "runwayml": []}},
                },
            }
        )
    )
    return path


def _move_team_model(proxy: Gateway, model_id: str, route: str) -> httpx.Response:
    return proxy.request(
        "PATCH", f"/model/{model_id}/update", {"litellm_params": {"model": route, "rpm": 10, "tpm": 20000}}
    )


def _new_project_without_limits(proxy: Gateway, team_id: str, alias: str) -> httpx.Response:
    return proxy.request("POST", "/project/new", {"team_id": team_id, "project_alias": alias, "models": [TEAM_MODEL]})


def test_quota_checks_read_the_route_a_sibling_proxy_just_wrote(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _policy_config(tmp_path)
    with scratch_database() as database_url:
        environment: Final = {"DATABASE_URL": database_url, "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "3600"}
        with owned_proxy(gateway, tmp_path / "writer", environment, config=config) as writer:
            team_id: Final = string_value(writer.post("/team/new", {"team_alias": "media-team"})["team_id"])
            created: Final = writer.post(
                "/model/new",
                {
                    "model_name": TEAM_MODEL,
                    "litellm_params": {"model": EXEMPT_ROUTE},
                    "model_info": {"team_id": team_id},
                },
            )
            model_id: Final = string_value(object_value(created["model_info"])["id"])

            with owned_proxy(
                gateway,
                tmp_path / "sibling",
                {**environment, "DISABLE_SCHEMA_UPDATE": "true"},
                config=config,
                database_setup=(),
            ) as sibling_loaded_the_exempt_route:
                moved_to_strict: Final = _move_team_model(writer, model_id, STRICT_ROUTE)
                assert moved_to_strict.status_code == 200, moved_to_strict.text

                uncapped: Final = _new_project_without_limits(
                    sibling_loaded_the_exempt_route, team_id, f"uncapped-{uuid.uuid4().hex}"
                )
                assert uncapped.status_code == 400, uncapped.text
                assert f"{TEAM_MODEL} (model_rpm_limit and model_tpm_limit)" in uncapped.text

                moved_to_exempt: Final = _move_team_model(writer, model_id, EXEMPT_ROUTE)
                assert moved_to_exempt.status_code == 200, moved_to_exempt.text
                exempt_alias: Final = f"exempt-{uuid.uuid4().hex}"
                exempt_project: Final = _new_project_without_limits(writer, team_id, exempt_alias)
                assert exempt_project.status_code == 200, exempt_project.text

                stale_strict_write: Final = _move_team_model(sibling_loaded_the_exempt_route, model_id, STRICT_ROUTE)
                assert stale_strict_write.status_code == 400, stale_strict_write.text
                assert f"projects {exempt_alias} list '{TEAM_MODEL}'" in stale_strict_write.text
