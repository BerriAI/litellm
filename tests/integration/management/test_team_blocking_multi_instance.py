from pathlib import Path
from typing import Final

from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration._support.process import owned_proxy


def test_team_blocking_is_visible_across_proxy_instances(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = Path("tests/integration/proxy_config.yaml")
    with owned_proxy(gateway, tmp_path / "first", {}, config=config) as first:
        with owned_proxy(gateway, tmp_path / "second", {}, config=config) as second:
            team: Final = string_value(
                first.post("/team/new", {"team_alias": "multi-instance-blocking"})["team_id"]
            )
            try:
                key: Final = string_value(
                    first.post("/key/generate", {"team_id": team})["key"]
                )
                initial: Final = object_value(
                    first.get("/team/info", {"team_id": team})["team_info"]
                )
                assert initial["blocked"] is False

                second.post("/team/update", {"team_id": team, "blocked": True})

                updated: Final = object_value(
                    first.get("/team/info", {"team_id": team})["team_info"]
                )
                assert updated["blocked"] is True

                response: Final = first.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": "unconfigured-model", "messages": [{"role": "user", "content": "blocked"}]},
                    key=key,
                )
                assert response.status_code == 401, response.text
                assert "blocked" in response.text.lower()
            finally:
                second.post("/team/delete", {"team_ids": [team]})
