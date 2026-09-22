import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("other.observability.phoenix.team_metadata_project_name_is_accepted_and_routes_traces")
def test_team_metadata_phoenix_project_name_is_accepted_and_routes_traces(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "phoenix" + uuid.uuid4().hex
    project: Final = "team-project-" + marker
    phoenix_secret: Final = "synthetic-phoenix-secret-" + marker

    def collector(request: Request) -> Reply:
        assert request.target == "/v1/traces", request.target
        return Reply()

    with wire_server(collector) as phoenix:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["litellm_settings"].update({"callbacks": ["arize_phoenix"]})
        path: Final = tmp_path / "phoenix.yaml"
        path.write_text(yaml.safe_dump(config))
        with (
            owned_proxy(
                gateway,
                tmp_path,
                {
                    "LITELLM_OTEL_V2": "true",
                    "PHOENIX_COLLECTOR_HTTP_ENDPOINT": phoenix.url + "/v1/traces",
                    "PHOENIX_API_KEY": phoenix_secret,
                    "PHOENIX_PROJECT_NAME": "default-project-" + marker,
                },
                config=path,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model()
            team: Final = scenario.team()
            updated: Final = candidate.request(
                "POST", "/team/update", {"team_id": team, "metadata": {"phoenix_project_name": project}}
            )
            assert updated.status_code == 200, updated.text
            assert updated.json()["data"]["metadata"] == {"phoenix_project_name": project}, updated.text
            key: Final = scenario.key(team_id=team, models=[model])
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
                key=key,
            )
            assert response.status_code == 200, response.text
            exports: Final[list[Request]] = []

            def routed_exports() -> tuple[Request, ...]:
                exports.extend(phoenix.drain())
                return tuple(request for request in exports if "x-project-name" in request.headers)

            routed: Final = eventually(routed_exports, lambda values: len(values) >= 1, seconds=30)
            assert {request.headers["x-project-name"] for request in routed} == {project}, routed
            assert {request.headers["authorization"] for request in routed} == {f"Bearer {phoenix_secret}"}, routed
            assert {request.headers["authorization"] for request in exports} == {f"Bearer {phoenix_secret}"}, exports
