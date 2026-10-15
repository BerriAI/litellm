import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy

pytestmark: Final = pytest.mark.timeout(180)

MODEL: Final = "tagged-model"
DEPLOYMENT_BY_TAG: Final = {"teamA": "team-a-deployment", "teamB": "team-b-deployment"}
CALLS: Final = 5


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("team-tag-routing")
    with gateway_from_environment() as base:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            {
                "model_name": MODEL,
                "litellm_params": {
                    "model": f"openai/{deployment}",
                    "api_base": f"{base.upstream_url}/v1",
                    "api_key": "synthetic-tag-key",
                    "tags": [tag],
                },
                "model_info": {"id": deployment},
            }
            for tag, deployment in DEPLOYMENT_BY_TAG.items()
        ]
        config["router_settings"] = {**config["router_settings"], "enable_tag_filtering": True}
        path: Final = directory / "team-tag-routing.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(base, directory, {}, config=path) as proxy:
            yield proxy


@pytest.mark.parametrize("tag", ["teamA", "teamB"])
def test_a_teams_tags_route_every_call_of_its_keys_to_the_matching_deployment(candidate: Gateway, tag: str) -> None:
    with (
        candidate.scenario() as scenario,
        httpx.Client(base_url=candidate.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        team_id: Final = scenario.team(tags=[tag])
        key: Final = scenario.key(team_id=team_id)
        upstream.get("/__observations").raise_for_status()
        responses: Final = tuple(
            candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": MODEL, "messages": [{"role": "user", "content": f"tagged {uuid.uuid4().hex}"}]},
                key=key,
            )
            for _ in range(CALLS)
        )
        assert [response.status_code for response in responses] == [200] * CALLS, [r.text for r in responses]
        assert [response.headers.get("x-litellm-model-id") for response in responses] == [
            DEPLOYMENT_BY_TAG[tag]
        ] * CALLS
        observed: Final = upstream.get("/__observations").json()["requests"]
        assert [request["body"]["model"] for request in observed] == [DEPLOYMENT_BY_TAG[tag]] * CALLS
