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

MODEL: Final = "regional-model"
UPSTREAM_BY_REGION: Final = {"eu": "regional-eu-upstream", "us": "regional-us-upstream"}
CALLS: Final = 5


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("region-routing")
    with gateway_from_environment() as base:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            {
                "model_name": MODEL,
                "litellm_params": {
                    "model": f"openai/{upstream}",
                    "api_base": f"{base.upstream_url}/v1",
                    "api_key": "synthetic-region-key",
                    "region_name": region,
                },
            }
            for region, upstream in UPSTREAM_BY_REGION.items()
        ]
        config["router_settings"] = {**config["router_settings"], "enable_pre_call_checks": True}
        path: Final = directory / "region-routing.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(base, directory, {}, config=path) as proxy:
            yield proxy


@pytest.mark.parametrize("region", ["eu", "us"])
def test_an_end_users_allowed_region_pins_every_call_to_that_regions_deployment(
    candidate: Gateway, region: str
) -> None:
    with (
        candidate.scenario() as scenario,
        httpx.Client(base_url=candidate.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        candidate.post("/end_user/new", {"user_id": end_user, "allowed_model_region": region})
        scenario.cleanups.callback(candidate.post, "/end_user/delete", {"user_ids": [end_user]})
        key: Final = scenario.key(models=[MODEL])
        upstream.get("/__observations").raise_for_status()
        responses: Final = tuple(
            candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": MODEL,
                    "user": end_user,
                    "messages": [{"role": "user", "content": f"region {uuid.uuid4().hex}"}],
                },
                key=key,
            )
            for _ in range(CALLS)
        )
        assert [response.status_code for response in responses] == [200] * CALLS, [r.text for r in responses]
        assert [response.headers.get("x-litellm-model-region") for response in responses] == [region] * CALLS
        observed: Final = upstream.get("/__observations").json()["requests"]
        assert [request["body"]["model"] for request in observed] == [UPSTREAM_BY_REGION[region]] * CALLS
