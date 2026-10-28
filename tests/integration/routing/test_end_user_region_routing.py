import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(180)

MODEL: Final = "regional-model"
UPSTREAM_BY_REGION: Final = {"eu": "regional-eu-upstream", "us": "regional-us-upstream"}
CALLS: Final = 5
UNCOUNTABLE_MODEL: Final = "regional-uncountable-model"
PINNED_REGION: Final = "eu"
WEIGHT_BY_REGION: Final = {"eu": 1, "us": 99}
SILENT_WAV: Final = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("region-routing")
    with gateway_from_environment() as base:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            *(
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
            ),
            *(
                {
                    "model_name": UNCOUNTABLE_MODEL,
                    "litellm_params": {
                        "model": f"openai/{upstream}",
                        "api_base": f"{base.upstream_url}/v1",
                        "api_key": "synthetic-region-key",
                        "region_name": region,
                        "weight": WEIGHT_BY_REGION[region],
                    },
                    "model_info": {"id": f"{UNCOUNTABLE_MODEL}-{region}", "max_input_tokens": 128000},
                }
                for region, upstream in UPSTREAM_BY_REGION.items()
            ),
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
        upstream.delete("/__observations").raise_for_status()
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


def _uncountable_chat(marker: str, end_user: str) -> tuple[str, dict[str, JsonValue]]:
    return (
        "/v1/chat/completions",
        {
            "model": UNCOUNTABLE_MODEL,
            "user": end_user,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"region {marker}"},
                        {"type": "input_audio", "input_audio": {"data": SILENT_WAV, "format": "wav"}},
                    ],
                }
            ],
        },
    )


def _uncountable_embedding(marker: str, end_user: str) -> tuple[str, dict[str, JsonValue]]:
    return "/v1/embeddings", {"model": UNCOUNTABLE_MODEL, "user": end_user, "input": [f"region {marker}", marker]}


@pytest.mark.parametrize(
    ("build", "recorded_chat_calls"),
    [
        pytest.param(_uncountable_chat, CALLS, id="chat-input-audio"),
        pytest.param(_uncountable_embedding, 0, id="embeddings-list-input"),
    ],
)
def test_an_uncountable_prompt_still_pins_an_end_user_to_their_region(
    candidate: Gateway, build: Callable[[str, str], tuple[str, dict[str, JsonValue]]], recorded_chat_calls: int
) -> None:
    with (
        candidate.scenario() as scenario,
        httpx.Client(base_url=candidate.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        candidate.post("/end_user/new", {"user_id": end_user, "allowed_model_region": PINNED_REGION})
        scenario.cleanups.callback(candidate.post, "/end_user/delete", {"user_ids": [end_user]})
        key: Final = scenario.key(models=[UNCOUNTABLE_MODEL])
        upstream.delete("/__observations").raise_for_status()
        requests: Final = tuple(build(uuid.uuid4().hex, end_user) for _ in range(CALLS))
        responses: Final = tuple(candidate.request("POST", path, body, key=key) for path, body in requests)
        assert [response.status_code for response in responses] == [200] * CALLS, [r.text for r in responses]
        pinned_deployment: Final = f"{UNCOUNTABLE_MODEL}-{PINNED_REGION}"
        assert [response.headers.get("x-litellm-model-id") for response in responses] == [pinned_deployment] * CALLS
        observed: Final = upstream.get("/__observations").json()["requests"]
        pinned_upstream: Final = UPSTREAM_BY_REGION[PINNED_REGION]
        assert [request["body"]["model"] for request in observed] == [pinned_upstream] * recorded_chat_calls
