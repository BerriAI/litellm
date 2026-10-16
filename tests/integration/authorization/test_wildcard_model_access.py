import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.process import owned_proxy
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(180)
SAMPLED_LISTED_IDS: Final = 3


def _deployment(model_name: str, model: str, upstream_url: str) -> dict[str, JsonValue]:
    return {
        "model_name": model_name,
        "litellm_params": {"model": model, "api_base": f"{upstream_url}/v1", "api_key": "synthetic-wildcard-key"},
    }


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("wildcard-access")
    with gateway_from_environment() as base:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            _deployment("*", "openai/*", base.upstream_url),
            _deployment("anthropic/*", "openai/*", base.upstream_url),
            _deployment("groq/*", "openai/*", base.upstream_url),
            _deployment("good-model", "openai/good-model-upstream", base.upstream_url),
        ]
        path: Final = directory / "wildcard-access.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(base, directory, {}, config=path) as proxy:
            yield proxy


@pytest.fixture
def upstream(candidate: Gateway) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=candidate.upstream_url, timeout=5, trust_env=False) as client:
        client.get("/__observations").raise_for_status()
        yield client


def _chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"wildcard {uuid.uuid4().hex}"}]},
        key=key,
    )


def _observed_models(upstream: httpx.Client) -> list[JsonValue]:
    return [request["body"]["model"] for request in upstream.get("/__observations").json()["requests"]]


def _listed_names(gateway: Gateway, path: str, field: str, key: str) -> list[str]:
    response: Final = gateway.request("GET", path, key=key)
    assert response.status_code == 200, response.text
    rows: Final = list_value(JSON_OBJECT.validate_json(response.content)["data"])
    return [string_value(object_value(row)[field]) for row in rows]


def test_an_all_models_key_reaches_a_model_served_only_by_the_catch_all_deployment(
    candidate: Gateway, upstream: httpx.Client
) -> None:
    with candidate.scenario() as scenario:
        key: Final = scenario.key(models=["*"])
        unlisted: Final = f"unlisted-{uuid.uuid4().hex}"
        response: Final = _chat(candidate, unlisted, key)
        assert response.status_code == 200, response.text
        assert _observed_models(upstream) == [unlisted]


def test_a_key_without_models_inherits_the_users_exact_and_wildcard_grants(
    candidate: Gateway, upstream: httpx.Client
) -> None:
    with candidate.scenario() as scenario:
        user_id: Final = scenario.user(models=["good-model", "anthropic/*"])
        key: Final = scenario.key(user_id=user_id, models=[])
        wildcard_model: Final = f"claude-{uuid.uuid4().hex}"
        assert _chat(candidate, f"anthropic/{wildcard_model}", key).status_code == 200
        assert _chat(candidate, "good-model", key).status_code == 200
        assert _observed_models(upstream) == [wildcard_model, "good-model-upstream"]
        denied: Final = tuple(
            _chat(candidate, outside, key)
            for outside in (f"groq/{wildcard_model}", f"bedrock/anthropic.{wildcard_model}")
        )
        assert [(response.status_code, response.json()["error"]["type"]) for response in denied] == [
            (403, "user_model_access_denied")
        ] * 2, [response.text for response in denied]
        assert _observed_models(upstream) == []
        assert _chat(candidate, f"groq/{wildcard_model}", candidate.key).status_code == 200
        assert _observed_models(upstream) == [wildcard_model]


def test_a_key_scoped_to_a_provider_wildcard_on_the_catch_all_lists_only_ids_that_reach_the_upstream_as_named(
    candidate: Gateway, upstream: httpx.Client
) -> None:
    with candidate.scenario() as scenario:
        key: Final = scenario.key(models=["openai/*"])
        listed: Final = _listed_names(candidate, "/v1/models", "id", key)
        sampled: Final = listed[:SAMPLED_LISTED_IDS]
        for model in sampled:
            _chat(candidate, model, key)
        assert _observed_models(upstream) == [model.removeprefix("openai/") for model in sampled]
        assert sorted(_listed_names(candidate, "/model/info", "model_name", key)) == sorted(listed)
