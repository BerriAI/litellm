import uuid
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(180)

_PRICED_DEPLOYMENT_ID: Final = f"priced-deployment-{uuid.uuid4().hex}"
_REGISTERED_AI21_CHAT_MODEL: Final = f"jamba-house-{uuid.uuid4().hex}"


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("wildcard-listing")
    with gateway_from_environment() as base:
        credentials: Final = {"api_base": f"{base.upstream_url}/v1", "api_key": "synthetic-listing-key"}
        config: Final = {
            **yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()),
            "model_list": [
                {"model_name": "openai/*", "litellm_params": {"model": "openai/*", **credentials}},
                {
                    "model_name": "priced-chat",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                        **credentials,
                    },
                    "model_info": {"id": _PRICED_DEPLOYMENT_ID},
                },
                {"model_name": "ai21/*", "litellm_params": {"model": "ai21/*", **credentials}},
                {
                    "model_name": "house-jamba",
                    "litellm_params": {"model": f"ai21/{_REGISTERED_AI21_CHAT_MODEL}", **credentials},
                    "model_info": {"litellm_provider": "ai21", "mode": "chat"},
                },
            ],
        }
        path: Final = directory / "wildcard-listing.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(base, directory, {}, config=path) as proxy:
            yield proxy


def _listed_ids(gateway: Gateway) -> list[JsonValue]:
    data: Final = gateway.get("/v1/models")["data"]
    assert isinstance(data, list), data
    return [entry["id"] for entry in data if isinstance(entry, dict)]


def test_wildcard_listing_after_a_priced_call_names_each_model_once_and_no_deployment_id(candidate: Gateway) -> None:
    candidate.chat("priced-chat")

    listed: Final = _listed_ids(candidate)

    assert "openai/gpt-4o-mini" in listed and "priced-chat" in listed, listed
    assert {model: count for model, count in Counter(listed).items() if count > 1} == {}, listed
    assert [model for model in listed if _PRICED_DEPLOYMENT_ID in str(model)] == [], listed


def test_ai21_wildcard_listing_names_a_chat_model_registered_from_the_config(candidate: Gateway) -> None:
    listed: Final = _listed_ids(candidate)

    assert f"ai21/{_REGISTERED_AI21_CHAT_MODEL}" in listed, listed
