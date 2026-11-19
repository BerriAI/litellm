"""The response cache keys an answer by the router's model group on every endpoint the router serves.

The router names the model group in ``metadata`` on ``/v1/chat/completions`` and in ``litellm_metadata`` on
``/v1/responses`` and ``/v1/messages``. Every cell gives each model group its own scripted upstream scenario on one
shared provider model, fills one group's entry, and reads which scenario the identical request reached next and which
answer the caller got: an answer served from the cache repeats the id of the request that filled it.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import pytest
import yaml
from anthropic import Anthropic
from openai import OpenAI
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value
from tests.integration._support.process import OwnedState, graceful_stop_seconds, owned_proxy_process
from tests.integration._support.upstream import ScenarioHandle
from tests.integration.caching.response_cache_case import (
    CHAT,
    CHAT_MODEL,
    MESSAGE,
    MESSAGES_MODEL,
    PROVIDER_KEY,
    RESPONSE,
    Upstream,
    await_entry,
    json_response,
    prompt,
    redis_store,
    scripted,
)

Ask: TypeAlias = Callable[[Gateway, str, str, Mapping[str, JsonValue]], str]


def _root(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _metadata(metadata: Mapping[str, JsonValue]) -> dict[str, JsonValue] | None:
    return {"metadata": dict(metadata)} if metadata else None


def _chat(gateway: Gateway, model: str, text: str, metadata: Mapping[str, JsonValue]) -> str:
    with OpenAI(base_url=f"{_root(gateway)}/v1", api_key=gateway.key, max_retries=0) as client:
        return client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": text}], extra_body=_metadata(metadata)
        ).id


def _responses(gateway: Gateway, model: str, text: str, metadata: Mapping[str, JsonValue]) -> str:
    with OpenAI(base_url=f"{_root(gateway)}/v1", api_key=gateway.key, max_retries=0) as client:
        identity: Final = client.responses.create(model=model, input=text, extra_body=_metadata(metadata)).output[0].id
    assert identity is not None
    return identity


def _messages(gateway: Gateway, model: str, text: str, metadata: Mapping[str, JsonValue]) -> str:
    with Anthropic(base_url=_root(gateway), api_key=gateway.key, max_retries=0) as client:
        return client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": text}],
            extra_body=_metadata(metadata),
        ).id


@dataclass(frozen=True, slots=True)
class Endpoint:
    provider_model: str
    sibling_model: str
    answer: Mapping[str, JsonValue]
    ask: Ask


@dataclass(frozen=True, slots=True)
class Group:
    name: str
    upstream: ScenarioHandle


ENDPOINTS: Final = {
    "chat_completions": Endpoint(CHAT_MODEL, "openai/gpt-5.6", CHAT, _chat),
    "responses": Endpoint(CHAT_MODEL, "openai/gpt-5.6", RESPONSE, _responses),
    "messages": Endpoint(MESSAGES_MODEL, "anthropic/claude-sonnet-5-5", MESSAGE, _messages),
}


def _group(scenario: Scenario, endpoint: Endpoint, name: str | None = None) -> Group:
    handle: Final = scripted(scenario, json_response(endpoint.answer))
    return Group(scenario.model(name=name, model=endpoint.provider_model, api_base=handle.api_base()), handle)


@pytest.mark.parametrize("endpoint", ENDPOINTS.values(), ids=ENDPOINTS.keys())
def test_identical_requests_to_two_groups_on_one_model_each_reach_their_own_deployment(
    gateway: Gateway, endpoint: Endpoint
) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario:
        first: Final = _group(scenario, endpoint)
        second: Final = _group(scenario, endpoint)
        filled: Final = endpoint.ask(gateway, first.name, text, {})
        await_entry(store, filled)
        answered: Final = endpoint.ask(gateway, second.name, text, {})
        assert answered != filled, f"{second.name} was served the answer cached for {first.name}"
        assert upstream.calls(second.upstream.scenario_id) == 1
        assert endpoint.ask(gateway, first.name, text, {}) == filled
        assert upstream.calls(first.upstream.scenario_id) == 1


PLANTS: Final[Mapping[str, Callable[[str], Mapping[str, JsonValue]]]] = {
    "model_group": lambda target: {"model_group": target},
    "caching_groups": lambda target: {"caching_groups": [target]},
}


@pytest.mark.parametrize("plant", PLANTS.values(), ids=PLANTS.keys())
@pytest.mark.parametrize("endpoint", ENDPOINTS.values(), ids=ENDPOINTS.keys())
def test_a_caller_naming_another_group_in_metadata_never_fills_that_groups_entry(
    gateway: Gateway, endpoint: Endpoint, plant: Callable[[str], Mapping[str, JsonValue]]
) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario:
        caller: Final = _group(scenario, endpoint)
        target: Final = _group(scenario, endpoint, name=f"{caller.name}-premium")
        planted: Final = endpoint.ask(gateway, caller.name, text, plant(target.name))
        assert upstream.calls(caller.upstream.scenario_id) == 1
        await_entry(store, planted)
        answered: Final = endpoint.ask(gateway, target.name, text, {})
        assert answered != planted, f"{target.name} was served the answer {caller.name} planted under its name"
        assert upstream.calls(target.upstream.scenario_id) == 1
        await_entry(store, answered)
        assert endpoint.ask(gateway, target.name, text, {}) == answered


def _caching_groups_config(directory: Path, endpoint: Endpoint, first: Group, second: Group) -> Path:
    rig: Final = JSON_OBJECT.validate_python(
        yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    )
    router_settings: Final = object_value(rig.get("router_settings") or {})
    config: Final = {
        **rig,
        "model_list": [
            {
                "model_name": group.name,
                "litellm_params": {"model": model, "api_key": PROVIDER_KEY, "api_base": group.upstream.api_base()},
            }
            for group, model in ((first, endpoint.provider_model), (second, endpoint.sibling_model))
        ],
        "router_settings": {**router_settings, "caching_groups": [[first.name, second.name]]},
    }
    path: Final = directory / f"proxy_config-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _configured_group(scenario: Scenario, endpoint: Endpoint) -> Group:
    return Group(f"cache-group-{uuid.uuid4().hex[:8]}", scripted(scenario, json_response(endpoint.answer)))


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
@pytest.mark.parametrize("endpoint", ENDPOINTS.values(), ids=ENDPOINTS.keys())
def test_groups_in_one_caching_group_on_different_models_share_one_entry(
    gateway: Gateway, endpoint: Endpoint, tmp_path: Path
) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario:
        first: Final = _configured_group(scenario, endpoint)
        second: Final = _configured_group(scenario, endpoint)
        config: Final = _caching_groups_config(tmp_path, endpoint, first, second)
        with owned_proxy_process(
            gateway, tmp_path, {}, config=config, workers=1, state=OwnedState.SHARED_WITH_WORKER
        ) as owned:
            filled: Final = endpoint.ask(owned.gateway, first.name, text, {})
            await_entry(store, filled)
            assert endpoint.ask(owned.gateway, second.name, text, {}) == filled
            assert upstream.calls(second.upstream.scenario_id) == 0
            assert upstream.calls(first.upstream.scenario_id) == 1
