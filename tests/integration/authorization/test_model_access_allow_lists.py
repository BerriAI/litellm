from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value, string_value
from tests.integration._support.provider import PROVIDER_URL, SharedProvider
from tests.integration._support.wire import Reply

_Denial = Literal["key_model_access_denied", "team_model_access_denied"]


@dataclass(frozen=True, slots=True)
class _Models:
    gpt: str
    gpt_mini: str
    claude: str
    bedrock_claude: str
    bedrock_titan: str


@dataclass(frozen=True, slots=True)
class _AccessCase:
    name: str
    allowed: Sequence[str] | None
    requested: str
    served: bool


_CASES: Final = (
    _AccessCase("openai_wildcard_denies_anthropic", ["openai/*"], "claude", False),
    _AccessCase("exact_name_allows_itself", ["gpt"], "gpt", True),
    _AccessCase("provider_wildcard_allows_bedrock", ["bedrock/*"], "bedrock_claude", True),
    _AccessCase("family_wildcard_allows_its_family", ["bedrock/anthropic.*"], "bedrock_claude", True),
    _AccessCase("family_wildcard_denies_another_family", ["bedrock/anthropic.*"], "bedrock_titan", False),
    _AccessCase("unset_models_allow_everything", None, "gpt", True),
    _AccessCase("empty_models_allow_everything", [], "gpt", True),
)


def _deployment(scenario: Scenario, gateway: Gateway, name: str) -> str:
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_base": f"{PROVIDER_URL}/v1",
                "api_key": "sk-fixture",
            },
            "model_info": {"id": f"access-{uuid.uuid4().hex}"},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return name


def _models(scenario: Scenario, gateway: Gateway) -> _Models:
    tag: Final = uuid.uuid4().hex[:10]
    return _Models(
        gpt=_deployment(scenario, gateway, f"openai/gpt-{tag}"),
        gpt_mini=_deployment(scenario, gateway, f"openai/gpt-mini-{tag}"),
        claude=_deployment(scenario, gateway, f"anthropic/claude-{tag}"),
        bedrock_claude=_deployment(scenario, gateway, f"bedrock/anthropic.claude-{tag}"),
        bedrock_titan=_deployment(scenario, gateway, f"bedrock/amazon.titan-{tag}"),
    )


def _pick(models: _Models, alias: str) -> str:
    return string_value(getattr(models, alias))


def _completion() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "scripted"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        ).encode()
    )


def _served(gateway: Gateway, provider: SharedProvider, key: str, model: str) -> None:
    provider.expect(_completion())
    body: Final = gateway.chat(model, key=key, text=f"access {uuid.uuid4().hex}")
    assert object_value(object_value(body["choices"][0])["message"])["content"] == "scripted", body
    assert len(provider.received()) == 1


def _denied(gateway: Gateway, provider: SharedProvider, key: str, model: str, denial: _Denial) -> None:
    refused: Final = gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}]}, key=key
    )
    assert refused.status_code == 403, f"{refused.status_code} {refused.text}"
    error: Final = object_value(JSON_OBJECT.validate_json(refused.content)["error"])
    assert error["type"] == denial, refused.text
    assert error["param"] == "model", refused.text
    assert error["code"] == "403", refused.text
    message: Final = string_value(error["message"])
    assert "is not available for this API key" in message, message
    assert "not allowed to access model" not in message, message
    assert provider.received() == ()


@pytest.mark.parametrize("case", _CASES, ids=[case.name for case in _CASES])
def test_a_key_model_allow_list_decides_which_models_it_reaches(
    gateway: Gateway, provider: SharedProvider, case: _AccessCase
) -> None:
    with gateway.scenario() as scenario:
        models: Final = _models(scenario, gateway)
        allowed: Final = (
            None
            if case.allowed is None
            else [_pick(models, item) if hasattr(models, item) else item for item in case.allowed]
        )
        key: Final = scenario.key(models=allowed)
        requested: Final = _pick(models, case.requested)
        if case.served:
            _served(gateway, provider, key, requested)
        else:
            _denied(gateway, provider, key, requested, "key_model_access_denied")


def test_widening_a_key_allow_list_to_a_wildcard_takes_effect_on_the_next_request(
    gateway: Gateway, provider: SharedProvider
) -> None:
    with gateway.scenario() as scenario:
        models: Final = _models(scenario, gateway)
        key: Final = scenario.key(models=[models.gpt])
        _served(gateway, provider, key, models.gpt)
        _denied(gateway, provider, key, models.gpt_mini, "key_model_access_denied")
        gateway.post("/key/update", {"key": key, "models": ["openai/*"]})
        _served(gateway, provider, key, models.gpt)
        _served(gateway, provider, key, models.gpt_mini)
        _denied(gateway, provider, key, models.claude, "key_model_access_denied")


def test_a_team_allow_list_denies_its_keys_a_model_outside_it(gateway: Gateway, provider: SharedProvider) -> None:
    with gateway.scenario() as scenario:
        models: Final = _models(scenario, gateway)
        team: Final = scenario.team(models=["openai/*"])
        key: Final = scenario.key(team_id=team)
        _served(gateway, provider, key, models.gpt)
        _denied(gateway, provider, key, models.claude, "team_model_access_denied")


def test_widening_a_team_allow_list_takes_effect_for_its_keys_on_the_next_request(
    gateway: Gateway, provider: SharedProvider
) -> None:
    with gateway.scenario() as scenario:
        models: Final = _models(scenario, gateway)
        team: Final = scenario.team(models=[models.gpt])
        key: Final = scenario.key(team_id=team)
        _served(gateway, provider, key, models.gpt)
        _denied(gateway, provider, key, models.gpt_mini, "team_model_access_denied")
        gateway.post("/team/update", {"team_id": team, "models": ["openai/*"]})
        _served(gateway, provider, key, models.gpt)
        _served(gateway, provider, key, models.gpt_mini)
        _denied(gateway, provider, key, models.claude, "team_model_access_denied")
