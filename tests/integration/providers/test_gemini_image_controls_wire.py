import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import product
from typing import Final, Literal, TypeAlias

import httpx
import pytest
from _pytest.mark.structures import ParameterSet
from integration._support.client import JSON_OBJECT, Gateway, Scenario, list_value, object_value, string_value
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

DropParams: TypeAlias = Literal["request", "deployment"]
ConfigCheck: TypeAlias = Callable[[Mapping[str, JsonValue]], bool]

_CHAT: Final = "/v1/chat/completions"
_RESPONSES: Final = "/v1/responses"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "global"
_TOKEN_PATH: Final = "/_oauth/token"
_GEMINI_KEY: Final = "synthetic-gemini-key"
_VERTEX_NANO: Final = "vertex_ai/gemini-nano-banana-2.1"
_STUDIO_NANO: Final = "gemini/gemini-nano-banana-2.1"
_VERTEX_FLASH_IMAGE: Final = "vertex_ai/gemini-3.1-flash-image"
_STUDIO_FLASH_IMAGE: Final = "gemini/gemini-3.1-flash-image"
_VERTEX_FLASH: Final = "vertex_ai/gemini-3.5-flash"
_VERTEX_PRO: Final = "vertex_ai/gemini-3.1-pro-preview"
_STUDIO_PRO: Final = "gemini/gemini-3.1-pro-preview"
_STUDIO_FLASH: Final = "gemini/gemini-3.5-flash"
_VERTEX_ARGON: Final = "vertex_ai/gemini-4-argon"
_STUDIO_ARGON: Final = "gemini/gemini-4-argon"
_VERTEX_FLASH_2_5: Final = "vertex_ai/gemini-2.5-flash"
_DROP_PLACEMENTS: Final[tuple[DropParams, ...]] = ("request", "deployment")
_IMAGE_MODELS: Final = (
    ("vertex-nano", _VERTEX_NANO),
    ("vertex-flash-image", _VERTEX_FLASH_IMAGE),
    ("studio-nano", _STUDIO_NANO),
    ("studio-flash-image", _STUDIO_FLASH_IMAGE),
)
# Google refused logprobs on these and accepted them on _LOGPROB_ACCEPTING_MODELS in live calls on 2026-11-18. The proof
# is in the PR that added these lists.
_LOGPROB_REFUSING_MODELS: Final = (
    *_IMAGE_MODELS,
    ("vertex-flash", _VERTEX_FLASH),
    ("studio-flash", _STUDIO_FLASH),
    ("vertex-pro", _VERTEX_PRO),
    ("studio-pro", _STUDIO_PRO),
    ("vertex-argon", _VERTEX_ARGON),
)
_LOGPROB_ACCEPTING_MODELS: Final = (
    ("vertex-flash-2-5", _VERTEX_FLASH_2_5),
    ("studio-argon", _STUDIO_ARGON),
)
_LOGPROB_ASKS: Final[tuple[tuple[str, Mapping[str, JsonValue]], ...]] = (
    ("logprobs", {"logprobs": True}),
    ("logprobs-and-top-logprobs", {"logprobs": True, "top_logprobs": 3}),
)
_LOGPROB_FIELDS: Final = frozenset({"responseLogprobs", "logprobs"})
_ACCEPTED_EFFORT: Final = "high"
# Levels Google AI Studio answered with a 400 in live calls on 2026-11-18, see the proof in PR #52514.
_STUDIO_REFUSED_EFFORTS: Final = (
    ("studio-nano", _STUDIO_NANO, "low"),
    ("studio-flash-image", _STUDIO_FLASH_IMAGE, "low"),
    ("studio-flash-image", _STUDIO_FLASH_IMAGE, "medium"),
)
_KEPT_EFFORTS: Final = (
    ("studio-nano", _STUDIO_NANO, "medium"),
    ("studio-nano", _STUDIO_NANO, "high"),
    ("studio-flash-image", _STUDIO_FLASH_IMAGE, "minimal"),
    ("studio-flash-image", _STUDIO_FLASH_IMAGE, "high"),
    ("vertex-nano", _VERTEX_NANO, "low"),
    ("vertex-flash-image", _VERTEX_FLASH_IMAGE, "low"),
    ("vertex-flash-image", _VERTEX_FLASH_IMAGE, "medium"),
)
_REJECTION: Final = "scripted rejection of this generation config"
_TOKEN_REPLY: Final = json.dumps({"access_token": "scripted", "token_type": "Bearer", "expires_in": 3600}).encode()
_REJECTION_REPLY: Final = json.dumps(
    {"error": {"code": 400, "message": _REJECTION, "status": "INVALID_ARGUMENT"}}
).encode()
_ANSWER_REPLY: Final = json.dumps(
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "done"}]}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4},
    }
).encode()


@dataclass(frozen=True, slots=True)
class Exchange:
    response: httpx.Response
    sent: tuple[Request, ...]


def _regression(name: str, *values: object) -> ParameterSet:
    return pytest.param(*values, id=f"regression-{name}")


def _guard(name: str, *values: object) -> ParameterSet:
    return pytest.param(*values, id=f"guard-{name}")


def _generation_config(request: Request) -> Mapping[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(request.body)["generationConfig"])


def _accepts_every_config(_config: Mapping[str, JsonValue]) -> bool:
    return False


def _has_logprobs(config: Mapping[str, JsonValue]) -> bool:
    return bool(_LOGPROB_FIELDS & config.keys())


def _peer(rejects: ConfigCheck) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == _TOKEN_PATH:
            return Reply(body=_TOKEN_REPLY)
        if rejects(_generation_config(request)):
            return Reply(status=400, body=_REJECTION_REPLY)
        return Reply(body=_ANSWER_REPLY)

    return respond


def _deployment(scenario: Scenario, wire: Wire, model: str, deployment_fields: Mapping[str, JsonValue]) -> str:
    if model.startswith("vertex_ai/"):
        return scenario.model(
            model=model,
            api_base=wire.url,
            api_key=None,
            vertex_project=_PROJECT,
            vertex_location=_LOCATION,
            vertex_credentials=service_account_json(_PROJECT, wire.url),
            num_retries=0,
            **deployment_fields,
        )
    return scenario.model(model=model, api_base=wire.url, api_key=_GEMINI_KEY, num_retries=0, **deployment_fields)


def _request_body(route: str, alias: str, fields: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    marker: Final = f"image controls {uuid.uuid4().hex}"
    if route == _RESPONSES:
        return {"model": alias, "input": marker, "max_output_tokens": 64, **fields}
    return {"model": alias, "messages": [{"role": "user", "content": marker}], "max_tokens": 64, **fields}


def _exchange(
    gateway: Gateway,
    model: str,
    fields: Mapping[str, JsonValue],
    *,
    route: str = _CHAT,
    deployment_fields: Mapping[str, JsonValue] | None = None,
    rejects: ConfigCheck = _accepts_every_config,
) -> Exchange:
    with wire_server(_peer(rejects)) as wire, gateway.scenario() as scenario:
        alias: Final = _deployment(scenario, wire, model, deployment_fields or {})
        response: Final = gateway.request("POST", route, _request_body(route, alias, fields))
        sent: Final = tuple(request for request in wire.drain() if request.target != _TOKEN_PATH)
    return Exchange(response, sent)


def _exchange_with_drop_params(
    gateway: Gateway,
    model: str,
    fields: Mapping[str, JsonValue],
    placement: DropParams,
    *,
    route: str = _CHAT,
    rejects: ConfigCheck = _accepts_every_config,
) -> Exchange:
    if placement == "request":
        return _exchange(gateway, model, {**fields, "drop_params": True}, route=route, rejects=rejects)
    return _exchange(gateway, model, fields, route=route, deployment_fields={"drop_params": True}, rejects=rejects)


def _generate_content_path(model: str) -> str:
    provider, _, name = model.partition("/")
    if provider == "vertex_ai":
        return f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{name}:generateContent"
    return f"/models/{name}:generateContent"


def _answered_config(exchange: Exchange, model: str) -> Mapping[str, JsonValue]:
    assert exchange.response.status_code == 200, exchange.response.text
    assert [(request.method, request.target) for request in exchange.sent] == [("POST", _generate_content_path(model))]
    return _generation_config(exchange.sent[0])


def _rejected_configs(exchange: Exchange) -> tuple[Mapping[str, JsonValue], ...]:
    assert exchange.response.status_code == 400, exchange.response.text
    assert _REJECTION in exchange.response.text, exchange.response.text
    return tuple(_generation_config(request) for request in exchange.sent)


def _thinking_config(config: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return object_value(config["thinkingConfig"])


def _thinking_level(config: Mapping[str, JsonValue]) -> str:
    return string_value(_thinking_config(config)["thinkingLevel"]).lower()


def _has_thinking_level(config: Mapping[str, JsonValue]) -> bool:
    return "thinkingLevel" in object_value(config.get("thinkingConfig", {}))


def _thinks_at(level: str) -> ConfigCheck:
    def rejects(config: Mapping[str, JsonValue]) -> bool:
        return _has_thinking_level(config) and _thinking_level(config) == level

    return rejects


@pytest.mark.parametrize(
    ("model", "effort", "level"),
    (
        _regression("vertex-nano-minimal", _VERTEX_NANO, "minimal", "minimal"),
        _regression("studio-nano-minimal", _STUDIO_NANO, "minimal", "minimal"),
        _guard("vertex-nano-low", _VERTEX_NANO, "low", "low"),
        _guard("vertex-nano-medium", _VERTEX_NANO, "medium", "medium"),
        _guard("vertex-nano-high", _VERTEX_NANO, "high", "high"),
        _guard("vertex-flash-image-low", _VERTEX_FLASH_IMAGE, "low", "low"),
        _guard("vertex-flash-image-medium", _VERTEX_FLASH_IMAGE, "medium", "medium"),
        _guard("vertex-flash-image-high", _VERTEX_FLASH_IMAGE, "high", "high"),
        _guard("vertex-flash-image-minimal", _VERTEX_FLASH_IMAGE, "minimal", "minimal"),
        _guard("vertex-flash-text-minimal", _VERTEX_FLASH, "minimal", "minimal"),
        _guard("vertex-pro-minimal-sends-low", _VERTEX_PRO, "minimal", "low"),
        _guard("studio-pro-minimal-sends-low", _STUDIO_PRO, "minimal", "low"),
    ),
)
def test_reasoning_effort_reaches_gemini_as_the_expected_thinking_level(
    gateway: Gateway, model: str, effort: str, level: str
) -> None:
    exchange: Final = _exchange(gateway, model, {"reasoning_effort": effort})
    assert _thinking_level(_answered_config(exchange, model)) == level


@pytest.mark.parametrize(
    "effort",
    (
        _regression("vertex-nano-none", "none"),
        _regression("vertex-nano-disable", "disable"),
    ),
)
def test_turning_reasoning_off_on_nano_sends_the_minimal_level_without_thoughts(gateway: Gateway, effort: str) -> None:
    exchange: Final = _exchange(gateway, _VERTEX_NANO, {"reasoning_effort": effort})
    config: Final = _answered_config(exchange, _VERTEX_NANO)
    assert _thinking_level(config) == "minimal"
    assert _thinking_config(config)["includeThoughts"] is False


@pytest.mark.parametrize("effort", (_guard("studio-nano-low", "low"),))
def test_a_low_effort_reaches_studio_nano_unchanged_and_its_rejection_reaches_the_caller(
    gateway: Gateway, effort: str
) -> None:
    exchange: Final = _exchange(gateway, _STUDIO_NANO, {"reasoning_effort": effort}, rejects=_thinks_at(effort))
    configs: Final = _rejected_configs(exchange)
    assert configs
    assert all(_thinking_level(config) == effort for config in configs), configs


@pytest.mark.parametrize(
    "effort",
    (
        _guard("studio-flash-image-low", "low"),
        _guard("studio-flash-image-medium", "medium"),
    ),
)
def test_without_drop_params_a_refused_level_reaches_studio_flash_image_and_its_rejection_reaches_the_caller(
    gateway: Gateway, effort: str
) -> None:
    exchange: Final = _exchange(gateway, _STUDIO_FLASH_IMAGE, {"reasoning_effort": effort}, rejects=_thinks_at(effort))
    configs: Final = _rejected_configs(exchange)
    assert configs
    assert all(_thinking_level(config) == effort for config in configs), configs


@pytest.mark.parametrize(
    ("model", "effort", "placement"),
    tuple(
        _regression(f"{name}-{effort}-drop-on-{placement}", model, effort, placement)
        for (name, model, effort), placement in product(_STUDIO_REFUSED_EFFORTS, _DROP_PLACEMENTS)
    ),
)
def test_drop_params_keeps_a_refused_level_away_from_studio_image_models_and_still_asks_for_thoughts(
    gateway: Gateway, model: str, effort: str, placement: DropParams
) -> None:
    refused: Final = _exchange_with_drop_params(
        gateway, model, {"reasoning_effort": effort}, placement, rejects=_thinks_at(effort)
    )
    accepted: Final = _exchange_with_drop_params(gateway, model, {"reasoning_effort": _ACCEPTED_EFFORT}, placement)
    thinking: Final = _thinking_config(_answered_config(refused, model))
    assert "thinkingLevel" not in thinking, thinking
    assert thinking["includeThoughts"] == _thinking_config(_answered_config(accepted, model))["includeThoughts"]


@pytest.mark.parametrize(
    ("model", "effort", "placement"),
    tuple(
        _guard(f"{name}-{effort}-allowed-params-drop-on-{placement}", model, effort, placement)
        for (name, model, effort), placement in product(_STUDIO_REFUSED_EFFORTS, _DROP_PLACEMENTS)
    ),
)
def test_allowed_openai_params_send_a_refused_level_to_studio_image_models_despite_drop_params(
    gateway: Gateway, model: str, effort: str, placement: DropParams
) -> None:
    exchange: Final = _exchange_with_drop_params(
        gateway,
        model,
        {"reasoning_effort": effort, "allowed_openai_params": ["reasoning_effort"]},
        placement,
        rejects=_thinks_at(effort),
    )
    configs: Final = _rejected_configs(exchange)
    assert configs
    assert all(_thinking_level(config) == effort for config in configs), configs


@pytest.mark.parametrize(
    ("model", "effort", "placement"),
    tuple(
        _guard(f"{name}-{effort}-drop-on-{placement}", model, effort, placement)
        for (name, model, effort), placement in product(_KEPT_EFFORTS, _DROP_PLACEMENTS)
    ),
)
def test_drop_params_leaves_a_level_the_route_accepts_unchanged(
    gateway: Gateway, model: str, effort: str, placement: DropParams
) -> None:
    exchange: Final = _exchange_with_drop_params(gateway, model, {"reasoning_effort": effort}, placement)
    assert _thinking_level(_answered_config(exchange, model)) == effort


@pytest.mark.parametrize("effort", (_regression("vertex-nano-minimal", "minimal"),))
def test_responses_reasoning_effort_reaches_nano_as_the_same_thinking_level(gateway: Gateway, effort: str) -> None:
    exchange: Final = _exchange(gateway, _VERTEX_NANO, {"reasoning": {"effort": effort}}, route=_RESPONSES)
    assert _thinking_level(_answered_config(exchange, _VERTEX_NANO)) == effort


@pytest.mark.parametrize("effort", (_regression("studio-flash-image-low-drop-on-request", "low"),))
def test_responses_drop_params_keeps_a_refused_level_away_from_studio_flash_image(
    gateway: Gateway, effort: str
) -> None:
    exchange: Final = _exchange_with_drop_params(
        gateway,
        _STUDIO_FLASH_IMAGE,
        {"reasoning": {"effort": effort}},
        "request",
        route=_RESPONSES,
        rejects=_thinks_at(effort),
    )
    config: Final = _answered_config(exchange, _STUDIO_FLASH_IMAGE)
    assert not _has_thinking_level(config), config


@pytest.mark.parametrize(
    ("model", "fields", "placement"),
    tuple(
        _regression(f"{name}-{asked}-drop-on-{placement}", model, fields, placement)
        for (name, model), (asked, fields), placement in product(
            _LOGPROB_REFUSING_MODELS, _LOGPROB_ASKS, _DROP_PLACEMENTS
        )
    ),
)
def test_drop_params_keeps_logprobs_away_from_models_that_reject_them(
    gateway: Gateway, model: str, fields: Mapping[str, JsonValue], placement: DropParams
) -> None:
    exchange: Final = _exchange_with_drop_params(gateway, model, fields, placement, rejects=_has_logprobs)
    config: Final = _answered_config(exchange, model)
    assert not _LOGPROB_FIELDS & config.keys(), config


@pytest.mark.parametrize(
    ("model", "placement"),
    tuple(
        _guard(f"{name}-drop-on-{placement}", model, placement)
        for (name, model), placement in product(_LOGPROB_ACCEPTING_MODELS, _DROP_PLACEMENTS)
    ),
)
def test_drop_params_leaves_logprobs_on_a_model_that_accepts_them(
    gateway: Gateway, model: str, placement: DropParams
) -> None:
    exchange: Final = _exchange_with_drop_params(gateway, model, {"logprobs": True}, placement)
    assert _answered_config(exchange, model)["responseLogprobs"] is True


@pytest.mark.parametrize(
    "model",
    (
        _guard("vertex-nano-drop-off", _VERTEX_NANO),
        _guard("studio-nano-drop-off", _STUDIO_NANO),
        _guard("vertex-flash-drop-off", _VERTEX_FLASH),
        _guard("studio-flash-drop-off", _STUDIO_FLASH),
        _guard("vertex-argon-drop-off", _VERTEX_ARGON),
    ),
)
def test_without_drop_params_logprobs_reach_gemini_and_its_rejection_reaches_the_caller(
    gateway: Gateway, model: str
) -> None:
    exchange: Final = _exchange(gateway, model, {"logprobs": True}, rejects=_has_logprobs)
    configs: Final = _rejected_configs(exchange)
    assert configs
    assert all(config["responseLogprobs"] is True for config in configs), configs


@pytest.mark.parametrize(
    ("model", "placement"),
    tuple(
        _guard(f"{name}-allowed-params-drop-on-{placement}", model, placement)
        for (name, model), placement in product(_LOGPROB_REFUSING_MODELS, _DROP_PLACEMENTS)
    ),
)
def test_allowed_openai_params_send_logprobs_to_refusing_models_despite_drop_params(
    gateway: Gateway, model: str, placement: DropParams
) -> None:
    exchange: Final = _exchange_with_drop_params(
        gateway,
        model,
        {"logprobs": True, "top_logprobs": 2, "allowed_openai_params": ["logprobs", "top_logprobs"]},
        placement,
        rejects=_has_logprobs,
    )
    configs: Final = _rejected_configs(exchange)
    assert configs
    assert all((config["responseLogprobs"], config["logprobs"]) == (True, 2) for config in configs), configs


@pytest.mark.parametrize(
    "placement",
    tuple(_guard(f"vertex-nano-drop-on-{placement}", placement) for placement in _DROP_PLACEMENTS),
)
def test_drop_params_leaves_sampling_controls_on_nano_unchanged(gateway: Gateway, placement: DropParams) -> None:
    exchange: Final = _exchange_with_drop_params(
        gateway, _VERTEX_NANO, {"temperature": 0.4, "top_p": 0.7, "seed": 11}, placement
    )
    config: Final = _answered_config(exchange, _VERTEX_NANO)
    assert (config["temperature"], config["top_p"], config["seed"]) == (0.4, 0.7, 11), config


@pytest.mark.parametrize("effort", (_guard("vertex-nano-unknown-effort", "not-a-real-effort"),))
def test_an_unknown_reasoning_effort_is_refused_before_reaching_gemini(gateway: Gateway, effort: str) -> None:
    exchange: Final = _exchange(gateway, _VERTEX_NANO, {"reasoning_effort": effort})
    assert exchange.response.status_code == 400, exchange.response.text
    assert effort in exchange.response.text, exchange.response.text
    assert exchange.sent == ()


def _advertised_reasoning_efforts(gateway: Gateway, model: str) -> JsonValue:
    with wire_server(_peer(_accepts_every_config)) as wire, gateway.scenario() as scenario:
        alias: Final = _deployment(scenario, wire, model, {})
        groups: Final = list_value(gateway.get("/model_group/info", {"model_group": alias})["data"])
    assert len(groups) == 1, groups
    return object_value(groups[0]).get("supported_reasoning_efforts")


@pytest.mark.parametrize("model", (_regression("vertex-nano-group", _VERTEX_NANO),))
def test_a_nano_model_group_advertises_the_minimal_reasoning_effort(gateway: Gateway, model: str) -> None:
    efforts: Final = _advertised_reasoning_efforts(gateway, model)
    assert isinstance(efforts, list) and "minimal" in efforts, efforts


@pytest.mark.parametrize("model", (_regression("studio-nano-group", _STUDIO_NANO),))
def test_a_studio_nano_model_group_advertises_minimal_and_not_low(gateway: Gateway, model: str) -> None:
    efforts: Final = _advertised_reasoning_efforts(gateway, model)
    assert isinstance(efforts, list) and "minimal" in efforts and "low" not in efforts, efforts


@pytest.mark.parametrize("model", (_regression("studio-flash-image-group", _STUDIO_FLASH_IMAGE),))
def test_a_studio_flash_image_model_group_advertises_minimal_and_high_but_not_low_or_medium(
    gateway: Gateway, model: str
) -> None:
    efforts: Final = _advertised_reasoning_efforts(gateway, model)
    assert isinstance(efforts, list) and "minimal" in efforts and "high" in efforts, efforts
    assert "low" not in efforts and "medium" not in efforts, efforts


@pytest.mark.parametrize("model", (_guard("vertex-pro-group", _VERTEX_PRO),))
def test_a_non_flash_gemini_model_group_keeps_advertising_no_reasoning_effort_list(
    gateway: Gateway, model: str
) -> None:
    assert _advertised_reasoning_efforts(gateway, model) is None
