import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import gateway_from_environment
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import marker_of, owned_config
from pydantic import JsonValue

_TOKEN: Final = "synthetic-bedrock-bearer"
_KIMI: Final = "moonshotai.kimi-k2.5"
_SONNET: Final = "global.anthropic.claude-sonnet-4-6"
_KIMI_BASE: Final = f"bedrock/us-east-1/{_KIMI}"
_KIMI_TARGET: Final = f"/model/{_KIMI}/converse"
_SONNET_TARGET: Final = f"/model/{_SONNET}/converse"
_PAST_ROUTE_ROW_LINES: Final = 16250
_PAST_ESCALATION_BUFFER_LINES: Final = 15480
_SMALL_LINES: Final = 1300
_REPLY: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": "8115"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 9, "outputTokens": 2, "totalTokens": 11},
        "metrics": {"latencyMs": 1},
    }
).encode()


@dataclass(frozen=True, slots=True)
class _Rig:
    owned: OwnedProxy
    wire: Wire


def _converse_peer(request: Request) -> Reply:
    assert request.headers["authorization"] == f"Bearer {_TOKEN}", request.headers
    return Reply(body=_REPLY)


def _deployment(
    name: str,
    model: str,
    api_base: str,
    *,
    model_info: Mapping[str, JsonValue] | None = None,
    **params: JsonValue,
) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"bedrock/converse/{model}",
            "api_key": _TOKEN,
            "aws_region_name": "us-east-1",
            "aws_bedrock_runtime_endpoint": api_base,
            **params,
        },
        "model_info": {"id": f"{name}-{uuid.uuid4().hex}", **(model_info or {})},
    }


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("base-model-window")
    with gateway_from_environment() as gateway, wire_server(_converse_peer) as wire:
        config: Final = owned_config(
            directory,
            [
                _deployment("base-in-model-info", _KIMI, wire.url, model_info={"base_model": _KIMI_BASE}),
                _deployment("base-in-litellm-params", _KIMI, wire.url, base_model=_KIMI_BASE),
                _deployment("no-base", _KIMI, wire.url),
                _deployment("other-model-base", _KIMI, wire.url, model_info={"base_model": _SONNET}),
                _deployment(
                    "declared-window",
                    _KIMI,
                    wire.url,
                    model_info={"base_model": _KIMI_BASE, "max_input_tokens": 257000},
                ),
                _deployment(
                    "unmapped-base", _KIMI, wire.url, model_info={"base_model": f"{_KIMI_BASE}-not-in-cost-map"}
                ),
                _deployment(
                    "smaller-other-model-base",
                    _KIMI,
                    wire.url,
                    model_info={"base_model": "bedrock/us-east-1/meta.llama3-8b-instruct-v1:0"},
                ),
                _deployment("one-million", _SONNET, wire.url),
                {
                    "model_name": "complexity",
                    "litellm_params": {
                        "model": "auto_router/complexity_router",
                        "complexity_router_config": {
                            "tiers": {
                                "SIMPLE": "base-in-model-info",
                                "MEDIUM": "base-in-model-info",
                                "COMPLEX": "one-million",
                                "REASONING": "one-million",
                            },
                            "enable_context_window_escalation": True,
                        },
                    },
                },
            ],
            litellm_settings={"cache": False},
            router_settings={"enable_pre_call_checks": True},
        )
        with owned_proxy_process(gateway, directory, {}, config=config, workers=2, client_timeout=120) as owned:
            yield _Rig(owned, wire)


def _prompt(lines: int, marker: str) -> str:
    listing: Final = "\n".join(f"Line {i}: the value of item {i} is {(i * 7) % 1000}." for i in range(1, lines + 1))
    return f"{listing}\n\nWhat is the value of item 12345? Answer with the number only. marker-{marker}"


def _request(surface: str, model: str, text: str) -> tuple[str, dict[str, JsonValue]]:
    if surface == "messages":
        return "/v1/messages", {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": text}]}
    if surface == "responses":
        return "/v1/responses", {"model": model, "max_output_tokens": 16, "input": text}
    return "/v1/chat/completions", {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": text}]}


def _send(rig: _Rig, surface: str, model: str, lines: int) -> tuple[int, str, tuple[str, ...]]:
    marker: Final = uuid.uuid4().hex
    path, body = _request(surface, model, _prompt(lines, marker))
    response: Final = rig.owned.gateway.request("POST", path, body)
    targets: Final = tuple(request.target for request in rig.wire.drain() if marker_of(request) == marker)
    return response.status_code, response.text, targets


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
@pytest.mark.parametrize("surface", ["chat", "messages", "responses"])
def test_a_same_model_base_model_row_widens_the_window_pre_call_checks_enforce(rig: _Rig, surface: str) -> None:
    status, text, targets = _send(rig, surface, "base-in-model-info", _PAST_ROUTE_ROW_LINES)
    assert status == 200, text
    assert targets == (_KIMI_TARGET,)


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_a_same_model_base_model_in_litellm_params_widens_the_window_too(rig: _Rig) -> None:
    status, text, targets = _send(rig, "chat", "base-in-litellm-params", _PAST_ROUTE_ROW_LINES)
    assert status == 200, text
    assert targets == (_KIMI_TARGET,)


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
@pytest.mark.parametrize("model", ["no-base", "other-model-base", "declared-window", "unmapped-base"])
def test_a_base_model_that_cannot_widen_keeps_the_route_window_and_calls_no_upstream(rig: _Rig, model: str) -> None:
    status, text, targets = _send(rig, "chat", model, _PAST_ROUTE_ROW_LINES)
    assert status == 400, text
    assert "ContextWindowExceededError" in text, text
    assert targets == ()


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_a_smaller_other_model_base_model_never_narrows_the_route_window(rig: _Rig) -> None:
    status, text, targets = _send(rig, "chat", "smaller-other-model-base", _SMALL_LINES)
    assert status == 200, text
    assert targets == (_KIMI_TARGET,)


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_complexity_router_keeps_a_prompt_on_the_tier_whose_base_model_row_holds_it(rig: _Rig) -> None:
    status, text, targets = _send(rig, "chat", "complexity", _PAST_ESCALATION_BUFFER_LINES)
    assert status == 200, text
    assert targets == (_KIMI_TARGET,)
