"""A /v1/responses deployment bridged to chat completions (`use_chat_completions_api: true`) answers a
`previous_response_id` follow-up with the prior turn replayed from LiteLLM_SpendLogs, and bills the follow-up
from its own usage (LIT-9604, #46398)"""

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration._support.workers import cost_map_environment
from integration.cost_calculation.cost_tracking_case import COST_MAP
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(240)

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])
_MODEL: Final = "gpt-5.6"
_PROMPT_TOKENS: Final = 1840
_COMPLETION_TOKENS: Final = 412
_FIRST_TURN: Final = "My favorite color is yellow. Remember it"
_SECOND_TURN: Final = "What is my favorite color?"
_ANSWER: Final = "scripted answer yellow"


@dataclass(frozen=True, slots=True)
class _Rig:
    wire: Wire
    proxy: OwnedProxy

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway


def _respond(request: Request) -> Reply:
    assert request.target.endswith("/chat/completions"), request.target
    body: Final = _JSON_OBJECT.validate_json(request.body)
    assert body["model"] == _MODEL, body
    completion: Final = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": 1,
        "model": _MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": _ANSWER}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": _PROMPT_TOKENS,
            "completion_tokens": _COMPLETION_TOKENS,
            "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
        },
    }
    return Reply(body=json.dumps(completion).encode())


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("bridged-previous-response-id-rig")
    with gateway_from_environment() as gateway, wire_server(_respond) as wire:
        overrides: Final = {**cost_map_environment(gateway.upstream_url), "STORE_PROMPTS_IN_SPEND_LOGS": "true"}
        with owned_proxy_process(
            gateway, directory, overrides, remove_environment=("LITELLM_LOCAL_MODEL_COST_MAP",)
        ) as owned:
            yield _Rig(wire, owned)


def _spend_rows(model: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, spend, status FROM "LiteLLM_SpendLogs" WHERE model_group = %s ORDER BY "startTime"',
        (model,),
    )


def _upstream_body(request: Request) -> dict[str, JsonValue]:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    assert "previous_response_id" not in body, body
    return body


def _upstream_messages(request: Request) -> list[dict[str, JsonValue]]:
    return _MESSAGES.validate_python(_upstream_body(request)["messages"])


def test_bridged_previous_response_id_replays_the_prior_turn_and_bills_the_follow_up(rig: _Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{_MODEL}",
            api_base=rig.wire.url,
            api_key="sk-scripted-provider",
            use_chat_completions_api=True,
        )
        first: Final = rig.gateway.request("POST", "/v1/responses", {"model": model, "input": _FIRST_TURN})
        assert first.status_code == 200, first.text
        first_id: Final = _JSON_OBJECT.validate_json(first.content)["id"]
        assert isinstance(first_id, str) and first_id.startswith("resp_"), first_id
        first_rows: Final = eventually(lambda: _spend_rows(model), lambda rows: len(rows) == 1, seconds=70)
        assert first_rows[0]["status"] == "success", first_rows

        second: Final = rig.gateway.request(
            "POST", "/v1/responses", {"model": model, "input": _SECOND_TURN, "previous_response_id": first_id}
        )
        assert second.status_code == 200, second.text
        sent: Final = tuple(request for request in rig.wire.drain() if request.method == "POST")
        assert len(sent) == 2, [request.target for request in sent]
        assert _upstream_messages(sent[0]) == [{"role": "user", "content": _FIRST_TURN}], sent[0].body
        assert _upstream_messages(sent[1]) == [
            {"role": "user", "content": _FIRST_TURN},
            {"role": "assistant", "content": _ANSWER},
            {"role": "user", "content": _SECOND_TURN},
        ], sent[1].body

        rows: Final = eventually(lambda: _spend_rows(model), lambda values: len(values) == 2, seconds=70)
        assert [row["status"] for row in rows] == ["success", "success"], rows
        rates: Final = COST_MAP[_MODEL]
        assert rates.input_cost_per_token is not None and rates.output_cost_per_token is not None, rates
        assert float(rows[1]["spend"]) == pytest.approx(
            _PROMPT_TOKENS * rates.input_cost_per_token + _COMPLETION_TOKENS * rates.output_cost_per_token
        ), rows
