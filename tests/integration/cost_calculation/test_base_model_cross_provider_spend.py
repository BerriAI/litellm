"""A chatgpt/ subscription deployment has a cost-map row without rates. With model_info.base_model pointing
at a priced row of another provider, the spend row is billed from that row (#46233)"""

import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support import codex_vendor as cv
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(240)

_MODEL: Final = "cc-subscription-sol"
_INPUT_TOKENS: Final = 1000
_OUTPUT_TOKENS: Final = 100
_COST_MAP: Final[Mapping[str, Mapping[str, JsonValue]]] = {
    f"chatgpt/{_MODEL}": {"litellm_provider": "chatgpt", "mode": "responses"},
    _MODEL: {
        "litellm_provider": "openai",
        "mode": "responses",
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 8e-6,
    },
    f"azure/{_MODEL}": {
        "litellm_provider": "azure",
        "mode": "responses",
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 5e-6,
    },
}


@dataclass(frozen=True, slots=True)
class _Rig:
    wire: Wire
    proxy: OwnedProxy

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway


def _completed_response(model: str) -> Mapping[str, JsonValue]:
    tag: Final = uuid.uuid4().hex
    message: Final[dict[str, JsonValue]] = {
        "id": f"msg_{tag}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "priced", "annotations": []}],
    }
    return {
        "id": f"resp_{tag}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [message],
        "usage": {
            "input_tokens": _INPUT_TOKENS,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": _OUTPUT_TOKENS,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
        },
    }


def _respond(request: Request) -> Reply:
    if request.method == "GET":
        assert request.target.endswith("/_cost_map"), request.target
        return Reply(body=json.dumps(_COST_MAP).encode())
    assert request.target.endswith("/responses"), request.target
    assert request.headers.get("authorization") == f"Bearer {cv.TOKEN}", request.headers
    body: Final = rv.JSON_OBJECT.validate_json(request.body)
    assert body["model"] == _MODEL, body
    response: Final = _completed_response(_MODEL)
    events: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        },
        {"type": "response.completed", "sequence_number": 1, "response": response},
    )
    return Reply(content_type="text/event-stream", chunks=tuple(rv.sse(event) for event in events))


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("base-model-cross-provider-rig")
    with gateway_from_environment() as gateway, wire_server(_respond) as wire:
        overrides: Final = {
            "CHATGPT_TOKEN_DIR": str(cv.login(directory)),
            "CHATGPT_API_BASE": wire.url,
            "LITELLM_MODEL_COST_MAP_URL": f"{wire.url}/_cost_map",
            "MODEL_COST_MAP_MIN_MODEL_COUNT": "1",
            "MODEL_COST_MAP_MAX_SHRINK_RATIO": "0",
        }
        config: Final = cv.proxy_config(directory, probe=False)
        with owned_proxy_process(
            gateway, directory, overrides, config=config, remove_environment=("LITELLM_LOCAL_MODEL_COST_MAP",)
        ) as owned:
            yield _Rig(wire, owned)


@pytest.mark.parametrize(
    ("base_model", "expected_spend"),
    [
        (_MODEL, _INPUT_TOKENS * 1e-6 + _OUTPUT_TOKENS * 8e-6),
        (f"openai/{_MODEL}", _INPUT_TOKENS * 1e-6 + _OUTPUT_TOKENS * 8e-6),
        (f"azure/{_MODEL}", _INPUT_TOKENS * 3e-6 + _OUTPUT_TOKENS * 5e-6),
    ],
)
def test_base_model_bills_the_priced_row_when_the_deployment_provider_row_has_no_rates(
    rig: _Rig, base_model: str, expected_spend: float
) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"chatgpt/{_MODEL}",
            api_base=rig.wire.url,
            api_key=None,
            model_info={"base_model": base_model},
        )
        response: Final = rig.gateway.request(
            "POST", "/v1/responses", {"model": model, "input": [{"role": "user", "content": "price control"}]}
        )
        assert response.status_code == 200, response.text
        assert float(response.headers.get("x-litellm-response-cost", "0")) == pytest.approx(expected_spend), (
            response.headers
        )
        rows: Final = eventually(
            lambda: read_rows('SELECT spend, status FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (model,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["status"] == "success", rows
        assert float(rows[0]["spend"]) == pytest.approx(expected_spend), rows
