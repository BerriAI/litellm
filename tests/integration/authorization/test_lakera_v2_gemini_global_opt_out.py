"""A key or team opted out of a global Lakera guardrail keeps skipping it on a Gemini pass-through body."""

import json
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, Scenario
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

GEMINI_ROUTE: Final = "/gemini/v1beta/models/gemini-2.5-flash:generateContent"
CARD: Final = "4111 1111 1111 1111"
GLOBAL_LAKERA: Final = "lakera-global"
_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_BODY: Final[dict[str, JsonValue]] = {"contents": [{"role": "user", "parts": [{"text": f"My card is {CARD}"}]}]}


def _lakera_verdict(request: Request) -> Reply:
    flagged: Final = CARD.encode() in request.body
    breakdown: Final = [{"detector_type": "pii/credit_card", "detected": flagged, "message_id": 0}]
    return Reply(body=json.dumps({"flagged": flagged, "breakdown": breakdown, "payload": []}).encode())


def _gemini_reply(_request: Request) -> Reply:
    return Reply(body=json.dumps({"candidates": [{"content": {"role": "model", "parts": [{"text": "Hi"}]}}]}).encode())


@dataclass(frozen=True, slots=True)
class GlobalLakeraRig:
    owned: OwnedProxy
    lakera: Wire
    upstream: Wire


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[GlobalLakeraRig]:
    directory: Final = tmp_path_factory.mktemp("lakera-gemini-global-opt-out")
    with ExitStack() as stack:
        lakera: Final = stack.enter_context(wire_server(_lakera_verdict))
        upstream: Final = stack.enter_context(wire_server(_gemini_reply))
        config: Final = _JSON.validate_python(yaml.safe_load(_CONFIG.read_text()))
        path: Final = directory / "lakera-gemini-global-opt-out.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    **config,
                    "guardrails": [
                        {
                            "guardrail_name": GLOBAL_LAKERA,
                            "litellm_params": {
                                "guardrail": "lakera_v2",
                                "mode": "pre_call",
                                "default_on": True,
                                "on_flagged": "block",
                                "api_base": lakera.url,
                                "api_key": "synthetic-lakera-key",
                            },
                        }
                    ],
                }
            )
        )
        environment: Final = {"GEMINI_API_BASE": upstream.url, "GEMINI_API_KEY": "synthetic-gemini-key"}
        owned: Final = stack.enter_context(owned_proxy_process(session_gateway, directory, environment, config=path))
        yield GlobalLakeraRig(owned, lakera, upstream)


def _key_disabling_global(scenario: Scenario) -> str:
    return scenario.key(disable_global_guardrails=True)


def _key_naming_it(scenario: Scenario) -> str:
    return scenario.key(metadata={"opted_out_global_guardrails": [GLOBAL_LAKERA]})


def _team_disabling_global(scenario: Scenario) -> str:
    return scenario.key(team_id=scenario.team(disable_global_guardrails=True))


def _team_naming_it(scenario: Scenario) -> str:
    return scenario.key(team_id=scenario.team(metadata={"opted_out_global_guardrails": [GLOBAL_LAKERA]}))


@pytest.mark.parametrize(
    "opted_out_key", [_key_disabling_global, _key_naming_it, _team_disabling_global, _team_naming_it]
)
def test_opted_out_key_skips_the_global_guardrail_on_a_gemini_body(
    rig: GlobalLakeraRig, opted_out_key: Callable[[Scenario], str]
) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = opted_out_key(scenario)
        rig.lakera.drain()
        rig.upstream.drain()

        response: Final = gateway.request("POST", GEMINI_ROUTE, _BODY, key=key, headers={"x-goog-api-key": key})

        assert response.status_code == 200, response.text
        assert rig.lakera.drain() == ()
        assert len(rig.upstream.drain()) == 1


def test_key_that_did_not_opt_out_is_inspected_by_the_global_guardrail(rig: GlobalLakeraRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        rig.lakera.drain()
        rig.upstream.drain()

        response: Final = gateway.request("POST", GEMINI_ROUTE, _BODY, key=key, headers={"x-goog-api-key": key})

        assert response.status_code == 400, response.text
        assert len(rig.lakera.drain()) == 1
        assert rig.upstream.drain() == ()
