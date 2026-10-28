import json
import uuid
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
import yaml
from integration._support.anthropic_sse import message_json
from integration._support.client import Gateway
from integration._support.openai_wire import chat_reply
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

GENERIC: Final = "passthrough-scan-generic"
AIRS: Final = "passthrough-scan-airs"
OPENAI_ROUTE: Final = "/openai/chat/completions"
ANTHROPIC_ROUTE: Final = "/anthropic/v1/messages"
ROUTE_BODIES: Final[Mapping[str, dict[str, JsonValue]]] = MappingProxyType(
    {
        OPENAI_ROUTE: {"model": "gpt-5.6", "messages": [{"role": "user", "content": "Say hi in one word"}]},
        ANTHROPIC_ROUTE: {
            "model": "claude-sonnet-5-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Say hi in one word"}],
        },
    }
)
_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_JSON: Final = TypeAdapter(dict[str, JsonValue])


def _generic_verdict(_request: Request) -> Reply:
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def _airs_verdict(_request: Request) -> Reply:
    return Reply(body=json.dumps({"action": "allow", "category": "benign", "scan_id": "s", "report_id": "r"}).encode())


def _openai_reply(_request: Request) -> Reply:
    return chat_reply("chatcmpl-passthrough-scan", "gpt-5.6", "Hi", stream=False)


def _anthropic_reply(_request: Request) -> Reply:
    return Reply(body=message_json("msg_passthrough_scan", "claude-sonnet-5-5", "Hi"))


def _parsed_texts(texts: JsonValue, vendor_body: dict[str, JsonValue]) -> list[JsonValue]:
    assert isinstance(texts, list), vendor_body
    return [json.loads(text) if isinstance(text, str) else text for text in texts]


def _scanned_payloads(guardrail: str, vendor_body: dict[str, JsonValue]) -> list[JsonValue]:
    if guardrail == GENERIC:
        return _parsed_texts(vendor_body["texts"], vendor_body)
    contents: Final = vendor_body["contents"]
    assert isinstance(contents, list), vendor_body
    return _parsed_texts([content.get("prompt") for content in contents if isinstance(content, dict)], vendor_body)


@dataclass(frozen=True, slots=True)
class PassThroughRig:
    owned: OwnedProxy
    vendors: Mapping[str, Wire]
    upstreams: Mapping[str, Wire]


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[PassThroughRig]:
    directory: Final = tmp_path_factory.mktemp("passthrough-guardrail-scan")
    with ExitStack() as stack:
        vendors: Final = MappingProxyType(
            {
                GENERIC: stack.enter_context(wire_server(_generic_verdict)),
                AIRS: stack.enter_context(wire_server(_airs_verdict)),
            }
        )
        upstreams: Final = MappingProxyType(
            {
                OPENAI_ROUTE: stack.enter_context(wire_server(_openai_reply)),
                ANTHROPIC_ROUTE: stack.enter_context(wire_server(_anthropic_reply)),
            }
        )
        config: Final = _JSON.validate_python(yaml.safe_load(_CONFIG.read_text()))
        path: Final = directory / "passthrough-guardrail-scan.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    **config,
                    "guardrails": [
                        {
                            "guardrail_name": GENERIC,
                            "litellm_params": {
                                "guardrail": "generic_guardrail_api",
                                "mode": "pre_call",
                                "default_on": False,
                                "api_base": vendors[GENERIC].url,
                                "api_key": "synthetic-guardrail-key",
                            },
                        },
                        {
                            "guardrail_name": AIRS,
                            "litellm_params": {
                                "guardrail": "panw_prisma_airs",
                                "mode": "pre_call",
                                "default_on": False,
                                "api_base": vendors[AIRS].url,
                                "api_key": "synthetic-airs-key",
                                "profile_name": "integration",
                            },
                        },
                    ],
                }
            )
        )
        environment: Final = {
            "OPENAI_API_BASE": upstreams[OPENAI_ROUTE].url,
            "OPENAI_API_KEY": "synthetic-openai-key",
            "ANTHROPIC_API_BASE": upstreams[ANTHROPIC_ROUTE].url,
            "ANTHROPIC_API_KEY": "synthetic-anthropic-key",
        }
        owned: Final = stack.enter_context(owned_proxy_process(session_gateway, directory, environment, config=path))
        yield PassThroughRig(owned, vendors, upstreams)


@pytest.mark.parametrize("guardrail", [GENERIC, AIRS])
@pytest.mark.parametrize("route", [OPENAI_ROUTE, ANTHROPIC_ROUTE])
def test_pass_through_guardrail_scans_the_client_body_without_the_callers_key_metadata(
    rig: PassThroughRig, route: str, guardrail: str
) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        user: Final = scenario.user(
            user_email=f"passthrough-scan-{uuid.uuid4().hex}@example.com", user_role="internal_user", teams=[team]
        )
        key: Final = scenario.key(user_id=user, team_id=team)
        body: Final = {**ROUTE_BODIES[route], "guardrails": [guardrail]}

        response: Final = gateway.request("POST", route, body, key=key)

        assert response.status_code == 200, response.text
        scanned: Final = [
            _scanned_payloads(guardrail, _JSON.validate_json(request.body))
            for request in rig.vendors[guardrail].drain()
        ]
        assert scanned == [[body]], scanned
        assert len(rig.upstreams[route].drain()) == 1
