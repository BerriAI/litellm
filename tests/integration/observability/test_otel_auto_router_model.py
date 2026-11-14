import json
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.otlp_sink import SpanSinks, recorded_spans
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(300)

MARKER: Final = re.compile(rb"autorouter-[0-9a-f]{32}")
TIERS: Final = ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")
USAGE: Final = {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}


@dataclass(frozen=True, slots=True)
class AutoRouterRig:
    proxy: Gateway
    sink: str
    router: str
    alias: str
    routed: str


def _upstream(request: Request) -> Reply:
    found: Final = MARKER.search(request.body)
    if found is None:
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    if request.target.endswith("/responses"):
        output: Final = {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
        }
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_" + found.group(0).decode(),
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-4o-mini-2024-07-18",
                    "output": [output],
                    "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
                }
            ).encode()
        )
    head: Final = {"id": "chatcmpl-" + found.group(0).decode(), "created": 1, "model": "gpt-4o-mini-2024-07-18"}
    if json.loads(request.body).get("stream") is not True:
        message: Final = {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        return Reply(
            body=json.dumps({**head, "object": "chat.completion", "choices": [message], "usage": USAGE}).encode()
        )
    chunks: Final = (
        {**head, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "ok"}}]},
        {
            **head,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": USAGE,
        },
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(*(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks), b"data: [DONE]\n\n"),
    )


@pytest.fixture(scope="module")
def gateway(audit_sinks: SpanSinks) -> Iterator[Gateway]:
    with gateway_from_environment() as base:
        yield base


@pytest.fixture(scope="module", params=[False, True], ids=["otel-v1", "otel-v2"])
def rig(
    request: pytest.FixtureRequest,
    gateway: Gateway,
    audit_sinks: SpanSinks,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[AutoRouterRig]:
    otel_v2: Final = request.param is True
    router: Final = "auto-router-" + uuid.uuid4().hex[:8]
    alias: Final = "smart-" + uuid.uuid4().hex[:8]
    routed: Final = "tier-model-" + uuid.uuid4().hex[:8]
    directory: Final = tmp_path_factory.mktemp("auto-router-otel")
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    with wire_server(_upstream) as provider:
        config["litellm_settings"] = {**config.get("litellm_settings", {}), "callbacks": ["otel"]}
        config["router_settings"] = {**config.get("router_settings", {}), "model_group_alias": {alias: router}}
        config["callback_settings"] = {
            "otel": {"exporter": "http/json", "endpoint": audit_sinks.operator, "use_simple_processor": True}
        }
        config["model_list"] = [
            {
                "model_name": routed,
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": provider.url, "api_key": "synthetic"},
            },
            {
                "model_name": router,
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": dict.fromkeys(TIERS, routed)},
                },
            },
        ]
        path: Final = directory / "auto_router_otel.yaml"
        path.write_text(yaml.safe_dump(config))
        overrides: Final = {
            "OTEL_EXPORTER": "http/json",
            "OTEL_ENDPOINT": audit_sinks.operator,
            **({"LITELLM_OTEL_V2": "1"} if otel_v2 else {}),
        }
        with owned_proxy(
            gateway,
            directory,
            overrides,
            config=path,
            remove_environment=() if otel_v2 else ("LITELLM_OTEL_V2",),
        ) as proxy:
            yield AutoRouterRig(proxy=proxy, sink=audit_sinks.operator, router=router, alias=alias, routed=routed)


@pytest.mark.parametrize(
    ("route", "stream", "via_alias"),
    [
        ("/v1/chat/completions", False, False),
        ("/v1/chat/completions", True, False),
        ("/v1/messages", False, False),
        ("/v1/responses", False, False),
        ("/v1/chat/completions", False, True),
    ],
    ids=["chat", "chat-stream", "messages", "responses", "chat-via-alias"],
)
def test_auto_router_llm_span_names_the_routed_model_and_carries_the_decision(
    rig: AutoRouterRig, route: str, stream: bool, via_alias: bool
) -> None:
    prompt: Final = "hi autorouter-" + uuid.uuid4().hex
    body: Final[dict[str, JsonValue]] = {
        "model": rig.alias if via_alias else rig.router,
        "stream": stream,
        **({"input": prompt} if route == "/v1/responses" else {"messages": [{"role": "user", "content": prompt}]}),
        **({"max_tokens": 16} if route == "/v1/messages" else {}),
    }
    start, _ = recorded_spans(rig.sink)
    response: Final = rig.proxy.request("POST", route, body)
    assert response.status_code == 200, response.text
    call_id: Final = response.headers["x-litellm-call-id"]

    def llm_span() -> Mapping[str, JsonValue] | None:
        return next(
            (
                span["attributes"]
                for span in recorded_spans(rig.sink, start)[1]
                if span["attributes"].get("litellm.call_id") == call_id
                and "litellm.routing.router_model_name" in span["attributes"]
            ),
            None,
        )

    attributes: Final = eventually(llm_span, lambda found: found is not None, seconds=60)
    assert attributes is not None
    shown: Final = json.dumps(attributes, sort_keys=True, default=str)
    underlying: Final = frozenset({rig.routed, "gpt-4o-mini"})
    assert attributes["gen_ai.request.model"] in underlying, shown
    assert attributes["gen_ai.response.model"] in underlying, shown
    assert attributes["litellm.routing.router_model_name"] == rig.router, shown
    assert attributes["litellm.routing.router_type"] == "complexity", shown
    assert attributes["litellm.routing.routed_model"] == rig.routed, shown
    assert attributes["litellm.routing.tier"] in TIERS, shown
