"""litellm_params.timeout bounds every HTTP guardrail's outbound call, through a real proxy.

Each guardrail is configured against an owned sink that records the request and then sleeps
~20s. With `timeout: 1` the outbound call must abort near the bound, so the chat round trip
completes in seconds instead of waiting on the sink. A control guardrail without `timeout`
points at a sink path that sleeps ~3s and must wait for the reply, proving unset keeps the
handler default.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, cast

import httpx
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server

SLOW_SECONDS: Final = 20
FAST_SECONDS: Final = 3
BOUND_SECONDS: Final = 8

EXCLUDED: Final = {
    "model_armor": "requires Google credentials before any HTTP call can be issued",
    "microsoft_purview": "token endpoint is the fixed login.microsoftonline.com and cannot point at a sink",
    "agent_365": "authenticates against Entra ID before any HTTP call can be issued",
    "mcp_jwt_signer": "only runs for pre_mcp_call, which /v1/chat/completions cannot trigger",
    "semantic_guard": "routes through litellm embeddings, not a guardrail provider HTTP client",
    "llm_as_a_judge": "routes through litellm completions, not a guardrail provider HTTP client",
    "litellm_content_filter": "local pattern matching with no outbound HTTP",
    "tool_permission": "policy evaluation with no outbound HTTP",
    "mcp_end_user_permission": "policy evaluation with no outbound HTTP",
    "block_code_execution": "local code analysis with no outbound HTTP",
    "custom_code": "runs user code with no provider HTTP client",
    "hide-secrets": "in-process masking with no outbound HTTP",
    "mcp_security": "MCP tool scanning with no provider HTTP client",
    "unified_guardrail": "delegates to other guardrails, makes no HTTP call of its own",
    "conduct": "requires the optional conduct-litellm-guard package, which is not installed",
    "grayswan": "honors its own guardrail_timeout param, not litellm_params.timeout",
}


PROVIDERS: Final = (
    pytest.param("aim", "aim", {}, "pre_call", False, id="aim"),
    pytest.param("aporia", "aporia", {}, "post_call", False, id="aporia"),
    pytest.param("alice", "alice", {}, "pre_call", False, id="alice"),
    pytest.param("azure-prompt-shield", "azure/prompt_shield", {}, "pre_call", False, id="azure-prompt-shield"),
    pytest.param(
        "azure-text-moderations", "azure/text_moderations", {}, "pre_call", False, id="azure-text-moderations"
    ),
    pytest.param("cato", "cato_networks", {}, "pre_call", False, id="cato-networks"),
    pytest.param("crowdstrike", "crowdstrike_aidr", {}, "pre_call", False, id="crowdstrike-aidr"),
    pytest.param(
        "deepkeep", "deepkeep", {"deepkeep_firewall_id": "synthetic-firewall"}, "pre_call", False, id="deepkeep"
    ),
    pytest.param("dynamoai", "dynamoai", {}, "pre_call", False, id="dynamoai"),
    pytest.param("enkryptai", "enkryptai", {}, "pre_call", False, id="enkryptai"),
    pytest.param("generic", "generic_guardrail_api", {}, "pre_call", False, id="generic-guardrail-api"),
    pytest.param(
        "ibm",
        "ibm_guardrails",
        {"auth_token": "synthetic-ibm-token", "detector_id": "synthetic-detector"},
        "pre_call",
        False,
        id="ibm-guardrails",
    ),
    pytest.param("javelin", "javelin", {"guard_name": "synthetic-guard"}, "pre_call", False, id="javelin"),
    pytest.param("lasso", "lasso", {}, "pre_call", False, id="lasso"),
    pytest.param("qualifire", "qualifire", {}, "pre_call", False, id="qualifire"),
    pytest.param("noma", "noma", {}, "pre_call", False, id="noma"),
    pytest.param("noma-v2", "noma_v2", {}, "pre_call", False, id="noma-v2"),
    pytest.param(
        "ovalix",
        "ovalix",
        {
            "tracker_api_key": "synthetic-tracker-key",
            "application_id": "synthetic-app",
            "pre_checkpoint_id": "synthetic-pre",
        },
        "pre_call",
        False,
        id="ovalix",
    ),
    pytest.param("pangea", "pangea", {}, "pre_call", False, id="pangea"),
    pytest.param("openai-moderation", "openai_moderation", {}, "pre_call", False, id="openai-moderation"),
    pytest.param("lakera", "lakera", {}, "pre_call", False, id="lakera"),
    pytest.param("lakera-v2", "lakera_v2", {}, "pre_call", False, id="lakera-v2"),
    pytest.param("promptguard", "promptguard", {}, "pre_call", False, id="promptguard"),
    pytest.param("xecguard", "xecguard", {"xecguard_model": "synthetic-model"}, "pre_call", False, id="xecguard"),
    pytest.param("typesafe", "typesafe", {}, "pre_call", True, id="typesafe"),
    pytest.param("compresr", "compresr", {}, "pre_call", True, id="compresr"),
    pytest.param("repelloai", "repelloai", {"asset_id": "synthetic-asset"}, "pre_call", False, id="repelloai"),
    pytest.param("prompt-security", "prompt_security", {}, "pre_call", False, id="prompt-security"),
    pytest.param("hiddenlayer", "hiddenlayer", {}, "pre_call", False, id="hiddenlayer"),
    pytest.param(
        "guardrails-ai", "guardrails_ai", {"guard_name": "synthetic-guard"}, "pre_call", False, id="guardrails-ai"
    ),
    pytest.param(
        "presidio",
        "presidio",
        {"pii_entities_config": {"EMAIL_ADDRESS": "BLOCK"}},
        "pre_call",
        False,
        id="presidio",
    ),
    pytest.param(
        "bedrock",
        "bedrock",
        {
            "guardrailIdentifier": "synthetic-guardrail",
            "guardrailVersion": "DRAFT",
            "aws_region_name": "us-east-1",
        },
        "pre_call",
        False,
        id="bedrock",
    ),
    pytest.param("rubrik", "rubrik", {}, "pre_call", False, id="rubrik"),
    pytest.param("qostodian", "qostodian_nexus", {}, "pre_call", False, id="qostodian-nexus"),
    pytest.param("straiker", "straiker", {"default_app": "synthetic-app"}, "pre_call", False, id="straiker"),
    pytest.param("zscaler", "zscaler_ai_guard", {}, "pre_call", False, id="zscaler-ai-guard"),
    pytest.param("pillar", "pillar", {}, "pre_call", False, id="pillar"),
    pytest.param("cisco", "cisco_ai_defense", {}, "pre_call", False, id="cisco-ai-defense"),
    pytest.param("vigil", "vigil_guard", {}, "pre_call", False, id="vigil-guard"),
    pytest.param("singulr", "singulr", {}, "pre_call", False, id="singulr"),
    pytest.param("headroom", "headroom", {}, "pre_call", True, id="headroom"),
    pytest.param(
        "akto",
        "akto",
        {
            "akto_api_key": "synthetic-akto-key",
            "akto_account_id": "synthetic-account",
            "akto_vxlan_id": "synthetic-vxlan",
        },
        "pre_call",
        False,
        id="akto",
    ),
    pytest.param("onyx", "onyx", {}, "post_call", False, id="onyx"),
    pytest.param("panw", "panw_prisma_airs", {}, "pre_call", False, id="panw-prisma-airs"),
)


@dataclass(frozen=True, slots=True)
class Seen:
    target: str
    headers: dict[str, str]
    body: str


@dataclass(slots=True)
class Sink:
    port: int
    seen: list[Seen] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    server: ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        sink: Final = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _handle(self) -> None:
                raw: Final = self.rfile.read(int(self.headers.get("content-length", "0")))
                with sink.lock:
                    sink.seen.append(
                        Seen(self.path, {k.lower(): v for k, v in self.headers.items()}, raw.decode(errors="replace"))
                    )
                time.sleep(SLOW_SECONDS if self.path.startswith("/slow/") else FAST_SECONDS)
                payload: Final = b"{}"
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(payload)

            do_POST = _handle
            do_GET = _handle
            do_PUT = _handle

            def log_message(self, format: str, *args: object) -> None:
                pass

        class Server(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        assert self.server is not None and self.thread is not None
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.server = None
        self.thread = None

    def calls_for(self, name: str) -> tuple[Seen, ...]:
        token: Final = f"key-{name}"
        with self.lock:
            return tuple(
                s
                for s in self.seen
                if f"/{name}" in s.target or any(token in v for v in s.headers.values()) or token in s.body
            )


def _provider(request: Request) -> Reply:
    body: Final = json.dumps(
        {
            "id": "chatcmpl-timeout",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "synthetic answer"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    ).encode()
    return Reply(body=body)


def _guardrail(
    name: str,
    provider: str,
    sink: str,
    timeout: object,
    extra: dict[str, object],
    mode: str,
) -> dict[str, object]:
    base: Final = f"{sink}/slow/{name}/" if timeout is not None else f"{sink}/fast/{name}/"
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": provider,
            "mode": mode,
            "default_on": False,
            "api_key": f"key-{name}",
            **extra,
            **_bases(provider, base),
            **({"timeout": timeout} if timeout is not None else {}),
        },
    }


def _bases(provider: str, base: str) -> dict[str, object]:
    if provider == "ibm_guardrails":
        return {"base_url": base}
    if provider == "ovalix":
        return {"tracker_api_base": base}
    if provider == "akto":
        return {"akto_base_url": base}
    if provider == "singulr":
        return {"singulr_api_base": base}
    if provider == "presidio":
        return {"presidio_analyzer_api_base": base + "/", "presidio_anonymizer_api_base": base + "/"}
    if provider == "bedrock":
        return {"aws_bedrock_runtime_endpoint": base}
    return {"api_base": base}


def _provider_values() -> Iterator[tuple[str, str, dict[str, object], str, bool]]:
    for param in PROVIDERS:
        yield cast("tuple[str, str, dict[str, object], str, bool]", param.values)


def _rig_config(sink_url: str, root: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = False
    config["guardrails"] = [
        _guardrail(name, provider, sink_url, 1, dict(extra), mode)
        for name, provider, extra, mode, _ in _provider_values()
    ] + [
        _guardrail("control-generic", "generic_guardrail_api", sink_url, None, {}, "pre_call"),
    ]
    path: Final = root / "guardrail-timeout.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    sink: Sink
    chat_model: str


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("guardrail-timeout")
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port: Final = reserve.getsockname()[1]
    sink: Final = Sink(port)
    sink.start()
    with gateway_from_environment() as gateway, wire_server(_provider) as provider:
        config: Final = _rig_config(sink.url, root)
        overrides: Final = {
            "AWS_ACCESS_KEY_ID": "synthetic-aws-key",
            "AWS_SECRET_ACCESS_KEY": "synthetic-aws-secret",
            "AWS_REGION_NAME": "us-east-1",
        }
        with (
            owned_proxy_process(gateway, root, overrides, config=config, workers=2) as owned,
            owned.gateway.scenario() as scenario,
        ):
            chat: Final = scenario.model(
                model="openai/gpt-4o-mini", api_base=provider.url + "/v1", api_key="synthetic-openai-key"
            )
            yield Rig(owned.gateway, sink, chat)
    if sink.server is not None:
        sink.stop()


def _chat(rig: Rig, guardrail_name: str, exchange: bool = False) -> tuple[httpx.Response, float]:
    def tool_call(index: int) -> dict[str, object]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_synthetic_{index}",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        }

    messages: Final = (
        [
            {"role": "user", "content": f"look up a fact for {guardrail_name}"},
            tool_call(0),
            {"role": "tool", "tool_call_id": "call_synthetic_0", "content": "synthetic tool output " * 200},
            tool_call(1),
            {"role": "tool", "tool_call_id": "call_synthetic_1", "content": "synthetic newer output " * 200},
            {"role": "user", "content": f"guardrail timeout probe {guardrail_name}"},
        ]
        if exchange
        else [{"role": "user", "content": f"guardrail timeout probe {guardrail_name}"}]
    )
    start: Final = time.monotonic()
    response: Final = rig.proxy.client.post(
        "/v1/chat/completions",
        json={"model": rig.chat_model, "messages": messages, "guardrails": [guardrail_name]},
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    return response, time.monotonic() - start


@pytest.mark.parametrize("name,provider,extra,mode,exchange", PROVIDERS)
def test_litellm_params_timeout_bounds_outbound_call(
    rig: Rig, name: str, provider: str, extra: dict[str, object], mode: str, exchange: bool
) -> None:
    response, elapsed = _chat(rig, name, exchange)
    calls: Final = rig.sink.calls_for(name)
    assert calls, f"{name}: sink saw no request for {provider}"
    assert elapsed < BOUND_SECONDS, f"{name}: elapsed {elapsed:.2f}s, expected under {BOUND_SECONDS}s with timeout=1"
    assert response.status_code != 504, response.text


def test_unset_timeout_waits_for_sink_response(rig: Rig) -> None:
    response, elapsed = _chat(rig, "control-generic")
    calls: Final = rig.sink.calls_for("control-generic")
    assert calls, "control-generic: sink saw no request"
    assert elapsed >= FAST_SECONDS - 0.5, (
        f"control-generic: elapsed {elapsed:.2f}s, expected to wait for the {FAST_SECONDS}s sink response"
    )
    assert response.status_code in (200, 400, 500), response.text
