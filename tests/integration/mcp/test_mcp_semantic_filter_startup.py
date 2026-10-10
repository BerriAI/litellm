import json
import os
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
import yaml
from integration._support.client import GATEWAY_LIMITS, Gateway, eventually
from integration._support.mcp import ScriptedTool, scripted_peer, text_result
from integration._support.process import (
    DB_PUSH,
    OwnedProxy,
    _launch,
    _proxy_environment,
    _proxy_root,
    _stop_launch,
    owned_proxy_process,
)
from integration._support.redis_process import OwnedRedis, owned_redis
from integration._support.wire import Reply, Request, wire_server

PROMPT: Final = "please refund the invoice for ACME"
MCP_REF: Final = {
    "type": "mcp",
    "server_url": "litellm_proxy",
    "server_label": "litellm",
    "require_approval": "never",
}
SEMANTIC_SETTINGS: Final = {
    "enabled": True,
    "embedding_model": "semantic-embed",
    "top_k": 3,
    "similarity_threshold": 0.5,
}
NO_REDIS_ENV: Final = (
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_URL",
    "REDIS_USERNAME",
    "REDIS_PASSWORD",
    "REDIS_CLUSTER_NODES",
    "REDIS_SENTINEL_NODES",
    "REDIS_SSL",
    "DATABASE_URL_READ_REPLICA",
)
Surface = Literal["chat", "responses"]
TOOL_COUNT: Final = 300


class EmbeddingDouble:
    def __init__(self, mode: Literal["healthy", "held"], delay: float = 0.0) -> None:
        self.release = threading.Event()
        self.mode = mode
        self.delay = delay
        self.arrived: list[tuple[tuple[str, ...], float]] = []
        self.answered: list[tuple[tuple[str, ...], float]] = []

    def document_inputs(self, descriptions: frozenset[str]) -> list[str]:
        """Embedding inputs that are tool descriptions; query embeddings ('test', prompts) excluded."""
        return [text for inputs, _ in self.arrived for text in inputs if text in descriptions]

    def respond(self, request: Request) -> Reply:
        if request.method == "GET" and request.target.endswith("/models"):
            return Reply(body=b'{"object": "list", "data": []}')
        assert request.method == "POST" and request.target.endswith("/embeddings"), request.target
        body: Final = json.loads(request.body)
        raw: Final = body.get("input", [])
        inputs: Final = tuple(str(text) for text in raw) if isinstance(raw, list) else (str(raw),)
        self.arrived.append((inputs, time.monotonic()))
        if self.mode == "held":
            self.release.wait(timeout=120)
        if self.delay:
            time.sleep(self.delay)
        data: Final = [
            {
                "object": "embedding",
                "index": index,
                "embedding": [1.0, 0.0] if "invoice" in text.lower() else [0.0, 1.0],
            }
            for index, text in enumerate(inputs)
        ]
        self.answered.append((inputs, time.monotonic()))
        return Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "data": data,
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            ).encode()
        )


def _chat_double() -> tuple[Callable[[Request], Reply], list[dict[str, object]]]:
    bodies: list[dict[str, object]] = []

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target.endswith("/models"):
            return Reply(body=b'{"object": "list", "data": []}')
        body: Final = json.loads(request.body)
        bodies.append(body)
        identity: Final = uuid.uuid4().hex[:12]
        usage: Final = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if request.target.endswith("/chat/completions"):
            return Reply(
                body=json.dumps(
                    {
                        "id": f"chatcmpl-{identity}",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "gpt-4o-mini",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": "done"},
                            }
                        ],
                        "usage": usage,
                    }
                ).encode()
            )
        assert request.target.endswith("/responses"), request.target
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{identity}",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-4o-mini",
                    "output": [
                        {
                            "type": "message",
                            "id": f"msg_{identity}",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "done", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    return respond, bodies


def _tool_names(body: dict[str, object]) -> tuple[str, ...]:
    tools: Final = body.get("tools")
    if not isinstance(tools, list):
        return ()
    return tuple(
        str(tool["name"] if "name" in tool else tool["function"]["name"]) for tool in tools if isinstance(tool, dict)
    )


def _peer_names(bodies: list[dict[str, object]], alias: str) -> frozenset[str]:
    prefix: Final = f"{alias}-"
    return frozenset(name for body in bodies for name in _tool_names(body) if name.startswith(prefix))


def _tools(invoice_description: str = "Refund an invoice for a customer") -> tuple[ScriptedTool, ...]:
    refund: Final = ScriptedTool(
        "refund_invoice",
        lambda params: text_result("refunded"),
        description=invoice_description,
    )
    weather: Final = tuple(
        ScriptedTool(
            f"weather_{n}",
            lambda params: text_result("sunny"),
            description=f"Look up the weather forecast for city {n}",
        )
        for n in range(TOOL_COUNT - 1)
    )
    return (refund, *weather)


def _all_names(alias: str) -> frozenset[str]:
    return frozenset((f"{alias}-refund_invoice", *(f"{alias}-weather_{n}" for n in range(TOOL_COUNT - 1))))


def _descriptions(invoice_description: str = "Refund an invoice for a customer") -> frozenset[str]:
    return frozenset(tool.description for tool in _tools(invoice_description))


def _config(
    directory: Path,
    *,
    alias: str,
    peer_url: str,
    embeddings_url: str,
    chat_url: str,
    redis: OwnedRedis | None,
) -> Path:
    document: Final = {
        "model_list": [
            {
                "model_name": "semantic-embed",
                "litellm_params": {
                    "model": "openai/text-embedding-3-small",
                    "api_base": f"{embeddings_url}/v1",
                    "api_key": "synthetic-embed-key",
                },
            },
            {
                "model_name": "chat-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{chat_url}/v1",
                    "api_key": "synthetic-chat-key",
                },
            },
            {
                "model_name": "responses-model",
                "litellm_params": {
                    "model": "openai/responses/gpt-4o-mini",
                    "api_base": f"{chat_url}/v1",
                    "api_key": "synthetic-chat-key",
                },
            },
        ],
        "general_settings": {
            "master_key": "os.environ/LITELLM_MASTER_KEY",
            "database_url": "os.environ/DATABASE_URL",
            "store_model_in_db": True,
            **({"coordination_redis": {"host": redis.host, "port": redis.port}} if redis is not None else {}),
        },
        "mcp_servers": {alias: {"transport": "http", "url": peer_url}},
        "litellm_settings": {"mcp_semantic_tool_filter": SEMANTIC_SETTINGS},
    }
    path: Final = directory / f"semantic-filter-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.dump(document))
    return path


def _send(proxy: Gateway, surface: Surface) -> httpx.Response:
    if surface == "chat":
        return proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": "chat-model", "messages": [{"role": "user", "content": PROMPT}], "tools": [MCP_REF]},
        )
    return proxy.request(
        "POST",
        "/v1/responses",
        {"model": "responses-model", "input": PROMPT, "tools": [MCP_REF]},
    )


def _request_is_filtered(proxy: Gateway, surface: Surface, bodies: list[dict[str, object]], alias: str) -> bool:
    try:
        response: Final = _send(proxy, surface)
    except httpx.TransportError:
        return False
    return response.status_code == 200 and _peer_names(bodies[-1:], alias) == frozenset((f"{alias}-refund_invoice",))


@contextmanager
def _proxy_without_readiness_gate(
    gateway: Gateway, directory: Path, overrides: dict[str, str], *, config: Path
) -> Iterator[OwnedProxy]:
    """owned_proxy_process without the readiness wait, for builds that must stay unready."""
    root: Final = _proxy_root()
    environment: Final = _proxy_environment(gateway, overrides, NO_REDIS_ENV)
    output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(directory)))
    output.mkdir(parents=True, exist_ok=True)
    command: Final = (
        sys.executable,
        "-m",
        "integration._support.proxy",
        "--config",
        str(config),
        "--host",
        "127.0.0.1",
        "--num_workers",
        "1",
        "--timeout_worker_healthcheck",
        "5",
        *DB_PUSH,
    )
    launch: Final = _launch(command, root, environment, output)
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{launch.port}", timeout=15, trust_env=False, limits=GATEWAY_LIMITS
        ) as client:
            yield OwnedProxy(Gateway(client, gateway.key, gateway.upstream_url), launch.process, launch.log)
    finally:
        _stop_launch(launch)


def test_redis_backed_proxy_is_live_while_300_tool_index_builds(gateway: Gateway, tmp_path: Path) -> None:
    alias: Final = "sf" + uuid.uuid4().hex[:8]
    embeddings: Final = EmbeddingDouble("held")
    respond, bodies = _chat_double()
    with (
        owned_redis(tmp_path) as redis,
        scripted_peer(*_tools()) as peer,
        wire_server(respond) as chat_wire,
        wire_server(embeddings.respond) as embedding_wire,
    ):
        try:
            config: Final = _config(
                tmp_path,
                alias=alias,
                peer_url=peer.url,
                embeddings_url=embedding_wire.url,
                chat_url=chat_wire.url,
                redis=redis,
            )
            with owned_proxy_process(gateway, tmp_path, {}, config=config) as proxy:
                liveliness: Final = proxy.gateway.client.get("/health/liveliness")
                assert liveliness.status_code == 200, liveliness.text
                response: Final = _send(proxy.gateway, "chat")
                assert response.status_code == 200, response.text
                assert _peer_names(bodies[-1:], alias) == _all_names(alias), _peer_names(bodies[-1:], alias)
                embeddings.release.set()
                eventually(
                    lambda: _request_is_filtered(proxy.gateway, "chat", bodies, alias),
                    lambda filtered: filtered,
                    seconds=60,
                )
        finally:
            embeddings.release.set()


def test_second_proxy_on_warm_redis_embeds_no_descriptions(gateway: Gateway, tmp_path: Path) -> None:
    alias: Final = "sf" + uuid.uuid4().hex[:8]
    descriptions: Final = _descriptions()
    with owned_redis(tmp_path) as redis, scripted_peer(*_tools()) as peer:
        first_embeddings: Final = EmbeddingDouble("healthy")
        respond, bodies = _chat_double()
        with (
            wire_server(respond) as chat_wire,
            wire_server(first_embeddings.respond) as first_embedding_wire,
        ):
            config: Final = _config(
                tmp_path,
                alias=alias,
                peer_url=peer.url,
                embeddings_url=first_embedding_wire.url,
                chat_url=chat_wire.url,
                redis=redis,
            )
            with owned_proxy_process(gateway, tmp_path, {}, config=config) as first_proxy:
                eventually(
                    lambda: _request_is_filtered(first_proxy.gateway, "chat", bodies, alias),
                    lambda filtered: filtered,
                    seconds=60,
                )
        second_embeddings: Final = EmbeddingDouble("healthy")
        second_respond, second_bodies = _chat_double()
        with (
            wire_server(second_respond) as second_chat_wire,
            wire_server(second_embeddings.respond) as second_embedding_wire,
        ):
            config = _config(
                tmp_path,
                alias=alias,
                peer_url=peer.url,
                embeddings_url=second_embedding_wire.url,
                chat_url=second_chat_wire.url,
                redis=redis,
            )
            with owned_proxy_process(gateway, tmp_path, {}, config=config) as second_proxy:
                assert second_embeddings.document_inputs(descriptions) == []
                assert _request_is_filtered(second_proxy.gateway, "chat", second_bodies, alias)


def test_two_proxies_cold_starting_together_embed_each_description_once(gateway: Gateway, tmp_path: Path) -> None:
    alias: Final = "sf" + uuid.uuid4().hex[:8]
    descriptions: Final = _descriptions()
    first_embeddings: Final = EmbeddingDouble("healthy", delay=10.0)
    second_embeddings: Final = EmbeddingDouble("healthy")
    respond_a, bodies_a = _chat_double()
    respond_b, bodies_b = _chat_double()
    with (
        owned_redis(tmp_path) as redis,
        scripted_peer(*_tools()) as peer,
        wire_server(respond_a) as chat_wire_a,
        wire_server(respond_b) as chat_wire_b,
        wire_server(first_embeddings.respond) as embedding_wire_a,
        wire_server(second_embeddings.respond) as embedding_wire_b,
    ):
        config_a: Final = _config(
            tmp_path,
            alias=alias,
            peer_url=peer.url,
            embeddings_url=embedding_wire_a.url,
            chat_url=chat_wire_a.url,
            redis=redis,
        )
        config_b: Final = _config(
            tmp_path,
            alias=alias,
            peer_url=peer.url,
            embeddings_url=embedding_wire_b.url,
            chat_url=chat_wire_b.url,
            redis=redis,
        )
        with (
            owned_proxy_process(gateway, tmp_path, {}, config=config_a) as proxy_a,
            owned_proxy_process(gateway, tmp_path, {}, config=config_b) as proxy_b,
        ):
            eventually(
                lambda: _request_is_filtered(proxy_a.gateway, "chat", bodies_a, alias),
                lambda filtered: filtered,
                seconds=120,
            )
            eventually(
                lambda: _request_is_filtered(proxy_b.gateway, "chat", bodies_b, alias),
                lambda filtered: filtered,
                seconds=60,
            )
        document_inputs: Final = Counter(
            first_embeddings.document_inputs(descriptions) + second_embeddings.document_inputs(descriptions)
        )
        assert document_inputs == Counter({description: 1 for description in descriptions})


def test_edited_description_reembeds_only_that_tool(gateway: Gateway, tmp_path: Path) -> None:
    alias: Final = "sf" + uuid.uuid4().hex[:8]
    descriptions_v1: Final = _descriptions()
    edited: Final = "Cancel and refund an invoice instantly"
    with owned_redis(tmp_path) as redis:
        first_embeddings: Final = EmbeddingDouble("healthy")
        respond, bodies = _chat_double()
        with (
            scripted_peer(*_tools()) as peer_v1,
            wire_server(respond) as chat_wire,
            wire_server(first_embeddings.respond) as first_embedding_wire,
        ):
            config: Final = _config(
                tmp_path,
                alias=alias,
                peer_url=peer_v1.url,
                embeddings_url=first_embedding_wire.url,
                chat_url=chat_wire.url,
                redis=redis,
            )
            with owned_proxy_process(gateway, tmp_path, {}, config=config) as first_proxy:
                eventually(
                    lambda: _request_is_filtered(first_proxy.gateway, "chat", bodies, alias),
                    lambda filtered: filtered,
                    seconds=60,
                )
        second_embeddings: Final = EmbeddingDouble("healthy")
        second_respond, second_bodies = _chat_double()
        with (
            scripted_peer(*_tools(edited)) as peer_v2,
            wire_server(second_respond) as second_chat_wire,
            wire_server(second_embeddings.respond) as second_embedding_wire,
        ):
            config = _config(
                tmp_path,
                alias=alias,
                peer_url=peer_v2.url,
                embeddings_url=second_embedding_wire.url,
                chat_url=second_chat_wire.url,
                redis=redis,
            )
            with owned_proxy_process(gateway, tmp_path, {}, config=config) as second_proxy:
                assert second_embeddings.document_inputs(descriptions_v1) == []
                document_inputs: Final = [
                    text
                    for inputs, _ in second_embeddings.arrived
                    for text in inputs
                    if text not in descriptions_v1 and text not in ("test", PROMPT)
                ]
                assert document_inputs == [edited]
                eventually(
                    lambda: _request_is_filtered(second_proxy.gateway, "chat", second_bodies, alias),
                    lambda filtered: filtered,
                    seconds=60,
                )


def test_without_redis_startup_waits_for_the_index(gateway: Gateway, tmp_path: Path) -> None:
    alias: Final = "sf" + uuid.uuid4().hex[:8]
    embeddings: Final = EmbeddingDouble("held")
    respond, bodies = _chat_double()
    with (
        scripted_peer(*_tools()) as peer,
        wire_server(respond) as chat_wire,
        wire_server(embeddings.respond) as embedding_wire,
    ):
        try:
            config: Final = _config(
                tmp_path,
                alias=alias,
                peer_url=peer.url,
                embeddings_url=embedding_wire.url,
                chat_url=chat_wire.url,
                redis=None,
            )
            with _proxy_without_readiness_gate(gateway, tmp_path, {}, config=config) as proxy:

                def liveliness() -> int | None:
                    try:
                        return proxy.gateway.client.get("/health/liveliness", timeout=2).status_code
                    except httpx.TransportError:
                        return None

                deadline: Final = time.monotonic() + 5
                while time.monotonic() < deadline:
                    assert liveliness() != 200, "blocking build should keep liveliness from 200 while held"
                    time.sleep(0.5)
                embeddings.release.set()
                eventually(liveliness, lambda status: status == 200, seconds=60)
                eventually(
                    lambda: _request_is_filtered(proxy.gateway, "chat", bodies, alias),
                    lambda filtered: filtered,
                    seconds=60,
                )
        finally:
            embeddings.release.set()
