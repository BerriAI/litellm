import json
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import scratch_database
from integration._support.mcp import ScriptedTool, scripted_peer, text_result
from integration._support.process import owned_proxy_process
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
BUILD_FAILED_LINE: Final = "MCP semantic tool index deferred build failed"
RELOAD_TICK_LINE: Final = "Successfully loaded 0 search tool(s) into router"
Surface = Literal["chat", "responses"]


class EmbeddingDouble:
    def __init__(self, mode: Literal["healthy", "held", "failing"], delay: float = 0.0) -> None:
        self.release = threading.Event()
        self.mode = mode
        self.delay = delay
        self.arrived: list[tuple[tuple[str, ...], float]] = []
        self.answered: list[tuple[tuple[str, ...], float]] = []

    def inputs(self) -> list[str]:
        return [text for inputs, _ in self.arrived for text in inputs]

    def respond(self, request: Request) -> Reply:
        if request.method == "GET" and request.target.endswith("/models"):
            return Reply(body=b'{"object": "list", "data": []}')
        assert request.method == "POST" and request.target.endswith("/embeddings"), request.target
        body: Final = json.loads(request.body)
        raw: Final = body.get("input", [])
        inputs: Final = tuple(str(text) for text in raw) if isinstance(raw, list) else (str(raw),)
        self.arrived.append((inputs, time.monotonic()))
        if self.mode == "failing":
            return Reply(status=500, body=b'{"error": {"message": "scripted embedding failure"}}')
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


def _tools() -> tuple[ScriptedTool, ...]:
    refund: Final = ScriptedTool(
        "refund_invoice",
        lambda params: text_result("refunded"),
        description="Refund an invoice for a customer",
    )
    weather: Final = tuple(
        ScriptedTool(
            f"weather_{n}",
            lambda params: text_result("sunny"),
            description=f"Look up the weather forecast for city {n}",
        )
        for n in range(49)
    )
    return (refund, *weather)


def _all_names(alias: str) -> frozenset[str]:
    return frozenset((f"{alias}-refund_invoice", *(f"{alias}-weather_{n}" for n in range(49))))


def _config(
    directory: Path,
    *,
    alias: str,
    peer_url: str,
    embeddings_url: str,
    chat_url: str,
    semantic_filter: dict[str, object] | None,
    reload_interval_seconds: int | None = None,
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
            **(
                {"proxy_config_reload_interval_seconds": reload_interval_seconds}
                if reload_interval_seconds is not None
                else {}
            ),
        },
        "mcp_servers": {alias: {"transport": "http", "url": peer_url}},
        "litellm_settings": ({"mcp_semantic_tool_filter": semantic_filter} if semantic_filter is not None else {}),
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
    response: Final = _send(proxy, surface)
    return response.status_code == 200 and _peer_names(bodies[-1:], alias) == frozenset((f"{alias}-refund_invoice",))


def _log_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def test_deferred_build_serves_liveliness_while_embeddings_are_held(gateway: Gateway, tmp_path: Path) -> None:
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
                semantic_filter={**SEMANTIC_SETTINGS, "defer_index_build": True},
            )
            with owned_proxy_process(gateway, tmp_path, {}, config=config) as proxy:
                liveliness: Final = proxy.gateway.client.get("/health/liveliness")
                assert liveliness.status_code == 200, liveliness.text
                eventually(lambda: len(embeddings.arrived), lambda count: count >= 1, seconds=30)
                assert embeddings.answered == [], "held embedding double released a response"
        finally:
            embeddings.release.set()


@pytest.mark.parametrize("surface", ("chat", "responses"))
def test_deferred_build_passes_tools_through_then_filters(gateway: Gateway, tmp_path: Path, surface: Surface) -> None:
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
                semantic_filter={**SEMANTIC_SETTINGS, "defer_index_build": True},
            )
            with owned_proxy_process(gateway, tmp_path, {}, config=config) as proxy:
                response: Final = _send(proxy.gateway, surface)
                assert response.status_code == 200, response.text
                assert _peer_names(bodies[-1:], alias) == _all_names(alias), _peer_names(bodies[-1:], alias)
                embeddings.release.set()
                eventually(
                    lambda: _request_is_filtered(proxy.gateway, surface, bodies, alias),
                    lambda filtered: filtered,
                    seconds=60,
                )
        finally:
            embeddings.release.set()


def test_failed_deferred_build_falls_back_to_request_time_indexing(gateway: Gateway, tmp_path: Path) -> None:
    alias: Final = "sf" + uuid.uuid4().hex[:8]
    embeddings: Final = EmbeddingDouble("failing")
    respond, bodies = _chat_double()
    with (
        scripted_peer(*_tools()) as peer,
        wire_server(embeddings.respond) as embedding_wire,
        wire_server(respond) as chat_wire,
    ):
        config: Final = _config(
            tmp_path,
            alias=alias,
            peer_url=peer.url,
            embeddings_url=embedding_wire.url,
            chat_url=chat_wire.url,
            semantic_filter={**SEMANTIC_SETTINGS, "defer_index_build": True},
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config) as proxy:
            eventually(
                lambda: _log_text(proxy.log),
                lambda text: BUILD_FAILED_LINE in text,
                seconds=60,
            )
            embeddings.mode = "healthy"
            eventually(
                lambda: _request_is_filtered(proxy.gateway, "chat", bodies, alias),
                lambda filtered: filtered,
                seconds=60,
            )


def test_default_build_completes_before_readiness(gateway: Gateway, tmp_path: Path) -> None:
    alias: Final = "sf" + uuid.uuid4().hex[:8]
    embeddings: Final = EmbeddingDouble("healthy", delay=2.0)
    respond, bodies = _chat_double()
    with (
        scripted_peer(*_tools()) as peer,
        wire_server(embeddings.respond) as embedding_wire,
        wire_server(respond) as chat_wire,
    ):
        config: Final = _config(
            tmp_path,
            alias=alias,
            peer_url=peer.url,
            embeddings_url=embedding_wire.url,
            chat_url=chat_wire.url,
            semantic_filter=SEMANTIC_SETTINGS,
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config) as proxy:
            ready_at: Final = time.monotonic()
            assert embeddings.answered, "startup never embedded the tool descriptions"
            last_response: Final = max(timestamp for _, timestamp in embeddings.answered)
            assert last_response <= ready_at, "index build finished after the proxy became ready"
            response: Final = _send(proxy.gateway, "chat")
            assert response.status_code == 200, response.text
            assert _peer_names(bodies, alias) == frozenset((f"{alias}-refund_invoice",)), _peer_names(bodies, alias)


def test_db_enabled_deferred_build_does_not_block_and_builds_once(gateway: Gateway, tmp_path: Path) -> None:
    alias: Final = "sf" + uuid.uuid4().hex[:8]
    embeddings: Final = EmbeddingDouble("held")
    respond, bodies = _chat_double()
    with scratch_database() as database_url:
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
                    semantic_filter=None,
                    reload_interval_seconds=2,
                )
                with owned_proxy_process(gateway, tmp_path, {"DATABASE_URL": database_url}, config=config) as proxy:
                    updated: Final = proxy.gateway.client.patch(
                        "/update/mcp_semantic_filter_settings",
                        json={**SEMANTIC_SETTINGS, "defer_index_build": True},
                        headers={"Authorization": f"Bearer {gateway.key}"},
                        timeout=15,
                    )
                    assert updated.status_code == 200, updated.text
                    fetched: Final = proxy.gateway.get("/get/mcp_semantic_filter_settings")
                    assert fetched["values"]["defer_index_build"] is True, fetched
                    liveliness: Final = proxy.gateway.client.get("/health/liveliness")
                    assert liveliness.status_code == 200, liveliness.text
                    embeddings.release.set()
                    eventually(
                        lambda: _request_is_filtered(proxy.gateway, "chat", bodies, alias),
                        lambda filtered: filtered,
                        seconds=60,
                    )
                    ticks_at_filtered: Final = _log_text(proxy.log).count(RELOAD_TICK_LINE)
                    eventually(
                        lambda: _log_text(proxy.log).count(RELOAD_TICK_LINE),
                        lambda ticks: ticks >= ticks_at_filtered + 4,
                        seconds=30,
                    )
                    descriptions: Final = [tool.description for tool in _tools()]
                    inputs: Final = embeddings.inputs()
                    for description in descriptions:
                        assert inputs.count(description) == 1, (
                            f"expected one index build; saw {inputs.count(description)} embeddings of {description!r}"
                        )
            finally:
                embeddings.release.set()
