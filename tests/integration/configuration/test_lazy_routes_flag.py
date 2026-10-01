"""Route table contract for the LITELLM_DISABLE_LAZY_ROUTES startup flag.

By default optional feature routers (``LAZY_FEATURES``) are registered on the first
request to their path prefix, so an operator inspecting the route table right after
boot cannot see or gate them. With the flag set every feature is registered at worker
startup, so ``GET /routes`` lists them before any feature request is served and the
first feature request changes nothing.
"""

import asyncio
import json
import os
import re
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from litellm.proxy._lazy_features import LAZY_FEATURES, LazyFeature
from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.mcp import McpPeer, call_tool, echo_tool, scripted_peer, tool_calls, tool_names
from tests.integration._support.process import OwnedProxy, owned_proxy_process
from tests.integration._support.wire import Reply, Request, Wire, wire_server

TICKET_FEATURES: Final = ("mcp_management", "mcp_byok_oauth")
FLAG: Final = "LITELLM_DISABLE_LAZY_ROUTES"
WARMUP_ROUTE: Final = "/lazy/warm/{name}"
MCP_WARM_PATH: Final = "/mcp/enabled"
MARKER: Final = re.compile(rb"lazyroutes-[0-9a-f]{32}")
FAILED_FEATURE: Final = re.compile(r"Failed to lazy-load optional feature '([a-z_]+)'")
JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
HOOK_MODULE: Final = "lazy_routes_route_filter_hook"
HOOK_SOURCE: Final = """from litellm.proxy.proxy_server import app


def drop_mcp_routes() -> None:
    app.router.routes[:] = [
        route for route in app.router.routes if not getattr(route, "path", "").startswith(("/mcp", "/v1/mcp"))
    ]
"""


def _paths(candidate: Gateway) -> tuple[str, ...]:
    routes: Final = candidate.get("/routes")["routes"]
    assert isinstance(routes, list), routes
    return tuple(string_value(object_value(route)["path"]) for route in routes)


def _routed_features(candidate: Gateway) -> Mapping[str, tuple[str, ...]]:
    paths: Final = _paths(candidate)
    return {feature.name: tuple(path for path in paths if feature.matches(path)) for feature in LAZY_FEATURES}


def _mcp_paths(candidate: Gateway) -> tuple[str, ...]:
    return tuple(path for path in _paths(candidate) if path.startswith(("/mcp", "/v1/mcp")))


def _route_filter_hook(directory: Path) -> Mapping[str, str]:
    (directory / f"{HOOK_MODULE}.py").write_text(HOOK_SOURCE)
    search_path: Final = (str(directory), os.environ.get("PYTHONPATH", ""))
    return {
        "PYTHONPATH": os.pathsep.join(entry for entry in search_path if entry),
        "LITELLM_WORKER_STARTUP_HOOKS": f"{HOOK_MODULE}:drop_mcp_routes",
    }


def test_lazy_routes_are_absent_from_the_route_table_until_first_request(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, {}, remove_environment=(FLAG,)) as owned:
        at_boot: Final = _routed_features(owned.gateway)
        assert {name: at_boot[name] for name in TICKET_FEATURES} == {name: () for name in TICKET_FEATURES}, at_boot
        listing: Final = owned.gateway.request("GET", "/v1/mcp/server")
        assert listing.status_code == 200, listing.text
        after_first_request: Final = _routed_features(owned.gateway)
        assert after_first_request["mcp_management"] != (), "first request did not register the router"
        assert after_first_request["mcp_byok_oauth"] == (), "only the requested feature is mounted"


@pytest.mark.parametrize("workers", (1, 4))
def test_disable_lazy_routes_flag_registers_every_feature_at_startup(
    gateway: Gateway, tmp_path: Path, workers: int
) -> None:
    with owned_proxy_process(gateway, tmp_path, {FLAG: "true"}, workers=workers) as owned:
        at_boot: Final = tuple(_routed_features(owned.gateway) for _ in range(2 * workers))
        unregistered: Final = sorted(name for name, paths in at_boot[0].items() if not paths)
        assert unregistered == [], f"features still missing from /routes at startup: {unregistered}"
        assert all(table == at_boot[0] for table in at_boot), "workers disagree on the route table"
        listing: Final = owned.gateway.request("GET", "/v1/mcp/server")
        assert listing.status_code == 200, listing.text
        assert _routed_features(owned.gateway) == at_boot[0], "first feature request changed the route table"


def test_startup_hook_cannot_remove_lazy_routes_that_register_after_it_ran(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, _route_filter_hook(tmp_path), remove_environment=(FLAG,)) as owned:
        assert _mcp_paths(owned.gateway) == (), "hook should have removed the routes registered before it ran"
        listing: Final = owned.gateway.request("GET", "/v1/mcp/server")
        assert listing.status_code == 200, listing.text
        assert _mcp_paths(owned.gateway) != (), "first request should have registered the routes the hook never saw"


def test_disable_lazy_routes_flag_lets_a_startup_hook_remove_optional_routes_for_good(
    gateway: Gateway, tmp_path: Path
) -> None:
    overrides: Final = {**_route_filter_hook(tmp_path), FLAG: "true"}
    with owned_proxy_process(gateway, tmp_path, overrides) as owned:
        assert _mcp_paths(owned.gateway) == (), "hook should have seen and removed every MCP route"
        listing: Final = owned.gateway.request("GET", "/v1/mcp/server")
        assert listing.status_code == 404, listing.text
        mounted: Final = owned.gateway.request("POST", "/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert mounted.status_code == 404, mounted.text
        guardrails: Final = owned.gateway.request("GET", "/guardrails/list")
        assert guardrails.status_code == 200, guardrails.text
        assert _mcp_paths(owned.gateway) == (), "a feature request re-registered routes the hook removed"


def _openapi_paths(candidate: Gateway) -> Mapping[str, tuple[str, ...]]:
    paths: Final = object_value(candidate.get("/openapi.json")["paths"])
    return {path: tuple(sorted(object_value(operations))) for path, operations in paths.items()}


def _published(feature: LazyFeature, paths: Mapping[str, tuple[str, ...]]) -> bool:
    return any(feature.matches(path) for path in paths)


def _warm_every_feature(candidate: Gateway) -> None:
    warmed: Final = tuple(
        (feature.name, candidate.request("POST", f"/lazy/warm/{feature.name}")) for feature in LAZY_FEATURES
    )
    cold: Final = [
        (name, response.status_code, response.text) for name, response in warmed if response.status_code != 200
    ]
    assert cold == [], cold
    enabled: Final = candidate.request("GET", MCP_WARM_PATH)
    assert enabled.status_code == 200, enabled.text
    unregistered: Final = sorted(name for name, paths in _routed_features(candidate).items() if not paths)
    assert unregistered == [], f"features still missing after warming every one of them: {unregistered}"


def _shadowed_dependency(directory: Path) -> Mapping[str, str]:
    package: Final = directory / "shadow" / "RestrictedPython"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('raise ImportError("shadowed by the lazy routes audit")\n')
    search_path: Final = (str(package.parent), os.environ.get("PYTHONPATH", ""))
    return {"PYTHONPATH": os.pathsep.join(entry for entry in search_path if entry)}


def _failed_features(owned: OwnedProxy) -> frozenset[str]:
    return frozenset(FAILED_FEATURE.findall(owned.log.read_text()))


def _marker() -> str:
    return "lazyroutes-" + uuid.uuid4().hex


def _chat_reply(identity: str, stream: bool) -> Reply:
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "lazy ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                }
            ).encode()
        )
    chunk: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
    }
    deltas: Final[tuple[dict[str, JsonValue], ...]] = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "lazy"}}]},
        {**chunk, "choices": [{"index": 0, "delta": {"content": " ok"}, "finish_reason": "stop"}]},
        {**chunk, "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(*(b"data: " + json.dumps(delta).encode() + b"\n\n" for delta in deltas), b"data: [DONE]\n\n"),
    )


def _responses_reply(identity: str, stream: bool) -> Reply:
    response: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "id": "msg_" + identity,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "lazy ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
    }
    if not stream:
        return Reply(body=json.dumps(response).encode())
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {"type": "response.created", "sequence_number": 0, "response": {**response, "status": "in_progress"}},
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_" + identity,
            "output_index": 0,
            "content_index": 0,
            "delta": "lazy ok",
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _upstream(request: Request) -> Reply:
    found: Final = MARKER.search(request.body)
    if found is None:
        return Reply(status=404, body=b'{"error":"no marker"}')
    marker: Final = found.group(0).decode()
    stream: Final = object_value(JSON.validate_json(request.body)).get("stream") is True
    if request.target.endswith("/responses"):
        return _responses_reply(f"resp_{marker}", stream)
    return _chat_reply(f"chatcmpl-{marker}", stream)


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_upstream) as wire:
        yield wire


async def _stream_chat(base_url: str, key: str, model: str, marker: str) -> tuple[frozenset[str], str]:
    client: Final = openai.AsyncOpenAI(base_url=base_url + "/v1", api_key=key, max_retries=0)
    stream: Final = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": marker}], stream=True
    )
    chunks: Final = [chunk async for chunk in stream]
    text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    return frozenset(chunk.id for chunk in chunks), text


async def _stream_message(base_url: str, key: str, model: str, marker: str) -> str:
    client: Final = anthropic.AsyncAnthropic(base_url=base_url, api_key=key, max_retries=0)
    async with client.messages.stream(
        model=model, max_tokens=16, messages=[{"role": "user", "content": marker}]
    ) as stream:
        return "".join([text async for text in stream.text_stream])


def _status(candidate: Gateway, path: str) -> int:
    return candidate.request("GET", path).status_code


def _workers(owned: OwnedProxy) -> tuple[psutil.Process, ...]:
    return tuple(child for child in psutil.Process(owned.process.pid).children() if _is_worker(child))


def _is_worker(child: psutil.Process) -> bool:
    try:
        return "spawn_main" in " ".join(child.cmdline()) and child.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


@pytest.mark.parametrize("spelling", ("1", "Yes", "ON"))
def test_every_truthy_spelling_of_the_flag_registers_the_ticket_features_at_startup(
    gateway: Gateway, tmp_path: Path, spelling: str
) -> None:
    with owned_proxy_process(gateway, tmp_path, {FLAG: spelling}) as owned:
        at_boot: Final = _routed_features(owned.gateway)
        assert all(at_boot[name] for name in TICKET_FEATURES), {name: at_boot[name] for name in TICKET_FEATURES}


@pytest.mark.parametrize(
    "spelling", ("", "0", "off", "maybe", "x" * 5000), ids=("empty", "zero", "off", "unknown-word", "five-kilobytes")
)
def test_a_falsey_or_unknown_flag_value_keeps_the_default_lazy_registration(
    gateway: Gateway, tmp_path: Path, spelling: str
) -> None:
    with owned_proxy_process(gateway, tmp_path, {FLAG: spelling}) as owned:
        at_boot: Final = _routed_features(owned.gateway)
        assert {name: at_boot[name] for name in TICKET_FEATURES} == {name: () for name in TICKET_FEATURES}, at_boot
        listing: Final = owned.gateway.request("GET", "/v1/mcp/server")
        assert listing.status_code == 200, listing.text
        assert _routed_features(owned.gateway)["mcp_management"] != (), "first request did not register the router"


def test_disable_lazy_routes_flag_publishes_the_live_route_table_in_openapi_before_any_feature_request(
    gateway: Gateway, tmp_path: Path
) -> None:
    with owned_proxy_process(gateway, tmp_path, {FLAG: "true"}) as owned:
        at_boot: Final = _openapi_paths(owned.gateway)
        unpublished: Final = sorted(
            feature.name
            for feature in LAZY_FEATURES
            if feature.name in TICKET_FEATURES and not _published(feature, at_boot)
        )
        assert unpublished == [], f"ticket features missing from /openapi.json at startup: {unpublished}"
        assert "get" in at_boot["/v1/mcp/server"], at_boot["/v1/mcp/server"]
        assert WARMUP_ROUTE not in at_boot
        stranger: Final = owned.gateway.request("GET", "/v1/mcp/server", key="sk-not-a-real-key")
        assert stranger.status_code == 401, stranger.text
        listing: Final = owned.gateway.request("GET", "/v1/mcp/server")
        assert listing.status_code == 200, listing.text
        assert _openapi_paths(owned.gateway) == at_boot, "first feature request changed /openapi.json"


def test_disable_lazy_routes_flag_removes_the_warmup_route(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, {FLAG: "true"}) as owned:
        assert WARMUP_ROUTE not in _paths(owned.gateway)
        warmed: Final = owned.gateway.request("POST", "/lazy/warm/mcp_management")
        assert warmed.status_code == 404, warmed.text
        listing: Final = owned.gateway.request("GET", "/v1/mcp/server")
        assert listing.status_code == 200, listing.text


def test_the_warmup_route_registers_a_feature_on_demand_by_default(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, {}, remove_environment=(FLAG,)) as owned:
        assert WARMUP_ROUTE in _paths(owned.gateway)
        warmed: Final = owned.gateway.request("POST", "/lazy/warm/mcp_management")
        assert warmed.status_code == 200, warmed.text
        assert "/v1/mcp/server" in object_value(object_value(JSON.validate_json(warmed.content))["paths"])
        assert _routed_features(owned.gateway)["mcp_management"] != (), "warmup did not register the router"


def test_disable_lazy_routes_flag_keeps_the_fixed_mcp_proxy_route_ahead_of_the_mcp_mount(
    gateway: Gateway, tmp_path: Path
) -> None:
    with owned_proxy_process(gateway, tmp_path, {}, remove_environment=(FLAG,)) as lazy:
        control: Final = lazy.gateway.request("POST", "/mcp/proxy", {})
        assert control.status_code == 400, control.text
        warmed: Final = _paths(lazy.gateway)
        assert warmed.count("/mcp") == 2, "expected the fixed /mcp route and the /mcp mount"
    with owned_proxy_process(gateway, tmp_path, {FLAG: "true"}) as eager:
        at_boot: Final = _paths(eager.gateway)
        assert at_boot.count("/mcp") == 2, "expected the fixed /mcp route and the /mcp mount at startup"
        mount: Final = max(index for index, path in enumerate(at_boot) if path == "/mcp")
        assert at_boot.index("/mcp/proxy") < mount, "the /mcp mount shadows /mcp/proxy"
        proxied: Final = eager.gateway.request("POST", "/mcp/proxy", {})
        assert (proxied.status_code, proxied.text) == (control.status_code, control.text)


def test_disable_lazy_routes_flag_matches_the_fully_warmed_lazy_route_table_and_openapi(
    gateway: Gateway, tmp_path: Path
) -> None:
    with owned_proxy_process(gateway, tmp_path, {}, remove_environment=(FLAG,)) as lazy:
        _warm_every_feature(lazy.gateway)
        warmed_paths: Final = tuple(path for path in _paths(lazy.gateway) if path != WARMUP_ROUTE)
        warmed_openapi: Final = _openapi_paths(lazy.gateway)
    with owned_proxy_process(gateway, tmp_path, {FLAG: "true"}) as eager:
        assert _paths(eager.gateway) == warmed_paths
        assert _openapi_paths(eager.gateway) == warmed_openapi


def test_disable_lazy_routes_flag_keeps_registering_after_an_optional_dependency_fails_to_import(
    gateway: Gateway, tmp_path: Path
) -> None:
    with owned_proxy_process(gateway, tmp_path, {**_shadowed_dependency(tmp_path), FLAG: "true"}) as owned:
        at_boot: Final = _routed_features(owned.gateway)
        unregistered: Final = frozenset(name for name, paths in at_boot.items() if not paths)
        failed: Final = _failed_features(owned)
        assert "guardrails" in failed, owned.log.read_text()
        assert unregistered == failed, (sorted(unregistered), sorted(failed))
        guardrails: Final = owned.gateway.request("GET", "/guardrails/list")
        assert guardrails.status_code == 404, guardrails.text
        listing: Final = owned.gateway.request("GET", "/v1/mcp/server")
        assert listing.status_code == 200, listing.text
        stores: Final = owned.gateway.request("GET", "/vector_store/list")
        assert stores.status_code == 200, stores.text
        assert _routed_features(owned.gateway) == at_boot, "feature requests changed the route table"


def test_a_broken_optional_dependency_only_404s_its_own_feature_by_default(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, _shadowed_dependency(tmp_path), remove_environment=(FLAG,)) as owned:
        guardrails: Final = owned.gateway.request("GET", "/guardrails/list")
        assert guardrails.status_code == 404, guardrails.text
        assert "guardrails" in _failed_features(owned), owned.log.read_text()
        listing: Final = owned.gateway.request("GET", "/v1/mcp/server")
        assert listing.status_code == 200, listing.text
        assert _routed_features(owned.gateway)["mcp_management"] != ()


def test_disable_lazy_routes_flag_leaves_the_completion_endpoints_serving_every_client(
    gateway: Gateway, tmp_path: Path, provider: Wire
) -> None:
    with owned_proxy_process(gateway, tmp_path, {FLAG: "true"}) as owned, owned.gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        base_url: Final = str(owned.gateway.client.base_url)
        key: Final = owned.gateway.key
        markers: Final = tuple(_marker() for _ in range(6))

        completion: Final = openai.OpenAI(
            base_url=base_url + "/v1", api_key=key, max_retries=0
        ).chat.completions.create(model=model, messages=[{"role": "user", "content": markers[0]}])
        assert (completion.id, completion.choices[0].message.content) == (f"chatcmpl-{markers[0]}", "lazy ok")

        assert asyncio.run(_stream_chat(base_url, key, model, markers[1])) == (
            frozenset({f"chatcmpl-{markers[1]}"}),
            "lazy ok",
        )

        message: Final = anthropic.Anthropic(base_url=base_url, api_key=key, max_retries=0).messages.create(
            model=model, max_tokens=16, messages=[{"role": "user", "content": markers[2]}]
        )
        assert [block.text for block in message.content if block.type == "text"] == ["lazy ok"]

        assert asyncio.run(_stream_message(base_url, key, model, markers[3])) == "lazy ok"

        responded: Final = owned.gateway.request("POST", "/v1/responses", {"model": model, "input": markers[4]})
        assert responded.status_code == 200, responded.text
        response: Final = object_value(JSON.validate_json(responded.content))
        assert (response["status"], response["object"]) == ("completed", "response"), responded.text
        assert "lazy ok" in responded.text, responded.text

        streamed: Final = owned.gateway.request(
            "POST", "/v1/responses", {"model": model, "input": markers[5], "stream": True}
        )
        assert streamed.status_code == 200, streamed.text
        assert "response.completed" in streamed.text and "lazy ok" in streamed.text, streamed.text

        reached: Final = tuple(request.target for request in provider.drain() if MARKER.search(request.body))
        assert len(reached) == 6, reached


def test_disable_lazy_routes_flag_route_table_survives_a_boot_burst_and_a_killed_worker(
    gateway: Gateway, tmp_path: Path
) -> None:
    probes: Final = ("/routes", "/v1/mcp/server", "/openapi.json", "/guardrails/list", "/vector_store/list") * 8
    with owned_proxy_process(gateway, tmp_path, {FLAG: "true"}, workers=2) as owned:
        at_boot: Final = _routed_features(owned.gateway)
        unregistered: Final = sorted(name for name, paths in at_boot.items() if not paths)
        assert unregistered == [], f"features still missing from /routes at startup: {unregistered}"
        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses: Final = tuple(pool.map(partial(_status, owned.gateway), probes))
        assert statuses == (200,) * len(probes), statuses
        assert _routed_features(owned.gateway) == at_boot, "the boot burst changed the route table"

        victim: Final = eventually(lambda: _workers(owned), lambda workers: len(workers) == 2)[0]
        victim.kill()
        with httpx.Client(base_url=owned.gateway.client.base_url, timeout=15, trust_env=False) as fresh:
            survivor: Final = Gateway(fresh, owned.gateway.key, owned.gateway.upstream_url)
            during: Final = tuple(survivor.request("GET", "/v1/mcp/server").status_code for _ in range(10))
        assert during == (200,) * 10, during
        respawned: Final = eventually(
            lambda: frozenset(worker.pid for worker in _workers(owned)),
            lambda pids: len(pids) == 2 and victim.pid not in pids,
            seconds=30,
        )
        assert f"Child process [{victim.pid}] died" in owned.log.read_text(), respawned
        tables: Final = tuple(_routed_features(owned.gateway) for _ in range(4))
        assert all(table == at_boot for table in tables), "the respawned worker disagrees on the route table"


def test_disable_lazy_routes_flag_yields_the_same_route_table_after_a_restart(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, {FLAG: "true"}) as first:
        table: Final = _paths(first.gateway)
    with owned_proxy_process(gateway, tmp_path, {FLAG: "true"}) as second:
        assert _paths(second.gateway) == table
        assert all(_routed_features(second.gateway).values()), "a feature is missing after restart"


SELF_HOSTED_LANGFUSE: Final = "/self-hosted-langfuse"


@dataclass(frozen=True, slots=True)
class _ConfiguredFeatures:
    alias: str
    config: Path
    policy: Wire
    langfuse: Wire
    peer: McpPeer


def _allow(request: Request) -> Reply:
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def _langfuse_health(request: Request) -> Reply:
    return Reply(body=json.dumps({"status": "OK"}).encode())


def _config_declaring(directory: Path, alias: str, policy: Wire, langfuse: Wire, peer: McpPeer) -> Path:
    base: Final = object_value(
        JSON.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    )
    config: Final = {
        **base,
        "guardrails": [
            {
                "guardrail_name": alias,
                "litellm_params": {
                    "guardrail": "generic_guardrail_api",
                    "mode": "pre_call",
                    "default_on": True,
                    "api_base": policy.url,
                    "api_key": "synthetic-guardrail-key",
                },
            }
        ],
        "mcp_servers": {alias: peer.registration()},
        "general_settings": {
            **object_value(base["general_settings"]),
            "pass_through_endpoints": [
                {
                    "path": "/langfuse",
                    "target": langfuse.url + SELF_HOSTED_LANGFUSE,
                    "include_subpath": True,
                    "auth": True,
                }
            ],
        },
    }
    path: Final = directory / "configured-features.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _configured_features(directory: Path) -> Iterator[_ConfiguredFeatures]:
    alias: Final = "lazyroutes" + uuid.uuid4().hex[:8]
    with (
        wire_server(_allow) as policy,
        wire_server(_langfuse_health) as langfuse,
        scripted_peer(echo_tool("add")) as peer,
    ):
        config: Final = _config_declaring(directory, alias, policy, langfuse, peer)
        yield _ConfiguredFeatures(alias, config, policy, langfuse, peer)


def _config_server_id(candidate: Gateway, alias: str) -> str:
    servers: Final = JSON.validate_json(candidate.request("GET", "/v1/mcp/server").content)
    assert isinstance(servers, list), servers
    return next(
        string_value(object_value(server)["server_id"])
        for server in servers
        if object_value(server)["server_name"] == alias
    )


def _assert_config_declared_features_serve(owned: OwnedProxy, features: _ConfiguredFeatures, provider: Wire) -> None:
    marker: Final = _marker()
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=provider.url + "/v1")
        completion: Final = owned.gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}
        )
        assert completion.status_code == 200, completion.text
        screened: Final = [request for request in features.policy.drain() if marker.encode() in request.body]
        assert len(screened) == 1, "the config-declared guardrail did not screen the completion"
        assert len([request for request in provider.drain() if marker.encode() in request.body]) == 1

        identity: Final = _config_server_id(owned.gateway, features.alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        tool: Final = tool_names(owned.gateway, key, identity)["add"]
        features.peer.drain()
        called: Final = call_tool(owned.gateway, key, identity, tool, {"marker": marker})
        assert called.status_code == 200, called.text
        reached_peer: Final = [
            object_value(object_value(call["body"])["params"]) for call in tool_calls(features.peer.drain())
        ]
        assert [(params["name"], params["arguments"]) for params in reached_peer] == [("add", {"marker": marker})]

    forwarded: Final = owned.gateway.request("GET", "/langfuse/api/public/health")
    assert forwarded.status_code == 200, forwarded.text
    reached_langfuse: Final = tuple(request.target for request in features.langfuse.drain())
    assert reached_langfuse == (SELF_HOSTED_LANGFUSE + "/api/public/health",), (
        f"the config pass-through for /langfuse lost to the built-in Langfuse route: {reached_langfuse}"
    )


def test_disable_lazy_routes_flag_serves_config_declared_features_like_the_warmed_lazy_proxy(
    gateway: Gateway, tmp_path: Path, provider: Wire
) -> None:
    with _configured_features(tmp_path) as features:
        with owned_proxy_process(gateway, tmp_path, {}, config=features.config, remove_environment=(FLAG,)) as lazy:
            _assert_config_declared_features_serve(lazy, features, provider)
            _warm_every_feature(lazy.gateway)
            warmed_paths: Final = tuple(path for path in _paths(lazy.gateway) if path != WARMUP_ROUTE)
            warmed_openapi: Final = _openapi_paths(lazy.gateway)
        with owned_proxy_process(gateway, tmp_path, {FLAG: "true"}, config=features.config) as eager:
            _assert_config_declared_features_serve(eager, features, provider)
            assert _paths(eager.gateway) == warmed_paths
            assert _openapi_paths(eager.gateway) == warmed_openapi
