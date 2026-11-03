import json
import threading
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, gateway_from_environment
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._dashscope_video_support import (
    CREATE_PATH,
    MODEL,
    MP4,
    OSS_PATH,
    PROMPT,
    TASKS_PATH,
    create_bodies,
    created_id,
    forged_id,
    json_reply,
    polled_tasks,
    task_api,
    task_of,
)
from pydantic import JsonValue

pytestmark: Final = pytest.mark.xdist_group("dashscope-video-owned-proxies")

_CHAOS_TIMEOUT: Final = int(4 * graceful_stop_seconds() + 240)
_GROUP: Final = "dashscope-video-group"
_MEMBER_A: Final = "dashscope-video-a"
_MEMBER_B: Final = "dashscope-video-b"
_CATCH_ALL_KEY: Final = "catch-all-openai-key"
_ROUTED_ENVIRONMENT: Final = (
    "DASHSCOPE_API_BASE",
    "DASHSCOPE_API_BASE_VIDEO",
    "DASHSCOPE_API_KEY",
    "QWENCLOUD_API_BASE",
    "QWENCLOUD_API_BASE_VIDEO",
    "QWENCLOUD_API_KEY",
    "QWEN_AI_PLATFORM_API_BASE",
    "QWEN_AI_PLATFORM_API_BASE_VIDEO",
    "QWEN_AI_PLATFORM_API_KEY",
)
_CHAT_REPLY: Final = json_reply(
    {
        "id": "chatcmpl-dashscope-env",
        "object": "chat.completion",
        "created": 1791943200,
        "model": "qwen-plus",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "kite"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
)


def _config(
    tmp_path: Path,
    name: str,
    model_list: tuple[Mapping[str, JsonValue], ...],
    router_settings: Mapping[str, JsonValue],
    litellm_settings: Mapping[str, JsonValue],
) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **base,
                "model_list": [dict(entry) for entry in model_list],
                "router_settings": {**base["router_settings"], **router_settings},
                "litellm_settings": {**base["litellm_settings"], **litellm_settings},
            }
        )
    )
    return path


def _member(name: str, wire: Wire, key: str) -> Mapping[str, JsonValue]:
    return {"model_name": name, "litellm_params": {"model": f"dashscope/{MODEL}", "api_base": wire.url, "api_key": key}}


def _create(gateway: Gateway, body: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", "/v1/videos", {"prompt": PROMPT, **body})


def _completed(gateway: Gateway, video_id: str) -> bool:
    response: Final = gateway.request("GET", f"/v1/videos/{video_id}")
    assert response.status_code == 200, response.text
    return JSON_OBJECT.validate_json(response.content)["status"] == "completed"


def _gated(
    release: threading.Event, armed: threading.Event, held: SimpleQueue[str], prefix: str
) -> Callable[[Request], Reply]:
    tasks: Final = task_api(prefix=prefix)

    def respond(request: Request) -> Reply:
        if armed.is_set() and request.method == "GET" and request.target.startswith(TASKS_PATH):
            held.put(request.target)
            assert release.wait(timeout=120), "The held polls were never released"
        return tasks(request)

    return respond


def _fresh_poll(gateway: Gateway, video_id: str) -> int | None:
    try:
        with httpx.Client(base_url=gateway.client.base_url, timeout=150, trust_env=False) as client:
            return client.get(f"/v1/videos/{video_id}", headers={"Authorization": f"Bearer {gateway.key}"}).status_code
    except httpx.TransportError:
        return None


def _held_connections(pid: int, wires: tuple[Wire, ...]) -> int:
    ports: Final = frozenset(urlsplit(wire.url).port for wire in wires)
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port in ports
    )


def _assert_each_polls_on_its_creating_member(gateway: Gateway, video_ids: tuple[str, ...], a: Wire, b: Wire) -> None:
    a.drain()
    b.drain()
    assert all(_completed(gateway, video_id) for video_id in video_ids)
    a_tasks: Final = sorted(task_of(video_id) for video_id in video_ids if task_of(video_id).startswith("a-"))
    b_tasks: Final = sorted(task_of(video_id) for video_id in video_ids if task_of(video_id).startswith("b-"))
    assert sorted(polled_tasks(a.drain())) == a_tasks
    assert sorted(polled_tasks(b.drain())) == b_tasks


@pytest.mark.timeout(_CHAOS_TIMEOUT)
def test_a_worker_killed_mid_poll_burst_and_a_restart_leave_every_task_polling_its_creating_member(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    armed: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    with (
        wire_server(_gated(release, armed, held, "a")) as a,
        wire_server(_gated(release, armed, held, "b")) as b,
    ):
        path: Final = _config(
            tmp_path,
            "dashscope-video-chaos",
            (
                _member(_MEMBER_A, a, "key-a"),
                _member(_MEMBER_B, b, "key-b"),
            ),
            {
                "routing_groups": [
                    {"group_name": _GROUP, "models": [_MEMBER_A, _MEMBER_B], "routing_strategy": "simple-shuffle"}
                ]
            },
            {},
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(owned.worker_pids, lambda pids: len(pids) == 2, seconds=60)
            video_ids: Final = tuple(
                created_id(_create(candidate, {"model": _GROUP, "seconds": "5", "size": "1280x720"})) for _ in range(24)
            )
            assert {task_of(video_id)[0] for video_id in video_ids} == {"a", "b"}
            a.drain()
            b.drain()
            armed.set()
            with ThreadPoolExecutor(max_workers=len(video_ids)) as pool:
                polls: Final = tuple(pool.submit(_fresh_poll, candidate, video_id) for video_id in video_ids)
                eventually(held.qsize, lambda size: size == len(video_ids), seconds=60)
                held_by: Final = MappingProxyType({pid: _held_connections(pid, (a, b)) for pid in workers})
                assert sum(held_by.values()) == len(video_ids), held_by
                victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
                with owned.replacing_worker(victim_pid) as kill:
                    kill.freeze()
                    kill()
                    armed.clear()
                    release.set()
                    statuses: Final = tuple(poll.result() for poll in polls)
            assert statuses.count(200) == held_by[survivor_pid], (held_by, statuses)
            assert all(status in (200, None) for status in statuses), statuses
            _assert_each_polls_on_its_creating_member(candidate, video_ids, a, b)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as restarted:
            _assert_each_polls_on_its_creating_member(restarted.gateway, video_ids, a, b)


@pytest.fixture(scope="module")
def environment_wires() -> Iterator[Mapping[str, Wire]]:
    with (
        wire_server(task_api()) as compatible,
        wire_server(task_api()) as qwencloud,
        wire_server(task_api()) as platform,
    ):
        yield MappingProxyType({"dashscope": compatible, "qwencloud": qwencloud, "qwen_ai_platform": platform})


@pytest.fixture(scope="module")
def catch_all_wire() -> Iterator[Wire]:
    with wire_server(
        lambda _request: json_reply({"error": {"message": "catch-all", "type": "invalid_request_error"}}, 404)
    ) as wire:
        yield wire


@pytest.fixture(scope="module")
def environment_proxy(
    environment_wires: Mapping[str, Wire], catch_all_wire: Wire, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("dashscope-video-environment")
    path: Final = _config(
        directory,
        "dashscope-video-environment",
        (
            *(
                {"model_name": f"env-{provider}", "litellm_params": {"model": f"{provider}/{MODEL}"}}
                for provider in environment_wires
            ),
            {
                "model_name": "*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_base": f"{catch_all_wire.url}/v1",
                    "api_key": _CATCH_ALL_KEY,
                },
            },
        ),
        {},
        {"drop_params": True},
    )
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            {
                "DASHSCOPE_API_BASE": f"{environment_wires['dashscope'].url}/compatible-mode/v1",
                "QWENCLOUD_API_BASE_VIDEO": environment_wires["qwencloud"].url,
                "QWEN_AI_PLATFORM_API_BASE_VIDEO": environment_wires["qwen_ai_platform"].url,
                "DASHSCOPE_API_KEY": "env-dashscope-key",
                "QWENCLOUD_API_KEY": "env-qwencloud-key",
                "QWEN_AI_PLATFORM_API_KEY": "env-qwen-ai-platform-key",
            },
            config=path,
            remove_environment=_ROUTED_ENVIRONMENT,
        ) as owned,
    ):
        yield owned


@pytest.mark.timeout(_CHAOS_TIMEOUT)
@pytest.mark.parametrize(
    ("provider", "key"),
    [
        ("dashscope", "env-dashscope-key"),
        ("qwencloud", "env-qwencloud-key"),
        ("qwen_ai_platform", "env-qwen-ai-platform-key"),
    ],
)
def test_each_dashscope_prefix_reaches_its_own_environment_base_with_its_own_environment_key(
    environment_proxy: OwnedProxy, environment_wires: Mapping[str, Wire], provider: str, key: str
) -> None:
    wire: Final = environment_wires[provider]
    wire.drain()
    candidate: Final = environment_proxy.gateway
    video_id: Final = created_id(_create(candidate, {"model": f"env-{provider}", "seconds": "5", "size": "1280x720"}))
    assert _completed(candidate, video_id)
    download: Final = candidate.request("GET", f"/v1/videos/{video_id}/content")
    assert download.status_code == 200 and download.content == MP4, download.text
    received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [
        ("POST", CREATE_PATH),
        ("GET", f"{TASKS_PATH}{task_of(video_id)}"),
        ("GET", f"{TASKS_PATH}{task_of(video_id)}"),
        ("GET", f"{OSS_PATH}{task_of(video_id)}.mp4"),
    ]
    assert [request.headers.get("authorization") for request in received[:3]] == [f"Bearer {key}"] * 3
    assert all(
        other.drain() == () for name, other in environment_wires.items() if name != provider
    ), "A prefix reached another prefix's environment base"


@pytest.mark.timeout(_CHAOS_TIMEOUT)
def test_a_catch_all_over_another_provider_never_takes_a_poll_for_an_id_only_its_pattern_matches(
    environment_proxy: OwnedProxy, environment_wires: Mapping[str, Wire], catch_all_wire: Wire
) -> None:
    wire: Final = environment_wires["dashscope"]
    candidate: Final = environment_proxy.gateway
    task: Final = task_of(created_id(_create(candidate, {"model": "env-dashscope", "seconds": "5"})))
    polled: Final = forged_id("dashscope", MODEL, task, None)
    wire.drain()
    catch_all_wire.drain()
    assert _completed(candidate, polled)
    download: Final = candidate.request("GET", f"/v1/videos/{polled}/content")
    assert download.status_code == 200 and download.content == MP4, download.text
    assert catch_all_wire.drain() == ()
    received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [
        ("GET", f"{TASKS_PATH}{task}"),
        ("GET", f"{TASKS_PATH}{task}"),
        ("GET", f"{OSS_PATH}{task}.mp4"),
    ]
    assert [request.headers.get("authorization") for request in received[:2]] == ["Bearer env-dashscope-key"] * 2


@pytest.mark.timeout(_CHAOS_TIMEOUT)
def test_global_drop_params_drops_a_malformed_seconds_and_seed_instead_of_answering_400(
    environment_proxy: OwnedProxy, environment_wires: Mapping[str, Wire]
) -> None:
    wire: Final = environment_wires["dashscope"]
    wire.drain()
    created_id(
        _create(
            environment_proxy.gateway,
            {"model": "env-dashscope", "seconds": "10s", "size": "1280x720", "parameters": {"seed": "abc"}},
        )
    )
    (body,) = create_bodies(wire.drain())
    assert body == {
        "model": MODEL,
        "input": {"prompt": PROMPT},
        "parameters": {"ratio": "16:9", "resolution": "720P"},
    }


@pytest.mark.timeout(_CHAOS_TIMEOUT)
def test_the_video_base_beats_the_chat_base_for_videos_while_chat_keeps_the_chat_base(
    gateway: Gateway, tmp_path: Path
) -> None:
    def chat(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/compatible-mode/v1/chat/completions"), request.target
        return _CHAT_REPLY

    with wire_server(task_api()) as video, wire_server(chat) as chat_wire:
        path: Final = _config(
            tmp_path,
            "dashscope-video-base",
            (
                {"model_name": "env-video", "litellm_params": {"model": f"dashscope/{MODEL}"}},
                {"model_name": "env-chat", "litellm_params": {"model": "dashscope/qwen-plus"}},
            ),
            {},
            {},
        )
        with owned_proxy_process(
            gateway,
            tmp_path,
            {
                "DASHSCOPE_API_BASE_VIDEO": video.url,
                "DASHSCOPE_API_BASE": f"{chat_wire.url}/compatible-mode/v1",
                "DASHSCOPE_API_KEY": "env-dashscope-key",
            },
            config=path,
            remove_environment=_ROUTED_ENVIRONMENT,
        ) as owned:
            candidate: Final = owned.gateway
            video_id: Final = created_id(_create(candidate, {"model": "env-video", "seconds": "5"}))
            assert _completed(candidate, video_id)
            completion: Final = candidate.post(
                "/v1/chat/completions", {"model": "env-chat", "messages": [{"role": "user", "content": "kite?"}]}
            )
            assert completion["id"] == "chatcmpl-dashscope-env", completion
        assert [(request.method, request.target) for request in video.drain()] == [
            ("POST", CREATE_PATH),
            ("GET", f"{TASKS_PATH}{task_of(video_id)}"),
        ]
        (chat_request,) = chat_wire.drain()
        assert json.loads(chat_request.body)["model"] == "qwen-plus"
        assert chat_request.headers["authorization"] == "Bearer env-dashscope-key"
