import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._dashscope_video_support import (
    API_KEY,
    MODEL,
    MP4,
    MP4_REPLY,
    OSS_PATH,
    PROMPT,
    create,
    create_bodies,
    created_id,
    decoded,
    deployed,
    error_message,
    forged_id,
    json_reply,
    polled_tasks,
    proxy_url,
    rate,
    settled_spend,
    task_api,
    task_of,
)
from pydantic import JsonValue

pytestmark: Final = pytest.mark.xdist_group("video-poll-routing")

_OPENAI_VIDEOS: Final = "/v1/videos"
_BURST: Final = 12
_OPENAI_KEY: Final = "synthetic-openai-key"


def _openai_videos(prefix: str) -> Callable[[Request], Reply]:
    def video(video_id: str, status: str) -> Mapping[str, JsonValue]:
        return {
            "id": video_id,
            "object": "video",
            "model": "sora-2",
            "status": status,
            "progress": 100 if status == "completed" else 0,
            "created_at": 1791943200,
            "seconds": "4",
            "size": "720x1280",
        }

    def respond(request: Request) -> Reply:
        if request.method == "POST":
            assert request.target == _OPENAI_VIDEOS, request.target
            return json_reply(video(f"video_{prefix}{uuid.uuid4().hex}", "queued"))
        assert request.method == "GET" and request.target.startswith(f"{_OPENAI_VIDEOS}/"), request.target
        video_id: Final = request.target.removeprefix(f"{_OPENAI_VIDEOS}/").removesuffix("/content")
        if not video_id.startswith(f"video_{prefix}"):
            return json_reply(
                {"error": {"message": f"Video {video_id} not found", "type": "invalid_request_error"}}, 404
            )
        if request.target.endswith("/content"):
            return MP4_REPLY
        return json_reply(video(video_id, "completed"))

    return respond


def _outage_switch(outage: threading.Event, respond: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def switched(request: Request) -> Reply:
        return Reply(drop_connection=True) if outage.is_set() else respond(request)

    return switched


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    scenario: Scenario
    a: Wire
    b: Wire
    b_outage: threading.Event
    first: Wire
    second: Wire
    member_a: str
    member_b: str
    group: str
    alias: str
    openai_first: str
    openai_second: str
    openai_alias: str

    def drain(self) -> None:
        for wire in (self.a, self.b, self.first, self.second):
            wire.drain()


def _fresh_status(gateway: Gateway, body: Mapping[str, JsonValue]) -> int:
    with httpx.Client(base_url=proxy_url(gateway), timeout=30, trust_env=False) as client:
        return client.post(
            "/v1/videos", json={"prompt": PROMPT, **body}, headers={"Authorization": f"Bearer {gateway.key}"}
        ).status_code


def _burst(gateway: Gateway, body: Mapping[str, JsonValue]) -> frozenset[int]:
    with ThreadPoolExecutor(max_workers=_BURST) as pool:
        return frozenset(pool.map(lambda _: _fresh_status(gateway, body), range(_BURST)))


def _settled(gateway: Gateway, body: Mapping[str, JsonValue]) -> None:
    eventually(
        lambda: _burst(gateway, body) | _burst(gateway, body), lambda seen: seen == frozenset({200}), seconds=90
    )


def _current(gateway: Gateway, field: str) -> JsonValue:
    return object_value(gateway.get("/router/settings")["current_values"]).get(field)


def _router_settings(gateway: Gateway, settings: Mapping[str, JsonValue]) -> None:
    gateway.post("/config/update", {"router_settings": dict(settings)})


def _without_ours(gateway: Gateway, group: str, aliases: frozenset[str]) -> Mapping[str, JsonValue]:
    groups: Final = list_value(_current(gateway, "routing_groups") or [])
    current_aliases: Final = object_value(_current(gateway, "model_group_alias") or {})
    return {
        "routing_groups": [entry for entry in groups if object_value(entry)["group_name"] != group],
        "model_group_alias": {name: target for name, target in current_aliases.items() if name not in aliases},
    }


@pytest.fixture(scope="module")
def rig() -> Iterator[_Rig]:
    suffix: Final = uuid.uuid4().hex[:10]
    b_outage: Final = threading.Event()
    with (
        gateway_from_environment() as gateway,
        wire_server(task_api(prefix="a")) as a,
        wire_server(_outage_switch(b_outage, task_api(prefix="b"))) as b,
        wire_server(_openai_videos("first")) as first,
        wire_server(_openai_videos("second")) as second,
        gateway.scenario() as scenario,
        ExitStack() as restore,
    ):
        member_a: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=a.url, api_key=API_KEY)
        member_b: Final = deployed(gateway, scenario, model=f"dashscope/{MODEL}", api_base=b.url, api_key=API_KEY)
        openai_first: Final = deployed(
            gateway, scenario, model=f"openai/sora-{suffix}-first", api_base=f"{first.url}/v1", api_key=_OPENAI_KEY
        )
        openai_second: Final = deployed(
            gateway, scenario, model=f"openai/sora-{suffix}-second", api_base=f"{second.url}/v1", api_key=_OPENAI_KEY
        )
        built: Final = _Rig(
            gateway=gateway,
            scenario=scenario,
            a=a,
            b=b,
            b_outage=b_outage,
            first=first,
            second=second,
            member_a=member_a,
            member_b=member_b,
            group=f"video-group-{suffix}",
            alias=f"video-alias-{suffix}",
            openai_first=openai_first,
            openai_second=openai_second,
            openai_alias=f"openai-video-alias-{suffix}",
        )
        aliases: Final = frozenset({built.alias, built.openai_alias})
        kept: Final = _without_ours(gateway, built.group, aliases)
        restore.callback(lambda: _router_settings(gateway, _without_ours(gateway, built.group, aliases)))
        _router_settings(
            gateway,
            {
                "routing_groups": [
                    *list_value(kept["routing_groups"]),
                    {"group_name": built.group, "models": [member_a, member_b], "routing_strategy": "simple-shuffle"},
                ],
                "model_group_alias": {
                    **object_value(kept["model_group_alias"]),
                    built.alias: member_b,
                    built.openai_alias: openai_first,
                },
            },
        )
        _settled(gateway, {"model": built.openai_alias, "seconds": "4"})
        built.drain()
        yield built


def _poll(rig: _Rig, video_id: str, key: str | None = None) -> httpx.Response:
    return rig.gateway.request("GET", f"/v1/videos/{video_id}", key=key)


def _download(rig: _Rig, video_id: str, key: str | None = None) -> httpx.Response:
    return rig.gateway.request("GET", f"/v1/videos/{video_id}/content", key=key)


def _assert_served(response: httpx.Response, video_id: str) -> None:
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert (body["id"], body["status"]) == (video_id, "completed"), body


def test_a_key_scoped_to_an_alias_polls_and_downloads_from_the_aliased_deployment(rig: _Rig) -> None:
    key: Final = rig.scenario.key(models=[rig.alias])
    rig.drain()
    video_id: Final = created_id(create(rig.gateway, {"model": rig.alias, "seconds": "5", "size": "1280x720"}, key))
    assert {name: decoded(video_id).get(name) for name in ("model_id", "served_model")} == {
        "model_id": rig.alias,
        "served_model": rig.member_b,
    }
    _assert_served(_poll(rig, video_id, key), video_id)
    download: Final = _download(rig, video_id, key)
    assert download.status_code == 200, download.text
    assert download.content == MP4
    assert rig.a.drain() == ()
    task: Final = task_of(video_id)
    assert polled_tasks(rig.b.drain()) == (task, task)


def test_a_routing_group_sends_every_poll_and_download_back_to_the_member_that_created_the_task(rig: _Rig) -> None:
    key: Final = rig.scenario.key(models=[rig.group])

    def create_one(_index: int) -> str:
        return created_id(create(rig.gateway, {"model": rig.group, "seconds": "5", "size": "1280x720"}, key))

    video_ids: Final = eventually(
        lambda: tuple(create_one(index) for index in range(_BURST)),
        lambda ids: {task_of(video_id)[0] for video_id in ids} == {"a", "b"},
        seconds=60,
    )
    served: Final = {video_id: decoded(video_id) for video_id in video_ids}
    assert all(entry["model_id"] == rig.group for entry in served.values()), served
    assert all(
        entry["served_model"] == (rig.member_a if task_of(video_id).startswith("a-") else rig.member_b)
        for video_id, entry in served.items()
    ), served
    rig.drain()
    polls: Final = tuple((video_id, _poll(rig, video_id, key), _download(rig, video_id, key)) for video_id in video_ids)
    for video_id, status, download in polls:
        _assert_served(status, video_id)
        assert download.status_code == 200 and download.content == MP4, download.text
    a_tasks: Final = tuple(task_of(video_id) for video_id in video_ids if task_of(video_id).startswith("a-"))
    b_tasks: Final = tuple(task_of(video_id) for video_id in video_ids if task_of(video_id).startswith("b-"))
    assert sorted(polled_tasks(rig.a.drain())) == sorted((*a_tasks, *a_tasks))
    assert sorted(polled_tasks(rig.b.drain())) == sorted((*b_tasks, *b_tasks))


def test_a_key_scoped_to_a_wildcard_route_gets_an_id_without_a_served_group_and_polls_through_the_wildcard(
    rig: _Rig,
) -> None:
    prefix: Final = f"video-wild-{uuid.uuid4().hex[:8]}"
    created: Final = rig.gateway.post(
        "/model/new",
        {
            "model_name": f"{prefix}/*",
            "litellm_params": {"model": "dashscope/*", "api_base": rig.a.url, "api_key": API_KEY},
            "model_info": {},
        },
    )
    rig.scenario.cleanups.callback(rig.scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    requested: Final = f"{prefix}/{MODEL}"
    _settled(rig.gateway, {"model": requested, "seconds": "4"})
    rig.drain()
    key: Final = rig.scenario.key(models=[f"{prefix}/*"])
    video_id: Final = created_id(create(rig.gateway, {"model": requested, "seconds": "5", "size": "1280x720"}, key))
    assert decoded(video_id).get("model_id") == requested
    assert decoded(video_id).get("served_model") is None
    _assert_served(_poll(rig, video_id, key), video_id)
    received: Final = rig.a.drain()
    assert JSON_OBJECT.validate_json(received[0].body)["model"] == MODEL
    assert polled_tasks(received) == (task_of(video_id),)


@pytest.mark.parametrize("level", ["key", "team"])
def test_an_alias_to_one_member_keeps_a_forged_served_member_from_moving_the_poll(rig: _Rig, level: str) -> None:
    task: Final = f"a-{uuid.uuid4().hex}"
    forged: Final = forged_id("dashscope", rig.group, task, rig.member_a)
    team: Final = (
        rig.scenario.team(models=[rig.group, rig.member_b], model_aliases={rig.group: rig.member_b})
        if level == "team"
        else None
    )
    key: Final = (
        rig.scenario.key(team_id=team)
        if team is not None
        else rig.scenario.key(models=[rig.group, rig.member_b], aliases={rig.group: rig.member_b})
    )
    rig.drain()
    status: Final = _poll(rig, forged, key)
    download: Final = _download(rig, forged, key)
    for response in (status, download):
        assert response.status_code == 400, response.text
        assert "Task not found" in error_message(response), response.text
    assert rig.a.drain() == ()
    assert polled_tasks(rig.b.drain()) == (task, task)


def test_a_served_group_the_ids_own_model_does_not_route_to_is_ignored(rig: _Rig) -> None:
    video_id: Final = created_id(create(rig.gateway, {"model": rig.member_a, "seconds": "5", "size": "1280x720"}))
    task: Final = task_of(video_id)
    forged: Final = forged_id("dashscope", rig.member_a, task, rig.member_b)
    rig.drain()
    _assert_served(_poll(rig, forged), video_id)
    assert rig.b.drain() == ()
    assert polled_tasks(rig.a.drain()) == (task,)


def test_an_id_without_a_served_group_still_polls_its_own_deployment(rig: _Rig) -> None:
    video_id: Final = created_id(create(rig.gateway, {"model": rig.member_a, "seconds": "5", "size": "1280x720"}))
    assert decoded(video_id).get("served_model") is None
    legacy: Final = forged_id("dashscope", rig.member_a, task_of(video_id), None)
    assert legacy == video_id
    rig.drain()
    _assert_served(_poll(rig, legacy), legacy)
    download: Final = _download(rig, legacy)
    assert download.status_code == 200 and download.content == MP4, download.text
    received: Final = rig.a.drain()
    assert polled_tasks(received) == (task_of(video_id), task_of(video_id))
    assert [request.target for request in received if request.target.startswith(OSS_PATH)] == [
        f"{OSS_PATH}{task_of(video_id)}.mp4"
    ]


def test_a_forged_served_group_on_an_openai_id_never_reaches_the_other_deployment(rig: _Rig) -> None:
    video_id: Final = created_id(create(rig.gateway, {"model": rig.openai_first, "seconds": "4"}))
    original: Final = decoded(video_id)
    forged: Final = forged_id(
        "openai", string_value(original["model_id"] or ""), string_value(original["video_id"] or ""), rig.openai_second
    )
    rig.drain()
    status: Final = _poll(rig, forged)
    assert status.status_code == 200, status.text
    assert JSON_OBJECT.validate_json(status.content)["status"] == "completed"
    assert rig.second.drain() == ()
    assert [(request.method, request.target) for request in rig.first.drain()] == [
        ("GET", f"{_OPENAI_VIDEOS}/{original['video_id']}")
    ]


def test_an_openai_video_created_through_a_narrow_wildcard_polls_through_the_wildcard(rig: _Rig) -> None:
    prefix: Final = f"sora-{uuid.uuid4().hex[:8]}"
    created: Final = rig.gateway.post(
        "/model/new",
        {
            "model_name": f"{prefix}-*",
            "litellm_params": {
                "model": f"openai/{prefix}-*",
                "api_base": f"{rig.first.url}/v1",
                "api_key": _OPENAI_KEY,
            },
            "model_info": {},
        },
    )
    rig.scenario.cleanups.callback(rig.scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    requested: Final = f"{prefix}-pro"
    _settled(rig.gateway, {"model": requested, "seconds": "4"})
    rig.drain()
    video_id: Final = created_id(create(rig.gateway, {"model": requested, "seconds": "4"}))
    status: Final = _poll(rig, video_id)
    assert status.status_code == 200, status.text
    assert JSON_OBJECT.validate_json(status.content)["status"] == "completed"
    content: Final = _download(rig, video_id)
    assert content.status_code == 200 and content.content == MP4, content.text
    original: Final = decoded(video_id)["video_id"]
    assert [(request.method, request.target) for request in rig.first.drain()] == [
        ("POST", _OPENAI_VIDEOS),
        ("GET", f"{_OPENAI_VIDEOS}/{original}"),
        ("GET", f"{_OPENAI_VIDEOS}/{original}/content"),
    ]


def test_an_openai_video_created_through_an_alias_polls_and_downloads_from_the_alias_target(rig: _Rig) -> None:
    key: Final = rig.scenario.key(models=[rig.openai_alias])
    rig.drain()
    video_id: Final = created_id(create(rig.gateway, {"model": rig.openai_alias, "seconds": "4"}, key))
    status: Final = _poll(rig, video_id, key)
    assert status.status_code == 200, status.text
    content: Final = _download(rig, video_id, key)
    assert content.status_code == 200 and content.content == MP4, content.text
    original: Final = decoded(video_id)["video_id"]
    assert rig.second.drain() == ()
    assert [(request.method, request.target) for request in rig.first.drain()] == [
        ("POST", _OPENAI_VIDEOS),
        ("GET", f"{_OPENAI_VIDEOS}/{original}"),
        ("GET", f"{_OPENAI_VIDEOS}/{original}/content"),
    ]


def test_a_member_outage_mid_burst_fails_cleanly_and_every_served_task_polls_on_its_creating_member(
    rig: _Rig,
) -> None:
    key: Final = rig.scenario.key(models=[rig.group])
    body: Final = {"model": rig.group, "seconds": "5", "size": "1280x720"}
    rig.drain()
    rig.b_outage.set()
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            during: Final = tuple(pool.map(lambda _: create(rig.gateway, body, key), range(16)))
    finally:
        rig.b_outage.clear()
    failed: Final = tuple(response for response in during if response.status_code != 200)
    assert all(error_message(response) for response in failed), [response.text for response in failed]
    served_during: Final = tuple(created_id(response) for response in during if response.status_code == 200)
    assert all(task_of(video_id).startswith("a-") for video_id in served_during), served_during
    assert len(create_bodies(rig.a.drain())) == len(served_during)
    assert create_bodies(rig.b.drain()) != ()
    with ThreadPoolExecutor(max_workers=12) as pool:
        recovered: Final = tuple(pool.map(lambda _: created_id(create(rig.gateway, body, key)), range(24)))
    assert {task_of(video_id)[0] for video_id in recovered} == {"a", "b"}
    video_ids: Final = (*served_during, *recovered)
    rig.drain()
    with ThreadPoolExecutor(max_workers=12) as pool:
        polls: Final = tuple(pool.map(lambda video_id: (video_id, _poll(rig, video_id, key)), video_ids))
    for video_id, response in polls:
        _assert_served(response, video_id)
    a_tasks: Final = sorted(task_of(video_id) for video_id in video_ids if task_of(video_id).startswith("a-"))
    b_tasks: Final = sorted(task_of(video_id) for video_id in video_ids if task_of(video_id).startswith("b-"))
    assert sorted(polled_tasks(rig.a.drain())) == a_tasks
    assert sorted(polled_tasks(rig.b.drain())) == b_tasks
    expected: Final = len(video_ids) * 5 * rate("dashscope", "output_cost_per_second_720p")
    settled_spend(key, expected)
