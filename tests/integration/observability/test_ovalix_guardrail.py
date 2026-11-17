"""Ovalix guardrail driven through a real proxy.

The Ovalix tracker is the only guardrail double: an owned wire peer that answers the `/tracking/beta` routes
the way the live tracker does, including allowing a checkpoint call whose `actor` is empty without inspecting
it. The provider is a second owned peer. The proxy, Postgres and Redis run for real with two workers.
"""

import json
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from integration._support.client import Gateway, JsonValue, gateway_from_environment
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.shares_worker_state

API_KEY: Final = "synthetic-ovalix-key"
APP_ID: Final = "synthetic-ovalix-app"
OTHER_APP: Final = "Synthetic Other App"
PRE_CHECKPOINT: Final = "synthetic-pre-checkpoint"
POST_CHECKPOINT: Final = "synthetic-post-checkpoint"
KEYWORD: Final = "Project Falcon"
NAME: Final = "David"
DENIAL: Final = "synthetic ovalix denial"
REPLY: Final = "synthetic provider reply"
PINNED: Final = "ovalix-pinned"
ENDPOINTS: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")
CHECKPOINT: Final = "/tracking/beta/checkpoint"
OUTAGE_AFTER: Final = 6


class _Outage:
    """Takes the tracker double down once it has answered a set number of checkpoint calls."""

    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self._remaining: int | None = None

    def arm(self, answered: int) -> None:
        with self._lock:
            self._remaining = answered

    def clear(self) -> None:
        with self._lock:
            self._remaining = None

    def admits(self, path: str) -> bool:
        with self._lock:
            if self._remaining is None:
                return True
            if self._remaining == 0:
                return False
            if path == CHECKPOINT:
                self._remaining -= 1
            return True


OUTAGE: Final = _Outage()


def _checkpoint_verdict(body: dict[str, object]) -> dict[str, object]:
    data: Final = body["data"]
    content: Final = str(data.get("content", "")) if isinstance(data, dict) else ""
    if not body["actor"]:
        return {"action_type": "allow"}
    if KEYWORD in content:
        return {"action_type": "block", "modified_data": {"content": DENIAL}}
    if NAME in content:
        return {"action_type": "anonymize", "modified_data": {"content": content.replace(NAME, "{Name}")}}
    return {"action_type": "allow"}


def _tracker(request: Request) -> Reply:
    if request.headers.get("x-api-key") != API_KEY:
        return Reply(status=401, body=b'{"detail": "missing api key"}')
    path: Final = urlsplit(request.target).path
    if not OUTAGE.admits(path):
        return Reply(status=503, body=b'{"detail": "tracker unavailable"}')
    if path != CHECKPOINT:
        return Reply(status=404, body=b'{"detail": "not found"}')
    return Reply(body=json.dumps(_checkpoint_verdict(json.loads(request.body))).encode())


def _responses_reply(marker: str) -> dict[str, object]:
    return {
        "id": "resp_" + marker,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_" + marker,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": REPLY, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def _chat_reply(marker: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-" + marker,
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": REPLY}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _provider(request: Request) -> Reply:
    marker: Final = uuid.uuid4().hex
    is_responses: Final = urlsplit(request.target).path.endswith("/responses")
    return Reply(body=json.dumps(_responses_reply(marker) if is_responses else _chat_reply(marker)).encode())


def _alias() -> str:
    return f"[{OTHER_APP}] integration key {uuid.uuid4().hex}"


def _ovalix(name: str, tracker: str, **routing: str) -> dict[str, object]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "ovalix",
            "mode": "pre_call",
            "default_on": False,
            "tracker_api_base": tracker,
            "tracker_api_key": API_KEY,
            **routing,
        },
    }


@dataclass(frozen=True, slots=True)
class Keys:
    no_user: str
    user_id: str
    with_user_id: str
    email: str
    with_email: str


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    tracker: Wire
    provider: Wire
    model: str
    keys: Keys


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("ovalix")
    with (
        gateway_from_environment() as gateway,
        wire_server(_tracker) as tracker,
        wire_server(_provider) as provider,
    ):
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["litellm_settings"]["cache"] = False
        config["guardrails"] = [
            _ovalix(
                PINNED,
                tracker.url,
                application_id=APP_ID,
                pre_checkpoint_id=PRE_CHECKPOINT,
                post_checkpoint_id=POST_CHECKPOINT,
            ),
        ]
        path: Final = root / "ovalix.yaml"
        path.write_text(yaml.safe_dump(config))
        with (
            owned_proxy_process(gateway, root, {}, config=path, workers=2) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model="openai/gpt-4o-mini", api_base=provider.url + "/v1", api_key="synthetic-openai-key"
            )
            user_id: Final = scenario.user()
            email: Final = f"ovalix-{uuid.uuid4().hex}@example.com"
            keys: Final = Keys(
                no_user=scenario.key(key_alias=_alias()),
                user_id=user_id,
                with_user_id=scenario.key(key_alias=_alias(), user_id=user_id),
                email=email,
                with_email=scenario.key(key_alias=_alias(), user_id=scenario.user(user_email=email)),
            )
            yield Rig(owned.gateway, tracker, provider, model, keys)


def _post(rig: Rig, endpoint: str, key: str, text: str, extra: dict[str, JsonValue] | None = None) -> httpx.Response:
    prompt: Final[dict[str, JsonValue]] = (
        {"input": text}
        if endpoint == "/v1/responses"
        else {"max_tokens": 64, "messages": [{"role": "user", "content": text}]}
        if endpoint == "/v1/messages"
        else {"messages": [{"role": "user", "content": text}]}
    )
    return rig.proxy.request(
        "POST", endpoint, {"model": rig.model, "guardrails": [PINNED], **prompt, **(extra or {})}, key=key
    )


def _send(rig: Rig, endpoint: str, key: str, text: str, extra: dict[str, JsonValue] | None = None) -> httpx.Response:
    rig.tracker.drain()
    rig.provider.drain()
    return _post(rig, endpoint, key, text, extra)


def _checkpoint_actors(rig: Rig) -> list[object]:
    return [
        json.loads(request.body)["actor"]
        for request in rig.tracker.drain()
        if urlsplit(request.target).path == CHECKPOINT
    ]


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_a_prompt_from_a_key_with_no_user_is_inspected(rig: Rig, endpoint: str, stream: bool) -> None:
    response: Final = _send(
        rig, endpoint, rig.keys.no_user, f"Tell me about {KEYWORD} {uuid.uuid4()}", {"stream": stream}
    )

    assert response.status_code == 400 and DENIAL in response.text, response.text
    assert rig.provider.drain() == ()
    assert _checkpoint_actors(rig) == ["unknown"]


@pytest.mark.parametrize(
    ("endpoint", "client_metadata"),
    (("/v1/messages", {"user_id": "client-chosen"}), ("/v1/responses", {"client": "chosen"})),
)
def test_a_client_metadata_object_does_not_switch_inspection_off(
    rig: Rig, endpoint: str, client_metadata: dict[str, JsonValue]
) -> None:
    response: Final = _send(
        rig,
        endpoint,
        rig.keys.with_user_id,
        f"Tell me about {KEYWORD} {uuid.uuid4()}",
        {"metadata": client_metadata},
    )

    assert response.status_code == 400 and DENIAL in response.text, response.text
    assert rig.provider.drain() == ()
    assert _checkpoint_actors(rig) == ["unknown"]


def test_a_name_from_a_key_with_no_user_reaches_the_provider_anonymized(rig: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _send(rig, "/v1/chat/completions", rig.keys.no_user, f"My name is {NAME} {marker}")

    assert response.status_code == 200, response.text
    sent: Final = [json.loads(request.body)["messages"][-1]["content"] for request in rig.provider.drain()]
    assert sent == [f"My name is {{Name}} {marker}"]


def test_the_actor_is_the_key_users_email_then_their_id(rig: Rig) -> None:
    by_id: Final = _send(rig, "/v1/chat/completions", rig.keys.with_user_id, f"hello {uuid.uuid4()}")
    assert by_id.status_code == 200, by_id.text
    assert _checkpoint_actors(rig) == [rig.keys.user_id]

    by_email: Final = _send(rig, "/v1/chat/completions", rig.keys.with_email, f"hello {uuid.uuid4()}")
    assert by_email.status_code == 200, by_email.text
    assert _checkpoint_actors(rig) == [rig.keys.email]


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_a_key_alias_naming_another_app_does_not_change_the_application_checked(rig: Rig, endpoint: str) -> None:
    response: Final = _send(rig, endpoint, rig.keys.with_email, f"Tell me about {KEYWORD} {uuid.uuid4()}")

    assert response.status_code == 400 and DENIAL in response.text, response.text
    assert rig.provider.drain() == ()
    calls: Final = [(urlsplit(request.target).path, json.loads(request.body)) for request in rig.tracker.drain()]
    assert [
        (path, body["application_id"], body["checkpoint_id"], "application_name" in body) for path, body in calls
    ] == [(CHECKPOINT, APP_ID, PRE_CHECKPOINT, False)]


def _burst(rig: Rig) -> tuple[httpx.Response, ...]:
    plan: Final = tuple(product(ENDPOINTS, (False, True), range(4)))
    with ThreadPoolExecutor(max_workers=12) as pool:
        return tuple(
            pool.map(
                lambda cell: _post(
                    rig,
                    cell[0],
                    rig.keys.no_user,
                    f"Tell me about {KEYWORD} {uuid.uuid4()}",
                    {"stream": cell[1]},
                ),
                plan,
            )
        )


def test_a_tracker_outage_mid_burst_fails_closed_and_recovers(rig: Rig) -> None:
    rig.tracker.drain()
    rig.provider.drain()
    OUTAGE.arm(OUTAGE_AFTER)
    try:
        during: Final = _burst(rig)
    finally:
        OUTAGE.clear()

    denied: Final = [response for response in during if response.status_code == 400 and DENIAL in response.text]
    failed: Final = [response for response in during if response.status_code >= 400 and DENIAL not in response.text]
    assert len(denied) == OUTAGE_AFTER, [response.text for response in during]
    assert len(failed) == len(during) - OUTAGE_AFTER, [response.text for response in during]
    assert all("Ovalix guardrail error" in response.text for response in failed), [response.text for response in failed]
    assert rig.provider.drain() == ()
    assert set(_checkpoint_actors(rig)) == {"unknown"}

    after: Final = _burst(rig)

    assert all(response.status_code == 400 and DENIAL in response.text for response in after), [
        response.text for response in after
    ]
    assert rig.provider.drain() == ()
    assert _checkpoint_actors(rig) == ["unknown"] * len(after)
