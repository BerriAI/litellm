import base64
import json
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from typing import Final

import httpx
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    held,
    list_value,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.wire import Reply, Request
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue

CREATE_PATH: Final = "/api/v1/services/aigc/video-generation/video-synthesis"
TASKS_PATH: Final = "/api/v1/tasks/"
OSS_PATH: Final = "/oss/"
API_KEY: Final = "synthetic-dashscope-key"
MODEL: Final = "wan3.0-video"
PROMPT: Final = "a red kite climbing over a grey sea at dawn"
MP4: Final = b"\x00\x00\x00\x18ftypmp42" + uuid.uuid4().bytes * 4
MP4_REPLY: Final = Reply(body=MP4, content_type="video/mp4")
SUBMITTED: Final = "2026-10-14 10:00:00.000"
FINISHED: Final = "2026-10-14 10:01:30.500"
SUBMITTED_EPOCH: Final = int(datetime(2026, 10, 14, 2, 0, 0, tzinfo=timezone.utc).timestamp())
FINISHED_EPOCH: Final = int(datetime(2026, 10, 14, 2, 1, 30, tzinfo=timezone.utc).timestamp())

TaskStatus = Callable[[str, str], Mapping[str, JsonValue]]


def json_reply(body: Mapping[str, JsonValue], status: int = 200) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode())


def pending(task_id: str, origin: str) -> Mapping[str, JsonValue]:
    return {"request_id": f"req-{task_id}", "output": {"task_id": task_id, "task_status": "PENDING"}}


def succeeded(task_id: str, origin: str) -> Mapping[str, JsonValue]:
    return {
        "request_id": f"req-{task_id}",
        "output": {
            "task_id": task_id,
            "task_status": "SUCCEEDED",
            "submit_time": SUBMITTED,
            "end_time": FINISHED,
            "video_url": f"{origin}{OSS_PATH}{task_id}.mp4",
        },
        "usage": {"duration": 5, "SR": 720, "ratio": "16:9", "video_count": 1},
    }


def unknown(task_id: str, origin: str) -> Mapping[str, JsonValue]:
    return {"request_id": f"req-{task_id}", "output": {"task_id": task_id, "task_status": "UNKNOWN"}}


def task_api(
    status: TaskStatus = succeeded, oss: Reply = MP4_REPLY, prefix: str = "task"
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target.startswith(OSS_PATH):
            return oss
        if request.method == "POST" and request.target == CREATE_PATH:
            return json_reply(pending(f"{prefix}-{uuid.uuid4().hex}", ""))
        assert request.method == "GET" and request.target.startswith(TASKS_PATH), (request.method, request.target)
        task_id: Final = request.target.removeprefix(TASKS_PATH)
        known: Final = task_id.startswith(f"{prefix}-")
        return json_reply((status if known else unknown)(task_id, f"http://{request.headers['host']}"))

    return respond


def polled_tasks(requests: tuple[Request, ...]) -> tuple[str, ...]:
    return tuple(
        request.target.removeprefix(TASKS_PATH)
        for request in requests
        if request.method == "GET" and request.target.startswith(TASKS_PATH)
    )


def create_bodies(requests: tuple[Request, ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(request.body)
        for request in requests
        if request.method == "POST" and request.target == CREATE_PATH
    )


def decoded(video_id: str) -> Mapping[str, str | None]:
    from litellm.types.videos.utils import decode_video_id_with_provider

    return {
        name: value if isinstance(value, str) else None
        for name, value in decode_video_id_with_provider(video_id).items()
    }


def task_of(video_id: str) -> str:
    task: Final = decoded(video_id).get("video_id")
    assert task is not None, video_id
    return task


def forged_id(provider: str, model: str, task_id: str, served: str | None) -> str:
    served_part: Final = f";served_model:{served}" if served else ""
    plain: Final = f"litellm:custom_llm_provider:{provider};model_id:{model};video_id:{task_id}{served_part}"
    return f"video_{base64.b64encode(plain.encode()).decode()}"


def rate(provider: str, tier_field: str, model: str = MODEL) -> float:
    import litellm

    value: Final = litellm.model_cost[f"{provider}/{model}"][tier_field]
    assert isinstance(value, float), (provider, model, tier_field, value)
    return value


def key_spend(key: str) -> float:
    rows: Final = read_rows(
        'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (sha256(key.encode()).hexdigest(),)
    )
    assert len(rows) == 1, rows
    return float(str(rows[0]["spend"]))


def settled_spend(key: str, expected: float) -> float:
    eventually(lambda: key_spend(key), lambda spend: abs(spend - expected) < 1e-9, seconds=70)
    return spend_stays(key, expected)


def spend_stays(key: str, expected: float) -> float:
    return held(lambda: key_spend(key), lambda spend: abs(spend - expected) < 1e-9, holding=4)


def spend_rows(key: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        read_rows(
            'SELECT request_id, call_type, spend, model_group, status FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
            (sha256(key.encode()).hexdigest(),),
        )
    )


def proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _listed_over_a_fresh_connection(gateway: Gateway, _: int) -> frozenset[str]:
    with httpx.Client(base_url=gateway.client.base_url, timeout=15, trust_env=False) as client:
        listed: Final = client.get("/v1/models", headers={"Authorization": f"Bearer {gateway.key}"})
    assert listed.status_code == 200, listed.text
    entries: Final = list_value(JSON_OBJECT.validate_json(listed.content)["data"])
    return frozenset(string_value(object_value(entry)["id"]) for entry in entries)


def _every_worker_lists(gateway: Gateway, model: str) -> bool:
    with ThreadPoolExecutor(max_workers=12) as pool:
        seen: Final = tuple(pool.map(lambda index: _listed_over_a_fresh_connection(gateway, index), range(24)))
    return all(model in listed for listed in seen)


def deployed(gateway: Gateway, scenario: Scenario, **parameters: JsonValue) -> str:
    model: Final = scenario.model(model_info=None, **parameters)
    eventually(lambda: _every_worker_lists(gateway, model), lambda listed: listed, seconds=90)
    return model


def openai_client(gateway: Gateway, key: str) -> OpenAI:
    return OpenAI(
        api_key=key,
        base_url=proxy_url(gateway) + "/v1",
        max_retries=0,
        http_client=httpx.Client(timeout=60, trust_env=False),
    )


def async_openai_client(gateway: Gateway, key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=key,
        base_url=proxy_url(gateway) + "/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(timeout=60, trust_env=False),
    )


def create(gateway: Gateway, body: Mapping[str, JsonValue], key: str | None = None) -> httpx.Response:
    return gateway.request("POST", "/v1/videos", {"prompt": PROMPT, **body}, key=key)


def created_id(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["id"])


def error_message(response: httpx.Response) -> str:
    return string_value(object_value(JSON_OBJECT.validate_json(response.content)["error"])["message"])
