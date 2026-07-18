from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

from locust import FastHttpUser, constant, events, task

_MODEL = os.environ["LOAD_MODEL"]
_HEADERS = {"Authorization": f"Bearer {os.environ['LOAD_API_KEY']}"}
_PAYLOAD = {
    "model": _MODEL,
    "messages": [{"role": "user", "content": "load test ping"}],
    "temperature": 0,
    "max_tokens": 16,
}
_FAILURE_REPORT_PATH = os.environ.get("LOAD_FAILURE_REPORT_PATH", "")

_failure_reasons: Counter[str] = Counter()


@events.request.add_listener
def _record_failure(  # pyright: ignore[reportUnusedFunction]  # locust event hook
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    response: Any,
    context: dict[str, object],
    exception: BaseException | None,
    start_time: float,
    url: str,
    **kwargs: object,
) -> None:
    if exception is not None:
        _failure_reasons[f"exception:{type(exception).__name__}"] += 1
        return
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and status >= 400:
        _failure_reasons[f"http:{status}"] += 1


@events.quitting.add_listener
def _write_failure_report(environment: object, **kwargs: object) -> None:  # pyright: ignore[reportUnusedFunction]
    if not _FAILURE_REPORT_PATH:
        return
    path = _FAILURE_REPORT_PATH
    payload = dict(_failure_reasons.most_common(20))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class ChatUser(FastHttpUser):
    wait_time = constant(0)
    connection_timeout = 10.0
    network_timeout = 60.0

    @task
    def chat(self) -> None:
        with self.client.post(  # pyright: ignore[reportUnknownMemberType]
            "/v1/chat/completions",
            json=_PAYLOAD,
            headers=_HEADERS,
            name="/v1/chat/completions",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code}")
                return
            response.success()
