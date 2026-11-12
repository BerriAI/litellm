from __future__ import annotations

import json
import time

from batch_client import BatchClient, BatchObject
from e2e_http import Success, UnknownApiError, unwrap
from pydantic import BaseModel

CREATED_BATCH_STATUSES = {"validating", "in_progress", "finalizing"}
BATCH_OP_RETRIES = 5


class BatchOutputBody(BaseModel):
    choices: list[object] = []


class BatchOutputResponse(BaseModel):
    status_code: int | None = None
    body: BatchOutputBody | None = None


class BatchOutputLine(BaseModel):
    custom_id: str | None = None
    response: BatchOutputResponse


def transient_status(status_code: int) -> bool:
    return status_code in {408, 429, 500, 502, 503, 504}


def backoff_seconds(attempt: int) -> float:
    delays: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 8.0)
    return delays[min(attempt, len(delays) - 1)]


def cancel_batch(
    client: BatchClient, batch_id: str, *, key: str, provider: str | None
) -> BatchObject:
    last = client.cancel_batch(batch_id, key=key, provider=provider)
    for attempt in range(BATCH_OP_RETRIES - 1):
        match last:
            case Success(data=data):
                return data
            case UnknownApiError(status_code=code) if transient_status(code):
                time.sleep(backoff_seconds(attempt))
                last = client.cancel_batch(batch_id, key=key, provider=provider)
            case _:
                break
    return unwrap(last)


def retrieve_batch(
    client: BatchClient, batch_id: str, *, key: str, provider: str | None
) -> BatchObject:
    last = client.retrieve_batch(batch_id, key=key, provider=provider)
    for attempt in range(BATCH_OP_RETRIES - 1):
        match last:
            case Success(data=data):
                return data
            case UnknownApiError(status_code=code) if transient_status(code):
                time.sleep(backoff_seconds(attempt))
                last = client.retrieve_batch(batch_id, key=key, provider=provider)
            case _:
                break
    return unwrap(last)


def render_jsonl(model: str) -> bytes:
    line = {
        "custom_id": "req-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
        },
    }
    return (json.dumps(line) + "\n").encode()


def assert_batch_object(batch: BatchObject) -> None:
    assert batch.object == "batch", f"batch.object={batch.object!r}"
    if batch.endpoint:
        assert (
            batch.endpoint == "/v1/chat/completions"
        ), f"batch.endpoint={batch.endpoint!r}"
    assert batch.completion_window == "24h", f"window={batch.completion_window!r}"
    assert batch.input_file_id, "batch.input_file_id missing"
    assert (
        batch.created_at is not None and batch.created_at > 0
    ), "batch.created_at missing"
