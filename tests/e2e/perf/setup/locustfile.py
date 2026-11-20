"""The Locust side of a perf case: simulated users take the next request off the pre-generated plan,
send it at the planned rate, and write one record per request with what the client saw: status,
time to headers, time to first byte, time to first token (streams), total time, the text read back
and whether it matched. Runs in its own process (`setup/run.py` starts it), never in the gateway's.

Reads `PERF_PLAN_FILE` (JSON list of wire requests), writes `PERF_RECORDS_FILE` (one JSON object per
line), paces to `PERF_RPS` over `PERF_USERS` users (or, when the plan carries `send_at_s`, sends each
request at its scheduled second), and quits once the last request is recorded.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final, cast

import gevent
from locust import FastHttpUser, task
from locust.contrib.fasthttp import FastResponse, ResponseContextManager
from pydantic import TypeAdapter

from setup.wire import Endpoint, PlanProgress, WireRecord, WireRequest, request_body, response_id

_PLAN_ADAPTER: Final = TypeAdapter(list[WireRequest])
_JSON: Final = TypeAdapter[object](object)
_PLAN: Final = _PLAN_ADAPTER.validate_json(Path(os.environ["PERF_PLAN_FILE"]).read_bytes())
_RECORDS_PATH: Final = Path(os.environ["PERF_RECORDS_FILE"])
_RPS: Final = float(os.environ["PERF_RPS"])
_USERS: Final = int(os.environ["PERF_USERS"])
_NEXT: Final[Iterator[WireRequest]] = iter(_PLAN)
_INTERVAL_S: Final = _USERS / _RPS
_JITTER: Final = 0.25
_RNG: Final = random.Random(_PLAN[0].marker if _PLAN else "empty-plan")
_RECORDS: Final = _RECORDS_PATH.open("a", encoding="utf-8")
_PROGRESS: Final = PlanProgress(len(_PLAN))
_SCHEDULED: Final = bool(_PLAN) and _PLAN[0].send_at_s is not None
_RUN_STARTED: Final = time.monotonic()
_STOP_FILE: Final = os.environ.get("PERF_STOP_FILE")
BLOCK_REASON: Final = "perf-test-block"
_TOKEN_MAX_AGE_S: Final = 240.0
_tokens: dict[str, tuple[str, float]] = {}


def _as_dict(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def _first_choice_field(payload: dict[str, object], holder: str) -> str:
    choices: Final = _as_list(payload.get("choices"))
    if not choices:
        return ""
    content: Final = _as_dict(_as_dict(choices[0]).get(holder)).get("content")
    return content if isinstance(content, str) else ""


def text_of_json(endpoint: Endpoint, payload: dict[str, object]) -> str:
    """The assistant text in a non-streaming body of each endpoint."""
    if endpoint == "chat_completions":
        return _first_choice_field(payload, "message")
    if endpoint == "responses":
        parts: list[str] = []
        for item in _as_list(payload.get("output")):
            for part in _as_list(_as_dict(item).get("content")):
                text = _as_dict(part).get("text")
                if _as_dict(part).get("type") == "output_text" and isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    blocks: list[str] = []
    for block in _as_list(payload.get("content")):
        text = _as_dict(block).get("text")
        if isinstance(text, str):
            blocks.append(text)
    return "".join(blocks)


def delta_of_event(endpoint: Endpoint, payload: dict[str, object]) -> str:
    """The text delta one SSE event of each endpoint carries, empty when it carries none."""
    if endpoint == "chat_completions":
        return _first_choice_field(payload, "delta")
    if endpoint == "responses":
        delta: Final = payload.get("delta")
        return delta if payload.get("type") == "response.output_text.delta" and isinstance(delta, str) else ""
    if payload.get("type") == "content_block_delta":
        text: Final = _as_dict(payload.get("delta")).get("text")
        return text if isinstance(text, str) else ""
    return ""


_RESPONSES_TOKEN_EVENTS: Final = frozenset(
    {
        "response.output_text.delta",
        "response.refusal.delta",
        "response.reasoning_text.delta",
        "response.reasoning_summary_text.delta",
        "response.function_call_arguments.delta",
    }
)


def token_in_event(endpoint: Endpoint, payload: dict[str, object]) -> bool:
    """Whether one SSE event carries a generated token of any kind: visible text, reasoning or tool-call arguments.
    The first such event is the stream's first token. `delta_of_event` stays visible text only because the body
    check compares it with the text the mock generated, and a tool-call reply has none."""
    if delta_of_event(endpoint, payload):
        return True
    if endpoint == "chat_completions":
        choices: Final = _as_list(payload.get("choices"))
        delta: Final = _as_dict(_as_dict(choices[0]).get("delta")) if choices else {}
        return any(delta.get(key) for key in ("reasoning_content", "reasoning", "tool_calls", "function_call"))
    if endpoint == "responses":
        return payload.get("type") in _RESPONSES_TOKEN_EVENTS
    return payload.get("type") == "content_block_delta"


def _events(buffer: bytes) -> tuple[list[dict[str, object]], bytes]:
    """Complete SSE events in `buffer` as their JSON data, plus the unfinished tail."""
    events: list[dict[str, object]] = []
    while b"\n\n" in buffer:
        raw, buffer = buffer.split(b"\n\n", 1)
        for line in raw.split(b"\n"):
            if line.startswith(b"data:"):
                data = line[5:].strip()
                if data and data != b"[DONE]":
                    events.append(_as_dict(_JSON.validate_json(data)))
    return events, buffer


def _header(response: FastResponse, name: str) -> str | None:
    headers: Final = response.headers
    if headers is None:
        return None
    value: object = headers.get(name)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return value if isinstance(value, str) else None


def _ms_header(response: FastResponse | None, name: str) -> float | None:
    """A gateway timing header as a number; the gateway writes the literal `None` when it has no value."""
    value: Final = _header(response, name) if response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _quit(user: FastHttpUser) -> None:
    """End the run once every record is written. Quitting from a separate greenlet: `runner.quit`
    kills the user greenlets, and the user calling it would be killed mid-shutdown."""
    runner = user.environment.runner  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if runner is not None:
        gevent.spawn(runner.quit)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnusedCallResult]


def _judge(item: WireRequest, status: int, body_text: str) -> tuple[bool, str | None]:
    if item.blocked:
        return (
            status == 400 and BLOCK_REASON in body_text,
            None if status == 400 else f"expected a 400 block, got {status}",
        )
    if status != 200:
        return False, f"status {status}: {body_text[:200]}"
    return True, None


def _record(record: WireRecord) -> None:
    _ = _RECORDS.write(record.model_dump_json() + "\n")
    _RECORDS.flush()
    _PROGRESS.completed += 1


def _authorization(user: FastHttpUser, item: WireRequest) -> str:
    """The bearer for this request: the key as given, or the caller's JWT, re-minted from Keycloak
    once it is older than the realm's access token lifespan allows."""
    if item.token_source is None:
        assert item.authorization is not None, f"request {item.seq} has neither a key nor a token source"
        return item.authorization
    source: Final = item.token_source
    cached: Final = _tokens.get(source.username)
    if cached is not None and time.monotonic() - cached[1] < _TOKEN_MAX_AGE_S:
        return f"Bearer {cached[0]}"
    minted = ""
    with user.client.request(  # pyright: ignore[reportUnknownMemberType]
        "POST",
        source.token_url,
        data={
            "grant_type": "password",
            "client_id": source.client_id,
            "username": source.username,
            "password": source.password,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        catch_response=True,
        name="keycloak token",
    ) as response:
        payload: Final = _as_dict(_JSON.validate_json(response.text or "{}"))
        token: Final = payload.get("access_token")
        if not isinstance(token, str):
            response.failure(f"no access_token in {response.status_code}")  # pyright: ignore[reportUnknownMemberType]
            raise RuntimeError(
                f"could not mint a token for {source.username}: {response.status_code} {response.text!r}"
            )
        response.success()
        minted = token
    _tokens[source.username] = (minted, time.monotonic())
    return f"Bearer {minted}"


class PlanUser(FastHttpUser):
    """One simulated client; all users share the plan and together hold the planned rate. Each user starts at
    a random point of its first interval and jitters every later interval by up to `_JITTER` either way, so the
    users never fire in lockstep; the mean interval is unchanged and `rps` stays the total rate. A scheduled
    plan (a ramp) is paced by the plan instead: a user takes the next request and sleeps until its second."""

    network_timeout = float(os.environ.get("PERF_REQUEST_TIMEOUT", "120"))
    connection_timeout = 30.0

    def on_start(self) -> None:
        self._last_run = 0.0
        self._last_wait = 0.0
        if not _SCHEDULED:
            gevent.sleep(_RNG.uniform(0.0, _INTERVAL_S))

    def wait_time(self) -> float:
        if _SCHEDULED:
            return 0.1
        target: Final = _INTERVAL_S * _RNG.uniform(1.0 - _JITTER, 1.0 + _JITTER)
        now: Final = time.monotonic()
        run_s: Final = now - self._last_run - self._last_wait if self._last_run else 0.0
        self._last_wait = max(0.0, target - run_s)
        self._last_run = now
        return self._last_wait

    @task
    def send(self) -> None:
        take: Final = _PROGRESS.take(stop=_STOP_FILE is not None and Path(_STOP_FILE).exists())
        item: Final = next(_NEXT, None) if take else None
        if item is None:
            if _PROGRESS.drained:
                _quit(self)
            return
        if item.send_at_s is not None:
            gevent.sleep(max(0.0, _RUN_STARTED + item.send_at_s - time.monotonic()))
        started_at: Final = time.time()
        started: Final = time.perf_counter()
        first_byte_s: float | None = None
        first_token_s: float | None = None
        text = ""
        body_text = ""
        status = 0
        ok = False
        error: str | None = None
        header_s = 0.0
        response: FastResponse | None = None
        identifier: str | None = None  # rebind-ok: the ID arrives in a JSON body or the first SSE event
        try:
            headers: Final = {"Authorization": _authorization(self, item), "Content-Type": "application/json"}
            context: ResponseContextManager
            with self.client.request(  # pyright: ignore[reportUnknownMemberType]
                "POST",
                item.path,
                json=request_body(item),
                headers=headers,
                stream=item.stream,
                catch_response=True,
                name=f"{item.endpoint}{' stream' if item.stream else ''}",
            ) as context:
                response = context
                header_s = time.perf_counter() - started
                status = int(context.status_code)
                if item.stream and status == 200:
                    buffer = b""
                    raw = context.stream  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the socket-backed response
                    while True:
                        line = cast(bytes, raw.readline(b"\n"))  # pyright: ignore[reportUnknownMemberType]  # returns as soon as a line is complete
                        if not line:
                            break
                        now = time.perf_counter() - started
                        if first_byte_s is None:
                            first_byte_s = now
                        buffer += line
                        events, buffer = _events(buffer)
                        for event in events:
                            identifier = identifier or response_id(event)
                            if first_token_s is None and token_in_event(item.endpoint, event):
                                first_token_s = now
                            text += delta_of_event(item.endpoint, event)
                else:
                    body_text = context.text or ""
                    first_byte_s = time.perf_counter() - started
                    if status == 200:
                        payload: Final = _as_dict(_JSON.validate_json(body_text))
                        identifier = response_id(payload)
                        text = text_of_json(item.endpoint, payload)
                ok, error = _judge(item, status, body_text)
                if ok:
                    context.success()
                else:
                    context.failure(error or "failed")  # pyright: ignore[reportUnknownMemberType]
        except Exception as exc:  # noqa: BLE001  # anything the client raised is the record's error
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        total_s: Final = time.perf_counter() - started
        delay_header: Final = _header(response, "llm_provider-x-mock-delay-s") if response is not None else None
        _record(
            WireRecord(
                seq=item.seq,
                marker=item.marker,
                caller=item.caller,
                endpoint=item.endpoint,
                stream=item.stream,
                blocked=item.blocked,
                started_at=started_at,
                status=status,
                header_s=header_s,
                first_byte_s=first_byte_s,
                first_token_s=first_token_s,
                total_s=total_s,
                text=text,
                ok=ok,
                mock_delay_s=float(delay_header) if delay_header else None,
                call_id=_header(response, "x-litellm-call-id") if response is not None else None,
                error=error,
                response_id=identifier,
                litellm_overhead_ms=_ms_header(response, "x-litellm-overhead-duration-ms"),
                litellm_response_ms=_ms_header(response, "x-litellm-response-duration-ms"),
            )
        )
        if _PROGRESS.drained:
            _quit(self)
