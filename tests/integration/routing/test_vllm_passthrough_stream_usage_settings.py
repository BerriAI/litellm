import asyncio
import json
import threading
import uuid
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.anthropic_sse import user_prompt
from integration._support.client import JSON_OBJECT, eventually, gateway_from_environment, held, object_value
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 120), pytest.mark.shares_worker_state]

_PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_SERVED_MODEL: Final = "Qwen/Qwen2.5-0.5B-Instruct"
_INPUT_COST: Final = 0.001
_OUTPUT_COST: Final = 0.002
_USAGE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
)
_STARTED_WORKER: Final = "Started server process ["
_STARTUP_COMPLETE: Final = "Application startup complete."
_BURST: Final = 20


@dataclass(frozen=True, slots=True)
class _Rig:
    wire: Wire
    proxy: OwnedProxy
    model: str

    @property
    def base_url(self) -> str:
        return str(self.proxy.gateway.client.base_url).rstrip("/")

    @property
    def key(self) -> str:
        return self.proxy.gateway.key


@dataclass(frozen=True, slots=True)
class _Outcome:
    identity: str
    status: int
    text: str


def _identity(body: Mapping[str, JsonValue]) -> str:
    return f"cmpl-{user_prompt(body)}"


def _frame(payload: Mapping[str, JsonValue] | str) -> bytes:
    return b"data: " + (payload if isinstance(payload, str) else json.dumps(payload)).encode() + b"\n\n"


def _chunk(identity: str, choices: Sequence[JsonValue], **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _SERVED_MODEL,
        "choices": list(choices),
        **extra,
    }


def _usage_frame(identity: str) -> bytes:
    return _frame(_chunk(identity, [], usage=dict(_USAGE)))


def _asks_usage(body: Mapping[str, JsonValue]) -> bool:
    options: Final = body.get("stream_options")
    return isinstance(options, dict) and options.get("include_usage") is True


def _stream_bytes(body: Mapping[str, JsonValue]) -> tuple[bytes, ...]:
    identity: Final = _identity(body)
    return (
        _frame(_chunk(identity, [{"index": 0, "delta": {"content": "Hello there"}, "finish_reason": None}])),
        _frame(_chunk(identity, [{"index": 0, "delta": {}, "finish_reason": "stop"}])),
        *((_usage_frame(identity),) if _asks_usage(body) else ()),
        _frame("[DONE]"),
    )


def _streaming(request: Request) -> Reply:
    return Reply(content_type="text/event-stream", chunks=_stream_bytes(JSON_OBJECT.validate_json(request.body)))


def _config(directory: Path, wire: Wire, model: str, general_settings: Mapping[str, JsonValue]) -> Path:
    base: Final = object_value(yaml.safe_load(_PROXY_CONFIG.read_text()))
    deployment: Final = {
        "model_name": model,
        "litellm_params": {
            "model": f"hosted_vllm/{_SERVED_MODEL}",
            "api_base": f"{wire.url}/v1",
            "input_cost_per_token": _INPUT_COST,
            "output_cost_per_token": _OUTPUT_COST,
        },
    }
    config: Final = {
        **base,
        "model_list": [*(base.get("model_list") or []), deployment],
        "general_settings": {**object_value(base["general_settings"]), **general_settings},
    }
    path: Final = directory / f"vllm-usage-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _vllm_proxy(
    directory: Path, respond: Callable[[Request], Reply], general_settings: Mapping[str, JsonValue]
) -> Generator[_Rig, None, None]:
    model: Final = f"vllm-usage-{uuid.uuid4().hex}"
    with gateway_from_environment() as gateway, wire_server(answering_model_discovery(respond)) as wire:
        config: Final = _config(directory, wire, model, general_settings)
        with owned_proxy_process(gateway, directory, {}, config=config, workers=2) as owned:
            yield _Rig(wire, owned, model)


def _chat_body(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": f"vllm-usage-{uuid.uuid4().hex}"}], **extra}


def _rows(model: str, count: int) -> list[dict[str, JsonValue]]:
    return held(
        lambda: read_rows(
            'SELECT request_id, status, prompt_tokens, completion_tokens, spend FROM "LiteLLM_SpendLogs" '
            "WHERE model_group=%s",
            (model,),
        ),
        lambda rows: len(rows) >= count,
        holding=2,
        seconds=70,
    )


def _posted(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    return tuple(JSON_OBJECT.validate_json(request.body) for request in wire.drain() if request.method == "POST")


def _spend(prompt: JsonValue, completion: JsonValue) -> float:
    assert isinstance(prompt, int) and isinstance(completion, int), (prompt, completion)
    return prompt * _INPUT_COST + completion * _OUTPUT_COST


@pytest.mark.parametrize("setting", [True, None], ids=["true", "null"])
def test_always_include_stream_usage_true_or_null_injects_usage_and_relays_the_stream_the_caller_asked_for(
    tmp_path: Path, setting: bool | None
) -> None:
    with _vllm_proxy(tmp_path, _streaming, {"always_include_stream_usage": setting}) as rig:
        body: Final = _chat_body(rig.model, stream=True)
        response: Final = rig.proxy.gateway.request("POST", "/vllm/chat/completions", body)
        assert response.status_code == 200, response.text
        (posted,) = _posted(rig.wire)
        assert posted["stream_options"] == {"include_usage": True}, posted
        assert response.content == b"".join(_stream_bytes(body)), response.text
        (row,) = _rows(rig.model, 1)
        assert row["request_id"] == _identity(body), row
        assert (row["status"], row["prompt_tokens"], row["completion_tokens"]) == ("success", 5, 3), row
        assert row["spend"] == pytest.approx(_spend(5, 3)), row


def test_always_include_stream_usage_false_forwards_the_body_untouched_and_counts_locally(tmp_path: Path) -> None:
    with _vllm_proxy(tmp_path, _streaming, {"always_include_stream_usage": False}) as rig:
        body: Final = _chat_body(rig.model, stream=True)
        response: Final = rig.proxy.gateway.request("POST", "/vllm/chat/completions", body)
        assert response.status_code == 200, response.text
        (posted,) = _posted(rig.wire)
        assert "stream_options" not in posted, posted
        assert response.content == b"".join(_stream_bytes(body)), response.text
        (row,) = _rows(rig.model, 1)
        assert row["request_id"] == _identity(body), row
        prompt, completion = row["prompt_tokens"], row["completion_tokens"]
        assert row["status"] == "success" and isinstance(prompt, int) and prompt > 0, row
        assert isinstance(completion, int) and completion > 0, row
        assert row["spend"] == pytest.approx(_spend(prompt, completion)), row


def _holding(release: threading.Event, arrived: SimpleQueue[str]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "POST" and urlsplit(request.target).path.endswith("/chat/completions"):
            arrived.put(request.target)
            assert release.wait(timeout=90), "The burst was never released"
        return _streaming(request)

    return respond


@contextmanager
def _releasing(release: threading.Event) -> Generator[None, None, None]:
    try:
        yield
    finally:
        release.set()


def _worker_pids(log: str) -> tuple[int, ...]:
    return tuple(
        int(line.split(_STARTED_WORKER, 1)[1].split("]", 1)[0]) for line in log.splitlines() if _STARTED_WORKER in line
    )


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


async def _send(client: httpx.AsyncClient, key: str, body: Mapping[str, JsonValue]) -> _Outcome:
    try:
        response: Final = await client.post(
            "/vllm/chat/completions", json=body, headers={"Authorization": f"Bearer {key}", "Connection": "close"}
        )
    except httpx.TransportError as error:
        return _Outcome(_identity(body), 0, repr(error))
    return _Outcome(_identity(body), response.status_code, response.text)


async def _burst(rig: _Rig, bodies: Sequence[Mapping[str, JsonValue]]) -> tuple[_Outcome, ...]:
    async with httpx.AsyncClient(base_url=rig.base_url, timeout=120, trust_env=False) as client:
        return tuple(await asyncio.gather(*(_send(client, rig.key, body) for body in bodies)))


def _served(outcomes: Sequence[_Outcome]) -> frozenset[str]:
    return frozenset(
        outcome.identity for outcome in outcomes if outcome.status == 200 and outcome.text.endswith("data: [DONE]\n\n")
    )


async def test_worker_sigkill_mid_burst_lands_every_served_call_exactly_once(tmp_path: Path) -> None:
    release: Final = threading.Event()
    arrived: Final[SimpleQueue[str]] = SimpleQueue()
    with _vllm_proxy(tmp_path, _holding(release, arrived), {}) as rig, _releasing(release):
        workers: Final = eventually(
            lambda: _worker_pids(rig.proxy.log.read_text()), lambda pids: len(pids) == 2, seconds=30
        )
        burst: Final = asyncio.create_task(
            _burst(rig, tuple(_chat_body(rig.model, stream=True) for _ in range(_BURST)))
        )
        await asyncio.to_thread(eventually, arrived.qsize, lambda size: size == _BURST, 60)
        held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, rig.wire.url) for pid in workers})
        assert sum(held_by.values()) == _BURST, held_by
        victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
        with rig.proxy.replacing_worker(victim_pid) as kill:
            kill.freeze()
            await asyncio.to_thread(kill)
            release.set()
            outcomes: Final = await burst
            served: Final = _served(outcomes)
            assert held_by[survivor_pid] >= _BURST // 2, held_by
            assert len(served) == held_by[survivor_pid], (held_by, outcomes)
            assert all(outcome.status != 200 for outcome in outcomes if outcome.identity not in served), outcomes
            assert all('"usage"' not in outcome.text for outcome in outcomes if outcome.identity in served), outcomes
            (follow_up,) = await _burst(rig, (_chat_body(rig.model, stream=True),))
            assert _served((follow_up,)) == {follow_up.identity}, follow_up
        rows: Final = _rows(rig.model, len(served) + 1)
        assert sorted(str(row["request_id"]) for row in rows) == sorted((*served, follow_up.identity)), rows
        assert all(row["status"] == "success" for row in rows), rows
        replaced: Final = eventually(
            lambda: rig.proxy.log.read_text(),
            lambda log: len(_worker_pids(log)) == 3 and log.count(_STARTUP_COMPLETE) == 3,
            seconds=2 * graceful_stop_seconds(),
        )
        assert victim_pid not in _worker_pids(replaced)[2:], replaced
