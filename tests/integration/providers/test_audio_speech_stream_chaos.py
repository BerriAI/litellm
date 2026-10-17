import asyncio
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import (
    OwnedProxy,
    graceful_stop_seconds,
    owned_proxy_process,
)
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(4 * graceful_stop_seconds() + 300)

_MODEL: Final = "chaos-speech"
_SPEECH: Final = "/v1/audio/speech"
_FIRST: Final = b"audio-frame-0:" + bytes(range(64))
_REST: Final = (
    b"audio-frame-1:" + bytes(range(64, 192)),
    b"audio-frame-2:" + bytes(range(192, 256)),
)
_CLIP: Final = _FIRST + b"".join(_REST)
_BURST: Final = 24
_HELD: Final = 8
_SPEND_SQL: Final = (
    'SELECT litellm_call_id, status, spend, metadata FROM "LiteLLM_SpendLogs" '
    "WHERE litellm_call_id = ANY(string_to_array(%s, ','))"
)

Kind = Literal["complete", "cut", "hang_up", "buffered", "held", "held_buffered"]
_MIXED: Final[tuple[Kind, ...]] = ("complete", "cut", "hang_up", "buffered")
_GATED: Final[frozenset[str]] = frozenset({"hang_up", "held", "held_buffered"})
_BUFFERED: Final[frozenset[str]] = frozenset({"buffered", "held_buffered"})


@dataclass(frozen=True, slots=True)
class Clip:
    kind: Kind
    call_id: str

    @property
    def text(self) -> str:
        return f"{self.kind} {self.call_id}"


@dataclass(frozen=True, slots=True)
class Heard:
    clip: Clip
    status: int
    body: bytes
    error: str


def _clips(marker: str, count: int, kinds: Sequence[Kind]) -> tuple[Clip, ...]:
    return tuple(
        Clip(kinds[index % len(kinds)], f"{marker}-{index:02d}-{uuid.uuid4().hex[:8]}") for index in range(count)
    )


def _peer(release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        kind: Final = string_value(object_value(JSON_OBJECT.validate_json(request.body))["input"]).split(" ", 1)[0]
        return Reply(
            content_type="audio/mpeg",
            chunks=(_FIRST, *_REST),
            gate_after_first=release if kind in _GATED else None,
            abort_after=1 if kind == "cut" else None,
        )

    return respond


def _config(directory: Path, wire: Wire) -> Path:
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    settings: Final = object_value(config["litellm_settings"])
    merged: Final = {
        **config,
        "model_list": [
            {
                "model_name": _MODEL,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini-tts",
                    "api_base": wire.url,
                    "api_key": "synthetic-key",
                },
                "model_info": {"mode": "audio_speech"},
            }
        ],
        "litellm_settings": {
            **settings,
            "cache_params": {
                **object_value(settings["cache_params"]),
                "namespace": f"tts-{uuid.uuid4().hex}",
            },
        },
        "router_settings": {
            **object_value(config["router_settings"]),
            "num_retries": 0,
        },
    }
    path: Final = directory / f"speech-chaos-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(merged))
    return path


@contextmanager
def _owned(
    gateway: Gateway,
    directory: Path,
    wire: Wire,
    overrides: Mapping[str, str] = MappingProxyType({}),
) -> Iterator[OwnedProxy]:
    with owned_proxy_process(gateway, directory, overrides, config=_config(directory, wire), workers=2) as proxy:
        yield proxy


def _body(clip: Clip) -> dict[str, JsonValue]:
    stream: Final[dict[str, JsonValue]] = {} if clip.kind in _BUFFERED else {"stream": True}
    return {"model": _MODEL, "input": clip.text, "voice": "alloy", **stream}


async def _hear(client: httpx.AsyncClient, key: str, clip: Clip) -> Heard:
    headers: Final = {
        "Authorization": f"Bearer {key}",
        "x-litellm-call-id": clip.call_id,
    }
    try:
        async with client.stream("POST", _SPEECH, json=_body(clip), headers=headers) as response:
            if clip.kind == "hang_up":
                return Heard(clip, response.status_code, await anext(response.aiter_raw()), "")
            return Heard(
                clip,
                response.status_code,
                b"".join([chunk async for chunk in response.aiter_raw()]),
                "",
            )
    except httpx.TransportError as error:
        return Heard(clip, -1, b"", type(error).__name__)


async def _burst(url: str, key: str, clips: Sequence[Clip]) -> tuple[Heard, ...]:
    async with httpx.AsyncClient(
        base_url=url,
        timeout=90,
        trust_env=False,
        limits=httpx.Limits(max_connections=len(clips)),
    ) as client:
        return tuple(await asyncio.gather(*(_hear(client, key, clip) for clip in clips)))


def _rows(clips: Sequence[Clip], database_url: str | None) -> list[dict[str, JsonValue]]:
    return read_rows(
        _SPEND_SQL,
        (",".join(clip.call_id for clip in clips),),
        database_url=database_url,
    )


def _billed_once(clips: Sequence[Clip], database_url: str | None = None) -> dict[str, dict[str, JsonValue]]:
    rows: Final = eventually(
        lambda: _rows(clips, database_url),
        lambda found: len(found) >= len(clips),
        seconds=120,
        return_last_on_timeout=True,
    )
    counts: Final = Counter(string_value(row["litellm_call_id"]) for row in rows)
    missing: Final = sorted(clip.call_id for clip in clips if clip.call_id not in counts)
    assert missing == [] and set(counts.values()) == {1}, (missing, counts)
    return {string_value(row["litellm_call_id"]): row for row in rows}


def _assert_heard(heard: Heard) -> None:
    match heard.clip.kind:
        case "cut":
            assert (heard.status, heard.error) == (-1, "RemoteProtocolError"), heard
        case "hang_up":
            assert (heard.status, heard.body) == (200, _FIRST), heard
        case _:
            assert (heard.status, heard.body, heard.error) == (200, _CLIP, ""), heard


def _assert_billed(clip: Clip, row: dict[str, JsonValue]) -> None:
    spend: Final = row["spend"]
    assert isinstance(spend, float) and spend > 0, (clip, row)
    match clip.kind:
        case "cut":
            assert row["status"] == "failure", (clip, row)
        case "hang_up":
            assert row["status"] == "success", (clip, row)
            error: Final = object_value(object_value(row["metadata"])["error_information"])
            assert error["error_code"] == "499", (clip, error)
        case _:
            assert row["status"] == "success", (clip, row)


def _held_by(pid: int, wire: Wire) -> int:
    port: Final = urlsplit(wire.url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _marker() -> str:
    return f"ttschaos{uuid.uuid4().hex[:8]}"


async def test_a_mixed_streamed_speech_burst_bills_every_call_once_and_leaks_no_key_slot(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    marker: Final = _marker()
    with wire_server(_peer(release)) as wire, _owned(gateway, tmp_path, wire) as proxy:
        with proxy.gateway.scenario() as scenario:
            key: Final = scenario.key(max_parallel_requests=_BURST)
            url: Final = str(proxy.gateway.client.base_url)
            first: Final = _clips(marker, _BURST, _MIXED)
            heard: Final = await _burst(url, key, first)
            release.set()
            billed: Final = await asyncio.to_thread(_billed_once, first)
            second: Final = _clips(f"{marker}again", _BURST, ("complete",))
            again: Final = await _burst(url, key, second)
            await asyncio.to_thread(_billed_once, second)
    for item in (*heard, *again):
        _assert_heard(item)
    for clip in first:
        _assert_billed(clip, billed[clip.call_id])
    assert len(wire.drain()) == 2 * _BURST


async def test_a_worker_killed_mid_streamed_speech_leaves_the_sibling_serving_and_billing(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    marker: Final = _marker()
    held: Final = _clips(marker, _HELD, ("held", "held_buffered"))
    after: Final = _clips(f"{marker}after", _HELD, ("complete", "buffered"))
    with wire_server(_peer(release)) as wire, _owned(gateway, tmp_path, wire) as proxy:
        workers: Final = eventually(
            proxy.worker_pids,
            lambda pids: len(pids) == 2,
            seconds=graceful_stop_seconds(),
        )
        url: Final = str(proxy.gateway.client.base_url)
        async with asyncio.TaskGroup() as tasks:
            burst: Final = tasks.create_task(_burst(url, proxy.gateway.key, held))
            try:
                await asyncio.to_thread(eventually, wire.received.qsize, lambda size: size >= _HELD, 90)
                held_by: Final = MappingProxyType({pid: _held_by(pid, wire) for pid in workers})
                victim: Final = max(workers, key=held_by.__getitem__)
                with proxy.replacing_worker(victim) as kill:
                    await asyncio.to_thread(kill)
                    release.set()
                    served: Final = await burst
            finally:
                release.set()
        later: Final = await _burst(url, proxy.gateway.key, after)
        survivors: Final = tuple(item for item in served if item.status != -1)
        billed: Final = await asyncio.to_thread(_billed_once, (*(item.clip for item in survivors), *after))
    assert sum(held_by.values()) == _HELD and held_by[victim] > 0, held_by
    assert len(survivors) == _HELD - held_by[victim], (held_by, served)
    for item in (*survivors, *later):
        _assert_heard(item)
        _assert_billed(item.clip, billed[item.clip.call_id])


async def test_a_proxy_restart_mid_streamed_speech_bills_every_served_clip_once_and_the_next_boot_serves(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    marker: Final = _marker()
    held: Final = _clips(marker, _HELD, ("held", "held_buffered"))
    after: Final = _clips(f"{marker}after", _HELD, ("complete", "buffered"))
    with wire_server(_peer(release)) as wire:
        with _owned(gateway, tmp_path, wire) as first:
            async with asyncio.TaskGroup() as tasks:
                burst: Final = tasks.create_task(_burst(str(first.gateway.client.base_url), first.gateway.key, held))
                try:
                    await asyncio.to_thread(eventually, wire.received.qsize, lambda size: size >= _HELD, 90)
                    first.begin_shutdown()
                finally:
                    release.set()
            served: Final = await burst
        with _owned(gateway, tmp_path, wire) as second:
            later: Final = await _burst(str(second.gateway.client.base_url), second.gateway.key, after)
            completed: Final = tuple(item for item in served if item.status == 200)
            billed: Final = await asyncio.to_thread(_billed_once, (*(item.clip for item in completed), *after))
    assert len(completed) == _HELD, served
    for item in (*completed, *later):
        _assert_heard(item)
        _assert_billed(item.clip, billed[item.clip.call_id])
