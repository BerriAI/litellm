import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, held, object_value, string_value
from integration._support.process import DB_PUSH, graceful_stop_seconds, owned_proxy
from integration._support.redis_process import owned_redis
from integration._support.responses_stream import OPENAI_TARGET, completed, created, delta, frame, response_object
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_MODEL: Final = "background-polling-cancel"
_NO_DATABASE: Final = ("DATABASE_URL", "DATABASE_URL_READ_REPLICA")
_DELTAS: Final = 40
_DELTA_PAUSE_SECONDS: Final = 0.25
_STREAM_SECONDS: Final = _DELTAS * _DELTA_PAUSE_SECONDS
_OWNED_PROXY_SECONDS: Final = 2 * graceful_stop_seconds() + 240
_MODEL_DISCOVERY: Final = ("GET", "/v1/models")
_NO_MODELS: Final = Reply(body=b'{"object":"list","data":[]}')
_POLLING_VIA_CACHE: Final = MappingProxyType({"responses": {"background_mode": {"polling_via_cache": "all"}}})
_KEEPALIVE_SECONDS: Final = 0.05


def _config(directory: Path, wire: Wire, litellm_settings: Mapping[str, JsonValue] = _POLLING_VIA_CACHE) -> Path:
    path: Final = directory / f"background-polling-cancel-{uuid.uuid4().hex}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    {
                        "model_name": _MODEL,
                        "litellm_params": {
                            "model": "openai/gpt-5.4-mini",
                            "api_base": wire.url,
                            "api_key": "synthetic-openai-key",
                        },
                    }
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "coordination_redis": {"host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
                },
                "litellm_settings": dict(litellm_settings),
            }
        )
    )
    return path


@contextmanager
def _proxies_sharing_one_redis(
    gateway: Gateway, directory: Path, config: Path, count: int, *, with_database: bool = False
) -> Iterator[tuple[Gateway, ...]]:
    with owned_redis(directory / "redis") as cache, ExitStack() as proxies:
        overrides: Final = {
            "REDIS_HOST": cache.host,
            "REDIS_PORT": str(cache.port),
            "STORE_MODEL_IN_DB": "False",
            "DISABLE_SCHEMA_UPDATE": str(not with_database).lower(),
        }
        yield tuple(
            proxies.enter_context(
                owned_proxy(
                    gateway,
                    directory / f"proxy-{index}",
                    overrides,
                    config=config,
                    database_setup=DB_PUSH if with_database else (),
                    remove_environment=() if with_database else _NO_DATABASE,
                )
            )
            for index in range(count)
        )


def _serving(reply: Reply, requested: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if (request.method, request.target) == _MODEL_DISCOVERY:
            return _NO_MODELS
        assert request.method == "POST" and request.target.startswith(OPENAI_TARGET), request.target
        requested.set()
        return reply

    return respond


def _create_background_response(proxy: Gateway, key: str | None = None) -> str:
    created_response: Final = proxy.request(
        "POST", "/v1/responses", {"model": _MODEL, "input": uuid.uuid4().hex, "background": True}, key=key
    )
    assert created_response.status_code == 200, created_response.text
    body: Final = object_value(created_response.json())
    assert body["status"] == "queued", body
    return string_value(body["id"])


def _cancel(proxy: Gateway, polling_id: str) -> None:
    cancelled: Final = proxy.request("POST", f"/v1/responses/{polling_id}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert object_value(cancelled.json())["status"] == "cancelled", cancelled.text


def _status(proxy: Gateway, polling_id: str) -> JsonValue:
    polled: Final = proxy.request("GET", f"/v1/responses/{polling_id}")
    assert polled.status_code == 200, polled.text
    return object_value(polled.json())["status"]


@pytest.mark.timeout(2 * _OWNED_PROXY_SECONDS + 3 * _STREAM_SECONDS + 60)
def test_a_cancel_sent_to_another_proxy_stops_the_background_stream_and_the_response_stays_cancelled(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    requested: Final = threading.Event()
    slow_stream: Final = Reply(
        content_type="text/event-stream",
        chunks=(
            frame(created(identity)),
            *(frame(delta(identity, "word ")) for _ in range(_DELTAS)),
            frame(completed(identity, "word " * _DELTAS)),
        ),
        pause_between_chunks=_DELTA_PAUSE_SECONDS,
    )
    with (
        wire_server(_serving(slow_stream, requested)) as wire,
        _proxies_sharing_one_redis(gateway, tmp_path, _config(tmp_path, wire), 2) as (streaming, cancelling),
    ):
        polling_id: Final = _create_background_response(streaming)
        eventually(requested.is_set, lambda seen: seen, seconds=30)
        _cancel(cancelling, polling_id)
        eventually(lambda: wire.disconnected.qsize(), lambda dropped: dropped == 1, seconds=_STREAM_SECONDS)
        held(
            lambda: _status(cancelling, polling_id),
            lambda status: status == "cancelled",
            holding=_STREAM_SECONDS,
            seconds=2 * _STREAM_SECONDS,
        )


@pytest.mark.timeout(_OWNED_PROXY_SECONDS + 60)
def test_a_background_response_cancelled_just_before_its_stream_finishes_stays_cancelled(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    requested: Final = threading.Event()
    cancel_sent: Final = threading.Event()
    held_stream: Final = Reply(
        content_type="text/event-stream",
        chunks=(
            frame(created(identity)),
            frame({"type": "response.in_progress", "response": response_object(identity, "in_progress")}),
            frame(delta(identity, "done")),
            frame(completed(identity, "done")),
        ),
        gate_after_first=cancel_sent,
    )
    with (
        wire_server(_serving(held_stream, requested)) as wire,
        _proxies_sharing_one_redis(gateway, tmp_path, _config(tmp_path, wire), 1) as (proxy,),
    ):
        polling_id: Final = _create_background_response(proxy)
        eventually(requested.is_set, lambda seen: seen, seconds=30)
        _cancel(proxy, polling_id)
        cancel_sent.set()
        held(lambda: _status(proxy, polling_id), lambda status: status == "cancelled", holding=3)


@pytest.mark.timeout(2 * _OWNED_PROXY_SECONDS + 3 * _STREAM_SECONDS + 60)
def test_a_delete_sent_to_another_proxy_stops_the_background_stream_and_the_response_stays_gone(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    requested: Final = threading.Event()
    slow_stream: Final = Reply(
        content_type="text/event-stream",
        chunks=(
            frame(created(identity)),
            *(frame(delta(identity, "word ")) for _ in range(_DELTAS)),
            frame(completed(identity, "word " * _DELTAS)),
        ),
        pause_between_chunks=_DELTA_PAUSE_SECONDS,
    )
    with (
        wire_server(_serving(slow_stream, requested)) as wire,
        _proxies_sharing_one_redis(gateway, tmp_path, _config(tmp_path, wire), 2) as (streaming, deleting),
    ):
        polling_id: Final = _create_background_response(streaming)
        eventually(requested.is_set, lambda seen: seen, seconds=30)
        deleted: Final = deleting.request("DELETE", f"/v1/responses/{polling_id}")
        assert deleted.status_code == 200, deleted.text
        eventually(lambda: wire.disconnected.qsize(), lambda dropped: dropped == 1, seconds=_STREAM_SECONDS)
        held(
            lambda: streaming.request("GET", f"/v1/responses/{polling_id}").status_code,
            lambda status_code: status_code == 404,
            holding=_STREAM_SECONDS,
            seconds=2 * _STREAM_SECONDS,
        )


def _streamed_response_status(proxy: Gateway, key: str) -> int:
    return proxy.request(
        "POST", "/v1/responses", {"model": _MODEL, "input": uuid.uuid4().hex, "stream": True}, key=key
    ).status_code


@pytest.mark.timeout(_OWNED_PROXY_SECONDS + 90)
def test_a_background_response_cancelled_during_keepalive_pings_frees_its_parallel_slot(
    gateway: Gateway, tmp_path: Path
) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    requested: Final = threading.Event()
    answer: Final = threading.Event()
    quick_stream: Final = Reply(
        content_type="text/event-stream",
        chunks=(frame(created(identity)), frame(delta(identity, "done")), frame(completed(identity, "done"))),
    )

    def respond(request: Request) -> Reply:
        if (request.method, request.target) == _MODEL_DISCOVERY:
            return _NO_MODELS
        assert request.method == "POST" and request.target.startswith(OPENAI_TARGET), request.target
        if not requested.is_set():
            requested.set()
            answer.wait()
        return quick_stream

    with (
        wire_server(respond) as wire,
        _proxies_sharing_one_redis(
            gateway,
            tmp_path,
            _config(tmp_path, wire, {**_POLLING_VIA_CACHE, "sse_keepalive_ping_interval_seconds": _KEEPALIVE_SECONDS}),
            1,
            with_database=True,
        ) as (proxy,),
        proxy.scenario() as scenario,
    ):
        try:
            key: Final = scenario.key(max_parallel_requests=1)
            polling_id: Final = _create_background_response(proxy, key)
            eventually(requested.is_set, lambda seen: seen, seconds=30)
            held(lambda: _status(proxy, polling_id), lambda status: status == "in_progress", holding=1)
            assert _streamed_response_status(proxy, key) == 429
            _cancel(proxy, polling_id)
            eventually(
                lambda: _streamed_response_status(proxy, key), lambda status_code: status_code == 200, seconds=20
            )
            assert _status(proxy, polling_id) == "cancelled"
        finally:
            answer.set()
