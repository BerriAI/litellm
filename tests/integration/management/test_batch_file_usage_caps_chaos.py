import asyncio
import re
import signal
import time
from collections.abc import Callable, Coroutine, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import SimpleQueue
from threading import Event
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, string_value
from integration._support.database import scratch_database
from integration._support.process import owned_proxy, owned_proxy_process
from integration._support.redis_process import OwnedRedis, owned_redis
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.management._batch_file_caps import (
    DAY_SECONDS,
    DOWNLOADS,
    IN_GENERAL_SETTINGS,
    IN_KEY,
    MINUTE_SECONDS,
    THIS_KEY,
    UPLOADS,
    Timed,
    assert_download_limited,
    assert_upload_limited,
    batch_file,
    caps_config,
    config_entry,
    download,
    downloads_seen,
    file_content,
    marker,
    provider,
    provider_environment,
    timed,
    upload,
    uploads_seen,
    window_end,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(600)

WORKERS: Final = 2
UPLOAD_CAP: Final = 5
DOWNLOAD_CAP: Final = 3
HELD_UPLOADS: Final = 20
RESTART_CAP: Final = 4
STORED_CAP: Final = 2
BURST: Final = 20
DAY_ROOM_SECONDS: Final = 180
MINUTE_ROOM_SECONDS: Final = 30
BURST_ROOM_SECONDS: Final = 30
RELOAD_SECONDS: Final = "3"
BREAKER_RECOVERY_SECONDS: Final = "2"
RECOVERY_SECONDS: Final = 60
STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")


@dataclass(frozen=True, slots=True)
class Chaos:
    candidate: Gateway
    sibling: Gateway
    provider: Wire
    redis: OwnedRedis
    config: Path
    environment: Mapping[str, str]
    directory: Path


@pytest.fixture(scope="module")
def chaos(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Chaos]:
    directory: Final = tmp_path_factory.mktemp("batch_file_caps_chaos")
    with (
        gateway_from_environment() as gateway,
        wire_server(provider) as files,
        owned_redis(directory) as redis,
    ):
        config: Final = caps_config(directory, files.url, {})
        environment: Final = {
            **provider_environment(files.url),
            "REDIS_HOST": redis.host,
            "REDIS_PORT": str(redis.port),
            "REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": BREAKER_RECOVERY_SECONDS,
        }
        with (
            owned_proxy(gateway, directory, environment, config=config, workers=WORKERS) as candidate,
            owned_proxy(gateway, directory, environment, config=config) as sibling,
        ):
            yield Chaos(candidate, sibling, files, redis, config, environment, directory)


def _key(scenario: Scenario, **limits: JsonValue) -> str:
    return scenario.key(metadata=limits)


def _statuses(responses: tuple[httpx.Response, ...]) -> list[int]:
    return sorted(response.status_code for response in responses)


def _accepted(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    return string_value(response.json()["id"])


async def _upload_burst(
    base_url: str, key: str, mark: str, count: int, *, tolerate_disconnects: bool = False
) -> tuple[httpx.Response, ...]:
    headers: Final = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(base_url=base_url, timeout=90, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(
                client.post(
                    "/v1/files",
                    data={"purpose": "batch"},
                    files={"file": ("batch.jsonl", batch_file(mark, 1), "application/jsonl")},
                    headers=headers,
                )
                for _ in range(count)
            ),
            return_exceptions=tolerate_disconnects,
        )
    for result in results:
        assert isinstance(result, httpx.Response | httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, httpx.Response))


async def _download_burst(base_url: str, key: str, file_id: str, count: int) -> tuple[httpx.Response, ...]:
    headers: Final = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(base_url=base_url, timeout=90, trust_env=False) as client:
        return await asyncio.gather(
            *(client.get(f"/v1/files/{file_id}/content", headers=headers) for _ in range(count))
        )


@dataclass(frozen=True, slots=True)
class Burst:
    subject: str
    window_ends: float
    answers: tuple[Timed, ...]

    def statuses(self) -> list[int]:
        return _statuses(tuple(observed.response for observed in self.answers))

    def refused(self) -> tuple[Timed, ...]:
        return tuple(observed for observed in self.answers if observed.response.status_code == 429)

    def allowed_exactly(self, count: int) -> bool:
        in_window: Final = all(observed.after < self.window_ends for observed in self.answers)
        return in_window and self.statuses() == [200] * count + [429] * (len(self.answers) - count)


def _timed_burst(burst: Coroutine[object, None, tuple[httpx.Response, ...]]) -> tuple[Timed, ...]:
    before: Final = time.time()
    responses: Final = asyncio.run(burst)
    after: Final = time.time()
    return tuple(Timed(response, before, after) for response in responses)


def _uploads_after_the_sibling_used_the_cap(chaos: Chaos, scenario: Scenario) -> Burst:
    key: Final = _key(scenario, **{UPLOADS: UPLOAD_CAP})
    mark: Final = marker()
    day_ends: Final = window_end(DAY_SECONDS, BURST_ROOM_SECONDS)
    counted: Final = tuple(upload(chaos.sibling, key, batch_file(mark, 1)) for _ in range(UPLOAD_CAP))
    assert _statuses(counted) == [200] * UPLOAD_CAP, [response.text for response in counted]
    return Burst(mark, day_ends, _timed_burst(_upload_burst(str(chaos.candidate.client.base_url), key, mark, BURST)))


def _downloads_after_the_sibling_used_the_cap(chaos: Chaos, scenario: Scenario) -> Burst:
    key: Final = _key(scenario, **{DOWNLOADS: DOWNLOAD_CAP})
    file_id: Final = f"file-{marker()}"
    minute_ends: Final = window_end(MINUTE_SECONDS, BURST_ROOM_SECONDS)
    counted: Final = tuple(download(chaos.sibling, key, file_id) for _ in range(DOWNLOAD_CAP))
    assert _statuses(counted) == [200] * DOWNLOAD_CAP, [response.text for response in counted]
    assert {response.content for response in counted} == {file_content(file_id)}
    return Burst(
        file_id, minute_ends, _timed_burst(_download_burst(str(chaos.candidate.client.base_url), key, file_id, BURST))
    )


async def test_uploads_fall_back_to_per_process_counts_while_redis_is_down(chaos: Chaos) -> None:
    with chaos.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{UPLOADS: UPLOAD_CAP})
        mark: Final = marker()
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        before: Final = tuple(upload(chaos.candidate, key, batch_file(mark, 1)) for _ in range(UPLOAD_CAP - 2))
        assert _statuses(before) == [200] * (UPLOAD_CAP - 2), [response.text for response in before]
        await asyncio.to_thread(chaos.redis.stop)
        try:
            during: Final = await _upload_burst(str(chaos.candidate.client.base_url), key, mark, BURST)
        finally:
            await asyncio.to_thread(chaos.redis.start)
        assert time.time() < day_ends
        statuses: Final = _statuses(during)
        assert set(statuses) <= {200, 429}, [response.text for response in during]
        accepted: Final = len(before) + statuses.count(200)
        assert UPLOAD_CAP <= accepted <= UPLOAD_CAP * WORKERS, statuses
        assert len(uploads_seen(chaos.provider.drain(), mark)) == accepted
        shared: Final = await asyncio.to_thread(
            eventually,
            partial(_uploads_after_the_sibling_used_the_cap, chaos, scenario),
            lambda burst: burst.allowed_exactly(0),
            RECOVERY_SECONDS,
        )
        for refusal in shared.refused():
            assert_upload_limited(refusal, shared.window_ends, UPLOAD_CAP, THIS_KEY, IN_KEY)
        assert len(uploads_seen(chaos.provider.drain(), shared.subject)) == UPLOAD_CAP


async def test_downloads_keep_answering_while_redis_hangs(chaos: Chaos) -> None:
    with chaos.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{DOWNLOADS: DOWNLOAD_CAP})
        file_id: Final = f"file-{marker()}"
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        chaos.redis.signal(signal.SIGSTOP)
        try:
            burst: Final = asyncio.create_task(
                _download_burst(str(chaos.candidate.client.base_url), key, file_id, BURST)
            )
            liveliness: Final = await asyncio.to_thread(
                timed, partial(chaos.candidate.request, "GET", "/health/liveliness")
            )
            during: Final = await burst
        finally:
            chaos.redis.signal(signal.SIGCONT)
        assert time.time() < minute_ends
        assert liveliness.response.status_code == 200, liveliness.response.text
        assert liveliness.after - liveliness.before < 5, liveliness
        statuses: Final = _statuses(during)
        assert set(statuses) <= {200, 429}, [response.text for response in during]
        assert DOWNLOAD_CAP <= statuses.count(200) <= DOWNLOAD_CAP * WORKERS, statuses
        assert len(downloads_seen(chaos.provider.drain(), file_id)) == statuses.count(200)
        shared: Final = await asyncio.to_thread(
            eventually,
            partial(_downloads_after_the_sibling_used_the_cap, chaos, scenario),
            lambda burst: burst.allowed_exactly(0),
            RECOVERY_SECONDS,
        )
        for refusal in shared.refused():
            assert_download_limited(refusal, shared.window_ends, shared.subject, DOWNLOAD_CAP, THIS_KEY, IN_KEY)
        assert len(downloads_seen(chaos.provider.drain(), shared.subject)) == DOWNLOAD_CAP


def _held_provider(release: Event, held: SimpleQueue[str]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if (request.method, request.target) == ("POST", "/v1/files"):
            held.put(request.target)
            assert release.wait(timeout=120), "The held uploads were never released"
        return provider(request)

    return respond


@contextmanager
def _released_on_exit(release: Event) -> Iterator[None]:
    try:
        yield
    finally:
        release.set()


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _worker_pids(log: Path) -> tuple[int, ...]:
    return tuple(int(pid) for pid in STARTED_WORKER.findall(log.read_text()))


async def test_a_killed_worker_keeps_its_upload_slots_used_and_the_sibling_serving(
    chaos: Chaos, tmp_path: Path
) -> None:
    release: Final = Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    with wire_server(_held_provider(release, held)) as files:
        config: Final = caps_config(tmp_path, files.url, {})
        environment: Final = {**chaos.environment, **provider_environment(files.url)}
        with (
            owned_proxy_process(chaos.candidate, tmp_path, environment, config=config, workers=WORKERS) as owned,
            owned.gateway.scenario() as scenario,
            _released_on_exit(release),
        ):
            candidate: Final = owned.gateway
            key: Final = _key(scenario, **{UPLOADS: HELD_UPLOADS})
            mark: Final = marker()
            day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
            workers: Final = eventually(partial(_worker_pids, owned.log), lambda pids: len(pids) == WORKERS, seconds=30)
            burst: Final = asyncio.create_task(
                _upload_burst(str(candidate.client.base_url), key, mark, HELD_UPLOADS, tolerate_disconnects=True)
            )
            await asyncio.to_thread(eventually, held.qsize, lambda size: size == HELD_UPLOADS, 90)
            held_by: Final = {pid: _open_upstream_connections(pid, files.url) for pid in workers}
            assert sum(held_by.values()) == HELD_UPLOADS, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            served: Final = await burst
            assert _statuses(served) == [200] * held_by[survivor_pid], (held_by, [response.text for response in served])
            assert_upload_limited(
                timed(partial(upload, candidate, key, batch_file(marker(), 1))),
                day_ends,
                HELD_UPLOADS,
                THIS_KEY,
                IN_KEY,
            )
            assert len(uploads_seen(files.drain(), mark)) == HELD_UPLOADS
            assert candidate.request("GET", "/health/liveliness").status_code == 200


def test_the_daily_count_survives_a_proxy_restart(chaos: Chaos) -> None:
    with chaos.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{UPLOADS: RESTART_CAP})
        mark: Final = marker()
        with owned_proxy(chaos.candidate, chaos.directory, chaos.environment, config=chaos.config) as first:
            day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
            before: Final = tuple(upload(first, key, batch_file(mark, 1)) for _ in range(RESTART_CAP - 1))
            assert _statuses(before) == [200] * (RESTART_CAP - 1), [response.text for response in before]
        with owned_proxy(chaos.candidate, chaos.directory, chaos.environment, config=chaos.config) as second:
            assert upload(second, key, batch_file(mark, 1)).status_code == 200
            assert_upload_limited(
                timed(partial(upload, second, key, batch_file(mark, 1))), day_ends, RESTART_CAP, THIS_KEY, IN_KEY
            )
        assert len(uploads_seen(chaos.provider.drain(), mark)) == RESTART_CAP


def _plain_key(gateway: Gateway) -> str:
    return string_value(gateway.post("/key/generate", {})["key"])


def _download_statuses(gateway: Gateway, key: str) -> list[int]:
    file_id: Final = f"file-{marker()}"
    responses: Final = asyncio.run(_download_burst(str(gateway.client.base_url), key, file_id, BURST))
    return _statuses(responses)


def _downloads_in_one_minute(gateway: Gateway, key: str) -> Burst:
    file_id: Final = f"file-{marker()}"
    minute_ends: Final = window_end(MINUTE_SECONDS, BURST_ROOM_SECONDS)
    return Burst(file_id, minute_ends, _timed_burst(_download_burst(str(gateway.client.base_url), key, file_id, BURST)))


def _assert_stored_cap(burst: Burst) -> None:
    assert burst.allowed_exactly(STORED_CAP), burst.statuses()
    for refusal in burst.refused():
        assert_download_limited(refusal, burst.window_ends, burst.subject, STORED_CAP, THIS_KEY, IN_GENERAL_SETTINGS)


def _stored_cap_on_every_worker(gateway: Gateway, key: str) -> Burst:
    return eventually(
        partial(_downloads_in_one_minute, gateway, key), lambda burst: burst.allowed_exactly(STORED_CAP), seconds=30
    )


def test_a_download_cap_stored_through_the_config_api_reaches_every_worker_and_survives_a_restart(
    chaos: Chaos, tmp_path: Path
) -> None:
    with scratch_database() as database_url:
        environment: Final = {
            **chaos.environment,
            "DATABASE_URL": database_url,
            "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": RELOAD_SECONDS,
        }
        boot: Final = partial(
            owned_proxy,
            chaos.candidate,
            tmp_path,
            environment,
            config=chaos.config,
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        )
        with boot(workers=WORKERS) as candidate:
            first: Final = _plain_key(candidate)
            second: Final = _plain_key(candidate)
            assert _download_statuses(candidate, first) == [200] * BURST
            assert config_entry(candidate, DOWNLOADS)["field_value"] is None
            stored: Final = candidate.request(
                "POST",
                "/config/field/update",
                {"field_name": DOWNLOADS, "field_value": STORED_CAP, "config_type": "general_settings"},
            )
            assert stored.status_code == 200, stored.text
            entry: Final = config_entry(candidate, DOWNLOADS)
            assert (entry["field_value"], entry["stored_in_db"]) == (STORED_CAP, True), entry
            _assert_stored_cap(_stored_cap_on_every_worker(candidate, first))
            _assert_stored_cap(_stored_cap_on_every_worker(candidate, second))
        with boot() as restarted:
            _assert_stored_cap(_downloads_in_one_minute(restarted, first))
            assert config_entry(restarted, DOWNLOADS)["field_value"] == STORED_CAP
            removed: Final = restarted.request(
                "POST", "/config/field/delete", {"field_name": DOWNLOADS, "config_type": "general_settings"}
            )
            assert removed.status_code == 200, removed.text
            eventually(
                partial(_download_statuses, restarted, second), lambda statuses: statuses == [200] * BURST, seconds=30
            )
            assert config_entry(restarted, DOWNLOADS)["field_value"] is None
