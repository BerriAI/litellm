import asyncio
import hashlib
import itertools
import re
import signal
import threading
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, TypeAlias

import httpcore
import httpx
import psutil
import pytest
import yaml
from integration._support import device_login as dl
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(900)

_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_DROPPED: Final = re.compile(r"original model: (\S+), ignoring and continuing")
_FIXED: Final[Mapping[str, str]] = MappingProxyType(
    {"chatgpt-fixed": "chatgpt/gpt-5.6-terra", "copilot-fixed": "github_copilot/gpt-5.2"}
)
_WILDCARDS: Final[Mapping[str, str]] = MappingProxyType(
    {"chatgpt/*": "chatgpt/gpt-5.6-terra", "github_copilot/*": "github_copilot/gpt-5.2"}
)
_LOGIN_PREFIXES: Final = ("chatgpt/", "github_copilot/")
_LOGIN_NAMES: Final = (*_FIXED, *_FIXED.values())
_CONTROL_CALLS_BEFORE_THE_RELOAD: Final = 1
_SHAPES: Final[Mapping[str, bool]] = MappingProxyType(
    {"docs-shapes": False, "docs-shapes-beside-a-universal-wildcard": True}
)
_NOT_FOUND: Final = "Invalid model name passed in model="
_NO_HEALTHY: Final = "There are no healthy deployments for this model"
_CONTROL: Final = "device-login-boot-control"
_EXTRA: Final[Mapping[str, JsonValue]] = MappingProxyType({"num_retries": 0, "cache": {"no-cache": True}})
_CALL_ID: Final = "x-litellm-call-id"
_REQUEST_TIMEOUT_SECONDS: Final = 6
_POLL_SECONDS: Final = 3
_ADDRESS: Final = TypeAdapter(tuple[str, int])


@dataclass(frozen=True, slots=True)
class _Served:
    status: int
    text: str
    call_id: str

    def message(self) -> str:
        error: Final = rv.JSON_OBJECT.validate_json(self.text)["error"]
        return str(rv.JSON_OBJECT.validate_python(error)["message"]) if isinstance(error, dict) else str(error)


@dataclass(frozen=True, slots=True)
class _TokenDirs:
    chatgpt: Path
    copilot: Path

    def environment(self) -> Mapping[str, str]:
        return MappingProxyType({"CHATGPT_TOKEN_DIR": str(self.chatgpt), "GITHUB_COPILOT_TOKEN_DIR": str(self.copilot)})

    def store_valid_tokens(self, api_url: str) -> None:
        dl.write_chatgpt(self.chatgpt, dl.chatgpt_record(dl.CHATGPT_STORED))
        dl.write_copilot_key(self.copilot, dl.COPILOT_STORED_KEY, api_url)


def _token_dirs(directory: Path) -> _TokenDirs:
    dirs: Final = _TokenDirs(directory / "chatgpt", directory / "copilot")
    dirs.chatgpt.mkdir()
    dirs.copilot.mkdir()
    return dirs


def _config(control_api_url: str, directory: Path, *, fixed: bool, wildcards: bool, universal: bool = False) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    fixed_entries: Final = [{"model_name": name, "litellm_params": {"model": model}} for name, model in _FIXED.items()]
    wildcard_entries: Final = [{"model_name": pattern, "litellm_params": {"model": pattern}} for pattern in _WILDCARDS]
    config["model_list"] = [
        *(fixed_entries if fixed else []),
        *(wildcard_entries if wildcards else []),
        *([{"model_name": "*", "litellm_params": {"model": "*"}}] if universal else []),
        {
            "model_name": _CONTROL,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_base": f"{control_api_url}/v1",
                "api_key": dl.CONTROL_KEY,
            },
        },
    ]
    path: Final = directory / "device-login-boot.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _chat(base_url: str, key: str, model: str, marker: str, seconds: float = 60) -> _Served:
    with httpx.Client(base_url=base_url, timeout=seconds, trust_env=False) as client:
        response: Final = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": f"Reply to marker-{marker}"}], **_EXTRA},
            headers={"Authorization": f"Bearer {key}"},
        )
    return _Served(response.status_code, response.text, response.headers.get(_CALL_ID, ""))


def _model_ids(text: str) -> tuple[str, ...]:
    return tuple(
        sorted(str(item["id"]) for item in rv.ITEMS.validate_python(rv.JSON_OBJECT.validate_json(text)["data"]))
    )


def _workers(owned: OwnedProxy) -> tuple[int, ...]:
    return eventually(
        lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
        lambda pids: len(pids) == 2,
        seconds=30,
    )


def _base_url(owned: OwnedProxy) -> str:
    return str(owned.gateway.client.base_url).rstrip("/")


def _proxy_port(owned: OwnedProxy) -> int:
    return int(httpx.URL(_base_url(owned)).port or 0)


def _accepted(pid: int, proxy_port: int, client_port: int) -> bool:
    return any(
        connection.status == psutil.CONN_ESTABLISHED
        and connection.laddr
        and connection.laddr.port == proxy_port
        and connection.raddr
        and connection.raddr.port == client_port
        for connection in psutil.Process(pid).net_connections(kind="tcp")
    )


def _listing_sample(owned: OwnedProxy, workers: tuple[int, ...]) -> tuple[int, tuple[str, ...]] | None:
    with httpx.Client(base_url=_base_url(owned), timeout=30, trust_env=False) as client:
        response: Final = client.get("/v1/models", headers={"Authorization": f"Bearer {owned.gateway.key}"})
        assert response.status_code == 200, response.text
        stream: Final[object] = response.extensions["network_stream"]
        assert isinstance(stream, httpcore.NetworkStream), stream
        client_port: Final = _ADDRESS.validate_python(stream.get_extra_info("client_addr"))[1]
        serving: Final = tuple(pid for pid in workers if _accepted(pid, _proxy_port(owned), client_port))
    return (serving[0], _model_ids(response.text)) if len(serving) == 1 else None


def _listing_by_worker(owned: OwnedProxy, workers: tuple[int, ...]) -> Mapping[int, tuple[str, ...]]:
    samples: Final = tuple(_listing_sample(owned, workers) for _ in range(8))
    return MappingProxyType(dict(sample for sample in samples if sample is not None))


def _login_names_listed(listed: tuple[str, ...]) -> frozenset[str]:
    return frozenset(name for name in listed if name in _FIXED or name.startswith(_LOGIN_PREFIXES))


def _lists_login_models(listed: tuple[str, ...]) -> bool:
    login_listed: Final = _login_names_listed(listed)
    return frozenset(_FIXED) <= login_listed and all(
        pattern in login_listed or example in login_listed for pattern, example in _WILDCARDS.items()
    )


def _every_worker_lists(
    owned: OwnedProxy, workers: tuple[int, ...], *, login_models: bool
) -> Mapping[int, tuple[str, ...]]:
    def settled(by_worker: Mapping[int, tuple[str, ...]]) -> bool:
        if set(by_worker) != set(workers):
            return False
        if login_models:
            return all(_lists_login_models(listed) for listed in by_worker.values())
        return all(_login_names_listed(listed) == frozenset() for listed in by_worker.values())

    return eventually(lambda: _listing_by_worker(owned, workers), settled, seconds=90)


def _successes_logged(key: str, at_least: int) -> None:
    eventually(
        lambda: read_rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND status = %s',
            (hashlib.sha256(key.encode()).hexdigest(), "success"),
        ),
        lambda rows: len(rows) >= at_least,
        seconds=40,
    )


def _logged(served: _Served, status: str) -> None:
    rows: Final = eventually(
        lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (served.call_id,)),
        lambda found: len(found) >= 1,
        seconds=40,
    )
    assert [str(row["status"]) for row in rows] == [status], rows


def _reload(owned: OwnedProxy) -> None:
    response: Final = owned.gateway.request("POST", "/reload/model_cost_map", None)
    assert response.status_code == 200, response.text


def _refusal_before_the_reload(name: str, universal: bool) -> str:
    if name in _FIXED:
        return _NO_HEALTHY
    if not universal:
        return _NOT_FOUND
    return dl.CHATGPT_REFUSAL if name.startswith("chatgpt/") else dl.COPILOT_REFUSAL


@pytest.mark.parametrize("universal", _SHAPES.values(), ids=_SHAPES.keys())
def test_booting_without_tokens_drops_the_login_deployments_and_a_reload_after_mounting_brings_them_back(
    gateway: Gateway, tmp_path: Path, universal: bool
) -> None:
    dirs: Final = _token_dirs(tmp_path)
    with dl.device_login_peers(tmp_path) as peers:
        peers.switches.grant.set()
        with owned_proxy_process(
            gateway,
            tmp_path,
            {
                **peers.environment(),
                **dirs.environment(),
                "REQUEST_TIMEOUT": str(_REQUEST_TIMEOUT_SECONDS),
                "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": str(_POLL_SECONDS),
            },
            config=_config(peers.api.url, tmp_path, fixed=True, wildcards=True, universal=universal),
            workers=2,
        ) as owned:
            workers: Final = _workers(owned)
            boot_log: Final = owned.log.read_text()
            assert peers.auth_connections() == ()
            assert dl.CHATGPT_USER_CODE not in boot_log and dl.COPILOT_USER_CODE not in boot_log, boot_log[-3000:]
            assert sorted(_DROPPED.findall(boot_log)) == sorted([*_FIXED.values(), *_WILDCARDS] * 2), boot_log[-3000:]
            _every_worker_lists(owned, workers, login_models=False)
            with owned.gateway.scenario() as scenario:
                key: Final = scenario.key()
                for name in _LOGIN_NAMES:
                    refused: Final = _chat(_base_url(owned), key, name, uuid.uuid4().hex)
                    assert refused.status == 400, refused.text
                    assert _refusal_before_the_reload(name, universal) in refused.message(), refused.text
                control: Final = _chat(_base_url(owned), key, _CONTROL, uuid.uuid4().hex)
                assert control.status == 200, control.text
                assert peers.auth_connections() == ()
                assert dl.bearer(peers.api.drain()[-1]) == dl.CONTROL_KEY

                dirs.store_valid_tokens(peers.api.url)
                _reload(owned)
                _every_worker_lists(owned, workers, login_models=True)
                for served_logins, name in enumerate(_LOGIN_NAMES, start=1):
                    served: Final = _chat(_base_url(owned), key, name, uuid.uuid4().hex)
                    assert served.status == 200, served.text
                    _successes_logged(key, at_least=_CONTROL_CALLS_BEFORE_THE_RELOAD + served_logins)
                assert peers.auth_connections() == ()
                forwarded: Final = peers.api.drain()
                assert {dl.bearer(request) for request in forwarded} == {dl.CHATGPT_STORED, dl.COPILOT_STORED_KEY}

                dl.write_chatgpt(dirs.chatgpt, dl.chatgpt_record(dl.CHATGPT_EXPIRED, refresh_token=dl.GOOD_REFRESH))
                peers.switches.hold_connect.set()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending: Final = pool.submit(_chat, _base_url(owned), key, "chatgpt-fixed", uuid.uuid4().hex, 120)
                    try:
                        dropped: Final = eventually(peers.dropped, lambda hangups: len(hangups) >= 1, seconds=40)
                    finally:
                        peers.switches.refuse_connect.set()
                        peers.switches.hold_connect.clear()
                    timed_out: Final = pending.result(timeout=120)
                assert dropped[0].authority == f"{dl.CHATGPT_AUTH_HOST}:443", dropped
                assert _REQUEST_TIMEOUT_SECONDS - 2 <= dropped[0].seconds <= _REQUEST_TIMEOUT_SECONDS + 6, dropped
                assert timed_out.status == 400, timed_out.text
                assert dl.CHATGPT_REFUSAL in timed_out.message(), timed_out.text
                _logged(timed_out, "failure")
                peers.switches.refuse_connect.clear()
                assert set(peers.auth_connections()) <= {f"{dl.CHATGPT_AUTH_HOST}:443"}
                assert peers.dropped() == ()

                marker: Final = uuid.uuid4().hex
                recovered: Final = _chat(_base_url(owned), key, "chatgpt-fixed", marker)
                assert recovered.status == 200, recovered.text
                assert set(rv.MARKER.findall(recovered.text)) == {marker}, recovered.text
                refreshes: Final = [(request.method, request.target) for request in peers.auth.drain()]
                assert refreshes == [("POST", "/oauth/token")], refreshes
                assert dl.bearer(peers.api.drain()[-1]) == dl.CHATGPT_REFRESHED


@pytest.mark.parametrize("shape", ("fixed", "wildcards"))
def test_a_token_removed_after_boot_turns_every_request_into_the_refusal(
    gateway: Gateway, tmp_path: Path, shape: str
) -> None:
    dirs: Final = _token_dirs(tmp_path)
    with dl.device_login_peers(tmp_path) as peers:
        dirs.store_valid_tokens(peers.api.url)
        with owned_proxy_process(
            gateway,
            tmp_path,
            {**peers.environment(), **dirs.environment()},
            config=_config(peers.api.url, tmp_path, fixed=shape == "fixed", wildcards=shape == "wildcards"),
            workers=2,
        ) as owned:
            _workers(owned)
            names: Final = tuple(_FIXED) if shape == "fixed" else tuple(_FIXED.values())
            with owned.gateway.scenario() as scenario:
                key: Final = scenario.key()
                for name in names:
                    served: Final = _chat(_base_url(owned), key, name, uuid.uuid4().hex)
                    assert served.status == 200, served.text
                dl.clear_tokens(dirs.chatgpt, dirs.copilot)
                peers.switches.grant.set()
                for name in names:
                    refusal: Final = dl.CHATGPT_REFUSAL if name.startswith("chatgpt") else dl.COPILOT_REFUSAL
                    for _ in range(4):
                        refused: Final = _chat(_base_url(owned), key, name, uuid.uuid4().hex)
                        assert refused.status == 400, refused.text
                        assert refusal in refused.message(), refused.text
                assert peers.auth_connections() == ()
                tail: Final = owned.log.read_text()
                assert dl.CHATGPT_USER_CODE not in tail and dl.COPILOT_USER_CODE not in tail, tail[-3000:]
                control: Final = _chat(_base_url(owned), key, _CONTROL, uuid.uuid4().hex)
                assert control.status == 200, control.text


async def _send(client: httpx.AsyncClient, key: str, model: str, marker: str) -> _Served | None:
    try:
        response: Final = await client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": f"Reply to marker-{marker}"}], **_EXTRA},
            headers={"Authorization": f"Bearer {key}"},
        )
    except httpx.TransportError:
        return None
    return _Served(response.status_code, response.text, response.headers.get(_CALL_ID, ""))


async def _burst(base_url: str, key: str, models: tuple[str, ...]) -> tuple[_Served | None, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=120, trust_env=False) as client:
        return tuple(await asyncio.gather(*(_send(client, key, model, uuid.uuid4().hex) for model in models)))


def _open_upstream_connections(pid: int, port: int) -> int:
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


_Bursts: TypeAlias = tuple[asyncio.Task[tuple[_Served | None, ...]], ...]
_HELD_PER_BURST: Final = 8
_HELD_PER_WORKER: Final = 4
_MOST_BURSTS: Final = 8


async def _held_on_every_worker(
    owned: OwnedProxy,
    key: str,
    workers: tuple[int, ...],
    control_port: int,
    held_markers: SimpleQueue[str],
    bursts: _Bursts,
) -> tuple[_Bursts, Mapping[int, int]]:
    started: Final = (*bursts, asyncio.create_task(_burst(_base_url(owned), key, (_CONTROL,) * _HELD_PER_BURST)))
    expected: Final = _HELD_PER_BURST * len(started)
    await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == expected, 60)
    held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, control_port) for pid in workers})
    assert sum(held_by.values()) == expected, held_by
    if min(held_by.values()) >= _HELD_PER_WORKER:
        return started, held_by
    assert len(started) < _MOST_BURSTS, held_by
    return await _held_on_every_worker(owned, key, workers, control_port, held_markers, started)


async def test_worker_sigkill_mid_burst_leaves_the_sibling_refusing_logins_and_serving_the_control(
    gateway: Gateway, tmp_path: Path
) -> None:
    dirs: Final = _token_dirs(tmp_path)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()
    vendor: Final = rv.ResponsesVendor()

    def held(request: Request) -> Reply:
        if request.method == "GET":
            return vendor.respond(request)
        held_markers.put(rv.newest_marker(request.body.decode()) or "")
        assert release.wait(timeout=120), "The burst was never released"
        return vendor.respond(request)

    with (
        dl.device_login_peers(tmp_path) as peers,
        wire_server(held) as control_api,
        owned_proxy_process(
            gateway,
            tmp_path,
            {**peers.environment(), **dirs.environment()},
            config=_config(control_api.url, tmp_path, fixed=False, wildcards=False, universal=True),
            workers=2,
        ) as owned,
    ):
        workers: Final = _workers(owned)
        control_port: Final = int(control_api.url.rsplit(":", 1)[1])
        with owned.gateway.scenario() as scenario:
            key: Final = scenario.key()
            bursts, held_by = await _held_on_every_worker(owned, key, workers, control_port, held_markers, ())
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            answers: Final = await asyncio.gather(*bursts)
            served: Final = tuple(item for item in itertools.chain.from_iterable(answers) if item is not None)
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                assert item.status == 200, item.text
            refusals: Final = await _burst(
                _base_url(owned), key, ("chatgpt/gpt-5.6-terra", "github_copilot/gpt-5.2") * 10
            )
            for index, answer in enumerate(refusals):
                assert answer is not None and answer.status == 400, answer
                refusal: Final = dl.CHATGPT_REFUSAL if index % 2 == 0 else dl.COPILOT_REFUSAL
                assert refusal in answer.message(), answer.text
            assert peers.auth_connections() == ()
            follow_up: Final = _chat(_base_url(owned), key, _CONTROL, uuid.uuid4().hex)
            assert follow_up.status == 200, follow_up.text
