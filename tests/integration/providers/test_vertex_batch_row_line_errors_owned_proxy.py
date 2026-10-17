import json
import threading
from itertools import chain
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Request, wire_server
from integration._support.ports import allocate_port
from pydantic import JsonValue, TypeAdapter

from integration.providers._vertex_batch_rows import (
    Route,
    chat_row,
    encoded,
    error_of,
    gcs_peer,
    held_gcs_peer,
    is_openai_upload,
    jsonl,
    openai_files_peer,
    openai_uploaded_lines,
    roleless_row,
    stored_custom_ids,
    unconvertible_line_message,
    upload,
    vertex_params,
)

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

VERTEX: Final = "lb-vertex"
OPENAI: Final = "lb-openai"
_JSON: Final = TypeAdapter(dict[str, JsonValue])
UploadRoute = Route | Literal["load_balanced"]
UPLOAD_ROUTES: Final[tuple[UploadRoute, ...]] = ("load_balanced", "model", "target_model_names")


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: OwnedProxy
    vertex_port: int
    openai_port: int

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway


def _config(gateway: Gateway, directory: Path, vertex_port: int, openai_port: int) -> Path:
    base: Final = _JSON.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final[dict[str, JsonValue]] = {
        **base,
        "model_list": [
            {"model_name": VERTEX, "litellm_params": vertex_params(gateway, f"http://127.0.0.1:{vertex_port}")},
            {
                "model_name": OPENAI,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "integration-provider-key",
                    "api_base": f"http://127.0.0.1:{openai_port}/v1",
                },
            },
        ],
        "litellm_settings": {
            **_JSON.validate_python(base["litellm_settings"]),
            "enable_loadbalancing_on_batch_endpoints": True,
        },
        "guardrails": [
            {
                "guardrail_name": "tripwire-filter",
                "litellm_params": {
                    "guardrail": "litellm_content_filter",
                    "mode": "pre_call",
                    "default_on": True,
                    "blocked_words": [{"keyword": "tripwire", "action": "BLOCK"}],
                },
            }
        ],
    }
    path: Final = directory / "vertex-batch-row-lines.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("vertex-batch-row-lines")
    vertex_port: Final = allocate_port()
    openai_port: Final = allocate_port()
    with gateway_from_environment() as shared:
        config: Final = _config(shared, directory, vertex_port, openai_port)
        with owned_proxy_process(shared, directory, {}, config=config, workers=2) as owned:
            yield _Rig(owned, vertex_port, openai_port)


def _upload(gateway: Gateway, route: UploadRoute, model: str, content: bytes) -> httpx.Response:
    return upload(gateway, None if route == "load_balanced" else route, model, content)


def _tripwire_row(custom_id: str, model: str) -> bytes:
    return encoded(chat_row(custom_id, model, "this row trips the tripwire filter"))


LAYOUTS: Final[tuple[tuple[str, int], ...]] = (("dropped_row_first", 1), ("dropped_row_second", 2))


def _layout(name: str, model: str) -> bytes:
    roleless: Final = encoded(roleless_row("bad", model))
    match name:
        case "dropped_row_first":
            return jsonl(_tripwire_row("dropped", model), b"", roleless)
        case "dropped_row_second":
            return jsonl(encoded(chat_row("first", model, "one")), _tripwire_row("dropped", model), b"", roleless)
        case _:
            raise AssertionError(name)


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_unconvertible_row_after_a_blank_line_names_its_file_line(rig: _Rig, route: UploadRoute) -> None:
    content: Final = jsonl(
        encoded(chat_row("first", VERTEX, "one")),
        b"",
        encoded(chat_row("third", VERTEX, "three")),
        encoded(roleless_row("bad", VERTEX)),
    )
    with wire_server(gcs_peer, port=rig.vertex_port) as wire:
        response: Final = _upload(rig.gateway, route, VERTEX, content)
        assert response.status_code == 400, response.text
        error: Final = error_of(response)
        assert str(error["message"]).startswith(unconvertible_line_message(4) + "missing field 'role'"), error
        assert error["type"] == "invalid_request_error", error
        assert wire.drain() == ()


@pytest.mark.parametrize(("layout", "bad_line"), LAYOUTS, ids=tuple(name for name, _ in LAYOUTS))
@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_row_after_a_guardrail_dropped_row_names_its_line_in_the_uploaded_file(
    rig: _Rig, route: UploadRoute, layout: str, bad_line: int
) -> None:
    with wire_server(gcs_peer, port=rig.vertex_port) as wire:
        response: Final = _upload(rig.gateway, route, VERTEX, _layout(layout, VERTEX))
        assert response.status_code == 400, response.text
        message: Final = str(error_of(response)["message"])
        assert message.startswith(unconvertible_line_message(bad_line) + "missing field 'role'"), message
        assert wire.drain() == ()


def _kept_and_dropped(model: str) -> bytes:
    return jsonl(
        encoded(chat_row("first", model, "one")),
        _tripwire_row("dropped", model),
        b"",
        encoded(chat_row("fourth", model, "four")),
    )


def _dropped_first(model: str) -> bytes:
    return jsonl(
        _tripwire_row("dropped", model),
        b"",
        encoded(chat_row("first", model, "one")),
        encoded(chat_row("fourth", model, "four")),
    )


@pytest.mark.parametrize("content", (_kept_and_dropped, _dropped_first), ids=("dropped_row_second", "dropped_row_first"))
@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_guardrail_dropped_and_blank_rows_never_reach_openai(
    rig: _Rig, route: UploadRoute, content: Callable[[str], bytes]
) -> None:
    with wire_server(openai_files_peer, port=rig.openai_port) as wire:
        response: Final = _upload(rig.gateway, route, OPENAI, content(OPENAI))
        assert response.status_code == 200, response.text
        uploads: Final = tuple(request for request in wire.drain() if is_openai_upload(request))
        assert len(uploads) == 1, uploads
        sent_model: Final = OPENAI if route == "model" else "gpt-4o-mini"
        rows: Final = tuple(json.loads(line) for line in openai_uploaded_lines(uploads[0]))
        assert [(row["custom_id"], row["body"]["model"]) for row in rows] == [
            ("first", sent_model),
            ("fourth", sent_model),
        ]


@pytest.mark.parametrize("route", UPLOAD_ROUTES)
def test_guardrail_dropped_row_never_reaches_the_bucket(rig: _Rig, route: UploadRoute) -> None:
    with wire_server(gcs_peer, port=rig.vertex_port) as wire:
        response: Final = _upload(rig.gateway, route, VERTEX, _kept_and_dropped(VERTEX))
        assert response.status_code == 200, response.text
        uploads: Final = wire.drain()
        assert len(uploads) == 1, uploads
        assert stored_custom_ids(uploads[0]) == ("first", "fourth")


def _bad_file(line: int) -> bytes:
    good: Final = tuple(encoded(chat_row(f"row-{line}-{index}", VERTEX, "fine")) for index in range(1, line))
    return jsonl(*good, encoded(roleless_row(f"bad-{line}", VERTEX)))


def _guarded_upload(client: httpx.Client, gateway: Gateway, content: bytes) -> httpx.Response | None:
    try:
        return upload(gateway, "model", VERTEX, content, client=client)
    except httpx.TransportError:
        return None


def _landed_ids(requests: tuple[Request, ...]) -> tuple[str, ...]:
    return tuple(chain.from_iterable(stored_custom_ids(request) for request in requests))


def _held_connections(pid: int, url: str) -> int:
    port: Final = urlsplit(url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _upload_bad_file(client: httpx.Client, gateway: Gateway, line: int) -> httpx.Response | None:
    return _guarded_upload(client, gateway, _bad_file(line))


def _assert_names_its_line(response: httpx.Response | None, line: int) -> None:
    assert response is not None
    assert response.status_code == 400, response.text
    assert str(error_of(response)["message"]).startswith(unconvertible_line_message(line)), response.text


@pytest.mark.timeout(4 * graceful_stop_seconds() + 240)
def test_worker_killed_mid_upload_leaves_the_sibling_naming_lines_and_landing_each_file_once(rig: _Rig) -> None:
    bad_lines: Final = tuple(range(2, 10))
    held_ids: Final = tuple(f"held-{index}" for index in range(8))
    release: Final = threading.Event()
    with (
        httpx.Client(
            base_url=str(rig.gateway.client.base_url),
            timeout=180,
            trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as client,
        ThreadPoolExecutor(max_workers=16) as pool,
    ):
        with wire_server(gcs_peer, port=rig.vertex_port) as wire:
            answers: Final = tuple(
                pool.map(partial(_upload_bad_file, client, rig.gateway), bad_lines)
            )
            for line, answer in zip(bad_lines, answers, strict=True):
                _assert_names_its_line(answer, line)
            assert wire.drain() == ()
        with wire_server(held_gcs_peer(release), port=rig.vertex_port) as held:
            try:
                workers: Final = eventually(rig.proxy.worker_pids, lambda pids: len(pids) == 2, seconds=30)
                pending: Final = tuple(
                    pool.submit(_guarded_upload, client, rig.gateway, jsonl(encoded(chat_row(identity, VERTEX, "x"))))
                    for identity in held_ids
                )
                eventually(held.received.qsize, lambda size: size == len(held_ids), seconds=60)
                held_by: Final = MappingProxyType({pid: _held_connections(pid, held.url) for pid in workers})
                assert sum(held_by.values()) == len(held_ids), held_by
                survivor_pid, victim_pid = sorted(workers, key=held_by.__getitem__)
                with rig.proxy.replacing_worker(victim_pid) as kill:
                    kill.freeze()
                    kill()
                    release.set()
                    outcomes: Final = tuple(future.result(timeout=120) for future in pending)
                served: Final = tuple(outcome for outcome in outcomes if outcome is not None)
                assert len(served) == held_by[survivor_pid], (held_by, [outcome and outcome.text for outcome in outcomes])
                assert all(outcome.status_code == 200 for outcome in served), [outcome.text for outcome in served]
                landed: Final = _landed_ids(held.drain())
                assert sorted(landed) == sorted(held_ids)
                _assert_names_its_line(_guarded_upload(client, rig.gateway, _bad_file(5)), 5)
                after: Final = _guarded_upload(client, rig.gateway, jsonl(encoded(chat_row("after", VERTEX, "x"))))
                assert after is not None and after.status_code == 200, after and after.text
                assert _landed_ids(held.drain()) == ("after",)
            finally:
                release.set()
