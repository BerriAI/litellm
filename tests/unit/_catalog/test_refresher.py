import contextlib
import json
import os
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm._catalog import Catalog, Remote

NEW_KEY: Final = "refresher/test-model"
REPO: Final = Path(litellm.__file__).resolve().parents[1]


def _remote_body() -> bytes:
    catalog: Final = json.loads(REPO.joinpath("model_prices_and_context_window.json").read_text())
    return json.dumps(
        {**catalog, NEW_KEY: {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 1e-6}}
    ).encode()


@contextlib.contextmanager
def _held_server(body: bytes) -> Iterator[tuple[str, threading.Event]]:
    """Serves ``body`` once the returned event is set, so a test controls when the fetch completes"""
    release: Final = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            release.wait(timeout=30)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    httpd: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/model_prices.json", release
    finally:
        release.set()
        httpd.shutdown()
        httpd.server_close()


def _eventually(condition: Callable[[], bool], timeout: float = 20.0) -> bool:
    deadline: Final = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def _bound_catalog() -> tuple[types.ModuleType, Catalog]:
    module: Final = types.ModuleType("fake_litellm")
    module.model_cost = {}
    catalog: Final = Catalog.bound_to(module, "model_cost")
    catalog.install(catalog.bundled())
    return module, catalog


def test_a_started_refresh_installs_the_remote_map_into_the_bound_dict() -> None:
    module, catalog = _bound_catalog()
    bound_before: Final = module.model_cost
    with _held_server(_remote_body()) as (url, release), catalog.refresher(Remote(url)):
        assert NEW_KEY not in module.model_cost
        release.set()
        assert _eventually(lambda: NEW_KEY in module.model_cost)

    assert module.model_cost is bound_before
    provenance: Final = catalog.provenance()
    assert provenance is not None and provenance.source == "remote"


def test_a_refresh_that_lands_after_an_explicit_install_is_dropped() -> None:
    module, catalog = _bound_catalog()
    with _held_server(_remote_body()) as (url, release), catalog.refresher(Remote(url)) as refresher:
        catalog.install(catalog.bundled())
        release.set()
        time.sleep(2)
        refresher.stop()

    assert NEW_KEY not in module.model_cost


def test_stop_cancels_a_held_fetch_without_waiting_for_it() -> None:
    module, catalog = _bound_catalog()
    with _held_server(_remote_body()) as (url, release):
        refresher: Final = catalog.refresher(Remote(url))
        refresher.start()
        started: Final = time.monotonic()
        refresher.stop()
        stopped_after: Final = time.monotonic() - started
        release.set()
        time.sleep(1)

    assert stopped_after < 2
    assert NEW_KEY not in module.model_cost


def test_import_serves_bundled_catalog_before_held_response_then_adopts_remote() -> None:
    """Decision 4. Import returns while the remote map is held, and the remote rows land after it"""
    script: Final = """
import sys, time
started = time.monotonic()
import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info
print(time.monotonic() - started, flush=True)
print(get_model_cost_map_source_info()["fallback_reason"], sys.argv[1] in litellm.model_cost, flush=True)
sys.stdin.readline()
deadline = time.monotonic() + 20
while sys.argv[1] not in litellm.model_cost and time.monotonic() < deadline:
    time.sleep(0.05)
print(get_model_cost_map_source_info()["source"], sys.argv[1] in litellm.model_cost, flush=True)
"""
    with _held_server(_remote_body()) as (url, release):
        env: Final = {
            **os.environ,
            "LITELLM_MODEL_COST_MAP_URL": url,
            "LITELLM_LOCAL_MODEL_COST_MAP": "",
            "PYTHONPATH": str(REPO),
        }
        child: Final = subprocess.Popen(
            [sys.executable, "-P", "-c", script, NEW_KEY],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert child.stdout is not None and child.stdin is not None
        import_seconds: Final = float(child.stdout.readline())
        before: Final = child.stdout.readline().split()
        release.set()
        child.stdin.write("\n")
        child.stdin.flush()
        after: Final = child.stdout.readline().split()
        child.wait(timeout=30)

    assert import_seconds < 30
    assert before == ["remote", "revalidation", "pending", "False"]
    assert after == ["remote", "True"]


@pytest.mark.parametrize("variable", ["LITELLM_LOCAL_MODEL_COST_MAP"])
def test_a_forced_local_map_never_starts_a_refresh(monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    from litellm.litellm_core_utils.get_model_cost_map import start_model_cost_map_refresh

    monkeypatch.setenv(variable, "True")
    _, catalog = _bound_catalog()

    assert start_model_cost_map_refresh("http://127.0.0.1:9/never", catalog) is None


def test_a_loop_over_the_map_sees_one_size_change_while_a_refresh_lands() -> None:
    """Decision 17. The install writes into the bound dict, so a loop over it at that moment raises once"""
    module, catalog = _bound_catalog()
    size_changes = 0
    with _held_server(_remote_body()) as (url, release), catalog.refresher(Remote(url)):
        release.set()
        deadline: Final = time.monotonic() + 20
        while NEW_KEY not in module.model_cost and time.monotonic() < deadline:
            try:
                for _ in module.model_cost:
                    time.sleep(0)
            except RuntimeError as error:
                assert "changed size during iteration" in str(error)
                size_changes += 1

    assert NEW_KEY in module.model_cost
    assert size_changes == 1


def test_an_s3_source_that_fails_transiently_is_retried_in_the_background(monkeypatch: pytest.MonkeyPatch) -> None:
    """On main a file or S3 source that failed on its first read stayed on the bundled map until the next reload"""
    body: Final = _remote_body()
    paths: Final[list[str]] = []  # mutable-ok: the requests the fake S3 endpoint saw

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            paths.append(self.path)
            if len(paths) == 1:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    httpd: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", f"http://127.0.0.1:{httpd.server_address[1]}")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    module, catalog = _bound_catalog()
    try:
        with catalog.refresher(Remote("s3://cost-maps/model_prices.json")):
            assert _eventually(lambda: NEW_KEY in module.model_cost)
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert len(paths) == 2
    assert all(path.split("?")[0] == "/cost-maps/model_prices.json" for path in paths)
    provenance: Final = catalog.provenance()
    assert provenance is not None and provenance.source == "remote"
