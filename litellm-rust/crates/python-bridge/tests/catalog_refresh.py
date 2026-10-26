from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import queue
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final
from unittest.mock import patch

import litellm
from litellm.rust_bridge import _native as native


class Response:
    def __init__(
        self, status: int = 200, body: bytes | None = None, hold: bool = False, retry_after: str | None = "0"
    ) -> None:
        self.status = status
        self.body = native.ModelRegistry.bundled_bytes() if body is None else body
        self.retry_after = retry_after
        self.requests: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self.request_times: queue.Queue[float] = queue.Queue()
        self.entered = threading.Event()
        self.release = threading.Event()
        if not hold:
            self.release.set()


@contextmanager
def server(
    response: Response, tls: ssl.SSLContext | None = None, *, following: tuple[Response, ...] = ()
) -> Iterator[str]:
    responses: Final = iter((response, *following))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            current: Final = next(responses, response)
            response.requests.put((self.path, self.headers.get("If-None-Match")))
            response.request_times.put(time.monotonic())
            current.entered.set()
            current.release.wait(15)
            self.send_response(current.status)
            self.send_header("ETag", '"catalog-v1"')
            if current.retry_after is not None:
                self.send_header("Retry-After", current.retry_after)
            self.send_header("Content-Length", str(len(current.body) if current.status == 200 else 0))
            self.end_headers()
            if current.status == 200:
                try:
                    self.wfile.write(current.body)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                    pass

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http:
        if tls is not None:
            http.socket = tls.wrap_socket(http.socket, server_side=True)
        worker: Final = threading.Thread(target=http.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        try:
            yield f"{'https' if tls else 'http'}://127.0.0.1:{http.server_port}"
        finally:
            response.release.set()
            for pending in following:
                pending.release.set()
            http.shutdown()
            worker.join(5)


def refresh_threads() -> frozenset[str]:
    return frozenset(path.name for path in Path("/proc/self/task").iterdir() if catalog_thread(path))


def catalog_thread(path: Path) -> bool:
    try:
        return (path / "comm").read_text().startswith("litellm-catalog")
    except FileNotFoundError:
        return False


def wait_for(predicate: Callable[[], bool], timeout: float = 5) -> None:
    deadline: Final = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not complete")
        threading.Event().wait(0.01)


class Observer:
    def __init__(self, accept: bool = True) -> None:
        self.value = 7
        self.accept = accept
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.provenance: tuple[str | None, str | None] = (None, None)

    def generation(self) -> int:
        return self.value

    def install(self, seen: int, registry: native.ModelRegistry, errors: list[native.RowError]) -> int | None:
        if self.accept and seen == self.value:
            self.provenance = (registry.source_revision, registry.source_validator)
            self.value += 1
            self.events.put(("install", (seen, registry, errors)))
            return self.value
        self.events.put(("install", (seen, registry, errors)))
        return None

    def checked(self, reason: str | None) -> None:
        self.events.put(("checked", reason))

    def start(self, url: str, validator: str | None = None, revision: str | None = None) -> native.CatalogRefresher:
        return native.CatalogRefresher(
            url,
            installed_revision=revision,
            installed_validator=validator,
            install=self.install,
            checked=self.checked,
            generation=self.generation,
        )


class CatalogRefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(os.environ, {"NO_PROXY": "*", "no_proxy": "*"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.ssl_verify = litellm.ssl_verify
        self.addCleanup(setattr, litellm, "ssl_verify", self.ssl_verify)
        self.threads = refresh_threads()
        self.assertFalse(native.process_state_started())

    def tearDown(self) -> None:
        wait_for(lambda: refresh_threads() == self.threads)
        self.assertFalse(native.process_state_started())

    def fetch(self, url: str, validator: str | None = None, timeout: float = 3) -> object:
        return native.fetch_once(url, if_none_match=validator, timeout_seconds=timeout)

    def retry_fetch(
        self,
        url: str,
        *,
        validator: str | None = None,
        timeout: float = 3,
        attempts: int = 3,
        max_wait: float = 0.05,
    ) -> native.FetchedRegistry | native.RegistryNotModified | native.RegistryUnavailable:
        return native.fetch_with_retry(
            url,
            if_none_match=validator,
            timeout_seconds=timeout,
            max_attempts=attempts,
            max_wait_seconds=max_wait,
        )

    def test_retry_success_preserves_provenance_row_errors_and_process_state(self) -> None:
        body: Final = json.dumps(
            {
                **json.loads(native.ModelRegistry.bundled_bytes()),
                "retry-invalid": {"litellm_provider": "openai", "input_cost_per_token": -1},
            }
        ).encode()
        response: Final = Response(503)
        original: Final = litellm.model_cost
        original_rows: Final = json.dumps(original, sort_keys=True)
        with server(response, following=(Response(body=body),)) as url:
            result: Final = self.retry_fetch(url, validator='"old"')
        assert isinstance(result, native.FetchedRegistry)
        self.assertEqual(response.requests.get_nowait(), ("/", '"old"'))
        self.assertEqual(response.requests.get_nowait(), ("/", '"old"'))
        self.assertTrue(response.requests.empty())
        self.assertEqual(result.revision, hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest())
        self.assertEqual(result.validator, '"catalog-v1"')
        self.assertEqual(result.registry.source_revision, result.revision)
        self.assertEqual(result.registry.source_validator, result.validator)
        self.assertIn("retry-invalid", result.registry.to_dict())
        self.assertEqual(
            tuple((error.key, error.field, error.rejects) for error in result.registry.errors()),
            (("retry-invalid", "input_cost_per_token", True),),
        )
        filtered, errors = result.registry.without_rejected()
        self.assertNotIn("retry-invalid", filtered.to_dict())
        self.assertEqual(tuple(error.key for error in errors), ("retry-invalid",))
        self.assertEqual(filtered.source_revision, result.revision)
        self.assertEqual(filtered.source_validator, result.validator)
        self.assertIs(litellm.model_cost, original)
        self.assertEqual(json.dumps(litellm.model_cost, sort_keys=True), original_rows)

    def test_retry_exhaustion_is_capped(self) -> None:
        response: Final = Response(503, retry_after="100")
        with server(response) as url:
            result: Final = self.retry_fetch(url, attempts=3, max_wait=0)
        assert isinstance(result, native.RegistryUnavailable)
        self.assertEqual(response.requests.qsize(), 3)
        self.assertTrue(result.retryable)
        self.assertEqual(result.retry_after, 100)
        self.assertIn("503", result.reason)

    def test_retry_terminal_response_does_not_retry(self) -> None:
        response: Final = Response(503)
        with server(response, following=(Response(404),)) as url:
            result: Final = self.retry_fetch(url, attempts=4)
        assert isinstance(result, native.RegistryUnavailable)
        self.assertEqual(response.requests.qsize(), 2)
        self.assertFalse(result.retryable)
        self.assertIsNone(result.retry_after)
        self.assertIn("404", result.reason)

    def test_retry_after_is_honored(self) -> None:
        response: Final = Response(429, retry_after="0.15")
        with server(response, following=(Response(304),)) as url:
            result: Final = self.retry_fetch(url, max_wait=1)
        self.assertIsInstance(result, native.RegistryNotModified)
        first, second = response.request_times.get_nowait(), response.request_times.get_nowait()
        self.assertGreaterEqual(second - first, 0.15)
        self.assertLess(second - first, 1)
        self.assertEqual(response.requests.qsize(), 2)

    def test_retry_after_is_capped(self) -> None:
        response: Final = Response(503, retry_after="100")
        with server(response, following=(Response(304),)) as url:
            result: Final = self.retry_fetch(url, max_wait=0.05)
        self.assertIsInstance(result, native.RegistryNotModified)
        first, second = response.request_times.get_nowait(), response.request_times.get_nowait()
        self.assertGreaterEqual(second - first, 0.05)
        self.assertLess(second - first, 1)
        self.assertEqual(response.requests.qsize(), 2)

    def test_retry_backoff_without_header_is_capped(self) -> None:
        response: Final = Response(503, retry_after=None)
        with server(response, following=(Response(304),)) as url:
            result: Final = self.retry_fetch(url, max_wait=0.05)
        self.assertIsInstance(result, native.RegistryNotModified)
        first, second = response.request_times.get_nowait(), response.request_times.get_nowait()
        self.assertGreaterEqual(second - first, 0.05)
        self.assertLess(second - first, 1)
        self.assertEqual(response.requests.qsize(), 2)

    def test_retry_conditional_not_modified_is_one_request(self) -> None:
        response: Final = Response(304)
        with server(response) as url:
            result: Final = self.retry_fetch(url, validator='"old"')
        self.assertIsInstance(result, native.RegistryNotModified)
        self.assertEqual(response.requests.get_nowait(), ("/", '"old"'))
        self.assertTrue(response.requests.empty())

    def assert_rejected_body(self, body: bytes) -> None:
        response: Final = Response(body=body)
        with server(response) as url:
            result: Final = self.retry_fetch(url)
        assert isinstance(result, native.RegistryUnavailable)
        self.assertFalse(result.retryable)
        self.assertIsNone(result.retry_after)
        self.assertTrue(result.reason)
        self.assertEqual(response.requests.qsize(), 1)

    def test_retry_parse_failure_is_terminal(self) -> None:
        self.assert_rejected_body(b"not-json")

    def test_retry_integrity_failure_is_terminal(self) -> None:
        self.assert_rejected_body(b'{"small": {"litellm_provider": "openai"}}')

    def test_retry_invalid_arguments_do_not_reach_transport(self) -> None:
        response: Final = Response()
        with server(response) as url:
            for value in (-1.0, float("nan"), float("inf"), float("-inf"), 1e100, 1e19):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        self.retry_fetch(url, timeout=value)
                    with self.assertRaises(ValueError):
                        self.retry_fetch(url, max_wait=value)
            for attempts in (0, -1, 2**128):
                with self.subTest(attempts=attempts), self.assertRaises((ValueError, OverflowError)):
                    self.retry_fetch(url, attempts=attempts)
            with self.assertRaises(ValueError):
                self.retry_fetch("ftp://127.0.0.1")
            self.assertTrue(response.requests.empty())

    def test_retry_releases_gil_during_fetch_and_wait(self) -> None:
        response: Final = Response(503, hold=True, retry_after="0.2")
        success: Final = Response(hold=True)
        progress: Final[queue.Queue[str]] = queue.Queue()

        def advance() -> None:
            if not response.entered.wait(5):
                return
            progress.put("fetch")
            response.release.set()
            time.sleep(0.05)
            progress.put("wait" if not success.entered.is_set() else "missed wait")
            success.release.set()

        worker: Final = threading.Thread(target=advance, daemon=True)
        with server(response, following=(success,)) as url:
            worker.start()
            result: Final = self.retry_fetch(url, max_wait=1)
            worker.join(5)
        self.assertIsInstance(result, native.FetchedRegistry)
        self.assertEqual(progress.get_nowait(), "fetch")
        self.assertEqual(progress.get_nowait(), "wait")
        self.assertFalse(worker.is_alive())
        self.assertEqual(response.requests.qsize(), 2)

    def test_retry_timeout_is_per_attempt_and_bounded(self) -> None:
        response: Final = Response(hold=True)
        with server(response) as url:
            result: Final = self.retry_fetch(url, timeout=0.05, attempts=2, max_wait=0)
        assert isinstance(result, native.RegistryUnavailable)
        self.assertTrue(result.retryable)
        self.assertEqual(response.requests.qsize(), 2)

    def test_fetch_installs_nothing_and_preserves_validator_revision(self) -> None:
        response: Final = Response()
        original: Final = litellm.model_cost
        original_rows: Final = dict(original)
        with server(response) as url:
            result: Final = self.fetch(url, '"previous"')
        self.assertIsInstance(result, native.FetchedRegistry)
        assert isinstance(result, native.FetchedRegistry)
        self.assertEqual(result.registry.to_dict(), native.ModelRegistry.bundled().to_dict())
        self.assertEqual(result.validator, '"catalog-v1"')
        self.assertEqual(result.registry.source_revision, result.revision)
        self.assertEqual(result.registry.source_validator, result.validator)
        filtered, _ = result.registry.without_rejected()
        self.assertEqual(filtered.source_revision, result.revision)
        self.assertEqual(filtered.source_validator, result.validator)
        self.assertEqual(
            result.revision,
            hashlib.sha1(b"blob " + str(len(response.body)).encode() + b"\0" + response.body).hexdigest(),
        )
        self.assertEqual(response.requests.get_nowait(), ("/", '"previous"'))
        self.assertTrue(response.requests.empty())
        self.assertIs(litellm.model_cost, original)
        self.assertEqual(litellm.model_cost, original_rows)

    def test_parse_and_bundled_have_no_source_metadata(self) -> None:
        bundled: Final = native.ModelRegistry.bundled()
        parsed: Final = native.ModelRegistry.parse(native.ModelRegistry.bundled_bytes())
        self.assertIsNone(bundled.source_revision)
        self.assertIsNone(bundled.source_validator)
        self.assertIsNone(parsed.source_revision)
        self.assertIsNone(parsed.source_validator)
        self.assertEqual(bundled.to_dict(), parsed.to_dict())
        with self.assertRaises(AttributeError):
            setattr(bundled, "source_revision", "not-writable")

    def test_not_modified(self) -> None:
        response: Final = Response(304)
        with server(response) as url:
            self.assertIsInstance(self.fetch(url, '"old"'), native.RegistryNotModified)
            observer: Final = Observer()
            refresher: Final = observer.start(url, '"old"')
            self.addCleanup(refresher.stop)
            self.assertEqual(observer.events.get(timeout=5), ("checked", None))
            wait_for(lambda: refresh_threads() == self.threads)
            self.assertTrue(observer.events.empty())
        self.assertEqual(response.requests.get_nowait(), ("/", '"old"'))
        self.assertEqual(response.requests.get_nowait(), ("/", '"old"'))

    def test_refresher_filters_rejected_rows_and_reports_them(self) -> None:
        body: Final = json.dumps(
            {
                **json.loads(native.ModelRegistry.bundled_bytes()),
                "p3-invalid": {"litellm_provider": "openai", "input_cost_per_token": -1},
            }
        ).encode()
        response: Final = Response(body=body)
        with server(response) as url:
            observer: Final = Observer()
            refresher: Final = observer.start(url)
            self.addCleanup(refresher.stop)
            kind, payload = observer.events.get(timeout=5)
            self.assertEqual(kind, "install")
            assert isinstance(payload, tuple)
            seen, registry, errors = payload
            self.assertEqual(seen, 7)
            self.assertNotIn("p3-invalid", registry.to_dict())
            self.assertTrue(any(error.key == "p3-invalid" for error in errors))
            self.assertGreater(len(registry.to_dict()), 100)
            self.assertEqual(
                registry.source_revision,
                hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest(),
            )
            self.assertEqual(registry.source_validator, '"catalog-v1"')
            self.assertEqual(observer.provenance, (registry.source_revision, registry.source_validator))
            wait_for(lambda: refresh_threads() == self.threads)
            self.assertTrue(observer.events.empty())
            self.assertEqual(response.requests.qsize(), 1)

    def test_stale_is_logged_once_without_checked_or_retry(self) -> None:
        response: Final = Response()
        with self.assertLogs(litellm.verbose_logger, level=logging.DEBUG) as logs, server(response) as url:
            observer: Final = Observer(accept=False)
            refresher: Final = observer.start(url)
            self.addCleanup(refresher.stop)
            self.assertEqual(observer.events.get(timeout=5)[0], "install")
            wait_for(lambda: refresh_threads() == self.threads)
            self.assertTrue(observer.events.empty())
            self.assertEqual(response.requests.qsize(), 1)
        self.assertTrue(
            any(getattr(record, "rust_fields", {}).get("outcome") == "stale" for record in logs.records), logs.output
        )
        self.assertEqual(observer.provenance, (None, None))

    def test_stop_held_response_has_no_callback_and_exits_thread(self) -> None:
        response: Final = Response(hold=True)
        with server(response) as url:
            observer: Final = Observer()
            refresher: Final = observer.start(url)
            self.addCleanup(refresher.stop)
            self.assertTrue(response.entered.wait(5))
            before: Final = time.monotonic()
            refresher.stop()
            self.assertLess(time.monotonic() - before, 0.5)
            wait_for(lambda: refresh_threads() == self.threads)
            response.release.set()
            refresher.stop()
            self.assertTrue(observer.events.empty())

    def test_new_generation_fences_inflight_result(self) -> None:
        response: Final = Response(hold=True)
        with server(response) as url:
            observer: Final = Observer()
            refresher: Final = observer.start(url)
            self.addCleanup(refresher.stop)
            self.assertTrue(response.entered.wait(5))
            observer.value = 8
            response.release.set()
            wait_for(lambda: refresh_threads() == self.threads)
            self.assertTrue(observer.events.empty())

        self.assertEqual(observer.provenance, (None, None))

    def test_retryable_once_does_not_retry(self) -> None:
        response: Final = Response(503, retry_after="100")
        with server(response) as url:
            result: Final = self.fetch(url)
        self.assertIsInstance(result, native.RegistryUnavailable)
        assert isinstance(result, native.RegistryUnavailable)
        self.assertTrue(result.retryable)
        self.assertEqual(result.retry_after, 100)
        self.assertIn("503", result.reason)
        self.assertEqual(response.requests.qsize(), 1)

    def test_failure_calls_checked_reason_once(self) -> None:
        response: Final = Response(404)
        with server(response) as url:
            observer: Final = Observer()
            refresher: Final = observer.start(url)
            self.addCleanup(refresher.stop)
            kind, reason = observer.events.get(timeout=5)
            self.assertEqual(kind, "checked")
            self.assertIn("404", str(reason))
            wait_for(lambda: refresh_threads() == self.threads)
            self.assertTrue(observer.events.empty())

    def test_file_source_ignores_http_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: Final = Path(directory) / "catalog.json"
            path.write_bytes(native.ModelRegistry.bundled_bytes())
            result: Final = self.fetch(path.as_uri())
            self.assertIsInstance(result, native.FetchedRegistry)
            assert isinstance(result, native.FetchedRegistry)
            self.assertIsNone(result.validator)
            self.assertIsInstance(self.fetch(path.as_uri(), '"old"'), native.FetchedRegistry)
            retried: Final = self.retry_fetch(path.as_uri(), validator='"old"')
            assert isinstance(retried, native.FetchedRegistry)
            self.assertIsNone(retried.validator)
            self.assertEqual(retried.revision, result.revision)
            self.assertEqual(retried.registry.to_dict(), result.registry.to_dict())

    def test_s3_endpoint_and_conditional_header(self) -> None:
        response: Final = Response()
        with (
            server(response) as url,
            patch.dict(
                os.environ,
                {
                    "AWS_ENDPOINT_URL_S3": url,
                    "AWS_ACCESS_KEY_ID": "local-test",
                    "AWS_SECRET_ACCESS_KEY": "local-test",
                    "AWS_REGION": "us-east-1",
                    "AWS_EC2_METADATA_DISABLED": "true",
                },
            ),
        ):
            result: Final = self.fetch("s3://bucket/catalog.json", '"old"')
            retried: Final = self.retry_fetch("s3://bucket/catalog.json", validator='"old"')
        self.assertIsInstance(result, native.FetchedRegistry)
        assert isinstance(result, native.FetchedRegistry)
        self.assertEqual(result.registry.to_dict(), native.ModelRegistry.bundled().to_dict())
        self.assertEqual(response.requests.get_nowait(), ("/bucket/catalog.json?x-id=GetObject", '"old"'))
        self.assertEqual(response.requests.get_nowait(), ("/bucket/catalog.json?x-id=GetObject", '"old"'))
        self.assertTrue(response.requests.empty())
        assert isinstance(retried, native.FetchedRegistry)
        self.assertEqual(retried.registry.to_dict(), result.registry.to_dict())
        self.assertEqual(retried.validator, result.validator)
        self.assertEqual(retried.revision, result.revision)

    def test_unsupported_source_and_invalid_timeout(self) -> None:
        with self.assertRaises(ValueError):
            self.fetch("ftp://x")
        with self.assertRaises(ValueError):
            Observer().start("ftp://x")
        with self.assertRaises(ValueError):
            self.fetch("http://127.0.0.1", timeout=float("nan"))

    def test_timeout_releases_gil_and_performs_one_request(self) -> None:
        response: Final = Response(hold=True)
        with server(response) as url:
            result: Final = self.fetch(url, timeout=0.05)
            self.assertTrue(response.entered.is_set())
            self.assertIsInstance(result, native.RegistryUnavailable)
            self.assertEqual(response.requests.qsize(), 1)

    def test_parse_and_integrity_failure(self) -> None:
        response: Final = Response(body=b'{"small": {"litellm_provider": "openai"}}')
        with server(response) as url:
            result: Final = self.fetch(url)
        self.assertIsInstance(result, native.RegistryUnavailable)
        assert isinstance(result, native.RegistryUnavailable)
        self.assertFalse(result.retryable)
        self.assertTrue(result.reason)

    def test_rejected_document_does_not_advance_provenance(self) -> None:
        response: Final = Response(body=b"not-json")
        observer: Final = Observer()
        with server(response) as url:
            refresher: Final = observer.start(url)
            self.addCleanup(refresher.stop)
            kind, reason = observer.events.get(timeout=5)
            self.assertEqual(kind, "checked")
            self.assertTrue(reason)
            wait_for(lambda: refresh_threads() == self.threads)
            self.assertTrue(observer.events.empty())
            self.assertEqual(observer.provenance, (None, None))

    def test_tls_verify_applies_to_fetch_and_refresher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            key: Final = Path(directory) / "key.pem"
            cert: Final = Path(directory) / "cert.pem"
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-days",
                    "1",
                    "-subj",
                    "/CN=localhost",
                    "-addext",
                    "subjectAltName=IP:127.0.0.1",
                    "-keyout",
                    str(key),
                    "-out",
                    str(cert),
                ],
                check=True,
                capture_output=True,
            )
            context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)
            response: Final = Response()
            with server(response, context) as url:
                litellm.ssl_verify = False
                self.assertIsInstance(self.fetch(url), native.FetchedRegistry)
                self.assertIsInstance(self.retry_fetch(url), native.FetchedRegistry)
                observer: Final = Observer()
                refresher: Final = observer.start(url)
                self.addCleanup(refresher.stop)
                litellm.ssl_verify = True
                self.assertEqual(observer.events.get(timeout=5)[0], "install")
                failed: Final = self.fetch(url)
                self.assertIsInstance(failed, native.RegistryUnavailable)
                assert isinstance(failed, native.RegistryUnavailable)
                self.assertTrue(failed.reason)
                self.assertIsInstance(self.retry_fetch(url, attempts=1), native.RegistryUnavailable)
                failure_observer: Final = Observer()
                failure_refresher: Final = failure_observer.start(url)
                self.addCleanup(failure_refresher.stop)
                kind, reason = failure_observer.events.get(timeout=15)
                self.assertEqual(kind, "checked")
                self.assertTrue(reason)

    def test_callback_exception_is_unraisable_and_terminal(self) -> None:
        failures: Final[queue.Queue[object]] = queue.Queue()

        def install(seen: int, registry: native.ModelRegistry, errors: list[native.RowError]) -> int:
            raise RuntimeError("callback-test")

        response: Final = Response()
        observer: Final = Observer()
        with patch.object(sys, "unraisablehook", failures.put), server(response) as url:
            refresher: Final = native.CatalogRefresher(
                url,
                installed_revision=None,
                installed_validator=None,
                install=install,
                checked=observer.checked,
                generation=observer.generation,
            )
            self.addCleanup(refresher.stop)
            failure: Final = failures.get(timeout=5)
            self.assertIn("callback-test", str(getattr(failure, "exc_value")))
            wait_for(lambda: refresh_threads() == self.threads)
            self.assertTrue(observer.events.empty())
            self.assertTrue(failures.empty())

    def test_generation_failure_starts_no_transport(self) -> None:
        failures: Final[queue.Queue[object]] = queue.Queue()

        def generation() -> int:
            raise RuntimeError("generation-test")

        response: Final = Response()
        observer: Final = Observer()
        with patch.object(sys, "unraisablehook", failures.put), server(response) as url:
            refresher: Final = native.CatalogRefresher(
                url,
                installed_revision=None,
                installed_validator=None,
                install=observer.install,
                checked=observer.checked,
                generation=generation,
            )
            self.addCleanup(refresher.stop)
            failure: Final = failures.get(timeout=5)
            self.assertIn("generation-test", str(getattr(failure, "exc_value")))
            wait_for(lambda: refresh_threads() == self.threads)
            self.assertTrue(response.requests.empty())
            self.assertTrue(observer.events.empty())

    def test_callback_cycle_collected_after_worker_completes(self) -> None:
        finalized: Final[queue.Queue[bool]] = queue.Queue()

        class Owner(Observer):
            def __init__(self, url: str) -> None:
                super().__init__()
                self.refresher = self.start(url)

            def __del__(self) -> None:
                self.refresher.stop()
                self.refresher.after_fork()
                finalized.put(True)

        response: Final = Response(304)
        with server(response) as url:

            def make_cycle() -> weakref.ReferenceType[Owner]:
                owner: Final = Owner(url)
                self.assertEqual(owner.events.get(timeout=5), ("checked", None))
                return weakref.ref(owner)

            reference: Final = make_cycle()
            wait_for(lambda: refresh_threads() == self.threads)
            gc.collect()
            self.assertIsNone(reference())
            self.assertTrue(finalized.get(timeout=1))

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_after_fork_restarts_unadopted_child(self) -> None:
        response: Final = Response(hold=True)
        with server(response) as url:
            observer: Final = Observer()
            refresher: Final = observer.start(url)
            self.addCleanup(refresher.stop)
            self.assertTrue(response.entered.wait(5))
            read_fd, write_fd = os.pipe()
            pid: Final = os.fork()
            if pid == 0:
                os.close(read_fd)
                try:
                    refresher.after_fork()
                    kind, _ = observer.events.get(timeout=10)
                    refresher.stop()
                    os.write(write_fd, kind.encode())
                finally:
                    os._exit(0)
            os.close(write_fd)
            try:
                response.release.set()
                self.assertEqual(os.read(read_fd, 32), b"install")
                self.assertEqual(os.waitpid(pid, 0)[1], 0)
                self.assertEqual(observer.events.get(timeout=5)[0], "install")
            finally:
                os.close(read_fd)


if __name__ == "__main__":
    unittest.main()
