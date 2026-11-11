import asyncio
import hashlib
import json
import logging
import os
import ssl
import sys
import threading
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlsplit

import pytest
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server

import litellm
from litellm.litellm_core_utils.get_model_cost_map import (
    GetModelCostMap,
    ModelCostMapReloaded,
    ModelCostMapReloadUnavailable,
    get_model_cost_map_loaded_at,
    get_model_cost_map_provenance,
    get_model_cost_map_source_info,
    model_cost_inputs,
    refetch_model_cost_map,
)
from litellm.litellm_core_utils.model_cost_table import publish
from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter

FALLBACK_GENERALIZATIONS_KEY: Final = "fallback_generalizations"
_BUNDLED: Final = GetModelCostMap.read_local_model_cost_map_bytes()
_IMPORTED_MODEL_COST_INPUTS: Final = model_cost_inputs.snapshot()


def _blob_id(body: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(body) + body, usedforsecurity=False).hexdigest()


def _document_with(**rows: Mapping[str, object]) -> bytes:
    """The bundled document with some rows replaced, so it passes the integrity limits."""
    return json.dumps({**json.loads(_BUNDLED), **rows}).encode()


@dataclass(frozen=True, slots=True)
class _Reply:
    status: int = 200
    body: bytes = _BUNDLED
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Server:
    url: str
    received: Sequence[Request]

    @property
    def targets(self) -> Sequence[str]:
        return [request.target for request in self.received]

    @property
    def authorizations(self) -> Sequence[str]:
        return [request.headers.get("authorization", "") for request in self.received]

    @property
    def requests(self) -> int:
        return len(self.received)


@contextmanager
def _serving(*replies: _Reply, tls: ssl.SSLContext | None = None) -> Generator[_Server]:
    """Answers each GET with the next scripted reply, repeating the last one."""
    received: Final[list[Request]] = []  # mutable-ok: respond records each request the peer answers

    def respond(request: Request) -> Reply:
        received.append(request)
        reply: Final = replies[min(len(received), len(replies)) - 1]
        return Reply(status=reply.status, body=reply.body, headers=reply.headers)

    with wire_server(respond, tls=tls) as wire:
        yield _Server(url=f"{wire.url}/cost.json", received=received)


@pytest.fixture(autouse=True)
def isolated_model_cost() -> Iterator[None]:
    inputs: Final = model_cost_inputs.snapshot()
    rows: Final = litellm.model_cost
    entries: Final = dict(rows)
    model_cost_inputs.restore(_IMPORTED_MODEL_COST_INPUTS)
    publish(rows)
    yield
    model_cost_inputs.restore(inputs)
    rows.clear()
    rows.update(entries)
    publish(rows)


@pytest.fixture
def remote_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.setattr(sys, "argv", ["python"])


@pytest.fixture
def own_model_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restores the binding of ``litellm.model_cost`` after a test that installs a new table."""
    monkeypatch.setattr(litellm, "model_cost", litellm.model_cost)


@pytest.fixture
def isolated_openai_model_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "open_ai_chat_completion_models", set(litellm.open_ai_chat_completion_models))
    monkeypatch.setattr(litellm, "open_ai_text_completion_models", set(litellm.open_ai_text_completion_models))


@pytest.mark.usefixtures("remote_source")
def test_boot_load_installs_the_served_document_and_records_its_provenance():
    body: Final = _document_with(**{"lit-boot-model": {"litellm_provider": "openai", "mode": "chat"}})
    with _serving(_Reply(body=body, headers={"ETag": '"v1"'})) as server:
        rows: Final = litellm.get_model_cost_map(url=server.url)

    assert "lit-boot-model" in rows
    assert FALLBACK_GENERALIZATIONS_KEY not in rows
    assert get_model_cost_map_provenance() == {"source_revision": _blob_id(body), "etag": '"v1"'}
    source: Final = get_model_cost_map_source_info()
    assert (source["source"], source["url"], source["fallback_reason"]) == ("remote", server.url, None)


@pytest.mark.usefixtures("remote_source")
@pytest.mark.parametrize(
    ("reply", "fallback_reason"),
    [
        (_Reply(status=404, body=b"missing"), "Remote fetch failed: HTTP 404 from "),
        (_Reply(body=b"[1, 2]"), "Remote fetch failed: invalid JSON from "),
        (
            _Reply(body=json.dumps({"only": {"litellm_provider": "openai"}}).encode()),
            "Remote data failed integrity validation",
        ),
    ],
)
def test_boot_load_falls_back_to_the_bundled_document(reply: _Reply, fallback_reason: str):
    with _serving(reply) as server:
        rows: Final = litellm.get_model_cost_map(url=server.url, max_attempts=1)

    assert "gpt-4o" in rows
    assert get_model_cost_map_provenance() == {"source_revision": _blob_id(_BUNDLED), "etag": None}
    source: Final = get_model_cost_map_source_info()
    assert (source["source"], source["url"], source["is_env_forced"]) == ("local", server.url, False)
    assert source["fallback_reason"] is not None and source["fallback_reason"].startswith(fallback_reason)


_MOVED: Final = _Reply(status=302, body=b"", headers={"Location": "/moved.json"})
_REDIRECTED_DOCUMENT: Final = _document_with(**{"lit-redirected-model": {"litellm_provider": "openai", "mode": "chat"}})


def _rows_loaded_through(load: Literal["boot", "reload", "register_model"], url: str) -> Mapping[str, object]:
    """The rows ``load`` produced from ``url``: the boot load's return value, or the live table after the others."""
    match load:
        case "boot":
            return litellm.get_model_cost_map(url=url, max_attempts=1)
        case "reload":
            assert isinstance(asyncio.run(refetch_model_cost_map(url=url, max_attempts=1)), ModelCostMapReloaded)
            return litellm.model_cost
        case "register_model":
            litellm.register_model(url, persist_across_reloads=False)
            return litellm.model_cost


@pytest.mark.usefixtures("remote_source", "own_model_cost")
@pytest.mark.parametrize("load", ["boot", "reload", "register_model"])
def test_every_load_follows_a_redirect_to_the_document(load: Literal["boot", "reload", "register_model"]):
    with _serving(_MOVED, _Reply(body=_REDIRECTED_DOCUMENT)) as server:
        rows: Final = _rows_loaded_through(load, server.url)

    assert server.targets == ["/cost.json", "/moved.json"]
    assert "lit-redirected-model" in rows


@pytest.mark.usefixtures("remote_source", "own_model_cost")
def test_boot_load_transient_failure_serves_the_bundled_map_then_installs_the_remote_one_in_the_background():
    body: Final = _document_with(**{"lit-retried-model": {"litellm_provider": "openai", "mode": "chat"}})
    with _serving(_Reply(status=503, body=b"busy", headers={"Retry-After": "0"}), _Reply(body=body)) as server:
        rows: Final = litellm.get_model_cost_map(url=server.url)
        assert "lit-retried-model" not in rows
        booted_at: Final = get_model_cost_map_loaded_at()
        retry: Final = next(t for t in threading.enumerate() if t.name == "litellm-model-cost-map-retry")
        retry.join(timeout=30)

    assert server.requests == 2
    assert "lit-retried-model" in litellm.model_cost
    source: Final = get_model_cost_map_source_info()
    assert (source["source"], source["url"], source["is_env_forced"], source["fallback_reason"]) == (
        "remote",
        server.url,
        False,
        None,
    )
    retried_at: Final = get_model_cost_map_loaded_at()
    assert booted_at is not None and retried_at is not None and retried_at > booted_at


@pytest.mark.usefixtures("remote_source")
def test_boot_load_does_not_retry_a_final_failure():
    with _serving(_Reply(status=404, body=b"missing")) as server:
        litellm.get_model_cost_map(url=server.url)

    assert not any(t.name == "litellm-model-cost-map-retry" for t in threading.enumerate())
    assert server.requests == 1


@pytest.mark.parametrize(
    ("argv0", "environment"),
    [
        ("/usr/bin/litellm-proxy", {}),
        ("/usr/bin/lite", {}),
        ("python", {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}),
        ("python", {"LITELLM_OFFLINE_MODE": "true"}),
    ],
    ids=["proxy-cli", "lite-cli", "local-map", "offline"],
)
def test_boot_load_serves_the_bundled_map_without_fetching(
    monkeypatch: pytest.MonkeyPatch, argv0: str, environment: Mapping[str, str]
):
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.setattr(sys, "argv", [argv0])
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with _serving(_Reply()) as server:
        rows: Final = litellm.get_model_cost_map(url=server.url)

    assert server.requests == 0
    assert "gpt-4o" in rows
    assert get_model_cost_map_provenance() == {"source_revision": _blob_id(_BUNDLED), "etag": None}
    assert get_model_cost_map_source_info()["is_env_forced"] is True


def test_importing_litellm_loads_the_map_litellm_model_cost_map_url_names():
    script: Final = """
import os
import litellm
assert litellm.model_cost_map_url == os.environ["LITELLM_MODEL_COST_MAP_URL"], litellm.model_cost_map_url
assert "lit-env-url-model" in litellm.model_cost
"""
    body: Final = _document_with(**{"lit-env-url-model": {"litellm_provider": "openai", "mode": "chat"}})
    environment: Final = {
        key: value
        for key, value in os.environ.items()
        if key not in {"LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_OFFLINE_MODE"}
    }
    with _serving(_Reply(body=body)) as server:
        result: Final = run_child_interpreter(
            script, env={**environment, "LITELLM_MODEL_COST_MAP_URL": server.url}, timeout=120
        )

    assert result.returncode == 0, result.stderr
    assert server.targets == ["/cost.json"]


@pytest.mark.skipif(sys.platform != "linux", reason="fork is the default start method only on Linux")
def test_importing_litellm_with_a_remote_source_leaves_the_process_free_to_fork():
    script: Final = """
import os
import litellm
from litellm.rust_bridge._native import process_state_started, reserve_process_for_forking
assert "lit-fork-model" in litellm.model_cost
assert not process_state_started()
reserve_process_for_forking()
pid = os.fork()
if pid == 0:
    os._exit(0 if "lit-fork-model" in litellm.model_cost else 1)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
"""
    body: Final = _document_with(**{"lit-fork-model": {"litellm_provider": "openai", "mode": "chat"}})
    environment: Final = {key: value for key, value in os.environ.items() if key != "LITELLM_LOCAL_MODEL_COST_MAP"}
    with _serving(_Reply(body=body)) as server:
        result: Final = run_child_interpreter(
            script, env={**environment, "LITELLM_MODEL_COST_MAP_URL": server.url}, timeout=120
        )

    assert result.returncode == 0, result.stderr


def test_register_model_leaves_the_native_runtime_unstarted() -> None:
    script: Final = """
import litellm
from litellm.rust_bridge._native import process_state_started
litellm.register_model({"lit-fork-safe-model": {"input_cost_per_token": 1e-06, "litellm_provider": "openai"}})
assert litellm.model_cost["lit-fork-safe-model"]["input_cost_per_token"] == 1e-06
assert not process_state_started()
"""
    result: Final = run_child_interpreter(
        script, env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"}, timeout=120
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.usefixtures("remote_source", "own_model_cost")
def test_reload_retries_a_429_after_its_retry_after():
    with _serving(_Reply(status=429, body=b"slow", headers={"Retry-After": "0"}), _Reply()) as server:
        result: Final = asyncio.run(refetch_model_cost_map(url=server.url))

    assert isinstance(result, ModelCostMapReloaded)
    assert server.requests == 2


@pytest.mark.usefixtures("remote_source", "own_model_cost")
@pytest.mark.parametrize(
    ("reply", "reason", "requests"),
    [
        (_Reply(status=503, body=b"busy", headers={"Retry-After": "0"}), "HTTP 503 from {url} (after 3 attempts)", 3),
        (_Reply(body=b"not json"), "invalid JSON from {url}", 1),
        (
            _Reply(body=json.dumps({"only": {"litellm_provider": "openai"}}).encode()),
            "model cost map from {url} failed integrity validation",
            1,
        ),
    ],
    ids=["gives-up-after-its-attempts", "not-json", "below-integrity-limits"],
)
def test_a_failed_reload_keeps_the_installed_map(reply: _Reply, reason: str, requests: int):
    installed: Final = litellm.model_cost
    provenance: Final = get_model_cost_map_provenance()
    with _serving(reply) as server:
        result: Final = asyncio.run(refetch_model_cost_map(url=server.url, max_attempts=3))

    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert result.reason.startswith(reason.format(url=server.url)), result.reason
    assert server.requests == requests
    assert litellm.model_cost is installed
    assert get_model_cost_map_provenance() == provenance


@pytest.mark.usefixtures("remote_source", "own_model_cost")
def test_a_reload_rejected_by_the_integrity_limits_logs_the_counts(caplog: pytest.LogCaptureFixture):
    with (
        _serving(_Reply(body=json.dumps({"only": {"litellm_provider": "openai"}}).encode())) as server,
        caplog.at_level(logging.WARNING, logger="LiteLLM"),
    ):
        asyncio.run(refetch_model_cost_map(url=server.url))

    assert "has 1 models, below minimum" in caplog.text


@pytest.mark.usefixtures("own_model_cost")
@pytest.mark.parametrize("argv0", ["/usr/bin/litellm-proxy", "/usr/bin/lite"])
def test_a_cli_process_that_booted_on_the_bundled_map_still_fetches_on_reload(
    monkeypatch: pytest.MonkeyPatch, argv0: str
):
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.setattr(sys, "argv", [argv0])
    with _serving(_Reply()) as server:
        litellm.get_model_cost_map(url=server.url)
        booted_requests: Final = server.requests
        result: Final = asyncio.run(refetch_model_cost_map(url=server.url))

    assert booted_requests == 0
    assert isinstance(result, ModelCostMapReloaded)
    assert server.requests == 1
    assert get_model_cost_map_source_info()["source"] == "remote"


@pytest.mark.usefixtures("own_model_cost")
def test_reload_with_the_local_override_installs_the_bundled_map_without_fetching(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    with _serving(_Reply()) as server:
        result: Final = asyncio.run(refetch_model_cost_map(url=server.url))

    assert isinstance(result, ModelCostMapReloaded)
    assert server.requests == 0
    assert get_model_cost_map_provenance() == {"source_revision": _blob_id(_BUNDLED), "etag": None}
    source: Final = get_model_cost_map_source_info()
    assert (source["source"], source["url"], source["is_env_forced"]) == ("local", None, True)


@pytest.mark.usefixtures("remote_source", "own_model_cost")
def test_only_a_reload_that_installs_moves_the_source_info_and_load_time():
    with _serving(_Reply()) as first:
        asyncio.run(refetch_model_cost_map(url=first.url))
    installed: Final = get_model_cost_map_source_info()
    loaded_at: Final = get_model_cost_map_loaded_at()

    with _serving(_Reply(status=404, body=b"missing")) as failing:
        asyncio.run(refetch_model_cost_map(url=failing.url))
    assert get_model_cost_map_source_info() == installed
    assert get_model_cost_map_loaded_at() is loaded_at

    with _serving(_Reply()) as second:
        asyncio.run(refetch_model_cost_map(url=second.url))
    source: Final = get_model_cost_map_source_info()
    assert (source["source"], source["url"], source["is_env_forced"], source["fallback_reason"]) == (
        "remote",
        second.url,
        False,
        None,
    )
    assert get_model_cost_map_loaded_at() is not loaded_at


@pytest.mark.usefixtures("remote_source", "own_model_cost")
def test_reload_records_the_served_bytes_and_etag():
    body: Final = _document_with(**{"lit-revision-model": {"litellm_provider": "openai", "mode": "chat"}})
    with _serving(_Reply(body=body, headers={"ETag": 'W/"etag"'})) as server:
        asyncio.run(refetch_model_cost_map(url=server.url))

    assert get_model_cost_map_provenance() == {"source_revision": _blob_id(body), "etag": 'W/"etag"'}


def _self_signed(directory: Path) -> tuple[ssl.SSLContext, Path]:
    """A server context for 127.0.0.1 and the path of its certificate, which nothing trusts by default."""
    certificate, key = write_self_signed_cert(directory)
    return server_context(certificate, key), certificate


@pytest.mark.usefixtures("remote_source", "own_model_cost")
def test_a_reload_rejects_an_untrusted_certificate_by_default(tmp_path: Path):
    context, _ = _self_signed(tmp_path)
    with _serving(_Reply(), tls=context) as server:
        result: Final = asyncio.run(refetch_model_cost_map(url=server.url, max_attempts=1))

    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert result.reason.startswith(f"error fetching {server.url}: ")
    assert server.requests == 0


@pytest.fixture
def no_tls_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SSL_VERIFY", "SSL_CERT_FILE", "SSL_CERTIFICATE"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("remote_source", "no_tls_environment", "own_model_cost")
@pytest.mark.parametrize(
    ("load", "setting", "trust"),
    [
        ("reload", "litellm.ssl_verify", "off"),
        ("reload", "litellm.ssl_verify", "certificate"),
        ("boot", "SSL_VERIFY", "off"),
        ("boot", "SSL_CERT_FILE", "certificate"),
    ],
)
def test_a_load_trusts_the_certificate_its_settings_allow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    load: Literal["boot", "reload"],
    setting: str,
    trust: Literal["off", "certificate"],
):
    context, certificate = _self_signed(tmp_path)
    if setting == "litellm.ssl_verify":
        monkeypatch.setattr(litellm, "ssl_verify", {"off": False, "certificate": str(certificate)}[trust])
    else:
        monkeypatch.setenv(setting, {"off": "False", "certificate": str(certificate)}[trust])
    with _serving(_Reply(), tls=context) as server:
        _rows_loaded_through(load, server.url)

    assert server.requests == 1
    assert get_model_cost_map_source_info()["source"] == "remote"


@pytest.mark.usefixtures("remote_source", "no_tls_environment")
def test_a_boot_load_ignores_litellm_ssl_verify_because_it_runs_before_settings_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    context, _ = _self_signed(tmp_path)
    monkeypatch.setattr(litellm, "ssl_verify", False)
    with _serving(_Reply(), tls=context) as server:
        litellm.get_model_cost_map(url=server.url, max_attempts=1)

    assert server.requests == 0
    source: Final = get_model_cost_map_source_info()
    assert source["source"] == "local"
    assert source["fallback_reason"] is not None
    assert source["fallback_reason"].startswith(f"Remote fetch failed: error fetching {server.url}: ")


_UNREACHABLE_ENDPOINT: Final = "http://127.0.0.1:9"


@pytest.mark.usefixtures("remote_source")
@pytest.mark.parametrize(
    ("environment", "endpoint_variable", "region"),
    [
        (
            {"AWS_REGION_NAME": "eu-west-3", "AWS_REGION": "ap-south-1", "AWS_DEFAULT_REGION": "sa-east-1"},
            "AWS_ENDPOINT_URL_S3",
            "eu-west-3",
        ),
        ({"AWS_REGION": "ap-south-1", "AWS_DEFAULT_REGION": "sa-east-1"}, "AWS_ENDPOINT_URL_S3", "ap-south-1"),
        ({"AWS_DEFAULT_REGION": "sa-east-1"}, "AWS_ENDPOINT_URL_S3", "sa-east-1"),
        ({"AWS_ENDPOINT_URL": _UNREACHABLE_ENDPOINT}, "AWS_ENDPOINT_URL_S3", "us-east-1"),
        ({}, "AWS_ENDPOINT_URL", "us-east-1"),
        ({"AWS_CONFIG_FILE": "profile_region"}, "AWS_ENDPOINT_URL_S3", "eu-north-1"),
    ],
)
def test_an_s3_load_signs_for_the_region_and_reaches_the_endpoint_the_environment_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    environment: Mapping[str, str],
    endpoint_variable: str,
    region: str,
):
    for name in (
        "AWS_REGION_NAME",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_SESSION_TOKEN",
        "AWS_ENDPOINT_URL_S3",
        "AWS_ENDPOINT_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    profile: Final = tmp_path / "profile_region"
    profile.write_text("[default]\nregion = eu-north-1\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no_config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no_credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDMODELCOST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "model-cost-secret")
    for name, value in environment.items():
        monkeypatch.setenv(name, str(tmp_path / value) if name == "AWS_CONFIG_FILE" else value)
    with _serving(_Reply()) as server:
        monkeypatch.setenv(endpoint_variable, server.url.removesuffix("/cost.json"))
        litellm.get_model_cost_map(url="s3://lit-prices/maps/cost.json", max_attempts=1)

    assert [urlsplit(target).path for target in server.targets] == ["/lit-prices/maps/cost.json"]
    assert f"/{region}/s3/aws4_request" in server.authorizations[0]
    assert get_model_cost_map_source_info()["source"] == "remote"


_GCS_URL: Final = "gs://lit-prices/maps/cost.json"
_GCS_OBJECT_TARGET: Final = "/storage/v1/b/lit-prices/o/maps%2Fcost.json?alt=media"
_GCS_TOKEN: Final = "ya29.lit-scripted-gcs-token"


@pytest.fixture
def gcs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[_Reply], AbstractContextManager[_Server]]:
    @contextmanager
    def serving(reply: _Reply) -> Generator[_Server]:
        received: Final[list[Request]] = []  # mutable-ok: respond records every request the peer answers

        def respond(request: Request) -> Reply:
            received.append(request)
            if request.target == "/_oauth/token":
                return Reply(body=json.dumps({"access_token": _GCS_TOKEN, "expires_in": 3600}).encode())
            return Reply(status=reply.status, body=reply.body, headers=reply.headers)

        with wire_server(respond) as wire:
            credentials: Final = tmp_path / "application_default_credentials.json"
            credentials.write_text(service_account_json("lit-prices-project", wire.url))
            monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))
            monkeypatch.setenv("STORAGE_EMULATOR_HOST", wire.url)
            yield _Server(url=_GCS_URL, received=received)

    return serving


@pytest.mark.usefixtures("remote_source", "own_model_cost")
@pytest.mark.parametrize("load", ["boot", "reload", "register_model"])
def test_every_load_reads_a_gs_url_from_the_emulator_without_credentials(
    gcs: Callable[[_Reply], AbstractContextManager[_Server]], load: Literal["boot", "reload", "register_model"]
):
    body: Final = _document_with(**{"lit-gcs-model": {"litellm_provider": "openai", "mode": "chat"}})
    with gcs(_Reply(body=body)) as server:
        rows: Final = _rows_loaded_through(load, server.url)

    assert server.targets == [_GCS_OBJECT_TARGET]
    assert server.authorizations == [""]
    assert "lit-gcs-model" in rows


@pytest.mark.usefixtures("remote_source", "own_model_cost")
def test_a_gs_load_reaches_an_emulator_host_given_without_a_scheme(
    gcs: Callable[[_Reply], AbstractContextManager[_Server]], monkeypatch: pytest.MonkeyPatch
):
    with gcs(_Reply()) as server:
        emulator: Final = os.environ["STORAGE_EMULATOR_HOST"]
        monkeypatch.setenv("STORAGE_EMULATOR_HOST", emulator.removeprefix("http://"))
        litellm.get_model_cost_map(url=server.url, max_attempts=1)

    assert server.targets == [_GCS_OBJECT_TARGET]
    assert get_model_cost_map_source_info()["source"] == "remote"


@pytest.mark.usefixtures("remote_source")
def test_a_gs_boot_load_records_the_remote_source(gcs: Callable[[_Reply], AbstractContextManager[_Server]]):
    with gcs(_Reply()) as server:
        litellm.get_model_cost_map(url=server.url)

    source: Final = get_model_cost_map_source_info()
    assert (source["source"], source["url"], source["fallback_reason"]) == ("remote", _GCS_URL, None)


@pytest.mark.usefixtures("remote_source")
@pytest.mark.parametrize("status", [403, 404], ids=["denied", "missing"])
def test_a_gs_boot_load_refused_by_storage_falls_back_without_a_retry(
    gcs: Callable[[_Reply], AbstractContextManager[_Server]], status: int
):
    with gcs(_Reply(status=status, body=b"refused")) as server:
        litellm.get_model_cost_map(url=server.url, max_attempts=3)

    assert server.requests == 1
    source: Final = get_model_cost_map_source_info()
    assert source["source"] == "local"
    assert source["fallback_reason"] == f"Remote fetch failed: HTTP {status} from {_GCS_URL}"


@pytest.mark.usefixtures("remote_source", "own_model_cost")
@pytest.mark.parametrize("status", [403, 404], ids=["denied", "missing"])
def test_a_gs_reload_refused_by_storage_keeps_the_installed_map(
    gcs: Callable[[_Reply], AbstractContextManager[_Server]], status: int
):
    installed: Final = litellm.model_cost
    with gcs(_Reply(status=status, body=b"refused")) as server:
        result: Final = asyncio.run(refetch_model_cost_map(url=server.url, max_attempts=3))

    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert result.reason == f"HTTP {status} from {_GCS_URL}"
    assert server.requests == 1
    assert litellm.model_cost is installed


@pytest.mark.usefixtures("remote_source", "own_model_cost")
@pytest.mark.parametrize("load", ["boot", "reload"])
def test_both_loads_fetch_through_the_proxy_the_environment_names(
    monkeypatch: pytest.MonkeyPatch, load: Literal["boot", "reload"]
):
    for name in ("http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    url: Final = "http://lit-cost-map.invalid/cost.json"
    with _serving(_Reply()) as proxy:
        monkeypatch.setenv("HTTP_PROXY", proxy.url.removesuffix("/cost.json"))
        rows: Final = _rows_loaded_through(load, url)

    assert "gpt-4o" in rows
    assert get_model_cost_map_source_info()["source"] == "remote"
    assert proxy.targets == [url]


@pytest.mark.usefixtures("remote_source", "own_model_cost")
def test_register_model_from_a_url_registers_the_served_rows_without_the_fallback_rules():
    document: Final = {
        "lit-url-model": {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 4e-06},
        FALLBACK_GENERALIZATIONS_KEY: {"rules": []},
    }
    with _serving(_Reply(body=json.dumps(document).encode())) as server:
        litellm.register_model(server.url)

    assert server.requests == 1
    assert litellm.model_cost["lit-url-model"]["input_cost_per_token"] == 4e-06
    assert FALLBACK_GENERALIZATIONS_KEY not in litellm.model_cost
    assert model_cost_inputs.registered_keys() == [*GetModelCostMap.loaded_model_cost_map(), "lit-url-model"]


@pytest.mark.usefixtures("remote_source", "own_model_cost")
@pytest.mark.parametrize(
    "reply", [_Reply(status=500, body=b"down"), _Reply(body=b"not json")], ids=["server_error", "invalid_json"]
)
def test_register_model_from_a_url_that_fails_registers_nothing(reply: _Reply):
    with _serving(reply) as server:
        litellm.register_model(server.url)

    assert model_cost_inputs.registered_keys() == list(GetModelCostMap.loaded_model_cost_map())


@pytest.mark.usefixtures("isolated_openai_model_sets")
def test_register_model_url_fetch_uses_single_attempt(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    before: Final = dict(litellm.model_cost)
    with wire_server(lambda _: Reply(status=503, body=b"")) as peer:
        threads_before: Final = {thread.name for thread in threading.enumerate()}
        litellm.register_model(model_cost=f"{peer.url}/custom_pricing.json")

    threads_after: Final = {thread.name for thread in threading.enumerate()}
    requests: Final = [request.target for request in peer.drain()]
    assert requests == ["/custom_pricing.json"]
    assert not (threads_after - threads_before) & {"litellm-model-cost-map-retry"}
    assert not any(
        thread.name == "litellm-model-cost-map-retry" and thread.is_alive() for thread in threading.enumerate()
    )
    assert litellm.model_cost.keys() >= before.keys()
