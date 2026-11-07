"""
The Python host of the native model cost map: loading and reloading the document over a real local HTTP server,
installing it as ``litellm.model_cost``, the ``register_model`` overlay across reloads, and the lookups over the dict.
"""

import asyncio
import hashlib
import ipaddress
import json
import os
import ssl
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlsplit

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

import litellm
from litellm.litellm_core_utils.fallback_generalizations import (
    get_fallback_generalization_rules,
    match_routing_generalization,
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.get_model_cost_map import (
    AliasedCostEntry,
    GetModelCostMap,
    ModelCostMapReloaded,
    ModelCostMapReloadUnavailable,
    get_model_cost_map_loaded_at,
    get_model_cost_map_provenance,
    get_model_cost_map_source_info,
    install_model_cost_map,
    model_cost_inputs,
    refetch_model_cost_map,
)
from litellm.rust_bridge._native import NativeInstalled, NativeModelCostMap
from litellm.utils import catalog_with_runtime_registrations, get_model_cost_key, runtime_model_cost_registrations
from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter

FALLBACK_GENERALIZATIONS_KEY: Final = "fallback_generalizations"
_BUNDLED: Final = GetModelCostMap.read_local_model_cost_map_bytes()


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


@dataclass(slots=True)
class _Server:
    url: str
    replies: list[_Reply]  # mutable-ok: the test scripts the replies and the handler pops them in order
    targets: list[str] = field(default_factory=list)  # mutable-ok: the handler records each request target
    authorizations: list[str] = field(default_factory=list)  # mutable-ok: the handler records each Authorization
    requests: int = 0


@contextmanager
def _serving(*replies: _Reply, tls: ssl.SSLContext | None = None) -> Iterator[_Server]:
    """Answers each GET with the next scripted reply, repeating the last one."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            server.requests += 1
            server.targets.append(self.path)
            server.authorizations.append(self.headers.get("Authorization", ""))
            reply = server.replies.pop(0) if len(server.replies) > 1 else server.replies[0]
            self.send_response(reply.status)
            for name, value in reply.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(reply.body)))
            self.end_headers()
            self.wfile.write(reply.body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    http_server: Final = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    if tls is not None:
        http_server.socket = tls.wrap_socket(http_server.socket, server_side=True)
    scheme: Final = "http" if tls is None else "https"
    server: Final = _Server(
        url=f"{scheme}://127.0.0.1:{http_server.server_address[1]}/cost.json", replies=list(replies)
    )
    thread: Final = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        http_server.shutdown()
        http_server.server_close()


@pytest.fixture
def remote_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.setattr(sys, "argv", ["python"])


@pytest.fixture
def own_model_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restores the binding of ``litellm.model_cost`` after a test that installs a new table."""
    monkeypatch.setattr(litellm, "model_cost", litellm.model_cost)


# ---------------------------------------------------------------------------
# The shipped document
# ---------------------------------------------------------------------------


def test_shipped_backup_carries_the_claude_routing_rules():
    """The bundled backup must ship the Claude routing rules so a fresh install
    (or an offline fallback) routes unknown Claude models without code changes.
    Bedrock-syntax ids must hit the bedrock rule before the bare-id Anthropic rule."""
    backup = GetModelCostMap.load_local_model_cost_map()
    rules = backup.get(FALLBACK_GENERALIZATIONS_KEY, {}).get("rules", [])
    names = [r.get("name") for r in rules]
    assert names.index("bedrock-claude-ids") < names.index("anthropic-claude-ids")

    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(rules)
        assert match_routing_generalization("claude-opus-4-9") == "anthropic"
        assert match_routing_generalization("global.anthropic.claude-opus-4-9") == "bedrock"
    finally:
        set_fallback_generalizations(previous)


def test_shipped_routing_rules_never_match_through_an_unrecognized_namespace():
    """Routing rules decide ``litellm_provider`` for otherwise-unknown ids, and the
    proxy's wildcard access check (``can_key_call_model`` with a ``bedrock/*`` key)
    trusts that inference: it rebuilds ``{provider}/{model}`` and matches it against
    the key's patterns. A routing pattern that matches as a substring lets
    ``bedrockz/anthropic.claude-...`` resolve to bedrock and slip through a
    ``bedrock/*`` key, so every shipped routing rule must anchor to the start of
    the name and never match an id carrying an unrecognized namespace prefix."""
    backup = GetModelCostMap.load_local_model_cost_map()
    rules = backup[FALLBACK_GENERALIZATIONS_KEY]["rules"]

    routing_rules = [r for r in rules if "litellm_provider" in r["model_info"]]
    assert routing_rules
    assert all(r["pattern"].startswith("^") for r in routing_rules)

    previous = list(get_fallback_generalization_rules())
    try:
        set_fallback_generalizations(rules)
        for bedrock_id in [
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-v2:1",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "global.anthropic.claude-fable-5-20260120-v1:0",
        ]:
            assert match_routing_generalization(bedrock_id) == "bedrock", bedrock_id
        for namespaced in [
            "bedrockz/anthropic.claude-3-5-sonnet-20240620",
            "bedrockz/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "bedrockz/claude-3-5-sonnet-20240620",
        ]:
            assert match_routing_generalization(namespaced) is None, namespaced
    finally:
        set_fallback_generalizations(previous)


def test_shipped_backup_marks_claude_4_6_plus_adaptive_not_4_0():
    """Adaptive thinking is data, not code. The bundled backup must carry
    supports_adaptive_thinking on genuine Claude >= 4.6 entries (every provider
    route) and on the version-gated anthropic-claude-adaptive-thinking rule for
    unmapped future Claudes, while leaving the dated Claude 4.0 names
    ("...-4-20250514") unflagged so a date can never be mistaken for a 4.6+ minor
    version. The version-neutral claude-family-baseline capability rule must not flag
    it, so an unmapped sub-4.6 name resolves but stays non-adaptive. The adaptive rule
    carries only its delta; capability unioning stacks it onto the baseline, so the
    baseline block is never duplicated across rules and no rule needs ``extends``."""
    backup = GetModelCostMap.load_local_model_cost_map()

    rules = backup[FALLBACK_GENERALIZATIONS_KEY]["rules"]
    baseline_rule = next(r for r in rules if r.get("name") == "claude-family-baseline")
    adaptive_rule = next(r for r in rules if r.get("name") == "claude-adaptive-thinking")
    assert "supports_adaptive_thinking" not in baseline_rule["model_info"]
    assert "litellm_provider" not in baseline_rule["model_info"]
    assert adaptive_rule["model_info"] == {"supports_adaptive_thinking": True}
    assert all("extends" not in r for r in rules)

    for adaptive in [
        "anthropic.claude-opus-4-8",
        "vertex_ai/claude-opus-4-6@default",
        "us.anthropic.claude-sonnet-4-6",
        "azure_ai/claude-opus-4-7",
    ]:
        assert backup[adaptive]["supports_adaptive_thinking"] is True, adaptive

    for non_adaptive in [
        "us.anthropic.claude-opus-4-20250514-v1:0",
        "claude-opus-4-5",
    ]:
        assert "supports_adaptive_thinking" not in backup[non_adaptive], non_adaptive


# ---------------------------------------------------------------------------
# Import-time load
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("argv0", ["/usr/bin/litellm-proxy", "/usr/bin/lite"])
def test_boot_load_never_fetches_for_cli_processes(monkeypatch: pytest.MonkeyPatch, argv0: str):
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.setattr(sys, "argv", [argv0])
    with _serving(_Reply()) as server:
        litellm.get_model_cost_map(url=server.url)

    assert server.requests == 0
    assert get_model_cost_map_source_info()["is_env_forced"] is True


@pytest.mark.parametrize(
    ("variable", "value"), [("LITELLM_LOCAL_MODEL_COST_MAP", "True"), ("LITELLM_OFFLINE_MODE", "true")]
)
def test_boot_load_with_the_bundled_map_forced_never_fetches(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
):
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.setattr(sys, "argv", ["python"])
    monkeypatch.setenv(variable, value)
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


# ---------------------------------------------------------------------------
# Runtime reload
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("remote_source", "own_model_cost")
def test_reload_retries_a_429_after_its_retry_after():
    with _serving(_Reply(status=429, body=b"slow", headers={"Retry-After": "0"}), _Reply()) as server:
        result: Final = asyncio.run(refetch_model_cost_map(url=server.url))

    assert isinstance(result, ModelCostMapReloaded)
    assert server.requests == 2


@pytest.mark.usefixtures("remote_source", "own_model_cost")
def test_reload_gives_up_after_its_attempts():
    with _serving(_Reply(status=503, body=b"busy", headers={"Retry-After": "0"})) as server:
        result: Final = asyncio.run(refetch_model_cost_map(url=server.url, max_attempts=3))

    assert result == ModelCostMapReloadUnavailable(reason=f"HTTP 503 from {server.url} (after 3 attempts)")
    assert server.requests == 3


@pytest.mark.usefixtures("remote_source", "own_model_cost")
@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (_Reply(status=404, body=b"missing"), "HTTP 404 from "),
        (_Reply(body=b"not json"), "invalid JSON from "),
        (_Reply(body=json.dumps({"only": {"litellm_provider": "openai"}}).encode()), "model cost map from "),
    ],
)
def test_a_failed_reload_keeps_the_installed_map(reply: _Reply, reason: str):
    installed: Final = litellm.model_cost
    provenance: Final = get_model_cost_map_provenance()
    with _serving(reply) as server:
        result: Final = asyncio.run(refetch_model_cost_map(url=server.url))

    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert result.reason.startswith(reason)
    assert server.requests == 1
    assert litellm.model_cost is installed
    assert get_model_cost_map_provenance() == provenance


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


# ---------------------------------------------------------------------------
# Fetch settings
# ---------------------------------------------------------------------------


def _self_signed(directory: Path) -> tuple[ssl.SSLContext, Path]:
    """A server context for 127.0.0.1 and the path of its certificate, which nothing trusts by default."""
    key: Final = ec.generate_private_key(ec.SECP256R1())
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "lit-cost-map")])
    now: Final = datetime.now(timezone.utc)
    certificate: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .sign(key, hashes.SHA256())
    )
    certificate_path: Final = directory / "cost-map.crt"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path: Final = directory / "cost-map.key"
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate_path, key_path)
    return context, certificate_path


@pytest.mark.usefixtures("remote_source", "own_model_cost")
def test_a_reload_rejects_an_untrusted_certificate_by_default(tmp_path: Path):
    context, _ = _self_signed(tmp_path)
    with _serving(_Reply(), tls=context) as server:
        result: Final = asyncio.run(refetch_model_cost_map(url=server.url, max_attempts=1))

    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert result.reason.startswith(f"error fetching {server.url}: ")
    assert server.requests == 0


@pytest.mark.usefixtures("remote_source", "own_model_cost")
@pytest.mark.parametrize("ssl_verify", ["off", "certificate"])
def test_a_reload_accepts_the_certificate_litellm_ssl_verify_allows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ssl_verify: Literal["off", "certificate"]
):
    context, certificate = _self_signed(tmp_path)
    monkeypatch.setattr(litellm, "ssl_verify", {"off": False, "certificate": str(certificate)}[ssl_verify])
    with _serving(_Reply(), tls=context) as server:
        result: Final = asyncio.run(refetch_model_cost_map(url=server.url, max_attempts=1))

    assert isinstance(result, ModelCostMapReloaded)
    assert server.requests == 1


@pytest.fixture
def no_tls_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SSL_VERIFY", "SSL_CERT_FILE", "SSL_CERTIFICATE"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("remote_source", "no_tls_environment")
@pytest.mark.parametrize("variable", ["SSL_VERIFY", "SSL_CERT_FILE"])
def test_a_boot_load_trusts_the_certificate_the_environment_allows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variable: Literal["SSL_VERIFY", "SSL_CERT_FILE"]
):
    context, certificate = _self_signed(tmp_path)
    monkeypatch.setenv(variable, {"SSL_VERIFY": "False", "SSL_CERT_FILE": str(certificate)}[variable])
    with _serving(_Reply(), tls=context) as server:
        litellm.get_model_cost_map(url=server.url, max_attempts=1)

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
    ],
)
def test_an_s3_load_signs_for_the_region_and_reaches_the_endpoint_the_environment_names(
    monkeypatch: pytest.MonkeyPatch, environment: Mapping[str, str], endpoint_variable: str, region: str
):
    for name in (
        "AWS_REGION_NAME",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_SESSION_TOKEN",
        "AWS_ENDPOINT_URL_S3",
        "AWS_ENDPOINT_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDMODELCOST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "model-cost-secret")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with _serving(_Reply()) as server:
        monkeypatch.setenv(endpoint_variable, server.url.removesuffix("/cost.json"))
        litellm.get_model_cost_map(url="s3://lit-prices/maps/cost.json", max_attempts=1)

    assert [urlsplit(target).path for target in server.targets] == ["/lit-prices/maps/cost.json"]
    assert f"/{region}/s3/aws4_request" in server.authorizations[0]
    assert get_model_cost_map_source_info()["source"] == "remote"


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
        loaded: Final = (
            "gpt-4o" in litellm.get_model_cost_map(url=url, max_attempts=1)
            and get_model_cost_map_source_info()["source"] == "remote"
            if load == "boot"
            else isinstance(asyncio.run(refetch_model_cost_map(url=url, max_attempts=1)), ModelCostMapReloaded)
        )

    assert loaded
    assert proxy.targets == [url]


# ---------------------------------------------------------------------------
# What litellm.model_cost promises
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("own_model_cost")
def test_model_cost_is_a_dict_that_reflects_each_install():
    first_body: Final = _document_with(**{"lit-first": {"litellm_provider": "openai"}})
    second_body: Final = _document_with(**{"lit-second": {"litellm_provider": "openai"}})

    install_model_cost_map(first_body)
    first: Final = litellm.model_cost
    install_model_cost_map(second_body)

    assert type(litellm.model_cost) is dict
    assert litellm.model_cost is not first
    assert "lit-second" in litellm.model_cost
    assert "lit-first" not in litellm.model_cost


@pytest.mark.usefixtures("own_model_cost")
def test_registrations_survive_a_reload_in_call_order():
    litellm.register_model(
        {"lit-registered": {"litellm_provider": "openai", "input_cost_per_token": 1, "mode": "chat"}}
    )
    litellm.register_model({"lit-registered": {"input_cost_per_token": 2, "output_cost_per_token": 3}})

    install_model_cost_map(_BUNDLED)

    assert litellm.model_cost["lit-registered"]["input_cost_per_token"] == 2
    assert litellm.model_cost["lit-registered"]["output_cost_per_token"] == 3
    assert litellm.get_model_info("lit-registered")["input_cost_per_token"] == 2


@pytest.mark.usefixtures("own_model_cost")
def test_a_registration_that_does_not_persist_is_gone_after_a_reload():
    litellm.register_model(
        {"lit-request-scoped": {"litellm_provider": "openai", "input_cost_per_token": 1}}, persist_across_reloads=False
    )
    assert "lit-request-scoped" in litellm.model_cost

    install_model_cost_map(_BUNDLED)

    assert "lit-request-scoped" not in litellm.model_cost


@pytest.mark.usefixtures("own_model_cost")
def test_a_registration_wins_the_fields_it_shares_with_a_request_scoped_row_after_a_reload():
    litellm.register_model({"gpt-4o": {"input_cost_per_token": 7.0}})
    litellm.register_model(
        {"gpt-4o": {"input_cost_per_token": 9.0, "output_cost_per_token": 8.0}}, persist_across_reloads=False
    )

    install_model_cost_map(_BUNDLED)

    assert litellm.model_cost["gpt-4o"]["input_cost_per_token"] == 7.0
    assert (
        litellm.model_cost["gpt-4o"]["output_cost_per_token"] == json.loads(_BUNDLED)["gpt-4o"]["output_cost_per_token"]
    )


@pytest.mark.usefixtures("own_model_cost")
def test_a_rejected_document_changes_nothing():
    installed: Final = litellm.model_cost
    rows: Final = dict(installed)
    provenance: Final = get_model_cost_map_provenance()

    result: Final = install_model_cost_map(json.dumps({"only": {"litellm_provider": "openai"}}).encode())

    assert isinstance(result, ModelCostMapReloadUnavailable)
    assert litellm.model_cost is installed
    assert litellm.model_cost == rows
    assert get_model_cost_map_provenance() == provenance


@pytest.mark.usefixtures("own_model_cost")
def test_a_reader_iterating_the_old_table_finishes_while_another_thread_installs():
    old: Final = litellm.model_cost
    installer: Final = threading.Thread(target=install_model_cost_map, args=(_BUNDLED,))
    walked: Final = iter(old.items())
    first: Final = next(walked)
    installer.start()
    installer.join()

    assert len((first, *walked)) == len(old)
    assert litellm.model_cost is not old


@pytest.mark.usefixtures("own_model_cost")
def test_get_model_info_follows_a_reload_and_a_registration():
    litellm.get_model_info("gpt-4o")
    install_model_cost_map(_document_with(**{"gpt-4o": {"litellm_provider": "openai", "input_cost_per_token": 0.5}}))
    assert litellm.get_model_info("gpt-4o")["input_cost_per_token"] == 0.5

    generation: Final = litellm.utils.get_model_cost_mutation_generation()
    litellm.register_model({"gpt-4o": {"input_cost_per_token": 0.25}})
    assert litellm.get_model_info("gpt-4o")["input_cost_per_token"] == 0.25
    assert litellm.utils.get_model_cost_mutation_generation() > generation


@pytest.mark.usefixtures("own_model_cost")
def test_the_registered_catalog_reads_the_document_under_registrations_and_never_request_scoped_rows():
    litellm.register_model({"lit-kept": {"litellm_provider": "openai", "input_cost_per_token": 1}})
    litellm.register_model({"lit-request-row": {"litellm_provider": "openai"}}, persist_across_reloads=False)

    for _ in range(2):
        catalog = catalog_with_runtime_registrations()
        assert catalog["lit-kept"]["input_cost_per_token"] == 1
        assert "lit-request-row" not in catalog
        assert catalog["gpt-4o"] == GetModelCostMap.loaded_model_cost_map()["gpt-4o"]
        install_model_cost_map(_BUNDLED)


@pytest.mark.usefixtures("own_model_cost")
def test_the_loaded_catalog_follows_installs_and_ignores_registrations():
    install_model_cost_map(_document_with(**{"lit-loaded": {"litellm_provider": "openai", "lit_field": 1}}))
    litellm.register_model({"lit-loaded": {"input_cost_per_token": 5}})

    assert GetModelCostMap.loaded_model_cost_map()["lit-loaded"] == {"litellm_provider": "openai", "lit_field": 1}
    assert "lit_field" in GetModelCostMap.loaded_model_cost_map_fields()

    install_model_cost_map(_document_with(**{"lit-loaded": {"litellm_provider": "openai", "lit_other_field": 2}}))

    assert GetModelCostMap.loaded_model_cost_map()["lit-loaded"] == {"litellm_provider": "openai", "lit_other_field": 2}
    assert "lit_field" not in GetModelCostMap.loaded_model_cost_map_fields()
    assert "lit_other_field" in GetModelCostMap.loaded_model_cost_map_fields()


@pytest.mark.usefixtures("own_model_cost")
@pytest.mark.parametrize("add_to_provider_model_sets", [True, False])
def test_register_model_adds_names_to_the_provider_sets_only_when_asked(add_to_provider_model_sets: bool):
    litellm.register_model(
        {"openrouter/lit-vendor/lit-model": {"litellm_provider": "openrouter", "mode": "chat"}},
        add_to_provider_model_sets=add_to_provider_model_sets,
    )

    assert ("lit-vendor/lit-model" in litellm.openrouter_models) is add_to_provider_model_sets


@pytest.mark.usefixtures("own_model_cost")
def test_install_hands_the_fallback_rules_to_the_generalization_lookup():
    rules: Final = [{"name": "lit-rule", "pattern": "^lit-family-", "model_info": {"litellm_provider": "openai"}}]
    previous: Final = list(get_fallback_generalization_rules())
    try:
        install_model_cost_map(_document_with(**{FALLBACK_GENERALIZATIONS_KEY: {"rules": rules}}))
        assert match_routing_generalization("lit-family-1") == "openai"
        assert FALLBACK_GENERALIZATIONS_KEY not in litellm.model_cost
    finally:
        set_fallback_generalizations(previous)


@pytest.mark.usefixtures("own_model_cost")
def test_alias_names_share_one_entry_after_an_install_and_a_registration():
    install_model_cost_map(
        _document_with(**{"lit-owner": {"litellm_provider": "openai", "mode": "chat", "aliases": ["lit-alias"]}})
    )
    assert litellm.model_cost["lit-alias"] is litellm.model_cost["lit-owner"]

    litellm.register_model({"lit-alias": {"input_cost_per_token": 3e-06}})

    entry: Final = litellm.model_cost["lit-owner"]
    assert litellm.model_cost["lit-alias"] is entry
    assert isinstance(entry, AliasedCostEntry)
    assert entry.registered_fields == frozenset({"input_cost_per_token"})
    assert runtime_model_cost_registrations("lit-owner", "openai") == (("lit-owner", {"input_cost_per_token": 3e-06}),)


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


# ---------------------------------------------------------------------------
# A second instance over a dict of its own
# ---------------------------------------------------------------------------


def _own_instance(document: Mapping[str, Mapping[str, object]]) -> tuple[NativeModelCostMap, dict]:
    instance: Final = NativeModelCostMap.bundled(0, 0.0)
    installed: Final = instance.install(json.dumps(document).encode())
    assert isinstance(installed, NativeInstalled)
    return instance, installed.rows


def test_an_instance_writes_only_the_dict_it_is_handed():
    instance, rows = _own_instance({"lit-own": {"litellm_provider": "openai", "mode": "chat"}})
    live: Final = dict(litellm.model_cost)

    written: Final = instance.register(rows, {"lit-own-new": {"litellm_provider": "openai"}}, True)

    assert written == ["lit-own-new"]
    assert "lit-own-new" in rows
    assert litellm.model_cost == live
    assert instance.find_key(rows, "LIT-OWN-NEW") == "lit-own-new"


def test_the_name_index_is_a_cache():
    instance, rows = _own_instance(
        {
            "lit-a": {"litellm_provider": "p1"},
            "LIT-B": {"litellm_provider": "p2"},
            "lit-c": {"litellm_provider": "p1"},
        }
    )

    def queries() -> tuple[list[str | None], list[list[str]]]:
        return (
            [instance.find_key(rows, name) for name in ("LIT-A", "lit-b", "missing")],
            [instance.keys_in(rows, provider) for provider in ("p1", "p2", "p3")],
        )

    before: Final = queries()
    instance.invalidate()

    assert queries() == before
    assert before == (["lit-a", "LIT-B", None], [["lit-a", "lit-c"], ["LIT-B"], []])


def test_a_key_written_straight_to_the_dict_is_found_until_the_next_install():
    instance, rows = _own_instance({"lit-a": {"litellm_provider": "p"}})
    rows.setdefault("Lit-Direct", {"litellm_provider": "p"})
    assert instance.find_key(rows, "lit-direct") == "Lit-Direct"

    reinstalled: Final = instance.install(json.dumps({"lit-a": {"litellm_provider": "p"}}).encode())
    assert isinstance(reinstalled, NativeInstalled)
    assert instance.find_key(reinstalled.rows, "lit-direct") is None


def test_a_key_swapped_in_for_a_popped_one_replaces_it_in_lookups():
    instance, rows = _own_instance({"lit-a": {"litellm_provider": "p"}, "Lit-Gone": {"litellm_provider": "p"}})
    assert instance.find_key(rows, "lit-gone") == "Lit-Gone"
    rows.pop("Lit-Gone")
    rows["Lit-New"] = {"litellm_provider": "p"}

    assert instance.find_key(rows, "lit-gone") is None
    assert instance.find_key(rows, "lit-new") == "Lit-New"


def test_a_rebound_dict_is_what_lookups_read_next():
    instance, rows = _own_instance({"lit-a": {"litellm_provider": "p"}})
    rebound: Final = {"Lit-Rebound": {"litellm_provider": "p"}}

    assert instance.find_key(rows, "lit-rebound") is None
    assert instance.find_key(rebound, "lit-rebound") == "Lit-Rebound"


def test_case_insensitive_lookups_go_through_the_live_dict(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "model_cost", {**litellm.model_cost, "Lit-Mixed-Case": {"litellm_provider": "p"}})

    assert get_model_cost_key("GPT-4O") == "gpt-4o"
    assert get_model_cost_key("lit-mixed-case") == "Lit-Mixed-Case"
    assert get_model_cost_key("lit-unknown") is None
