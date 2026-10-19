from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

import httpx
import pytest
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from pydantic import JsonValue

import litellm
from litellm.rust_bridge.host.secret_manager import capture_secret_manager, native_secret_manager_config
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
from litellm.secret_managers.cyberark_secret_manager import CyberArkSecretManager
from litellm.secret_managers.hashicorp_secret_manager import HashicorpSecretManager
from litellm.secret_managers.secret_manager_handler import get_secret_from_manager
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem
from tests._support.provider import SharedProvider, shared_provider
from tests._support.wire import Reply

if TYPE_CHECKING:
    from litellm.rust_bridge._native import _SecretManagerRuntime


@pytest.fixture(autouse=True)
def preserve_manager_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "secret_manager_client", litellm.secret_manager_client)
    monkeypatch.setattr(litellm, "_key_management_system", litellm._key_management_system)
    monkeypatch.setattr(litellm, "_key_management_settings", litellm._key_management_settings)


def _vault(monkeypatch: pytest.MonkeyPatch, address: str) -> HashicorpSecretManager:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setenv("HCP_VAULT_ADDR", address)
    monkeypatch.setenv("HCP_VAULT_TOKEN", "token")
    return HashicorpSecretManager()


def _vault_body(value: str) -> dict[str, JsonValue]:
    return {
        "data": {
            "data": {"key": value},
            "metadata": {
                "created_time": "",
                "deletion_time": "",
                "custom_metadata": None,
                "destroyed": False,
                "version": 1,
            },
        },
        "lease_id": "",
        "lease_duration": 0,
        "renewable": False,
        "request_id": "",
        "warnings": None,
        "wrap_info": None,
    }


def _reply(body: JsonValue | bytes, status: int = 200, gate: threading.Event | None = None) -> Reply:
    payload = body if isinstance(body, bytes) else json.dumps(body).encode()
    if gate is None:
        return Reply(status=status, body=payload)
    return Reply(status=status, chunks=(payload,), gate_before_first=gate)


def _replies(body: JsonValue | bytes, count: int, status: int = 200) -> tuple[Reply, ...]:
    return tuple(_reply(body, status) for _ in range(count))


def _native_handle(client: object) -> "_SecretManagerRuntime":
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    handle: Final = native._SecretManagerRuntime.from_client(client)
    assert handle is not None
    return handle


@pytest.mark.parametrize("system", ("aws_secret_manager", "hashicorp_vault", "cyberark"))
async def test_python_native_handle_reuses_backend_across_sync_and_async_reads(system: str) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_python_native_handle_reuses_backend_across_sync_and_async_reads"),
    ):
        environment: Final = {
            "AWS_REGION_NAME": "us-east-1",
            "AWS_ACCESS_KEY_ID": "captured-access",
            "AWS_SECRET_ACCESS_KEY": "captured-secret",
            "AWS_BEDROCK_RUNTIME_ENDPOINT": server.wire.url,
            "AZURE_KEY_VAULT_URI": server.wire.url,
            "AZURE_AD_TOKEN": "azure-token",
            "HCP_VAULT_ADDR": server.wire.url,
            "HCP_VAULT_TOKEN": "vault-token",
            "CYBERARK_API_BASE": server.wire.url,
            "CYBERARK_API_KEY": "cyberark-key",
            "CYBERARK_ACCOUNT": "account",
            "CYBERARK_USERNAME": "reader",
        }
        responses: Final = {
            "aws_secret_manager": {"SecretString": "native-value"},
            "azure_key_vault": {"value": "native-value"},
            "hashicorp_vault": _vault_body("native-value"),
            "cyberark": "native-value",
        }
        request_count: Final = 2 if system == "cyberark" else 1 if system == "hashicorp_vault" else 3
        if system == "cyberark":
            server.expect(_reply("authentication-token"), *_replies(responses[system], 1))
        else:
            server.expect(*_replies(responses[system], request_count))
        handle: Final = native._SecretManagerRuntime.from_config(system, environment, enterprise_enabled=True)
        expected: Final = '"native-value"' if system == "cyberark" else "native-value"
        assert handle.read_secret("KEY") == expected
        assert await handle.async_read_secret("KEY") == expected
        assert handle.read_secret("KEY") == expected
        requests: Final = server.received()
        assert len(requests) == request_count


def test_shared_initializer_captures_credentials_and_tracks_instance_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_shared_initializer_captures_credentials_and_tracks_instance_settings"),
    ):
        server.expect(*_replies({"SecretString": "native-value"}, 2))
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "captured-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "captured-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.wire.url)
        from litellm.proxy.proxy_server import ProxyConfig

        monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(aws_region_name="us-east-1"))
        ProxyConfig().initialize_secret_manager(KeyManagementSystem.AWS_SECRET_MANAGER.value)
        manager: Final = litellm.secret_manager_client
        assert isinstance(manager, AWSSecretsManagerV2)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "changed-access")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", "http://127.0.0.1:1")
        first: Final = native._SecretManagerRuntime.from_client(manager)
        assert first is not None
        assert native._SecretManagerRuntime.from_client(manager) is first
        assert first.read_secret("KEY") == "native-value"
        manager.aws_region_name = "us-west-2"
        second: Final = native._SecretManagerRuntime.from_client(manager)
        assert second is not None
        assert second is not first
        assert second.read_secret("KEY") == "native-value"
        requests: Final = server.received()
        assert len(requests) == 2
        assert "Credential=captured-access/" in requests[0].headers["authorization"]
        assert "/us-east-1/" in requests[0].headers["authorization"]
        assert "/us-west-2/" in requests[1].headers["authorization"]


def test_configuration_replacement_rebuilds_without_invalidating_existing_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as first_server,
        first_server.checked("test_configuration_replacement_rebuilds_without_invalidating_existing_handles:first"),
        shared_provider() as second_server,
        second_server.checked("test_configuration_replacement_rebuilds_without_invalidating_existing_handles:second"),
    ):
        first_server.expect(_reply(_vault_body("first")))
        second_server.expect(_reply(_vault_body("second")))
        manager: Final = _vault(monkeypatch, first_server.wire.url)
        first: Final = native._SecretManagerRuntime.from_client(manager)
        assert first is not None
        assert first.read_secret("KEY") == "first"
        manager.vault_addr = second_server.wire.url
        second: Final = native._SecretManagerRuntime.from_client(manager)
        assert second is not None
        assert second is not first
        assert second.read_secret("KEY") == "second"
        assert first.read_secret("KEY") == "first"
        first_requests: Final = first_server.received()
        second_requests: Final = second_server.received()
        assert len(first_requests) == len(second_requests) == 1


def test_custom_subclass_is_not_captured_for_a_native_handle() -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")

    class CustomManager(AWSSecretsManagerV2):
        def sync_read_secret(self, secret_name: str, primary_secret_name: str | None = None) -> str:
            return f"custom:{secret_name}"

    manager: Final = CustomManager(aws_region_name="us-east-1")
    assert native._SecretManagerRuntime.from_client(manager) is None
    assert manager.sync_read_secret("KEY") == "custom:KEY"


@pytest.mark.parametrize("system", ("google_secret_manager", "hashicorp_vault", "cyberark"))
def test_enterprise_backends_cannot_initialize_without_entitlement(system: str) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with pytest.raises(ValueError, match=r"[Ee]nterprise|[Pp]remium"):
        native._SecretManagerRuntime.from_config(system, {"CYBERARK_API_KEY": "key"})


def test_azure_factory_rejects_unencrypted_vault_endpoints() -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with pytest.raises(ValueError, match="https"):
        native._SecretManagerRuntime.from_config("azure_key_vault", {"AZURE_KEY_VAULT_URI": "http://127.0.0.1:1"})


def test_direct_builtin_constructor_can_use_native_without_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_direct_builtin_constructor_can_use_native_without_registration"),
    ):
        server.expect(_reply(_vault_body("native-value")))
        manager: Final = _vault(monkeypatch, server.wire.url)
        handle: Final = native._SecretManagerRuntime.from_client(manager)
        assert handle is not None
        assert handle.read_secret("KEY") == "native-value"
        requests: Final = server.received()
        assert len(requests) == 1


def test_uncaptured_clients_have_no_native_handle() -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    assert native._SecretManagerRuntime.from_client(SimpleNamespace()) is None


def test_native_failure_is_not_replayed_in_python(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_native_failure_is_not_replayed_in_python"):
        server.expect(_reply({"errors": ["denied"]}, status=403))
        manager: Final = _vault(monkeypatch, server.wire.url)
        handle: Final = _native_handle(manager)
        with pytest.raises(ValueError, match="HashiCorp Vault"):
            handle.read_secret("KEY")
        requests: Final = server.received()
        assert len(requests) == 1


@pytest.mark.parametrize("explicit_capture", (False, True))
def test_config_capture_preserves_credentials_and_excludes_unrelated_environment(
    monkeypatch: pytest.MonkeyPatch, explicit_capture: bool
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "initial-access")
    monkeypatch.setenv("SECRET_MANAGER_REFRESH_INTERVAL", "45")
    monkeypatch.setenv("UNRELATED_PRIVATE_TOKEN", "unrelated-secret")
    manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
    if explicit_capture:
        capture_secret_manager(manager, "aws_secret_manager")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "replacement-access")
    captured: Final = native_secret_manager_config(manager)
    assert captured is not None
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "replacement-access")

    retained: Final = native_secret_manager_config(manager)

    assert retained is captured
    assert retained.system == "aws_secret_manager"
    assert dict(retained.environment)["AWS_ACCESS_KEY_ID"] == "initial-access"
    assert dict(retained.environment)["SECRET_MANAGER_REFRESH_INTERVAL"] == "45"
    assert "UNRELATED_PRIVATE_TOKEN" not in dict(retained.environment)
    assert "initial-access" not in repr(retained)
    assert retained.settings == KeyManagementSettings().model_dump(mode="json")


def test_kms_sdk_client_capture_preserves_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3

    monkeypatch.setenv("AWS_REGION_NAME", "captured-region")
    client: Final = boto3.client(
        "kms", region_name="us-east-1", aws_access_key_id="access", aws_secret_access_key="secret"
    )
    try:
        capture_secret_manager(client, "aws_kms")
        monkeypatch.setenv("AWS_REGION_NAME", "replacement-region")

        captured: Final = native_secret_manager_config(client)

        assert captured is not None
        assert captured.system == "aws_kms"
        assert dict(captured.environment)["AWS_REGION_NAME"] == "captured-region"
    finally:
        client.close()


def test_same_named_custom_client_is_not_captured() -> None:
    class AWSSecretsManagerV2:
        pass

    client: Final = AWSSecretsManagerV2()
    capture_secret_manager(client, "aws_secret_manager")

    assert native_secret_manager_config(client) is None


@pytest.mark.parametrize("value", ("text", "", True, False, 42, 2**100, [1, "two"], {"nested": True}, None))
async def test_aws_primary_values_match_python_handler(monkeypatch: pytest.MonkeyPatch, value: JsonValue) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_aws_primary_values_match_python_handler"):
        server.expect(*_replies({"SecretString": json.dumps({"KEY": value})}, 3))
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.wire.url)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        settings: Final = KeyManagementSettings(primary_secret_name="primary")
        reference: Final = get_secret_from_manager(manager, "aws_secret_manager", "KEY", settings)
        handle: Final = native._SecretManagerRuntime.from_client(manager)
        assert handle is not None
        actual: Final = handle.read_secret("KEY", settings.model_dump(mode="json"))
        asynchronous: Final = await handle.read_secret_async("KEY", settings.model_dump(mode="json"))
        assert type(actual) is type(reference) is type(value)
        assert actual == reference == value
        assert type(asynchronous) is type(reference)
        assert asynchronous == reference
        requests: Final = server.received()
        assert len(requests) == 3
        assert tuple(json.loads(request.body) for request in requests) == ({"SecretId": "primary"},) * 3


@pytest.mark.parametrize("primary", (None, "primary"))
async def test_aws_primary_values_match_python_reads(monkeypatch: pytest.MonkeyPatch, primary: str | None) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_aws_primary_values_match_python_reads"):
        server.expect(*_replies({"SecretString": '{"KEY":"value"}'}, 3))
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.wire.url)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        settings: Final = KeyManagementSettings(primary_secret_name=primary)
        reference: Final = get_secret_from_manager(manager, "aws_secret_manager", "KEY", settings)
        handle: Final = _native_handle(manager)
        settings_dump: Final = settings.model_dump(mode="json")
        synchronous: Final = handle.read_secret("KEY", settings_dump)
        asynchronous: Final = await handle.read_secret_async("KEY", settings_dump)
        assert synchronous == reference
        assert asynchronous == reference
        requests: Final = server.received()
        assert len(requests) == 3
        assert tuple(json.loads(request.body) for request in requests) == ({"SecretId": primary or "KEY"},) * 3


@pytest.mark.parametrize("primary", (None, "primary"))
@pytest.mark.parametrize(
    ("status", "body"),
    (
        (400, {"__type": "ResourceNotFoundException"}),
        (403, {"__type": "AccessDeniedException"}),
        (500, {"__type": "InternalServiceError"}),
        (200, {"Name": "without-string"}),
        (200, {"SecretString": ""}),
    ),
)
def test_aws_absence_and_failed_reads_match_python_without_environment_fallback(
    monkeypatch: pytest.MonkeyPatch, primary: str | None, status: int, body: dict[str, str]
) -> None:
    with (
        shared_provider() as server,
        server.checked("test_aws_absence_and_failed_reads_match_python_without_environment_fallback"),
    ):
        server.expect(*_replies(body, 2, status))
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.wire.url)
        monkeypatch.setenv("KEY", "must-not-fall-back")
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        settings: Final = KeyManagementSettings(primary_secret_name=primary)
        reference: Final = get_secret_from_manager(manager, "aws_secret_manager", "KEY", settings)
        handle: Final = _native_handle(manager)
        actual: Final = handle.read_secret("KEY", settings.model_dump(mode="json"))
        assert actual == reference
        assert actual == ("" if primary is None and body.get("SecretString") == "" else None)
        requests: Final = server.received()
        assert len(requests) == 2


@pytest.mark.parametrize("document", ("{", "not-json", "[1]", "null", "true", "42", '"text"'))
async def test_aws_primary_json_errors_preserve_python_exception_details(
    monkeypatch: pytest.MonkeyPatch, document: str
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_aws_primary_json_errors_preserve_python_exception_details"):
        server.expect(*_replies({"SecretString": document}, 3))
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.wire.url)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        settings: Final = KeyManagementSettings(primary_secret_name="primary")
        with pytest.raises((json.JSONDecodeError, AttributeError)) as reference:
            get_secret_from_manager(manager, "aws_secret_manager", "KEY", settings)
        handle: Final = _native_handle(manager)
        settings_dump: Final = settings.model_dump(mode="json")
        with pytest.raises(type(reference.value)) as actual:
            handle.read_secret("KEY", settings_dump)
        assert actual.value.args == reference.value.args
        if isinstance(reference.value, json.JSONDecodeError):
            assert isinstance(actual.value, json.JSONDecodeError)
            assert (actual.value.doc, actual.value.pos) == (reference.value.doc, reference.value.pos)
        with pytest.raises(type(reference.value)) as asynchronous:
            await handle.read_secret_async("KEY", settings_dump)
        assert asynchronous.value.args == reference.value.args
        requests: Final = server.received()
        assert len(requests) == 3


async def test_public_aws_reads_preserve_coroutines_and_per_call_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as initial,
        initial.checked("test_public_aws_reads_preserve_coroutines_and_per_call_credentials:initial"),
        shared_provider() as selected,
        selected.checked("test_public_aws_reads_preserve_coroutines_and_per_call_credentials:selected"),
    ):
        selected.expect(*_replies({"SecretString": "value"}, 2))
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "environment-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "environment-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", initial.wire.url)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        options: Final = {
            "aws_region_name": "us-west-2",
            "aws_bedrock_runtime_endpoint": selected.wire.url,
            "aws_access_key_id": "operation-access",
            "aws_secret_access_key": "operation-secret",
            "aws_session_token": "operation-session",
        }
        pending: Final = manager.async_read_secret(secret_name="KEY", optional_params=dict(options), timeout=2)
        assert inspect.iscoroutine(pending)
        pending_requests: Final = selected.received()
        assert pending_requests == ()
        assert await asyncio.create_task(pending) == "value"
        async_requests: Final = selected.received()
        assert len(async_requests) == 1
        assert manager.sync_read_secret("KEY", dict(options), 2) == "value"
        sync_requests: Final = selected.received()
        assert len(sync_requests) == 1
        requests: Final = (*async_requests, *sync_requests)
        assert tuple(json.loads(request.body) for request in requests) == ({"SecretId": "KEY"},) * 2
        assert all(
            "Credential=operation-access/" in request.headers["authorization"]
            and "/us-west-2/" in request.headers["authorization"]
            and request.headers["x-amz-security-token"] == "operation-session"
            for request in requests
        )
        for request in requests:
            signed_headers: Final = request.headers["authorization"].split("SignedHeaders=")[1].split(",")[0].split(";")
            signed_request: Final = AWSRequest(
                method=request.method,
                url=selected.wire.url + request.target,
                data=request.body,
                headers={name: request.headers[name] for name in signed_headers},
            )
            signed_request.context["timestamp"] = request.headers["x-amz-date"]
            signer: Final = SigV4Auth(
                Credentials(
                    options["aws_access_key_id"], options["aws_secret_access_key"], options["aws_session_token"]
                ),
                "secretsmanager",
                options["aws_region_name"],
            )
            string_to_sign: Final = signer.string_to_sign(signed_request, signer.canonical_request(signed_request))
            assert request.headers["authorization"].split("Signature=")[1] == signer.signature(
                string_to_sign, signed_request
            )


async def test_public_aws_primary_reads_ignore_operation_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_aws_primary_reads_ignore_operation_overrides:server"),
        shared_provider() as unused,
        unused.checked("test_public_aws_primary_reads_ignore_operation_overrides:unused"),
    ):
        server.expect(*_replies({"SecretString": '{"KEY":true}'}, 2))
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.wire.url)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        options: Final = {"aws_bedrock_runtime_endpoint": unused.wire.url}
        assert manager.sync_read_secret("KEY", options, 0, "primary") is True
        assert (
            await manager.async_read_secret("KEY", optional_params=options, timeout=0, primary_secret_name="primary")
            is True
        )
        requests: Final = server.received()
        unused_requests: Final = unused.received()
        assert len(requests) == 2
        assert unused_requests == ()
        assert tuple(json.loads(request.body) for request in requests) == ({"SecretId": "primary"},) * 2


async def test_public_aws_bootstrap_names_only_bypass_sync_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_public_aws_bootstrap_names_only_bypass_sync_reads"):
        server.expect(_reply({"SecretString": "remote-access"}))
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "environment-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "environment-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.wire.url)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        assert manager.sync_read_secret("AWS_ACCESS_KEY_ID") == "environment-access"
        bootstrap_requests: Final = server.received()
        assert bootstrap_requests == ()
        assert await manager.async_read_secret("AWS_ACCESS_KEY_ID") == "remote-access"
        provider_requests: Final = server.received()
        assert len(provider_requests) == 1


@pytest.mark.parametrize("timeout", (0.05, httpx.Timeout(1, read=0.05)))
async def test_public_aws_read_timeouts_follow_the_python_http_handler(
    monkeypatch: pytest.MonkeyPatch, timeout: float | httpx.Timeout
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_public_aws_read_timeouts_follow_the_python_http_handler"):
        gate: Final = threading.Event()
        server.expect(*(_reply({"SecretString": "too-late"}, gate=gate) for _ in range(2)))
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.wire.url)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        try:
            assert manager.sync_read_secret("KEY", timeout=timeout) is None
            assert await manager.async_read_secret("KEY", timeout=timeout) is None
            requests: Final = server.received()
            assert len(requests) == 2
        finally:
            gate.set()


async def test_public_vault_reads_keep_overrides_cache_and_coroutines(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_public_vault_reads_keep_overrides_cache_and_coroutines"):
        server.expect(*_replies(_vault_body("value"), 2))
        manager: Final = _vault(monkeypatch, server.wire.url)
        options: Final = {"secret_manager_settings": {"mount": "team", "path_prefix": "keys", "data": "key"}}
        pending: Final = manager.async_read_secret("KEY", options)
        assert inspect.iscoroutine(pending)
        pending_requests: Final = server.received()
        assert pending_requests == ()
        assert await asyncio.create_task(pending) == "value"
        async_requests: Final = server.received()
        assert len(async_requests) == 1
        assert urlsplit(async_requests[0].target).path == "/v1/team/data/keys/KEY"
        assert manager.sync_read_secret(secret_name="KEY", optional_params=options) == "value"
        cached_requests: Final = server.received()
        assert cached_requests == ()
        assert manager.sync_read_secret("KEY") == "value"
        default_requests: Final = server.received()
        assert len(default_requests) == 1
        assert urlsplit(default_requests[0].target).path == "/v1/secret/data/KEY"


@pytest.mark.parametrize("status", (404, 403))
async def test_public_vault_failed_reads_return_none_without_replay(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_public_vault_failed_reads_return_none_without_replay"):
        server.expect(*_replies({"errors": ["unavailable"]}, 2, status))
        manager: Final = _vault(monkeypatch, server.wire.url)
        assert manager.sync_read_secret("KEY") is None
        sync_requests: Final = server.received()
        assert len(sync_requests) == 1
        assert await manager.async_read_secret("KEY") is None
        async_requests: Final = server.received()
        assert len(async_requests) == 1


async def test_public_cyberark_reads_reuse_authentication_and_cached_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    from litellm.proxy import proxy_server
    from litellm.secret_managers.cyberark_secret_manager import CyberArkSecretManager

    monkeypatch.setattr(proxy_server, "premium_user", True)
    with (
        shared_provider() as server,
        server.checked("test_public_cyberark_reads_reuse_authentication_and_cached_values"),
    ):
        server.expect(_reply("authentication-token"), _reply("secret-value"))
        monkeypatch.setenv("CYBERARK_API_BASE", server.wire.url)
        monkeypatch.setenv("CYBERARK_API_KEY", "api-key")
        monkeypatch.setenv("CYBERARK_ACCOUNT", "account")
        monkeypatch.setenv("CYBERARK_USERNAME", "reader")
        manager: Final = CyberArkSecretManager()
        pending: Final = manager.async_read_secret(secret_name="KEY", timeout=0)
        assert inspect.iscoroutine(pending)
        pending_requests: Final = server.received()
        assert pending_requests == ()
        assert await asyncio.create_task(pending) == '"secret-value"'
        requests: Final = server.received()
        assert len(requests) == 2
        assert manager.sync_read_secret("KEY", timeout=0) == "secret-value"
        cached_requests: Final = server.received()
        assert cached_requests == ()
        assert tuple(request.target for request in requests) == (
            "/authn/account/reader/authenticate",
            "/secrets/account/variable/KEY",
        )


def test_public_aws_bootstrap_read_does_not_hit_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "bootstrap-value")
    assert AWSSecretsManagerV2().sync_read_secret("AWS_ACCESS_KEY_ID") == "bootstrap-value"


@pytest.mark.parametrize("override", ("", 42))
def test_public_vault_prefix_overrides_match_python_string_conversion(
    monkeypatch: pytest.MonkeyPatch, override: str | int
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setenv("HCP_VAULT_PATH_PREFIX", "default-prefix")
    with (
        shared_provider() as server,
        server.checked("test_public_vault_prefix_overrides_match_python_string_conversion"),
    ):
        server.expect(_reply(_vault_body("value")))
        manager: Final = _vault(monkeypatch, server.wire.url)
        assert manager.sync_read_secret("KEY", {"path_prefix": override}) == "value"
        requests: Final = server.received()
        assert len(requests) == 1
        assert urlsplit(requests[0].target).path == (
            f"/v1/secret/data/{override}/KEY" if override else "/v1/secret/data/KEY"
        )


def _cyberark(monkeypatch: pytest.MonkeyPatch, address: str) -> CyberArkSecretManager:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setenv("CYBERARK_API_BASE", address)
    monkeypatch.setenv("CYBERARK_API_KEY", "api-key")
    monkeypatch.setenv("CYBERARK_ACCOUNT", "account")
    monkeypatch.setenv("CYBERARK_USERNAME", "reader")
    return CyberArkSecretManager()


async def test_public_cyberark_writes_and_deletes_share_the_read_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_public_cyberark_writes_and_deletes_share_the_read_cache"):
        server.expect(
            _reply(b"token"),
            _reply(b"old"),
            _reply({}),
            _reply({}),
            _reply(b"provider-after-delete"),
        )
        manager: Final = _cyberark(monkeypatch, server.wire.url)
        assert manager.sync_read_secret("KEY") == "old"
        read_requests: Final = server.received()
        assert len(read_requests) == 2
        pending: Final = manager.async_write_secret(
            "KEY",
            "new-value",
            "ignored",
            {"ignored": object()},
            0,
            {"ignored": object()},
        )
        assert inspect.iscoroutine(pending)
        write_pending_requests: Final = server.received()
        assert write_pending_requests == ()
        assert await asyncio.create_task(pending) == {
            "status": "success",
            "message": "Secret KEY written successfully",
        }
        write_requests: Final = server.received()
        assert len(write_requests) == 2
        assert manager.sync_read_secret("KEY") == "new-value"
        cache_requests: Final = server.received()
        assert cache_requests == ()
        assert await manager.async_read_secret("KEY") == "new-value"
        async_cache_requests: Final = server.received()
        assert async_cache_requests == ()
        assert await manager.async_delete_secret(secret_name="KEY", recovery_window_in_days=None, timeout=0) == {
            "status": "not_supported",
            "message": "CyberArk Conjur does not support direct secret deletion. Use policy updates to remove variables.",
        }
        deletion_requests: Final = server.received()
        assert deletion_requests == ()
        assert manager.sync_read_secret("KEY") == "provider-after-delete"
        after_delete_requests: Final = server.received()
        assert len(after_delete_requests) == 1
        requests: Final = (*read_requests, *write_requests, *after_delete_requests)
        assert tuple(request.target for request in requests) == (
            "/authn/account/reader/authenticate",
            "/secrets/account/variable/KEY",
            "/policies/account/policy/root",
            "/secrets/account/variable/KEY",
            "/secrets/account/variable/KEY",
        )
        assert write_requests[1].body == b"new-value"


@pytest.mark.parametrize("status", (401, 403, 500))
@pytest.mark.parametrize("authentication", (False, True))
async def test_public_cyberark_write_errors_match_python_without_http_retries(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    authentication: bool,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_cyberark_write_errors_match_python_without_http_retries"),
    ):
        responses: Final = (
            (_reply({}, status),) * 2
            if authentication
            else (
                _reply(b"token"),
                _reply({}),
                _reply({}, status),
            )
        )
        server.expect(*(responses * 2))
        reference: Final = await _cyberark(monkeypatch, server.wire.url).async_write_secret("KEY", "value")
        native_handle: Final = _native_handle(_cyberark(monkeypatch, server.wire.url))
        actual: Final = await native_handle.async_write_secret("KEY", "value")
        assert actual == reference
        assert tuple(actual) == tuple(reference)
        assert actual["status"] == "error"
        assert str(status) in actual["message"]
        requests: Final = server.received()
        assert len(requests) == len(responses) * 2


async def test_public_cyberark_write_recovers_from_initial_policy_authentication_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_cyberark_write_recovers_from_initial_policy_authentication_failure"),
    ):
        server.expect(_reply({}, 401), _reply(b"token"), _reply({}))
        manager: Final = _cyberark(monkeypatch, server.wire.url)
        assert await manager.async_write_secret("KEY", "value") == {
            "status": "success",
            "message": "Secret KEY written successfully",
        }
        requests: Final = server.received()
        assert len(requests) == 3
        assert tuple(request.target for request in requests) == (
            "/authn/account/reader/authenticate",
            "/authn/account/reader/authenticate",
            "/secrets/account/variable/KEY",
        )


@pytest.mark.parametrize("name", ("../KEY", "line\nKEY", "a b"))
async def test_public_cyberark_write_rejects_unsafe_names_before_authentication(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_cyberark_write_rejects_unsafe_names_before_authentication"),
    ):
        manager: Final = _cyberark(monkeypatch, server.wire.url)
        assert await manager.async_write_secret(name, "value") == {
            "status": "error",
            "message": f"Invalid secret_name {name!r}",
        }
        requests: Final = server.received()
        assert requests == ()


@pytest.mark.parametrize("same_name", (False, True))
async def test_public_cyberark_rotation_returns_the_write_response_and_retains_old_alias(
    monkeypatch: pytest.MonkeyPatch,
    same_name: bool,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_cyberark_rotation_returns_the_write_response_and_retains_old_alias"),
    ):
        server.expect(_reply(b"token"), _reply(b"old-value"), _reply({}), _reply({}))
        manager: Final = _cyberark(monkeypatch, server.wire.url)
        new_name: Final = "OLD" if same_name else "NEW"
        pending: Final = manager.async_rotate_secret("OLD", new_name, "new-value", {"ignored": object()}, 0)
        assert inspect.iscoroutine(pending)
        pending_requests: Final = server.received()
        assert pending_requests == ()
        assert await asyncio.create_task(pending) == {
            "status": "success",
            "message": f"Secret {new_name} written successfully",
        }
        requests: Final = server.received()
        assert len(requests) == 4
        assert tuple(request.method for request in requests) == ("POST", "GET", "POST", "POST")
        assert requests[3].body == b"new-value"


@pytest.mark.parametrize("replacement", (None, b"wrong-value"))
async def test_public_native_cyberark_rotation_requires_a_fresh_matching_replacement(
    monkeypatch: pytest.MonkeyPatch,
    replacement: bytes | None,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_native_cyberark_rotation_requires_a_fresh_matching_replacement"),
    ):
        server.expect(
            _reply(b"token"),
            _reply(b"old-value"),
            _reply({}),
            _reply({}),
            _reply(replacement, 404 if replacement is None else 200),
        )
        manager: Final = _cyberark(monkeypatch, server.wire.url)
        handle: Final = _native_handle(manager)
        message: Final = "Failed to verify new secret NEW" if replacement is None else "New secret value mismatch"
        with pytest.raises(ValueError, match=message):
            await handle.async_rotate_secret("OLD", "NEW", "new-value")
        assert handle.sync_read_secret("OLD") == "old-value"
        requests: Final = server.received()
        assert len(requests) == 5
        assert tuple(request.target for request in requests) == (
            "/authn/account/reader/authenticate",
            "/secrets/account/variable/OLD",
            "/policies/account/policy/root",
            "/secrets/account/variable/NEW",
            "/secrets/account/variable/NEW",
        )


async def test_public_cyberark_cached_authentication_does_not_retry_denied_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_cyberark_cached_authentication_does_not_retry_denied_reads"),
    ):
        server.expect(_reply(b"token"), _reply(b"value"), _reply({}, 401))
        manager: Final = _cyberark(monkeypatch, server.wire.url)
        assert manager.sync_read_secret("OLD") == "value"
        initial_requests: Final = server.received()
        assert len(initial_requests) == 2
        assert await manager.async_read_secret("NEW") is None
        denied_requests: Final = server.received()
        assert len(denied_requests) == 1


async def test_cyberark_handler_errors_match_python_after_cached_authentication_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_cyberark_handler_errors_match_python_after_cached_authentication_is_denied"),
    ):
        server.expect(*((_reply(b"token"), _reply(b"value"), _reply({}, 401)) * 2))
        reference_manager: Final = _cyberark(monkeypatch, server.wire.url)
        assert get_secret_from_manager(reference_manager, "cyberark", "OLD") == "value"
        with pytest.raises(ValueError, match="No secret found in CyberArk Secret Manager for NEW") as reference:
            get_secret_from_manager(reference_manager, "cyberark", "NEW")
        reference_requests: Final = server.received()
        assert len(reference_requests) == 3
        native_handle: Final = _native_handle(_cyberark(monkeypatch, server.wire.url))
        assert native_handle.read_secret("OLD") == "value"
        with pytest.raises(ValueError, match="No secret found in CyberArk Secret Manager for NEW") as actual:
            native_handle.read_secret("NEW")
        assert actual.value.args == reference.value.args
        native_requests: Final = server.received()
        assert len(native_requests) == 3


async def test_public_cyberark_connection_errors_match_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_public_cyberark_connection_errors_match_python"):
        address: Final = server.wire.url
        requests: Final = server.received()
        assert requests == ()
    reference: Final = await _cyberark(monkeypatch, address).async_write_secret("KEY", "value")
    native_handle: Final = _native_handle(_cyberark(monkeypatch, address))
    actual: Final = await native_handle.async_write_secret("KEY", "value")
    assert actual == reference
    assert actual["status"] == "error"


async def test_public_native_cyberark_rotation_stops_after_a_failed_write(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_public_native_cyberark_rotation_stops_after_a_failed_write"):
        server.expect(_reply(b"token"), _reply(b"old-value"), _reply({}), _reply({}, 401))
        handle: Final = _native_handle(_cyberark(monkeypatch, server.wire.url))
        response: Final = await handle.async_rotate_secret("OLD", "NEW", "new-value")
        assert response["status"] == "error"
        assert "401" in response["message"]
        assert handle.sync_read_secret("OLD") == "old-value"
        requests: Final = server.received()
        assert len(requests) == 4
        assert tuple(request.method for request in requests) == ("POST", "GET", "POST", "POST")


@pytest.mark.parametrize("description", (None, "", "purpose"))
async def test_public_vault_writes_preserve_complete_responses_and_request_fields(
    monkeypatch: pytest.MonkeyPatch,
    description: str | None,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_vault_writes_preserve_complete_responses_and_request_fields"),
    ):
        response: Final = {
            "request_id": "test-request",
            "data": {"version": 2, "custom_metadata": {"large": 2**100}},
            "warnings": ["test-warning"],
            "unknown_field": {"nested": [None, True, ""]},
        }
        server.expect(_reply(response))
        manager: Final = _vault(monkeypatch, server.wire.url)
        options: Final = {
            "secret_manager_settings": {"namespace": "team", "mount": "kv", "path_prefix": "app", "data": "token"}
        }
        pending: Final = manager.async_write_secret("KEY", "value", description, options, 2, {"ignored": object()})
        assert inspect.iscoroutine(pending)
        pending_requests: Final = server.received()
        assert pending_requests == ()
        result: Final = await asyncio.create_task(pending)
        assert result == response
        assert tuple(result) == tuple(response)
        requests: Final = server.received()
        assert len(requests) == 1
        request: Final = requests[0]
        assert request.method == "POST"
        assert request.headers["x-vault-token"] == "token"
        namespace: Final = request.headers.get("x-vault-namespace")
        assert request.target == ("/v1/kv/data/app/KEY" if namespace else "/v1/team/kv/data/app/KEY")
        assert namespace in (None, "team")
        assert json.loads(request.body) == {
            "data": {"token": "value", **({"description": description} if description else {})},
        }
        assert options == {
            "secret_manager_settings": {"namespace": "team", "mount": "kv", "path_prefix": "app", "data": "token"}
        }


@pytest.mark.parametrize("operation", ("write", "delete"))
@pytest.mark.parametrize("status", (400, 403, 500))
async def test_public_vault_mutation_http_errors_match_python_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    status: int,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_vault_mutation_http_errors_match_python_without_retry"),
    ):
        server.expect(*_replies({"errors": ["denied"]}, 2, status))
        options: Final = {"namespace": "team", "mount": "kv", "path_prefix": "prefix"}
        reference: Final = (
            await _vault(monkeypatch, server.wire.url).async_write_secret("KEY", "value", optional_params=options)
            if operation == "write"
            else await _vault(monkeypatch, server.wire.url).async_delete_secret("KEY", optional_params=options)
        )
        native_handle: Final = _native_handle(_vault(monkeypatch, server.wire.url))
        actual: Final = (
            await native_handle.async_write_secret("KEY", "value", optional_params=options)
            if operation == "write"
            else await native_handle.async_delete_secret("KEY", optional_params=options)
        )
        assert actual == reference
        assert tuple(actual) == tuple(reference)
        assert actual["status"] == "error"
        assert str(status) in actual["message"]
        requests: Final = server.received()
        assert len(requests) == 2


@pytest.mark.parametrize("body", (b"{", b"", b"null", b"[1,2]", b'{"large":1267650600228229401496703205376}'))
async def test_public_vault_write_response_conversion_matches_python(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_public_vault_write_response_conversion_matches_python"):
        server.expect(_reply(body), _reply(body))
        reference: Final = await _vault(monkeypatch, server.wire.url).async_write_secret("KEY", "value")
        native_handle: Final = _native_handle(_vault(monkeypatch, server.wire.url))
        actual: Final = await native_handle.async_write_secret("KEY", "value")
        assert type(actual) is type(reference)
        assert actual == reference
        requests: Final = server.received()
        assert len(requests) == 2


async def test_public_vault_deletion_invalidates_cached_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_public_vault_deletion_invalidates_cached_fields"):
        server.expect(_reply(_vault_body("old")), _reply(b"", 204), _reply(_vault_body("after-delete")))
        manager: Final = _vault(monkeypatch, server.wire.url)
        assert manager.sync_read_secret("KEY") == "old"
        read_requests: Final = server.received()
        assert len(read_requests) == 1
        pending: Final = manager.async_delete_secret("KEY", None, {"ignored": object()}, 2)
        assert inspect.iscoroutine(pending)
        pending_requests: Final = server.received()
        assert pending_requests == ()
        assert await asyncio.create_task(pending) == {"status": "success", "message": "Secret KEY deleted successfully"}
        delete_requests: Final = server.received()
        assert len(delete_requests) == 1
        assert await manager.async_read_secret("KEY") == "after-delete"
        final_requests: Final = server.received()
        assert len(final_requests) == 1
        assert tuple(request.method for request in (*read_requests, *delete_requests, *final_requests)) == (
            "GET",
            "DELETE",
            "GET",
        )


@pytest.mark.parametrize("same_name", (False, True))
@pytest.mark.parametrize("delete_status", (204, 403))
async def test_public_vault_rotation_preserves_response_and_best_effort_deletion(
    monkeypatch: pytest.MonkeyPatch,
    same_name: bool,
    delete_status: int,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_vault_rotation_preserves_response_and_best_effort_deletion"),
    ):
        response: Final = {"request_id": "write-id", "data": {"version": 3}, "extra": [1, 2]}
        server.expect(
            _reply(b"current-existence-is-status-only"),
            _reply(response),
            _reply(_vault_body("replacement")),
            *(() if same_name else (_reply(b"", delete_status),)),
        )
        manager: Final = _vault(monkeypatch, server.wire.url)
        new_name: Final = "OLD" if same_name else "NEW"
        pending: Final = manager.async_rotate_secret("OLD", new_name, "replacement", timeout=2)
        assert inspect.iscoroutine(pending)
        pending_requests: Final = server.received()
        assert pending_requests == ()
        assert await asyncio.create_task(pending) == response
        requests: Final = server.received()
        assert len(requests) == (3 if same_name else 4)
        assert tuple(request.method for request in requests) == (
            ("GET", "POST", "GET") if same_name else ("GET", "POST", "GET", "DELETE")
        )
        assert json.loads(requests[1].body) == {
            "data": {"key": "replacement", "description": "Rotated from OLD"},
        }
        assert urlsplit(requests[2].target).path == f"/v1/secret/data/{new_name}"


@pytest.mark.parametrize("stage", ("current", "write", "verify"))
@pytest.mark.parametrize("status", (404, 403, 500))
async def test_public_vault_rotation_failure_messages_and_request_counts_match_python(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    status: int,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_vault_rotation_failure_messages_and_request_counts_match_python"),
    ):
        before: Final = (
            ()
            if stage == "current"
            else (
                (_reply(_vault_body("old")),)
                if stage == "write"
                else (
                    _reply(_vault_body("old")),
                    _reply({"data": {"version": 2}}),
                )
            )
        )
        responses: Final = (*before, _reply({"errors": ["denied"]}, status))
        server.expect(*(responses * 2))
        reference: Final = await _vault(monkeypatch, server.wire.url).async_rotate_secret("OLD", "NEW", "value")
        native_handle: Final = _native_handle(_vault(monkeypatch, server.wire.url))
        actual: Final = await native_handle.async_rotate_secret("OLD", "NEW", "value")
        assert actual == reference
        assert actual["status"] == "error"
        expected_paths: Final = (
            ("/v1/secret/data/OLD",)
            + (("/v1/secret/data/NEW",) if stage != "current" else ())
            + (("/v1/secret/data/NEW",) if stage == "verify" else ())
        )
        requests: Final = server.received()
        assert len(requests) == len(responses) * 2
        assert tuple(urlsplit(request.target).path for request in requests) == expected_paths * 2


@pytest.mark.parametrize("value", (None, "different", True, 42, 2**100, [1, "two"], [2**100], {"nested": 2**100}))
async def test_public_vault_rotation_mismatches_do_not_delete_the_old_alias(
    monkeypatch: pytest.MonkeyPatch,
    value: JsonValue,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_vault_rotation_mismatches_do_not_delete_the_old_alias"),
    ):
        responses: Final = (
            _reply(_vault_body("old")),
            _reply({"data": {"version": 2}}),
            _reply({"data": {"data": {"key": value}}}),
        )
        server.expect(*(responses * 2))
        reference: Final = await _vault(monkeypatch, server.wire.url).async_rotate_secret("OLD", "NEW", "value")
        native_handle: Final = _native_handle(_vault(monkeypatch, server.wire.url))
        actual: Final = await native_handle.async_rotate_secret("OLD", "NEW", "value")
        assert actual == reference
        assert actual["status"] == "error"
        requests: Final = server.received()
        assert len(requests) == 6
        assert all(request.method != "DELETE" for request in requests)


@pytest.mark.parametrize("operation", ("write", "delete", "rotate"))
async def test_public_vault_mutation_timeouts_match_python(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_public_vault_mutation_timeouts_match_python"):
        gate: Final = threading.Event()
        delayed_reply: Final = _reply(_vault_body("value"), gate=gate)
        server.expect(delayed_reply, delayed_reply)
        reference_manager: Final = _vault(monkeypatch, server.wire.url)
        try:
            reference: Final = await {
                "write": partial(reference_manager.async_write_secret, "KEY", "value"),
                "delete": partial(reference_manager.async_delete_secret, "KEY"),
                "rotate": partial(reference_manager.async_rotate_secret, "OLD", "NEW", "value"),
            }[operation](timeout=0.05)
            native_handle: Final = _native_handle(_vault(monkeypatch, server.wire.url))
            actual: Final = await {
                "write": partial(native_handle.async_write_secret, "KEY", "value"),
                "delete": partial(native_handle.async_delete_secret, "KEY"),
                "rotate": partial(native_handle.async_rotate_secret, "OLD", "NEW", "value"),
            }[operation](timeout=0.05)
            if operation == "write":
                assert isinstance(actual["message"], str)
                assert isinstance(reference["message"], str)
                pattern: Final = r"time taken=(\d+(?:\.\d+)?) seconds"
                assert re.sub(pattern, "time taken=<elapsed> seconds", actual["message"]) == re.sub(
                    pattern,
                    "time taken=<elapsed> seconds",
                    reference["message"],
                )
                elapsed: Final = re.search(pattern, actual["message"])
                assert elapsed is not None
                assert float(elapsed[1]) >= 0.05
            else:
                assert actual == reference
            assert actual["status"] == "error"
            requests: Final = server.received()
            assert len(requests) == 2
        finally:
            gate.set()


@pytest.mark.parametrize("operation", ("write", "delete", "rotate"))
async def test_public_vault_unsafe_names_fail_before_authentication(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_public_vault_unsafe_names_fail_before_authentication"):
        manager: Final = _vault(monkeypatch, server.wire.url)
        result: Final = await {
            "write": partial(manager.async_write_secret, "../KEY", "value"),
            "delete": partial(manager.async_delete_secret, "../KEY"),
            "rotate": partial(manager.async_rotate_secret, "../KEY", "NEW", "value"),
        }[operation]()
        assert result == {"status": "error", "message": "Invalid secret_name '../KEY'"}
        requests: Final = server.received()
        assert requests == ()


@pytest.mark.parametrize("operation", ("write", "delete", "rotate"))
async def test_public_vault_authentication_errors_match_python(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setenv("HCP_VAULT_APPROLE_ROLE_ID", "role")
    monkeypatch.setenv("HCP_VAULT_APPROLE_SECRET_ID", "secret-id")
    with shared_provider() as server, server.checked("test_public_vault_authentication_errors_match_python"):
        server.expect(*_replies({"errors": ["denied"]}, 2, 403))
        reference_manager: Final = _vault(monkeypatch, server.wire.url)
        reference: Final = await {
            "write": partial(reference_manager.async_write_secret, "KEY", "value"),
            "delete": partial(reference_manager.async_delete_secret, "KEY"),
            "rotate": partial(reference_manager.async_rotate_secret, "OLD", "NEW", "value"),
        }[operation]()
        native_handle: Final = _native_handle(_vault(monkeypatch, server.wire.url))
        actual: Final = await {
            "write": partial(native_handle.async_write_secret, "KEY", "value"),
            "delete": partial(native_handle.async_delete_secret, "KEY"),
            "rotate": partial(native_handle.async_rotate_secret, "OLD", "NEW", "value"),
        }[operation]()
        assert actual == reference
        assert actual["status"] == "error"
        requests: Final = server.received()
        assert len(requests) == 2
        assert tuple(request.target for request in requests) == ("/v1/auth/approle/login",) * 2


async def test_public_vault_rotation_stops_on_a_success_response_containing_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_vault_rotation_stops_on_a_success_response_containing_an_error"),
    ):
        result: Final = {"status": "error", "message": "write rejected", "extra": 2**100}
        server.expect(*((_reply(_vault_body("old")), _reply(result)) * 2))
        reference: Final = await _vault(monkeypatch, server.wire.url).async_rotate_secret("OLD", "NEW", "value")
        native_handle: Final = _native_handle(_vault(monkeypatch, server.wire.url))
        actual: Final = await native_handle.async_rotate_secret("OLD", "NEW", "value")
        assert actual == reference == result
        requests: Final = server.received()
        assert len(requests) == 4
        assert tuple(request.method for request in requests) == ("GET", "POST", "GET", "POST")


async def test_public_native_vault_write_invalidates_stale_cached_values(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with shared_provider() as server, server.checked("test_public_native_vault_write_invalidates_stale_cached_values"):
        server.expect(
            _reply(_vault_body("old")),
            _reply({"data": {"version": 2}}),
            _reply(_vault_body("new")),
        )
        handle: Final = _native_handle(_vault(monkeypatch, server.wire.url))
        assert handle.sync_read_secret("KEY") == "old"
        read_requests: Final = server.received()
        assert len(read_requests) == 1
        assert await handle.async_write_secret("KEY", "new") == {"data": {"version": 2}}
        write_requests: Final = server.received()
        assert len(write_requests) == 1
        assert handle.sync_read_secret("KEY") == "new"
        refresh_requests: Final = server.received()
        assert len(refresh_requests) == 1
        assert tuple(request.method for request in (*read_requests, *write_requests, *refresh_requests)) == (
            "GET",
            "POST",
            "GET",
        )


async def test_public_native_vault_write_rejects_description_overwriting_the_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_native_vault_write_rejects_description_overwriting_the_secret"),
    ):
        handle: Final = _native_handle(_vault(monkeypatch, server.wire.url))
        assert await handle.async_write_secret("KEY", "value", "description", {"data": "description"}) == {
            "status": "error",
            "message": "HashiCorp Vault data key conflicts with description",
        }
        requests: Final = server.received()
        assert requests == ()


@pytest.mark.parametrize(
    "body", (None, [], 42, 2**100, {"data": 2**100}, {"data": None}, {"data": {"data": []}}, {}, {"data": {}})
)
async def test_public_vault_rotation_preserves_malformed_verification_errors(
    monkeypatch: pytest.MonkeyPatch,
    body: JsonValue,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with (
        shared_provider() as server,
        server.checked("test_public_vault_rotation_preserves_malformed_verification_errors"),
    ):
        responses: Final = (
            _reply(_vault_body("old")),
            _reply({"data": {"version": 2}}),
            _reply(body),
        )
        server.expect(*(responses * 2))
        reference: Final = await _vault(monkeypatch, server.wire.url).async_rotate_secret("OLD", "NEW", "value")
        native_handle: Final = _native_handle(_vault(monkeypatch, server.wire.url))
        actual: Final = await native_handle.async_rotate_secret("OLD", "NEW", "value")
        assert actual == reference
        assert actual["status"] == "error"
        requests: Final = server.received()
        assert len(requests) == 6
        assert all(request.method != "DELETE" for request in requests)
