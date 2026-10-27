from __future__ import annotations

import asyncio
import inspect
import json
import re
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
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service

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


def _vault_body(value: str) -> dict[str, object]:
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


def _native_handle(client: object) -> "_SecretManagerRuntime":
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    handle: Final = native._SecretManagerRuntime.from_client(client)
    assert handle is not None
    return handle


@pytest.mark.parametrize("system", ("aws_secret_manager", "hashicorp_vault", "cyberark"))
async def test_python_native_handle_reuses_backend_across_sync_and_async_reads(system: str) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        environment: Final = {
            "AWS_REGION_NAME": "us-east-1",
            "AWS_ACCESS_KEY_ID": "captured-access",
            "AWS_SECRET_ACCESS_KEY": "captured-secret",
            "AWS_BEDROCK_RUNTIME_ENDPOINT": server.base_url,
            "AZURE_KEY_VAULT_URI": server.base_url,
            "AZURE_AD_TOKEN": "azure-token",
            "HCP_VAULT_ADDR": server.base_url,
            "HCP_VAULT_TOKEN": "vault-token",
            "CYBERARK_API_BASE": server.base_url,
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
        server.default_response = ResponseSpec(body=responses[system])
        server.expected_requests = 2 if system == "cyberark" else 1 if system == "hashicorp_vault" else 3
        if system == "cyberark":
            server.enqueue(ResponseSpec(body="authentication-token"))
        handle: Final = native._SecretManagerRuntime.from_config(system, environment, enterprise_enabled=True)
        expected: Final = '"native-value"' if system == "cyberark" else "native-value"
        assert handle.read_secret("KEY") == expected
        assert await handle.async_read_secret("KEY") == expected
        assert handle.read_secret("KEY") == expected


def test_shared_initializer_captures_credentials_and_tracks_instance_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(body={"SecretString": "native-value"})
        server.expected_requests = 2
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "captured-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "captured-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.base_url)
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
        assert "Credential=captured-access/" in server.requests[0].headers["authorization"]
        assert "/us-east-1/" in server.requests[0].headers["authorization"]
        assert "/us-west-2/" in server.requests[1].headers["authorization"]


def test_configuration_replacement_rebuilds_without_invalidating_existing_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as first_server, recording_service() as second_server:
        first_server.default_response = ResponseSpec(body=_vault_body("first"))
        second_server.default_response = ResponseSpec(body=_vault_body("second"))
        manager: Final = _vault(monkeypatch, first_server.base_url)
        first: Final = native._SecretManagerRuntime.from_client(manager)
        assert first is not None
        assert first.read_secret("KEY") == "first"
        manager.vault_addr = second_server.base_url
        second: Final = native._SecretManagerRuntime.from_client(manager)
        assert second is not None
        assert second is not first
        assert second.read_secret("KEY") == "second"
        assert first.read_secret("KEY") == "first"


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
    with recording_service() as server:
        server.default_response = ResponseSpec(body=_vault_body("native-value"))
        manager: Final = _vault(monkeypatch, server.base_url)
        handle: Final = native._SecretManagerRuntime.from_client(manager)
        assert handle is not None
        assert handle.read_secret("KEY") == "native-value"


def test_uncaptured_clients_have_no_native_handle() -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    assert native._SecretManagerRuntime.from_client(SimpleNamespace()) is None


def test_native_failure_is_not_replayed_in_python(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(status=403, body={"errors": ["denied"]})
        server.expected_requests = 1
        manager: Final = _vault(monkeypatch, server.base_url)
        handle: Final = _native_handle(manager)
        with pytest.raises(ValueError, match="HashiCorp Vault"):
            handle.read_secret("KEY")
        assert len(server.requests) == 1


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
    with recording_service() as server:
        server.default_response = ResponseSpec(body={"SecretString": json.dumps({"KEY": value})})
        server.expected_requests = 3
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.base_url)
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
        assert tuple(json.loads(request.raw_body) for request in server.requests) == ({"SecretId": "primary"},) * 3


@pytest.mark.parametrize("primary", (None, "primary"))
async def test_aws_primary_values_match_python_reads(monkeypatch: pytest.MonkeyPatch, primary: str | None) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(body={"SecretString": '{"KEY":"value"}'})
        server.expected_requests = 3
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.base_url)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        settings: Final = KeyManagementSettings(primary_secret_name=primary)
        reference: Final = get_secret_from_manager(manager, "aws_secret_manager", "KEY", settings)
        handle: Final = _native_handle(manager)
        settings_dump: Final = settings.model_dump(mode="json")
        synchronous: Final = handle.read_secret("KEY", settings_dump)
        asynchronous: Final = await handle.read_secret_async("KEY", settings_dump)
        assert synchronous == reference
        assert asynchronous == reference
        assert tuple(json.loads(request.raw_body) for request in server.requests) == ({"SecretId": primary or "KEY"},) * 3


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
    with recording_service() as server:
        server.default_response = ResponseSpec(status=status, body=body)
        server.expected_requests = 2
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.base_url)
        monkeypatch.setenv("KEY", "must-not-fall-back")
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        settings: Final = KeyManagementSettings(primary_secret_name=primary)
        reference: Final = get_secret_from_manager(manager, "aws_secret_manager", "KEY", settings)
        handle: Final = _native_handle(manager)
        actual: Final = handle.read_secret("KEY", settings.model_dump(mode="json"))
        assert actual == reference
        assert actual == ("" if primary is None and body.get("SecretString") == "" else None)


@pytest.mark.parametrize("document", ("{", "not-json", "[1]", "null", "true", "42", '"text"'))
async def test_aws_primary_json_errors_preserve_python_exception_details(
    monkeypatch: pytest.MonkeyPatch, document: str
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(body={"SecretString": document})
        server.expected_requests = 3
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.base_url)
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


async def test_public_aws_reads_preserve_coroutines_and_per_call_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as initial, recording_service() as selected:
        initial.expected_requests = 0
        selected.expected_requests = 2
        selected.default_response = ResponseSpec(body={"SecretString": "value"})
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "environment-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "environment-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", initial.base_url)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        options: Final = {
            "aws_region_name": "us-west-2",
            "aws_bedrock_runtime_endpoint": selected.base_url,
            "aws_access_key_id": "operation-access",
            "aws_secret_access_key": "operation-secret",
            "aws_session_token": "operation-session",
        }
        pending: Final = manager.async_read_secret(secret_name="KEY", optional_params=dict(options), timeout=2)
        assert inspect.iscoroutine(pending)
        assert selected.requests == []
        assert await asyncio.create_task(pending) == "value"
        assert manager.sync_read_secret("KEY", dict(options), 2) == "value"
        assert tuple(json.loads(request.raw_body) for request in selected.requests) == ({"SecretId": "KEY"},) * 2
        assert all(
            "Credential=operation-access/" in request.headers["authorization"]
            and "/us-west-2/" in request.headers["authorization"]
            and request.headers["x-amz-security-token"] == "operation-session"
            for request in selected.requests
        )
        for request in selected.requests:
            signed_headers: Final = request.headers["authorization"].split("SignedHeaders=")[1].split(",")[0].split(";")
            signed_request: Final = AWSRequest(
                method=request.method,
                url=selected.base_url + request.path,
                data=request.raw_body,
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
    with recording_service() as server, recording_service() as unused:
        server.default_response = ResponseSpec(body={"SecretString": '{"KEY":true}'})
        server.expected_requests = 2
        unused.expected_requests = 0
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.base_url)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        options: Final = {"aws_bedrock_runtime_endpoint": unused.base_url}
        assert manager.sync_read_secret("KEY", options, 0, "primary") is True
        assert (
            await manager.async_read_secret("KEY", optional_params=options, timeout=0, primary_secret_name="primary")
            is True
        )
        assert tuple(json.loads(request.raw_body) for request in server.requests) == ({"SecretId": "primary"},) * 2


async def test_public_aws_bootstrap_names_only_bypass_sync_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(body={"SecretString": "remote-access"})
        server.expected_requests = 1
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "environment-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "environment-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.base_url)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        assert manager.sync_read_secret("AWS_ACCESS_KEY_ID") == "environment-access"
        assert server.requests == []
        assert await manager.async_read_secret("AWS_ACCESS_KEY_ID") == "remote-access"


@pytest.mark.parametrize("timeout", (0.05, httpx.Timeout(1, read=0.05)))
async def test_public_aws_read_timeouts_follow_the_python_http_handler(
    monkeypatch: pytest.MonkeyPatch, timeout: float | httpx.Timeout
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(body={"SecretString": "too-late"}, delay=0.25)
        server.expected_requests = 2
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", server.base_url)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        assert manager.sync_read_secret("KEY", timeout=timeout) is None
        assert await manager.async_read_secret("KEY", timeout=timeout) is None


async def test_public_vault_reads_keep_overrides_cache_and_coroutines(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(body=_vault_body("value"))
        server.expected_requests = 2
        manager: Final = _vault(monkeypatch, server.base_url)
        options: Final = {"secret_manager_settings": {"mount": "team", "path_prefix": "keys", "data": "key"}}
        pending: Final = manager.async_read_secret("KEY", options)
        assert inspect.iscoroutine(pending)
        assert server.requests == []
        assert await asyncio.create_task(pending) == "value"
        assert manager.sync_read_secret(secret_name="KEY", optional_params=options) == "value"
        assert manager.sync_read_secret("KEY") == "value"
        assert urlsplit(server.requests[0].path).path == "/v1/team/data/keys/KEY"
        assert urlsplit(server.requests[1].path).path == "/v1/secret/data/KEY"


@pytest.mark.parametrize("status", (404, 403))
async def test_public_vault_failed_reads_return_none_without_replay(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(status=status, body={"errors": ["unavailable"]})
        server.expected_requests = 2
        manager: Final = _vault(monkeypatch, server.base_url)
        assert manager.sync_read_secret("KEY") is None
        assert await manager.async_read_secret("KEY") is None


async def test_public_cyberark_reads_reuse_authentication_and_cached_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    from litellm.proxy import proxy_server
    from litellm.secret_managers.cyberark_secret_manager import CyberArkSecretManager

    monkeypatch.setattr(proxy_server, "premium_user", True)
    with recording_service() as server:
        server.enqueue(ResponseSpec(body="authentication-token"))
        server.default_response = ResponseSpec(body="secret-value")
        server.expected_requests = 2
        monkeypatch.setenv("CYBERARK_API_BASE", server.base_url)
        monkeypatch.setenv("CYBERARK_API_KEY", "api-key")
        monkeypatch.setenv("CYBERARK_ACCOUNT", "account")
        monkeypatch.setenv("CYBERARK_USERNAME", "reader")
        manager: Final = CyberArkSecretManager()
        pending: Final = manager.async_read_secret(secret_name="KEY", timeout=0)
        assert inspect.iscoroutine(pending)
        assert server.requests == []
        assert await asyncio.create_task(pending) == '"secret-value"'
        assert manager.sync_read_secret("KEY", timeout=0) == "secret-value"
        assert tuple(request.path for request in server.requests) == (
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
    with recording_service() as server:
        server.default_response = ResponseSpec(body=_vault_body("value"))
        server.expected_requests = 1
        manager: Final = _vault(monkeypatch, server.base_url)
        assert manager.sync_read_secret("KEY", {"path_prefix": override}) == "value"
        assert urlsplit(server.requests[0].path).path == (
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
    with recording_service() as server:
        for body in (b"token", b"old", {}, {}, b"provider-after-delete"):
            server.enqueue(ResponseSpec(body=body))
        server.expected_requests = 5
        manager: Final = _cyberark(monkeypatch, server.base_url)
        assert manager.sync_read_secret("KEY") == "old"
        pending: Final = manager.async_write_secret(
            "KEY",
            "new-value",
            "ignored",
            {"ignored": object()},
            0,
            {"ignored": object()},
        )
        assert inspect.iscoroutine(pending)
        assert len(server.requests) == 2
        assert await asyncio.create_task(pending) == {
            "status": "success",
            "message": "Secret KEY written successfully",
        }
        assert manager.sync_read_secret("KEY") == "new-value"
        assert await manager.async_read_secret("KEY") == "new-value"
        assert len(server.requests) == 4
        assert await manager.async_delete_secret(secret_name="KEY", recovery_window_in_days=None, timeout=0) == {
            "status": "not_supported",
            "message": "CyberArk Conjur does not support direct secret deletion. Use policy updates to remove variables.",
        }
        assert len(server.requests) == 4
        assert manager.sync_read_secret("KEY") == "provider-after-delete"
        assert tuple(request.path for request in server.requests) == (
            "/authn/account/reader/authenticate",
            "/secrets/account/variable/KEY",
            "/policies/account/policy/root",
            "/secrets/account/variable/KEY",
            "/secrets/account/variable/KEY",
        )
        assert server.requests[3].raw_body == b"new-value"


@pytest.mark.parametrize("status", (401, 403, 500))
@pytest.mark.parametrize("authentication", (False, True))
async def test_public_cyberark_write_errors_match_python_without_http_retries(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    authentication: bool,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        responses: Final = (
            (ResponseSpec(status=status, body={}),) * 2
            if authentication
            else (
                ResponseSpec(body=b"token"),
                ResponseSpec(body={}),
                ResponseSpec(status=status, body={}),
            )
        )
        for response in responses * 2:
            server.enqueue(response)
        server.expected_requests = len(responses) * 2
        reference: Final = await _cyberark(monkeypatch, server.base_url).async_write_secret("KEY", "value")
        native_handle: Final = _native_handle(_cyberark(monkeypatch, server.base_url))
        actual: Final = await native_handle.async_write_secret("KEY", "value")
        assert actual == reference
        assert tuple(actual) == tuple(reference)
        assert actual["status"] == "error"
        assert str(status) in actual["message"]


async def test_public_cyberark_write_recovers_from_initial_policy_authentication_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.enqueue(ResponseSpec(status=401, body={}))
        server.enqueue(ResponseSpec(body=b"token"))
        server.enqueue(ResponseSpec(body={}))
        server.expected_requests = 3
        manager: Final = _cyberark(monkeypatch, server.base_url)
        assert await manager.async_write_secret("KEY", "value") == {
            "status": "success",
            "message": "Secret KEY written successfully",
        }
        assert tuple(request.path for request in server.requests) == (
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
    with recording_service() as server:
        server.expected_requests = 0
        manager: Final = _cyberark(monkeypatch, server.base_url)
        assert await manager.async_write_secret(name, "value") == {
            "status": "error",
            "message": f"Invalid secret_name {name!r}",
        }


@pytest.mark.parametrize("same_name", (False, True))
async def test_public_cyberark_rotation_returns_the_write_response_and_retains_old_alias(
    monkeypatch: pytest.MonkeyPatch,
    same_name: bool,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        for body in (b"token", b"old-value", {}, {}):
            server.enqueue(ResponseSpec(body=body))
        server.expected_requests = 4
        manager: Final = _cyberark(monkeypatch, server.base_url)
        new_name: Final = "OLD" if same_name else "NEW"
        pending: Final = manager.async_rotate_secret("OLD", new_name, "new-value", {"ignored": object()}, 0)
        assert inspect.iscoroutine(pending)
        assert server.requests == []
        assert await asyncio.create_task(pending) == {
            "status": "success",
            "message": f"Secret {new_name} written successfully",
        }
        assert tuple(request.method for request in server.requests) == ("POST", "GET", "POST", "POST")
        assert server.requests[3].raw_body == b"new-value"


@pytest.mark.parametrize("replacement", (None, b"wrong-value"))
async def test_public_native_cyberark_rotation_requires_a_fresh_matching_replacement(
    monkeypatch: pytest.MonkeyPatch,
    replacement: bytes | None,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        for body in (b"token", b"old-value", {}, {}):
            server.enqueue(ResponseSpec(body=body))
        server.enqueue(ResponseSpec(status=404 if replacement is None else 200, body=replacement))
        server.expected_requests = 5
        manager: Final = _cyberark(monkeypatch, server.base_url)
        handle: Final = _native_handle(manager)
        message: Final = "Failed to verify new secret NEW" if replacement is None else "New secret value mismatch"
        with pytest.raises(ValueError, match=message):
            await handle.async_rotate_secret("OLD", "NEW", "new-value")
        assert handle.sync_read_secret("OLD") == "old-value"
        assert tuple(request.path for request in server.requests) == (
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
    with recording_service() as server:
        server.enqueue(ResponseSpec(body=b"token"))
        server.enqueue(ResponseSpec(body=b"value"))
        server.enqueue(ResponseSpec(status=401, body={}))
        server.expected_requests = 3
        manager: Final = _cyberark(monkeypatch, server.base_url)
        assert manager.sync_read_secret("OLD") == "value"
        assert await manager.async_read_secret("NEW") is None


async def test_cyberark_handler_errors_match_python_after_cached_authentication_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        for response in (
            ResponseSpec(body=b"token"),
            ResponseSpec(body=b"value"),
            ResponseSpec(status=401, body={}),
        ) * 2:
            server.enqueue(response)
        server.expected_requests = 6
        reference_manager: Final = _cyberark(monkeypatch, server.base_url)
        assert get_secret_from_manager(reference_manager, "cyberark", "OLD") == "value"
        with pytest.raises(ValueError, match="No secret found in CyberArk Secret Manager for NEW") as reference:
            get_secret_from_manager(reference_manager, "cyberark", "NEW")
        native_handle: Final = _native_handle(_cyberark(monkeypatch, server.base_url))
        assert native_handle.read_secret("OLD") == "value"
        with pytest.raises(ValueError, match="No secret found in CyberArk Secret Manager for NEW") as actual:
            native_handle.read_secret("NEW")
        assert actual.value.args == reference.value.args


async def test_public_cyberark_connection_errors_match_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.expected_requests = 0
        address: Final = server.base_url
    reference: Final = await _cyberark(monkeypatch, address).async_write_secret("KEY", "value")
    native_handle: Final = _native_handle(_cyberark(monkeypatch, address))
    actual: Final = await native_handle.async_write_secret("KEY", "value")
    assert actual == reference
    assert actual["status"] == "error"


async def test_public_native_cyberark_rotation_stops_after_a_failed_write(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        for body in (b"token", b"old-value", {}):
            server.enqueue(ResponseSpec(body=body))
        server.enqueue(ResponseSpec(status=401, body={}))
        server.expected_requests = 4
        handle: Final = _native_handle(_cyberark(monkeypatch, server.base_url))
        response: Final = await handle.async_rotate_secret("OLD", "NEW", "new-value")
        assert response["status"] == "error"
        assert "401" in response["message"]
        assert handle.sync_read_secret("OLD") == "old-value"
        assert tuple(request.method for request in server.requests) == ("POST", "GET", "POST", "POST")


@pytest.mark.parametrize("description", (None, "", "purpose"))
async def test_public_vault_writes_preserve_complete_responses_and_request_fields(
    monkeypatch: pytest.MonkeyPatch,
    description: str | None,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        response: Final = {
            "request_id": "test-request",
            "data": {"version": 2, "custom_metadata": {"large": 2**100}},
            "warnings": ["test-warning"],
            "unknown_field": {"nested": [None, True, ""]},
        }
        server.enqueue(ResponseSpec(body=response))
        server.expected_requests = 1
        manager: Final = _vault(monkeypatch, server.base_url)
        options: Final = {
            "secret_manager_settings": {"namespace": "team", "mount": "kv", "path_prefix": "app", "data": "token"}
        }
        pending: Final = manager.async_write_secret("KEY", "value", description, options, 2, {"ignored": object()})
        assert inspect.iscoroutine(pending)
        assert server.requests == []
        result: Final = await asyncio.create_task(pending)
        assert result == response
        assert tuple(result) == tuple(response)
        request: Final = server.requests[0]
        assert request.method == "POST"
        assert request.headers["x-vault-token"] == "token"
        namespace: Final = request.headers.get("x-vault-namespace")
        assert request.path == ("/v1/kv/data/app/KEY" if namespace else "/v1/team/kv/data/app/KEY")
        assert namespace in (None, "team")
        assert json.loads(request.raw_body) == {
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
    with recording_service() as server:
        server.default_response = ResponseSpec(status=status, body={"errors": ["denied"]})
        server.expected_requests = 2
        options: Final = {"namespace": "team", "mount": "kv", "path_prefix": "prefix"}
        reference: Final = (
            await _vault(monkeypatch, server.base_url).async_write_secret("KEY", "value", optional_params=options)
            if operation == "write"
            else await _vault(monkeypatch, server.base_url).async_delete_secret("KEY", optional_params=options)
        )
        native_handle: Final = _native_handle(_vault(monkeypatch, server.base_url))
        actual: Final = (
            await native_handle.async_write_secret("KEY", "value", optional_params=options)
            if operation == "write"
            else await native_handle.async_delete_secret("KEY", optional_params=options)
        )
        assert actual == reference
        assert tuple(actual) == tuple(reference)
        assert actual["status"] == "error"
        assert str(status) in actual["message"]


@pytest.mark.parametrize("body", (b"{", b"", b"null", b"[1,2]", b'{"large":1267650600228229401496703205376}'))
async def test_public_vault_write_response_conversion_matches_python(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(body=body)
        server.expected_requests = 2
        reference: Final = await _vault(monkeypatch, server.base_url).async_write_secret("KEY", "value")
        native_handle: Final = _native_handle(_vault(monkeypatch, server.base_url))
        actual: Final = await native_handle.async_write_secret("KEY", "value")
        assert type(actual) is type(reference)
        assert actual == reference


async def test_public_vault_deletion_invalidates_cached_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.enqueue(ResponseSpec(body=_vault_body("old")))
        server.enqueue(ResponseSpec(status=204, body=b""))
        server.enqueue(ResponseSpec(body=_vault_body("after-delete")))
        server.expected_requests = 3
        manager: Final = _vault(monkeypatch, server.base_url)
        assert manager.sync_read_secret("KEY") == "old"
        pending: Final = manager.async_delete_secret("KEY", None, {"ignored": object()}, 2)
        assert inspect.iscoroutine(pending)
        assert len(server.requests) == 1
        assert await asyncio.create_task(pending) == {"status": "success", "message": "Secret KEY deleted successfully"}
        assert await manager.async_read_secret("KEY") == "after-delete"
        assert tuple(request.method for request in server.requests) == ("GET", "DELETE", "GET")


@pytest.mark.parametrize("same_name", (False, True))
@pytest.mark.parametrize("delete_status", (204, 403))
async def test_public_vault_rotation_preserves_response_and_best_effort_deletion(
    monkeypatch: pytest.MonkeyPatch,
    same_name: bool,
    delete_status: int,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        response: Final = {"request_id": "write-id", "data": {"version": 3}, "extra": [1, 2]}
        server.enqueue(ResponseSpec(body=b"current-existence-is-status-only"))
        server.enqueue(ResponseSpec(body=response))
        server.enqueue(ResponseSpec(body=_vault_body("replacement")))
        if not same_name:
            server.enqueue(ResponseSpec(status=delete_status, body=b""))
        server.expected_requests = 3 if same_name else 4
        manager: Final = _vault(monkeypatch, server.base_url)
        new_name: Final = "OLD" if same_name else "NEW"
        pending: Final = manager.async_rotate_secret("OLD", new_name, "replacement", timeout=2)
        assert inspect.iscoroutine(pending)
        assert server.requests == []
        assert await asyncio.create_task(pending) == response
        assert tuple(request.method for request in server.requests) == (
            ("GET", "POST", "GET") if same_name else ("GET", "POST", "GET", "DELETE")
        )
        assert json.loads(server.requests[1].raw_body) == {
            "data": {"key": "replacement", "description": "Rotated from OLD"},
        }
        assert urlsplit(server.requests[2].path).path == f"/v1/secret/data/{new_name}"


@pytest.mark.parametrize("stage", ("current", "write", "verify"))
@pytest.mark.parametrize("status", (404, 403, 500))
async def test_public_vault_rotation_failure_messages_and_request_counts_match_python(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    status: int,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        before: Final = (
            ()
            if stage == "current"
            else (
                (ResponseSpec(body=_vault_body("old")),)
                if stage == "write"
                else (
                    ResponseSpec(body=_vault_body("old")),
                    ResponseSpec(body={"data": {"version": 2}}),
                )
            )
        )
        responses: Final = (*before, ResponseSpec(status=status, body={"errors": ["denied"]}))
        for response in responses * 2:
            server.enqueue(response)
        server.expected_requests = len(responses) * 2
        reference: Final = await _vault(monkeypatch, server.base_url).async_rotate_secret("OLD", "NEW", "value")
        native_handle: Final = _native_handle(_vault(monkeypatch, server.base_url))
        actual: Final = await native_handle.async_rotate_secret("OLD", "NEW", "value")
        assert actual == reference
        assert actual["status"] == "error"
        expected_paths: Final = (
            ("/v1/secret/data/OLD",)
            + (("/v1/secret/data/NEW",) if stage != "current" else ())
            + (("/v1/secret/data/NEW",) if stage == "verify" else ())
        )
        assert tuple(urlsplit(request.path).path for request in server.requests) == expected_paths * 2


@pytest.mark.parametrize("value", (None, "different", True, 42, 2**100, [1, "two"], [2**100], {"nested": 2**100}))
async def test_public_vault_rotation_mismatches_do_not_delete_the_old_alias(
    monkeypatch: pytest.MonkeyPatch,
    value: JsonValue,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        responses: Final = (
            ResponseSpec(body=_vault_body("old")),
            ResponseSpec(body={"data": {"version": 2}}),
            ResponseSpec(body={"data": {"data": {"key": value}}}),
        )
        for response in responses * 2:
            server.enqueue(response)
        server.expected_requests = 6
        reference: Final = await _vault(monkeypatch, server.base_url).async_rotate_secret("OLD", "NEW", "value")
        native_handle: Final = _native_handle(_vault(monkeypatch, server.base_url))
        actual: Final = await native_handle.async_rotate_secret("OLD", "NEW", "value")
        assert actual == reference
        assert actual["status"] == "error"
        assert all(request.method != "DELETE" for request in server.requests)


@pytest.mark.parametrize("operation", ("write", "delete", "rotate"))
async def test_public_vault_mutation_timeouts_match_python(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.default_response = ResponseSpec(body=_vault_body("value"), delay=0.25)
        server.expected_requests = 2
        reference_manager: Final = _vault(monkeypatch, server.base_url)
        reference: Final = await {
            "write": partial(reference_manager.async_write_secret, "KEY", "value"),
            "delete": partial(reference_manager.async_delete_secret, "KEY"),
            "rotate": partial(reference_manager.async_rotate_secret, "OLD", "NEW", "value"),
        }[operation](timeout=0.05)
        native_handle: Final = _native_handle(_vault(monkeypatch, server.base_url))
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


@pytest.mark.parametrize("operation", ("write", "delete", "rotate"))
async def test_public_vault_unsafe_names_fail_before_authentication(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.expected_requests = 0
        manager: Final = _vault(monkeypatch, server.base_url)
        result: Final = await {
            "write": partial(manager.async_write_secret, "../KEY", "value"),
            "delete": partial(manager.async_delete_secret, "../KEY"),
            "rotate": partial(manager.async_rotate_secret, "../KEY", "NEW", "value"),
        }[operation]()
        assert result == {"status": "error", "message": "Invalid secret_name '../KEY'"}


@pytest.mark.parametrize("operation", ("write", "delete", "rotate"))
async def test_public_vault_authentication_errors_match_python(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setenv("HCP_VAULT_APPROLE_ROLE_ID", "role")
    monkeypatch.setenv("HCP_VAULT_APPROLE_SECRET_ID", "secret-id")
    with recording_service() as server:
        server.default_response = ResponseSpec(status=403, body={"errors": ["denied"]})
        server.expected_requests = 2
        reference_manager: Final = _vault(monkeypatch, server.base_url)
        reference: Final = await {
            "write": partial(reference_manager.async_write_secret, "KEY", "value"),
            "delete": partial(reference_manager.async_delete_secret, "KEY"),
            "rotate": partial(reference_manager.async_rotate_secret, "OLD", "NEW", "value"),
        }[operation]()
        native_handle: Final = _native_handle(_vault(monkeypatch, server.base_url))
        actual: Final = await {
            "write": partial(native_handle.async_write_secret, "KEY", "value"),
            "delete": partial(native_handle.async_delete_secret, "KEY"),
            "rotate": partial(native_handle.async_rotate_secret, "OLD", "NEW", "value"),
        }[operation]()
        assert actual == reference
        assert actual["status"] == "error"
        assert tuple(request.path for request in server.requests) == ("/v1/auth/approle/login",) * 2


async def test_public_vault_rotation_stops_on_a_success_response_containing_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        result: Final = {"status": "error", "message": "write rejected", "extra": 2**100}
        for response in (ResponseSpec(body=_vault_body("old")), ResponseSpec(body=result)) * 2:
            server.enqueue(response)
        server.expected_requests = 4
        reference: Final = await _vault(monkeypatch, server.base_url).async_rotate_secret("OLD", "NEW", "value")
        native_handle: Final = _native_handle(_vault(monkeypatch, server.base_url))
        actual: Final = await native_handle.async_rotate_secret("OLD", "NEW", "value")
        assert actual == reference == result
        assert tuple(request.method for request in server.requests) == ("GET", "POST", "GET", "POST")


async def test_public_native_vault_write_invalidates_stale_cached_values(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.enqueue(ResponseSpec(body=_vault_body("old")))
        server.enqueue(ResponseSpec(body={"data": {"version": 2}}))
        server.enqueue(ResponseSpec(body=_vault_body("new")))
        server.expected_requests = 3
        handle: Final = _native_handle(_vault(monkeypatch, server.base_url))
        assert handle.sync_read_secret("KEY") == "old"
        assert await handle.async_write_secret("KEY", "new") == {"data": {"version": 2}}
        assert handle.sync_read_secret("KEY") == "new"
        assert tuple(request.method for request in server.requests) == ("GET", "POST", "GET")


async def test_public_native_vault_write_rejects_description_overwriting_the_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        server.expected_requests = 0
        handle: Final = _native_handle(_vault(monkeypatch, server.base_url))
        assert await handle.async_write_secret("KEY", "value", "description", {"data": "description"}) == {
            "status": "error",
            "message": "HashiCorp Vault data key conflicts with description",
        }


@pytest.mark.parametrize(
    "body", (None, [], 42, 2**100, {"data": 2**100}, {"data": None}, {"data": {"data": []}}, {}, {"data": {}})
)
async def test_public_vault_rotation_preserves_malformed_verification_errors(
    monkeypatch: pytest.MonkeyPatch,
    body: JsonValue,
) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    with recording_service() as server:
        responses: Final = (
            ResponseSpec(body=_vault_body("old")),
            ResponseSpec(body={"data": {"version": 2}}),
            ResponseSpec(body=body),
        )
        for response in responses * 2:
            server.enqueue(response)
        server.expected_requests = 6
        reference: Final = await _vault(monkeypatch, server.base_url).async_rotate_secret("OLD", "NEW", "value")
        native_handle: Final = _native_handle(_vault(monkeypatch, server.base_url))
        actual: Final = await native_handle.async_rotate_secret("OLD", "NEW", "value")
        assert actual == reference
        assert actual["status"] == "error"
        assert all(request.method != "DELETE" for request in server.requests)
