from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeAlias, cast

import httpx
import pytest

import litellm
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge import settings
from litellm.rust_bridge.ocr.entrypoints import NATIVE_AOCR, NATIVE_OCR, LiteLLMOcrRequest
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec, recording_service
from tests.test_litellm_rust.support.requests import OCR_DOCUMENT, OCR_MODEL, OCR_RESPONSE

native: Final = pytest.importorskip("litellm.rust_bridge._native")

AccessMode: TypeAlias = Literal["read_only", "write_only", "read_and_write"]


class Ocr(Protocol):
    def __call__(self, api_base: str, /) -> Awaitable[OCRResponse]: ...


class _VaultSecrets(CustomSecretManager):
    def __init__(self, failure: BaseException | None = None) -> None:
        super().__init__(secret_manager_name="rust_bridge_ocr_test")
        self.failure: Final = failure
        self.reads: tuple[tuple[str, Mapping[str, object] | None], ...] = ()

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        raise AssertionError("get_secret reads custom managers synchronously")

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        self.reads = (*self.reads, (secret_name, optional_params))
        if secret_name != "MISTRAL_API_KEY":
            return None
        if self.failure is not None:
            raise self.failure
        return "vault-key"

    def key_reads(self) -> tuple[Mapping[str, object] | None, ...]:
        return tuple(params for name, params in self.reads if name == "MISTRAL_API_KEY")


def _native_request(api_base: str) -> LiteLLMOcrRequest:
    return LiteLLMOcrRequest(
        model=OCR_MODEL,
        document=OCR_DOCUMENT,
        api_key=None,
        api_base=api_base,
        timeout=None,
        custom_llm_provider=None,
        extra_headers=None,
        kwargs=MappingProxyType({}),
    )


def _public_kwargs(api_base: str) -> dict[str, object]:
    return {"model": OCR_MODEL, "document": OCR_DOCUMENT, "api_base": api_base}


async def _rust_ocr(api_base: str) -> OCRResponse:
    route: Final = NATIVE_OCR.load()
    assert route is not None
    return route(_native_request(api_base), (), _public_kwargs(api_base))


async def _rust_aocr(api_base: str) -> OCRResponse:
    route: Final = NATIVE_AOCR.load()
    assert route is not None
    return await route(_native_request(api_base), (), _public_kwargs(api_base))


_RUST_PATHS: Final = (_rust_ocr, _rust_aocr)
_RUST_IDS: Final = ("rust-sync", "rust-async")


@pytest.fixture(params=_RUST_PATHS, ids=_RUST_IDS)
def ocr(request: pytest.FixtureRequest) -> Ocr:
    return cast(Ocr, request.param)


@pytest.fixture(params=_RUST_PATHS, ids=_RUST_IDS)
def rust_ocr(request: pytest.FixtureRequest) -> Ocr:
    return cast(Ocr, request.param)


@contextmanager
def _mistral_service(expected_requests: int = 1) -> Generator[RecordingServer]:
    with recording_service() as server:
        server.default_response = ResponseSpec(body=OCR_RESPONSE)
        server.expected_requests = expected_requests
        yield server


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manager: _VaultSecrets,
    key_management: KeyManagementSettings,
    native_secret_manager: bool = True,
    environment_key: str | None = "environment-key",
) -> None:
    if environment_key is None:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    else:
        monkeypatch.setenv("MISTRAL_API_KEY", environment_key)
    monkeypatch.setattr(litellm, "secret_manager_client", manager)
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.CUSTOM)
    monkeypatch.setattr(litellm, "_key_management_settings", key_management)
    configured: Final = settings.secret_manager
    monkeypatch.setattr(settings, "secret_manager", lambda: replace(configured(), native=native_secret_manager))


@pytest.mark.parametrize(
    ("access_mode", "hosted_keys"),
    (("read_only", None), ("read_and_write", None), ("read_only", ["MISTRAL_API_KEY"])),
)
async def test_custom_secret_manager_supplies_the_ocr_key(
    monkeypatch: pytest.MonkeyPatch, ocr: Ocr, access_mode: AccessMode, hosted_keys: list[str] | None
) -> None:
    manager: Final = _VaultSecrets()
    key_management: Final = KeyManagementSettings(access_mode=access_mode, hosted_keys=hosted_keys)
    _configure(monkeypatch, manager=manager, key_management=key_management)

    with _mistral_service() as server:
        await ocr(server.base_url)

    assert server.requests[0].headers["authorization"] == "Bearer vault-key"
    assert manager.key_reads(), "the custom manager was never asked for MISTRAL_API_KEY"
    assert all(params == key_management.model_dump() for params in manager.key_reads()), manager.key_reads()


@pytest.mark.parametrize(("access_mode", "hosted_keys"), (("read_only", ["OTHER"]), ("write_only", None)))
async def test_custom_secret_manager_is_not_read_when_settings_exclude_the_key(
    monkeypatch: pytest.MonkeyPatch, ocr: Ocr, access_mode: AccessMode, hosted_keys: list[str] | None
) -> None:
    manager: Final = _VaultSecrets()
    _configure(
        monkeypatch,
        manager=manager,
        key_management=KeyManagementSettings(access_mode=access_mode, hosted_keys=hosted_keys),
    )

    with _mistral_service() as server:
        await ocr(server.base_url)

    assert server.requests[0].headers["authorization"] == "Bearer environment-key"
    assert manager.key_reads() == ()


async def test_custom_secret_manager_exceptions_fall_back_to_the_environment_key(
    monkeypatch: pytest.MonkeyPatch, ocr: Ocr
) -> None:
    _configure(
        monkeypatch,
        manager=_VaultSecrets(ValueError("secret manager failed")),
        key_management=KeyManagementSettings(access_mode="read_only"),
    )

    with _mistral_service() as server:
        await ocr(server.base_url)

    assert server.requests[0].headers["authorization"] == "Bearer environment-key"


async def test_custom_secret_manager_exceptions_without_environment_key_raise_missing_key(
    monkeypatch: pytest.MonkeyPatch, ocr: Ocr
) -> None:
    _configure(
        monkeypatch,
        manager=_VaultSecrets(ValueError("secret manager failed")),
        key_management=KeyManagementSettings(access_mode="read_only"),
        environment_key=None,
    )

    with _mistral_service(expected_requests=0) as server:
        with pytest.raises(litellm.APIConnectionError, match="Missing Mistral API Key"):
            await ocr(server.base_url)


async def test_custom_secret_manager_cancellation_propagates_without_provider_io(
    monkeypatch: pytest.MonkeyPatch, ocr: Ocr
) -> None:
    failure: Final = asyncio.CancelledError("secret manager cancelled")
    _configure(
        monkeypatch, manager=_VaultSecrets(failure), key_management=KeyManagementSettings(access_mode="read_only")
    )

    with _mistral_service(expected_requests=0) as server:
        with pytest.raises(asyncio.CancelledError) as raised:
            await ocr(server.base_url)

    assert raised.value is failure


async def test_rust_reads_a_python_only_secret_manager_through_python(
    monkeypatch: pytest.MonkeyPatch, rust_ocr: Ocr
) -> None:
    manager: Final = _VaultSecrets()
    key_management: Final = KeyManagementSettings(access_mode="read_only")
    _configure(monkeypatch, manager=manager, key_management=key_management, native_secret_manager=False)

    with _mistral_service() as server:
        await rust_ocr(server.base_url)

    assert server.requests[0].headers["authorization"] == "Bearer vault-key"
    assert manager.key_reads(), "the Python manager was never asked for MISTRAL_API_KEY"
    assert all(params == key_management.model_dump() for params in manager.key_reads()), manager.key_reads()


class _PythonManagerReads:
    def __init__(self) -> None:
        self.reads: tuple[tuple[object, str, str], ...] = ()

    def __call__(
        self,
        client: object,
        key_manager: str,
        secret_name: str,
        key_management_settings: KeyManagementSettings | None = None,
    ) -> str | None:
        self.reads = (*self.reads, (client, key_manager, secret_name))
        return "python-key" if secret_name == "MISTRAL_API_KEY" else None


async def test_python_only_builtin_manager_is_read_through_python_not_its_native_backend(
    monkeypatch: pytest.MonkeyPatch, rust_ocr: Ocr
) -> None:
    from litellm.secret_managers import main as secret_manager_main
    from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2

    python_reads: Final = _PythonManagerReads()
    monkeypatch.setattr(secret_manager_main, "get_secret_from_manager", python_reads)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "native-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "native-secret")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
    monkeypatch.setattr(litellm, "secret_manager_client", manager)
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.AWS_SECRET_MANAGER)
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(hosted_keys=["MISTRAL_API_KEY"]))
    monkeypatch.setattr(settings, "secret_manager", lambda: settings.SecretManager(readable=True, native=False))

    with _mistral_service() as provider:
        await rust_ocr(provider.base_url)

    assert provider.requests[0].headers["authorization"] == "Bearer python-key"
    assert (manager, "aws_secret_manager", "MISTRAL_API_KEY") in python_reads.reads
    assert getattr(manager, "_litellm_native_secret_manager", None) is None


async def test_no_secret_client_leaves_dormant_binding_settings_unread(
    monkeypatch: pytest.MonkeyPatch, rust_ocr: Ocr
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "environment-key")
    monkeypatch.setattr(litellm, "secret_manager_client", None)
    monkeypatch.setattr(litellm, "_key_management_settings", object())

    with _mistral_service() as server:
        await rust_ocr(server.base_url)

    assert server.requests[0].headers["authorization"] == "Bearer environment-key"


class _FixedSecrets(CustomSecretManager):
    def __init__(self, value: str) -> None:
        super().__init__(secret_manager_name="rust_bridge_ocr_fixed")
        self.value: Final = value

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        raise AssertionError("get_secret reads custom managers synchronously")

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        return self.value if secret_name == "MISTRAL_API_KEY" else None


class _PlainSecretReader:
    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        return "vault-key"


class _AzureSecret:
    def __init__(self, value: str | None) -> None:
        self.value: Final = value


def _azure_sdk_client(value: str | None) -> object:
    class SecretClient:
        def get_secret(self, name: str) -> _AzureSecret:
            return _AzureSecret(value if name == "MISTRAL_API_KEY" else None)

    SecretClient.__module__ = "azure.keyvault.secrets._client"
    return SecretClient()


def _configure_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    client: object,
    system: KeyManagementSystem,
    key_management: KeyManagementSettings,
    environment_key: str = "environment-key",
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", environment_key)
    monkeypatch.setattr(litellm, "secret_manager_client", client)
    monkeypatch.setattr(litellm, "_key_management_system", system)
    monkeypatch.setattr(litellm, "_key_management_settings", key_management)
    configured: Final = settings.secret_manager
    monkeypatch.setattr(settings, "secret_manager", lambda: replace(configured(), native=True))


async def _assert_missing_key(ocr: Ocr) -> None:
    with _mistral_service(expected_requests=0) as server:
        with pytest.raises(litellm.APIConnectionError, match="Missing Mistral API Key"):
            await ocr(server.base_url)


@pytest.mark.parametrize("environment_key", ("true", " FALSE ", "True"))
async def test_boolean_environment_keys_count_as_missing(
    monkeypatch: pytest.MonkeyPatch, ocr: Ocr, environment_key: str
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", environment_key)
    monkeypatch.setattr(litellm, "secret_manager_client", None)

    await _assert_missing_key(ocr)


@pytest.mark.parametrize("manager_key", ("True", "(False)"))
async def test_boolean_manager_keys_count_as_missing(
    monkeypatch: pytest.MonkeyPatch, ocr: Ocr, manager_key: str
) -> None:
    _configure_client(
        monkeypatch,
        client=_FixedSecrets(manager_key),
        system=KeyManagementSystem.CUSTOM,
        key_management=KeyManagementSettings(access_mode="read_only"),
    )

    await _assert_missing_key(ocr)


async def test_boolean_environment_fallback_after_a_manager_exception_counts_as_missing(
    monkeypatch: pytest.MonkeyPatch, ocr: Ocr
) -> None:
    _configure(
        monkeypatch,
        manager=_VaultSecrets(ValueError("secret manager failed")),
        key_management=KeyManagementSettings(access_mode="read_only"),
        environment_key="True",
    )

    await _assert_missing_key(ocr)


async def test_manager_without_the_key_does_not_fall_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch, ocr: Ocr
) -> None:
    _configure_client(
        monkeypatch,
        client=_azure_sdk_client(None),
        system=KeyManagementSystem.AZURE_KEY_VAULT,
        key_management=KeyManagementSettings(access_mode="read_only"),
    )

    await _assert_missing_key(ocr)


async def test_rust_hosted_keys_exclude_azure_sdk_clients_too(monkeypatch: pytest.MonkeyPatch, rust_ocr: Ocr) -> None:
    _configure_client(
        monkeypatch,
        client=_azure_sdk_client("vault-key"),
        system=KeyManagementSystem.AZURE_KEY_VAULT,
        key_management=KeyManagementSettings(access_mode="read_only", hosted_keys=["OTHER"]),
    )

    with _mistral_service() as server:
        await rust_ocr(server.base_url)

    assert server.requests[0].headers["authorization"] == "Bearer environment-key", (
        "recorded divergence: Python's get_secret_from_manager recognizes Azure SDK clients by type and ignores hosted_keys"
    )


async def test_custom_system_with_a_foreign_client_falls_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch, ocr: Ocr
) -> None:
    _configure_client(
        monkeypatch,
        client=_PlainSecretReader(),
        system=KeyManagementSystem.CUSTOM,
        key_management=KeyManagementSettings(access_mode="read_only"),
    )

    with _mistral_service() as server:
        await ocr(server.base_url)

    assert server.requests[0].headers["authorization"] == "Bearer environment-key"


async def test_native_backend_supplies_ocr_credentials_without_a_python_reader(
    monkeypatch: pytest.MonkeyPatch, rust_ocr: Ocr
) -> None:
    from litellm.secret_managers import main as secret_manager_main
    from litellm.secret_managers import secret_manager_handler
    from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2

    def reject_python_read(
        client: object,
        key_manager: str,
        secret_name: str,
        key_management_settings: KeyManagementSettings | None = None,
    ) -> str | None:
        raise AssertionError("Rust must read the native backend directly")

    monkeypatch.setattr(secret_manager_handler, "get_secret_from_manager", reject_python_read)
    monkeypatch.setattr(secret_manager_main, "get_secret_from_manager", reject_python_read)
    with recording_service() as secrets, _mistral_service(expected_requests=2) as provider:
        secrets.default_response = ResponseSpec(body={"SecretString": "native-key"})
        secrets.expected_requests = None
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "native-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "native-secret")
        monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", secrets.base_url)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        manager: Final = AWSSecretsManagerV2(aws_region_name="us-east-1")
        monkeypatch.setattr(litellm, "secret_manager_client", manager)
        monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.AWS_SECRET_MANAGER)
        monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(hosted_keys=["MISTRAL_API_KEY"]))
        monkeypatch.setattr(settings, "secret_manager", lambda: settings.SecretManager(readable=True, native=True))
        await rust_ocr(provider.base_url)
        await rust_ocr(provider.base_url)

        assert len(secrets.requests) == 2, [(request.path, request.body) for request in secrets.requests]
        assert all(request.headers["authorization"] == "Bearer native-key" for request in provider.requests)
        assert all("Credential=native-access/" in request.headers["authorization"] for request in secrets.requests)
        assert native._SecretManagerRuntime.from_client(manager) is getattr(manager, "_litellm_native_secret_manager")
