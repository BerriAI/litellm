from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict

from litellm.secret_managers.secret_manager_handler import get_secret_from_manager
from litellm.types.secret_managers.main import KeyManagementSystem


def _azure_exception_types() -> tuple[type[Exception], type[Exception]]:
    try:
        from azure.core.exceptions import (
            HttpResponseError,
            ResourceNotFoundError,
        )
    except ImportError:
        return Exception, Exception
    return HttpResponseError, ResourceNotFoundError


_AZURE_EXCEPTION_TYPES: Final[tuple[type[Exception], type[Exception]]] = _azure_exception_types()
AzureHttpResponseError: Final[type[Exception]] = _AZURE_EXCEPTION_TYPES[0]
AzureResourceNotFoundError: Final[type[Exception]] = _AZURE_EXCEPTION_TYPES[1]


class FixtureResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: int
    body: dict[str, object]


class FixtureExpected(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str | None = None
    missing: bool = False
    error: bool = False


class FixtureCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    secret_name: str
    response: FixtureResponse
    expected: FixtureExpected


class Fixture(BaseModel):
    model_config = ConfigDict(frozen=True)

    cases: tuple[FixtureCase, ...]


@dataclass(frozen=True, slots=True)
class FakeSecret:
    value: str | None


@dataclass(frozen=True, slots=True)
class FakeAzureKeyVaultClient:
    status: int
    value: str | None

    def get_secret(self, name: str) -> FakeSecret:
        if self.status == 404:
            raise AzureResourceNotFoundError()
        if self.status != 200:
            raise AzureHttpResponseError()
        return FakeSecret(value=self.value)


FIXTURE_PATH: Path = (
    Path(__file__).parents[3]
    / "litellm-rust/crates/secrets-azure/tests/fixtures/key_vault_parity.json"
)


def test_azure_key_vault_matches_rust_parity_fixture() -> None:
    fixture: Fixture = Fixture.model_validate_json(FIXTURE_PATH.read_text())
    for case in fixture.cases:
        value: object = case.response.body.get("value")
        secret: str | None = value if isinstance(value, str) else None
        client: FakeAzureKeyVaultClient = FakeAzureKeyVaultClient(
            status=case.response.status,
            value=secret,
        )
        if case.expected.missing or case.expected.error:
            with pytest.raises(
                AzureResourceNotFoundError if case.expected.missing else AzureHttpResponseError
            ):
                get_secret_from_manager(
                    secret_name=case.secret_name,
                    key_manager=KeyManagementSystem.AZURE_KEY_VAULT.value,
                    client=client,
                )
            continue

        result: str | None = get_secret_from_manager(
            secret_name=case.secret_name,
            key_manager=KeyManagementSystem.AZURE_KEY_VAULT.value,
            client=client,
        )
        assert result == case.expected.value
