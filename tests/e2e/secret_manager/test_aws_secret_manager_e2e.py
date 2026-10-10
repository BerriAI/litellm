import hashlib
import json
import uuid
from typing import Final, Protocol, cast

import pytest
from e2e_metadata import Domain, Subject, meta
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2
from litellm.secret_managers.main import get_secret

pytestmark = pytest.mark.e2e

_AWS_FIXTURE_MASTER_KEY_SHA256: Final = "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_STRING_MAP: Final = TypeAdapter(dict[str, str])
_STRING: Final = TypeAdapter(str)


class _AWSSecretManagerOperations(Protocol):
    async def async_write_secret(self, secret_name: str, secret_value: str) -> object: ...

    async def async_delete_secret(
        self, secret_name: str, recovery_window_in_days: int | None = 7
    ) -> object: ...


@pytest.mark.asyncio
@meta(Subject(domain=Domain.DEPLOY_OPS))
async def test_aws_secret_manager() -> None:
    AWSSecretsManagerV2.load_aws_secret_manager(use_aws_secret_manager=True)

    fixture_value: Final[str | None] = cast(
        str | None, TypeAdapter(str | None).validate_python(get_secret("litellm_master_key"))
    )
    if fixture_value is not None:
        fixture_data: Final = _STRING_MAP.validate_json(_STRING.validate_python(fixture_value))
        assert hashlib.sha256(fixture_data["litellm_master_key"].encode()).hexdigest() == (
            _AWS_FIXTURE_MASTER_KEY_SHA256
        )
        return

    secret_name: Final = f"litellm-e2e-secret-manager-{uuid.uuid4().hex}"
    secret_value: Final = f"sk-litellm-e2e-{uuid.uuid4().hex}"
    manager: Final = litellm.secret_manager_client
    assert isinstance(manager, AWSSecretsManagerV2)
    secret_manager: Final = cast(_AWSSecretManagerOperations, manager)
    try:
        creation: Final = _JSON_OBJECT.validate_python(
            await secret_manager.async_write_secret(
                secret_name,
                json.dumps({"litellm_master_key": secret_value}),
            )
        )
        assert creation.get("Name") == secret_name
        created_value: Final = _STRING.validate_python(get_secret(secret_name))
        created_data: Final = _STRING_MAP.validate_json(created_value)
        assert created_data["litellm_master_key"] == secret_value
    finally:
        deletion: Final = _JSON_OBJECT.validate_python(
            await secret_manager.async_delete_secret(secret_name, recovery_window_in_days=7)
        )
        assert deletion.get("Name") == secret_name
