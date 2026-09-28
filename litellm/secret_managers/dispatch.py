from typing import Final

from pydantic import JsonValue

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Rules
from litellm.rust_bridge.secret_manager import (
    NATIVE_SECRET_MANAGER,
    NativeSecretManagerFactory,
    resolve_native_secret_manager,
)
from litellm.secret_managers.secret_manager_handler import get_secret_from_manager as python_get_secret_from_manager
from litellm.types.secret_managers.main import KeyManagementSettings


def get_secret_from_manager(
    client: object,
    key_manager: str,
    secret_name: str,
    key_management_settings: KeyManagementSettings | None = None,
    *,
    rules: Rules | None = None,
    binding: NativeBinding[NativeSecretManagerFactory] = NATIVE_SECRET_MANAGER,
) -> JsonValue:
    native: Final = resolve_native_secret_manager(client, key_manager, rules, binding=binding)
    if native is None:
        return python_get_secret_from_manager(client, key_manager, secret_name, key_management_settings)
    return native.read_secret(
        secret_name, key_management_settings.model_dump(mode="json") if key_management_settings is not None else None
    )
