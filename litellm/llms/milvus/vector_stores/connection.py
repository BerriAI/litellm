from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Final

import litellm
from litellm.constants import MILVUS_ADMIN_CONFIGURED_CONNECTION

MILVUS_MANAGED_CONFIGURATION_FIELDS: Final = frozenset(
    {
        "api_base",
        "api_key",
        "custom_llm_provider",
        "litellm_credential_name",
        "milvus_transport",
        "milvus_db_name",
        "milvus_partition_names",
        "litellm_embedding_config",
        "litellm_embedding_model",
        "milvus_text_field",
    }
)


class MilvusConnectionRejection(Enum):
    ADMIN_REQUIRED = "Only proxy admins can configure vector store connections. Contact your LiteLLM administrator."
    ADMIN_SAVE_REQUIRED = "This managed Milvus gRPC connection must be re-saved by a proxy admin before it can be used."


def _normalize_provider(custom_llm_provider: object) -> str | None:
    if not isinstance(custom_llm_provider, str) or not custom_llm_provider:
        return None
    if "/" not in custom_llm_provider:
        return custom_llm_provider
    try:
        _, provider, _, _ = litellm.get_llm_provider(model=custom_llm_provider)
        return provider
    except Exception:  # noqa: BLE001  # provider parsing failures fall back to the explicit prefix
        return custom_llm_provider.split("/", 1)[0]


def _targets_milvus(custom_llm_provider: object, litellm_params: object) -> bool:
    return _normalize_provider(custom_llm_provider) == "milvus" or (
        isinstance(litellm_params, Mapping)
        and _normalize_provider(litellm_params.get("custom_llm_provider")) == "milvus"
    )


def _is_grpc_connection(custom_llm_provider: object, litellm_params: object) -> bool:
    return (
        isinstance(litellm_params, Mapping)
        and _targets_milvus(custom_llm_provider, litellm_params)
        and litellm_params.get("milvus_transport") == "grpc"
    )


def _connection_fields(litellm_params: object) -> Mapping[str, object]:
    if not isinstance(litellm_params, Mapping):
        return MappingProxyType({})
    return MappingProxyType(
        {key: value for key, value in litellm_params.items() if key != MILVUS_ADMIN_CONFIGURED_CONNECTION}
    )


def connection_rejection(
    custom_llm_provider: object,
    litellm_params: object,
    *,
    is_proxy_admin: bool,
    managed: bool,
) -> MilvusConnectionRejection | None:
    if not _is_grpc_connection(custom_llm_provider, litellm_params):
        return None
    if managed:
        if isinstance(litellm_params, Mapping) and litellm_params.get(MILVUS_ADMIN_CONFIGURED_CONNECTION) is True:
            return None
        return MilvusConnectionRejection.ADMIN_SAVE_REQUIRED
    if is_proxy_admin:
        return None
    return MilvusConnectionRejection.ADMIN_REQUIRED


def prepare_connection_for_persistence(
    *,
    custom_llm_provider: object,
    litellm_params: object,
    is_proxy_admin: bool,
    existing_custom_llm_provider: object | None = None,
    existing_litellm_params: object | None = None,
    litellm_credential_name: object | None = None,
    existing_litellm_credential_name: object | None = None,
    litellm_credential_name_supplied: bool = False,
) -> Mapping[str, object] | MilvusConnectionRejection:
    existing: Final = _connection_fields(existing_litellm_params)
    supplied: Final = _connection_fields(litellm_params)
    merged: Final = MappingProxyType({**existing, **supplied})
    previous_is_grpc: Final = _is_grpc_connection(existing_custom_llm_provider, existing)
    effective_is_grpc: Final = _is_grpc_connection(custom_llm_provider, merged)
    effective: Final = (
        merged
        if previous_is_grpc or effective_is_grpc
        else supplied
        if isinstance(litellm_params, Mapping)
        else existing
    )
    is_create: Final = existing_custom_llm_provider is None
    provider_changed: Final = not is_create and custom_llm_provider != existing_custom_llm_provider
    credential_changed: Final = litellm_credential_name_supplied and (
        litellm_credential_name != existing_litellm_credential_name
    )
    connection_changed: Final = not is_create and (provider_changed or credential_changed or effective != existing)
    missing_marker: Final = effective_is_grpc and (
        not isinstance(existing_litellm_params, Mapping)
        or existing_litellm_params.get(MILVUS_ADMIN_CONFIGURED_CONNECTION) is not True
    )
    if (connection_changed or missing_marker) and not is_proxy_admin:
        return MilvusConnectionRejection.ADMIN_REQUIRED
    return MappingProxyType({**effective, MILVUS_ADMIN_CONFIGURED_CONNECTION: True}) if effective_is_grpc else effective


def managed_connection_fields(custom_llm_provider: object, litellm_params: object) -> frozenset[str]:
    return frozenset((MILVUS_ADMIN_CONFIGURED_CONNECTION, "custom_llm_provider", "litellm_credential_name")) | (
        MILVUS_MANAGED_CONFIGURATION_FIELDS if _targets_milvus(custom_llm_provider, litellm_params) else frozenset()
    )


def approve_configured_connection(
    custom_llm_provider: object, litellm_params: Mapping[str, object]
) -> Mapping[str, object]:
    if _normalize_provider(custom_llm_provider) == "milvus" and litellm_params.get("milvus_transport") == "grpc":
        return MappingProxyType({**litellm_params, MILVUS_ADMIN_CONFIGURED_CONNECTION: True})
    return litellm_params
