from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.rust_bridge.catalog import Rules


@dataclass(frozen=True, slots=True)
class HttpSettings:
    ssl_verify: object
    ssl_certificate: object
    ssl_security_level: object
    ssl_ecdh_curve: object
    force_ipv4: object
    http2: object
    aiohttp_trust_env: object
    disable_aiohttp_trust_env: object
    disable_aiohttp_transport: object
    user_agent: str


@dataclass(frozen=True, slots=True)
class UrlPolicy:
    user_url_validation: object
    user_url_allowed_hosts: object


@dataclass(frozen=True, slots=True)
class ProviderDefaults:
    vertex_project: object
    vertex_location: object
    enable_azure_ad_token_refresh: object


@dataclass(frozen=True, slots=True)
class SecretManager:
    readable: bool
    native: bool


@dataclass(frozen=True, slots=True)
class SecretManagerBinding:
    system: object
    access_mode: object
    hosted_keys: object
    primary_secret_name: object
    store_virtual_keys: object
    prefix_for_stored_virtual_keys: object
    kms_key_id: object
    custom_secret_manager: object
    aws_region_name: object
    aws_role_name: object
    aws_session_name: object
    aws_external_id: object
    aws_profile_name: object
    aws_web_identity_token: object
    aws_sts_endpoint: object
    replica_regions: object
    client: object
    settings_object: object


def secret_manager(rules: Rules | None = None) -> SecretManager:
    import litellm
    from litellm.rust_bridge.catalog import SecretManagerContext, decision
    from litellm.rust_bridge.configuration import Decision
    from litellm.secret_managers.main import (
        _should_read_secret_from_secret_manager,  # pyright: ignore[reportPrivateUsage]  # canonical resolver is private
    )

    readable: Final = _should_read_secret_from_secret_manager()
    system: Final = (
        litellm._key_management_system  # pyright: ignore[reportPrivateUsage]  # canonical key management globals are private
    )
    native: Final = (
        readable
        and system is not None
        and decision(SecretManagerContext(system=system.value), rules) is not Decision.PYTHON
    )
    return SecretManager(readable=readable, native=native)


def secret_manager_binding() -> SecretManagerBinding:
    import litellm
    from litellm.types.secret_managers.main import KeyManagementSettings

    configured_system: Final = (
        litellm._key_management_system  # pyright: ignore[reportPrivateUsage]  # canonical key management globals are private
    )
    configured_settings: Final = (
        litellm._key_management_settings  # pyright: ignore[reportPrivateUsage]  # canonical key management globals are private
    )
    settings: Final = configured_settings or KeyManagementSettings()
    system: Final = (
        configured_system.value if litellm.secret_manager_client is not None and configured_system is not None else None
    )
    return SecretManagerBinding(
        system=system,
        access_mode=settings.access_mode,
        hosted_keys=settings.hosted_keys,
        primary_secret_name=settings.primary_secret_name,
        store_virtual_keys=settings.store_virtual_keys,
        prefix_for_stored_virtual_keys=settings.prefix_for_stored_virtual_keys,
        kms_key_id=settings.kms_key_id,
        custom_secret_manager=settings.custom_secret_manager,
        aws_region_name=settings.aws_region_name,
        aws_role_name=settings.aws_role_name,
        aws_session_name=settings.aws_session_name,
        aws_external_id=settings.aws_external_id,
        aws_profile_name=settings.aws_profile_name,
        aws_web_identity_token=settings.aws_web_identity_token,
        aws_sts_endpoint=settings.aws_sts_endpoint,
        replica_regions=settings.replica_regions,
        client=litellm.secret_manager_client,
        settings_object=configured_settings,
    )


def provider_defaults() -> ProviderDefaults:
    import litellm

    return ProviderDefaults(
        vertex_project=litellm.vertex_project,
        vertex_location=litellm.vertex_location,
        enable_azure_ad_token_refresh=litellm.enable_azure_ad_token_refresh,
    )


def url_policy() -> UrlPolicy:
    import litellm

    return UrlPolicy(
        user_url_validation=litellm.user_url_validation,
        user_url_allowed_hosts=litellm.user_url_allowed_hosts,
    )


def http_settings() -> HttpSettings:
    import litellm
    from litellm.llms.custom_httpx.http_handler import default_user_agent

    return HttpSettings(
        ssl_verify=litellm.ssl_verify,
        ssl_certificate=litellm.ssl_certificate,
        ssl_security_level=litellm.ssl_security_level,
        ssl_ecdh_curve=litellm.ssl_ecdh_curve,
        force_ipv4=litellm.force_ipv4,
        http2=litellm.http2,
        aiohttp_trust_env=litellm.aiohttp_trust_env,
        disable_aiohttp_trust_env=litellm.disable_aiohttp_trust_env,
        disable_aiohttp_transport=litellm.disable_aiohttp_transport,
        user_agent=default_user_agent(),
    )
