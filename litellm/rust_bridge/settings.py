from __future__ import annotations

from dataclasses import dataclass


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


def warn(message: str) -> None:
    from litellm._logging import verbose_logger

    verbose_logger.warning("%s", message)


def secret_manager() -> SecretManager:
    from litellm.secret_managers.main import (
        _should_read_secret_from_secret_manager,  # pyright: ignore[reportPrivateUsage]  # canonical resolver is private
    )

    return SecretManager(readable=_should_read_secret_from_secret_manager())


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
