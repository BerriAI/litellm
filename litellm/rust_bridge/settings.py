from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HttpSettings:
    ssl_verify: bool | str
    ssl_certificate: str | None
    ssl_security_level: str | None
    ssl_ecdh_curve: str | None
    force_ipv4: bool
    http2: bool
    aiohttp_trust_env: bool
    disable_aiohttp_transport: bool
    user_agent: str


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
        disable_aiohttp_transport=litellm.disable_aiohttp_transport,
        user_agent=default_user_agent(),
    )
