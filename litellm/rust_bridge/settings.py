"""The `litellm.*` module globals the native routes read.

Environment variables that override these are applied in Rust, so nothing here reads `os.environ`.
`litellm-rust/crates/python-bridge/python_settings.json` pins the fields each function returns.
"""

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
        user_agent=default_user_agent(),
    )
