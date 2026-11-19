from __future__ import annotations

import os
from typing import Final

import httpx

from litellm.secret_managers.native_config import NativeSecretManagerConfig, captured_config
from litellm.types.secret_managers.main import KeyManagementSettings


def native_secret_manager_config(client: object) -> NativeSecretManagerConfig | None:
    return captured_config(client)


def read_secret_str(name: str) -> str | None:
    from litellm.secret_managers.main import get_secret_str

    return get_secret_str(name)


def read_from_manager(
    client: object,
    key_manager: str,
    secret_name: str,
    key_management_settings: KeyManagementSettings | None,
) -> str | None:
    from litellm.secret_managers.secret_manager_handler import get_secret_from_manager

    return get_secret_from_manager(
        client=client,
        key_manager=key_manager,
        secret_name=secret_name,
        key_management_settings=key_management_settings,
    )


def timeout_message(message: str) -> str:
    from litellm.exceptions import Timeout

    return str(Timeout(message=message, model="default-model-name", llm_provider="litellm-httpx-handler"))


def response_text(body: bytes) -> str:
    return httpx.Response(200, content=body).text


def http_status_message(method: str, url: str, status: int) -> str:
    response: Final = httpx.Response(status, request=httpx.Request(method, url))
    try:
        _ = response.raise_for_status()
    except httpx.HTTPStatusError as error:
        return str(error)
    return f"HTTP {status}"


def os_error_message(code: int) -> str:
    return str(OSError(code, os.strerror(code)))
