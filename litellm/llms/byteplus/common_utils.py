from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import httpx

from litellm.llms.volcengine.common_utils import VolcEngineError


class BytePlusError(VolcEngineError):
    def __init__(self, status_code: int, message: str, headers: httpx.Headers | None = None) -> None:
        self.status_code = status_code
        self.message = message
        self.headers = headers or httpx.Headers()
        super().__init__(status_code=status_code, message=message, headers=self.headers)


def get_byteplus_base_url(api_base: str | None = None) -> str:
    if api_base:
        return api_base
    return "https://ark.ap-southeast.bytepluses.com"


def get_byteplus_headers(
    api_key: str, extra_headers: Mapping[str, str] | None = None
) -> dict[str, str]:  # mutable-ok: required by HTTP handler signatures
    headers: Final[dict[str, str]] = {  # mutable-ok: mutable dictionary required by HTTP client
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    if extra_headers:
        headers.update(extra_headers)

    return headers
