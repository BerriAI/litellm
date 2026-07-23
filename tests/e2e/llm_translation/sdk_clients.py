"""Real provider SDK clients pointed at the proxy, connected the way customers
connect (LIT-4577).

The OpenAI SDK drives the OpenAI-compatible surface (/responses, /embeddings,
/images/generations, /moderations, /audio/*) and the Anthropic SDK drives
/v1/messages, each authenticated with a litellm virtual key. Errors surface as
the SDK's own exceptions, exactly what an end user sees. Retries are disabled
so a proxy fault fails the test instead of being papered over, and the timeout
matches the shared transport's request budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from anthropic import Anthropic
from openai import OpenAI

from e2e_config import PROXY_BASE_URL, REQUEST_TIMEOUT


def response_header(headers: Mapping[str, str], name: str) -> str | None:
    """Typed read of an SDK response header: httpx.Headers.get returns Any and
    httpx itself is a banned import in suite code, so tests read headers through
    the Mapping[str, str] interface Headers fulfils."""
    return headers[name] if name in headers else None


@dataclass(frozen=True, slots=True)
class SdkClients:
    base_url: str
    request_timeout: float

    def openai(self, key: str) -> OpenAI:
        return OpenAI(
            base_url=self.base_url,
            api_key=key,
            timeout=self.request_timeout,
            max_retries=0,
        )

    def anthropic(self, key: str) -> Anthropic:
        return Anthropic(
            base_url=self.base_url,
            api_key=key,
            timeout=self.request_timeout,
            max_retries=0,
        )


def build_sdk_clients() -> SdkClients:
    return SdkClients(base_url=PROXY_BASE_URL, request_timeout=REQUEST_TIMEOUT)
