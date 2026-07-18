from __future__ import annotations

from dataclasses import dataclass

from e2e_http import unwrap
from load_constants import LOAD_MOCK_BODY_SNIPPET
from models import ChatBody, ChatMessage
from proxy_client import ProxyClient


@dataclass(frozen=True, slots=True)
class LoadClient:
    proxy: ProxyClient

    def preflight_mock_chat(self, *, key: str, model: str) -> None:
        """One real chat before Locust; fail with the body if mock routing is broken."""
        response = unwrap(
            self.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content="preflight load mock")],
                    max_tokens=16,
                ),
            )
        )
        content = ""
        if response.choices and response.choices[0].message is not None:
            content = response.choices[0].message.content or ""
        assert LOAD_MOCK_BODY_SNIPPET in content, (
            f"load preflight to {model!r} did not return the mock body "
            f"(mock_response not applied or wrong deployment). content={content!r} "
            f"full={response!r}"
        )


def build_client(proxy: ProxyClient) -> LoadClient:
    return LoadClient(proxy=proxy)
