"""Client for the logging e2e suite: drive traffic and scrape the proxy's
Prometheus ``/metrics`` endpoint.

Holds the shared Gateway so the ``resources`` fixture cleans up keys it creates.
``/metrics`` is exposed as plaintext (not a typed JSON body), so scraping goes
through ``transport.probe`` and returns the raw exposition text for a Prometheus
parser to read.
"""

from __future__ import annotations

from dataclasses import dataclass

from e2e_gateway import Gateway, build_gateway
from e2e_http import NoBody, unwrap
from models import ChatBody, ChatMessage, ChatResponse, KeyGenerateBody


@dataclass(frozen=True, slots=True)
class LoggingClient:
    gateway: Gateway

    def key_with_alias(self, alias: str, *, models: list[str]) -> str:
        return self.gateway.generate_key(
            KeyGenerateBody(key_alias=alias, models=models, user_id=f"e2e-{alias}")
        )

    def delete_key(self, key: str) -> None:
        self.gateway.delete_key(key)

    def chat(self, key: str, model: str, text: str) -> ChatResponse:
        return unwrap(
            self.gateway.chat(
                key,
                ChatBody(
                    model=model,
                messages=[ChatMessage(role="user", content=text)],
                max_tokens=64,
                ),
            )
        )

    def scrape_metrics(self) -> str:
        return self.gateway.probe("/metrics", params=NoBody()).body


def build_logging_client() -> LoggingClient:
    return LoggingClient(gateway=build_gateway())
