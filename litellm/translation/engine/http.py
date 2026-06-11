"""The only I/O in the package, defined as an injected port.

The pure core never imports an HTTP client; the imperative shell passes one in
that satisfies ``HttpPort``. That keeps the translation functions deterministic
and unit-testable without a network, and lets the proxy choose the client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from expression.collections import Map

from ..ir import Body, PlainJson


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Map[str, str]
    body: PlainJson


class HttpPort(Protocol):
    async def post(
        self, url: str, headers: Map[str, str], body: Body
    ) -> HttpResponse: ...
