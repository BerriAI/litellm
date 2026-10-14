from typing import Final
from dataclasses import dataclass
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import httpx
from hypothesis import Phase, settings

from tests.integration._support.client import Gateway

LIFECYCLE_SETTINGS: Final = settings(
    max_examples=20,
    stateful_step_count=8,
    deadline=None,
    database=None,
    phases=(Phase.generate, Phase.shrink),
    print_blob=True,
)


@dataclass(slots=True)
class RequestBudget:
    limit: int
    requests: int = 0
    cleaning: bool = False

    def observe(self, _request: httpx.Request) -> None:
        if self.cleaning:
            return
        self.requests += 1
        assert self.requests <= self.limit, f"Generated HTTP operation budget exceeded: {self.limit}"

    @contextmanager
    def cleanup(self) -> Iterator[None]:
        self.cleaning = True
        try:
            yield
        finally:
            self.cleaning = False


@contextmanager
def bounded_http_requests(gateways: Sequence[Gateway], limit: int) -> Iterator[RequestBudget]:
    budget: Final = RequestBudget(limit)
    for gateway in gateways:
        gateway.client.event_hooks["request"].append(budget.observe)
    try:
        yield budget
    finally:
        for gateway in gateways:
            gateway.client.event_hooks["request"].remove(budget.observe)
        print(f"Generated HTTP operations: {budget.requests}/{budget.limit}; cleanup excluded")
