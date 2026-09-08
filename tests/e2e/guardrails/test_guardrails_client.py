from dataclasses import dataclass
from itertools import chain, repeat
from typing import Final

import pytest

from e2e_http import StreamingResponse
from guardrails_client import poll_until_guardrail_applied


@dataclass
class Clock:
    elapsed: float = 0.0

    def now(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


def _response(applied: str, status: int = 200) -> StreamingResponse:
    return StreamingResponse(status_code=status, body="{}", headers={"x-litellm-applied-guardrails": applied})


def test_waits_for_requested_guardrail_after_an_unrelated_global_guardrail() -> None:
    clock: Final = Clock()
    expected: Final = _response("global-filter, tool-permission")
    responses: Final = iter((_response("global-filter"), expected))

    result: Final = poll_until_guardrail_applied(
        lambda: next(responses), "tool-permission", timeout=5, interval=2, now=clock.now, sleep=clock.sleep
    )

    assert result is expected
    assert clock.elapsed == 2


@pytest.mark.parametrize("applied", ("", "global-filter", "tool-permission-sibling"))
def test_missing_exact_guardrail_returns_failure_evidence_at_deadline(applied: str) -> None:
    clock: Final = Clock()
    missing: Final = _response(applied)

    result: Final = poll_until_guardrail_applied(
        lambda: missing, "tool-permission", timeout=5, interval=2, now=clock.now, sleep=clock.sleep
    )

    assert result is missing
    assert clock.elapsed == 5


@pytest.mark.parametrize("status", (400, 401, 429, 500))
def test_http_failure_is_not_hidden_by_a_later_success(status: int) -> None:
    clock: Final = Clock()
    failed: Final = _response("", status)
    responses: Final = iter(chain((failed,), repeat(_response("tool-permission"))))

    result: Final = poll_until_guardrail_applied(
        lambda: next(responses), "tool-permission", timeout=5, interval=2, now=clock.now, sleep=clock.sleep
    )

    assert result is failed
    assert clock.elapsed == 0
