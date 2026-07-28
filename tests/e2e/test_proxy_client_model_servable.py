"""Harness coverage for the bounded wait after /model/new (no live proxy).

Model propagation is polled to a deadline so a stuck control/data-plane reload fails
fast instead of stalling every test that creates a model. The clock and sleep are
injected, so these assert the deadline arithmetic without waiting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from e2e_http import NetworkError, Result, Success
from models import ModelListEntry, ModelsListResponse
from proxy_client import (
    MODEL_SERVABLE_REQUEST_TIMEOUT,
    MODEL_SERVABLE_TIMEOUT,
    NotServable,
    Servable,
    await_servable,
    servable_timeout_message,
)


def _listing(*model_names: str) -> Result[ModelsListResponse]:
    return Success(
        status_code=200,
        data=ModelsListResponse(data=tuple(ModelListEntry(id=name) for name in model_names)),
    )


@dataclass(slots=True)
class FakeClock:
    """A clock that only advances when the code under test sleeps or a slow poll runs."""

    seconds: float = 0.0
    slept: list[float] = field(default_factory=list)  # mutable-ok: records calls for assertions

    def now(self) -> float:
        return self.seconds

    def sleep(self, duration: float) -> None:
        self.slept.append(duration)
        self.seconds += duration


@dataclass(slots=True)
class FakeModelList:
    """Returns each queued /v1/models read in turn, repeating the last forever."""

    responses: tuple[Result[ModelsListResponse], ...]
    calls: int = 0
    timeouts: list[float] = field(default_factory=list)  # mutable-ok: records call timeouts

    def __call__(self, request_timeout: float) -> Result[ModelsListResponse]:
        self.timeouts.append(request_timeout)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def test_returns_servable_on_first_listing_without_sleeping() -> None:
    clock = FakeClock()
    list_models = FakeModelList(responses=(_listing("my-model"),))

    outcome = await_servable(
        list_models,
        model_name="my-model",
        timeout=40.0,
        interval=2.0,
        request_timeout=5.0,
        now=clock.now,
        sleep=clock.sleep,
    )

    assert outcome == Servable()
    assert list_models.calls == 1
    assert list_models.timeouts == [5.0]
    assert clock.slept == []


def test_polls_until_the_model_appears() -> None:
    clock = FakeClock()
    list_models = FakeModelList(responses=(_listing("other"), _listing("other"), _listing("other", "my-model")))

    outcome = await_servable(
        list_models,
        model_name="my-model",
        timeout=40.0,
        interval=2.0,
        request_timeout=5.0,
        now=clock.now,
        sleep=clock.sleep,
    )

    assert outcome == Servable()
    assert list_models.calls == 3
    assert clock.seconds == 4.0


def test_gives_up_at_the_deadline_rather_than_polling_forever() -> None:
    clock = FakeClock()
    list_models = FakeModelList(responses=(_listing("other"),))

    outcome = await_servable(
        list_models,
        model_name="my-model",
        timeout=10.0,
        interval=2.0,
        request_timeout=5.0,
        now=clock.now,
        sleep=clock.sleep,
    )

    assert isinstance(outcome, NotServable)
    assert clock.seconds == 8.0
    assert list_models.calls == 5
    assert list_models.timeouts == [5.0, 5.0, 5.0, 4.0, 2.0]


def test_does_not_wait_past_the_overall_budget() -> None:
    clock = FakeClock()

    outcome = await_servable(
        FakeModelList(responses=(_listing("other"),)),
        model_name="my-model",
        timeout=MODEL_SERVABLE_TIMEOUT,
        interval=2.0,
        request_timeout=MODEL_SERVABLE_REQUEST_TIMEOUT,
        now=clock.now,
        sleep=clock.sleep,
    )

    assert isinstance(outcome, NotServable)
    assert clock.seconds <= MODEL_SERVABLE_TIMEOUT


def test_clamps_request_timeout_to_remaining_deadline() -> None:
    """A slow final poll must not receive the full request cap when less budget remains.

    Without the clamp, remaining=3 and cap=5 lets the transport block for 5s and the
    overall wait overruns model_servable_timeout by up to ~cap seconds.
    """
    clock = FakeClock()
    timeouts: list[float] = []

    def list_models(request_timeout: float) -> Result[ModelsListResponse]:
        timeouts.append(request_timeout)
        clock.seconds += request_timeout
        return _listing("other")

    outcome = await_servable(
        list_models,
        model_name="my-model",
        timeout=10.0,
        interval=2.0,
        request_timeout=5.0,
        now=clock.now,
        sleep=clock.sleep,
    )

    assert isinstance(outcome, NotServable)
    assert timeouts[0] == 5.0
    assert any(timeout < 5.0 for timeout in timeouts)
    assert timeouts[-1] == 3.0
    assert clock.seconds <= 10.0


def test_reports_a_failed_read_distinctly_from_a_missing_model() -> None:
    clock = FakeClock()
    unreachable: Result[ModelsListResponse] = NetworkError(message="connection refused")

    outcome = await_servable(
        FakeModelList(responses=(unreachable,)),
        model_name="my-model",
        timeout=1.0,
        interval=0.5,
        request_timeout=5.0,
        now=clock.now,
        sleep=clock.sleep,
    )

    assert outcome == NotServable(last_result=unreachable)
    message = servable_timeout_message(model_name="my-model", timeout=1.0, last_result=unreachable)
    assert "connection refused" in message

    listed_without_model = _listing("other")
    propagation_message = servable_timeout_message(
        model_name="my-model", timeout=1.0, last_result=listed_without_model
    )
    assert "did not succeed" not in propagation_message
