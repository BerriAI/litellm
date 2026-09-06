from __future__ import annotations

from typing import Final

import pytest

from litellm.rust_bridge import runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("state", ("disabled", "ineligible", "unavailable", "handled"))
async def test_attempt_only_prepares_selected_requests(asynchronous: bool, state: str) -> None:
    events: Final[list[str]] = []

    def load() -> object | None:
        events.append("load")
        return None if state == "unavailable" else object()

    def prepare() -> int:
        events.append("prepare")
        return 3

    def call(_binding: object, request: int) -> int:
        events.append("call")
        return request * 2

    async def acall(binding: object, request: int) -> int:
        return call(binding, request)

    def adapt(value: int) -> str:
        events.append("adapt")
        return str(value)

    result: Final = (
        await runtime.aattempt(
            load=load,
            enabled=state != "disabled",
            eligible=state != "ineligible",
            prepare=prepare,
            call=acall,
            adapt=adapt,
        )
        if asynchronous
        else runtime.attempt(
            load=load,
            enabled=state != "disabled",
            eligible=state != "ineligible",
            prepare=prepare,
            call=call,
            adapt=adapt,
        )
    )
    if state == "handled":
        assert result == runtime.Handled("6")
        assert events == ["load", "prepare", "call", "adapt"]
    else:
        assert result == runtime.NativeSkipped(runtime.NativeSkipReason(state))
        assert events == (["load"] if state == "unavailable" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("phase", ("prepare", "call"))
async def test_attempt_reports_failure_without_deciding_retry(asynchronous: bool, phase: str) -> None:
    error: Final = RuntimeError("native failure")

    def prepare() -> int:
        if phase == "prepare":
            raise error
        return 3

    def call(_binding: object, request: int) -> int:
        raise error

    async def acall(binding: object, request: int) -> int:
        return call(binding, request)

    def adapt(value: int) -> str:
        pytest.fail("failed attempts cannot be adapted")

    result: Final = (
        await runtime.aattempt(load=object, enabled=True, eligible=True, prepare=prepare, call=acall, adapt=adapt)
        if asynchronous
        else runtime.attempt(load=object, enabled=True, eligible=True, prepare=prepare, call=call, adapt=adapt)
    )
    assert isinstance(result, runtime.NativeFailed)
    assert result.error is error


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_adaptation_failure_remains_distinct_from_native_failure(asynchronous: bool) -> None:
    error: Final = ValueError("invalid response")

    async def acall(_binding: object, request: int) -> int:
        return request

    def adapt(value: int) -> str:
        raise error

    async def run() -> None:
        if asynchronous:
            await runtime.aattempt(load=object, enabled=True, eligible=True, prepare=lambda: 3, call=acall, adapt=adapt)
        else:
            runtime.attempt(
                load=object,
                enabled=True,
                eligible=True,
                prepare=lambda: 3,
                call=lambda binding, request: request,
                adapt=adapt,
            )

    with pytest.raises(ValueError, match="invalid response") as caught:
        await run()
    assert caught.value is error
