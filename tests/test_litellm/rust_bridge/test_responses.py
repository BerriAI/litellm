from __future__ import annotations

from collections.abc import Generator, Mapping

import pytest

from litellm.rust_bridge import configuration, responses


@pytest.fixture(autouse=True)
def reset_responses_bridge(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    responses.set_rust_responses(sync=None, asynchronous=None)
    yield
    configuration.reset_rust_configuration()
    responses.set_rust_responses(sync=None, asynchronous=None)


class RecordingPrepare:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> Mapping[str, object]:
        self.calls += 1
        return {"model": "gpt-5"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", (pytest.param("sync", id="sync"), pytest.param("async", id="async")))
async def test_unavailable_bridge_does_not_prepare_request(mode: str) -> None:
    prepare = RecordingPrepare()
    if mode == "sync":
        result = responses.dispatch_responses(
            prepare=lambda: responses.NativeResponsesRequest(prepare()),
            fallback=lambda: "python",
            adapt=lambda value: value,
            model="gpt-5",
            provider="openai",
            request_override=True,
        )
    else:
        result = await responses.adispatch_responses(
            prepare=lambda: responses.NativeResponsesRequest(prepare()),
            fallback=_async_python_fallback,
            adapt=lambda value: value,
            model="gpt-5",
            provider="openai",
            request_override=True,
        )

    assert result == "python"
    assert prepare.calls == 0


async def _async_python_fallback() -> str:
    return "python"
