import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Final, Literal

import anyio

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import UnifiedLLMGuardrails
from litellm.types.utils import Delta, GenericGuardrailAPIInputs, ModelResponseStream

_WORDS: Final = ("Hello", " ", "world", "!")


class _Observer(CustomGuardrail):
    def __init__(self, *, failure: Exception | None = None) -> None:
        super().__init__(guardrail_name="observer", event_hook="post_call", default_on=True)
        self.streaming_observe_only = True
        self.observed: Final[list[list[str]]] = []  # mutable-ok: records each observed call
        self._failure: Final = failure

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        loop: Final = asyncio.get_running_loop()
        next_iteration: Final = loop.create_future()
        loop.call_soon(next_iteration.set_result, None)
        await next_iteration
        if self._failure is not None:
            raise self._failure
        self.observed.append(list(inputs.get("texts", [])))
        return inputs


def _chunk(word: str) -> ModelResponseStream:
    return ModelResponseStream(
        model="gpt-4", choices=[litellm.StreamingChoices(index=0, delta=Delta(role="assistant", content=word))]
    )


async def _upstream(
    *, words: tuple[str, ...] = _WORDS, stall_after: int | None = None
) -> AsyncIterator[ModelResponseStream]:
    for i, word in enumerate(words):
        if i == stall_after:
            await asyncio.Event().wait()
        yield _chunk(word)


def _guarded_stream(
    observer: _Observer, upstream: AsyncIterator[ModelResponseStream], *, route: str | None = "/chat/completions"
) -> AsyncIterator[object]:
    return UnifiedLLMGuardrails().async_post_call_streaming_iterator_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="test", request_route=route),
        response=upstream,
        request_data={
            "messages": [{"role": "user", "content": "hi"}],
            "guardrail_to_apply": observer,
            "metadata": {"guardrails": ["observer"]},
        },
    )


async def test_a_client_disconnect_mid_stream_is_still_observed() -> None:
    observer: Final = _Observer()
    two_chunks_sent: Final = asyncio.Event()
    sent: Final[list[object]] = []  # mutable-ok: records what reached the client

    async def client() -> None:
        async for chunk in _guarded_stream(observer, _upstream(stall_after=2)):
            sent.append(chunk)
            if len(sent) == 2:
                two_chunks_sent.set()

    async with anyio.create_task_group() as requests:
        requests.start_soon(client)
        await two_chunks_sent.wait()
        requests.cancel_scope.cancel()

    assert observer.observed == [["Hello "]]


async def test_a_failing_observer_never_breaks_the_stream(warning_messages: Callable[[str], list[str]]) -> None:
    observer: Final = _Observer(failure=RuntimeError("observer down"))

    sent: Final = [chunk async for chunk in _guarded_stream(observer, _upstream())]

    assert len(sent) == len(_WORDS)
    assert warning_messages("observe-only stream check") == [
        "UnifiedLLMGuardrails: observe-only stream check for observer failed: observer down"
    ]


async def test_an_empty_stream_is_not_observed(warning_messages: Callable[[str], list[str]]) -> None:
    observer: Final = _Observer()

    sent: Final = [chunk async for chunk in _guarded_stream(observer, _upstream(words=()), route=None)]

    assert (sent, observer.observed, warning_messages("observe-only stream check")) == ([], [], [])
