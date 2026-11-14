import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final, TypeAlias

import pytest

import litellm
from litellm.rust_bridge.public_call import (
    AsyncFlag,
    NativeCall,
    native_acall_hook,
    native_acall_hook_on_loop,
    native_call,
    signature,
)

NativeAcallHook: TypeAlias = Callable[
    [Callable[[NativeCall], Awaitable[None]], NativeCall, tuple[object, ...], Mapping[str, object]], Awaitable[None]
]


def _messages(
    max_tokens: int,
    messages: Sequence[object],
    model: str,
    temperature: float | None = None,
    api_key: str | None = None,
    **kwargs: object,  # kwargs-ok: exercise the public signature binding contract
) -> None:
    return None


_SIGNATURE: Final = signature(_messages)
_FLAG: Final = AsyncFlag("aflag", sync=_SIGNATURE)


@pytest.mark.parametrize("supplied", ({}, {"api_key": None}, {"api_key": "explicit"}))
def test_base_holds_positionals_and_defaults_and_never_a_keyword(supplied: Mapping[str, object]) -> None:
    messages: Final[Sequence[object]] = [{"role": "user", "content": "hello"}]
    args: Final = (128, messages, "model", 0.25)

    call: Final = native_call(_SIGNATURE, args, supplied)

    assert call.args is args
    assert call.kwargs is supplied
    assert call.base == {
        "max_tokens": 128,
        "messages": messages,
        "model": "model",
        "temperature": 0.25,
        "api_key": None,
    }
    assert call.base["messages"] is messages
    assert call.resolved["api_key"] == supplied.get("api_key")
    assert ("api_key" in call.kwargs) == ("api_key" in supplied)


def test_resolved_lays_the_keywords_over_the_base_and_equals_the_full_binding() -> None:
    messages: Final[Sequence[object]] = []
    metadata: Final = {"trace": "caller"}
    supplied: Final = {"model": "keyword-model", "metadata": metadata}
    args: Final = (128, messages)

    call: Final = native_call(_SIGNATURE, args, supplied)

    assert "model" not in call.base
    assert "kwargs" not in call.base
    assert "metadata" not in call.base
    assert call.resolved == {
        "max_tokens": 128,
        "messages": messages,
        "model": "keyword-model",
        "temperature": None,
        "api_key": None,
        "metadata": metadata,
    }
    assert call.resolved["metadata"] is metadata
    assert supplied == {"model": "keyword-model", "metadata": metadata}


def test_flag_keywords_name_positionals_and_drop_only_the_flag() -> None:
    messages: Final[Sequence[object]] = [{"role": "user", "content": "hello"}]

    keywords: Final = _FLAG.keywords((128, messages), {"model": "m", "stream": True, "aflag": True})

    assert keywords == {"max_tokens": 128, "messages": messages, "model": "m", "stream": True}


def test_flag_keywords_reject_a_parameter_given_twice() -> None:
    with pytest.raises(TypeError):
        _FLAG.keywords((128,), {"max_tokens": 64})


_CALL: Final = native_call(_SIGNATURE, (128, [], "model"), {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "expected_timeout", "expected_num_retries"),
    (
        ({"timeout": 30.0, "stream_timeout": 0.5, "num_retries": 0}, 30.0, 0),
        ({"stream": True, "timeout": 30.0, "stream_timeout": 0.5}, 0.5, None),
        ({"stream": True, "timeout": 30.0}, 30.0, None),
        ({}, None, None),
    ),
)
@pytest.mark.parametrize("hook", (native_acall_hook, native_acall_hook_on_loop))
async def test_async_hook_failures_carry_the_client_wrapper_stamp(
    hook: NativeAcallHook,
    kwargs: Mapping[str, object],
    expected_timeout: float | None,
    expected_num_retries: int | None,
) -> None:
    async def native(call: NativeCall) -> None:
        raise litellm.Timeout(message="timed out", model="model", llm_provider="anthropic")

    with pytest.raises(litellm.Timeout) as raised:
        await hook(native, _CALL, (), kwargs)

    assert getattr(raised.value, "timeout", "unset") == expected_timeout
    assert getattr(raised.value, "num_retries", None) == expected_num_retries


@pytest.mark.asyncio
@pytest.mark.parametrize(("hook", "yields"), ((native_acall_hook, True), (native_acall_hook_on_loop, False)))
async def test_only_the_executor_hook_lets_a_task_started_before_the_call_run_first(
    hook: NativeAcallHook, yields: bool
) -> None:
    messages: Final = [{"role": "user", "content": "during about secret"}]
    sent: Final[list[object]] = []

    async def mask() -> None:
        messages[0]["content"] = "during about [REDACTED]"

    async def native(call: NativeCall) -> None:
        sent.append(messages[0]["content"])

    guardrail: Final = asyncio.create_task(mask())
    await hook(native, _CALL, (), {})
    await guardrail

    assert sent == ["during about [REDACTED]" if yields else "during about secret"]
