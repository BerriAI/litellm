from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.rust_bridge.public_call import bind, native_call, signature


def _messages(
    max_tokens: int,
    messages: Sequence[object],
    model: str,
    temperature: float | None = None,
    api_key: str | None = None,
    **kwargs: object,  # kwargs-ok: exercise the public signature binding contract
) -> None:
    return None


@pytest.mark.parametrize("supplied", ({}, {"api_key": None}, {"api_key": "explicit"}))
def test_native_call_preserves_omission_separately_from_bound_defaults(supplied: Mapping[str, object]) -> None:
    messages: Final[Sequence[object]] = [{"role": "user", "content": "hello"}]
    args: Final = (128, messages, "model", 0.25)
    fields: Final = bind(signature(_messages), args, supplied)
    assert fields is not None

    call: Final = native_call(args, supplied, fields)

    assert call.args is args
    assert call.kwargs is supplied
    assert call.bound == {
        "max_tokens": 128,
        "messages": messages,
        "model": "model",
        "temperature": 0.25,
        "api_key": supplied.get("api_key"),
    }
    assert call.bound["messages"] is messages
    assert ("api_key" in call.kwargs) == ("api_key" in supplied)


def test_native_call_keeps_extra_option_objects_without_nested_kwargs() -> None:
    messages: Final[Sequence[object]] = []
    metadata: Final = {"trace": "caller"}
    supplied: Final = {"metadata": metadata}
    args: Final = (128, messages, "model")
    fields: Final = bind(signature(_messages), args, supplied)
    assert fields is not None

    call: Final = native_call(args, supplied, fields)

    assert call.bound == {
        "max_tokens": 128,
        "messages": messages,
        "model": "model",
        "temperature": None,
        "api_key": None,
        "metadata": metadata,
    }
    assert call.bound["metadata"] is metadata
    assert supplied == {"metadata": metadata}
