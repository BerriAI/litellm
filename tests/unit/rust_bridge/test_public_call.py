from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.rust_bridge.public_call import native_call, signature


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
