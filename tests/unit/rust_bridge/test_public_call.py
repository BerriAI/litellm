import warnings
from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.rust_bridge.public_call import AsyncFlag, native_call, native_call_hook, signature


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


def _hook_kwargs(supplied: Mapping[str, object]) -> tuple[Mapping[str, object], tuple[warnings.WarningMessage, ...]]:
    call: Final = native_call(_SIGNATURE, (128, [], "model"), supplied)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seen: Final = native_call_hook(lambda hooked: hooked.kwargs, call, (), {})
    return seen, tuple(warning for warning in caught if issubclass(warning.category, DeprecationWarning))


def test_hook_drops_a_caller_client_and_warns_at_the_callers_line() -> None:
    extra_body: Final = {"pages": [0]}

    seen, caught = _hook_kwargs({"client": object(), "pages": [1, 2], "extra_body": extra_body})

    assert seen == {"pages": [1, 2], "extra_body": extra_body}
    assert seen["extra_body"] is extra_body
    assert len(caught) == 1
    assert "client=" in str(caught[0].message)
    assert caught[0].filename == __file__


@pytest.mark.parametrize(
    "supplied",
    ({"shared_session": object(), "api_key": "k"}, {"client": None, "api_key": "k"}, {"api_key": "k"}),
)
def test_hook_drops_internal_or_unset_clients_without_warning(supplied: Mapping[str, object]) -> None:
    seen, caught = _hook_kwargs(supplied)

    assert seen == {"api_key": "k"}
    assert caught == ()
