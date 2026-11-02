"""Helpers for the fallbacks characterization tests (``test_fallbacks_characterization*.py``)."""

from collections.abc import Callable, Mapping
from typing import Final

from litellm.router_core import RouterCore
from tests.unit.test_router.core.scripted_host import Outcome, ScriptedHost, deployment

MakeCore = Callable[..., RouterCore]

MESSAGES: Final = [{"role": "user", "content": "hi"}]
GROUPS_ABC: Final = [deployment("a", "a"), deployment("b", "b"), deployment("c", "c")]
GROUP_A_OF_THREE: Final = [deployment("a", "a1"), deployment("a", "a2"), deployment("a", "a3"), deployment("b", "b")]
TIERED_A: Final = [
    deployment("a", "a1", order=1),
    deployment("a", "a2", order=2),
    deployment("b", "b"),
    deployment("c", "c"),
]


def build(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: the router's constructor arguments
    return make_core(**{"model_list": GROUPS_ABC, "num_retries": 0, **arguments})


async def complete(core: RouterCore, model: str = "a", **request: object) -> object:  # kwargs-ok: request kwargs
    return await core.acompletion(model=model, messages=MESSAGES, **request)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method


def complete_sync(core: RouterCore, model: str = "a", **request: object) -> object:  # kwargs-ok: request kwargs
    return core.completion(model=model, messages=MESSAGES, **request)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method


def hidden(response: object) -> Mapping[str, object]:
    return getattr(response, "_hidden_params")


def headers(response: object) -> Mapping[str, object]:
    found: Final = hidden(response)["additional_headers"]
    assert isinstance(found, Mapping)
    return found


def metadata_of(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    found: Final = kwargs.get("metadata") or kwargs.get("litellm_metadata") or {}
    assert isinstance(found, Mapping)
    return found


def show_router_debug(host: ScriptedHost) -> None:
    host.set_dynamic(expose_router_debug_in_errors=True)


def fails_once_then_answers(error: BaseException) -> Callable[..., Outcome]:
    """One scripted outcome shared by several models: the first call to any of them raises
    ``error``, every later call answers the default response."""
    calls: Final = [0]

    def outcome(**_request: object) -> Outcome:  # kwargs-ok: the scripted request
        calls[0] += 1
        return error if calls[0] == 1 else None

    return outcome


def anthropic_refusal(*, with_stop_details: bool = True) -> dict[str, object]:  # mutable-ok: a provider response
    response: Final[dict[str, object]] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "content": [],
        "stop_reason": "refusal",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    if with_stop_details:
        response["stop_details"] = {"type": "refusal", "category": None, "explanation": "no"}
    return response


def anthropic_message(text: str = "fine") -> dict[str, object]:  # mutable-ok: a provider response
    return {
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
