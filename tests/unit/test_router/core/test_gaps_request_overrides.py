"""
Per-request routing overrides a router core reads (or does not read) on each call:

- ``default_fallbacks`` and ``retry_policy`` in a request are no routing input: the router neither
  falls back on the request's ``default_fallbacks`` nor retries on its ``retry_policy``. Both reach
  the deployment call as given (litellm's own ``completion`` reads ``retry_policy`` there), on every
  attempt, sync and async, streamed or not.
- ``model_group_retry_policy`` that is not a mapping of policies, in a request or as the router's
  setting: the retry loop looks the model group up in it (``group in value``, then ``value[group]``),
  so a value that does not hold the group leaves the router's ``retry_policy`` in charge, a ``None``
  entry is no policy at all, and an explicit request ``None`` replaces the router's per-group policies.
"""

from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.router_api import ChangeSet
from litellm.router_core import RouterCore
from litellm.types.router import RetryPolicy
from tests.unit.test_router.core.retries_health_helpers import attempt, attempt_sync
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit
from tests.unit.test_router.core.streaming_helpers import chat_stream, collect_texts, collect_texts_sync

MakeCore = Callable[..., RouterCore]
MESSAGES: Final = [{"role": "user", "content": "hi"}]
GROUPS: Final = [deployment("a", "a"), deployment("b", "b"), deployment("c", "c")]
OVERRIDES: Final[Mapping[str, object]] = {"default_fallbacks": ["c"], "retry_policy": {"RateLimitErrorRetries": 3}}


def sent_overrides(host: ScriptedHost) -> list[tuple[str | None, object, object]]:  # mutable-ok: a fresh list
    return [
        (call.deployment_id, call.kwargs.get("default_fallbacks"), call.kwargs.get("retry_policy"))
        for call in host.calls_to()
    ]


# --- default_fallbacks and retry_policy: passed to the deployment, never routed on ----------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "request_body"),
    [
        ("acompletion", {"messages": MESSAGES}),
        ("atext_completion", {"prompt": "p"}),
        ("aembedding", {"input": ["x"]}),
        ("aresponses", {"input": "hi"}),
    ],
)
async def test_request_default_fallbacks_and_retry_policy_reach_the_deployment_as_given(
    make_core: MakeCore, host: ScriptedHost, endpoint: str, request_body: Mapping[str, object]
) -> None:
    core: Final = make_core(model_list=GROUPS, num_retries=0)

    await core.call(endpoint, {"model": "a", **request_body, **OVERRIDES})

    assert sent_overrides(host) == [("a", ["c"], {"RateLimitErrorRetries": 3})]


@pytest.mark.asyncio
async def test_a_failed_call_neither_falls_back_on_nor_retries_by_the_request_overrides(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=GROUPS, num_retries=0)
    host.answer("openai/a", rate_limit())

    error: Final = await attempt(core, "a", **OVERRIDES)

    assert isinstance(error, litellm.RateLimitError)
    assert sent_overrides(host) == [("a", ["c"], {"RateLimitErrorRetries": 3})]


def test_a_failed_sync_call_neither_falls_back_on_nor_retries_by_the_request_overrides(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=GROUPS, num_retries=0)
    host.answer("openai/a", rate_limit())

    error: Final = attempt_sync(core, "a", **OVERRIDES)

    assert isinstance(error, litellm.RateLimitError)
    assert sent_overrides(host) == [("a", ["c"], {"RateLimitErrorRetries": 3})]


@pytest.mark.asyncio
async def test_the_router_retries_and_fallbacks_carry_the_request_overrides_on_every_attempt(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=GROUPS, num_retries=1, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", rate_limit(), rate_limit())

    response: Final = await attempt(core, "a", **OVERRIDES)

    assert not isinstance(response, Exception)
    assert sent_overrides(host) == [("a", ["c"], {"RateLimitErrorRetries": 3})] * 2 + [
        ("b", ["c"], {"RateLimitErrorRetries": 3})
    ]


@pytest.mark.asyncio
async def test_a_stream_that_fails_before_content_falls_back_with_the_request_overrides(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=GROUPS, num_retries=0, fallbacks=[{"a": ["b"]}])
    host.answer("openai/a", chat_stream("x", error=rate_limit(), fail_after=0))
    host.answer("openai/b", chat_stream("B"))

    stream: Final = await core.call("acompletion", {"model": "a", "messages": MESSAGES, "stream": True, **OVERRIDES})

    assert await collect_texts(stream) == "B"
    assert sent_overrides(host) == [
        ("a", ["c"], {"RateLimitErrorRetries": 3}),
        ("b", ["c"], {"RateLimitErrorRetries": 3}),
    ]


def test_a_sync_stream_carries_the_request_overrides(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=GROUPS, num_retries=0)
    host.answer("openai/a", chat_stream("A", sync=True))

    completion: Final[Callable[..., object]] = getattr(core, "completion")  # pyright: ignore[reportAny]  # an endpoint method
    stream: Final = completion(model="a", messages=MESSAGES, stream=True, **OVERRIDES)

    assert collect_texts_sync(stream) == "A"
    assert sent_overrides(host) == [("a", ["c"], {"RateLimitErrorRetries": 3})]


# --- model_group_retry_policy that is not a mapping of policies --------------------------------------


def calls_after_a_rate_limit(
    core: RouterCore, host: ScriptedHost, **request: object
) -> int:  # kwargs-ok: request kwargs
    host.answer("openai/a", rate_limit())
    error: Final = attempt_sync(core, "a", **request)
    assert isinstance(error, litellm.RateLimitError)
    return len(host.calls_to())


@pytest.mark.parametrize(
    "policies",
    [None, {"a": None}, "xyz", [], ("b",)],
    ids=["none", "none-entry", "string-without-the-group", "empty-list", "tuple-without-the-group"],
)
def test_a_request_policy_that_holds_no_policy_for_the_group_keeps_the_plain_retries(
    make_core: MakeCore, host: ScriptedHost, policies: object
) -> None:
    core: Final = make_core(model_list=GROUPS, num_retries=2)

    assert calls_after_a_rate_limit(core, host, model_group_retry_policy=policies) == 3


@pytest.mark.parametrize(
    ("policies", "calls"),
    [
        ("xyz", 2),  # the group is not in it: the router's retry_policy applies
        ({"a": None}, 3),  # a None entry is no policy: the plain retries apply
        (None, 2),  # an explicit None replaces the router's per-group policies
    ],
    ids=["string-without-the-group", "none-entry", "none"],
)
def test_a_request_policy_without_the_group_leaves_the_router_retry_policy_in_charge(
    make_core: MakeCore, host: ScriptedHost, policies: object, calls: int
) -> None:
    core: Final = make_core(
        model_list=GROUPS,
        num_retries=2,
        retry_policy=RetryPolicy(RateLimitErrorRetries=1),
        model_group_retry_policy={"a": RetryPolicy(RateLimitErrorRetries=0)},
    )

    assert calls_after_a_rate_limit(core, host, model_group_retry_policy=policies) == calls


@pytest.mark.parametrize("policies", [5, "abc", {"a": 3}], ids=["int", "string-holding-the-group", "int-entry"])
def test_a_request_that_opts_out_of_retries_never_reads_its_policy(
    make_core: MakeCore, host: ScriptedHost, policies: object
) -> None:
    core: Final = make_core(model_list=GROUPS, num_retries=2)

    assert calls_after_a_rate_limit(core, host, model_group_retry_policy=policies, num_retries=0) == 1


@pytest.mark.parametrize(
    "policies", [{"a": None}, "xyz", []], ids=["none-entry", "string-without-the-group", "empty-list"]
)
@pytest.mark.parametrize("how", ["constructor", "apply"])
def test_a_router_policy_that_holds_no_policy_for_the_group_keeps_the_plain_retries(
    make_core: MakeCore, host: ScriptedHost, policies: object, how: str
) -> None:
    if how == "constructor":
        core = make_core(model_list=GROUPS, num_retries=2, model_group_retry_policy=policies)
    else:
        core = make_core(model_list=GROUPS, num_retries=2)
        core.apply(ChangeSet(settings={"model_group_retry_policy": policies}))

    stored: Final[object] = getattr(core, "model_group_retry_policy")  # pyright: ignore[reportAny]  # the router attribute
    assert stored == policies
    assert calls_after_a_rate_limit(core, host) == 3


@pytest.mark.parametrize("how", ["constructor", "apply"])
def test_a_router_policy_that_is_no_mapping_is_not_read_when_the_request_opts_out(
    make_core: MakeCore, host: ScriptedHost, how: str
) -> None:
    if how == "constructor":
        core = make_core(model_list=GROUPS, num_retries=2, model_group_retry_policy=5)
    else:
        core = make_core(model_list=GROUPS, num_retries=2)
        core.apply(ChangeSet(settings={"model_group_retry_policy": 5}))

    assert calls_after_a_rate_limit(core, host, num_retries=0) == 1


def test_get_num_retries_from_retry_policy_reads_a_router_policy_that_is_no_mapping(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=GROUPS, retry_policy=RetryPolicy(RateLimitErrorRetries=4), model_group_retry_policy="xyz"
    )
    error: Final = rate_limit()
    retries_for: Final[Callable[..., object]] = getattr(core, "get_num_retries_from_retry_policy")  # pyright: ignore[reportAny]  # the router member

    assert retries_for(error, model_group="a") == 4
    with pytest.raises(TypeError, match="string indices must be integers"):
        retries_for(error, model_group="x")
