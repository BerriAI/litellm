"""
Parity of the Rust core with ``router_python_core.py`` on what a call counts: which deployments track
per-minute usage, when a failure's counters are written on a logging object the attempts share, the
exits of the failed fallback hop's cooldown, the reads and writes that reach Redis, and the keys a
wildcard stream counts under.
"""

import asyncio
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Final

import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.router_core import RouterCore
from litellm.router_utils import cooldown_handlers
from litellm.router_utils.cooldown_handlers import mark_deployment_selection_failure
from tests.unit.test_router.core.endpoints_helpers import EndpointHost
from tests.unit.test_router.core.retries_health_helpers import (
    MESSAGES,
    ErrorBuilder,
    MakeCore,
    attempt,
    attempt_sync,
    cooled_down,
    fresh,
    not_found,
    plain_error,
    rate_limited,
)
from tests.unit.test_router.core.scripted_host import Call, deployment, server_error, timeout
from tests.unit.test_router.core.streaming_helpers import chat_stream, collect_texts, request_logging

USAGE: Final = {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12}
PER_USER_COPILOT: Final = {
    "litellm_credential_name": "ghc",
    "github_copilot_auth_type": "per_user_oauth",
}


async def settle() -> None:
    """Let the recording tasks a stream's end scheduled run."""
    for _ in range(20):
        await asyncio.sleep(0)


class SharedLoggingHost(EndpointHost):
    """``EndpointHost`` that logs on a shared logging object as litellm does: every attempt stamps its
    marker, and only the first failure runs the failure callbacks. ``_arealtime`` answers ``None``."""

    __slots__ = ("timeout_cause",)

    def __init__(self) -> None:
        super().__init__()
        self.timeout_cause: BaseException | None = None

    async def _answer_async(self, op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> object:
        logging_obj: Final = kwargs.get("litellm_logging_obj")
        if isinstance(logging_obj, LiteLLMLogging):
            logging_obj.model_call_details["current_attempt_api_call_start_time"] = object()
        return await super()._answer_async(op, args, kwargs)

    def _finish(self, op: str, call: Call, outcome: object) -> tuple[BaseException | None, object]:
        error, response = super()._finish(op, call, outcome)
        if isinstance(error, litellm.Timeout) and self.timeout_cause is not None:
            error.__cause__ = self.timeout_cause  # rebind-ok: a scripted error is rebuilt, which drops its cause
        return error, None if op == "_arealtime" else response

    async def log_async(self, call: Call, error: BaseException | None, response: object) -> None:
        logging_obj: Final = call.kwargs.get("litellm_logging_obj")
        if error is None or not isinstance(logging_obj, LiteLLMLogging):
            await super().log_async(call, error, response)
            return
        details: Final = logging_obj.model_call_details
        if details.get("has_logged_async_failure") is True:
            return
        details["has_logged_async_failure"] = True
        details["async_failure_logged_attempt"] = details.get("current_attempt_api_call_start_time")
        await super().log_async(call, error, response)

    def log_sync(self, call: Call, error: BaseException | None, response: object) -> None:
        logging_obj: Final = call.kwargs.get("litellm_logging_obj")
        if error is None or not isinstance(logging_obj, LiteLLMLogging):
            super().log_sync(call, error, response)
            return
        if logging_obj.model_call_details.get("has_logged_sync_failure") is True:
            return
        logging_obj.model_call_details["has_logged_sync_failure"] = True
        super().log_sync(call, error, response)


@pytest.fixture
def host() -> SharedLoggingHost:
    return SharedLoggingHost()


def minute() -> str:
    return datetime.now(timezone.utc).strftime("%H-%M")


def chain(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: router arguments
    return make_core(
        model_list=[deployment("a", "a"), deployment("b", "b"), deployment("c", "c")],
        fallbacks=[{"a": ["b"]}, {"b": ["c"]}],
        num_retries=0,
        **arguments,
    )


# --- PV4: which deployments track usage ----------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        (deployment("g", "d1"), (None, None)),
        (deployment("g", "d1", rpm=100), (10, 1)),
        (deployment("g", "d1", itpm=100), (10, 1)),
        (deployment("g", "d1", model_info={"rpm": 100}), (10, 1)),
        (deployment("g", "d1", model_info={"tpm": 100}), (10, 1)),
        (deployment("g", "d1", model_info={"otpm": 100}), (10, 1)),
        ({**deployment("g", "d1"), "tpm": 100}, (10, 1)),
        (deployment("g", "d1", model="gemini/gemini-3.1-flash-lite-preview"), (10, 1)),
    ],
    ids=[
        "no-limit",
        "params-rpm",
        "params-itpm",
        "model-info-rpm",
        "model-info-tpm",
        "model-info-otpm",
        "top-level-tpm",
        "catalog-rpm",
    ],
)
async def test_usage_is_counted_for_a_limit_anywhere_the_python_core_reads_one(
    make_core: MakeCore, entry: dict[str, object], expected: tuple[int | None, int | None]
) -> None:
    core: Final = make_core(model_list=[entry])

    # The scripted host logs a sync call's success apart from the usage `make_call` writes.
    _ = attempt_sync(core, "g")

    assert await core.health().usage("g") == expected


async def test_a_stream_counts_usage_for_a_limit_in_model_info(make_core: MakeCore, host: SharedLoggingHost) -> None:
    host.answer("openai/d1", chat_stream("hello", usage=USAGE, logged=True))
    core: Final = make_core(model_list=[deployment("g", "d1", model_info={"rpm": 100})])

    stream: Final = await core.acompletion(model="g", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # every core serves acompletion
    _ = await collect_texts(stream)
    await settle()

    assert await core.health().usage("g") == (12, 1)


# --- realtime ------------------------------------------------------------------------------------


@pytest.mark.parametrize("limit", ["tpm", "rpm", "itpm", "otpm"])
async def test_a_realtime_session_counts_no_request_against_a_limited_deployment(
    make_core: MakeCore, limit: str
) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1", **{limit: 100})], num_retries=0)

    await core._arealtime(model="g")  # pyright: ignore[reportAttributeAccessIssue]  # every core serves it

    assert await core.health().usage("g") == (None, None)


# --- P3: what a shared logging object lets through -------------------------------------------------


async def test_every_failure_on_a_shared_logging_object_counts_a_request(
    make_core: MakeCore, host: SharedLoggingHost
) -> None:
    core: Final = chain(make_core)
    host.answer("*", fresh(server_error))

    _ = await attempt(core, "a", litellm_logging_obj=request_logging())

    assert await core.health().usage("a") == (None, 1)
    assert await core.health().usage("b") == (None, 1)
    assert await core.health().usage("c") == (None, 1)


async def test_every_failure_counts_a_request_when_each_call_logs_on_its_own(
    make_core: MakeCore, host: SharedLoggingHost
) -> None:
    core: Final = chain(make_core)
    host.answer("*", fresh(server_error))

    _ = await attempt(core, "a")

    assert await core.health().usage("a") == (None, 1)
    assert await core.health().usage("b") == (None, 1)
    assert await core.health().usage("c") == (None, 1)


async def test_a_retry_past_the_first_logged_failure_is_recorded(make_core: MakeCore, host: SharedLoggingHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=1, allowed_fails=5)
    host.answer("*", fresh(server_error))

    _ = await attempt(core, "g", litellm_logging_obj=request_logging())

    assert core.health().failures_this_minute("a") + core.health().failures_this_minute("b") == 2


def test_a_sync_retry_past_the_first_logged_failure_is_recorded(make_core: MakeCore, host: SharedLoggingHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=1, allowed_fails=5)
    host.answer("*", fresh(server_error))

    _ = attempt_sync(core, "g", litellm_logging_obj=request_logging())

    assert core.health().failures_this_minute("a") + core.health().failures_this_minute("b") == 2


async def test_a_retried_deployment_is_cooled_down_past_the_first_logged_failure(
    make_core: MakeCore, host: SharedLoggingHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=1, allowed_fails=0, cooldown_time=60
    )
    host.answer("*", fresh(rate_limited))

    _ = await attempt(core, "g", litellm_logging_obj=request_logging())

    assert sorted(cooled_down(core, ["a", "b"])) == ["a", "b"]


async def test_a_hop_whose_group_is_cooled_records_its_first_failure(
    make_core: MakeCore, host: SharedLoggingHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("a", "a"), deployment("b", "b"), deployment("c", "c")],
        fallbacks=[{"a": ["b"]}, {"b": ["c"]}],
        num_retries=0,
        allowed_fails=0,
        cooldown_time=60,
    )
    host.answer("openai/a", fresh(server_error))
    host.answer("openai/b", fresh(rate_limited))
    _ = await attempt(core, "a", litellm_logging_obj=request_logging())
    assert sorted(cooled_down(core, ["a", "b"])) == ["a", "b"]

    _ = await attempt(core, "a", litellm_logging_obj=request_logging())

    assert core.health().failures_this_minute("b") == 1


def test_a_sync_failure_counts_no_request(make_core: MakeCore, host: SharedLoggingHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    host.answer("*", fresh(server_error))

    with pytest.raises(litellm.InternalServerError):
        core.completion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # see above

    assert core.health().failures_this_minute("d1") == 1
    assert asyncio.run(core.health().usage("g")) == (None, None)


# --- failure counting ------------------------------------------------------------------------------


async def test_a_failure_raised_before_the_litellm_call_records_nothing(
    make_core: MakeCore, host: SharedLoggingHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "d1", model="github_copilot/gpt-4o", **PER_USER_COPILOT)], num_retries=0
    )

    raised: Final = await attempt(core, "g", litellm_credential_name="another")

    assert isinstance(raised, litellm.BadRequestError)
    assert host.calls == []
    assert core.health().failures_this_minute("d1") == 0
    assert await core.health().usage("g") == (None, None)


async def test_an_exception_with_no_status_is_counted_and_never_cooled(
    make_core: MakeCore, host: SharedLoggingHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=0, allowed_fails=0, cooldown_time=60
    )
    host.answer("*", fresh(plain_error))

    _ = await attempt(core, "g")

    assert cooled_down(core, ["a", "b"]) == {}
    assert core.health().failures_this_minute("a") + core.health().failures_this_minute("b") == 1


# --- _trigger_cooldown_for_failed_deployment ---------------------------------------------------------


def selection_failure() -> Exception:
    error: Final = rate_limited()
    mark_deployment_selection_failure(error)
    return error


@pytest.mark.parametrize(
    ("builder", "stalled", "counted", "requests"),
    [
        (server_error, False, 1, 1),
        (selection_failure, False, 1, None),
        (timeout, True, 0, None),
        (timeout, False, 1, 1),
    ],
    ids=["server-error", "selection-tag-not-exempt", "stream-stall-exempt", "plain-timeout"],
)
async def test_a_failed_hop_past_the_first_logged_failure_takes_the_fallback_paths_exits(
    make_core: MakeCore,
    host: SharedLoggingHost,
    builder: ErrorBuilder,
    stalled: bool,
    counted: int,
    requests: int | None,
) -> None:
    core: Final = make_core(
        model_list=[deployment("a", "a"), deployment("b", "b")], fallbacks=[{"a": ["b"]}], num_retries=0
    )
    host.answer("openai/a", fresh(server_error))
    host.answer("openai/b", fresh(builder))
    if stalled:
        host.timeout_cause = MidStreamFallbackError(message="stalled", model="b", llm_provider="openai")

    _ = await attempt(core, "a", litellm_logging_obj=request_logging())

    assert core.health().failures_this_minute("b") == counted
    assert await core.health().usage("b") == (None, requests)


@pytest.mark.parametrize(("logged", "counted"), [(True, 0), (False, 1)], ids=["generic", "chat"])
async def test_a_failed_hops_404_is_the_callers_only_on_a_generic_call(
    make_core: MakeCore, host: SharedLoggingHost, logged: bool, counted: int
) -> None:
    core: Final = make_core(
        model_list=[deployment("a", "a"), deployment("b", "b")], fallbacks=[{"a": ["b"]}], num_retries=0
    )
    host.answer("openai/a", fresh(server_error))
    host.answer("openai/b", fresh(not_found))

    logging_obj: Final = request_logging()
    if logged:
        with pytest.raises(litellm.InternalServerError):
            await core.agenerate_content(model="a", contents=[], litellm_logging_obj=logging_obj)  # pyright: ignore[reportAttributeAccessIssue]  # see above
    else:
        _ = await attempt(core, "a", litellm_logging_obj=logging_obj)

    assert core.health().failures_this_minute("b") == counted


@pytest.mark.parametrize(
    ("params", "router_cooldown", "error", "expected"),
    [
        pytest.param({}, 7, server_error, 7, id="router-int"),
        pytest.param({}, 7.0, server_error, 7.0, id="router-float"),
        pytest.param({"cooldown_time": 7}, 7.0, server_error, 7, id="deployment-int-over-router-float"),
        pytest.param({"cooldown_time": 7.0}, 7, server_error, 7.0, id="deployment-float-over-router-int"),
        pytest.param({}, 9.0, lambda: rate_limited("9"), 9, id="header-int-over-router-float"),
    ],
)
async def test_a_failed_hop_fires_the_number_python_chose_as_it_was_written(
    make_core: MakeCore,
    host: SharedLoggingHost,
    monkeypatch: pytest.MonkeyPatch,
    params: Mapping[str, object],
    router_cooldown: float,
    error: Callable[[], Exception],
    expected: float,
) -> None:
    fired: Final[list[tuple[str, object]]] = []  # mutable-ok: appended as they fire

    async def record(ctx: object, deployment_id: str, exception_status: object, cooldown_time: object) -> None:
        fired.append((deployment_id, cooldown_time))

    monkeypatch.setattr(cooldown_handlers, "router_cooldown_event_callback", record)
    core: Final = make_core(
        model_list=[deployment("a", "a"), deployment("b", "b", **params), deployment("b", "c", **params)],
        fallbacks=[{"a": ["b"]}],
        num_retries=0,
        allowed_fails=0,
        cooldown_time=router_cooldown,
    )
    host.answer("openai/a", fresh(server_error))
    host.answer("openai/b", fresh(error))
    host.answer("openai/c", fresh(error))

    _ = await attempt(core, "a", litellm_logging_obj=request_logging())
    for _ in range(20):
        await asyncio.sleep(0)

    assert [repr(seconds) for name, seconds in fired if name != "a"] == [repr(expected)]


# --- wildcard streams --------------------------------------------------------------------------------


async def test_a_wildcard_stream_counts_its_tokens_under_the_filled_model(
    make_core: MakeCore, host: SharedLoggingHost
) -> None:
    host.answer("openai/gpt-4o", chat_stream("hello", usage=USAGE, logged=True))
    core: Final = make_core(model_list=[deployment("openai/*", "w", model="openai/*", rpm=100)])

    stream: Final = await core.acompletion(model="openai/gpt-4o", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # see above
    _ = await collect_texts(stream)
    await settle()

    filled_tpm, _ = await core.health().usage("openai/gpt-4o")
    wildcard_tpm, _ = await core.health().usage("openai/*")
    assert filled_tpm == 12
    assert wildcard_tpm is None
