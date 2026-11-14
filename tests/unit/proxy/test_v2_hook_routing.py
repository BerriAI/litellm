import asyncio
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from typing import Final

import pytest
from fastapi import HTTPException
from mcp.types import CallToolResult, TextContent

import litellm
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.exceptions import RejectedRequestError, SensitiveDataRouteException
from litellm.integrations.custom_logger_v2 import (
    Abort,
    CallContext,
    CustomLoggerV2,
    Field,
    Hook,
    Patch,
    Respond,
    on,
)
from litellm.integrations.custom_logger_v2.host import V2Outcome, reset_host_for_tests
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.policy_engine.pipeline_types import GuardrailPipeline, PipelineStep
from litellm.types.utils import CallTypesLiteral, GenericGuardrailAPIInputs, ModelResponse
from litellm.utils import (
    async_post_call_failure_deployment_hook,
    async_post_call_success_deployment_hook,
    async_pre_call_deployment_hook,
)

AUTH: Final = UserAPIKeyAuth(api_key="sk-test")


@pytest.fixture(autouse=True)
def fresh_host() -> Generator[None]:
    reset_host_for_tests()
    yield
    reset_host_for_tests()


@pytest.fixture
def proxy_logging() -> ProxyLogging:
    return ProxyLogging(user_api_key_cache=DualCache())


class _Legacy(CustomLogger):
    def __init__(self, order: list[str], reply: object = None, error: Exception | None = None) -> None:
        super().__init__()
        self.order = order
        self.reply = reply
        self.error = error
        self.seen: list[object] = []

    def _record(self, seen: object) -> object:
        self.order.append("legacy")
        self.seen.append(seen)
        if self.error is not None:
            raise self.error
        return self.reply

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: CallTypesLiteral
    ) -> object:
        return self._record(data)

    async def async_moderation_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, call_type: CallTypesLiteral
    ) -> object:
        return self._record(data)

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        return self._record(response)

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> object:
        return self._record(original_exception)

    async def async_post_call_response_headers_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
        request_headers: dict[str, str] | None = None,
        litellm_call_info: dict[str, object] | None = None,
    ) -> object:
        return self._record(response)

    async def async_filter_listed_models(
        self, user_api_key_dict: UserAPIKeyAuth, model_names: Sequence[str]
    ) -> Sequence[str]:
        self._record(tuple(model_names))
        return ("kept",)


class _StreamGuardrail(CustomGuardrail):
    def __init__(self, order: list[str], reply: object) -> None:
        super().__init__(guardrail_name="stream-witness", default_on=True, event_hook=GuardrailEventHooks.post_call)
        self.order = order
        self.reply = reply

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        self.order.append("legacy")
        return self.reply


class _SuccessDispatch:
    def __init__(self) -> None:
        self.model_call_details: dict[str, object] = {"metadata": {}}
        self.dispatched: list[object] = []

    async def dispatch_success_handlers(self, result: object, **kwargs: object) -> None:
        self.dispatched.append(result)


async def test_pre_call_passthrough_hands_the_request_to_legacy_once_and_returns_its_reply(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    reply: Final[dict[str, object]] = {"model": "replaced", "metadata": {}}
    legacy: Final = _Legacy(order, reply=reply)
    monkeypatch.setattr(litellm, "callbacks", [legacy])
    data: Final[dict[str, object]] = {"model": "gpt-4o", "metadata": {}}

    result: Final = await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data=data, call_type="completion")

    assert result is reply
    assert legacy.seen == [data]
    assert legacy.seen[0] is data


async def test_pre_call_passthrough_propagates_the_legacy_exception(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    error: Final = ValueError("rejected by a legacy hook")
    monkeypatch.setattr(litellm, "callbacks", [_Legacy([], error=error)])

    with pytest.raises(ValueError, match="rejected by a legacy hook") as raised:
        await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "gpt-4o"}, call_type="completion")

    assert raised.value is error


async def test_during_call_passthrough_returns_the_request_and_propagates_the_legacy_exception(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    order: Final[list[str]] = []
    legacy: Final = _Legacy(order)
    monkeypatch.setattr(litellm, "callbacks", [legacy])
    data: Final = {"model": "gpt-4o"}

    result: Final = await proxy_logging.during_call_hook(data=data, user_api_key_dict=AUTH, call_type="completion")

    assert result is data
    assert legacy.seen == [data]
    error: Final = RuntimeError("moderation failed")
    monkeypatch.setattr(litellm, "callbacks", [_Legacy([], error=error)])
    with pytest.raises(RuntimeError) as raised:
        await proxy_logging.during_call_hook(data=data, user_api_key_dict=AUTH, call_type="completion")
    assert raised.value is error


async def test_post_call_success_passthrough_returns_the_legacy_replacement(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    replacement: Final = ModelResponse()
    legacy: Final = _Legacy([], reply=replacement)
    monkeypatch.setattr(litellm, "callbacks", [legacy])
    response: Final = ModelResponse()

    result: Final = await proxy_logging.post_call_success_hook(
        data={"model": "gpt-4o"}, response=response, user_api_key_dict=AUTH
    )

    assert result is replacement
    assert legacy.seen == [response]
    assert legacy.seen[0] is response


async def test_post_call_success_passthrough_propagates_the_legacy_exception(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    error: Final = ValueError("response rejected")
    monkeypatch.setattr(litellm, "callbacks", [_Legacy([], error=error)])

    with pytest.raises(ValueError, match="response rejected") as raised:
        await proxy_logging.post_call_success_hook(
            data={"model": "gpt-4o"}, response=ModelResponse(), user_api_key_dict=AUTH
        )

    assert raised.value is error


async def test_post_call_failure_passthrough_returns_the_first_legacy_http_exception(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    transformed: Final = HTTPException(status_code=418, detail="rewritten")
    legacy: Final = _Legacy([], reply=transformed)
    monkeypatch.setattr(litellm, "callbacks", [legacy, _Legacy([], reply=HTTPException(status_code=500))])
    original: Final = RuntimeError("upstream failed")

    result: Final = await proxy_logging.post_call_failure_hook(
        request_data={"model": "gpt-4o"}, original_exception=original, user_api_key_dict=AUTH
    )

    assert result is transformed
    assert legacy.seen == [original]
    assert legacy.seen[0] is original


async def test_response_headers_passthrough_returns_the_legacy_headers(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    legacy: Final = _Legacy([], reply={"x-legacy": "1"})
    monkeypatch.setattr(litellm, "callbacks", [legacy])
    response: Final = ModelResponse()

    result: Final = await proxy_logging.post_call_response_headers_hook(
        data={"model": "gpt-4o"}, user_api_key_dict=AUTH, response=response
    )

    assert result == {"x-legacy": "1"}
    assert legacy.seen == [response]


async def test_list_models_passthrough_returns_the_names_legacy_hides(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    legacy: Final = _Legacy([])
    monkeypatch.setattr(litellm, "callbacks", [legacy])

    result: Final = await proxy_logging.hidden_by_listing_callbacks(AUTH, ["kept", "hidden"])

    assert result == frozenset({"hidden"})
    assert legacy.seen == [("kept", "hidden")]


async def test_post_mcp_call_passthrough_returns_the_result_untouched_without_guardrails(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    response: Final = CallToolResult(content=[TextContent(type="text", text="tool output")])

    result: Final = await proxy_logging.post_mcp_call_hook(response=response, request_data={}, user_api_key_dict=AUTH)

    assert result is response


async def test_deferred_stream_guardrails_passthrough_dispatches_the_guardrail_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: Final[list[str]] = []
    replacement: Final = ModelResponse()
    monkeypatch.setattr(litellm, "callbacks", [_StreamGuardrail(order, replacement)])
    dispatch: Final = _SuccessDispatch()

    await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
        captured_data={"model": "gpt-4o", "metadata": {}},
        captured_user_api_key_dict=AUTH,
        captured_logging_obj=dispatch,
        assembled_response=ModelResponse(),
        cache_hit=None,
    )
    await _drain()

    assert order == ["legacy"]
    assert dispatch.dispatched == [replacement]
    assert dispatch.dispatched[0] is replacement


async def _drain() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


Action = Callable[[dict[str, object]], Awaitable[None] | Exception | str | dict[str, object] | None]


def _passes(data: dict[str, object]) -> None:
    return None


class _PreCall(CustomLogger):
    def __init__(self, name: str, journal: list[object], action: Action = _passes, enforces: bool = False) -> None:
        super().__init__()
        self.name = name
        self.journal = journal
        self.action = action
        self.enforces_request_content = enforces
        self.seen: list[dict[str, object]] = []

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: CallTypesLiteral
    ) -> Exception | str | dict[str, object] | None:
        self.journal.append(self.name)
        self.seen.append(data)
        return await _settled(self.action(data))

    async def async_moderation_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, call_type: CallTypesLiteral
    ) -> object:
        self.journal.append(self.name)
        await asyncio.sleep(0)
        return await _settled(self.action(data))


class _Guard(CustomGuardrail):
    def __init__(
        self,
        name: str,
        journal: list[object],
        action: Action = _passes,
        parallel: bool = False,
        delay: float = 0.0,
        event: GuardrailEventHooks = GuardrailEventHooks.pre_call,
    ) -> None:
        super().__init__(guardrail_name=name, default_on=True, event_hook=event, run_in_parallel=parallel)
        self.name = name
        self.journal = journal
        self.action = action
        self.delay = delay
        self.seen_models: list[object] = []

    async def _run(self, data: dict[str, object]) -> Exception | str | dict[str, object] | None:
        self.journal.append(self.name)
        self.seen_models.append(data.get("model"))
        await asyncio.sleep(self.delay)
        return await _settled(self.action(data))

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: CallTypesLiteral
    ) -> Exception | str | dict[str, object] | None:
        return await self._run(data)

    async def async_moderation_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, call_type: CallTypesLiteral
    ) -> object:
        return await self._run(data)


class _V2Callback(CustomLoggerV2):
    def __init__(self, journal: list[object], replacement: object = None, outcome: Abort | None = None) -> None:
        self.journal = journal
        self.replacement = replacement
        self.outcome = outcome

    @on(Hook.pre_call, reads={Field.messages}, writes={Field.messages}, may_abort=True)
    async def pre_call(self, view: object, ctx: CallContext[object]) -> Patch | Abort | None:
        self.journal.append(("v2", getattr(view, "messages", None)))
        if self.outcome is not None:
            return self.outcome
        return None if self.replacement is None else Patch().set(Field.messages, self.replacement)

    @on(Hook.during_call, reads={Field.messages}, may_abort=True)
    async def during_call(self, view: object, ctx: CallContext[object]) -> Abort | None:
        self.journal.append("v2")
        await asyncio.sleep(0)
        return self.outcome


async def _settled(
    value: Awaitable[None] | Exception | str | dict[str, object] | None,
) -> Exception | str | dict[str, object] | None:
    return await value if isinstance(value, Awaitable) else value


def _reroute(name: str) -> Action:
    def action(data: dict[str, object]) -> None:
        raise SensitiveDataRouteException(
            route_to_model=name, session_id="s", guardrail_name=name, sticky_session_routing=False
        )

    return action


def _raise(error: BaseException) -> Action:
    def action(data: dict[str, object]) -> None:
        raise error

    return action


def _returning(value: str | dict[str, object] | None) -> Action:
    def action(data: dict[str, object]) -> str | dict[str, object] | None:
        return value

    return action


def _with_inert_v2(callbacks: list[object], inert: bool) -> list[object]:
    return [*callbacks, _V2Callback([])] if inert else callbacks


@pytest.mark.parametrize("inert", [False, True])
async def test_pre_call_runs_sequential_entries_in_list_order_then_the_parallel_guardrails(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    journal: Final[list[object]] = []
    callbacks: Final[list[object]] = [
        _Guard("parallel-1", journal, parallel=True),
        _PreCall("logger-1", journal),
        _Guard("sequential", journal),
        _Guard("parallel-2", journal, parallel=True),
        _PreCall("logger-2", journal),
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert_v2(callbacks, inert))

    await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type="completion")

    assert journal == ["logger-1", "sequential", "logger-2", "parallel-1", "parallel-2"]


@pytest.mark.parametrize("inert", [False, True])
async def test_pre_call_dict_replaces_the_request_and_none_keeps_it(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    replacement: Final[dict[str, object]] = {"model": "replaced"}
    first: Final = _PreCall("first", [], _returning(replacement))
    keeper: Final = _PreCall("keeper", [])
    last: Final = _PreCall("last", [])
    monkeypatch.setattr(litellm, "callbacks", _with_inert_v2([first, keeper, last], inert))
    data: Final[dict[str, object]] = {"model": "m"}

    result: Final = await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data=data, call_type="completion")

    assert first.seen[0] is data
    assert keeper.seen[0] is replacement
    assert last.seen[0] is replacement
    assert result is replacement


@pytest.mark.parametrize(("call_type", "error"), [("completion", RejectedRequestError), ("embeddings", HTTPException)])
async def test_pre_call_string_rejects_and_stops_the_hook(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, call_type: CallTypesLiteral, error: type[Exception]
) -> None:
    journal: Final[list[object]] = []
    monkeypatch.setattr(
        litellm, "callbacks", [_PreCall("rejecter", journal, _returning("no thanks")), _PreCall("after", journal)]
    )

    with pytest.raises(error):
        await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type=call_type)

    assert journal == ["rejecter"]


async def test_pre_call_sequential_exception_propagates_unchanged_and_stops_the_hook(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    error: Final = HTTPException(status_code=403, detail="blocked")
    monkeypatch.setattr(litellm, "callbacks", [_Guard("blocker", journal, _raise(error)), _PreCall("after", journal)])

    with pytest.raises(HTTPException) as raised:
        await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type="completion")

    assert raised.value is error
    assert journal == ["blocker"]


@pytest.mark.parametrize("inert", [False, True])
async def test_parallel_guardrails_pick_the_first_block_in_list_order_and_a_block_beats_a_reroute(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    journal: Final[list[object]] = []
    slow_error: Final = ValueError("slow but first")
    callbacks: Final[list[object]] = [
        _Guard("reroute", journal, _reroute("onprem"), parallel=True),
        _Guard("slow", journal, _raise(slow_error), parallel=True, delay=0.02),
        _Guard("fast", journal, _raise(HTTPException(status_code=400)), parallel=True),
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert_v2(callbacks, inert))
    metadata: Final[dict[str, object]] = {}
    data: Final[dict[str, object]] = {"model": "m", "metadata": metadata}

    with pytest.raises(ValueError, match="slow but first") as raised:
        await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data=data, call_type="completion")

    assert raised.value is slow_error
    assert data["model"] == "m"
    assert metadata["applied_guardrails"] == ["slow", "fast"]


async def test_parallel_reroute_alone_reroutes_the_request(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    monkeypatch.setattr(litellm, "callbacks", [_Guard("reroute", journal, _reroute("onprem"), parallel=True)])

    result: Final = await proxy_logging.pre_call_hook(
        user_api_key_dict=AUTH, data={"model": "m", "metadata": {}}, call_type="completion"
    )

    assert result["model"] == "onprem"


@pytest.mark.parametrize("inert", [False, True])
async def test_deferred_reroute_lets_later_guardrails_run_and_the_first_reroute_wins(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    journal: Final[list[object]] = []
    later: Final = _Guard("later", journal)
    parallel: Final = _Guard("parallel", journal, parallel=True)
    callbacks: Final[list[object]] = [
        _Guard("first-reroute", journal, _reroute("first")),
        _Guard("second-reroute", journal, _reroute("second")),
        later,
        parallel,
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert_v2(callbacks, inert))

    result: Final = await proxy_logging.pre_call_hook(
        user_api_key_dict=AUTH, data={"model": "m", "metadata": {}}, call_type="completion"
    )

    assert journal == ["first-reroute", "second-reroute", "later", "parallel"]
    assert later.seen_models == ["m"]
    assert parallel.seen_models == ["first"]
    assert result["model"] == "first"
    assert result["metadata"]["sensitive_data_routing_original_model"] == "m"


async def test_a_block_after_a_deferred_reroute_still_propagates(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    error: Final = HTTPException(status_code=400, detail="blocked")
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [_Guard("reroute", [], _reroute("onprem")), _Guard("blocker", [], _raise(error))],
    )
    data: Final[dict[str, object]] = {"model": "m", "metadata": {}}

    with pytest.raises(HTTPException) as raised:
        await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data=data, call_type="completion")

    assert raised.value is error
    assert data["model"] == "m"


async def test_a_pipeline_managed_guardrail_runs_in_its_pipeline_and_not_again_in_the_hook(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    monkeypatch.setattr(litellm, "callbacks", [_Guard("managed", journal), _Guard("unmanaged", journal)])
    pipeline: Final = GuardrailPipeline(mode="pre_call", steps=[PipelineStep(guardrail="managed", on_pass="next")])
    data: Final[dict[str, object]] = {
        "model": "m",
        "messages": [],
        "metadata": {"_guardrail_pipelines": [("p", pipeline)]},
    }

    await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data=data, call_type="completion")

    assert journal == ["managed", "unmanaged"]


@pytest.mark.parametrize(
    ("skip_guardrails", "guardrails_only", "expected"),
    [
        (False, False, ["logger", "guard", ("v2", None), "enforcer"]),
        (True, False, ["logger", ("v2", None), "enforcer"]),
        (False, True, ["guard", "enforcer"]),
    ],
)
async def test_hook_point_modes_choose_which_entries_run(
    monkeypatch: pytest.MonkeyPatch,
    proxy_logging: ProxyLogging,
    skip_guardrails: bool,
    guardrails_only: bool,
    expected: list[object],
) -> None:
    journal: Final[list[object]] = []
    callbacks: Final[list[object]] = [
        _PreCall("logger", journal),
        _Guard("guard", journal),
        _V2Callback(journal),
        _PreCall("enforcer", journal, enforces=True),
    ]
    monkeypatch.setattr(litellm, "callbacks", callbacks)

    await proxy_logging.pre_call_hook(
        user_api_key_dict=AUTH,
        data={"model": "m"},
        call_type="completion",
        skip_guardrails=skip_guardrails,
        guardrails_only=guardrails_only,
    )

    assert journal == expected


async def test_a_runtime_change_to_the_callback_list_reaches_the_next_request(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    first: Final = _PreCall("first", journal)
    callbacks: Final[list[object]] = [first]
    monkeypatch.setattr(litellm, "callbacks", callbacks)

    await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type="completion")
    callbacks.append(_PreCall("added", journal))
    await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type="completion")
    callbacks.remove(first)
    await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type="completion")

    assert journal == ["first", "first", "added", "added"]


async def test_a_v2_callback_between_two_legacy_loggers_sees_the_first_edit_and_the_second_sees_its_patch(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []

    def edit(data: dict[str, object]) -> None:
        data["messages"] = ["edited by the first"]
        return None

    seen_by_second: Final[list[object]] = []

    def record(data: dict[str, object]) -> None:
        seen_by_second.append(data["messages"])

    first: Final = _PreCall("first", [], edit)
    second: Final = _PreCall("second", [], record)
    monkeypatch.setattr(
        litellm, "callbacks", [first, _V2Callback(journal, replacement=["patched by the v2 callback"]), second]
    )
    data: Final[dict[str, object]] = {"model": "m", "messages": ["original"]}

    result: Final = await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data=data, call_type="completion")

    assert journal == [("v2", ["edited by the first"])]
    assert seen_by_second == [["patched by the v2 callback"]]
    assert result is data
    assert data["messages"] == ["patched by the v2 callback"]


async def test_a_v2_callback_abort_stops_the_hook_before_later_legacy_loggers(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    abort: Final = Abort(status=429, message="slow down")
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [_PreCall("before", journal), _V2Callback(journal, outcome=abort), _PreCall("after", journal)],
    )

    with pytest.raises(HTTPException) as raised:
        await proxy_logging.pre_call_hook(user_api_key_dict=AUTH, data={"model": "m"}, call_type="completion")

    assert raised.value.status_code == 429
    assert raised.value.detail == {"error": "slow down"}
    assert journal == ["before", ("v2", None)]


@pytest.mark.parametrize("inert", [False, True])
async def test_during_call_starts_every_entry_in_list_order_and_returns_on_the_first_failure(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    journal: Final[list[object]] = []
    release: Final = asyncio.Event()
    error: Final = RuntimeError("moderation failed")

    async def hold(data: dict[str, object]) -> None:
        await release.wait()

    slow: Final = _Guard("slow", journal, hold, event=GuardrailEventHooks.during_call)
    callbacks: Final[list[object]] = [
        slow,
        _PreCall("moderator", journal, _raise(error)),
        _Guard("guard", journal, event=GuardrailEventHooks.during_call),
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert_v2(callbacks, inert))

    with pytest.raises(RuntimeError) as raised:
        await asyncio.wait_for(
            proxy_logging.during_call_hook(data={"model": "m"}, user_api_key_dict=AUTH, call_type="completion"), 1
        )

    release.set()
    await _drain()
    assert raised.value is error
    assert journal[:3] == ["slow", "moderator", "guard"]


async def test_during_call_v2_callback_between_legacy_runs_with_them_and_its_abort_wins_by_list_order(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    abort: Final = Abort(status=400, message="flagged")
    callbacks: Final[list[object]] = [
        _PreCall("first", journal),
        _V2Callback(journal, outcome=abort),
        _PreCall("second", journal),
    ]
    monkeypatch.setattr(litellm, "callbacks", callbacks)

    with pytest.raises(HTTPException) as raised:
        await proxy_logging.during_call_hook(data={"model": "m"}, user_api_key_dict=AUTH, call_type="completion")

    assert raised.value.status_code == 400
    assert raised.value.detail == {"error": "flagged"}
    assert journal == ["first", "v2", "second"]


class _Inert(CustomLoggerV2):
    @on(Hook.post_call_success, may_abort=True)
    async def post_call_success(self, view: object, ctx: CallContext[object]) -> None:
        return None

    @on(Hook.post_call_failure, may_abort=True)
    async def post_call_failure(self, view: object, ctx: CallContext[object]) -> None:
        return None

    @on(Hook.response_headers, writes={Field.response_headers})
    async def response_headers(self, view: object, ctx: CallContext[object]) -> None:
        return None

    @on(Hook.list_models, may_abort=True)
    async def list_models(self, view: object, ctx: CallContext[object]) -> None:
        return None

    @on(Hook.mcp_tool_result, may_abort=True)
    async def mcp_tool_result(self, view: object, ctx: CallContext[object]) -> None:
        return None


def _with_inert(callbacks: list[object], inert: bool) -> list[object]:
    return [*callbacks[:1], _Inert(), *callbacks[1:]] if inert else callbacks


def _read(view: object, name: str) -> object:
    return getattr(view, name)  # pyright: ignore[reportAny]  # a view exposes each declared field as an attribute


class _Patcher(CustomLoggerV2):
    def __init__(self, replacement: object = None, outcome: Abort | None = None) -> None:
        self.replacement = replacement
        self.outcome = outcome
        self.seen: list[object] = []

    def _answer(self, field: Field, current: object) -> Patch | Abort | None:
        self.seen.append(current)
        if self.outcome is not None:
            return self.outcome
        return None if self.replacement is None else Patch().set(field, self.replacement)

    @on(Hook.post_call_success, reads={Field.response}, writes={Field.response}, may_abort=True)
    async def post_call_success(self, view: object, ctx: CallContext[object]) -> Patch | Abort | None:
        return self._answer(Field.response, _read(view, "response"))

    @on(Hook.post_call_failure, reads={Field.error}, writes={Field.error}, may_abort=True)
    async def post_call_failure(self, view: object, ctx: CallContext[object]) -> Patch | Abort | None:
        return self._answer(Field.error, _read(view, "error"))

    @on(Hook.list_models, reads={Field.model_list}, writes={Field.model_list}, may_abort=True)
    async def list_models(self, view: object, ctx: CallContext[object]) -> Patch | Abort | None:
        return self._answer(Field.model_list, _read(view, "model_list"))

    @on(Hook.mcp_tool_result, reads={Field.tool_result}, writes={Field.tool_result}, may_abort=True)
    async def mcp_tool_result(self, view: object, ctx: CallContext[object]) -> Patch | Abort | None:
        return self._answer(Field.tool_result, _read(view, "tool_result"))


class _HeaderMerger(CustomLoggerV2):
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers
        self.seen: list[dict[str, object]] = []

    @on(Hook.response_headers, reads={Field.response_headers}, writes={Field.response_headers})
    async def response_headers(self, view: object, ctx: CallContext[object]) -> Patch:
        headers: Final = _read(view, "response_headers")
        assert isinstance(headers, Mapping)
        self.seen.append(dict(headers))
        return Patch().merge(Field.response_headers, self.headers)


class _PostGuard(CustomGuardrail):
    def __init__(
        self,
        name: str,
        journal: list[object],
        reply: object = None,
        error: BaseException | None = None,
        parallel: bool = False,
        delay: float = 0.0,
    ) -> None:
        super().__init__(
            guardrail_name=name, default_on=True, event_hook=GuardrailEventHooks.post_call, run_in_parallel=parallel
        )
        self.name = name
        self.journal = journal
        self.reply = reply
        self.error = error
        self.delay = delay
        self.seen: list[object] = []

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        self.journal.append(self.name)
        self.seen.append(response)
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        self.journal.append(f"{self.name} done")
        return self.reply


class _PostLogger(CustomLogger):
    def __init__(self, name: str, journal: list[object], reply: object = None) -> None:
        super().__init__()
        self.name = name
        self.journal = journal
        self.reply = reply
        self.seen: list[object] = []

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        self.journal.append(self.name)
        self.seen.append(response)
        return self.reply


def _flow_change(name: str) -> SensitiveDataRouteException:
    return SensitiveDataRouteException(
        route_to_model=name, session_id="s", guardrail_name=name, sticky_session_routing=False
    )


async def _post_call(proxy_logging: ProxyLogging, response: ModelResponse) -> object:
    return await proxy_logging.post_call_success_hook(
        data={"model": "m", "metadata": {}}, response=response, user_api_key_dict=AUTH
    )


@pytest.mark.parametrize("inert", [False, True])
async def test_post_call_success_runs_guardrails_then_parallel_guardrails_then_loggers_and_chains_replies(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    journal: Final[list[object]] = []
    from_guardrail: Final = ModelResponse()
    from_logger: Final = ModelResponse()
    parallel: Final = _PostGuard("parallel", journal, reply=ModelResponse(), parallel=True)
    second_logger: Final = _PostLogger("logger-2", journal)
    callbacks: Final[list[object]] = [
        _PostLogger("logger-1", journal, reply=from_logger),
        _PostGuard("guard-1", journal, reply=from_guardrail),
        parallel,
        CustomLogger(),
        second_logger,
        _PostGuard("guard-2", journal),
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert(callbacks, inert))

    result: Final = await _post_call(proxy_logging, ModelResponse())

    assert [entry for entry in journal if isinstance(entry, str) and not entry.endswith("done")] == [
        "guard-1",
        "guard-2",
        "parallel",
        "logger-1",
        "logger-2",
    ]
    assert parallel.seen[0] is from_guardrail
    assert second_logger.seen[0] is from_logger
    assert result is from_logger


@pytest.mark.parametrize("inert", [False, True])
async def test_post_call_success_parallel_guardrails_wait_for_each_other_and_the_first_block_in_list_order_wins(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    journal: Final[list[object]] = []
    slow_block: Final = ValueError("slow but first")
    callbacks: Final[list[object]] = [
        _PostGuard("flow", journal, error=_flow_change("flow"), parallel=True),
        _PostGuard("slow", journal, error=slow_block, parallel=True, delay=0.02),
        _PostGuard("fast", journal, error=HTTPException(status_code=400), parallel=True),
        _PostGuard("passes", journal, parallel=True, delay=0.03),
        _PostLogger("logger", journal),
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert(callbacks, inert))

    with pytest.raises(ValueError, match="slow but first") as raised:
        await _post_call(proxy_logging, ModelResponse())

    assert raised.value is slow_block
    assert "passes done" in journal
    assert "logger" not in journal


@pytest.mark.parametrize("inert", [False, True])
async def test_post_call_success_parallel_flow_change_propagates_only_when_nothing_blocked(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    journal: Final[list[object]] = []
    first: Final = _flow_change("first")
    callbacks: Final[list[object]] = [
        _PostGuard("first", journal, error=first, parallel=True, delay=0.02),
        _PostGuard("second", journal, error=_flow_change("second"), parallel=True),
        _PostLogger("logger", journal),
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert(callbacks, inert))

    with pytest.raises(SensitiveDataRouteException) as raised:
        await _post_call(proxy_logging, ModelResponse())

    assert raised.value is first
    assert "logger" not in journal


async def test_post_call_success_v2_callback_between_two_loggers_sees_the_first_reply_and_the_second_sees_its_patch(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    from_first: Final = ModelResponse()
    from_v2: Final = ModelResponse()
    v2_callback: Final = _Patcher(replacement=from_v2)
    second: Final = _PostLogger("second", [])
    monkeypatch.setattr(litellm, "callbacks", [_PostLogger("first", [], reply=from_first), v2_callback, second])

    result: Final = await _post_call(proxy_logging, ModelResponse())

    assert v2_callback.seen[0] is from_first
    assert second.seen[0] is from_v2
    assert result is from_v2


async def test_post_call_success_v2_callback_abort_stops_the_loggers_after_it(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    abort: Final = Abort(status=400, message="unsafe output")
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [_PostLogger("before", journal), _Patcher(outcome=abort), _PostLogger("after", journal)],
    )

    with pytest.raises(HTTPException) as raised:
        await _post_call(proxy_logging, ModelResponse())

    assert raised.value.status_code == 400
    assert raised.value.detail == {"error": "unsafe output"}
    assert journal == ["before"]


@pytest.mark.parametrize("inert", [False, True])
async def test_post_call_failure_calls_every_logger_and_the_first_http_exception_wins(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    order: Final[list[str]] = []
    raised_first: Final = HTTPException(status_code=429, detail="raised first")
    callbacks: Final[list[object]] = [
        _Legacy(order, error=RuntimeError("swallowed")),
        "not-a-known-integration",
        _Legacy(order, reply="not an exception"),
        _Legacy(order, error=raised_first),
        _passes,
        _Legacy(order, reply=HTTPException(status_code=418, detail="returned later")),
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert(callbacks, inert))

    result: Final = await proxy_logging.post_call_failure_hook(
        request_data={"model": "m"}, original_exception=RuntimeError("upstream"), user_api_key_dict=AUTH
    )

    assert result is raised_first
    assert order == ["legacy", "legacy", "legacy", "legacy"]


async def test_post_call_failure_v2_callback_between_two_loggers_rewrites_the_error_the_second_sees(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    original: Final = RuntimeError("upstream")
    rewritten: Final = RuntimeError("rewritten by the v2 callback")
    first: Final = _Legacy([])
    v2_callback: Final = _Patcher(replacement=rewritten)
    second: Final = _Legacy([])
    monkeypatch.setattr(litellm, "callbacks", [first, v2_callback, second])

    result: Final = await proxy_logging.post_call_failure_hook(
        request_data={"model": "m"}, original_exception=original, user_api_key_dict=AUTH
    )

    assert result is None
    assert first.seen[0] is original
    assert v2_callback.seen[0] is original
    assert second.seen[0] is rewritten


@pytest.mark.parametrize("inert", [False, True])
async def test_response_headers_merge_in_list_order_and_a_failure_keeps_what_was_merged(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    order: Final[list[str]] = []
    callbacks: Final[list[object]] = [
        _Legacy(order, reply={"x-a": "1", "x-b": "1"}),
        CustomLogger(),
        _Legacy(order, reply={"x-b": "2"}),
        _Legacy(order, error=RuntimeError("header hook failed")),
        _Legacy(order, reply={"x-c": "never"}),
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert(callbacks, inert))

    result: Final = await proxy_logging.post_call_response_headers_hook(
        data={"model": "m"}, user_api_key_dict=AUTH, response=ModelResponse()
    )

    assert result == {"x-a": "1", "x-b": "2"}
    assert order == ["legacy", "legacy", "legacy"]


async def test_response_headers_v2_callback_between_two_loggers_adds_headers_and_later_entries_win(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    v2_callback: Final = _HeaderMerger({"x-b": "v2", "x-n": "v2"})
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [_Legacy([], reply={"x-a": "1", "x-b": "1"}), v2_callback, _Legacy([], reply={"x-n": "2"})],
    )

    result: Final = await proxy_logging.post_call_response_headers_hook(
        data={"model": "m"}, user_api_key_dict=AUTH, response=ModelResponse()
    )

    assert v2_callback.seen == [{"x-a": "1", "x-b": "1"}]
    assert result == {"x-a": "1", "x-b": "v2", "x-n": "2"}


class _Lister(CustomLogger):
    def __init__(self, journal: list[object], hides: frozenset[str] = frozenset(), reply: object = None) -> None:
        super().__init__()
        self.journal = journal
        self.hides = hides
        self.reply = reply

    async def async_filter_listed_models(
        self, user_api_key_dict: UserAPIKeyAuth, model_names: Sequence[str]
    ) -> Sequence[str]:
        self.journal.append(tuple(model_names))
        if self.reply is not None:
            return self.reply  # pyright: ignore[reportReturnType]  # a malformed return is the case under test
        return tuple(name for name in model_names if name not in self.hides)


@pytest.mark.parametrize("inert", [False, True])
async def test_list_models_filters_see_the_previous_output_and_stop_once_nothing_is_left(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    journal: Final[list[object]] = []
    callbacks: Final[list[object]] = [
        _Lister(journal, hides=frozenset({"b"})),
        CustomLogger(),
        _Lister(journal, hides=frozenset({"a", "c"})),
        _Lister(journal),
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert(callbacks, inert))

    hidden: Final = await proxy_logging.hidden_by_listing_callbacks(AUTH, ["a", "b", "c"])

    assert hidden == frozenset({"a", "b", "c"})
    assert journal == [("a", "b", "c"), ("a", "c")]


@pytest.mark.parametrize("inert", [False, True])
async def test_list_models_malformed_return_raises_500_and_stops_the_filters(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    journal: Final[list[object]] = []
    monkeypatch.setattr(litellm, "callbacks", _with_inert([_Lister(journal, reply=42), _Lister(journal)], inert))

    with pytest.raises(ProxyException) as raised:
        await proxy_logging.hidden_by_listing_callbacks(AUTH, ["a"])

    assert raised.value.code == "500"
    assert journal == [("a",)]


async def test_list_models_v2_callback_between_two_filters_narrows_what_the_second_sees(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    journal: Final[list[object]] = []
    v2_callback: Final = _Patcher(replacement=("c",))
    monkeypatch.setattr(litellm, "callbacks", [_Lister(journal, hides=frozenset({"a"})), v2_callback, _Lister(journal)])

    hidden: Final = await proxy_logging.hidden_by_listing_callbacks(AUTH, ["a", "b", "c"])

    assert v2_callback.seen == [("b", "c")]
    assert journal == [("a", "b", "c"), ("c",)]
    assert hidden == frozenset({"a", "b"})


class _McpSuffix(CustomGuardrail):
    def __init__(self, suffix: str) -> None:
        super().__init__(guardrail_name=f"mcp{suffix}", default_on=True, event_hook=GuardrailEventHooks.post_mcp_call)
        self.suffix = suffix
        self.seen: list[list[str]] = []

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: str,
        logging_obj: object = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = list(inputs.get("texts") or [])
        self.seen.append(texts)
        return GenericGuardrailAPIInputs(texts=[f"{text}{self.suffix}" for text in texts])


def _tool_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)])


def _texts(result: CallToolResult) -> list[str]:
    return [item.text for item in result.content if isinstance(item, TextContent)]


@pytest.mark.parametrize("inert", [False, True])
async def test_post_mcp_call_guardrails_run_in_list_order_on_the_previous_result(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging, inert: bool
) -> None:
    second: Final = _McpSuffix("-2")
    monkeypatch.setattr(litellm, "callbacks", _with_inert([_McpSuffix("-1"), CustomLogger(), second], inert))

    result: Final = await proxy_logging.post_mcp_call_hook(
        response=_tool_result("out"), request_data={"metadata": {}}, user_api_key_dict=AUTH
    )

    assert second.seen == [["out-1"]]
    assert _texts(result) == ["out-1-2"]


async def test_post_mcp_call_v2_callback_between_two_guardrails_replaces_the_result_the_second_scans(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    replacement: Final = _tool_result("from v2")
    v2_callback: Final = _Patcher(replacement=replacement)
    second: Final = _McpSuffix("-2")
    monkeypatch.setattr(litellm, "callbacks", [_McpSuffix("-1"), v2_callback, second])

    result: Final = await proxy_logging.post_mcp_call_hook(
        response=_tool_result("out"), request_data={"metadata": {}}, user_api_key_dict=AUTH
    )

    seen_by_v2: Final = v2_callback.seen[0]
    assert isinstance(seen_by_v2, CallToolResult)
    assert _texts(seen_by_v2) == ["out-1"]
    assert second.seen == [["from v2"]]
    assert result is replacement
    assert _texts(result) == ["from v2-2"]


async def _deferred(dispatch: _SuccessDispatch) -> None:
    await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
        captured_data={"model": "m", "metadata": {}},
        captured_user_api_key_dict=AUTH,
        captured_logging_obj=dispatch,
        assembled_response=ModelResponse(),
        cache_hit=None,
    )
    await _drain()


@pytest.mark.parametrize("inert", [False, True])
async def test_deferred_stream_guardrails_run_one_at_a_time_and_a_block_is_logged_then_success_is_dispatched(
    monkeypatch: pytest.MonkeyPatch, inert: bool
) -> None:
    journal: Final[list[object]] = []
    replacement: Final = ModelResponse()
    last: Final = _PostGuard("last", journal)
    callbacks: Final[list[object]] = [
        _PostGuard("parallel", journal, reply=replacement, parallel=True, delay=0.01),
        _PostGuard("blocker", journal, error=HTTPException(status_code=400, detail="blocked")),
        last,
    ]
    monkeypatch.setattr(litellm, "callbacks", _with_inert(callbacks, inert))
    dispatch: Final = _SuccessDispatch()

    await _deferred(dispatch)

    assert journal == ["parallel", "parallel done", "blocker", "last", "last done"]
    assert last.seen[0] is replacement
    assert dispatch.model_call_details["metadata"] == {"guardrail_blocked": True}
    assert dispatch.dispatched == [replacement]


async def test_deferred_stream_v2_callback_between_two_guardrails_patches_what_the_second_sees_and_what_is_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from_first: Final = ModelResponse()
    from_v2: Final = ModelResponse()
    v2_callback: Final = _Patcher(replacement=from_v2)
    second: Final = _PostGuard("second", [])
    monkeypatch.setattr(litellm, "callbacks", [_PostGuard("first", [], reply=from_first), v2_callback, second])
    dispatch: Final = _SuccessDispatch()

    await _deferred(dispatch)

    assert v2_callback.seen[0] is from_first
    assert second.seen[0] is from_v2
    assert dispatch.dispatched == [from_v2]


class _Responder(CustomLoggerV2):
    @on(Hook.post_call_success, may_abort=True)
    async def post_call_success(self, view: object, ctx: CallContext[object]) -> Respond:
        return Respond(response={"answer": "from the v2 callback"})


async def test_post_call_success_v2_abort_raises_the_exception_it_carries(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    error: Final = ProxyException(message="budget exceeded", type="budget_exceeded", param=None, code=400)
    monkeypatch.setattr(litellm, "callbacks", [_Patcher(outcome=Abort(status=400, message="ignored", exception=error))])

    with pytest.raises(ProxyException) as raised:
        await _post_call(proxy_logging, ModelResponse())

    assert raised.value is error


async def test_post_call_success_v2_respond_is_not_wired_and_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_Responder()])

    with pytest.raises(V2Outcome) as raised:
        await _post_call(proxy_logging, ModelResponse())

    assert isinstance(raised.value.outcome, Respond)


async def test_post_call_failure_v2_abort_becomes_the_transformed_error_and_never_raises(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    first: Final = _Legacy([])
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [first, _Patcher(outcome=Abort(status=429, message="slow down", headers={"retry-after": "3"}))],
    )

    result: Final = await proxy_logging.post_call_failure_hook(
        request_data={"model": "m"}, original_exception=RuntimeError("upstream"), user_api_key_dict=AUTH
    )

    assert isinstance(result, HTTPException)
    assert result.status_code == 429
    assert result.detail == {"error": "slow down"}
    assert result.headers == {"retry-after": "3"}
    assert len(first.seen) == 1


async def test_list_models_v2_abort_raises_its_status(
    monkeypatch: pytest.MonkeyPatch, proxy_logging: ProxyLogging
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_Patcher(outcome=Abort(status=403, message="no models for you"))])

    with pytest.raises(HTTPException) as raised:
        await proxy_logging.hidden_by_listing_callbacks(AUTH, ["a"])

    assert raised.value.status_code == 403


class _DeploymentAbort(CustomLoggerV2):
    def __init__(self, outcome: Abort) -> None:
        self.outcome = outcome

    @on(Hook.pre_deployment_call, may_abort=True)
    async def pre_deployment_call(self, view: object, ctx: CallContext[object]) -> Abort:
        return self.outcome

    @on(Hook.post_deployment_success, may_abort=True)
    async def post_deployment_success(self, view: object, ctx: CallContext[object]) -> Abort:
        return self.outcome

    @on(Hook.post_deployment_failure, may_abort=True)
    async def post_deployment_failure(self, view: object, ctx: CallContext[object]) -> Abort:
        return self.outcome


def _deployment_hook(hook: Hook) -> Awaitable[object]:
    request: Final[dict[str, object]] = {"model": "m", "messages": []}
    if hook is Hook.pre_deployment_call:
        return async_pre_call_deployment_hook(request, "acompletion", include_v2=True)
    return async_post_call_success_deployment_hook(request, ModelResponse(), "acompletion", include_v2=True)


@pytest.mark.parametrize("hook", [Hook.pre_deployment_call, Hook.post_deployment_success])
async def test_a_v2_abort_at_a_deployment_hook_raises_the_exception_it_carries(
    monkeypatch: pytest.MonkeyPatch, hook: Hook
) -> None:
    error: Final = litellm.RateLimitError(message="no capacity", llm_provider="openai", model="m")
    monkeypatch.setattr(litellm, "callbacks", [_DeploymentAbort(Abort(status=429, message="ignored", exception=error))])

    with pytest.raises(litellm.RateLimitError) as raised:
        await _deployment_hook(hook)

    assert raised.value is error


class _FailingAtDeploymentFailure(CustomLoggerV2):
    @on(Hook.post_deployment_failure, may_abort=True)
    async def post_deployment_failure(self, view: object, ctx: CallContext[object]) -> None:
        raise KeyError("usage")


@pytest.mark.parametrize("raising", [False, True])
async def test_a_v2_abort_or_error_at_post_deployment_failure_keeps_the_original_failure(
    monkeypatch: pytest.MonkeyPatch, raising: bool
) -> None:
    error: Final = litellm.RateLimitError(message="no capacity", llm_provider="openai", model="m")
    v2: Final = (
        _FailingAtDeploymentFailure()
        if raising
        else _DeploymentAbort(Abort(status=429, message="ignored", exception=error))
    )
    monkeypatch.setattr(litellm, "callbacks", [v2])

    returned: Final = await async_post_call_failure_deployment_hook(
        {"model": "m"}, RuntimeError("upstream"), "acompletion", include_v2=True
    )

    assert returned is None
