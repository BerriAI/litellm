"""
Parity of the Rust core with ``router_python_core.py`` on the error causes of the regress5 pack: when
the routing note is added and what it says, the error an order chain raises, what a blocked response
and a failed attempt stamp and record, the names the errors carry, and non-string names at the lookup
members.
"""

import asyncio
import datetime
from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.exceptions import GuardrailRaisedException, UnsupportedEndpointError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.router import Router
from litellm.router_core import RouterCore
from tests.unit.test_router.core.endpoints_helpers import EndpointHost, patch_direct_endpoints
from tests.unit.test_router.core.fallbacks_helpers import (
    MESSAGES,
    MakeCore,
    build,
    complete,
    complete_sync,
    show_router_debug,
)
from tests.unit.test_router.core.scripted_host import (
    auth_error,
    bad_request,
    completion_response,
    deployment,
    rate_limit,
    server_error,
    timeout,
)
from tests.unit.test_router.core.streaming_helpers import chat_stream, collect_texts, install_orjson_stub_if_missing

install_orjson_stub_if_missing()

NOTE: Final = "LiteLLM: model group"
NO_FALLBACK: Final = "\n\nLiteLLM: model group 'a' failed with the error above. No fallback was attempted."
LONG_MESSAGE: Final = "x" * 900


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> EndpointHost:
    endpoint_host: Final = EndpointHost()
    patch_direct_endpoints(monkeypatch, endpoint_host)
    return endpoint_host


def as_router(core: RouterCore) -> Router:
    router: Final = Router.__new__(Router)
    router._attach(core)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType]  # the facade's own attach
    return router


def content_filtered() -> object:
    response: Final = completion_response()
    response.choices[0].finish_reason = "content_filter"  # pyright: ignore[reportAttributeAccessIssue]  # Choices
    return response


def cooled(core: RouterCore, *ids: str) -> Mapping[str, Mapping[str, object]]:
    return dict(core.health().cooldowns.get_active_cooldowns(list(ids), None))  # pyright: ignore[reportReturnType]  # a mapping of entries


async def settle() -> None:
    """Let the tasks a failed stream scheduled run."""
    while pending := asyncio.all_tasks() - {asyncio.current_task()}:
        _, still_running = await asyncio.wait(pending, timeout=5)
        assert not still_running, still_running


async def failed_with(core: RouterCore, **request: object) -> BaseException:  # kwargs-ok: request kwargs
    with pytest.raises(Exception) as raised:  # noqa: PT011  # any failure: each caller asserts on what it gets back
        await complete(core, **request)
    return raised.value


# --- N2: when the note is added -------------------------------------------------------------------


async def test_disable_fallbacks_raises_with_no_note(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}])
    show_router_debug(host)
    host.answer("openai/a", server_error())
    error: Final = await failed_with(core, disable_fallbacks=True)
    assert NOTE not in str(error)
    assert host.deployments_called() == ["a"]


async def test_a_guardrail_block_raises_with_no_note(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}])
    show_router_debug(host)
    host.answer("openai/a", GuardrailRaisedException(guardrail_name="g", message="blocked"))
    error: Final = await failed_with(core)
    assert isinstance(error, GuardrailRaisedException)
    assert NOTE not in str(error)


async def test_a_providers_bad_request_falls_back_and_notes_the_failed_fallback(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}])
    show_router_debug(host)
    host.answer("openai/a", bad_request())
    host.answer("openai/b", bad_request())
    error: Final = await failed_with(core)
    assert f"{NOTE} 'a' failed with the error above. Fallback to b also failed" in str(error)
    assert host.deployments_called() == ["a", "b"]


def test_a_single_attempt_endpoint_raises_with_no_note(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}], num_retries=1)
    show_router_debug(host)
    host.answer("openai/a", server_error())
    with pytest.raises(litellm.InternalServerError) as raised:
        core.ocr(model="a", document={})  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert NOTE not in str(raised.value)
    assert host.deployments_called() == ["a"]


ROUTER_RETRIED: Final[dict[str, tuple[str, str, Mapping[str, object]]]] = {
    "responses": ("responses", "aresponses", {"input": "hi"}),
    "generate_content": ("generate_content", "agenerate_content", {"contents": []}),
}


async def call_routed(core: RouterCore, endpoint: str, mode: str) -> object:
    sync_name, async_name, request = ROUTER_RETRIED[endpoint]
    if mode == "sync":
        return getattr(core, sync_name)(model="a", **request)
    return await getattr(core, async_name)(model="a", **request)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("endpoint", sorted(ROUTER_RETRIED))
async def test_a_router_retried_endpoint_retries_a_failed_attempt_with_the_router_num_retries(
    make_core: MakeCore, host: EndpointHost, endpoint: str, mode: str
) -> None:
    core: Final = build(make_core, num_retries=1)
    recovered: Final = completion_response(content="recovered")
    host.answer("openai/a", server_error(), recovered)
    assert await call_routed(core, endpoint, mode) is recovered
    assert host.deployments_called() == ["a", "a"]


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("endpoint", sorted(ROUTER_RETRIED))
async def test_a_router_retried_endpoint_falls_back_to_the_next_group_when_its_group_fails(
    make_core: MakeCore, host: EndpointHost, endpoint: str, mode: str
) -> None:
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}])
    fallback: Final = completion_response(content="from b")
    host.answer("openai/a", server_error())
    host.answer("openai/b", fallback)
    assert await call_routed(core, endpoint, mode) is fallback
    assert host.deployments_called() == ["a", "b"]


async def test_an_empty_request_list_says_no_fallbacks_are_configured(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = build(make_core)
    show_router_debug(host)
    host.answer("openai/a", rate_limit())
    message: Final = str(await failed_with(core, fallbacks=[]))
    assert "and no fallback model group was found for it" in message
    assert "No fallbacks are configured." in message


def test_a_strategy_marker_group_with_an_empty_list_says_no_fallbacks_are_configured(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(
        model_list=[
            {
                "model_name": "qr",
                "litellm_params": {
                    "model": "auto_router/quality_router",
                    "quality_router_default_model": "t1",
                    "quality_router_config": {
                        "available_models": ["t1", "t3"],
                        "complexity_to_quality": {"SIMPLE": 2, "REASONING": 3},
                    },
                },
                "model_info": {"id": "qr"},
            },
            deployment("t1", "t1", model_info={"litellm_routing_preferences": {"quality_tier": 1}}),
            deployment("t3", "t3", model_info={"litellm_routing_preferences": {"quality_tier": 3}}),
        ],
        num_retries=0,
    )
    show_router_debug(host)
    with pytest.raises(
        Exception, match="model group 'qr' failed with the error above and no fallback model group was found"
    ) as raised:
        complete_sync(core, "qr", fallbacks=[])
    assert "No fallbacks are configured." in str(raised.value)


async def test_a_chain_that_only_names_the_failed_group_reads_the_primary_error(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = build(make_core, fallbacks=[{"a": ["a"]}])
    show_router_debug(host)
    host.answer("openai/a", rate_limit())
    assert str(await failed_with(core)).endswith(
        "LiteLLM: model group 'a' failed with the error above. "
        "Fallback to a also failed: litellm.RateLimitError: slow down"
    )


async def test_a_depth_spent_before_any_hop_reads_the_primary_error(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}], max_fallbacks=0)
    show_router_debug(host)
    host.answer("openai/a", rate_limit())
    assert str(await failed_with(core)).endswith("Fallback to b also failed: litellm.RateLimitError: slow down")
    assert host.deployments_called() == ["a"]


# --- U11: an order chain ---------------------------------------------------------------------------


async def test_every_order_level_failing_raises_the_last_levels_error_with_no_note(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("bb", "bb1", order=1), deployment("bb", "bb2", order=2)], num_retries=0
    )
    show_router_debug(host)
    host.answer("openai/bb1", rate_limit(message="first"))
    host.answer("openai/bb2", rate_limit(message="second"))
    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core, "bb")
    assert str(raised.value) == "litellm.RateLimitError: second"
    assert getattr(raised.value, "failed_deployment_id", None) == "bb2"


# --- R13: a blocked response ------------------------------------------------------------------------


async def test_a_blocked_response_stamps_the_deployment(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = build(make_core, content_policy_fallbacks=[{"a": ["a"]}])
    host.answer("openai/a", content_filtered())
    error: Final = await failed_with(core)
    assert isinstance(error, litellm.ContentPolicyViolationError)
    assert getattr(error, "failed_deployment_id", None) == "a"


def test_a_blocked_sync_response_stamps_the_deployment(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = build(make_core, content_policy_fallbacks=[{"a": ["a"]}])
    host.answer("openai/a", content_filtered())
    with pytest.raises(litellm.ContentPolicyViolationError) as raised:
        complete_sync(core)
    assert getattr(raised.value, "failed_deployment_id", None) == "a"


async def test_a_blocked_response_counts_as_a_failure(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = build(make_core, content_policy_fallbacks=[{"a": ["a"]}])
    host.answer("openai/a", content_filtered())
    await failed_with(core)
    assert core.fail_calls["openai/a"] == 1  # pyright: ignore[reportAttributeAccessIssue]  # the router's live counters
    assert core.success_calls["openai/a"] == 0  # pyright: ignore[reportAttributeAccessIssue]  # see above


# --- R12, S11, U6: what a cooldown keeps and what a timeout says --------------------------------------


async def test_a_cooldown_keeps_the_timeout_as_it_was_raised(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = make_core(
        model_list=[deployment("a", "a1", weight=1), deployment("a", "a2", weight=0)], num_retries=0, allowed_fails=0
    )
    show_router_debug(host)
    host.answer("openai/a1", timeout())
    error: Final = await failed_with(core)
    assert "Deployment Info" in str(error)
    assert len(str(cooled(core, "a1")["a1"]["exception_received"])) == len(str(timeout()))


async def test_a_cooldown_keeps_the_whole_error(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = make_core(
        model_list=[deployment("a", "a1", weight=1), deployment("a", "a2", weight=0)], num_retries=0, allowed_fails=0
    )
    host.answer("openai/a1", rate_limit(message=LONG_MESSAGE))
    await failed_with(core)
    expected: Final = str(rate_limit(message=LONG_MESSAGE))
    assert len(str(cooled(core, "a1")["a1"]["exception_received"])) == len(expected)


async def test_a_stream_failing_on_its_first_chunk_cools_with_the_routing_note(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(model_list=[deployment("a", "a")], num_retries=2, retry_after=0)
    show_router_debug(host)
    host.answer("openai/a", chat_stream(error=auth_error(), fail_after=0, logged=True))
    error: Final = await failed_with(core, stream=True)
    await settle()
    assert NOTE in str(error)
    assert len(str(cooled(core, "a")["a"]["exception_received"])) == len(str(error))


def test_a_sync_timeout_says_nothing_of_the_deployment(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = build(make_core)
    show_router_debug(host)
    host.answer("openai/a", timeout())
    with pytest.raises(litellm.Timeout) as raised:
        complete_sync(core)
    assert "Deployment Info" not in str(raised.value)


async def test_an_async_timeout_counts_no_failure(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = build(make_core)
    host.answer("openai/a", timeout())
    await failed_with(core)
    assert core.fail_calls["openai/a"] == 0  # pyright: ignore[reportAttributeAccessIssue]  # the router's live counters


# --- R14, N19: a name nothing serves ------------------------------------------------------------------


async def test_a_default_fallback_with_no_deployment_is_the_name_the_error_gives(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0, default_fallbacks=["missing"])
    with pytest.raises(litellm.BadRequestError) as raised:
        await complete(core, "nope")
    assert raised.value.model == "missing"
    assert "You passed in model=missing." in str(raised.value)


def test_an_empty_router_names_the_first_default_fallback(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[], num_retries=1, retry_after=0, default_fallbacks=["team-gpt", "gpt-4o-mini"])
    with pytest.raises(litellm.BadRequestError) as raised:
        complete_sync(core, "gpt-4o")
    assert raised.value.model == "team-gpt"


async def test_an_unknown_alias_target_is_the_name_the_error_gives(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0, model_group_alias={"al": "nowhere"})
    with pytest.raises(litellm.BadRequestError) as raised:
        await complete(core, "al")
    assert raised.value.model == "nowhere"


async def test_a_missing_message_list_through_a_wildcard_alias_reaches_the_call(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("openai/*", "w", model="openai/*"), deployment("g", "d1")],
        num_retries=0,
        model_group_alias={"al": {"hidden": False, "model": "gpt-4o-mini"}},
    )
    await core.acompletion(model="al", messages=None)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.deployments_called("acompletion") == ["w"]


async def test_moderation_of_a_name_only_a_wildcard_serves_uses_the_wildcard(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(model_list=[deployment("openai/*", "w", model="openai/*")], num_retries=0)
    await core.amoderation(model="openai/omni-moderation-latest", input="hello")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.deployments_called("amoderation") == ["w"]


# --- R15, T7: pass-through ------------------------------------------------------------------------


async def test_a_listed_pick_of_an_id_names_its_litellm_model(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "p1", model="openai/b"), deployment("g", "p4", use_in_pass_through=True)],
        num_retries=0,
    )
    with pytest.raises(litellm.BadRequestError, match="Deployment p1 does not support pass-through") as raised:
        await core.select("p1", {}, kind="pass_through")  # pyright: ignore[reportCallIssue]  # PythonCore's select takes keyword options
    assert raised.value.model == "openai/b"


async def test_an_async_pick_of_an_id_names_the_id(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "p1", model="openai/b"), deployment("g", "p4", use_in_pass_through=True)],
        num_retries=0,
    )
    with pytest.raises(litellm.BadRequestError, match="Deployment p1 does not support pass-through") as raised:
        await core.async_get_available_deployment_for_pass_through("p1", {}, messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the async pass-through pick
    assert raised.value.model == "p1"


def unknown_provider(model: str = "nope") -> litellm.BadRequestError:
    return litellm.BadRequestError(message="LLM Provider NOT provided", model=model, llm_provider="")


async def test_a_pass_through_to_an_unknown_model_carries_the_note(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    show_router_debug(host)
    host.answer("nope", unknown_provider())
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.allm_passthrough_route(model="nope", method="GET", endpoint="/models")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert str(raised.value).endswith(
        "\n\nLiteLLM: model group 'nope' failed with the error above. No fallback was attempted."
    )


async def test_a_pass_through_with_no_deployment_calls_the_handler_before_any_fallback(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0, fallbacks=[{"nope": ["g"]}])
    host.answer("nope", unknown_provider())
    await core.allm_passthrough_route(model="nope", method="GET", endpoint="/models")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert [call.model for call in host.calls_to("allm_passthrough_route")] == ["nope", "openai/d1"]


# --- S9: the deployment an exception names --------------------------------------------------------


def unsupported() -> UnsupportedEndpointError:
    return UnsupportedEndpointError(message="not here", llm_provider="openai", model="gpt-4o")


async def test_an_exception_naming_a_retried_deployment_exhausts_the_retries(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "d1", weight=1), deployment("g", "d2", weight=0)], num_retries=1, retry_after=0
    )
    reused: Final = unsupported()
    reused.failed_deployment_id = "d1"  # pyright: ignore[reportAttributeAccessIssue]  # what the first failure stamped
    host.answer("openai/d1", unsupported())
    host.answer("openai/d2", reused)
    with pytest.raises(UnsupportedEndpointError) as raised:
        await complete(core, "g")
    assert (raised.value.num_retries, raised.value.max_retries) == (1, 1)
    assert "LiteLLM Retried: 1 times" in str(raised.value)
    assert host.deployments_called("acompletion") == ["d1", "d2"]


async def test_every_deployment_refusing_raises_before_the_retries_are_exhausted(
    make_core: MakeCore, host: EndpointHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "d1", weight=1), deployment("g", "d2", weight=0)], num_retries=1, retry_after=0
    )
    host.answer("*", unsupported())
    with pytest.raises(UnsupportedEndpointError) as raised:
        await complete(core, "g")
    assert raised.value.max_retries is None
    assert host.deployments_called("acompletion") == ["d1", "d2"]


# --- N5: model=None -----------------------------------------------------------------------------------


async def test_model_none_is_a_bad_request(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    show_router_debug(host)
    with pytest.raises(litellm.BadRequestError, match="You passed in model=None") as raised:
        await core.acompletion(model=None, messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert raised.value.status_code == 400
    assert NOTE not in str(raised.value)


def test_sync_model_none_is_a_bad_request(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    with pytest.raises(litellm.BadRequestError, match="You passed in model=None"):
        core.completion(model=None, messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint


# --- N4: lookup members given a non-string name ---------------------------------------------------


def test_a_non_string_name_is_not_recognized(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    assert core.registry().resolve(True, kind="recognized") is False


def test_an_unhashable_name_raises_as_the_index_lookup_raises(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1")], num_retries=0)
    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        core.registry().resolve(["g"], kind="recognized")  # pyright: ignore[reportArgumentType]  # raising is the point


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        ("get_configured_display_name", None),
        ("get_configured_mode", None),
        ("get_deployment", None),
        ("get_model_ids", []),
    ],
)
def test_a_lookup_given_an_int_finds_nothing(make_core: MakeCore, member: str, expected: object) -> None:
    router: Final = as_router(make_core(model_list=[deployment("g", "d1")], num_retries=0))
    with pytest.warns(DeprecationWarning, match="is deprecated"):
        assert getattr(router, member)(1) == expected  # pyright: ignore[reportAny]  # each old member's own type


def test_a_model_list_lookup_given_an_int_raises_as_the_wildcard_step_raises(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("g", "d1")], num_retries=0))
    with (
        pytest.warns(DeprecationWarning, match="is deprecated"),
        pytest.raises(TypeError, match="argument of type 'int' is not"),
    ):
        router.get_model_list(1)  # pyright: ignore[reportArgumentType]  # raising is the point


# --- V6: a base_model that is no string -----------------------------------------------------------


def test_an_int_base_model_fails_as_startswith_fails(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "az1", model="azure/my-dep", api_base="https://x", base_model=7)], num_retries=0
    )
    with pytest.raises(AttributeError, match=r"^'int' object has no attribute 'startswith'$"):
        core.registry().router_model_info(None, "g", id="az1")


# --- a shared logging object ----------------------------------------------------------------------


async def test_a_failure_already_logged_blocks_a_streams_count(make_core: MakeCore, host: EndpointHost) -> None:
    core: Final = make_core(
        model_list=[deployment("a", "a1", weight=1), deployment("a", "a2", weight=0)], num_retries=0, allowed_fails=0
    )
    logging_obj: Final = LiteLLMLogging(
        model="a",
        messages=MESSAGES,
        stream=True,
        call_type="acompletion",
        start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="c",
        function_id="f",
    )
    logging_obj.update_environment_variables(model="a", user=None, optional_params={}, litellm_params={})
    logging_obj.model_call_details["has_logged_async_failure"] = True
    host.answer("openai/a1", chat_stream("hi", error=rate_limit(), fail_after=1))
    stream: Final = await complete(core, stream=True, litellm_logging_obj=logging_obj)
    with pytest.raises(litellm.RateLimitError):
        await collect_texts(stream)
    await settle()
    assert cooled(core, "a1", "a2") == {}
