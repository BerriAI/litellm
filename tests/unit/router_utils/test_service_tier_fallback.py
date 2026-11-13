from collections.abc import Mapping
from typing import Final

import pytest

from litellm import Router
from litellm.cost_calculator import completion_cost
from litellm.exceptions import AuthenticationError, BadRequestError, RateLimitError, Timeout, UnsupportedParamsError
from litellm.litellm_core_utils.hidden_params import get_hidden_params
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.router_config import build_router_config
from litellm.router_utils.service_tier_fallback import FlexTierFallback
from tests.unit.test_router.core.scripted_host import ScriptedHost, completion_response, deployment
from tests.unit.test_router.core.streaming_helpers import chat_stream, collect_texts


@pytest.mark.parametrize(
    "error,trigger",
    (
        (RateLimitError("busy", "openai", "model"), True),
        (Timeout("slow", "model", "openai"), True),
        (UnsupportedParamsError("service_tier='flex' unsupported"), True),
        (UnsupportedParamsError("temperature unsupported"), False),
        (BadRequestError("bad", "model", "openai"), False),
        (AuthenticationError("unauthorized", "model", "openai"), False),
    ),
)
def test_trigger_only_accepts_retryable_flex_errors(error: Exception, trigger: bool) -> None:
    assert FlexTierFallback.is_trigger(error) is trigger


@pytest.mark.parametrize(
    "enabled,tier,expected",
    ((True, "flex", True), (False, "flex", False), (True, "default", False), (True, None, False)),
)
def test_fallback_requires_opt_in_and_flex(enabled: bool, tier: str | None, expected: bool) -> None:
    assert FlexTierFallback.applies({"service_tier": tier}, enabled) is expected


def _router(host: ScriptedHost, **settings: object) -> Router:  # kwargs-ok: router constructor settings
    return Router.from_config(
        build_router_config(
            {
                "model_list": [deployment("group", "only")],
                "num_retries": 0,
                "allowed_fails": 0,
                "cooldown_time": 60,
                **settings,
            }
        ),
        host,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    (
        RateLimitError("flex unavailable", "openai", "model"),
        Timeout("slow", "model", "openai"),
        UnsupportedParamsError("service_tier='flex' unsupported"),
    ),
)
async def test_flex_failure_does_not_cool_down_successful_default_deployment(error: Exception) -> None:
    host: Final = ScriptedHost({"*": (error, None)})
    router: Final = _router(host)
    response: Final = await router.acompletion(
        model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex"
    )
    await router.acompletion(model="group", messages=[{"role": "user", "content": "again"}])
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == ("flex", None, None)
    assert (get_hidden_params(response) or {})["additional_headers"]["x-litellm-service-tier"] == "default"
    metadata: Final = host.calls[1].kwargs.get("metadata")
    assert isinstance(metadata, Mapping)
    assert metadata["service_tier_fallback"] == {"from": "flex", "to": "default"}
    assert StandardLoggingPayloadSetup.get_standard_logging_metadata(metadata)["service_tier_fallback"] == {
        "from": "flex",
        "to": "default",
    }


@pytest.mark.asyncio
async def test_default_failure_reaches_normal_router_fallback() -> None:
    host: Final = ScriptedHost(
        {
            "*": (
                RateLimitError("busy", "openai", "model"),
                RateLimitError("busy", "openai", "model"),
                UnsupportedParamsError("service_tier='flex' unsupported"),
                None,
            )
        }
    )
    router: Final = _router(
        host,
        model_list=[deployment("group", "only"), deployment("backup", "backup")],
        fallbacks=[{"group": ["backup"]}],
    )
    await router.acompletion(model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex")
    assert tuple(call.deployment_id for call in host.calls) == ("only", "only", "backup", "backup")
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == ("flex", None, "flex", None)


@pytest.mark.asyncio
async def test_downgrade_rebuilds_default_tier_timeout() -> None:
    host: Final = ScriptedHost({"*": (Timeout("slow", "model", "openai"), None)})
    router: Final = _router(
        host, model_list=[deployment("group", "only", timeout=90, timeout_by_service_tier={"flex": 3})]
    )
    await router.acompletion(model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex")
    assert tuple(call.kwargs.get("timeout") for call in host.calls) == (3, 90)


def test_sync_completion_retries_default_tier() -> None:
    host: Final = ScriptedHost({"*": (RateLimitError("busy", "openai", "model"), None)})
    router: Final = _router(host)
    router.completion(model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex")
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == ("flex", None)


@pytest.mark.asyncio
async def test_explicit_false_keeps_original_failure() -> None:
    host: Final = ScriptedHost({"*": (RateLimitError("busy", "openai", "model"), None)})
    router: Final = _router(host, service_tier_fallbacks=False)
    with pytest.raises(RateLimitError):
        await router.acompletion(model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex")
    assert len(host.calls) == 1


@pytest.mark.asyncio
async def test_default_router_downgrades_flex_failure() -> None:
    host: Final = ScriptedHost({"*": (RateLimitError("busy", "openai", "model"), None)})
    router: Final = Router.from_config(build_router_config({"model_list": [deployment("group", "only")]}), host)
    response: Final = await router.acompletion(
        model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex"
    )
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == ("flex", None)
    assert (get_hidden_params(response) or {})["additional_headers"]["x-litellm-service-tier"] == "default"


@pytest.mark.asyncio
async def test_downgraded_cost_uses_base_pricing_without_response_tier() -> None:
    host: Final = ScriptedHost({"*": (RateLimitError("busy", "openai", "model"), completion_response())})
    router: Final = _router(host)
    response: Final = await router.acompletion(
        model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex"
    )
    assert completion_cost(completion_response=response, optional_params=dict(host.calls[1].kwargs)) == completion_cost(
        completion_response=response
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", (BadRequestError("bad", "model", "openai"), AuthenticationError("unauthorized", "model", "openai"))
)
async def test_non_trigger_errors_do_not_downgrade(error: Exception) -> None:
    host: Final = ScriptedHost({"*": (error, None)})
    router: Final = _router(host)
    with pytest.raises(type(error)):
        await router.acompletion(model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex")
    assert len(host.calls) == 1


def test_setting_can_be_updated() -> None:
    router: Final = _router(ScriptedHost(), service_tier_fallbacks=False)
    router.update_settings(service_tier_fallbacks=True)
    assert router.get_settings()["service_tier_fallbacks"] is True


@pytest.mark.asyncio
async def test_responses_non_stream_retries_default_tier() -> None:
    host: Final = ScriptedHost({"*": (RateLimitError("busy", "openai", "model"), None)})
    router: Final = _router(host)
    response: Final = await router.aresponses(model="group", input="hello", service_tier="flex")
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == ("flex", None)
    assert (get_hidden_params(response) or {})["additional_headers"]["x-litellm-service-tier"] == "default"


@pytest.mark.asyncio
async def test_chat_stream_failure_before_first_chunk_downgrades() -> None:
    host: Final = ScriptedHost(
        {
            "*": (
                chat_stream(error=RateLimitError("busy", "openai", "model")),
                chat_stream("default reply"),
            )
        }
    )
    router: Final = _router(host)
    response: Final = await router.acompletion(
        model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex", stream=True
    )
    assert await collect_texts(response) == "default reply"
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == ("flex", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
async def test_successful_flex_run_stops_suppressing_later_failures(stream: bool) -> None:
    error: Final = RateLimitError("late failure", "openai", "model")
    host: Final = ScriptedHost({"*": (chat_stream("first", error=error, logged=True) if stream else None,)})
    router: Final = _router(host)
    response: Final = await router.acompletion(
        model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex", stream=stream
    )
    metadata: Final = host.calls[0].kwargs.get("metadata")
    assert isinstance(metadata, Mapping)
    logged: Final = {"litellm_params": {"metadata": dict(metadata)}}
    assert not FlexTierFallback.suppress_failure(logged, error)
    if stream:
        with pytest.raises(RateLimitError):
            await collect_texts(response)


def test_successful_sync_flex_run_stops_suppressing_later_failures() -> None:
    host: Final = ScriptedHost()
    router: Final = _router(host)
    router.completion(model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex")
    assert not FlexTierFallback.suppress_failure(host.calls[0].kwargs, RateLimitError("late", "openai", "model"))


def test_pending_scope_is_shared_with_logging_metadata_copy() -> None:
    error: Final = RateLimitError("busy", "openai", "model")
    pending: Final = FlexTierFallback.pending({"service_tier": "flex"}, "metadata")
    metadata: Final = pending["metadata"]
    assert isinstance(metadata, Mapping)
    logged: Final = {"litellm_params": {"metadata": dict(metadata)}}
    assert FlexTierFallback.suppress_failure(logged, error)
    FlexTierFallback.finish(pending, "metadata")
    assert not FlexTierFallback.suppress_failure(logged, error)
