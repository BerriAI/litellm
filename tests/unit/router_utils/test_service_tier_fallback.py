from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm import Router
from litellm.cost_calculator import completion_cost
from litellm.exceptions import AuthenticationError, BadRequestError, RateLimitError, Timeout, UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.hidden_params import get_hidden_params
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.llms.compaction import CompactionProtocol
from litellm.router_api import ChangeSet
from litellm.router_config import build_router_config
from litellm.router_strategy.complexity_router.context_compaction import compaction_executor
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
@pytest.mark.asyncio
async def test_trigger_only_accepts_retryable_flex_errors(error: Exception, trigger: bool) -> None:
    host: Final = ScriptedHost({"*": (error, None)})
    router: Final = _router(host)
    if trigger:
        await router.acompletion(model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex")
    else:
        with pytest.raises(type(error)):
            await router.acompletion(
                model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex"
            )
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == (("flex", None) if trigger else ("flex",))


@pytest.mark.parametrize(
    "enabled,tier,expected",
    ((True, "flex", True), (False, "flex", False), (True, "default", False), (True, None, False)),
)
@pytest.mark.asyncio
async def test_fallback_requires_enabled_and_flex(enabled: bool, tier: str | None, expected: bool) -> None:
    host: Final = ScriptedHost({"*": (RateLimitError("busy", "openai", "model"), None)})
    router: Final = _router(host, service_tier_fallbacks=enabled)
    if expected:
        await router.acompletion(model="group", messages=[{"role": "user", "content": "hello"}], service_tier=tier)
    else:
        with pytest.raises(RateLimitError):
            await router.acompletion(model="group", messages=[{"role": "user", "content": "hello"}], service_tier=tier)
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == ((tier, None) if expected else (tier,))


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
        model="group",
        messages=[{"role": "user", "content": "hello"}],
        service_tier="flex",
        metadata={"custom": "provider-metadata"},
        litellm_metadata={"audit": "kept"},
    )
    await router.acompletion(model="group", messages=[{"role": "user", "content": "again"}])
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == ("flex", None, None)
    assert (get_hidden_params(response) or {})["additional_headers"]["x-litellm-service-tier"] == "default"
    metadata: Final = host.calls[1].kwargs.get("litellm_metadata")
    assert isinstance(metadata, Mapping)
    assert metadata["service_tier_fallback"] == {"from": "flex", "to": "default"}
    assert metadata["audit"] == "kept"
    assert StandardLoggingPayloadSetup.get_standard_logging_metadata(metadata)["service_tier_fallback"] == {
        "from": "flex",
        "to": "default",
    }
    assert "service_tier_fallback" not in (as_str_mapping(host.calls[1].kwargs.get("metadata")) or {})


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
async def test_downgraded_fallback_hop_keeps_walk_counts() -> None:
    host: Final = ScriptedHost(
        {"*": (BadRequestError("bad", "model", "openai"), RateLimitError("busy", "openai", "model"), None)}
    )
    router: Final = _router(
        host,
        model_list=[deployment("group", "only"), deployment("backup", "backup")],
        fallbacks=[{"group": ["backup"]}],
    )
    await router.acompletion(model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex")
    assert tuple(call.deployment_id for call in host.calls) == ("only", "backup", "backup")
    metadata: Final = as_str_mapping(host.calls[-1].kwargs.get("metadata"))
    assert metadata is not None
    assert {key: metadata.get(key) for key in ("attempted_fallbacks", "attempted_retries", "original_model_group")} == {
        "attempted_fallbacks": 1,
        "attempted_retries": 0,
        "original_model_group": "group",
    }
    assert (as_str_mapping(host.calls[-1].kwargs.get("litellm_metadata")) or {})["service_tier_fallback"] == {
        "from": "flex",
        "to": "default",
    }


@pytest.mark.asyncio
async def test_a_copied_default_deployment_does_not_restore_flex() -> None:
    host: Final = ScriptedHost({"*": (RateLimitError("busy", "openai", "model"), None)})
    router: Final = _router(host)
    router.apply(
        ChangeSet(
            assign={
                "default_deployment": {
                    **deployment("*", "default", service_tier="flex", timeout=33, timeout_by_service_tier={"flex": 2}),
                    "model_info": {"id": 7},
                }
            }
        )
    )
    await router.acompletion(model="unknown", messages=[{"role": "user", "content": "hello"}], service_tier="flex")
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == ("flex", None)
    assert tuple(call.kwargs.get("timeout") for call in host.calls) == (2.0, 33)


@pytest.mark.asyncio
async def test_disabled_fallback_keeps_dictionary_target_metadata_replacement() -> None:
    host: Final = ScriptedHost({"*": (BadRequestError("bad", "model", "openai"), None)})
    router: Final = _router(
        host,
        service_tier_fallbacks=False,
        model_list=[deployment("group", "only"), deployment("backup", "backup")],
        fallbacks=[{"group": [{"model": "backup", "litellm_metadata": {"hop": "target"}}]}],
    )
    await router.acompletion(
        model="group",
        messages=[{"role": "user", "content": "hello"}],
        litellm_metadata={"original": "replaced"},
    )
    assert tuple(call.deployment_id for call in host.calls) == ("only", "backup")
    assert host.calls[-1].kwargs.get("litellm_metadata") == {"hop": "target"}


@pytest.mark.asyncio
async def test_downgraded_retry_preserves_compacted_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "flex-compactor-fixture",
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "max_input_tokens": 32000,
            "max_output_tokens": 512,
            "supports_anthropic_compaction": True,
        },
    )
    host: Final = ScriptedHost({"*": (RateLimitError("busy", "openai", "model"), None)})
    router: Final = _router(
        host,
        model_list=[
            deployment(
                "group",
                "auto",
                model="auto_router/complexity_router",
                complexity_router_config={
                    "tiers": {"SIMPLE": "small", "MEDIUM": "small", "COMPLEX": "small", "REASONING": "small"},
                    "enable_context_window_escalation": False,
                    "max_tokens_from_tier_model": False,
                    "context_compaction": {"model": "compactor", "max_tokens": 512},
                },
            ),
            deployment("small", "only", model_info={"max_input_tokens": 256, "max_output_tokens": 64}),
            deployment("compactor", "compactor", model="anthropic/flex-compactor-fixture"),
        ],
    )
    messages: Final = [
        {"role": "user", "content": "old history " * 300},
        {"role": "assistant", "content": "recorded"},
        {"role": "user", "content": "latest question"},
    ]

    async def summarize(
        protocol: CompactionProtocol, payload: Mapping[str, object], parent_model: str | None = None
    ) -> Mapping[str, object]:
        assert protocol == "chat" and parent_model == "group"
        assert "old history" in str(payload["messages"])
        return {
            "choices": [
                {
                    "message": {
                        "provider_specific_fields": {
                            "compaction_blocks": [
                                {
                                    "type": "compaction",
                                    "content": "short history summary",
                                    "signature": "fixture-signature",
                                },
                            ]
                        }
                    }
                }
            ]
        }

    token: Final = compaction_executor.set(summarize)
    try:
        await router.acompletion(model="group", messages=messages, max_tokens=32, service_tier="flex")
    finally:
        compaction_executor.reset(token)
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == ("flex", None)
    compacted: Final = host.calls[-1].kwargs.get("messages")
    assert compacted == host.calls[0].kwargs.get("messages")
    assert compacted != messages
    assert "short history summary" in str(compacted) and "old history" not in str(compacted)


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


@pytest.mark.asyncio
async def test_setting_can_be_updated() -> None:
    host: Final = ScriptedHost({"*": (RateLimitError("busy", "openai", "model"), None)})
    router: Final = _router(host, service_tier_fallbacks=False)
    router.update_settings(service_tier_fallbacks=True)
    assert router.get_settings()["service_tier_fallbacks"] is True
    await router.acompletion(model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex")
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == ("flex", None)


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
async def test_successful_flex_run_does_not_retry_later_stream_failure(stream: bool) -> None:
    error: Final = RateLimitError("late failure", "openai", "model")
    host: Final = ScriptedHost({"*": (chat_stream("first", error=error, logged=True) if stream else None,)})
    router: Final = _router(host)
    response: Final = await router.acompletion(
        model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex", stream=stream
    )
    if stream:
        with pytest.raises(RateLimitError):
            await collect_texts(response)
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == ("flex",)


def test_successful_sync_flex_run_is_not_downgraded() -> None:
    host: Final = ScriptedHost()
    router: Final = _router(host)
    router.completion(model="group", messages=[{"role": "user", "content": "hello"}], service_tier="flex")
    assert tuple(call.kwargs.get("service_tier") for call in host.calls) == ("flex",)
