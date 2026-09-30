"""Pin ``ProxyLogging.post_call_failure_hook``, ``_is_proxy_only_llm_api_error``,
and ``_handle_logging_proxy_only_error``."""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.constants import PROXY_REJECTED_BEFORE_ROUTING_KEY
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import ProxyErrorTypes, UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.types.utils import CachingDetails


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ProxyLogging._callback_capabilities_cache.clear()
    yield
    ProxyLogging._callback_capabilities_cache.clear()


# ---------------------------------------------------------------------------
# _is_proxy_only_llm_api_error
# ---------------------------------------------------------------------------


def test_is_proxy_only_llm_api_truth_table(proxy_logging):
    """Pin the truth table of ``_is_proxy_only_llm_api_error`` in a single
    snapshot. Covers no-route, non-LLM route, HTTPException on LLM route,
    and auth-error short-circuit."""
    snapshot = {
        "no_route": proxy_logging._is_proxy_only_llm_api_error(original_exception=Exception(), route=None),
        "non_llm_route": proxy_logging._is_proxy_only_llm_api_error(
            original_exception=HTTPException(status_code=429, detail="rate"),
            route="/random/path",
        ),
        "http_on_llm_route": proxy_logging._is_proxy_only_llm_api_error(
            original_exception=HTTPException(status_code=429, detail="rate"),
            route="/chat/completions",
        ),
        "auth_short_circuit": proxy_logging._is_proxy_only_llm_api_error(
            original_exception=Exception("auth"),
            error_type=ProxyErrorTypes.auth_error,
            route="/chat/completions",
        ),
        "guardrail_raised_on_llm_route": proxy_logging._is_proxy_only_llm_api_error(
            original_exception=GuardrailRaisedException(guardrail_name="g", message="blocked"),
            route="/chat/completions",
        ),
    }
    assert snapshot == {
        "no_route": False,
        "non_llm_route": False,
        "http_on_llm_route": True,
        "auth_short_circuit": True,
        "guardrail_raised_on_llm_route": True,
    }


def test_is_proxy_only_llm_api_missing_exception_raises(proxy_logging):
    """Passing nothing should TypeError on the missing positional kwarg."""
    with pytest.raises(TypeError):
        proxy_logging._is_proxy_only_llm_api_error()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# post_call_failure_hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_call_failure_hook_no_callbacks_returns_none(
    proxy_logging, make_user_api_key_auth, mock_callbacks_disabled
):
    proxy_logging.alert_types = []
    request_data = {"litellm_call_id": "abc", "model": "m", "messages": []}
    out = await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=ValueError("oops"),
        user_api_key_dict=make_user_api_key_auth(),
    )
    snapshot = {
        "out_is_none": out is None,
        "litellm_logging_obj_popped": "litellm_logging_obj" not in request_data,
        "call_id_preserved": request_data["litellm_call_id"] == "abc",
        "first_api_call_start_time_present": "first_api_call_start_time" in request_data,
    }
    assert snapshot == {
        "out_is_none": True,
        "litellm_logging_obj_popped": True,
        "call_id_preserved": True,
        "first_api_call_start_time_present": False,
    }


@pytest.mark.asyncio
async def test_post_call_failure_hook_attributes_single_router_deployment(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"provider": "acme"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "internal-model", "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=403, detail="blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs["custom_llm_provider"] == "openai"
    assert kwargs["litellm_params"]["custom_llm_provider"] == "openai"
    assert kwargs["litellm_params"]["metadata"]["model_info"]["provider"] == "acme"
    assert kwargs["litellm_params"]["metadata"]["deployment"] == "openai/gpt-4.1"
    assert kwargs["litellm_params"][PROXY_REJECTED_BEFORE_ROUTING_KEY] is True
    assert kwargs["standard_logging_object"]["custom_llm_provider"] == "openai"
    assert (
        kwargs["standard_logging_object"]["model_id"] == proxy_server.llm_router.get_model_list()[0]["model_info"]["id"]
    )


@pytest.mark.asyncio
async def test_pre_routing_reject_spend_log_keeps_public_model_group(proxy_logging, make_user_api_key_auth, monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "internal-model", "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=401, detail="blocked key"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    assert recorded[0]["standard_logging_object"]["model_group"] == "internal-model"
    now: Final = datetime.now()
    payload = get_logging_payload(
        kwargs={**recorded[0], "completion_start_time": now}, response_obj=None, start_time=now, end_time=now
    )
    assert payload["model"] == "openai/gpt-4.1"
    assert payload["model_group"] == "internal-model"


@pytest.mark.asyncio
async def test_post_call_failure_hook_keeps_router_stamped_metadata_for_post_call_failures(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """A post-call guardrail block arrives after the provider handoff with the router's own
    ``model_info`` in the request metadata. The pre-routing flag must stay off so deployment
    metrics keep attributing the failure to the deployment that actually served the call."""
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"id": "routed-deployment"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    request_data = {
        "litellm_call_id": "post-call-guardrail",
        "model": "internal-model",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"model_info": {"id": "routed-deployment", "served": True}},
    }
    logging_obj, request_data = litellm.utils.function_setup(
        original_function="acompletion", rules_obj=litellm.utils.Rules(), start_time=datetime.now(), **request_data
    )
    logging_obj.model_call_details["first_api_call_start_time"] = datetime.now()
    request_data["litellm_logging_obj"] = logging_obj

    await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=GuardrailRaisedException(guardrail_name="g", message="response blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs["litellm_params"]["metadata"]["model_info"] == {"id": "routed-deployment", "served": True}
    assert PROXY_REJECTED_BEFORE_ROUTING_KEY not in kwargs["litellm_params"]
    assert kwargs["standard_logging_object"]["model_id"] == "routed-deployment"


@pytest.mark.asyncio
async def test_post_call_failure_hook_keeps_deployment_attribution_for_cache_hit_post_call_failures(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """A post-call guardrail blocks a response served from the litellm cache. No provider call was made,
    so ``first_api_call_start_time`` is unset, but the router did pick the deployment: the pre-routing
    flag must stay off so ``litellm_deployment_failure_responses`` keeps its model_id and provider labels."""
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"id": "routed-deployment"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    request_data = {
        "litellm_call_id": "cache-hit-post-call-guardrail",
        "model": "internal-model",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"model_info": {"id": "routed-deployment"}},
    }
    logging_obj, request_data = litellm.utils.function_setup(
        original_function="acompletion", rules_obj=litellm.utils.Rules(), start_time=datetime.now(), **request_data
    )
    logging_obj.caching_details = CachingDetails(cache_hit=True, cache_duration_ms=1.0)
    request_data["litellm_logging_obj"] = logging_obj

    await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=GuardrailRaisedException(guardrail_name="g", message="response blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert PROXY_REJECTED_BEFORE_ROUTING_KEY not in kwargs["litellm_params"], kwargs["litellm_params"]
    assert kwargs["standard_logging_object"]["model_id"] == "routed-deployment"
    assert kwargs["standard_logging_object"]["custom_llm_provider"] == "openai"
    assert kwargs["model"] == "internal-model"
    assert kwargs["litellm_params"]["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_post_call_failure_hook_flags_pre_routing_reject_despite_caller_model_info(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """A key allowed to override pricing keeps caller-supplied ``metadata.model_info``. A reject
    before any provider handoff must still carry the pre-routing flag so deployment metrics do
    not record an outage for a deployment the request never reached."""
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"id": "real-deployment"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={
            "model": "internal-model",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"model_info": {"id": "spoofed-deployment"}},
        },
        original_exception=HTTPException(status_code=429, detail="key over limit"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs["litellm_params"][PROXY_REJECTED_BEFORE_ROUTING_KEY] is True
    assert kwargs["litellm_params"]["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_post_call_failure_hook_attribution_does_not_count_against_the_deployment(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """The router's failure callbacks run on this path too. A proxy-side reject must not
    bump the deployment's failure or rpm counters, or a key hitting its own limit
    could cool down the only deployment for everyone."""
    from litellm.proxy import proxy_server

    router = litellm.Router(
        model_list=[
            {
                "model_name": "internal-model",
                "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test", "rpm": 100},
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    proxy_logging.alert_types = []
    deployment_id = router.get_model_list()[0]["model_info"]["id"]

    for status in (403, 429):
        await proxy_logging.post_call_failure_hook(
            request_data={"model": "internal-model", "messages": [{"role": "user", "content": "hi"}]},
            original_exception=HTTPException(status_code=status, detail="blocked"),
            user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
            route="/chat/completions",
        )
    pending = asyncio.all_tasks() - {asyncio.current_task()}
    await asyncio.gather(*pending, return_exceptions=True)

    deployment_keys = [key for key in router.cache.in_memory_cache.cache_dict if deployment_id in key]
    assert deployment_keys == [], f"proxy reject was counted against the deployment: {deployment_keys}"


@pytest.mark.asyncio
async def test_post_call_failure_hook_attributes_the_keys_team_deployment_over_the_global_group(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """A team key requesting its team public model name must be attributed to the team's
    deployment, not to a global group that happens to share the public name."""
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "shared-name",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"id": "global-deployment"},
                },
                {
                    "model_name": "shared-name_test-team_deadbeef",
                    "litellm_params": {"model": "anthropic/claude-sonnet-4-5", "api_key": "sk-test"},
                    "model_info": {
                        "id": "team-deployment",
                        "team_id": "test-team",
                        "team_public_model_name": "shared-name",
                    },
                },
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "shared-name", "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=429, detail="rate limited"),
        user_api_key_dict=make_user_api_key_auth(team_id="test-team", request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs["custom_llm_provider"] == "anthropic"
    assert kwargs["litellm_params"]["metadata"]["deployment"] == "anthropic/claude-sonnet-4-5"
    assert kwargs["standard_logging_object"]["model_id"] == "team-deployment"


@pytest.mark.asyncio
async def test_post_call_failure_hook_omits_provider_for_mixed_router_deployments(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                },
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "anthropic/claude-sonnet-4-5", "api_key": "sk-test"},
                },
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "internal-model", "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=403, detail="blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs.get("custom_llm_provider") is None
    assert "model_info" not in (kwargs["litellm_params"].get("metadata") or {})


@pytest.mark.asyncio
async def test_post_call_failure_hook_omits_provider_when_a_deployment_does_not_resolve(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """One deployment resolves to openai and its sibling resolves to nothing: the group
    is not known to be single-provider, so no provider is stamped on the failure."""
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    router = MagicMock()
    router.get_model_list.return_value = [
        {"model_name": "internal-model", "litellm_params": {"model": "openai/gpt-4.1"}},
        {"model_name": "internal-model", "litellm_params": {"model": "unmapped-model-with-no-provider"}},
    ]
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "internal-model", "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=403, detail="blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs.get("custom_llm_provider") is None
    assert kwargs["litellm_params"].get("custom_llm_provider") is None


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_attributes_with_read_only_metadata(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """With a logging object already on the request, its metadata is taken as given;
    a read-only mapping there must not crash the stamp, and the failure handler
    still receives the provider attribution."""
    from litellm.proxy import proxy_server

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"provider": "acme"},
                }
            ]
        ),
    )
    logging_obj = MagicMock()
    logging_obj.call_type = "acompletion"
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock()

    await proxy_logging._handle_logging_proxy_only_error(
        request_data={
            "litellm_logging_obj": logging_obj,
            "model": "internal-model",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": MappingProxyType({"user_api_key_alias": "frozen"}),
        },
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
        original_exception=HTTPException(status_code=403, detail="blocked"),
    )

    assert logging_obj.async_failure_handler.called
    update_kwargs = logging_obj.update_environment_variables.call_args.kwargs
    assert update_kwargs["custom_llm_provider"] == "openai"
    assert update_kwargs["litellm_params"]["custom_llm_provider"] == "openai"
    assert update_kwargs["litellm_params"]["metadata"] == {"user_api_key_alias": "frozen"}


@pytest.mark.asyncio
async def test_post_call_failure_hook_fires_without_router_attribution(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "different-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "internal-model", "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=403, detail="blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs.get("custom_llm_provider") is None
    assert "model_info" not in (kwargs["litellm_params"].get("metadata") or {})


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [123, ["internal-model"], {"name": "internal-model"}, None])
async def test_post_call_failure_hook_fires_for_non_string_model(
    proxy_logging, make_user_api_key_auth, monkeypatch, model: object
):
    """A body whose ``model`` is not a string is rejected by the proxy before routing; its
    failure callback must still fire, unattributed, instead of a TypeError escaping the hook."""
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=400, detail="'model' must be a string."),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs.get("custom_llm_provider") is None
    assert "model_info" not in (kwargs["litellm_params"].get("metadata") or {})


@pytest.mark.asyncio
async def test_post_call_failure_hook_callback_returns_http_exception(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    transformed = HTTPException(status_code=418, detail="teapot")

    class _Cb(CustomLogger):
        async def async_post_call_failure_hook(self, **kwargs):  # type: ignore[override]
            return transformed

    monkeypatch.setattr(litellm, "callbacks", [_Cb()])
    proxy_logging.alert_types = []
    out = await proxy_logging.post_call_failure_hook(
        request_data={"litellm_call_id": "abc"},
        original_exception=ValueError("oops"),
        user_api_key_dict=make_user_api_key_auth(),
    )
    assert out is transformed


@pytest.mark.asyncio
async def test_post_call_failure_hook_callback_raises_http_exception_first_wins(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    err = HTTPException(status_code=418, detail="raised teapot")

    class _Cb(CustomLogger):
        async def async_post_call_failure_hook(self, **kwargs):  # type: ignore[override]
            raise err

    monkeypatch.setattr(litellm, "callbacks", [_Cb()])
    proxy_logging.alert_types = []
    out = await proxy_logging.post_call_failure_hook(
        request_data={"litellm_call_id": "abc"},
        original_exception=ValueError("oops"),
        user_api_key_dict=make_user_api_key_auth(),
    )
    assert out is err


@pytest.mark.asyncio
async def test_post_call_failure_hook_non_http_exception_in_callback_swallowed(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    class _Cb(CustomLogger):
        async def async_post_call_failure_hook(self, **kwargs):  # type: ignore[override]
            raise RuntimeError("non-http inside cb")

    monkeypatch.setattr(litellm, "callbacks", [_Cb()])
    proxy_logging.alert_types = []
    out = await proxy_logging.post_call_failure_hook(
        request_data={"litellm_call_id": "abc"},
        original_exception=ValueError("oops"),
        user_api_key_dict=make_user_api_key_auth(),
    )
    assert out is None


@pytest.mark.asyncio
@pytest.mark.parametrize("logging_value", (None, "caller-controlled", {"baseline_cache_context": "untrusted"}))  # mutable-ok: emulate an untrusted JSON request field
async def test_terminal_baseline_cleanup_ignores_missing_or_untrusted_logging(
    proxy_logging: ProxyLogging, monkeypatch: pytest.MonkeyPatch, logging_value: object
) -> None:
    monkeypatch.setattr(litellm, "callbacks", ())
    proxy_logging.alert_types = []  # mutable-ok: disable optional alert sinks for this boundary test  # rebind-ok: isolate the fixture-owned alert configuration
    request_data: Final = {"litellm_call_id": "untrusted-logging", "litellm_logging_obj": logging_value}  # mutable-ok: the production failure owner removes internal fields in place
    result: Final = await proxy_logging.post_call_failure_hook(  # pyright: ignore[reportUnknownMemberType]  # exercise the existing proxy terminal owner with its legacy request dictionary contract
        request_data=request_data,
        original_exception=ValueError("original provider failure"),
        user_api_key_dict=UserAPIKeyAuth(request_route="/v1/messages"),
    )
    assert result is None
    assert "litellm_logging_obj" not in request_data


# ---------------------------------------------------------------------------
# _handle_logging_proxy_only_error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_uses_existing_logging_obj(proxy_logging, make_user_api_key_auth):
    logging_obj = MagicMock()
    logging_obj.call_type = "acompletion"
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock()

    request_data = {
        "litellm_logging_obj": logging_obj,
        "messages": [{"role": "user", "content": "x"}],
        "model": "m",
        "metadata": {},
    }
    await proxy_logging._handle_logging_proxy_only_error(
        request_data=request_data,
        user_api_key_dict=make_user_api_key_auth(),
        route="/chat/completions",
        original_exception=HTTPException(status_code=429, detail="rate"),
    )
    from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL

    snapshot = {
        "input_logged": "messages" in logging_obj.model_call_details,
        "call_type_normalized": logging_obj.call_type,
        "marker_present": logging_obj.model_call_details.get(LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL) is True,
        "async_failure_called": logging_obj.async_failure_handler.called,
    }
    assert snapshot == {
        "input_logged": True,
        "call_type_normalized": "acompletion",
        "marker_present": True,
        "async_failure_called": True,
    }


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_skips_for_pass_through(proxy_logging, make_user_api_key_auth):
    from litellm.types.utils import CallTypes

    logging_obj = MagicMock()
    logging_obj.call_type = CallTypes.pass_through.value
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock()
    logging_obj.pre_call = MagicMock()
    request_data = {
        "litellm_logging_obj": logging_obj,
        "messages": [{"role": "user"}],
        "model": "m",
    }
    await proxy_logging._handle_logging_proxy_only_error(
        request_data=request_data,
        user_api_key_dict=make_user_api_key_auth(),
        route="/chat/completions",
        original_exception=HTTPException(status_code=429, detail="rate"),
    )
    logging_obj.pre_call.assert_not_called()
    logging_obj.async_failure_handler.assert_not_called()


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_no_logging_obj_creates_one(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    fake_logging_obj = MagicMock()
    fake_logging_obj.call_type = "acompletion"
    fake_logging_obj.model_call_details = {}
    fake_logging_obj.async_failure_handler = AsyncMock()

    def fake_function_setup(**kwargs):
        return fake_logging_obj, {}

    monkeypatch.setattr(litellm.utils, "function_setup", fake_function_setup)
    request_data = {"messages": [{"role": "user"}], "model": "m"}
    await proxy_logging._handle_logging_proxy_only_error(
        request_data=request_data,
        user_api_key_dict=make_user_api_key_auth(),
        route="/chat/completions",
        original_exception=HTTPException(status_code=429, detail="rate"),
    )
    assert "litellm_call_id" in request_data
    fake_logging_obj.async_failure_handler.assert_called_once()


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_propagates_async_failure_raises(proxy_logging, make_user_api_key_auth):
    logging_obj = MagicMock()
    logging_obj.call_type = "acompletion"
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock(side_effect=RuntimeError("boom"))
    request_data = {
        "litellm_logging_obj": logging_obj,
        "messages": [{"role": "user"}],
        "model": "m",
    }
    with pytest.raises(RuntimeError):
        await proxy_logging._handle_logging_proxy_only_error(
            request_data=request_data,
            user_api_key_dict=make_user_api_key_auth(),
            route="/chat/completions",
            original_exception=Exception("x"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route, request_data, expected_call_type",
    [
        ("/v1/chat/completions", {}, "acompletion"),
        ("/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, "acompletion"),
        ("/v1/messages", {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, "anthropic_messages"),
        ("/v1/responses", {"model": "m", "input": "hi"}, "aresponses"),
        ("/v1/embeddings", {"model": "m", "input": ["hi"]}, "aembedding"),
        ("/model/info", {}, "/model/info"),
    ],
)
async def test_post_call_failure_hook_lifts_route_call_type_for_gate_rejections(
    proxy_logging, make_user_api_key_auth, route, request_data, expected_call_type
):
    """Regression for LIT-5884: the matched route, not the body shape, sets the
    spend-log call_type for requests rejected before dispatch."""
    proxy_logging.alert_types = []
    await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=Exception("Authentication Error, No api key passed in."),
        user_api_key_dict=make_user_api_key_auth(request_route=route),
        error_type=ProxyErrorTypes.auth_error,
        route=route,
    )
    assert request_data["call_type"] == expected_call_type
    assert "start_time" in request_data


@pytest.mark.asyncio
async def test_post_call_failure_hook_falls_back_to_body_shape_without_a_route(proxy_logging, make_user_api_key_auth):
    proxy_logging.alert_types = []
    request_data = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=Exception("Authentication Error, No api key passed in."),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        error_type=ProxyErrorTypes.auth_error,
    )
    assert request_data["call_type"] == "acompletion"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/v1/files", "/files/file-abc", "/v1/containers"])
async def test_post_call_failure_hook_keeps_the_route_for_multi_operation_routes(
    proxy_logging, make_user_api_key_auth, route
):
    """Routes shared by several operations (POST create vs GET list) cannot be attributed without the
    method, so a rejected request there is filed under its route, not under whichever operation the
    mapping lists first."""
    proxy_logging.alert_types = []
    request_data: dict = {}
    await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=Exception("Authentication Error, No api key passed in."),
        user_api_key_dict=make_user_api_key_auth(request_route=route),
        error_type=ProxyErrorTypes.auth_error,
        route=route,
    )
    assert request_data["call_type"] == route


@pytest.mark.asyncio
async def test_post_call_failure_hook_guardrail_block_fires_failure_callback(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """A ``GuardrailRaisedException`` on an LLM route must reach the logging
    object's ``async_failure_handler`` so custom loggers see a ``failure``
    status - without this, guardrail blocks produce only
    ``post_call_failure_hook`` and no failure logging event."""
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    recorded: list[object] = []

    class _StatusRecorder(CustomLogger):
        async def async_log_failure_event(
            self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            standard_logging_object = kwargs.get("standard_logging_object")
            recorded.append(standard_logging_object.get("status") if isinstance(standard_logging_object, dict) else None)

    monkeypatch.setattr(litellm, "_async_failure_callback", [_StatusRecorder()])
    logging_obj = LiteLLMLoggingObj(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="test_guardrail_block_failure_cb",
        function_id="test_guardrail_block_failure_cb",
    )
    request_data = {
        "litellm_logging_obj": logging_obj,
        "litellm_call_id": "test_guardrail_block_failure_cb",
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {},
    }
    proxy_logging.alert_types = []
    await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=GuardrailRaisedException(guardrail_name="g", message="blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert recorded == ["failure"]
