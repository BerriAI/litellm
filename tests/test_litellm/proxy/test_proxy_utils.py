import datetime as real_datetime
import os
import smtplib
import sys

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import ProxyErrorTypes
from litellm.proxy.utils import ProxyLogging

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from unittest.mock import MagicMock, patch

from litellm.proxy.utils import get_custom_url, join_paths


def test_get_custom_url(monkeypatch):
    monkeypatch.setenv("SERVER_ROOT_PATH", "/litellm")
    custom_url = get_custom_url(request_base_url="http://0.0.0.0:4000", route="ui/")
    assert custom_url == "http://0.0.0.0:4000/litellm/ui/"


def test_proxy_only_error_true_for_llm_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert proxy_logging_obj._is_proxy_only_llm_api_error(
        original_exception=Exception(),
        error_type=ProxyErrorTypes.auth_error,
        route="/v1/chat/completions",
    )


def test_proxy_only_error_true_for_info_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=ProxyErrorTypes.auth_error,
            route="/key/info",
        )
        is True
    )


def test_proxy_only_error_false_for_non_llm_non_info_route():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=ProxyErrorTypes.auth_error,
            route="/key/generate",
        )
        is False
    )


def test_proxy_only_error_false_for_other_error_type():
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    assert (
        proxy_logging_obj._is_proxy_only_llm_api_error(
            original_exception=Exception(),
            error_type=None,
            route="/v1/chat/completions",
        )
        is False
    )


@pytest.mark.asyncio
async def test_proxy_only_error_log_marks_no_upstream_llm_call():
    """A proxy-gate error (auth/rate-limit) synthesizes a ``Logging`` object and
    fires ``pre_call`` so the failure is logged — but it must tag the object with
    ``LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL`` so tracing callbacks don't fabricate
    an LLM-call span for a request that never reached a provider (root cause of the
    misplaced gen-AI span on auth failure)."""
    from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL
    from litellm.proxy._types import UserAPIKeyAuth

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    captured = {}

    def fake_pre_call(self, *args, **kwargs):
        captured["flag"] = self.model_call_details.get(
            LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL
        )

    from litellm.litellm_core_utils.litellm_logging import Logging

    orig_pre_call = Logging.pre_call
    orig_async_failure = Logging.async_failure_handler
    Logging.pre_call = fake_pre_call

    async def _noop_async_failure(self, *args, **kwargs):
        return None

    Logging.async_failure_handler = _noop_async_failure
    try:
        await proxy_logging_obj._handle_logging_proxy_only_error(
            request_data={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
            },
            user_api_key_dict=UserAPIKeyAuth(
                api_key="sk-bad", request_route="/v1/chat/completions"
            ),
            route="/v1/chat/completions",
            original_exception=Exception("bad key"),
        )
    finally:
        Logging.pre_call = orig_pre_call
        Logging.async_failure_handler = orig_async_failure

    assert captured.get("flag") is True


@pytest.mark.asyncio
async def test_proxy_only_error_log_keeps_litellm_metadata_in_litellm_params():
    """Responses API requests carry guardrail info under ``litellm_metadata``
    (not ``metadata``). It must land in litellm_params so
    ``merge_litellm_metadata`` can surface ``guardrail_information`` in the
    spend-log failure row, matching the chat completions path."""
    from litellm.proxy._types import UserAPIKeyAuth

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    captured = {}
    guardrail_info = [{"guardrail_name": "test-guard", "guardrail_status": "blocked"}]

    def fake_update_environment_variables(self, *args, **kwargs):
        captured["litellm_params"] = kwargs.get("litellm_params")
        captured["optional_params"] = kwargs.get("optional_params")

    from litellm.litellm_core_utils.litellm_logging import Logging

    orig_update_env = Logging.update_environment_variables
    orig_pre_call = Logging.pre_call
    orig_async_failure = Logging.async_failure_handler

    async def _noop_async_failure(self, *args, **kwargs):
        return None

    Logging.update_environment_variables = fake_update_environment_variables
    Logging.pre_call = lambda self, *args, **kwargs: None
    Logging.async_failure_handler = _noop_async_failure
    try:
        await proxy_logging_obj._handle_logging_proxy_only_error(
            request_data={
                "model": "gpt-4o",
                "input": "blocked prompt",
                "litellm_metadata": {
                    "standard_logging_guardrail_information": guardrail_info
                },
            },
            user_api_key_dict=UserAPIKeyAuth(
                api_key="sk-1234", request_route="/v1/responses"
            ),
            route="/v1/responses",
            original_exception=HTTPException(status_code=400, detail="blocked"),
        )
    finally:
        Logging.update_environment_variables = orig_update_env
        Logging.pre_call = orig_pre_call
        Logging.async_failure_handler = orig_async_failure

    assert (
        captured["litellm_params"]["litellm_metadata"][
            "standard_logging_guardrail_information"
        ]
        == guardrail_info
    )
    assert "litellm_metadata" not in captured["optional_params"]


def test_get_model_group_info_order():
    from litellm import Router
    from litellm.proxy.proxy_server import _get_model_group_info

    router = Router(
        model_list=[
            {
                "model_name": "openai/tts-1",
                "litellm_params": {
                    "model": "openai/tts-1",
                    "api_key": "sk-1234",
                },
            },
            {
                "model_name": "openai/gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": "sk-1234",
                },
            },
        ]
    )
    model_list = _get_model_group_info(
        llm_router=router,
        all_models_str=["openai/tts-1", "openai/gpt-3.5-turbo"],
        model_group=None,
    )

    model_groups = [m.model_group for m in model_list]
    assert model_groups == ["openai/tts-1", "openai/gpt-3.5-turbo"]


def test_join_paths_no_duplication():
    """Test that join_paths doesn't duplicate route when base_path already ends with it"""
    result = join_paths(
        base_path="http://0.0.0.0:4000/my-custom-path/", route="/my-custom-path"
    )
    assert result == "http://0.0.0.0:4000/my-custom-path"


def test_join_paths_normal_join():
    """Test normal path joining"""
    result = join_paths(base_path="http://0.0.0.0:4000", route="/api/v1")
    assert result == "http://0.0.0.0:4000/api/v1"


def test_join_paths_with_trailing_slash():
    """Test path joining with trailing slash on base_path"""
    result = join_paths(base_path="http://0.0.0.0:4000/", route="api/v1")
    assert result == "http://0.0.0.0:4000/api/v1"


def test_join_paths_empty_base():
    """Test path joining with empty base_path"""
    result = join_paths(base_path="", route="api/v1")
    assert result == "/api/v1"


def test_join_paths_empty_route():
    """Test path joining with empty route"""
    result = join_paths(base_path="http://0.0.0.0:4000", route="")
    assert result == "http://0.0.0.0:4000"


def test_join_paths_both_empty():
    """Test path joining with both empty"""
    result = join_paths(base_path="", route="")
    assert result == "/"


def test_join_paths_nested_path():
    """Test path joining with nested paths"""
    result = join_paths(base_path="http://0.0.0.0:4000/v1", route="chat/completions")
    assert result == "http://0.0.0.0:4000/v1/chat/completions"


def _patch_today(monkeypatch, year, month, day):
    class PatchedDate(real_datetime.date):
        @classmethod
        def today(cls):
            return real_datetime.date(year, month, day)

    monkeypatch.setattr("litellm.proxy.utils.date", PatchedDate)


def test_get_projected_spend_over_limit_day_one(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 1, 1)
    result = _get_projected_spend_over_limit(100.0, 1.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == 3100.0
    assert projected_exceeded_date == real_datetime.date(2026, 1, 1)


def test_get_projected_spend_over_limit_december(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 12, 15)
    result = _get_projected_spend_over_limit(100.0, 1.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == pytest.approx(214.28571428571428)
    assert projected_exceeded_date == real_datetime.date(2026, 12, 15)


def test_get_projected_spend_over_limit_includes_current_spend(monkeypatch):
    from litellm.proxy.utils import _get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 4, 11)
    result = _get_projected_spend_over_limit(100.0, 200.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == 290.0
    assert projected_exceeded_date == real_datetime.date(2026, 4, 21)


# ---------------------------------------------------------------------------
# L2: _enrich_http_exception_with_guardrail_context
# Regression coverage for case 2026-04-10-internal-bedrock-guardrail-streaming-error.
# ---------------------------------------------------------------------------


def test_enrich_http_exception_with_guardrail_context_dict_detail():
    """L2: dict-detail HTTPException is enriched with guardrail_name and mode."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "bedrock-pii-guard"
        event_hook = "post_call"

    exc = HTTPException(status_code=400, detail={"error": "Violated guardrail policy"})
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail["guardrail_name"] == "bedrock-pii-guard"
    assert exc.detail["guardrail_mode"] == "post_call"


def test_enrich_http_exception_string_detail_noop():
    """L2: string-detail HTTPException is not mutated (can't add fields to a str)."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "x"
        event_hook = "pre_call"

    exc = HTTPException(status_code=400, detail="Content blocked")
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail == "Content blocked"


def test_enrich_http_exception_setdefault_does_not_overwrite():
    """L2: a guardrail that already populates guardrail_name explicitly wins."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "inferred-name"
        event_hook = "pre_call"

    exc = HTTPException(
        status_code=400,
        detail={"error": "x", "guardrail_name": "explicit-name"},
    )
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail["guardrail_name"] == "explicit-name"


def test_enrich_http_exception_non_http_exception_noop():
    """L2: non-HTTPException is left alone and the helper does not raise."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "x"
        event_hook = "pre_call"

    exc = ValueError("not an HTTPException")
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert str(exc) == "not an HTTPException"


def test_enrich_http_exception_callback_without_guardrail_name_noop():
    """L2: callback without guardrail_name attribute leaves detail alone."""
    from litellm.proxy.utils import _enrich_http_exception_with_guardrail_context

    class StubCallback:
        pass

    exc = HTTPException(status_code=400, detail={"error": "x"})
    _enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail == {"error": "x"}


class TestPostCallFailureHookLiftsFirstApiCallStartTime:
    """post_call_failure_hook lifts first_api_call_start_time off the
    logging object into request_data (an internal top-level key) before
    the non-serialisable logging object is popped, so failure-path
    callbacks (OTel preprocessing latency) can still read it. It must
    never land in request_data["metadata"] (user request metadata,
    echoed downstream and typed Dict[str, str] in batch objects).
    """

    async def _run(self, request_data):
        from unittest.mock import AsyncMock, patch

        from litellm.proxy._types import UserAPIKeyAuth

        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging_obj.alert_types = []  # skip alerting branch
        with patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()):
            await proxy_logging_obj.post_call_failure_hook(
                request_data=request_data,
                original_exception=Exception("boom"),
                user_api_key_dict=UserAPIKeyAuth(),
            )

    @pytest.mark.asyncio
    async def test_lifts_to_top_level_and_pops_logging_obj(self):
        handoff = real_datetime.datetime(2026, 1, 1, 0, 0, 0)
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"first_api_call_start_time": handoff}
        user_meta = {}
        request_data = {
            "litellm_logging_obj": logging_obj,
            "metadata": user_meta,
        }
        await self._run(request_data)

        assert request_data["first_api_call_start_time"] == handoff
        assert "litellm_logging_obj" not in request_data
        # user metadata is never touched
        assert user_meta == {}
        assert "first_api_call_start_time" not in request_data["metadata"]

    @pytest.mark.asyncio
    async def test_no_logging_obj_is_noop(self):
        request_data = {"metadata": {}}
        await self._run(request_data)
        assert "first_api_call_start_time" not in request_data

    @pytest.mark.asyncio
    async def test_logging_obj_without_anchor_is_noop(self):
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        request_data = {"litellm_logging_obj": logging_obj}
        await self._run(request_data)
        assert "first_api_call_start_time" not in request_data
        assert "litellm_logging_obj" not in request_data


class TestPostCallFailureHookLiftsRecoveredPartialSpend:
    """A stream that broke mid-flight still billed the provider for the chunks
    already delivered. The streaming handler stashes that recovered usage and
    cost on the logging object; post_call_failure_hook must lift them onto
    request_data before the logging object is popped, so the failure-path spend
    callbacks (which run after the pop) record the real partial spend.
    """

    async def _run(self, request_data):
        from unittest.mock import AsyncMock, patch

        from litellm.proxy._types import UserAPIKeyAuth

        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging_obj.alert_types = []
        with patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()):
            await proxy_logging_obj.post_call_failure_hook(
                request_data=request_data,
                original_exception=Exception("boom"),
                user_api_key_dict=UserAPIKeyAuth(),
            )

    @pytest.mark.asyncio
    async def test_lifts_recovered_usage_and_cost(self):
        from litellm.types.utils import Usage

        recovered_usage = Usage(prompt_tokens=30, completion_tokens=1, total_tokens=31)
        logging_obj = MagicMock()
        logging_obj.model_call_details = {
            "combined_usage_object": recovered_usage,
            "response_cost": 3.5e-05,
        }
        request_data = {"litellm_logging_obj": logging_obj, "metadata": {}}
        await self._run(request_data)

        assert request_data["combined_usage_object"] is recovered_usage
        assert request_data["response_cost"] == 3.5e-05
        assert "litellm_logging_obj" not in request_data

    @pytest.mark.asyncio
    async def test_no_recovered_usage_is_noop(self):
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        request_data = {"litellm_logging_obj": logging_obj, "metadata": {}}
        await self._run(request_data)
        assert "combined_usage_object" not in request_data
        assert "response_cost" not in request_data


from typing import cast

import litellm
from litellm.proxy.utils import create_model_info_response
from litellm.types.utils import ModelInfo


def _fake_model_info(**fields: object) -> ModelInfo:
    return cast(ModelInfo, dict(fields))


def _raise_unmapped(model_id: str) -> ModelInfo:
    raise ValueError(f"This model isn't mapped yet: {model_id}")


def test_create_model_info_response_includes_max_tokens_from_lookup():
    response = create_model_info_response(
        model_id="some-model",
        provider="openai",
        llm_router=None,
        get_model_info=lambda _model: _fake_model_info(
            max_input_tokens=128000, max_output_tokens=16384
        ),
    )

    assert response["id"] == "some-model"
    assert response["object"] == "model"
    assert response["max_input_tokens"] == 128000
    assert response["max_output_tokens"] == 16384


def test_create_model_info_response_does_not_call_router_group_info():
    router = MagicMock()
    router.get_configured_token_limits.return_value = (None, None)

    response = create_model_info_response(
        model_id="some-model",
        provider="openai",
        llm_router=router,
        get_model_info=lambda _model: _fake_model_info(
            max_input_tokens=128000, max_output_tokens=16384
        ),
    )

    router.get_model_group_info.assert_not_called()
    assert response["max_input_tokens"] == 128000


def test_create_model_info_response_uses_deployment_limits_when_not_in_cost_map():
    router = MagicMock()
    router.get_configured_token_limits.return_value = (32000, 8000)

    response = create_model_info_response(
        model_id="my-custom-deployment",
        provider="openai",
        llm_router=router,
        get_model_info=_raise_unmapped,
    )

    router.get_model_group_info.assert_not_called()
    assert response["max_input_tokens"] == 32000
    assert response["max_output_tokens"] == 8000


def test_create_model_info_response_deployment_limits_override_cost_map():
    router = MagicMock()
    router.get_configured_token_limits.return_value = (200000, None)

    response = create_model_info_response(
        model_id="gpt-4o",
        provider="openai",
        llm_router=router,
        get_model_info=lambda _model: _fake_model_info(
            max_input_tokens=128000, max_output_tokens=16384
        ),
    )

    assert response["max_input_tokens"] == 200000
    assert response["max_output_tokens"] == 16384


def test_create_model_info_response_survives_malformed_configured_limits():
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "bad-limit-model",
                "litellm_params": {"model": "openai/some-unmapped-model"},
                "model_info": {"max_input_tokens": "128,000"},
            }
        ]
    )

    response = create_model_info_response(
        model_id="bad-limit-model",
        provider="openai",
        llm_router=router,
        get_model_info=_raise_unmapped,
    )

    assert response["id"] == "bad-limit-model"
    assert "max_input_tokens" not in response
    assert "max_output_tokens" not in response


@pytest.mark.parametrize("bad_value", ["128,000", "", "unlimited", [128000], {"max": 128000}, True])
def test_create_model_info_response_survives_malformed_cost_map_limits(bad_value):
    response = create_model_info_response(
        model_id="some-model",
        provider="openai",
        llm_router=None,
        get_model_info=lambda _model: _fake_model_info(
            max_input_tokens=bad_value, max_output_tokens=bad_value
        ),
    )

    assert response["id"] == "some-model"
    assert "max_input_tokens" not in response
    assert "max_output_tokens" not in response


def test_create_model_info_response_keeps_valid_cost_map_limit_beside_malformed_one():
    response = create_model_info_response(
        model_id="some-model",
        provider="openai",
        llm_router=None,
        get_model_info=lambda _model: _fake_model_info(
            max_input_tokens="128,000", max_output_tokens=16384
        ),
    )

    assert "max_input_tokens" not in response
    assert response["max_output_tokens"] == 16384


def test_create_model_info_response_survives_malformed_limits_registered_by_router():
    """A deployment's model_info is registered into litellm.model_cost verbatim, so a
    malformed configured limit reaches the listing through the real cost-map lookup and
    not just the router index. Guarding only the index path still 500s the whole listing."""
    from litellm import Router

    saved_model_cost = dict(litellm.model_cost)
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "openai/some-unmapped-model",
                    "litellm_params": {"model": "openai/some-unmapped-model"},
                    "model_info": {"max_input_tokens": "128,000"},
                }
            ]
        )

        response = create_model_info_response(
            model_id="openai/some-unmapped-model",
            provider="openai",
            llm_router=router,
        )
    finally:
        litellm.model_cost.clear()
        litellm.model_cost.update(saved_model_cost)

    assert response["id"] == "openai/some-unmapped-model"
    assert "max_input_tokens" not in response


def test_create_model_info_response_emits_integer_token_counts():
    response = create_model_info_response(
        model_id="some-model",
        provider="openai",
        llm_router=None,
        get_model_info=lambda _model: _fake_model_info(
            max_input_tokens=128000, max_output_tokens=16384
        ),
    )

    assert isinstance(response["max_input_tokens"], int)
    assert isinstance(response["max_output_tokens"], int)


def test_create_model_info_response_omits_unknown_individual_limit():
    response = create_model_info_response(
        model_id="some-embedding",
        provider="openai",
        llm_router=None,
        get_model_info=lambda _model: _fake_model_info(max_input_tokens=8191),
    )

    assert response["max_input_tokens"] == 8191
    assert "max_output_tokens" not in response


def test_create_model_info_response_omits_limits_when_lookup_raises():
    response = create_model_info_response(
        model_id="openai/*",
        provider="openai",
        llm_router=None,
        get_model_info=_raise_unmapped,
    )

    assert response["id"] == "openai/*"
    assert "max_input_tokens" not in response
    assert "max_output_tokens" not in response


def test_create_model_info_response_no_router_keeps_base_fields():
    response = create_model_info_response(
        model_id="totally-unknown-model-xyz",
        provider="openai",
        llm_router=None,
        get_model_info=_raise_unmapped,
    )

    assert response == {
        "id": "totally-unknown-model-xyz",
        "object": "model",
        "created": response["created"],
        "owned_by": "openai",
    }


def test_create_model_info_response_reads_real_cost_map():
    response = create_model_info_response(
        model_id="gpt-4o", provider="openai", llm_router=None
    )

    assert isinstance(response["max_input_tokens"], int)
    assert response["max_input_tokens"] > 0
    assert isinstance(response["max_output_tokens"], int)
    assert response["max_output_tokens"] > 0


class TestPostCallFailureHookLLMExceptionAlerting:
    """The llm_exceptions alert is for infra / LLM-API failures, not user
    errors (https://github.com/BerriAI/litellm/issues/3395). Already-normalized
    client errors must be excluded so a guardrail content-policy block never
    pages on-call. ProxyException is such an error; before LIT-3751 only
    HTTPException was excluded, so AIM blocks paged as if the LLM API failed."""

    async def _alerted(self, exc) -> bool:
        import asyncio
        from unittest.mock import AsyncMock

        from litellm.proxy._types import AlertType, UserAPIKeyAuth

        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging_obj.alert_types = [AlertType.llm_exceptions]
        alerting_handler = AsyncMock()
        with (
            patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()),
            patch.object(proxy_logging_obj, "alerting_handler", new=alerting_handler),
        ):
            await proxy_logging_obj.post_call_failure_hook(
                request_data={},
                original_exception=exc,
                user_api_key_dict=UserAPIKeyAuth(),
            )
        await asyncio.sleep(0)  # let the fire-and-forget alert task run
        return alerting_handler.called

    @pytest.mark.asyncio
    async def test_proxy_exception_does_not_alert(self):
        from litellm.proxy._types import ProxyException

        exc = ProxyException(
            message="content blocked",
            type="invalid_request_error",
            param=None,
            code=400,
            openai_code="content_policy_violation",
        )
        assert await self._alerted(exc) is False

    @pytest.mark.asyncio
    async def test_http_exception_does_not_alert(self):
        assert (
            await self._alerted(HTTPException(status_code=400, detail="blocked"))
            is False
        )

    @pytest.mark.asyncio
    async def test_genuine_llm_api_error_still_alerts(self):
        assert await self._alerted(Exception("upstream 503")) is True


class TestPostCallFailureHookProxyExceptionLogging:
    """A guardrail block raises a ProxyException; on an LLM route it must still
    drive proxy-only failure logging (_handle_logging_proxy_only_error) so the
    blocked request is recorded, exactly as the old HTTPException did. Before
    LIT-3751 the classifier only matched HTTPException, so switching AIM to
    ProxyException silently dropped the rejected prompt from failure logs."""

    async def _logged(self, exc, *, request_route) -> bool:
        from unittest.mock import AsyncMock

        from litellm.proxy._types import UserAPIKeyAuth

        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging_obj.alert_types = []
        handle_mock = AsyncMock()
        with (
            patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()),
            patch.object(
                proxy_logging_obj,
                "_handle_logging_proxy_only_error",
                new=handle_mock,
            ),
        ):
            await proxy_logging_obj.post_call_failure_hook(
                request_data={},
                original_exception=exc,
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="sk-test", request_route=request_route
                ),
            )
        return handle_mock.await_count > 0

    def _block(self):
        from litellm.proxy._types import ProxyException

        return ProxyException(
            message="content blocked",
            type="invalid_request_error",
            param=None,
            code=400,
            openai_code="content_policy_violation",
        )

    @pytest.mark.asyncio
    async def test_proxy_exception_on_llm_route_is_logged(self):
        assert (
            await self._logged(self._block(), request_route="/v1/chat/completions")
            is True
        )

    @pytest.mark.asyncio
    async def test_generic_exception_on_llm_route_is_not_logged(self):
        # A raw provider/unknown exception is logged by the LLM call path, not here.
        assert (
            await self._logged(
                Exception("upstream 503"), request_route="/v1/chat/completions"
            )
            is False
        )


class TestShouldUseSmtpSsl:
    def test_port_465_uses_ssl(self, monkeypatch):
        from litellm.proxy.utils import _should_use_smtp_ssl

        monkeypatch.delenv("SMTP_USE_SSL", raising=False)
        assert _should_use_smtp_ssl(smtp_port=465) is True

    def test_smtp_use_ssl_env_var_forces_ssl_on_any_port(self, monkeypatch):
        from litellm.proxy.utils import _should_use_smtp_ssl

        monkeypatch.setenv("SMTP_USE_SSL", "True")
        assert _should_use_smtp_ssl(smtp_port=2465) is True

    def test_port_587_uses_plain_smtp(self, monkeypatch):
        from litellm.proxy.utils import _should_use_smtp_ssl

        monkeypatch.delenv("SMTP_USE_SSL", raising=False)
        assert _should_use_smtp_ssl(smtp_port=587) is False


class TestCreateSmtpConnection:
    def test_port_465_creates_smtp_ssl_with_verified_context(self, monkeypatch):
        import ssl

        from litellm.proxy.utils import _create_smtp_connection

        monkeypatch.delenv("SMTP_USE_SSL", raising=False)
        with (
            patch("smtplib.SMTP_SSL") as mock_smtp_ssl,
            patch("smtplib.SMTP") as mock_smtp,
        ):
            result = _create_smtp_connection(
                smtp_host="mail.example.com", smtp_port=465
            )

        mock_smtp.assert_not_called()
        assert result is mock_smtp_ssl.return_value
        _, kwargs = mock_smtp_ssl.call_args
        assert kwargs["host"] == "mail.example.com"
        assert kwargs["port"] == 465
        context = kwargs["context"]
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_port_587_creates_plain_smtp(self, monkeypatch):
        from litellm.proxy.utils import _create_smtp_connection

        monkeypatch.delenv("SMTP_USE_SSL", raising=False)
        with (
            patch("smtplib.SMTP_SSL") as mock_smtp_ssl,
            patch("smtplib.SMTP") as mock_smtp,
        ):
            result = _create_smtp_connection(
                smtp_host="mail.example.com", smtp_port=587
            )

        mock_smtp_ssl.assert_not_called()
        assert result is mock_smtp.return_value
        mock_smtp.assert_called_once_with(host="mail.example.com", port=587)


class TestSendEmailStartTls:
    @pytest.mark.asyncio
    async def test_starttls_uses_verified_context(self, monkeypatch):
        import ssl

        from litellm.proxy.utils import send_email

        monkeypatch.setenv("SMTP_HOST", "mail.example.com")
        monkeypatch.setenv("SMTP_PORT", "587")
        monkeypatch.setenv("SMTP_SENDER_EMAIL", "sender@example.com")
        monkeypatch.delenv("SMTP_TLS", raising=False)
        monkeypatch.delenv("SMTP_USE_SSL", raising=False)

        mock_server = MagicMock(spec=smtplib.SMTP)
        with patch(
            "litellm.proxy.utils._create_smtp_connection"
        ) as mock_create_connection:
            mock_create_connection.return_value.__enter__.return_value = mock_server
            await send_email(
                receiver_email="receiver@example.com",
                subject="test",
                html="<p>test</p>",
            )

        _, kwargs = mock_server.starttls.call_args
        context = kwargs["context"]
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
