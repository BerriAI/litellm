import asyncio
import datetime as real_datetime
import json
import smtplib
import traceback
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.exceptions import InternalServerError
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.bug_report import ISSUE_URL_BASE
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import ProxyErrorTypes, UserAPIKeyAuth
from litellm.proxy.utils import (
    PrismaClient,
    ProxyLogging,
    _failure_fields_to_lift,
    handle_exception_on_proxy,
)
from litellm.router_utils.fallback_event_handlers import carry_superseded_attempt_usage
from litellm.types.guardrails import GuardrailEventHooks


from unittest.mock import AsyncMock, MagicMock, patch

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
        captured["flag"] = self.model_call_details.get(LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL)

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
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-bad", request_route="/v1/chat/completions"),
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
                "litellm_metadata": {"standard_logging_guardrail_information": guardrail_info},
            },
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-9876", request_route="/v1/responses"),
            route="/v1/responses",
            original_exception=HTTPException(status_code=400, detail="blocked"),
        )
    finally:
        Logging.update_environment_variables = orig_update_env
        Logging.pre_call = orig_pre_call
        Logging.async_failure_handler = orig_async_failure

    assert captured["litellm_params"]["litellm_metadata"]["standard_logging_guardrail_information"] == guardrail_info
    assert "litellm_metadata" not in captured["optional_params"]


@pytest.mark.asyncio
async def test_proxy_only_error_log_keeps_the_request_litellm_call_id(monkeypatch: pytest.MonkeyPatch):
    """LIT-7836: a route that already stamped the caller's litellm_call_id must
    keep it when the failure is a proxy-only error, so the spend-log row and the
    error line share one id instead of a fresh uuid minted here."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    call_id: Final = "caller-supplied-7836"
    captured: dict[str, object] = {}

    def fake_pre_call(self, *args, **kwargs):
        captured["litellm_call_id"] = self.litellm_call_id

    async def _noop_async_failure(self, *args, **kwargs):
        return None

    monkeypatch.setattr(Logging, "pre_call", fake_pre_call)
    monkeypatch.setattr(Logging, "async_failure_handler", _noop_async_failure)
    request_data: Final[dict[str, object]] = {"model": "gpt-4o", "input": "hi", "litellm_call_id": call_id}

    await ProxyLogging(user_api_key_cache=DualCache())._handle_logging_proxy_only_error(
        request_data=request_data,
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-bad", request_route="/v1/moderations"),
        route="/v1/moderations",
        original_exception=Exception("bad key"),
    )

    assert request_data["litellm_call_id"] == call_id
    assert captured["litellm_call_id"] == call_id


def test_get_model_group_info_order():
    from litellm import Router
    from litellm.proxy.proxy_server import get_model_group_info

    router = Router(
        model_list=[
            {
                "model_name": "openai/tts-1",
                "litellm_params": {
                    "model": "openai/tts-1",
                    "api_key": "sk-9876",
                },
            },
            {
                "model_name": "openai/gpt-3.5-turbo",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": "sk-9876",
                },
            },
        ]
    )
    model_list = get_model_group_info(
        llm_router=router,
        all_models_str=["openai/tts-1", "openai/gpt-3.5-turbo"],
        model_group=None,
    )

    model_groups = [m.model_group for m in model_list]
    assert model_groups == ["openai/tts-1", "openai/gpt-3.5-turbo"]


def test_join_paths_no_duplication():
    """Test that join_paths doesn't duplicate route when base_path already ends with it"""
    result = join_paths(base_path="http://0.0.0.0:4000/my-custom-path/", route="/my-custom-path")
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
    from litellm.proxy.utils import get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 1, 1)
    result = get_projected_spend_over_limit(100.0, 1.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == 3100.0
    assert projected_exceeded_date == real_datetime.date(2026, 1, 1)


def test_get_projected_spend_over_limit_december(monkeypatch):
    from litellm.proxy.utils import get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 12, 15)
    result = get_projected_spend_over_limit(100.0, 1.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == pytest.approx(214.28571428571428)
    assert projected_exceeded_date == real_datetime.date(2026, 12, 15)


def test_get_projected_spend_over_limit_includes_current_spend(monkeypatch):
    from litellm.proxy.utils import get_projected_spend_over_limit

    _patch_today(monkeypatch, 2026, 4, 11)
    result = get_projected_spend_over_limit(100.0, 200.0)

    assert result is not None
    projected_spend, projected_exceeded_date = result
    assert projected_spend == 290.0
    assert projected_exceeded_date == real_datetime.date(2026, 4, 21)


# ---------------------------------------------------------------------------
# L2: enrich_http_exception_with_guardrail_context
# Regression coverage for case 2026-04-10-internal-bedrock-guardrail-streaming-error.
# ---------------------------------------------------------------------------


def test_enrich_http_exception_with_guardrail_context_dict_detail():
    """L2: dict-detail HTTPException is enriched with guardrail_name and mode."""
    from litellm.proxy.guardrails.exception_utils import enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "bedrock-pii-guard"
        event_hook = "post_call"

    exc = HTTPException(status_code=400, detail={"error": "Violated guardrail policy"})
    enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail["guardrail_name"] == "bedrock-pii-guard"
    assert exc.detail["guardrail_mode"] == "post_call"


def test_enrich_http_exception_string_detail_noop():
    """L2: string-detail HTTPException is not mutated (can't add fields to a str)."""
    from litellm.proxy.guardrails.exception_utils import enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "x"
        event_hook = "pre_call"

    exc = HTTPException(status_code=400, detail="Content blocked")
    enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail == "Content blocked"


def test_enrich_http_exception_setdefault_does_not_overwrite():
    """L2: a guardrail that already populates guardrail_name explicitly wins."""
    from litellm.proxy.guardrails.exception_utils import enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "inferred-name"
        event_hook = "pre_call"

    exc = HTTPException(
        status_code=400,
        detail={"error": "x", "guardrail_name": "explicit-name"},
    )
    enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert exc.detail["guardrail_name"] == "explicit-name"


def test_enrich_http_exception_non_http_exception_noop():
    """L2: non-HTTPException is left alone and the helper does not raise."""
    from litellm.proxy.guardrails.exception_utils import enrich_http_exception_with_guardrail_context

    class StubCallback:
        guardrail_name = "x"
        event_hook = "pre_call"

    exc = ValueError("not an HTTPException")
    enrich_http_exception_with_guardrail_context(exc, StubCallback())
    assert str(exc) == "not an HTTPException"


def test_enrich_http_exception_callback_without_guardrail_name_noop():
    """L2: callback without guardrail_name attribute leaves detail alone."""
    from litellm.proxy.guardrails.exception_utils import enrich_http_exception_with_guardrail_context

    class StubCallback:
        pass

    exc = HTTPException(status_code=400, detail={"error": "x"})
    enrich_http_exception_with_guardrail_context(exc, StubCallback())
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


class TestPostCallFailureHookDoesNotInventAProviderHandoff:
    """A failure the proxy raised before any provider call (here an auth error) is logged through
    _handle_logging_proxy_only_error, which fires pre_call for the callbacks. That must not stamp a
    handoff instant onto the request, or every downstream reader (preprocessing latency, the
    dispatched check behind the failure-row estimate) sees a provider call that never happened."""

    @pytest.mark.asyncio
    async def test_proxy_only_failure_lifts_no_first_api_call_start_time(self):
        from datetime import datetime
        from unittest.mock import AsyncMock, patch

        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.proxy._types import ProxyErrorTypes, UserAPIKeyAuth

        logging_obj = Logging(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            call_type="acompletion",
            start_time=datetime.now(),
            litellm_call_id="pre-handoff-auth-failure",
            function_id="pre-handoff",
        )
        request_data: Final[dict[str, object]] = {"litellm_logging_obj": logging_obj, "model": "gpt-4.1", "metadata": {}}
        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging_obj.alert_types = []
        with (
            patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()),
            patch.object(proxy_logging_obj, "_dispatch_proxy_only_failure_handlers", new=AsyncMock()),
        ):
            await proxy_logging_obj.post_call_failure_hook(
                request_data=request_data,
                original_exception=Exception("Authentication Error, invalid key"),
                user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
                error_type=ProxyErrorTypes.auth_error,
            )

        assert request_data.get("first_api_call_start_time") is None
        assert "api_call_start_time" not in logging_obj.model_call_details


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
    async def test_recovered_usage_without_cost_clobbers_client_cost_with_zero(self):
        from litellm.types.utils import Usage

        recovered_usage = Usage(prompt_tokens=30, completion_tokens=1, total_tokens=31)
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"combined_usage_object": recovered_usage}
        request_data = {
            "litellm_logging_obj": logging_obj,
            "response_cost": 999.0,
            "metadata": {},
        }
        await self._run(request_data)

        assert request_data["combined_usage_object"] is recovered_usage
        assert request_data["response_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_no_recovered_usage_is_noop(self):
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        request_data = {"litellm_logging_obj": logging_obj, "metadata": {}}
        await self._run(request_data)
        assert "combined_usage_object" not in request_data
        assert "response_cost" not in request_data


class TestPostCallFailureHookLiftsStandardLoggingObject:
    """Failure callbacks read standard_logging_object from request_data, but
    post_call_failure_hook pops litellm_logging_obj before they run. The hook
    must lift the logging obj's standard_logging_object onto request_data so
    failed-request spend logs keep deployment attribution (LIT-5795).
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
    async def test_lifts_standard_logging_object(self):
        sl_object = {"model_id": "mid-123", "model_group": "group-x"}
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"standard_logging_object": sl_object}
        request_data = {"litellm_logging_obj": logging_obj, "metadata": {}}
        await self._run(request_data)
        assert request_data["standard_logging_object"] is sl_object
        assert "litellm_logging_obj" not in request_data

    @pytest.mark.asyncio
    async def test_logging_obj_value_overwrites_preexisting_key(self):
        authoritative = {"model_id": "from-logging-obj"}
        logging_obj = MagicMock()
        logging_obj.model_call_details = {"standard_logging_object": authoritative}
        request_data = {
            "litellm_logging_obj": logging_obj,
            "standard_logging_object": {"model_id": "client-injected"},
            "metadata": {},
        }
        await self._run(request_data)
        assert request_data["standard_logging_object"] is authoritative

    @pytest.mark.asyncio
    async def test_client_supplied_key_is_stripped_when_logging_obj_supplies_none(self):
        spoofed = {"model_id": "client-injected"}
        request_data = {"standard_logging_object": spoofed, "metadata": {}}
        await self._run(request_data)
        assert "standard_logging_object" not in request_data

        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        request_data_with_obj = {
            "litellm_logging_obj": logging_obj,
            "standard_logging_object": spoofed,
            "metadata": {},
        }
        await self._run(request_data_with_obj)
        assert "standard_logging_object" not in request_data_with_obj

    @pytest.mark.asyncio
    async def test_pass_through_failure_never_relifts_client_supplied_key(self):
        from datetime import datetime
        from unittest.mock import AsyncMock, patch

        from fastapi import HTTPException

        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.proxy._types import UserAPIKeyAuth

        logging_obj = Logging(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            call_type="pass_through_endpoint",
            start_time=datetime.now(),
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        )
        request_data = {
            "litellm_logging_obj": logging_obj,
            "standard_logging_object": {"model_id": "client-injected"},
            "metadata": {},
        }
        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging_obj.alert_types = []
        with patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()):
            await proxy_logging_obj.post_call_failure_hook(
                request_data=request_data,
                original_exception=HTTPException(status_code=401, detail="unauthorized"),
                user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
            )
        assert "standard_logging_object" not in request_data
        assert "standard_logging_object" not in logging_obj.model_call_details

    @pytest.mark.asyncio
    async def test_no_standard_logging_object_is_noop(self):
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        request_data = {"litellm_logging_obj": logging_obj, "metadata": {}}
        await self._run(request_data)
        assert "standard_logging_object" not in request_data


class TestPostCallFailureHookLiftsCallTypeAndStartTime:
    """A guardrail-blocked MCP tool call fails before any LLM call. The failure
    spend row is built from request_data after ``litellm_logging_obj`` is popped,
    so ``call_type`` and the request ``start_time`` must be lifted off the logging
    object first, or the Logs page shows the row as an LLM call with a blank call
    type and a 0s duration (LIT-7453).
    """

    @pytest.mark.asyncio
    async def test_failed_mcp_tool_call_spend_row_keeps_call_type_model_and_duration(self):
        import traceback
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
        from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

        request_start = real_datetime.datetime.now() - real_datetime.timedelta(seconds=2)
        logging_obj = Logging(
            model="MCP: deepwiki-ask_question",
            messages=[],
            stream=False,
            call_type="call_mcp_tool",
            start_time=request_start,
            litellm_call_id="call-1",
            function_id="fn-1",
        )
        logging_obj.update_environment_variables(
            model="MCP: deepwiki-ask_question",
            user="",
            optional_params={},
            litellm_params={"metadata": {"user_api_key_hash": "hashed"}},
        )
        blocked = Exception("Content blocked: keyword 'confidential' detected")
        logging_obj.failure_handler(blocked, traceback.format_exc(), request_start, real_datetime.datetime.now())
        request_data = {
            "name": "deepwiki-ask_question",
            "arguments": {"question": "confidential"},
            "litellm_logging_obj": logging_obj,
        }
        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging_obj.alert_types = []
        spend_writer = SimpleNamespace(update_database=AsyncMock())
        original_callbacks = list(litellm.callbacks)
        litellm.callbacks = [ProxyDBLogger(spend_writer=lambda: spend_writer)]
        try:
            with patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()):
                await proxy_logging_obj.post_call_failure_hook(
                    request_data=request_data,
                    original_exception=blocked,
                    user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
                )
        finally:
            litellm.callbacks = original_callbacks
            ProxyLogging._callback_capabilities_cache.clear()

        db_call = spend_writer.update_database.call_args.kwargs
        payload = get_logging_payload(
            kwargs=db_call["kwargs"],
            response_obj=db_call["completion_response"],
            start_time=db_call["start_time"],
            end_time=db_call["end_time"],
        )
        assert payload["call_type"] == "call_mcp_tool"
        assert payload["model"] == "MCP: deepwiki-ask_question"
        assert payload["endTime"] - payload["startTime"] >= real_datetime.timedelta(seconds=2)


class TestPostCallFailureHookEstimatesDispatchedInputTokens:
    """A non-stream request that failed after dispatch (timeout, provider
    error) consumed provider-billed input tokens but recovered no usage.
    post_call_failure_hook must estimate the input side onto request_data so
    the spend log's failure row records what was sent instead of zero
    (LIT-5690). On a timeout that estimate is priced at the model's input
    rate since the provider bills the prompt it accepted (LIT-145); other
    post-dispatch failures keep zero spend.
    """

    async def _run(
        self, request_data, original_exception: Exception | None = None, request_route=None, dispatched_call_type=None
    ):
        from unittest.mock import AsyncMock, patch

        from litellm.proxy._types import UserAPIKeyAuth

        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging_obj.alert_types = []
        with patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()):
            await proxy_logging_obj.post_call_failure_hook(
                request_data=request_data,
                original_exception=original_exception or Exception("boom"),
                user_api_key_dict=UserAPIKeyAuth(request_route=request_route),
                dispatched_call_type=dispatched_call_type,
            )

    def _timeout(self, model: str, cause: Exception) -> Exception:
        try:
            try:
                raise cause
            except Exception as inner:
                raise litellm.Timeout("Request timed out.", model=model, llm_provider="openai") from inner
        except litellm.Timeout as timed_out:
            return timed_out

    def _logging_obj(self, model_call_details):
        logging_obj = MagicMock()
        logging_obj.model_call_details = model_call_details
        return logging_obj

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "request_route, dispatched_call_type",
        [("/v1/responses", None), ("/cursor/chat/completions", "aresponses")],
    )
    async def test_a_failed_responses_request_counts_its_input_from_the_newest_compaction_item(
        self, request_route, dispatched_call_type
    ):
        from datetime import datetime

        async def prompt_tokens(request_input):
            request_data = {
                "litellm_logging_obj": self._logging_obj(
                    {
                        "first_api_call_start_time": datetime.now(),
                        "model": FAILURE_USAGE_MODEL,
                        "messages": request_input,
                        "call_type": "aresponses",
                    }
                ),
                "metadata": {},
            }
            await self._run(request_data, request_route=request_route, dispatched_call_type=dispatched_call_type)
            return request_data["combined_usage_object"].prompt_tokens

        resent = await prompt_tokens(
            [COMPACTION_INSTRUCTIONS, *COMPACTED_HISTORY, COMPACTION_ITEM, TURN_AFTER_COMPACTION]
        )

        assert resent == await prompt_tokens([COMPACTION_INSTRUCTIONS, COMPACTION_ITEM, TURN_AFTER_COMPACTION])
        assert resent < await prompt_tokens([COMPACTION_INSTRUCTIONS, *COMPACTED_HISTORY, TURN_AFTER_COMPACTION])

    @pytest.mark.asyncio
    async def test_timed_out_request_estimates_input_tokens_priced_at_input_rate(self):
        from datetime import datetime

        from litellm.types.utils import Usage

        request_data = {
            "litellm_logging_obj": self._logging_obj(
                {
                    "first_api_call_start_time": datetime.now(),
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "count these input tokens please"}],
                    "call_type": "acompletion",
                }
            ),
            "metadata": {},
            "response_cost": 123.0,
        }
        await self._run(
            request_data, original_exception=self._timeout("gpt-3.5-turbo", httpx.ReadTimeout("no bytes in time"))
        )

        estimated = request_data["combined_usage_object"]
        assert isinstance(estimated, Usage)
        assert estimated.prompt_tokens > 0
        assert estimated.completion_tokens == 0
        assert estimated.total_tokens == estimated.prompt_tokens
        expected_input_cost, _ = litellm.cost_per_token(model="gpt-3.5-turbo", prompt_tokens=estimated.prompt_tokens)
        assert expected_input_cost > 0
        assert request_data["response_cost"] == pytest.approx(expected_input_cost)

    @pytest.mark.asyncio
    async def test_client_disconnect_after_dispatch_is_priced_at_input_rate(self):
        from datetime import datetime

        from fastapi import HTTPException

        from litellm.constants import LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED, LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL
        from litellm.types.utils import Usage

        request_data = {
            "litellm_logging_obj": self._logging_obj(
                {
                    "first_api_call_start_time": datetime.now(),
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "count these input tokens please"}],
                    "call_type": "acompletion",
                    LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL: True,
                }
            ),
            "metadata": {},
        }
        await self._run(
            request_data,
            original_exception=HTTPException(
                status_code=LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED, detail="Client disconnected the request"
            ),
        )

        estimated = request_data["combined_usage_object"]
        assert isinstance(estimated, Usage)
        expected_input_cost, _ = litellm.cost_per_token(model="gpt-3.5-turbo", prompt_tokens=estimated.prompt_tokens)
        assert expected_input_cost > 0
        assert request_data["response_cost"] == pytest.approx(expected_input_cost)

    @pytest.mark.asyncio
    async def test_upstream_499_error_is_a_plain_failure_not_a_priced_disconnect(self):
        """A provider answering HTTP 499 surfaces as litellm.APIError with status_code 499. Only the
        proxy's own disconnect HTTPException means the caller hung up mid-generation, so the upstream
        error keeps the estimated usage at zero cost like any other post-dispatch provider error."""
        from datetime import datetime

        from litellm.constants import LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED
        from litellm.types.utils import Usage

        request_data = {
            "litellm_logging_obj": self._logging_obj(
                {
                    "first_api_call_start_time": datetime(2026, 1, 1, 12, 0, 0),
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "count these input tokens please"}],
                    "call_type": "acompletion",
                }
            ),
            "metadata": {},
        }
        await self._run(
            request_data,
            original_exception=litellm.APIError(
                status_code=LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED,
                message="client closed request",
                llm_provider="openai",
                model="gpt-3.5-turbo",
            ),
        )

        estimated = request_data["combined_usage_object"]
        assert isinstance(estimated, Usage)
        assert estimated.prompt_tokens > 0
        assert request_data["response_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_client_disconnect_after_the_call_already_completed_is_not_priced_again(self):
        """The LLM task finished and claimed its success log (which carries the real usage and cost)
        while a during_call guardrail task was still pending, then the caller hung up. The 499 raised
        for that disconnect must not price the prompt a second time."""
        from datetime import datetime

        from fastapi import HTTPException

        from litellm.constants import LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED
        from litellm.litellm_core_utils.litellm_logging import Logging

        logging_obj = Logging(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "count these input tokens please"}],
            stream=False,
            call_type="acompletion",
            start_time=datetime(2026, 1, 1, 12, 0, 0),
            litellm_call_id="completed-before-disconnect",
            function_id="fn",
        )
        logging_obj.model_call_details.update(
            {
                "first_api_call_start_time": datetime(2026, 1, 1, 12, 0, 1),
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "count these input tokens please"}],
                "call_type": "acompletion",
            }
        )
        assert logging_obj.claim_async_success_log()
        logging_obj.mark_async_success_log_enqueued()
        request_data = {"litellm_logging_obj": logging_obj, "metadata": {}}
        await self._run(
            request_data,
            original_exception=HTTPException(
                status_code=LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED, detail="Client disconnected the request"
            ),
        )

        assert request_data.get("response_cost", 0.0) == 0.0

    @pytest.mark.asyncio
    async def test_client_disconnect_with_a_claimed_but_deferred_success_log_is_still_priced(self):
        """With a post_call guardrail registered the success log is claimed but parked in the deferred slot,
        and a 499 raised before the flush means it never runs. Nothing else bills that request, so the
        failure row keeps the estimated input cost."""
        from datetime import datetime

        from fastapi import HTTPException

        from litellm.constants import LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED
        from litellm.litellm_core_utils.litellm_logging import Logging

        logging_obj = Logging(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "count these input tokens please"}],
            stream=False,
            call_type="acompletion",
            start_time=datetime(2026, 1, 1, 12, 0, 0),
            litellm_call_id="deferred-success-never-flushed",
            function_id="fn",
        )
        logging_obj.model_call_details.update(
            {
                "first_api_call_start_time": datetime(2026, 1, 1, 12, 0, 1),
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "count these input tokens please"}],
                "call_type": "acompletion",
            }
        )
        logging_obj.defer_async_logging = True
        assert logging_obj.claim_async_success_log()
        logging_obj.enqueue_deferred_logging = lambda: None
        request_data = {"litellm_logging_obj": logging_obj, "metadata": {}}
        await self._run(
            request_data,
            original_exception=HTTPException(
                status_code=LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED, detail="Client disconnected the request"
            ),
        )

        assert request_data["response_cost"] > 0.0

    @pytest.mark.asyncio
    async def test_client_disconnect_before_provider_handoff_is_not_priced(self):
        """A caller that hung up while the request was still in pre-call hooks never reached the
        provider, so there is no handoff instant on the logging object and the 499 row is not priced."""
        from datetime import datetime
        from unittest.mock import AsyncMock, patch

        from fastapi import HTTPException

        from litellm.constants import LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.proxy._types import UserAPIKeyAuth

        logging_obj = Logging(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "count these input tokens please"}],
            stream=False,
            call_type="acompletion",
            start_time=datetime.now(),
            litellm_call_id="lit145-pre-handoff",
            function_id="lit145",
        )
        assert logging_obj.model_call_details.get("first_api_call_start_time") is None
        request_data = {"litellm_logging_obj": logging_obj, "model": "gpt-3.5-turbo", "metadata": {}}

        proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
        proxy_logging_obj.alert_types = []
        with (
            patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()),
            patch.object(proxy_logging_obj, "_dispatch_proxy_only_failure_handlers", new=AsyncMock()),
        ):
            await proxy_logging_obj.post_call_failure_hook(
                request_data=request_data,
                original_exception=HTTPException(
                    status_code=LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED, detail="Client disconnected the request"
                ),
                user_api_key_dict=UserAPIKeyAuth(request_route="/v1/chat/completions"),
            )

        assert request_data.get("first_api_call_start_time") is None
        assert "combined_usage_object" not in request_data
        assert request_data.get("response_cost", 0.0) == 0.0

    @pytest.mark.asyncio
    async def test_connect_timeout_never_reached_the_provider_stays_zero_cost(self):
        from datetime import datetime

        from litellm.types.utils import Usage

        request_data = {
            "litellm_logging_obj": self._logging_obj(
                {
                    "first_api_call_start_time": datetime.now(),
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "count these input tokens please"}],
                    "call_type": "acompletion",
                }
            ),
            "metadata": {},
        }
        await self._run(
            request_data, original_exception=self._timeout("gpt-3.5-turbo", httpx.ConnectTimeout("no route"))
        )

        assert isinstance(request_data["combined_usage_object"], Usage)
        assert request_data["response_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_dispatched_non_timeout_failure_estimates_input_tokens_with_zero_cost(self):
        from datetime import datetime

        from litellm.types.utils import Usage

        request_data = {
            "litellm_logging_obj": self._logging_obj(
                {
                    "first_api_call_start_time": datetime.now(),
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "count these input tokens please"}],
                    "call_type": "acompletion",
                }
            ),
            "metadata": {},
            "response_cost": 123.0,
        }
        await self._run(request_data)

        estimated = request_data["combined_usage_object"]
        assert isinstance(estimated, Usage)
        assert estimated.prompt_tokens > 0
        assert request_data["response_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_failure_before_dispatch_stays_zero(self):
        request_data = {
            "litellm_logging_obj": self._logging_obj(
                {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "never dispatched"}],
                }
            ),
            "metadata": {},
        }
        await self._run(request_data)

        assert "combined_usage_object" not in request_data
        assert "response_cost" not in request_data

    @pytest.mark.asyncio
    async def test_proxy_only_error_never_dispatched_stays_zero(self):
        from datetime import datetime

        from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL

        request_data = {
            "litellm_logging_obj": self._logging_obj(
                {
                    "first_api_call_start_time": datetime.now(),
                    "model": "no-such-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL: True,
                }
            ),
            "metadata": {},
        }
        await self._run(request_data)

        assert "combined_usage_object" not in request_data
        assert "response_cost" not in request_data

    @pytest.mark.asyncio
    async def test_recovered_partial_usage_wins_over_estimate(self):
        from datetime import datetime

        from litellm.types.utils import Usage

        recovered_usage = Usage(prompt_tokens=30, completion_tokens=7, total_tokens=37)
        request_data = {
            "litellm_logging_obj": self._logging_obj(
                {
                    "first_api_call_start_time": datetime.now(),
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "mid-stream failure"}],
                    "call_type": "acompletion",
                    "combined_usage_object": recovered_usage,
                    "response_cost": 3.5e-05,
                }
            ),
            "metadata": {},
        }
        await self._run(request_data)

        assert request_data["combined_usage_object"] is recovered_usage
        assert request_data["response_cost"] == 3.5e-05

    @pytest.mark.asyncio
    async def test_dispatched_failure_with_text_completion_prompt(self):
        from datetime import datetime

        from litellm.types.utils import Usage

        request_data = {
            "litellm_logging_obj": self._logging_obj(
                {
                    "first_api_call_start_time": datetime.now(),
                    "model": "gpt-3.5-turbo",
                    "messages": "a plain text-completion prompt string",
                    "call_type": "atext_completion",
                }
            ),
            "metadata": {},
        }
        await self._run(request_data)

        estimated = request_data["combined_usage_object"]
        assert isinstance(estimated, Usage)
        assert estimated.prompt_tokens > 0
        assert estimated.completion_tokens == 0

    def _dispatched_request_data(self, messages, optional_params, call_type="acompletion"):
        from datetime import datetime

        return {
            "litellm_logging_obj": self._logging_obj(
                {
                    "first_api_call_start_time": datetime.now(),
                    "model": "gpt-3.5-turbo",
                    "messages": messages,
                    "optional_params": optional_params,
                    "call_type": call_type,
                }
            ),
            "metadata": {},
        }

    @pytest.mark.asyncio
    async def test_image_message_estimated_without_fetching_image(self):
        import litellm as litellm_module
        from litellm.types.utils import Usage

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://127.0.0.1:1/unreachable.png", "detail": "high"},
                    },
                ],
            }
        ]
        request_data = self._dispatched_request_data(messages, {})
        await self._run(request_data)

        estimated = request_data["combined_usage_object"]
        assert isinstance(estimated, Usage)
        expected = litellm_module.token_counter(
            model="gpt-3.5-turbo", messages=messages, use_default_image_token_count=True
        )
        assert estimated.prompt_tokens == expected
        assert estimated.prompt_tokens > 0

    @pytest.mark.asyncio
    async def test_embedding_string_list_input_counted_in_estimate(self):
        import litellm as litellm_module
        from litellm.types.utils import Usage

        embedding_input = ["first embedding text", "second embedding text"]
        request_data = self._dispatched_request_data(embedding_input, {}, call_type="aembedding")
        await self._run(request_data)

        estimated = request_data["combined_usage_object"]
        assert isinstance(estimated, Usage)
        expected = litellm_module.token_counter(model="gpt-3.5-turbo", text="".join(embedding_input))
        assert estimated.prompt_tokens == expected

    @pytest.mark.asyncio
    async def test_transcription_checksum_not_estimated(self):
        request_data = self._dispatched_request_data("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6", {}, call_type="atranscription")
        await self._run(request_data)

        assert "combined_usage_object" not in request_data
        assert "response_cost" not in request_data

    @pytest.mark.asyncio
    async def test_anthropic_system_prompt_counted_in_estimate(self):
        import litellm as litellm_module
        from litellm.types.utils import Usage

        system_prompt = "You are a verbose historian who narrates every fact in exhaustive detail."
        messages = [{"role": "user", "content": "write a short essay"}]
        request_data = self._dispatched_request_data(messages, {"system": system_prompt, "max_tokens": 100})
        await self._run(request_data)

        estimated = request_data["combined_usage_object"]
        assert isinstance(estimated, Usage)
        expected = litellm_module.token_counter(
            model="gpt-3.5-turbo", messages=messages
        ) + litellm_module.token_counter(model="gpt-3.5-turbo", text=system_prompt)
        assert estimated.prompt_tokens == expected

    @pytest.mark.asyncio
    async def test_anthropic_system_text_blocks_counted_in_estimate(self):
        import litellm as litellm_module
        from litellm.types.utils import Usage

        system_blocks = [
            {"type": "text", "text": "part one of the system prompt. "},
            {"type": "text", "text": "part two of the system prompt."},
        ]
        messages = [{"role": "user", "content": "write a short essay"}]
        request_data = self._dispatched_request_data(messages, {"system": system_blocks})
        await self._run(request_data)

        estimated = request_data["combined_usage_object"]
        assert isinstance(estimated, Usage)
        expected = litellm_module.token_counter(
            model="gpt-3.5-turbo", messages=messages
        ) + litellm_module.token_counter(
            model="gpt-3.5-turbo", text="part one of the system prompt. part two of the system prompt."
        )
        assert estimated.prompt_tokens == expected

    @pytest.mark.asyncio
    async def test_responses_instructions_counted_in_estimate(self):
        import litellm as litellm_module
        from litellm.types.utils import Usage

        instructions = "Answer every question as a meticulous archivist."
        request_data = self._dispatched_request_data("summarize the archive", {"instructions": instructions})
        await self._run(request_data)

        estimated = request_data["combined_usage_object"]
        assert isinstance(estimated, Usage)
        expected = litellm_module.token_counter(
            model="gpt-3.5-turbo", text="summarize the archive"
        ) + litellm_module.token_counter(model="gpt-3.5-turbo", text=instructions)
        assert estimated.prompt_tokens == expected

    @pytest.mark.asyncio
    async def test_request_body_system_counted_when_optional_params_empty(self):
        import litellm as litellm_module
        from litellm.types.utils import Usage

        system_prompt = "You are a meticulous cartographer who labels every landmark."
        messages = [{"role": "user", "content": "draw me a map"}]
        request_data = {
            **self._dispatched_request_data(messages, {}, call_type="aanthropic_messages"),
            "system": system_prompt,
        }
        await self._run(request_data)

        estimated = request_data["combined_usage_object"]
        assert isinstance(estimated, Usage)
        expected = litellm_module.token_counter(
            model="gpt-3.5-turbo", messages=messages
        ) + litellm_module.token_counter(model="gpt-3.5-turbo", text=system_prompt)
        assert estimated.prompt_tokens == expected

    @pytest.mark.asyncio
    async def test_optional_params_system_wins_over_request_body_system(self):
        import litellm as litellm_module
        from litellm.types.utils import Usage

        dispatched_system = "short dispatched system prompt"
        messages = [{"role": "user", "content": "hello"}]
        request_data = {
            **self._dispatched_request_data(messages, {"system": dispatched_system}),
            "system": "a much longer request body system prompt that must not be double counted here",
        }
        await self._run(request_data)

        estimated = request_data["combined_usage_object"]
        assert isinstance(estimated, Usage)
        expected = litellm_module.token_counter(
            model="gpt-3.5-turbo", messages=messages
        ) + litellm_module.token_counter(model="gpt-3.5-turbo", text=dispatched_system)
        assert estimated.prompt_tokens == expected


from typing import cast

import litellm
from litellm.proxy.utils import create_model_info_response
from litellm.types.router import DeploymentModelListingInfo
from litellm.types.utils import ModelInfo, ModelResponse, Usage


def _fake_model_info(**fields: object) -> ModelInfo:
    return cast(ModelInfo, dict(fields))


def _raise_unmapped(model_id: str) -> ModelInfo:
    raise ValueError(f"This model isn't mapped yet: {model_id}")


def test_create_model_info_response_includes_max_tokens_from_lookup():
    response = create_model_info_response(
        model_id="some-model",
        provider="openai",
        llm_router=None,
        get_model_info=lambda _model: _fake_model_info(max_input_tokens=128000, max_output_tokens=16384),
    )

    assert response["id"] == "some-model"
    assert response["object"] == "model"
    assert response["max_input_tokens"] == 128000
    assert response["max_output_tokens"] == 16384


def test_create_model_info_response_does_not_call_router_group_info():
    router = MagicMock()
    router.get_model_listing_info.return_value = None

    response = create_model_info_response(
        model_id="some-model",
        provider="openai",
        llm_router=router,
        get_model_info=lambda _model: _fake_model_info(max_input_tokens=128000, max_output_tokens=16384),
    )

    router.get_model_group_info.assert_not_called()
    assert response["max_input_tokens"] == 128000


def test_create_model_info_response_uses_deployment_limits_when_not_in_cost_map():
    router = MagicMock()
    router.get_model_listing_info.return_value = DeploymentModelListingInfo(
        cost_map_keys=("my-custom-deployment",), max_input_tokens=32000, max_output_tokens=8000
    )

    response = create_model_info_response(
        model_id="my-custom-deployment",
        provider="openai",
        llm_router=router,
        get_model_info=_raise_unmapped,
    )

    router.get_model_group_info.assert_not_called()
    assert response["max_input_tokens"] == 32000
    assert response["max_output_tokens"] == 8000


def test_create_model_info_response_uses_deployment_mode_for_auto_router():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "claude-sonnet",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "test-key"},
            },
            {
                "model_name": "claude-auto",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {
                            "SIMPLE": "claude-sonnet",
                            "MEDIUM": "claude-sonnet",
                            "COMPLEX": "claude-sonnet",
                        }
                    },
                    "complexity_router_default_model": "claude-sonnet",
                },
                "model_info": {
                    "mode": "chat",
                    "max_input_tokens": 1_000_000,
                    "max_output_tokens": 128_000,
                },
            },
        ]
    )

    response = create_model_info_response(
        model_id="claude-auto",
        provider="openai",
        llm_router=router,
        get_model_info=_raise_unmapped,
    )

    assert response["mode"] == "chat"
    assert response["max_input_tokens"] == 1_000_000
    assert response["max_output_tokens"] == 128_000


def test_create_model_info_response_deployment_limits_override_cost_map():
    router = MagicMock()
    router.get_model_listing_info.return_value = DeploymentModelListingInfo(
        cost_map_keys=("gpt-4o",), max_input_tokens=200000, max_output_tokens=None
    )

    response = create_model_info_response(
        model_id="gpt-4o",
        provider="openai",
        llm_router=router,
        get_model_info=lambda _model: _fake_model_info(max_input_tokens=128000, max_output_tokens=16384),
    )

    assert response["max_input_tokens"] == 200000
    assert response["max_output_tokens"] == 16384


def test_create_model_info_response_reports_widest_window_in_a_mixed_group():
    """A group mixing models advertises the widest window, not whichever is listed first."""
    limits = {
        "small-model": _fake_model_info(max_input_tokens=200000, max_output_tokens=4096, mode="chat"),
        "large-model": _fake_model_info(max_input_tokens=1000000, max_output_tokens=128000, mode="chat"),
    }

    for keys in (("small-model", "large-model"), ("large-model", "small-model")):
        router = MagicMock()
        router.get_model_listing_info.return_value = DeploymentModelListingInfo(
            cost_map_keys=keys, max_input_tokens=None, max_output_tokens=None
        )

        response = create_model_info_response(
            model_id="house-claude",
            provider="openai",
            llm_router=router,
            get_model_info=lambda model: limits[model],
        )

        assert response["max_input_tokens"] == 1000000, keys
        assert response["max_output_tokens"] == 128000, keys


def test_create_model_info_response_resolves_alias_once_per_listing():
    """The alias is the same for every deployment in the group, so it is looked up once."""
    seen: list[str] = []

    def _tracking_get_model_info(model: str) -> ModelInfo:
        seen.append(model)
        return _fake_model_info(max_input_tokens=128000)

    router = MagicMock()
    router.get_model_listing_info.return_value = DeploymentModelListingInfo(
        cost_map_keys=("model-a", "model-b"), max_input_tokens=None, max_output_tokens=None
    )

    create_model_info_response(
        model_id="house-model",
        provider="openai",
        llm_router=router,
        get_model_info=_tracking_get_model_info,
    )

    assert seen.count("house-model") == 1
    assert sorted(seen) == ["house-model", "model-a", "model-b"]


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
        get_model_info=lambda _model: _fake_model_info(max_input_tokens=bad_value, max_output_tokens=bad_value),
    )

    assert response["id"] == "some-model"
    assert "max_input_tokens" not in response
    assert "max_output_tokens" not in response


def test_create_model_info_response_keeps_valid_cost_map_limit_beside_malformed_one():
    response = create_model_info_response(
        model_id="some-model",
        provider="openai",
        llm_router=None,
        get_model_info=lambda _model: _fake_model_info(max_input_tokens="128,000", max_output_tokens=16384),
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
        get_model_info=lambda _model: _fake_model_info(max_input_tokens=128000, max_output_tokens=16384),
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
    response = create_model_info_response(model_id="gpt-4o", provider="openai", llm_router=None)

    assert isinstance(response["max_input_tokens"], int)
    assert response["max_input_tokens"] > 0
    assert isinstance(response["max_output_tokens"], int)
    assert response["max_output_tokens"] > 0


def test_create_model_info_response_includes_mode_from_lookup():
    response = create_model_info_response(
        model_id="text-embedding-3-small",
        provider="openai",
        llm_router=None,
        get_model_info=lambda _model: _fake_model_info(mode="embedding"),
    )

    assert response["mode"] == "embedding"


def test_create_model_info_response_omits_mode_when_lookup_raises():
    response = create_model_info_response(
        model_id="my-custom-deployment",
        provider="openai",
        llm_router=None,
        get_model_info=_raise_unmapped,
    )

    assert "mode" not in response


def test_create_model_info_response_omits_non_string_mode():
    response = create_model_info_response(
        model_id="some-model",
        provider="openai",
        llm_router=None,
        get_model_info=lambda _model: _fake_model_info(mode=None),
    )

    assert "mode" not in response


class TestPostCallFailureHookLLMExceptionAlerting:
    """The llm_exceptions alert is for infra / LLM-API failures, not user
    errors (https://github.com/BerriAI/litellm/issues/3395). Already-normalized
    client errors must be excluded so a guardrail content-policy block never
    pages on-call. 5xx proxy errors still alert."""

    async def _alerted(self, exc: Exception) -> AsyncMock:
        import asyncio

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
        return alerting_handler

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
        assert (await self._alerted(exc)).called is False

    @pytest.mark.asyncio
    async def test_http_exception_does_not_alert(self):
        assert (await self._alerted(HTTPException(status_code=400, detail="blocked"))).called is False

    @pytest.mark.asyncio
    async def test_genuine_llm_api_error_still_alerts(self):
        assert (await self._alerted(Exception("upstream 503"))).called is True

    @pytest.mark.asyncio
    async def test_http_exception_5xx_alerts(self):
        alerting_handler = await self._alerted(
            HTTPException(
                status_code=502,
                detail={
                    "error": "Headroom compression service returned an error",
                    "status_code": 503,
                    "guardrail_name": "headroom-compression-global",
                },
            )
        )
        assert alerting_handler.called is True
        assert "headroom-compression-global" in alerting_handler.call_args.kwargs["message"]

    @pytest.mark.asyncio
    async def test_proxy_exception_5xx_alerts(self):
        from litellm.proxy._types import ProxyException

        alerting_handler = await self._alerted(
            ProxyException(
                message="guardrail backend down",
                type="internal_server_error",
                param=None,
                code=503,
            )
        )
        assert alerting_handler.called is True

    @pytest.mark.asyncio
    async def test_http_exception_429_does_not_alert(self):
        alerting_handler = await self._alerted(HTTPException(status_code=429, detail="rate limited"))
        assert alerting_handler.called is False


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
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", request_route=request_route),
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
        assert await self._logged(self._block(), request_route="/v1/chat/completions") is True

    @pytest.mark.asyncio
    async def test_generic_exception_on_llm_route_is_not_logged(self):
        # A raw provider/unknown exception is logged by the LLM call path, not here.
        assert await self._logged(Exception("upstream 503"), request_route="/v1/chat/completions") is False


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
            result = _create_smtp_connection(smtp_host="mail.example.com", smtp_port=465, timeout=30.0)

        mock_smtp.assert_not_called()
        assert result is mock_smtp_ssl.return_value
        _, kwargs = mock_smtp_ssl.call_args
        assert kwargs["host"] == "mail.example.com"
        assert kwargs["port"] == 465
        assert kwargs["timeout"] == 30.0
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
            result = _create_smtp_connection(smtp_host="mail.example.com", smtp_port=587, timeout=30.0)

        mock_smtp_ssl.assert_not_called()
        assert result is mock_smtp.return_value
        mock_smtp.assert_called_once_with(host="mail.example.com", port=587, timeout=30.0)


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
        with patch("litellm.proxy.utils._create_smtp_connection") as mock_create_connection:
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


class _RecordingMCPGuardrail(CustomGuardrail):
    """Unified guardrail that masks every text it is handed."""

    def __init__(self, event_hook, masked_text="<MASKED>", raises=None):
        super().__init__(guardrail_name="mcp-output-guardrail", event_hook=event_hook, default_on=True)
        self.masked_text = masked_text
        self.raises = raises
        self.call_count = 0
        self.last_input_type = None

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        self.call_count += 1
        self.last_input_type = input_type
        if self.raises is not None:
            raise self.raises
        return {"texts": [self.masked_text for _ in inputs.get("texts", [])]}


class _NativeMCPGuardrail(CustomGuardrail):
    """Guardrail that only implements the MCP logging hook (cisco-style)."""

    def __init__(self):
        super().__init__(
            guardrail_name="native-mcp-guardrail",
            event_hook=GuardrailEventHooks.post_mcp_call,
            default_on=True,
        )
        self.considered_count = 0

    def should_run_guardrail(self, data, event_type):
        self.considered_count += 1
        return super().should_run_guardrail(data=data, event_type=event_type)

    async def async_post_mcp_tool_call_hook(self, kwargs, response_obj, start_time, end_time):
        return None


@pytest.fixture
def restore_callbacks():
    """Restore the process-wide callback state post_mcp_call_hook reads.

    ProxyLogging caches callback capabilities keyed on id()s of litellm.callbacks,
    so a restored-but-different list can collide with a stale entry after GC and
    leak a has_guardrail verdict into unrelated tests in the same worker.
    """
    original = list(litellm.callbacks)
    yield
    litellm.callbacks = original
    ProxyLogging._callback_capabilities_cache.clear()


@pytest.mark.asyncio
async def test_post_mcp_call_hook_masks_tool_result(restore_callbacks):
    """A post_mcp_call guardrail must see the tool result text and mask it in the returned result."""
    from mcp.types import CallToolResult, TextContent

    guardrail = _RecordingMCPGuardrail(event_hook=GuardrailEventHooks.post_mcp_call)
    litellm.callbacks = [guardrail]
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    result = CallToolResult(content=[TextContent(type="text", text="jane@example.com")], isError=False)

    returned = await proxy_logging_obj.post_mcp_call_hook(
        response=result,
        request_data={"mcp_tool_name": "echo"},
        user_api_key_dict=None,
    )

    assert guardrail.call_count == 1
    assert guardrail.last_input_type == "response"
    assert [item.text for item in returned.content] == ["<MASKED>"]


@pytest.mark.asyncio
async def test_post_mcp_call_hook_skips_guardrail_configured_for_other_hooks(restore_callbacks):
    """A guardrail not configured for post_mcp_call must not scan MCP tool results."""
    from mcp.types import CallToolResult, TextContent

    guardrail = _RecordingMCPGuardrail(event_hook=GuardrailEventHooks.post_call)
    litellm.callbacks = [guardrail]
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    result = CallToolResult(content=[TextContent(type="text", text="jane@example.com")], isError=False)

    returned = await proxy_logging_obj.post_mcp_call_hook(
        response=result,
        request_data={"mcp_tool_name": "echo"},
        user_api_key_dict=None,
    )

    assert guardrail.call_count == 0
    assert [item.text for item in returned.content] == ["jane@example.com"]


@pytest.mark.asyncio
async def test_post_mcp_call_hook_skips_guardrail_without_apply_guardrail(restore_callbacks):
    """Guardrails that implement async_post_mcp_tool_call_hook are dispatched by the
    logging object, so this hook must not run them a second time."""
    from mcp.types import CallToolResult, TextContent

    guardrail = _NativeMCPGuardrail()
    litellm.callbacks = [guardrail]
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    result = CallToolResult(content=[TextContent(type="text", text="jane@example.com")], isError=False)

    returned = await proxy_logging_obj.post_mcp_call_hook(
        response=result,
        request_data={"mcp_tool_name": "echo"},
        user_api_key_dict=None,
    )

    assert guardrail.considered_count == 0
    assert [item.text for item in returned.content] == ["jane@example.com"]


@pytest.mark.asyncio
async def test_post_mcp_call_hook_propagates_guardrail_block(restore_callbacks):
    """A guardrail rejecting the tool result must raise out of the hook."""
    from mcp.types import CallToolResult, TextContent

    from litellm.exceptions import BlockedPiiEntityError

    guardrail = _RecordingMCPGuardrail(
        event_hook=GuardrailEventHooks.post_mcp_call,
        raises=BlockedPiiEntityError(entity_type="EMAIL_ADDRESS", guardrail_name="mcp-output-guardrail"),
    )
    litellm.callbacks = [guardrail]
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    result = CallToolResult(content=[TextContent(type="text", text="jane@example.com")], isError=False)

    with pytest.raises(BlockedPiiEntityError):
        await proxy_logging_obj.post_mcp_call_hook(
            response=result,
            request_data={"mcp_tool_name": "echo"},
            user_api_key_dict=None,
        )


@pytest.mark.asyncio
async def test_prisma_health_check_failure_names_itself_at_operator_visible_level(caplog):
    """A failing DB health check has to name the check that failed, at a level
    operators actually run at.

    Reporting it as ``disconnect()`` sends anyone grepping the logs to the wrong
    function and reads as "the check never ran", and reporting it only at debug
    level hides a database fault behind a flag nobody enables in production."""
    import logging
    from functools import partial
    from unittest.mock import AsyncMock

    from litellm.proxy.utils import PrismaClient

    client = MagicMock()
    client.db.query_raw = AsyncMock(side_effect=Exception("connection refused"))
    client.proxy_logging_obj.failure_handler = AsyncMock()
    client._probe_target_wrapper = MagicMock(return_value=client.db)
    client._run_health_probe = partial(PrismaClient._run_health_probe, client)
    client._report_health_check_failure = AsyncMock()

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        with pytest.raises(Exception, match="connection refused"):
            await PrismaClient.health_check(client)

    assert "health_check()" in caplog.text
    assert "disconnect()" not in caplog.text
    assert "connection refused" in caplog.text


@pytest.mark.asyncio
async def test_prisma_connect_failure_is_reported_at_operator_visible_level(caplog):
    """The sibling connect failure is labelled correctly but was equally
    invisible. A database the proxy could not connect to at startup must not be
    a debug-only record."""
    import logging
    from unittest.mock import AsyncMock

    from litellm.proxy.utils import PrismaClient

    client = MagicMock()
    client.db.is_connected = MagicMock(return_value=False)
    client.db.connect = AsyncMock(side_effect=Exception("could not reach database"))
    client.proxy_logging_obj.failure_handler = AsyncMock()

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        with pytest.raises(Exception, match="could not reach database"):
            await PrismaClient.connect(client)

    assert "connect()" in caplog.text
    assert "could not reach database" in caplog.text


@pytest.mark.asyncio
async def test_prisma_health_check_failure_redacts_database_credentials(caplog):
    """Raising the level must not widen what reaches the logs. The exception
    text can carry a full connection string, so the credential has to be gone
    from the emitted record."""
    import logging
    from functools import partial
    from unittest.mock import AsyncMock

    from litellm.proxy.utils import PrismaClient

    client = MagicMock()
    client.db.query_raw = AsyncMock(
        side_effect=Exception("could not connect to postgresql://admin:hunter2@db.internal:5432/litellm")
    )
    client.proxy_logging_obj.failure_handler = AsyncMock()
    client._probe_target_wrapper = MagicMock(return_value=client.db)
    client._run_health_probe = partial(PrismaClient._run_health_probe, client)
    client._report_health_check_failure = AsyncMock()

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        with pytest.raises(Exception, match="could not connect to"):
            await PrismaClient.health_check(client)

    emitted = [record.getMessage() for record in caplog.records if record.name == "LiteLLM Proxy"]

    assert emitted
    assert all("hunter2" not in message for message in emitted)
    assert any("postgresql://REDACTED@db.internal" in message for message in emitted)


@pytest.mark.asyncio
async def test_update_data_key_branch_stamps_settings_updated_at():
    """`updated_at` carries Prisma's @updatedAt and is rewritten by every spend
    flush, so key config edits need their own audit column."""
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock

    from litellm.proxy.utils import PrismaClient

    client = MagicMock()
    client.jsonify_object = MagicMock(side_effect=lambda data: dict(data))
    client.db.litellm_verificationtoken.update = AsyncMock(return_value=None)

    before = datetime.now(timezone.utc)
    await PrismaClient.update_data(client, token="sk-test-key", data={"models": ["gpt-4"]})
    after = datetime.now(timezone.utc)

    sent = client.db.litellm_verificationtoken.update.call_args.kwargs["data"]
    assert sent["models"] == ["gpt-4"]
    assert before <= sent["settings_updated_at"] <= after


@pytest.mark.asyncio
async def test_post_mcp_call_hook_skips_opted_out_guardrail(restore_callbacks):
    """A guardrail that keeps its native lifecycle hooks must not have MCP tool results
    scanned through the unified path, even though it implements apply_guardrail."""
    from mcp.types import CallToolResult, TextContent

    class _OptedOutMCPGuardrail(_RecordingMCPGuardrail):
        # apply_guardrail is redefined rather than inherited because the dispatch check
        # reads the leaf class __dict__, so an inherited override would skip for the
        # wrong reason and leave the flag untested
        use_native_lifecycle_hooks = True

        async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
            return await super().apply_guardrail(inputs, request_data, input_type, **kwargs)

    guardrail = _OptedOutMCPGuardrail(event_hook=GuardrailEventHooks.post_mcp_call)
    litellm.callbacks = [guardrail]
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    result = CallToolResult(content=[TextContent(type="text", text="jane@example.com")], isError=False)

    returned = await proxy_logging_obj.post_mcp_call_hook(
        response=result,
        request_data={"mcp_tool_name": "echo"},
        user_api_key_dict=None,
    )

    assert guardrail.call_count == 0
    assert [item.text for item in returned.content] == ["jane@example.com"]


FAILURE_USAGE_MODEL = "gpt-4o"
ONE_USER_MESSAGE = [{"role": "user", "content": "hi"}]


class _LoggingObj:
    def __init__(self, model_call_details):
        self.model_call_details = model_call_details


@pytest.mark.parametrize(
    "system_input, expected",
    [
        ("be brief", "be brief"),
        ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "ab"),
        (["a", {"text": "b"}], "ab"),
        ([{"type": "image"}], ""),
        (None, ""),
        (17, ""),
    ],
)
def test_a_system_prompt_reads_the_same_whatever_shape_it_arrived_in(system_input, expected):
    from litellm.proxy.utils import _system_prompt_text

    assert _system_prompt_text(system_input) == expected


def test_a_system_prompt_is_counted_on_top_of_the_request():
    from litellm.proxy.utils import count_request_input_tokens

    without = count_request_input_tokens(FAILURE_USAGE_MODEL, "hello world", None)
    with_system = count_request_input_tokens(FAILURE_USAGE_MODEL, "hello world", "be brief")

    assert without > 0
    assert with_system > without


def test_a_request_with_nothing_in_it_counts_zero():
    from litellm.proxy.utils import count_request_input_tokens

    assert count_request_input_tokens(FAILURE_USAGE_MODEL, [], None) == 0
    assert count_request_input_tokens(FAILURE_USAGE_MODEL, None, None) == 0


def test_a_failed_dispatch_is_estimated_as_input_only():
    from litellm.proxy.utils import count_request_input_tokens, _estimate_dispatched_failure_usage

    usage = _estimate_dispatched_failure_usage(FAILURE_USAGE_MODEL, ONE_USER_MESSAGE, None)

    assert usage is not None
    assert usage.prompt_tokens == count_request_input_tokens(FAILURE_USAGE_MODEL, ONE_USER_MESSAGE, None)
    assert usage.completion_tokens == 0
    assert usage.total_tokens == usage.prompt_tokens


@pytest.mark.parametrize("request_input", [[], object()])
def test_nothing_is_estimated_when_there_is_nothing_to_count(request_input):
    from litellm.proxy.utils import _estimate_dispatched_failure_usage

    assert _estimate_dispatched_failure_usage(FAILURE_USAGE_MODEL, request_input, None) is None


def test_usage_the_stream_already_recovered_beats_an_estimate():
    from litellm.proxy.utils import _failure_usage_to_lift
    from litellm.types.utils import Usage

    recovered = Usage(prompt_tokens=5, completion_tokens=7, total_tokens=12)

    lifted = _failure_usage_to_lift(
        model_call_details={"combined_usage_object": recovered, "response_cost": 0.25},
        request_body={},
        dispatched=True,
    )

    assert lifted == (recovered, 0.25)


def test_a_failed_request_counts_its_instructions_even_after_logging_redacted_them():
    from litellm.constants import REDACTED_BY_LITELLM
    from litellm.proxy.utils import _failure_usage_to_lift

    instructions: Final = "Review every changed line and explain each risk you find. " * 40
    call: Final[dict[str, object]] = {
        "call_type": "aresponses",
        "model": FAILURE_USAGE_MODEL,
        "messages": ONE_USER_MESSAGE,
    }

    def estimated_prompt_tokens(dispatched_instructions: str) -> int:
        lifted: Final = _failure_usage_to_lift(
            model_call_details={**call, "optional_params": {"instructions": dispatched_instructions}},
            request_body={"instructions": instructions},
            dispatched=True,
        )
        assert lifted is not None
        usage: Final = lifted[0]
        assert isinstance(usage, Usage)
        return usage.prompt_tokens

    assert estimated_prompt_tokens(REDACTED_BY_LITELLM) == estimated_prompt_tokens(instructions)


def test_a_timed_out_request_on_an_unmapped_model_keeps_its_row_at_zero_cost():
    from litellm.proxy.utils import _failure_usage_to_lift

    lifted = _failure_usage_to_lift(
        model_call_details={
            "call_type": "acompletion",
            "model": "my-unmapped-model",
            "custom_llm_provider": "openai",
            "messages": ONE_USER_MESSAGE,
        },
        request_body={},
        dispatched=True,
        provider_billed=True,
    )

    assert lifted is not None
    usage, response_cost = lifted
    assert usage.prompt_tokens > 0
    assert response_cost == 0.0


def test_a_timed_out_request_is_priced_at_the_rate_the_router_registered_for_the_deployment(monkeypatch):
    """Router._create_deployment registers a deployment's own ``input_cost_per_token`` under its
    ``model_info.id``; the failure row prices with that entry through the same
    ``deployment_pricing_model_info`` the success path uses."""
    from litellm.proxy.utils import _failure_usage_to_lift

    monkeypatch.setitem(
        litellm.model_cost,
        "dep-lit145",
        {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0.25, "output_cost_per_token": 1.0},
    )

    lifted = _failure_usage_to_lift(
        model_call_details={
            "call_type": "acompletion",
            "model": FAILURE_USAGE_MODEL,
            "messages": ONE_USER_MESSAGE,
        },
        request_body={},
        dispatched=True,
        provider_billed=True,
        deployment_id="dep-lit145",
    )

    assert lifted is not None
    usage, response_cost = lifted
    assert usage.prompt_tokens > 0
    assert response_cost == pytest.approx(usage.prompt_tokens * 0.25)


@pytest.mark.parametrize(("prompt_tokens", "expected_rate"), [(1_000, 1e-7), (250_000, 5e-7)])
def test_a_timed_out_request_is_priced_at_the_deployments_long_context_tier(monkeypatch, prompt_tokens, expected_rate):
    """A deployment that declares ``input_cost_per_token_above_200k_tokens`` next to its standard rate
    bills a 250k-token prompt at the tier on the success path; the failure row must not flatten it to
    the standard rate."""
    from litellm.proxy.utils import _estimated_input_cost

    monkeypatch.setitem(
        litellm.model_cost,
        "dep-lit-f9-tiered",
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 1e-7,
            "input_cost_per_token_above_200k_tokens": 5e-7,
            "output_cost_per_token": 4e-7,
        },
    )

    response_cost = _estimated_input_cost(
        model_call_details={"model": FAILURE_USAGE_MODEL, "custom_llm_provider": "openai"},
        prompt_tokens=prompt_tokens,
        deployment_id="dep-lit-f9-tiered",
        base_model=None,
    )

    assert response_cost == pytest.approx(prompt_tokens * expected_rate)


def test_every_earlier_timed_out_attempt_is_billed_at_its_own_deployment_rate(monkeypatch):
    """The router leaves one RetryAttemptRecord per failed attempt in metadata.previous_models, the
    final attempt included; the earlier Timeout ones are billed (each at the deployment that served
    it), a 429 attempt is not, and the final attempt is left to the failure row itself."""
    from litellm.proxy.utils import _billed_earlier_attempts

    monkeypatch.setitem(
        litellm.model_cost,
        "dep-lit145-b",
        {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0.25, "output_cost_per_token": 1.0},
    )
    usage = litellm.Usage(prompt_tokens=24, completion_tokens=0, total_tokens=24)
    public_rate, _ = litellm.cost_per_token(model=FAILURE_USAGE_MODEL, prompt_tokens=24)

    attempts = _billed_earlier_attempts(
        model_call_details={"model": FAILURE_USAGE_MODEL, "custom_llm_provider": "openai"},
        litellm_params={
            "metadata": {
                "attempted_retries": 3,
                "previous_models": (
                    {"deployment_id": "dep-lit145-a", "exception_type": "Timeout", "attempted_retries": 0},
                    {"deployment_id": "dep-lit145-b", "exception_type": "RateLimitError", "attempted_retries": 1},
                    {"deployment_id": "dep-lit145-b", "exception_type": "Timeout", "attempted_retries": 2},
                    {"deployment_id": "dep-lit145-a", "exception_type": "Timeout", "attempted_retries": 3},
                ),
            }
        },
        estimated_usage=usage,
        base_model=None,
        deployment_id="dep-lit145-a",
    )

    assert [(a.attempted_retries, a.deployment_id) for a in attempts] == [(0, "dep-lit145-a"), (2, "dep-lit145-b")]
    assert attempts[0].response_cost == pytest.approx(public_rate)
    assert attempts[0].response_cost > 0
    assert attempts[1].response_cost == pytest.approx(24 * 0.25)
    assert all(a.usage is usage for a in attempts)


def test_an_earlier_attempt_on_another_model_is_priced_at_that_models_rate_not_the_final_models(monkeypatch):
    """After a fallback hop the final attempt runs on another model group, so the records of the group
    that timed out first name their own deployment model (and its ``base_model`` when the deployment
    name is not in the cost map). Each is billed at that model's input rate, not at the final model's."""
    from litellm.proxy.utils import _billed_earlier_attempts

    for pricing_model, provider, input_rate in (
        ("lit145-final-model", "openai", 0.5),
        ("lit145-first-model", "openai", 0.02),
        ("azure/lit145-first-model", "azure", 0.02),
    ):
        monkeypatch.setitem(
            litellm.model_cost,
            pricing_model,
            {"litellm_provider": provider, "mode": "chat", "input_cost_per_token": input_rate, "output_cost_per_token": 1.0},
        )
    usage = litellm.Usage(prompt_tokens=24, completion_tokens=0, total_tokens=24)

    attempts = _billed_earlier_attempts(
        model_call_details={"model": "openai/lit145-final-model", "custom_llm_provider": "openai"},
        litellm_params={
            "metadata": {
                "attempted_retries": 1,
                "model_info": {"id": "dep-lit145-final"},
                "previous_models": (
                    {
                        "deployment_id": "dep-lit145-first",
                        "model": "openai/lit145-first-model",
                        "base_model": None,
                        "exception_type": "Timeout",
                        "attempted_retries": 0,
                    },
                    {
                        "deployment_id": "dep-lit145-first-azure",
                        "model": "azure/lit145-first-deployment",
                        "base_model": "azure/lit145-first-model",
                        "exception_type": "Timeout",
                        "attempted_retries": 1,
                    },
                    {
                        "deployment_id": "dep-lit145-final",
                        "model": "openai/lit145-final-model",
                        "base_model": None,
                        "exception_type": "Timeout",
                        "attempted_retries": 0,
                    },
                    {
                        "deployment_id": "dep-lit145-final",
                        "model": "openai/lit145-final-model",
                        "base_model": None,
                        "exception_type": "Timeout",
                        "attempted_retries": 1,
                    },
                ),
            }
        },
        estimated_usage=usage,
        base_model=None,
        deployment_id="dep-lit145-final",
    )

    assert [(a.attempted_retries, a.deployment_id, a.response_cost) for a in attempts] == [
        (0, "dep-lit145-first", pytest.approx(24 * 0.02)),
        (1, "dep-lit145-first-azure", pytest.approx(24 * 0.02)),
        (0, "dep-lit145-final", pytest.approx(24 * 0.5)),
    ]


def test_earlier_attempts_are_priced_from_the_prompt_estimate_not_the_final_streams_recovered_usage():
    """A final attempt whose stream broke mid-flight lifts its recovered usage (prompt plus the chunks
    delivered); the earlier timed-out attempts never produced output, so they are priced from the
    input-only estimate and not from that recovered usage."""
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.proxy.utils import _estimated_prompt_usage, _failure_fields_to_lift

    logging_obj = _dispatched_failure_logging_obj(
        FAILURE_USAGE_MODEL,
        get_litellm_params(
            metadata={
                "attempted_retries": 1,
                "model_info": {"id": "dep-lit145-final"},
                "previous_models": (
                    {"deployment_id": "dep-lit145-final", "exception_type": "Timeout", "attempted_retries": 0},
                ),
            }
        ),
    )
    recovered = litellm.Usage(prompt_tokens=500, completion_tokens=40, total_tokens=540)
    logging_obj.model_call_details["combined_usage_object"] = recovered
    logging_obj.model_call_details["response_cost"] = 0.01

    lifted = _failure_fields_to_lift({"litellm_logging_obj": logging_obj}, provider_billed=True)

    estimate = _estimated_prompt_usage(logging_obj.model_call_details, {}, None)
    assert lifted["combined_usage_object"] is recovered
    assert [(a.usage.prompt_tokens, a.usage.completion_tokens) for a in lifted["billed_failed_attempts"]] == [
        (estimate.prompt_tokens, 0)
    ]
    assert estimate.prompt_tokens < recovered.prompt_tokens


def test_attempts_already_summed_into_the_final_row_are_not_lifted_again():
    """``Logging.carry_failed_attempt_usage`` moves a cut stream's usage into ``failed_attempts_usage``
    and the final row already bills every entry of it, so no earlier attempt gets a row of its own."""
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.proxy.utils import _failure_fields_to_lift

    logging_obj = _dispatched_failure_logging_obj(
        FAILURE_USAGE_MODEL,
        get_litellm_params(
            metadata={
                "attempted_retries": 1,
                "model_info": {"id": "dep-lit145-final"},
                "previous_models": (
                    {"deployment_id": "dep-lit145-final", "exception_type": "Timeout", "attempted_retries": 0},
                ),
            }
        ),
    )
    logging_obj.model_call_details["failed_attempts_usage"] = (
        (litellm.Usage(prompt_tokens=24, completion_tokens=3, total_tokens=27), 0.001),
    )

    lifted = _failure_fields_to_lift({"litellm_logging_obj": logging_obj}, provider_billed=True)

    current_only_logging_obj: Final = _dispatched_failure_logging_obj(
        FAILURE_USAGE_MODEL,
        get_litellm_params(
            metadata={
                "attempted_retries": 1,
                "model_info": {"id": "dep-lit145-final"},
                "previous_models": (
                    {"deployment_id": "dep-lit145-final", "exception_type": "Timeout", "attempted_retries": 0},
                ),
            }
        ),
    )
    current_only: Final = _failure_fields_to_lift(
        {"litellm_logging_obj": current_only_logging_obj}, provider_billed=True
    )
    current_only_usage: Final = current_only["combined_usage_object"]

    assert current_only_usage.prompt_tokens > 0
    assert lifted["combined_usage_object"].prompt_tokens == 24 + current_only_usage.prompt_tokens
    assert "billed_failed_attempts" not in lifted


def test_carry_superseded_attempt_usage_without_stashed_usage_preserves_failure_estimate() -> None:
    request_start: Final = real_datetime.datetime(2026, 1, 1)
    logging_obj: Final = Logging(
        model=FAILURE_USAGE_MODEL,
        messages=ONE_USER_MESSAGE,
        stream=False,
        call_type="acompletion",
        start_time=request_start,
        litellm_call_id="call-lit145-no-stashed-usage",
        function_id="fn-lit145-no-stashed-usage",
    )
    logging_obj.pre_call(input=ONE_USER_MESSAGE, api_key="sk-test", model=FAILURE_USAGE_MODEL)
    logging_obj.model_call_details["call_type"] = "acompletion"
    logging_obj.model_call_details["messages"] = ONE_USER_MESSAGE

    try:
        try:
            raise httpx.ReadTimeout("no bytes in time")
        except httpx.ReadTimeout as cause:
            raise litellm.Timeout("Request timed out.", model=FAILURE_USAGE_MODEL, llm_provider="openai") from cause
    except litellm.Timeout as timeout:
        logging_obj.failure_handler(
            timeout,
            traceback.format_exc(),
            request_start,
            request_start + real_datetime.timedelta(seconds=1),
        )

    assert "combined_usage_object" not in logging_obj.model_call_details
    request_data: Final = {"litellm_logging_obj": logging_obj}
    before: Final = _failure_fields_to_lift(request_data, provider_billed=True)

    carry_superseded_attempt_usage(request_data)

    after: Final = _failure_fields_to_lift(request_data, provider_billed=True)
    before_usage: Final = before.get("combined_usage_object")
    after_usage: Final = after.get("combined_usage_object")
    assert before_usage is not None
    assert after_usage is not None
    assert after_usage == before_usage
    assert before["response_cost"] > 0
    assert after["response_cost"] == before["response_cost"]


@pytest.mark.asyncio
async def test_dropped_stream_failure_attribution_reaches_terminal_spend_row() -> None:
    from litellm.exceptions import MidStreamFallbackError
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage

    request_start: Final = real_datetime.datetime(2026, 1, 1)
    model: Final = "deployment-model"
    logging_obj: Final = Logging(
        model=model,
        messages=ONE_USER_MESSAGE,
        stream=True,
        call_type="acompletion",
        start_time=request_start,
        litellm_call_id="call-lit145-attribution",
        function_id="fn-lit145-attribution",
    )
    logging_obj.update_environment_variables(
        model=model,
        user="",
        optional_params={},
        litellm_params=get_litellm_params(metadata={"model_info": {"id": "deployment-id"}}),
    )
    logging_obj.model_call_details = {
        **logging_obj.model_call_details,
        "custom_llm_provider": "openai",
        "first_api_call_start_time": request_start,
        "current_attempt_api_call_start_time": request_start,
    }
    wrapper: Final = CustomStreamWrapper(
        completion_stream=None,
        model=model,
        logging_obj=logging_obj,
        custom_llm_provider="openai",
    )
    wrapper.chunks = [
        ModelResponseStream(
            id="chatcmpl-attribution",
            created=1742056047,
            model=model,
            object="chat.completion.chunk",
            choices=[StreamingChoices(finish_reason=None, index=0, delta=Delta(content="The"))],
            usage=Usage(prompt_tokens=30, completion_tokens=1, total_tokens=31),
        )
    ]
    exception: Final = httpx.ReadError("connection dropped")

    with pytest.raises(MidStreamFallbackError):
        wrapper._log_stream_failure_and_raise(exception)
    await wrapper.wait_for_failure_logging(timeout=10)

    request_data: Final = {"litellm_logging_obj": logging_obj}
    lifted: Final = _failure_fields_to_lift(request_data, provider_billed=True)
    terminal_kwargs: Final = {
        key: value
        for key, value in {**request_data, **lifted}.items()
        if key != "litellm_logging_obj"
    }
    row: Final = get_logging_payload(
        kwargs=terminal_kwargs,
        response_obj=exception,
        start_time=request_start,
        end_time=request_start + real_datetime.timedelta(seconds=1),
    )

    assert row["model_id"] == "deployment-id"
    assert row["custom_llm_provider"] == "openai"
    assert row["model"] == model


class _ReplaceSharedFailurePayload(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logging_obj: Logging | None = None
        self.replaced: asyncio.Event = asyncio.Event()
        self.started: asyncio.Event = asyncio.Event()
        self.release: asyncio.Event = asyncio.Event()
        self.pause_until_released: bool = False
        self.replacement_payload: object | None = None

    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: real_datetime.datetime,
        end_time: real_datetime.datetime,
    ) -> None:
        from pydantic import TypeAdapter

        from litellm.types.utils import StandardLoggingPayload

        payload: Final = TypeAdapter(StandardLoggingPayload).validate_python(kwargs["standard_logging_object"])
        self.started.set()
        if self.pause_until_released:
            await self.release.wait()
        logging_obj: Final = self.logging_obj
        if logging_obj is None:
            return
        replacement: Final = self.replacement_payload or TypeAdapter(StandardLoggingPayload).validate_python(
            {
                **payload,
                "model_id": "newer-deployment-id",
                "custom_llm_provider": "newer-provider",
                "model": "newer-model",
            }
        )
        logging_obj.model_call_details = {
            **logging_obj.model_call_details,
            "standard_logging_object": replacement,
        }
        self.replaced.set()


class _OrderedAttemptFailureRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.attempt_a_started: asyncio.Event = asyncio.Event()
        self.attempt_b_started: asyncio.Event = asyncio.Event()
        self.release_attempt_a: asyncio.Event = asyncio.Event()
        self.release_attempt_b: asyncio.Event = asyncio.Event()

    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: real_datetime.datetime,
        end_time: real_datetime.datetime,
    ) -> None:
        from pydantic import TypeAdapter

        from litellm.types.utils import StandardLoggingPayload

        payload: Final = TypeAdapter(StandardLoggingPayload).validate_python(kwargs["standard_logging_object"])
        if payload["model_id"] == "deployment-a":
            self.attempt_a_started.set()
            await self.release_attempt_a.wait()
            return
        if payload["model_id"] == "deployment-b":
            self.attempt_b_started.set()
            await self.release_attempt_b.wait()


class _PausedBaselineFailureRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.started: asyncio.Event = asyncio.Event()
        self.release: asyncio.Event = asyncio.Event()
        self.standard_logging_object: object | None = None

    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: real_datetime.datetime,
        end_time: real_datetime.datetime,
    ) -> None:
        self.standard_logging_object = kwargs.get("standard_logging_object")
        self.started.set()
        await self.release.wait()


@pytest.mark.asyncio
async def test_latest_dropped_attempt_publishes_its_payload_after_earlier_dispatch_finishes() -> None:
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
    from litellm.types.utils import Usage

    request_start: Final = real_datetime.datetime(2026, 1, 1)
    marker_a: Final = request_start
    marker_b: Final = request_start + real_datetime.timedelta(seconds=1)
    recorder: Final = _OrderedAttemptFailureRecorder()
    logging_obj: Final = Logging(
        model="deployment-model-a",
        messages=ONE_USER_MESSAGE,
        stream=True,
        call_type="acompletion",
        start_time=request_start,
        litellm_call_id="call-lit145-attempt-order",
        function_id="fn-lit145-attempt-order",
        dynamic_async_failure_callbacks=[recorder],
    )
    logging_obj.update_environment_variables(
        model="deployment-model-a",
        user="",
        optional_params={},
        litellm_params=get_litellm_params(metadata={"model_info": {"id": "deployment-a"}}),
    )
    logging_obj.model_call_details = {
        **logging_obj.model_call_details,
        "custom_llm_provider": "openai",
        "first_api_call_start_time": request_start,
        "current_attempt_api_call_start_time": marker_a,
        "combined_usage_object": Usage(prompt_tokens=4, completion_tokens=1, total_tokens=5),
        "response_cost": 0.001,
    }

    attempt_a: Final = asyncio.create_task(
        logging_obj.dispatch_attempt_failure_handlers(httpx.ReadError("attempt A dropped"), "")
    )
    await asyncio.wait_for(recorder.attempt_a_started.wait(), timeout=10)

    logging_obj.carry_failed_attempt_usage()
    logging_obj.update_environment_variables(
        model="deployment-model-b",
        user="",
        optional_params={},
        litellm_params=get_litellm_params(metadata={"model_info": {"id": "deployment-b"}}),
    )
    logging_obj.model_call_details = {
        **logging_obj.model_call_details,
        "model": "deployment-model-b",
        "custom_llm_provider": "anthropic",
        "current_attempt_api_call_start_time": marker_b,
        "combined_usage_object": Usage(prompt_tokens=7, completion_tokens=2, total_tokens=9),
        "response_cost": 0.002,
    }

    attempt_b: Final = asyncio.create_task(
        logging_obj.dispatch_attempt_failure_handlers(httpx.ReadError("attempt B dropped"), "")
    )
    await asyncio.wait_for(recorder.attempt_b_started.wait(), timeout=10)
    recorder.release_attempt_a.set()
    await attempt_a
    recorder.release_attempt_b.set()
    await attempt_b

    request_data: Final = {"litellm_logging_obj": logging_obj}
    lifted: Final = _failure_fields_to_lift(request_data, provider_billed=True)
    terminal_kwargs: Final = {
        key: value
        for key, value in {**request_data, **lifted}.items()
        if key != "litellm_logging_obj"
    }
    row: Final = get_logging_payload(
        kwargs=terminal_kwargs,
        response_obj=httpx.ReadError("attempt B dropped"),
        start_time=request_start,
        end_time=request_start + real_datetime.timedelta(seconds=1),
    )

    assert row["model_id"] == "deployment-b"
    assert row["custom_llm_provider"] == "anthropic"
    assert row["model"] == "deployment-model-b"


@pytest.mark.asyncio
async def test_dropped_attempt_attribution_is_published_while_failure_callback_is_paused() -> None:
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
    from litellm.types.utils import Usage

    request_start: Final = real_datetime.datetime(2026, 1, 1)
    marker_a: Final = request_start
    failure: Final = httpx.ReadError("attempt A dropped")
    recorder: Final = _OrderedAttemptFailureRecorder()
    logging_obj: Final = Logging(
        model="deployment-model-a",
        messages=ONE_USER_MESSAGE,
        stream=True,
        call_type="acompletion",
        start_time=request_start,
        litellm_call_id="call-lit145-attempt-paused",
        function_id="fn-lit145-attempt-paused",
        dynamic_async_failure_callbacks=[recorder],
    )
    logging_obj.update_environment_variables(
        model="deployment-model-a",
        user="",
        optional_params={},
        litellm_params=get_litellm_params(metadata={"model_info": {"id": "deployment-a"}}),
    )
    logging_obj.model_call_details = {
        **logging_obj.model_call_details,
        "custom_llm_provider": "openai",
        "first_api_call_start_time": request_start,
        "current_attempt_api_call_start_time": marker_a,
        "combined_usage_object": Usage(prompt_tokens=4, completion_tokens=1, total_tokens=5),
        "response_cost": 0.001,
    }

    attempt: Final = asyncio.create_task(logging_obj.dispatch_attempt_failure_handlers(failure, ""))
    await asyncio.wait_for(recorder.attempt_a_started.wait(), timeout=10)

    request_data: Final = {"litellm_logging_obj": logging_obj}
    lifted: Final = _failure_fields_to_lift(request_data, provider_billed=True)
    terminal_kwargs: Final = {
        key: value
        for key, value in {**request_data, **lifted}.items()
        if key != "litellm_logging_obj"
    }
    row: Final = get_logging_payload(
        kwargs=terminal_kwargs,
        response_obj=failure,
        start_time=request_start,
        end_time=request_start + real_datetime.timedelta(seconds=1),
    )

    try:
        assert row["model_id"] == "deployment-a"
        assert row["custom_llm_provider"] == "openai"
        assert row["model"] == "deployment-model-a"
        assert row["prompt_tokens"] == 4
        assert row["completion_tokens"] == 1
    finally:
        recorder.release_attempt_a.set()
        await attempt


@pytest.mark.asyncio
async def test_failed_attempt_publishes_baseline_reason_after_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import TypeAdapter

    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.proxy.hooks.autorouter_baseline_cache import AutoRouterBaselineCache, CapturedBaselineObservation
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
    from litellm.router import Router
    from litellm.types.utils import CallTypes, StandardLoggingPayload, Usage
    from tests.unit.proxy.hooks.test_autorouter_baseline_cache import _MODELS, _kwargs

    request_start: Final = real_datetime.datetime(2026, 1, 1)
    marker: Final = request_start
    model: Final = "deployment-model-a"
    router: Final = Router(model_list=_MODELS, disable_cooldowns=True)
    baseline_cache: Final = AutoRouterBaselineCache(
        None,
        router=lambda: router,
        clock=lambda: request_start.timestamp() + 10,
    )
    recorder: Final = _PausedBaselineFailureRecorder()
    logging_obj: Final = Logging(
        model=model,
        messages=ONE_USER_MESSAGE,
        stream=True,
        call_type=CallTypes.anthropic_messages.value,
        start_time=request_start,
        litellm_call_id="call-lit145-baseline-failure",
        function_id="fn-lit145-baseline-failure",
        dynamic_async_failure_callbacks=[recorder],
    )
    logging_obj.update_environment_variables(
        model=model,
        user="",
        optional_params={},
        litellm_params=get_litellm_params(metadata={"model_info": {"id": "deployment-a"}}),
    )
    logging_obj.model_call_details = {
        **logging_obj.model_call_details,
        "custom_llm_provider": "openai",
        "first_api_call_start_time": request_start,
        "current_attempt_api_call_start_time": marker,
        "combined_usage_object": Usage(prompt_tokens=4, completion_tokens=1, total_tokens=5),
        "response_cost": 0.001,
    }
    await baseline_cache.async_pre_call_deployment_hook(_kwargs(logging_obj), CallTypes.anthropic_messages)
    assert logging_obj.baseline_cache_context is not None

    failure: Final = httpx.ReadError("attempt A dropped")
    attempt: Final = asyncio.create_task(logging_obj.dispatch_attempt_failure_handlers(failure, ""))
    try:
        await asyncio.wait_for(recorder.started.wait(), timeout=10)
        request_data: Final = {"litellm_logging_obj": logging_obj}
        lifted: Final = _failure_fields_to_lift(request_data, provider_billed=True)
        terminal_kwargs: Final = {
            key: value
            for key, value in {**request_data, **lifted}.items()
            if key != "litellm_logging_obj"
        }
        row: Final = get_logging_payload(
            kwargs=terminal_kwargs,
            response_obj=failure,
            start_time=request_start,
            end_time=request_start + real_datetime.timedelta(seconds=1),
        )

        assert row["model_id"] == "deployment-a"
        assert row["custom_llm_provider"] == "openai"
        assert row["model"] == model
        assert row["prompt_tokens"] == 4
        assert row["completion_tokens"] == 1
        lifted_payload: Final = TypeAdapter(StandardLoggingPayload).validate_python(lifted["standard_logging_object"])
        callback_payload: Final = TypeAdapter(StandardLoggingPayload).validate_python(
            recorder.standard_logging_object
        )

        def baseline_reason(payload: StandardLoggingPayload) -> str:
            encoded: Final = payload["autorouter_baseline_observation"]
            assert isinstance(encoded, str)
            return CapturedBaselineObservation.model_validate_json(encoded).observation.reason or ""

        assert baseline_reason(lifted_payload) == "failed_request"
        assert baseline_reason(callback_payload) == "failed_request"
    finally:
        recorder.release.set()
        await attempt


@pytest.mark.asyncio
async def test_queued_stale_attempt_dispatch_does_not_publish_over_current_attempt() -> None:
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
    from litellm.types.utils import Usage

    request_start: Final = real_datetime.datetime(2026, 1, 1)
    marker_a: Final = request_start
    marker_b: Final = request_start + real_datetime.timedelta(seconds=1)
    recorder: Final = _OrderedAttemptFailureRecorder()
    logging_obj: Final = Logging(
        model="deployment-model-a",
        messages=ONE_USER_MESSAGE,
        stream=True,
        call_type="acompletion",
        start_time=request_start,
        litellm_call_id="call-lit145-stale-attempt",
        function_id="fn-lit145-stale-attempt",
        dynamic_async_failure_callbacks=[recorder],
    )
    logging_obj.update_environment_variables(
        model="deployment-model-a",
        user="",
        optional_params={},
        litellm_params=get_litellm_params(metadata={"model_info": {"id": "deployment-a"}}),
    )
    logging_obj.model_call_details = {
        **logging_obj.model_call_details,
        "custom_llm_provider": "openai",
        "first_api_call_start_time": request_start,
        "current_attempt_api_call_start_time": marker_a,
        "combined_usage_object": Usage(prompt_tokens=4, completion_tokens=1, total_tokens=5),
        "response_cost": 0.001,
    }
    failure_a: Final = httpx.ReadError("attempt A dropped")
    coro_a: Final = logging_obj.dispatch_attempt_failure_handlers(failure_a, "")

    logging_obj.carry_failed_attempt_usage()
    logging_obj.update_environment_variables(
        model="deployment-model-b",
        user="",
        optional_params={},
        litellm_params=get_litellm_params(metadata={"model_info": {"id": "deployment-b"}}),
    )
    logging_obj.model_call_details = {
        **logging_obj.model_call_details,
        "model": "deployment-model-b",
        "custom_llm_provider": "anthropic",
        "current_attempt_api_call_start_time": marker_b,
        "combined_usage_object": Usage(prompt_tokens=7, completion_tokens=2, total_tokens=9),
        "response_cost": 0.002,
    }
    failure_b: Final = httpx.ReadError("attempt B dropped")
    coro_b: Final = logging_obj.dispatch_attempt_failure_handlers(failure_b, "")

    task_a: Final = asyncio.create_task(coro_a)
    task_b: Final = asyncio.create_task(coro_b)
    try:
        await asyncio.wait_for(
            asyncio.gather(recorder.attempt_a_started.wait(), recorder.attempt_b_started.wait()),
            timeout=10,
        )
        request_data: Final = {"litellm_logging_obj": logging_obj}
        lifted: Final = _failure_fields_to_lift(request_data, provider_billed=True)
        terminal_kwargs: Final = {
            key: value
            for key, value in {**request_data, **lifted}.items()
            if key != "litellm_logging_obj"
        }
        row: Final = get_logging_payload(
            kwargs=terminal_kwargs,
            response_obj=failure_b,
            start_time=request_start,
            end_time=request_start + real_datetime.timedelta(seconds=1),
        )

        assert row["model_id"] == "deployment-b"
        assert row["custom_llm_provider"] == "anthropic"
        assert row["model"] == "deployment-model-b"
    finally:
        recorder.release_attempt_a.set()
        recorder.release_attempt_b.set()
        await asyncio.gather(task_a, task_b)


@pytest.mark.asyncio
@pytest.mark.parametrize("queued_start", [False, True], ids=["running-start", "queued-start"])
async def test_dropped_stream_failure_does_not_overwrite_newer_terminal_payload(queued_start: bool) -> None:
    from litellm.exceptions import MidStreamFallbackError
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
    from litellm.types.utils import Delta, ModelResponseStream, StandardLoggingPayload, StreamingChoices, Usage
    from pydantic import TypeAdapter

    request_start: Final = real_datetime.datetime(2026, 1, 1)
    model: Final = "deployment-model"
    recorder: Final = _ReplaceSharedFailurePayload()
    logging_obj: Final = Logging(
        model=model,
        messages=ONE_USER_MESSAGE,
        stream=True,
        call_type="acompletion",
        start_time=request_start,
        litellm_call_id="call-lit145-identity",
        function_id="fn-lit145-identity",
        dynamic_async_failure_callbacks=[recorder],
    )
    recorder.logging_obj = logging_obj
    initial_payload: Final = {
        "model_id": "older-deployment-id",
        "custom_llm_provider": "older-provider",
        "model": "older-model",
    }
    logging_obj.update_environment_variables(
        model=model,
        user="",
        optional_params={},
        litellm_params=get_litellm_params(metadata={"model_info": {"id": "deployment-id"}}),
    )
    logging_obj.model_call_details = {
        **logging_obj.model_call_details,
        "custom_llm_provider": "openai",
        "first_api_call_start_time": request_start,
        "current_attempt_api_call_start_time": request_start,
        "standard_logging_object": initial_payload,
    }

    def assert_newer_payload_attribution() -> None:
        request_data: Final = {"litellm_logging_obj": logging_obj}
        lifted: Final = _failure_fields_to_lift(request_data, provider_billed=True)
        terminal_kwargs: Final = {
            key: value
            for key, value in {**request_data, **lifted}.items()
            if key != "litellm_logging_obj"
        }
        row: Final = get_logging_payload(
            kwargs=terminal_kwargs,
            response_obj=exception,
            start_time=request_start,
            end_time=request_start + real_datetime.timedelta(seconds=1),
        )

        assert row["model_id"] == "newer-deployment-id"
        assert row["custom_llm_provider"] == "newer-provider"
        assert row["model"] == "newer-model"

    exception: Final = httpx.ReadError("connection dropped")
    if queued_start:
        dispatch: Final = logging_obj.dispatch_attempt_failure_handlers(exception, "")
        logging_obj._failure_handler_helper_fn(
            exception=httpx.ReadError("newer failure"),
            traceback_exception="",
            start_time=request_start,
            end_time=request_start + real_datetime.timedelta(seconds=1),
        )
        replacement_base: Final = logging_obj.model_call_details["standard_logging_object"]
        replacement_payload: Final = TypeAdapter(StandardLoggingPayload).validate_python(
            {
                **replacement_base,
                "model_id": "newer-deployment-id",
                "custom_llm_provider": "newer-provider",
                "model": "newer-model",
            }
        )
        recorder.pause_until_released = True
        recorder.replacement_payload = replacement_payload
        logging_obj.model_call_details = {
            **logging_obj.model_call_details,
            "standard_logging_object": replacement_payload,
        }
        assert logging_obj.model_call_details["current_attempt_api_call_start_time"] == request_start

        task: Final = asyncio.create_task(dispatch)
        try:
            await asyncio.wait_for(recorder.started.wait(), timeout=10)
            assert_newer_payload_attribution()
        finally:
            recorder.release.set()
            await task

        await asyncio.wait_for(recorder.replaced.wait(), timeout=10)
        assert_newer_payload_attribution()
        return

    wrapper: Final = CustomStreamWrapper(
        completion_stream=None,
        model=model,
        logging_obj=logging_obj,
        custom_llm_provider="openai",
    )
    wrapper.chunks = [
        ModelResponseStream(
            id="chatcmpl-identity",
            created=1742056047,
            model=model,
            object="chat.completion.chunk",
            choices=[StreamingChoices(finish_reason=None, index=0, delta=Delta(content="The"))],
            usage=Usage(prompt_tokens=30, completion_tokens=1, total_tokens=31),
        )
    ]

    with pytest.raises(MidStreamFallbackError):
        wrapper._log_stream_failure_and_raise(exception)
    await wrapper.wait_for_failure_logging(timeout=10)
    await asyncio.wait_for(recorder.replaced.wait(), timeout=10)
    assert_newer_payload_attribution()


def test_a_timed_out_attempt_from_the_group_a_fallback_left_is_still_billed():
    """After a fallback hop the router restarts ``attempted_retries`` at 0 for the new model group and
    re-logs the attempt that exhausted the previous group, so the final attempt (dep-b, retry 1) must be
    told from the previous group's attempt with the same count (dep-a, retry 1), and the re-logged
    record must bill once."""
    from litellm.proxy.utils import _billed_earlier_attempts

    usage = litellm.Usage(prompt_tokens=24, completion_tokens=0, total_tokens=24)

    attempts = _billed_earlier_attempts(
        model_call_details={"model": FAILURE_USAGE_MODEL, "custom_llm_provider": "openai"},
        litellm_params={
            "metadata": {
                "attempted_retries": 1,
                "model_info": {"id": "dep-lit145-b"},
                "previous_models": (
                    {"deployment_id": "dep-lit145-a", "exception_type": "Timeout", "attempted_retries": 0},
                    {"deployment_id": "dep-lit145-a", "exception_type": "Timeout", "attempted_retries": 1},
                    {"deployment_id": "dep-lit145-a", "exception_type": "Timeout", "attempted_retries": 1},
                    {"deployment_id": "dep-lit145-b", "exception_type": "Timeout", "attempted_retries": 0},
                    {"deployment_id": "dep-lit145-b", "exception_type": "Timeout", "attempted_retries": 1},
                ),
            }
        },
        estimated_usage=usage,
        base_model=None,
        deployment_id="dep-lit145-b",
    )

    assert [(a.attempted_retries, a.deployment_id) for a in attempts] == [
        (0, "dep-lit145-a"),
        (1, "dep-lit145-a"),
        (0, "dep-lit145-b"),
    ]
    assert all(a.response_cost > 0 for a in attempts)


def _dispatched_failure_logging_obj(model: str, litellm_params: dict) -> object:
    """A Logging object as the failure hook sees it: litellm_params in the real get_litellm_params()
    shape (every key present, ``litellm_metadata`` None on chat completions) and a first-handoff
    instant, so the lift treats the request as dispatched."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj = Logging(
        model=model,
        messages=ONE_USER_MESSAGE,
        stream=False,
        call_type="acompletion",
        start_time=real_datetime.datetime.now(),
        litellm_call_id="call-lit145",
        function_id="fn-lit145",
    )
    logging_obj.update_environment_variables(model=model, user="", optional_params={}, litellm_params=litellm_params)
    logging_obj.model_call_details["first_api_call_start_time"] = real_datetime.datetime.now()
    logging_obj.model_call_details["messages"] = ONE_USER_MESSAGE
    logging_obj.model_call_details["call_type"] = "acompletion"
    return logging_obj


def test_the_deployment_rate_is_found_on_the_logging_objects_real_litellm_params(monkeypatch):
    """``get_litellm_params()`` always carries a ``litellm_metadata`` key (None on chat
    completions) next to the router-stamped ``metadata.model_info.id``; the failure row must
    still price at the deployment's own rate, like ``Logging.get_router_model_id`` on the
    success path."""
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.proxy.utils import _failure_fields_to_lift

    monkeypatch.setitem(
        litellm.model_cost,
        "dep-lit145-real-shape",
        {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 0.25, "output_cost_per_token": 0.25},
    )
    litellm_params = get_litellm_params(metadata={"model_info": {"id": "dep-lit145-real-shape"}})
    assert "litellm_metadata" in litellm_params and litellm_params["litellm_metadata"] is None
    logging_obj = _dispatched_failure_logging_obj(FAILURE_USAGE_MODEL, litellm_params)

    lifted = _failure_fields_to_lift({"litellm_logging_obj": logging_obj}, provider_billed=True)

    assert lifted["response_cost"] == pytest.approx(lifted["combined_usage_object"].prompt_tokens * 0.25)


def test_an_azure_deployment_is_priced_at_its_base_model_like_the_success_path():
    """An Azure deployment name is not in the cost map; the success path prices it through
    ``model_info.base_model`` (``get_base_model_from_metadata``) and so must the failure row."""
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.proxy.utils import _failure_fields_to_lift

    logging_obj = _dispatched_failure_logging_obj(
        "azure/lit145-not-in-cost-map",
        get_litellm_params(metadata={"model_info": {"base_model": "azure/gpt-4.1"}}),
    )
    logging_obj.model_call_details["custom_llm_provider"] = "azure"

    lifted = _failure_fields_to_lift({"litellm_logging_obj": logging_obj}, provider_billed=True)

    expected_cost, _ = litellm.cost_per_token(
        model="azure/gpt-4.1", prompt_tokens=lifted["combined_usage_object"].prompt_tokens
    )
    assert expected_cost > 0
    assert lifted["response_cost"] == pytest.approx(expected_cost)


def test_an_unmapped_base_model_falls_back_to_the_deployment_model_like_the_success_path():
    """``completion_cost`` tries ``base_model`` first and then the deployment model, so a
    ``model_info.base_model`` the cost map does not know still prices a success row at the
    deployment model's rate. The failure row must land on the same number instead of $0."""
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.proxy.utils import _failure_fields_to_lift

    logging_obj = _dispatched_failure_logging_obj(
        FAILURE_USAGE_MODEL,
        get_litellm_params(metadata={"model_info": {"base_model": "not-in-the-cost-map"}}),
    )
    logging_obj.model_call_details["custom_llm_provider"] = "openai"

    lifted = _failure_fields_to_lift({"litellm_logging_obj": logging_obj}, provider_billed=True)

    prompt_tokens = lifted["combined_usage_object"].prompt_tokens
    success_cost = litellm.completion_cost(
        completion_response=ModelResponse(
            model=FAILURE_USAGE_MODEL, usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=0)
        ),
        model=FAILURE_USAGE_MODEL,
        custom_llm_provider="openai",
        base_model="not-in-the-cost-map",
    )
    assert success_cost > 0
    assert lifted["response_cost"] == pytest.approx(success_cost)


def test_a_litellm_params_key_in_the_request_body_cannot_zero_the_timeout_price():
    """``model_call_details`` is flattened with the request's optional_params, so a body key
    named ``litellm_params`` can land there; the deployment rate must come from the logging
    object's own ``litellm_params`` instead."""
    from litellm.proxy.utils import _failure_usage_to_lift

    lifted = _failure_usage_to_lift(
        model_call_details={
            "call_type": "acompletion",
            "model": FAILURE_USAGE_MODEL,
            "messages": ONE_USER_MESSAGE,
            "litellm_params": {"input_cost_per_token": 0},
        },
        request_body={},
        dispatched=True,
        provider_billed=True,
        deployment_id=None,
    )

    assert lifted is not None
    usage, response_cost = lifted
    expected_cost, _ = litellm.cost_per_token(model=FAILURE_USAGE_MODEL, prompt_tokens=usage.prompt_tokens)
    assert response_cost == pytest.approx(expected_cost)
    assert response_cost > 0


def test_a_request_that_timed_out_at_a_provider_bills_its_input_at_the_input_rate():
    from litellm.proxy.utils import _failure_usage_to_lift

    lifted = _failure_usage_to_lift(
        model_call_details={
            "call_type": "acompletion",
            "model": FAILURE_USAGE_MODEL,
            "messages": ONE_USER_MESSAGE,
        },
        request_body={},
        dispatched=True,
        provider_billed=True,
    )

    assert lifted is not None
    usage, response_cost = lifted
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens == 0
    expected_input_cost, _ = litellm.cost_per_token(model=FAILURE_USAGE_MODEL, prompt_tokens=usage.prompt_tokens)
    assert expected_input_cost > 0
    assert response_cost == pytest.approx(expected_input_cost)



COMPACTION_INSTRUCTIONS = {"role": "system", "content": "answer briefly"}
COMPACTED_HISTORY = [
    {"role": "user", "content": "a long turn " * 200},
    {"role": "assistant", "content": "a long reply " * 200},
]
COMPACTION_ITEM = {
    "type": "compaction",
    "encrypted_content": json.dumps({"type": "compaction", "content": "the cat is named Biscuit"}),
}
TURN_AFTER_COMPACTION = {"role": "user", "content": "what is the cat's name?"}


def _failed_call_prompt_tokens(call_type, dispatched_call_type, request_input):
    from litellm.proxy.utils import _failure_usage_to_lift

    lifted = _failure_usage_to_lift(
        model_call_details={"call_type": call_type, "model": FAILURE_USAGE_MODEL, "messages": request_input},
        request_body={},
        dispatched=True,
        dispatched_call_type=dispatched_call_type,
    )
    assert lifted is not None
    return lifted[0].prompt_tokens


@pytest.mark.parametrize(
    "call_type, dispatched_call_type",
    [
        ("acompletion", "acompletion"),
        ("aanthropic_messages", "anthropic_messages"),
        ("responses", "acompletion"),
    ],
)
def test_a_failed_call_off_the_responses_route_counts_the_history_before_a_compaction_typed_message(
    call_type, dispatched_call_type
):
    def prompt_tokens(request_input):
        return _failed_call_prompt_tokens(call_type, dispatched_call_type, request_input)

    sent = prompt_tokens([COMPACTION_INSTRUCTIONS, *COMPACTED_HISTORY, COMPACTION_ITEM, TURN_AFTER_COMPACTION])

    assert sent > prompt_tokens([COMPACTION_INSTRUCTIONS, COMPACTION_ITEM, TURN_AFTER_COMPACTION])


def test_an_earlier_timed_out_responses_attempt_counts_its_input_from_the_newest_compaction_item():
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params
    from litellm.proxy.utils import _failure_fields_to_lift

    def earlier_attempt_prompt_tokens(request_input):
        logging_obj = _dispatched_failure_logging_obj(
            FAILURE_USAGE_MODEL,
            get_litellm_params(
                metadata={
                    "attempted_retries": 1,
                    "model_info": {"id": "dep-compaction-final"},
                    "previous_models": (
                        {"deployment_id": "dep-compaction-final", "exception_type": "Timeout", "attempted_retries": 0},
                    ),
                }
            ),
        )
        logging_obj.model_call_details["call_type"] = "aresponses"
        logging_obj.model_call_details["messages"] = request_input
        lifted = _failure_fields_to_lift(
            {"litellm_logging_obj": logging_obj}, provider_billed=True, dispatched_call_type="aresponses"
        )
        (earlier_attempt,) = lifted["billed_failed_attempts"]
        return earlier_attempt.usage.prompt_tokens

    sent = earlier_attempt_prompt_tokens(
        [COMPACTION_INSTRUCTIONS, *COMPACTED_HISTORY, COMPACTION_ITEM, TURN_AFTER_COMPACTION]
    )

    assert sent == earlier_attempt_prompt_tokens([COMPACTION_INSTRUCTIONS, COMPACTION_ITEM, TURN_AFTER_COMPACTION])


@pytest.mark.parametrize(
    "current_attempt_api_call_start_time",
    [None, 1700000001.0],
    ids=["retry-not-dispatched", "retry-dispatched"],
)
def test_a_retry_failure_bills_input_only_for_dispatched_attempts(
    current_attempt_api_call_start_time: float | None,
) -> None:
    from litellm.proxy.utils import _estimate_dispatched_failure_usage, _failure_fields_to_lift
    from litellm.types.utils import Usage

    prior_usage: Final = Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6)
    estimated_retry_usage: Final = _estimate_dispatched_failure_usage(FAILURE_USAGE_MODEL, ONE_USER_MESSAGE, None)
    assert estimated_retry_usage is not None

    lifted: Final = _failure_fields_to_lift(
        {
            "litellm_logging_obj": _LoggingObj(
                {
                    "first_api_call_start_time": 1700000000.0,
                    "current_attempt_api_call_start_time": current_attempt_api_call_start_time,
                    "failed_attempts_usage": ((prior_usage, 0.25),),
                    "call_type": "acompletion",
                    "model": FAILURE_USAGE_MODEL,
                    "messages": ONE_USER_MESSAGE,
                }
            )
        }
    )

    usage: Final = lifted["combined_usage_object"]
    assert isinstance(usage, Usage)
    expected_retry_tokens: Final = (
        estimated_retry_usage.prompt_tokens if current_attempt_api_call_start_time is not None else 0
    )
    assert usage.prompt_tokens == prior_usage.prompt_tokens + expected_retry_tokens
    assert usage.completion_tokens == prior_usage.completion_tokens
    assert usage.total_tokens == prior_usage.total_tokens + expected_retry_tokens
    assert lifted["response_cost"] == 0.25


@pytest.mark.parametrize(
    "model_call_details, dispatched",
    [
        ({"call_type": "acompletion", "model": FAILURE_USAGE_MODEL, "messages": ONE_USER_MESSAGE}, False),
        (
            {
                "litellm_no_upstream_llm_call": True,
                "call_type": "acompletion",
                "model": FAILURE_USAGE_MODEL,
                "messages": ONE_USER_MESSAGE,
            },
            True,
        ),
        ({"call_type": "afile_content", "model": FAILURE_USAGE_MODEL, "messages": ONE_USER_MESSAGE}, True),
    ],
    ids=["never dispatched", "no upstream call", "call type has no input to price"],
)
def test_a_failure_that_cost_the_provider_nothing_lifts_nothing(model_call_details, dispatched):
    from litellm.proxy.utils import _failure_usage_to_lift

    assert _failure_usage_to_lift(model_call_details=model_call_details, request_body={}, dispatched=dispatched) is None


def test_the_no_upstream_call_key_the_module_uses_is_the_one_asserted_above():
    from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL

    assert LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL == "litellm_no_upstream_llm_call"


def test_the_dispatched_system_prompt_wins_over_the_one_in_the_request_body():
    from litellm.proxy.utils import _failure_usage_to_lift

    def lift(model_call_details, request_body):
        lifted = _failure_usage_to_lift(
            model_call_details=model_call_details, request_body=request_body, dispatched=True
        )
        assert lifted is not None
        return lifted[0].prompt_tokens

    base = {
        "call_type": "aanthropic_messages",
        "model": FAILURE_USAGE_MODEL,
        "messages": ONE_USER_MESSAGE,
    }
    long_system = "answer as briefly as you possibly can, in one short sentence"

    from_body = lift(base, {"system": long_system})
    from_params = lift({**base, "optional_params": {"system": "x"}}, {"system": long_system})
    body_only_short = lift(base, {"system": "x"})

    assert from_body > body_only_short
    assert from_params == body_only_short


def test_a_failure_with_no_logging_object_lifts_nothing():
    from litellm.proxy.utils import _failure_fields_to_lift

    assert dict(_failure_fields_to_lift({})) == {}
    assert dict(_failure_fields_to_lift({"litellm_logging_obj": _LoggingObj({})})) == {}


def test_a_dispatched_failure_lifts_the_fields_the_spend_log_needs():
    from litellm.proxy.utils import _failure_fields_to_lift

    lifted = _failure_fields_to_lift(
        {
            "litellm_logging_obj": _LoggingObj(
                {
                    "start_time": 1699999999.0,
                    "first_api_call_start_time": 1700000000.0,
                    "call_type": "acompletion",
                    "model": FAILURE_USAGE_MODEL,
                    "messages": ONE_USER_MESSAGE,
                    "standard_logging_object": {"id": "log-1"},
                }
            )
        }
    )

    assert set(lifted) == {
        "start_time",
        "first_api_call_start_time",
        "call_type",
        "combined_usage_object",
        "response_cost",
        "standard_logging_object",
    }
    assert lifted["start_time"] == 1699999999.0
    assert lifted["first_api_call_start_time"] == 1700000000.0
    assert lifted["call_type"] == "acompletion"
    assert lifted["response_cost"] == 0.0
    assert lifted["combined_usage_object"].prompt_tokens > 0
    assert lifted["standard_logging_object"] == {"id": "log-1"}


@pytest.mark.asyncio
async def test_a_dispatched_failure_is_counted_off_the_event_loop():
    from unittest.mock import AsyncMock, patch

    from tests.large_text import text
    from tests.unit.litellm_core_utils.event_loop_lag import (
        assert_loop_stayed_free,
        timed_with_loop_lags,
        warm_tokenizer,
    )

    warm_tokenizer("claude-fable-5")
    request_data = {
        "litellm_logging_obj": _LoggingObj(
            {
                "first_api_call_start_time": 1700000000.0,
                "call_type": "acompletion",
                "model": "claude-fable-5",
                "messages": [{"role": "user", "content": text * 100}],
            }
        ),
        "metadata": {},
    }
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    proxy_logging_obj.alert_types = []
    with patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()):
        _, took, lags = await timed_with_loop_lags(
            lambda: proxy_logging_obj.post_call_failure_hook(
                request_data=request_data,
                original_exception=Exception("boom"),
                user_api_key_dict=UserAPIKeyAuth(),
            )
        )

    assert request_data["combined_usage_object"].prompt_tokens > 0
    assert_loop_stayed_free(took, lags)


@pytest.mark.asyncio
async def test_proxy_only_error_expected_4xx_skips_traceback_for_both_handlers(monkeypatch):
    """Regression for LIT-6043: an expected 4xx must not format a traceback for
    either the async or the threaded sync failure handler."""
    import asyncio

    import litellm
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy._types import UserAPIKeyAuth

    monkeypatch.setattr(litellm, "failure_callback", [])
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    captured = {}
    sync_ran = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def fake_async_failure(self, exception, traceback_exception, *args, **kwargs):
        captured["async_traceback"] = traceback_exception

    def fake_sync_failure(self, exception, traceback_exception, *args, **kwargs):
        captured["sync_traceback"] = traceback_exception
        loop.call_soon_threadsafe(sync_ran.set)

    orig_async_failure = Logging.async_failure_handler
    orig_sync_failure = Logging.failure_handler
    Logging.async_failure_handler = fake_async_failure
    Logging.failure_handler = fake_sync_failure
    try:
        try:
            raise HTTPException(status_code=400, detail="Invalid model name passed in")
        except HTTPException as exc:
            await proxy_logging_obj._handle_logging_proxy_only_error(
                request_data={
                    "model": "does-not-exist",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-9876", request_route="/v1/chat/completions"),
                route="/v1/chat/completions",
                original_exception=exc,
            )
        await asyncio.wait_for(sync_ran.wait(), timeout=5)
    finally:
        Logging.async_failure_handler = orig_async_failure
        Logging.failure_handler = orig_sync_failure

    assert captured["async_traceback"] == ""
    assert captured["sync_traceback"] == ""


@pytest.mark.asyncio
async def test_proxy_only_error_5xx_keeps_traceback_and_runs_sync_callbacks(monkeypatch):
    """Unexpected (5xx) errors keep the full traceback, and a configured
    sync-only failure callback still gets its threaded handler."""
    import asyncio

    import litellm
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy._types import UserAPIKeyAuth

    def _custom_sync_callback(kwargs, completion_response, start_time, end_time):
        pass

    monkeypatch.setattr(litellm, "failure_callback", [_custom_sync_callback])
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    captured = {}
    sync_ran = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def fake_async_failure(self, exception, traceback_exception, *args, **kwargs):
        captured["async_traceback"] = traceback_exception

    def fake_sync_failure(self, *args, **kwargs):
        loop.call_soon_threadsafe(sync_ran.set)

    orig_async_failure = Logging.async_failure_handler
    orig_sync_failure = Logging.failure_handler
    Logging.async_failure_handler = fake_async_failure
    Logging.failure_handler = fake_sync_failure
    try:
        try:
            raise HTTPException(status_code=500, detail="internal error")
        except HTTPException as exc:
            await proxy_logging_obj._handle_logging_proxy_only_error(
                request_data={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-9876", request_route="/v1/chat/completions"),
                route="/v1/chat/completions",
                original_exception=exc,
            )
        await asyncio.wait_for(sync_ran.wait(), timeout=5)
    finally:
        Logging.async_failure_handler = orig_async_failure
        Logging.failure_handler = orig_sync_failure

    assert "test_proxy_utils" in captured["async_traceback"]


def test_create_model_info_response_resolves_mode_through_deployment_model():
    """`mode` is derived from the same lookup, so an aliased embedding deployment
    currently reports no mode at all; it must report `embedding`."""
    from litellm import Router

    saved_model_cost = dict(litellm.model_cost)
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "my-embeddings",
                    "litellm_params": {"model": "openai/text-embedding-3-small"},
                }
            ]
        )

        response = create_model_info_response(
            model_id="my-embeddings", provider="openai", llm_router=router
        )
    finally:
        litellm.model_cost.clear()
        litellm.model_cost.update(saved_model_cost)

    assert response["mode"] == "embedding"


@pytest.mark.parametrize(
    "model_group_alias",
    [
        {"team-embeddings": "my-embeddings"},
        {"team-embeddings": {"model": "my-embeddings", "hidden": False}},
    ],
)
def test_create_model_info_response_resolves_model_group_alias_to_target(model_group_alias, local_model_cost_map):
    """A `model_group_alias` row must report the metadata of the group it points at,
    not the cost-map generalization or nothing that the alias name resolves to."""
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "my-embeddings",
                "litellm_params": {"model": "openai/text-embedding-3-small"},
            }
        ],
        model_group_alias=model_group_alias,
    )

    alias_response = create_model_info_response(
        model_id="team-embeddings", provider="openai", llm_router=router
    )
    target_response = create_model_info_response(
        model_id="my-embeddings", provider="openai", llm_router=router
    )

    assert alias_response["id"] == "team-embeddings"
    for field in ("mode", "max_input_tokens", "max_output_tokens"):
        assert alias_response.get(field) == target_response.get(field)
    assert alias_response["mode"] == "embedding"


@pytest.mark.parametrize(
    "key_metadata, team_metadata, expected_to_run",
    [
        ({"guardrails": ["key-scoped-guardrail"]}, None, True),
        ({}, {"guardrails": ["key-scoped-guardrail"]}, True),
        ({"guardrails": ["some-other-guardrail"]}, None, False),
        ({}, None, False),
    ],
)
def test_convert_mcp_to_llm_format_carries_key_and_team_guardrails(key_metadata, team_metadata, expected_to_run):
    from litellm.responses.mcp.request_context import MCPRequestContext

    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    guardrail = CustomGuardrail(guardrail_name="key-scoped-guardrail", event_hook="pre_mcp_call", default_on=False)
    kwargs = {
        "name": "ask_question",
        "arguments": {"question": "hello"},
        "server_name": "deepwiki",
        "guardrail_context": MCPRequestContext.resolve_guardrail_context({"guardrails": ["parent-rule"]}),
        "user_api_key_auth": UserAPIKeyAuth(metadata=key_metadata, team_metadata=team_metadata),
    }
    request_obj = proxy_logging._create_mcp_request_object_from_kwargs(kwargs)

    with patch(  # test-quality-ok: the key-guardrail premium gate reads this proxy_server module global and has no injection seam
        "litellm.proxy.proxy_server.premium_user", True
    ):
        synthetic = proxy_logging._convert_mcp_to_llm_format(request_obj, kwargs)

    assert guardrail.should_run_guardrail(synthetic, GuardrailEventHooks.pre_mcp_call) is expected_to_run

    assert "parent-rule" in synthetic["metadata"]["guardrails"]


class _TracebackRecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.received_traceback: str | None = None

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> HTTPException | None:
        self.received_traceback = traceback_str
        return None


@pytest.mark.asyncio
async def test_post_call_failure_hook_redacts_traceback_before_callbacks(monkeypatch):
    """A pass-through upstream failure hands the hook the httpx traceback, whose
    message quotes the upstream URL with the provider key in its query string.
    Every callback, custom loggers included, must receive it redacted."""
    import traceback
    from unittest.mock import AsyncMock, patch

    import httpx

    provider_key = "AIza" + "S" * 35
    upstream_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent?key={provider_key}"
    response = httpx.Response(400, request=httpx.Request("POST", upstream_url))
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        upstream_traceback = traceback.format_exc()
    assert provider_key in upstream_traceback

    recorder = _TracebackRecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    proxy_logging_obj.alert_types = []
    with patch.object(proxy_logging_obj, "update_request_status", new=AsyncMock()):
        await proxy_logging_obj.post_call_failure_hook(
            request_data={"metadata": {}},
            original_exception=HTTPException(status_code=400, detail="Upstream passthrough request failed with status 400"),
            user_api_key_dict=UserAPIKeyAuth(),
            traceback_str=upstream_traceback,
        )

    assert recorder.received_traceback is not None
    assert provider_key not in recorder.received_traceback
    assert "REDACTED" in recorder.received_traceback


class TestPrismaClientTokenAuthBehindThePool:
    """Behind the in-container pool the supervisor renews the writer's database
    token and hands the workers a loopback URL with a static password, so the
    writer wrapper must not run its own refresh loop. The reader is not pooled
    and keeps refreshing its own token."""

    UPSTREAM: Final = "postgresql://litellm:TOKEN@db.internal:5432/litellm"
    READER: Final = "postgresql://litellm:TOKEN@reader.internal:5432/litellm"

    def _client(self, monkeypatch: pytest.MonkeyPatch, pooled: bool) -> PrismaClient:
        from litellm.proxy.db.pgbouncer import PGBOUNCER_POOLED_ENV_VAR

        monkeypatch.delenv("AZURE_POSTGRESQL_AUTH", raising=False)
        monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
        monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
        monkeypatch.setenv("DATABASE_URL", self.UPSTREAM)
        monkeypatch.setenv("DATABASE_URL_READ_REPLICA", self.READER)
        if pooled:
            monkeypatch.setenv(PGBOUNCER_POOLED_ENV_VAR, "true")
        else:
            monkeypatch.delenv(PGBOUNCER_POOLED_ENV_VAR, raising=False)
        rds: Final = MagicMock()
        rds.generate_db_auth_token.return_value = "TOKEN"
        with patch("boto3.client", return_value=rds):
            return PrismaClient(database_url=self.UPSTREAM, proxy_logging_obj=MagicMock(spec=ProxyLogging))

    def test_a_pooled_writer_leaves_token_refresh_to_the_pooler_while_the_reader_keeps_its_own(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

        client = self._client(monkeypatch, pooled=True)
        assert isinstance(client.db, RoutingPrismaWrapper)
        assert client.db.writer.iam_token_db_auth is False
        assert client.db.reader.iam_token_db_auth is True
        assert client.token_auth is not None

    def test_an_unpooled_writer_still_refreshes_its_own_token(self, monkeypatch: pytest.MonkeyPatch):
        from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

        client = self._client(monkeypatch, pooled=False)
        assert isinstance(client.db, RoutingPrismaWrapper)
        assert client.db.writer.iam_token_db_auth is True
        assert client.db.reader.iam_token_db_auth is True


@pytest.mark.parametrize("bucket", ["metadata", "litellm_metadata"])
def test_mcp_conversion_preserves_request_policy_and_isolates_guardrail_data(bucket):
    from copy import deepcopy
    from litellm.responses.mcp.request_context import MCPRequestContext

    parent = {
        "model": "parent-model",
        bucket: {
            "guardrails": ["policy-rule"], "guardrail_config": {"language": "en"},
            "applied_policies": ["parent-policy"], "policy_sources": {"parent-policy": "model"},
            "_guardrail_pipelines": [], "_pipeline_managed_guardrails": ["pipeline-rule"], "tags": ["review"],
        },
        "guardrails": [{"request-rule": {"extra_body": {"threshold": 0.9}}}],
        "guardrail_config": {"entities": ["EMAIL_ADDRESS"]},
    }
    original = deepcopy(parent)
    context = MCPRequestContext.resolve(kwargs=parent, tools=None)
    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    kwargs = {"name": "execute", "arguments": {"text": "hello"}, "guardrail_context": context.guardrail_context}
    request_obj = proxy_logging._create_mcp_request_object_from_kwargs(kwargs)
    first = proxy_logging._convert_mcp_to_llm_format(request_obj, kwargs)
    assert first["model"] == "parent-model"
    assert first["metadata"]["guardrails"] == ["policy-rule", {"request-rule": {"extra_body": {"threshold": 0.9}}}]
    assert first["metadata"]["guardrail_config"] == {"language": "en", "entities": ["EMAIL_ADDRESS"]}
    assert first["metadata"]["applied_policies"] == ["parent-policy"]
    assert first["metadata"]["policy_sources"] == {"parent-policy": "model"}
    assert first["metadata"]["_pipeline_managed_guardrails"] == ["pipeline-rule"]
    first["metadata"]["guardrails"].clear()
    first["metadata"]["guardrail_config"]["entities"].clear()
    assert parent == original
    second = proxy_logging._convert_mcp_to_llm_format(request_obj, kwargs)
    assert second["metadata"]["guardrails"] == ["policy-rule", {"request-rule": {"extra_body": {"threshold": 0.9}}}]
    assert second["metadata"]["guardrail_config"]["entities"] == ["EMAIL_ADDRESS"]


@pytest.mark.parametrize("opt_out", [False, True])
def test_mcp_conversion_honors_only_authenticated_global_guardrail_opt_outs(opt_out):
    from litellm.responses.mcp.request_context import MCPRequestContext

    auth = UserAPIKeyAuth(metadata={"opted_out_global_guardrails": ["global-rule"] if opt_out else []})
    context = MCPRequestContext.resolve(kwargs={"metadata": {
        "user_api_key_auth": auth, "disable_global_guardrails": True,
        "user_api_key_metadata": {"disable_global_guardrails": True},
    }}, tools=None)
    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    kwargs = {"name": "execute", "arguments": {}, "user_api_key_auth": auth, "guardrail_context": context.guardrail_context}
    synthetic = proxy_logging._convert_mcp_to_llm_format(proxy_logging._create_mcp_request_object_from_kwargs(kwargs), kwargs)
    guardrail = CustomGuardrail(guardrail_name="global-rule", event_hook="pre_mcp_call", default_on=True)
    assert guardrail.should_run_guardrail(synthetic, GuardrailEventHooks.pre_mcp_call) is (not opt_out)
    synthetic["metadata"]["user_api_key_metadata"]["opted_out_global_guardrails"].append("unrelated")
    assert auth.metadata == {"opted_out_global_guardrails": ["global-rule"] if opt_out else []}


@pytest.mark.parametrize("model, expected", [("parent-model", True), ("unmatched-model", False)])
def test_mcp_auth_policy_uses_original_request_model(monkeypatch, model, expected):
    from litellm.responses.mcp.request_context import MCPRequestContext
    from litellm.proxy.policy_engine import policy_registry
    from litellm.types.proxy.policy_engine import Policy, PolicyCondition, PolicyGuardrails

    registry = policy_registry.PolicyRegistry()
    registry._policies = {"model-policy": Policy(
        condition=PolicyCondition(model="parent-model"), guardrails=PolicyGuardrails(add=["model-rule"])
    )}
    registry._initialized = True
    monkeypatch.setattr(policy_registry, "_policy_registry", registry)
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    kwargs = {
        "name": "execute", "arguments": {},
        "user_api_key_auth": UserAPIKeyAuth(metadata={"policies": ["model-policy"]}),
        "guardrail_context": MCPRequestContext.resolve_guardrail_context({"model": model, "guardrails": ["request-rule"]}),
    }
    synthetic = proxy_logging._convert_mcp_to_llm_format(proxy_logging._create_mcp_request_object_from_kwargs(kwargs), kwargs)
    assert ("model-rule" in synthetic["metadata"]["guardrails"]) is expected
    assert "request-rule" in synthetic["metadata"]["guardrails"]


def test_handle_exception_on_proxy_logs_bug_report_only_for_unmapped_500(caplog):
    with caplog.at_level("ERROR", logger="LiteLLM Proxy"):
        provider_result = handle_exception_on_proxy(
            InternalServerError(message="upstream 500", llm_provider="openai", model="gpt-4")
        )
        assert ISSUE_URL_BASE not in caplog.text
        internal_result = handle_exception_on_proxy(KeyError("missing"))

    assert provider_result.code == internal_result.code == "500"
    assert ISSUE_URL_BASE in caplog.text
    assert ISSUE_URL_BASE not in internal_result.message
