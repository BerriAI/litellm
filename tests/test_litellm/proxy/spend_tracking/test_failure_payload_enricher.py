"""
Tests for ``litellm.proxy.spend_tracking.failure_payload_enricher``.

These cover the streaming-cancel observability fix: when a request fails
(especially a streaming cancel), the enricher pulls identification,
upstream-attribution, and timing data from the Logging object so the spend
log row carries the same fields a success row would.
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from litellm.proxy.spend_tracking.failure_payload_enricher import (
    enrich_failure_request_data,
    resolve_failure_start_time,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logging_obj(model_call_details=None, start_time=None, trace_id=None):
    """Build a stand-in for the litellm Logging object with the attributes
    ``get_standard_logging_object_payload`` reads off ``logging_obj``."""
    return SimpleNamespace(
        model_call_details=model_call_details or {},
        start_time=start_time,
        litellm_trace_id=trace_id,
        completion_start_time=None,
        cost_breakdown=None,
        _response_cost_calculator=lambda **_: 0.0,
    )


def _full_model_call_details():
    """A model_call_details dict shaped like a real streaming request that
    has reached the upstream call before being cancelled."""
    return {
        "model": "kimi-k2-6-dev",
        "custom_llm_provider": "hosted_vllm",
        "call_type": "acompletion",
        "stream": True,
        "litellm_call_id": "call-123",
        "litellm_trace_id": "trace-abc",
        "start_time": datetime(2026, 5, 21, 18, 30, 0),
        "litellm_params": {
            "api_base": "http://103.48.43.99:8000/v1",
            "model": "kimi-k2-6-dev",
            "custom_llm_provider": "hosted_vllm",
            "proxy_server_request": {
                "headers": {"user-agent": "claude-cli/2.1.119"},
                "body": {"model": "open-large"},
                "url": "http://litellm/v1/chat/completions",
            },
            "metadata": {
                "model_group": "open-large",
                "model_info": {"id": "deployment-uuid-1"},
                "deployment": "kimi-k2-6-dev",
                "requester_ip_address": "10.7.1.144",
                "user_agent": "claude-cli/2.1.119",
                "tags": ["User-Agent: claude-cli", "team-internal"],
                "user_api_key_alias": "alias-prod",
                "user_api_key_hash": "hash-deadbeef",
                "user_api_key_user_id": "user-1",
                "user_api_key_team_id": "team-1",
                "user_api_key_org_id": "org-1",
                "endpoint": "/v1/chat/completions",
            },
        },
    }


# ---------------------------------------------------------------------------
# enrich_failure_request_data — direct unit tests
# ---------------------------------------------------------------------------


def test_enricher_promotes_upstream_attribution_fields():
    """The most important assertion: api_base, model_group, model_id all land
    on request_data after enrichment of a streaming-cancel-shaped failure."""
    request_data = {"metadata": {}}
    logging_obj = _make_logging_obj(model_call_details=_full_model_call_details())

    enrich_failure_request_data(
        request_data=request_data,
        litellm_logging_obj=logging_obj,
        original_exception=asyncio.CancelledError(),
    )

    assert request_data["model"] == "kimi-k2-6-dev"
    assert request_data["custom_llm_provider"] == "hosted_vllm"
    assert request_data["litellm_params"]["api_base"] == "http://103.48.43.99:8000/v1"
    assert request_data["litellm_params"]["model"] == "kimi-k2-6-dev"

    # Identification metadata must end up on BOTH top-level and litellm_params
    # so the hook's subsequent metadata-rebuild preserves it.
    for md in (
        request_data["metadata"],
        request_data["litellm_params"]["metadata"],
    ):
        assert md["model_group"] == "open-large"
        assert md["model_info"]["id"] == "deployment-uuid-1"
        assert md["requester_ip_address"] == "10.7.1.144"
        assert md["tags"] == ["User-Agent: claude-cli", "team-internal"]
        assert md["user_api_key_alias"] == "alias-prod"


def test_enricher_builds_standard_logging_object_when_missing():
    """Building the SLO is what unlocks the rich metadata JSON column on the
    spend log row. The enricher must do this when neither request_data nor
    model_call_details already has one."""
    request_data = {}
    logging_obj = _make_logging_obj(model_call_details=_full_model_call_details())

    enrich_failure_request_data(
        request_data=request_data,
        litellm_logging_obj=logging_obj,
        original_exception=Exception("boom"),
    )

    slp = request_data.get("standard_logging_object")
    assert slp is not None
    assert slp.get("status") == "failure"
    # Some core fields the SLO must carry — these flow into the
    # spend log payload constructor.
    assert slp.get("model_group") == "open-large" or slp.get("model") == "kimi-k2-6-dev"


def test_enricher_skips_slo_build_when_already_present():
    """If something else has already built an SLO (e.g. core
    async_failure_handler ran first), the enricher must not clobber it."""
    existing_slo = {"status": "failure", "id": "pre-built"}
    request_data = {"standard_logging_object": existing_slo}
    logging_obj = _make_logging_obj(model_call_details=_full_model_call_details())

    enrich_failure_request_data(
        request_data=request_data,
        litellm_logging_obj=logging_obj,
        original_exception=asyncio.CancelledError(),
    )

    assert request_data["standard_logging_object"] is existing_slo


def test_enricher_is_idempotent_does_not_overwrite_real_values():
    """If request_data already has a real value, the enricher must leave it
    alone (idempotency). Empty/None values are still filled in."""
    request_data = {
        "model": "user-provided-model",
        "litellm_params": {
            "api_base": "http://existing-base/v1",
            "metadata": {"model_group": "user-set-group"},
        },
        "metadata": {"model_group": "user-set-group"},
    }
    logging_obj = _make_logging_obj(model_call_details=_full_model_call_details())

    enrich_failure_request_data(
        request_data=request_data,
        litellm_logging_obj=logging_obj,
        original_exception=Exception("err"),
    )

    # Existing real values must survive.
    assert request_data["model"] == "user-provided-model"
    assert (
        request_data["litellm_params"]["api_base"] == "http://existing-base/v1"
    )
    assert request_data["metadata"]["model_group"] == "user-set-group"
    # But blanks must be filled.
    assert request_data["custom_llm_provider"] == "hosted_vllm"
    assert request_data["metadata"]["requester_ip_address"] == "10.7.1.144"


def test_enricher_handles_missing_logging_obj():
    """``litellm_logging_obj=None`` is the pre-router-failure case. Enricher
    must not crash and must leave the request_data unchanged."""
    request_data = {"model": "x", "metadata": {"k": "v"}}
    enrich_failure_request_data(
        request_data=request_data,
        litellm_logging_obj=None,
        original_exception=Exception("err"),
    )
    assert request_data == {"model": "x", "metadata": {"k": "v"}}


def test_enricher_handles_empty_model_call_details():
    """Logging obj exists but model_call_details is empty (very early failure
    where nothing was logged yet). Should not crash; should be a no-op."""
    request_data = {"model": "x"}
    logging_obj = _make_logging_obj(model_call_details={})
    enrich_failure_request_data(
        request_data=request_data,
        litellm_logging_obj=logging_obj,
        original_exception=Exception("err"),
    )
    # api_base / model_group remain unset (acceptable for very early
    # failures — we don't have the data).
    assert "api_base" not in request_data.get("litellm_params", {})
    assert request_data["model"] == "x"


def test_enricher_swallows_internal_errors_and_returns_request_data():
    """Any unexpected exception inside the enricher must not propagate;
    enrichment failures must never block the spend log write."""

    class _Booby:
        @property
        def model_call_details(self):
            raise RuntimeError("Boom")

    request_data = {"model": "x"}
    result = enrich_failure_request_data(
        request_data=request_data,
        litellm_logging_obj=_Booby(),
        original_exception=Exception("err"),
    )
    assert result is request_data


# ---------------------------------------------------------------------------
# resolve_failure_start_time
# ---------------------------------------------------------------------------


def test_resolve_failure_start_time_prefers_logging_obj_attr():
    real_start = datetime(2026, 5, 21, 18, 0, 0)
    logging_obj = _make_logging_obj(start_time=real_start)
    assert resolve_failure_start_time(logging_obj) == real_start


def test_resolve_failure_start_time_falls_back_to_model_call_details():
    real_start = datetime(2026, 5, 21, 18, 5, 0)
    logging_obj = _make_logging_obj(
        model_call_details={"start_time": real_start},
        start_time=None,
    )
    assert resolve_failure_start_time(logging_obj) == real_start


def test_resolve_failure_start_time_default_is_now():
    before = datetime.now()
    got = resolve_failure_start_time(litellm_logging_obj=None)
    after = datetime.now()
    assert before <= got <= after + timedelta(seconds=1)


# ---------------------------------------------------------------------------
# End-to-end through ``async_post_call_failure_hook``
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_hook_persists_upstream_attribution_on_streaming_cancel():
    """Simulate the streaming-cancel path: the hook is called with a
    Logging object whose model_call_details is fully populated. After the
    hook runs, the kwargs passed to ``update_database`` must carry api_base,
    model_group, model_id, requester_ip_address, tags, and a real start_time."""
    logger = _ProxyDBLogger()

    real_start = datetime(2026, 5, 21, 18, 30, 0)
    logging_obj = _make_logging_obj(
        model_call_details=_full_model_call_details(),
        start_time=real_start,
        trace_id="trace-abc",
    )

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        key_alias="alias-prod",
        user_id="user-1",
        team_id="team-1",
        org_id="org-1",
        end_user_id="end-user-1",
    )

    request_data = {
        "model": "open-large",  # user-facing alias
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {},
        "litellm_logging_obj": logging_obj,
    }

    with patch(
        "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
        new_callable=AsyncMock,
    ) as mock_update_database:
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=asyncio.CancelledError(),
            user_api_key_dict=user_api_key_dict,
        )

    mock_update_database.assert_called_once()
    call_args = mock_update_database.call_args[1]
    kwargs = call_args["kwargs"]

    # Upstream attribution — the headline assertion.
    assert kwargs["litellm_params"]["api_base"] == "http://103.48.43.99:8000/v1"
    assert kwargs["litellm_params"]["model"] == "kimi-k2-6-dev"
    assert kwargs["custom_llm_provider"] == "hosted_vllm"

    # Identification metadata propagated to litellm_params.metadata
    # (which is what get_logging_payload reads).
    final_md = kwargs["litellm_params"]["metadata"]
    assert final_md["model_group"] == "open-large"
    assert final_md["model_info"]["id"] == "deployment-uuid-1"
    assert final_md["requester_ip_address"] == "10.7.1.144"
    assert final_md["tags"] == ["User-Agent: claude-cli", "team-internal"]

    # Failure-context fields layered on top.
    assert final_md["status"] == "failure"
    assert "error_information" in final_md
    assert final_md["user_api_key"] == "test_api_key"

    # Timing reflects reality, not datetime.now().
    assert call_args["start_time"] == real_start

    # SLO built so the rich metadata JSON column gets populated downstream.
    assert kwargs.get("standard_logging_object") is not None


@pytest.mark.asyncio
async def test_failure_hook_backward_compat_no_logging_obj():
    """When request_data has no litellm_logging_obj (very early failure or
    test harness), the hook must not crash and must preserve original
    behaviour — original_key etc still land in metadata."""
    logger = _ProxyDBLogger()

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="user-1",
        team_id="team-1",
        org_id="org-1",
        end_user_id="end-user-1",
    )
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"original_key": "original_value"},
        "proxy_server_request": {"request_id": "test_request_id"},
    }

    with patch(
        "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
        new_callable=AsyncMock,
    ) as mock_update_database:
        await logger.async_post_call_failure_hook(
            request_data=request_data,
            original_exception=Exception("boom"),
            user_api_key_dict=user_api_key_dict,
        )

    mock_update_database.assert_called_once()
    call_args = mock_update_database.call_args[1]
    final_md = call_args["kwargs"]["litellm_params"]["metadata"]
    assert final_md["original_key"] == "original_value"
    assert final_md["status"] == "failure"
    assert "error_information" in final_md


# ---------------------------------------------------------------------------
# End-to-end through the orchestrator (``ProxyLogging.post_call_failure_hook``)
# This is the path the real proxy code uses — the orchestrator pops
# ``litellm_logging_obj`` from request_data before iterating callbacks, so
# the enricher MUST run inside the orchestrator (before the pop), not just
# inside the callback (where logging_obj is already gone).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_bakes_upstream_attribution_before_popping_logging_obj():
    """Regression test for the silent-no-op bug: without the orchestrator
    enrichment, the callback would see a stripped request_data and produce
    a blank spend log row. This test exercises the real production path
    (``ProxyLogging.post_call_failure_hook``) and asserts every field
    survives the pop."""
    import litellm
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    real_start = datetime(2026, 5, 21, 18, 30, 0)
    logging_obj = _make_logging_obj(
        model_call_details=_full_model_call_details(),
        start_time=real_start,
        trace_id="trace-abc",
    )

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        key_alias="alias-prod",
        user_id="user-1",
        team_id="team-1",
        org_id="org-1",
        end_user_id="end-user-1",
        request_route="/v1/chat/completions",
    )

    request_data = {
        "model": "open-large",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {},
        "litellm_logging_obj": logging_obj,
    }

    proxy_logger = ProxyLogging(user_api_key_cache=DualCache())
    db_logger = _ProxyDBLogger()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [db_logger]

    try:
        with patch(
            "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
            new_callable=AsyncMock,
        ) as mock_update_database, patch.object(
            proxy_logger, "alerting_handler", new_callable=AsyncMock
        ):
            await proxy_logger.post_call_failure_hook(
                request_data=request_data,
                original_exception=asyncio.CancelledError(),
                user_api_key_dict=user_api_key_dict,
            )
    finally:
        litellm.callbacks = original_callbacks

    # The orchestrator must have popped the logging obj.
    assert "litellm_logging_obj" not in request_data

    mock_update_database.assert_called_once()
    call_args = mock_update_database.call_args[1]
    kwargs = call_args["kwargs"]

    assert kwargs["litellm_params"]["api_base"] == "http://103.48.43.99:8000/v1"
    final_md = kwargs["litellm_params"]["metadata"]
    assert final_md["model_group"] == "open-large"
    assert final_md["model_info"]["id"] == "deployment-uuid-1"
    assert final_md["requester_ip_address"] == "10.7.1.144"
    assert call_args["start_time"] == real_start
    assert kwargs.get("standard_logging_object") is not None
    assert kwargs.get("litellm_trace_id") == "trace-abc"


@pytest.mark.asyncio
async def test_orchestrator_handles_request_data_with_no_logging_obj():
    """Auth-error / very-early-failure path: ``request_data`` never had a
    ``litellm_logging_obj`` attached. Orchestrator must not crash, must still
    call the spend-log writer, and the start_time fallback must not be None."""
    import litellm
    from litellm.caching import DualCache
    from litellm.proxy.utils import ProxyLogging

    user_api_key_dict = UserAPIKeyAuth(
        api_key="auth-error-key",
        request_route="/v1/chat/completions",
    )

    request_data = {
        "model": "claude-haiku",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {},
        # No litellm_logging_obj
    }

    proxy_logger = ProxyLogging(user_api_key_cache=DualCache())
    db_logger = _ProxyDBLogger()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [db_logger]

    try:
        with patch(
            "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database",
            new_callable=AsyncMock,
        ) as mock_update_database, patch.object(
            proxy_logger, "alerting_handler", new_callable=AsyncMock
        ):
            await proxy_logger.post_call_failure_hook(
                request_data=request_data,
                original_exception=Exception("401 — invalid key"),
                user_api_key_dict=user_api_key_dict,
            )
    finally:
        litellm.callbacks = original_callbacks

    mock_update_database.assert_called_once()
    call_args = mock_update_database.call_args[1]

    # start_time must be a real datetime (datetime.now() fallback), never None.
    assert isinstance(call_args["start_time"], datetime)
    # And the baked-in start_time on request_data must be set even though
    # logging_obj was absent — preserves contract for any other callback.
    assert isinstance(request_data.get("_litellm_failure_start_time"), datetime)
