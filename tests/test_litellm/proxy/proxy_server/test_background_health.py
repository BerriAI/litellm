"""Behavior pins for proxy_server background health-check helpers.

Pins covered:
- ``_get_process_rss_mb``
- ``_rss_mb_for_log``
- ``_run_direct_health_check_with_instrumentation``
- ``_schedule_background_health_check_db_save``
- ``_get_endpoint_exception_status``
- ``_write_health_state_to_router_cache``
- ``_adaptive_router_flusher_loop``
- ``_run_background_health_check``
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.proxy_server as proxy_server
from litellm.proxy.proxy_server import (
    _adaptive_router_flusher_loop,
    _get_endpoint_exception_status,
    _get_process_rss_mb,
    _run_background_health_check,
    _run_direct_health_check_with_instrumentation,
    _rss_mb_for_log,
    _schedule_background_health_check_db_save,
    _write_health_state_to_router_cache,
)

from .conftest import normalize

# ---------------------------------------------------------------------------
# _get_process_rss_mb
# ---------------------------------------------------------------------------


def test_get_process_rss_mb_returns_positive_float():
    value = _get_process_rss_mb()
    assert value is not None
    assert normalize(
        {
            "value_present": value is not None,
            "value_type": type(value).__name__,
            "positive": value > 0,
        }
    ) == {
        "value_present": True,
        "value_type": "float",
        "positive": True,
    }


def test_get_process_rss_mb_returns_none_when_resource_raises(monkeypatch):
    import resource

    def _boom(*_args, **_kwargs):
        raise OSError("nope")

    monkeypatch.setattr(resource, "getrusage", _boom)
    assert _get_process_rss_mb() is None


# ---------------------------------------------------------------------------
# _rss_mb_for_log
# ---------------------------------------------------------------------------


def test_rss_mb_for_log_formats_numeric_value(monkeypatch):
    monkeypatch.setattr(proxy_server, "_get_process_rss_mb", lambda: 100.5)
    result = _rss_mb_for_log()
    assert normalize(
        {
            "format": result,
            "is_string": isinstance(result, str),
            "contains_mb": "100.50" in result,
        }
    ) == {
        "format": "100.50",
        "is_string": True,
        "contains_mb": True,
    }


def test_rss_mb_for_log_unknown_when_rss_missing(monkeypatch):
    monkeypatch.setattr(proxy_server, "_get_process_rss_mb", lambda: None)
    assert _rss_mb_for_log() == "unknown"


# ---------------------------------------------------------------------------
# _run_direct_health_check_with_instrumentation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_direct_health_check_with_instrumentation_returns_results(
    monkeypatch,
):
    expected = (["healthy_ep"], ["unhealthy_ep"], {"m1": Exception("boom")})

    async def _fake_perform(model_list, details, max_concurrency, **kwargs):
        return expected

    monkeypatch.setattr(proxy_server, "perform_health_check", _fake_perform)
    monkeypatch.setattr(
        proxy_server,
        "health_check_filter_kwargs_from_general_settings",
        lambda _gs: {},
    )

    healthy, unhealthy, exceptions = (
        await _run_direct_health_check_with_instrumentation(
            model_list=[{"model_name": "gpt-4"}],
            details=False,
            max_concurrency=1,
            instrumentation_context={"source": "test"},
        )
    )

    assert normalize(
        {
            "healthy": healthy,
            "unhealthy": unhealthy,
            "exception_keys": list(exceptions.keys()),
        }
    ) == {
        "healthy": ["healthy_ep"],
        "unhealthy": ["unhealthy_ep"],
        "exception_keys": ["m1"],
    }


@pytest.mark.asyncio
async def test_run_direct_health_check_raises_non_kwarg_typeerror(monkeypatch):
    async def _boom(model_list, details, max_concurrency, **kwargs):
        raise TypeError("totally unrelated")

    monkeypatch.setattr(proxy_server, "perform_health_check", _boom)
    monkeypatch.setattr(
        proxy_server,
        "health_check_filter_kwargs_from_general_settings",
        lambda _gs: {},
    )

    with pytest.raises(TypeError):
        await _run_direct_health_check_with_instrumentation(
            model_list=[],
            details=False,
            max_concurrency=1,
            instrumentation_context={},
        )


# ---------------------------------------------------------------------------
# _schedule_background_health_check_db_save
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_background_health_check_db_save_creates_task(monkeypatch):
    captured = {}

    async def _fake_save(
        prisma_client,
        model_list,
        healthy,
        unhealthy,
        start_time,
        checked_by,
    ):
        captured["prisma_client"] = prisma_client
        captured["model_list"] = model_list
        captured["healthy"] = healthy
        captured["unhealthy"] = unhealthy
        captured["checked_by"] = checked_by

    import litellm.proxy.health_endpoints._health_endpoints as he

    monkeypatch.setattr(he, "_save_background_health_checks_to_db", _fake_save)

    prisma_client = MagicMock()
    shared_manager = SimpleNamespace(pod_id="pod-xyz")

    _schedule_background_health_check_db_save(
        prisma_client=prisma_client,
        shared_health_manager=shared_manager,
        model_list=[{"model_name": "gpt-4"}],
        healthy_endpoints=[{"model_id": "h1"}],
        unhealthy_endpoints=[{"model_id": "u1"}],
    )

    await asyncio.sleep(0)

    assert normalize(
        {
            "prisma_present": captured.get("prisma_client") is prisma_client,
            "checked_by": captured.get("checked_by"),
            "healthy": captured.get("healthy"),
            "unhealthy": captured.get("unhealthy"),
        }
    ) == {
        "prisma_present": True,
        "checked_by": "pod-xyz",
        "healthy": [{"model_id": "h1"}],
        "unhealthy": [{"model_id": "u1"}],
    }


def test_schedule_background_health_check_db_save_noop_when_prisma_none():
    _schedule_background_health_check_db_save(
        prisma_client=None,
        shared_health_manager=None,
        model_list=[],
        healthy_endpoints=[],
        unhealthy_endpoints=[],
    )


@pytest.mark.asyncio
async def test_schedule_background_health_check_db_save_invalid_no_event_loop_raises(
    monkeypatch,
):
    async def _fake_save(*_args, **_kwargs):
        return None

    import litellm.proxy.health_endpoints._health_endpoints as he

    monkeypatch.setattr(he, "_save_background_health_checks_to_db", _fake_save)

    def _broken_create_task(_coro):
        raise RuntimeError("no running event loop")

    monkeypatch.setattr(asyncio, "create_task", _broken_create_task)

    with pytest.raises(RuntimeError):
        _schedule_background_health_check_db_save(
            prisma_client=MagicMock(),
            shared_health_manager=None,
            model_list=[],
            healthy_endpoints=[],
            unhealthy_endpoints=[],
        )


# ---------------------------------------------------------------------------
# _get_endpoint_exception_status
# ---------------------------------------------------------------------------


def test_get_endpoint_exception_status_prefers_live_exception():
    endpoint = {"model_id": "m1", "exception_status": 999}
    exceptions = {"m1": SimpleNamespace(status_code=429)}
    status = _get_endpoint_exception_status(endpoint, exceptions)
    assert normalize(
        {
            "input_endpoint": endpoint,
            "exceptions_keys": list(exceptions.keys()),
            "status": status,
        }
    ) == {
        "input_endpoint": {"model_id": "m1", "exception_status": 999},
        "exceptions_keys": ["m1"],
        "status": 429,
    }


def test_get_endpoint_exception_status_falls_back_to_stored_int():
    endpoint = {"model_id": "m-missing", "exception_status": 503}
    assert _get_endpoint_exception_status(endpoint, {}) == 503


def test_get_endpoint_exception_status_default_500_when_no_data():
    assert _get_endpoint_exception_status({}, {}) == 500


def test_get_endpoint_exception_status_invalid_endpoint_type_raises():
    with pytest.raises(AttributeError):
        _get_endpoint_exception_status(None, {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _write_health_state_to_router_cache
# ---------------------------------------------------------------------------


def test_write_health_state_to_router_cache_sets_states(monkeypatch):
    fake_router = MagicMock()
    fake_router.enable_health_check_routing = True
    fake_router.health_check_ignore_transient_errors = False
    fake_router.cooldown_time = 30
    fake_router.health_state_cache = MagicMock()

    monkeypatch.setattr(proxy_server, "llm_router", fake_router)

    fake_states = {"m1": {"is_healthy": True}, "m2": {"is_healthy": False}}

    import litellm.proxy.health_check as hc

    monkeypatch.setattr(hc, "build_deployment_health_states", lambda **_kw: fake_states)

    import litellm.router_utils.cooldown_handlers as cd

    monkeypatch.setattr(cd, "_set_cooldown_deployments", lambda **_kw: None)

    import litellm.router_utils.router_callbacks.track_deployment_metrics as tdm

    monkeypatch.setattr(
        tdm,
        "increment_deployment_failures_for_current_minute",
        lambda **_kw: None,
    )

    healthy = [{"model_id": "m1"}]
    unhealthy = [{"model_id": "m2"}]
    exceptions = {"m2": SimpleNamespace(status_code=500)}

    _write_health_state_to_router_cache(healthy, unhealthy, exceptions)

    fake_router.health_state_cache.set_deployment_health_states.assert_called_once_with(
        fake_states
    )

    call_args = fake_router.health_state_cache.set_deployment_health_states.call_args[
        0
    ][0]
    assert normalize(
        {
            "states_keys": sorted(call_args.keys()),
            "m1_healthy": call_args["m1"]["is_healthy"],
            "m2_healthy": call_args["m2"]["is_healthy"],
        }
    ) == {
        "states_keys": ["m1", "m2"],
        "m1_healthy": True,
        "m2_healthy": False,
    }


def test_write_health_state_to_router_cache_noop_when_router_none(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", None)
    _write_health_state_to_router_cache([], [], {})


def test_write_health_state_to_router_cache_swallows_internal_failures(monkeypatch):
    """The function logs and swallows exceptions so a bad cache call never crashes the loop."""
    fake_router = MagicMock()
    fake_router.enable_health_check_routing = True
    fake_router.health_check_ignore_transient_errors = False
    fake_router.health_state_cache.set_deployment_health_states.side_effect = (
        RuntimeError("cache exploded")
    )

    monkeypatch.setattr(proxy_server, "llm_router", fake_router)

    import litellm.proxy.health_check as hc

    monkeypatch.setattr(
        hc,
        "build_deployment_health_states",
        lambda **_kw: {"m1": {"is_healthy": True}},
    )

    _write_health_state_to_router_cache([{"model_id": "m1"}], [], {})


# ---------------------------------------------------------------------------
# _adaptive_router_flusher_loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adaptive_router_flusher_loop_flushes_each_router(monkeypatch):
    fake_ar = MagicMock()
    fake_ar._state_loaded = True
    fake_ar.queue.flush_state_to_db = AsyncMock()
    fake_ar.queue.flush_session_to_db = AsyncMock()

    fake_router = MagicMock()
    fake_router.adaptive_routers = {"alpha": fake_ar}

    monkeypatch.setattr(proxy_server, "llm_router", fake_router)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())

    # asyncio.sleep is awaited at the top of every iteration; raise CancelledError
    # on the SECOND call so the first iteration completes its flush work.
    call_count = {"n": 0}
    _real_sleep = asyncio.sleep

    async def _short_sleep(_seconds):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise asyncio.CancelledError()
        await _real_sleep(0)

    monkeypatch.setattr(proxy_server.asyncio, "sleep", _short_sleep)

    with pytest.raises(asyncio.CancelledError):
        await _adaptive_router_flusher_loop()

    assert fake_ar.queue.flush_state_to_db.await_count == 1
    assert fake_ar.queue.flush_session_to_db.await_count == 1


@pytest.mark.asyncio
async def test_adaptive_router_flusher_loop_times_out_when_sleep_real(monkeypatch):
    """Confirms the loop is infinite — wait_for must raise TimeoutError."""
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock(adaptive_routers={}))
    monkeypatch.setattr(proxy_server, "prisma_client", None)

    # Bind the real asyncio.sleep before the patch so the replacement does not
    # recurse into itself.
    _real_sleep = asyncio.sleep

    async def _instant_sleep(_seconds):
        await _real_sleep(0)

    monkeypatch.setattr(proxy_server.asyncio, "sleep", _instant_sleep)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_adaptive_router_flusher_loop(), timeout=0.2)


# ---------------------------------------------------------------------------
# _run_background_health_check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_background_health_check_returns_immediately_when_interval_invalid(
    monkeypatch,
):
    monkeypatch.setattr(proxy_server, "health_check_interval", None)

    result = await _run_background_health_check()

    assert normalize(
        {
            "result_is_none": result is None,
            "loop_active": proxy_server.background_health_check_loop_active,
            "interval": proxy_server.health_check_interval,
        }
    ) == {
        "result_is_none": True,
        "loop_active": False,
        "interval": None,
    }


@pytest.mark.asyncio
async def test_run_background_health_check_runs_one_cycle_then_cancels(monkeypatch):
    monkeypatch.setattr(proxy_server, "health_check_interval", 60)
    monkeypatch.setattr(proxy_server, "health_check_concurrency", 1)
    monkeypatch.setattr(proxy_server, "health_check_details", True)
    monkeypatch.setattr(proxy_server, "use_shared_health_check", False)
    monkeypatch.setattr(proxy_server, "redis_usage_cache", None)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "background_health_check_loop_active", False)
    monkeypatch.setattr(
        proxy_server,
        "llm_model_list",
        [{"model_name": "gpt-4", "model_info": {}}],
    )
    monkeypatch.setattr(
        proxy_server,
        "health_check_results",
        {"healthy_endpoints": [], "unhealthy_endpoints": []},
    )

    async def _fake_direct(*_a, **_kw):
        return ([{"model_id": "h"}], [{"model_id": "u"}], {})

    monkeypatch.setattr(
        proxy_server,
        "_run_direct_health_check_with_instrumentation",
        _fake_direct,
    )
    monkeypatch.setattr(
        proxy_server, "_schedule_background_health_check_db_save", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        proxy_server, "_write_health_state_to_router_cache", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        proxy_server,
        "health_check_filter_kwargs_from_general_settings",
        lambda _gs: {},
    )

    sleep_calls = {"n": 0}

    async def _stop_sleep(_seconds):
        sleep_calls["n"] += 1
        raise asyncio.CancelledError()

    monkeypatch.setattr(proxy_server.asyncio, "sleep", _stop_sleep)

    with pytest.raises(asyncio.CancelledError):
        await _run_background_health_check()

    assert normalize(
        {
            "healthy_count": proxy_server.health_check_results["healthy_count"],
            "unhealthy_count": proxy_server.health_check_results["unhealthy_count"],
            "sleep_invoked": sleep_calls["n"] >= 1,
        }
    ) == {
        "healthy_count": 1,
        "unhealthy_count": 1,
        "sleep_invoked": True,
    }
