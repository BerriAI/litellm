"""
LIT-6611: every unique client-supplied model name that fails routing used to
mint permanent Prometheus series carrying ``requested_model="<junk>"`` on the
proxy request metrics and the deployment metrics, with no eviction. The fix
collapses any requested model the router does not recognize (and no wildcard
pattern matches) into the single ``other`` label bucket, while recognized
names, aliases, and wildcard-matched names keep their own label values.
"""

import sys
import types
from unittest.mock import patch

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import (
    UNRECOGNIZED_REQUESTED_MODEL_LABEL,
    PrometheusLogger,
)
from litellm.proxy._types import UserAPIKeyAuth


class _ClientSideError(Exception):
    status_code = 400


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

    yield

    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture
def router():
    return litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake-key"},
            },
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_key": "fake-key"},
            },
        ],
        model_group_alias={"gpt4o-alias": "gpt-4o-mini"},
    )


@pytest.fixture
def team_router():
    return litellm.Router(
        model_list=[
            {
                "model_name": "team-internal-gpt",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake-key"},
                "model_info": {"team_id": "team-1", "team_public_model_name": "team-alias-gpt"},
            },
            {
                "model_name": "team-internal-bedrock",
                "litellm_params": {"model": "openai/*", "api_key": "fake-key"},
                "model_info": {"team_id": "team-1", "team_public_model_name": "team-models/*"},
            },
        ]
    )


def _requested_model_values(metric) -> set[str]:
    index = metric._labelnames.index("requested_model")
    return {sample_key[index] for sample_key in metric._metrics}


def _series_count(metric) -> int:
    return len(metric._metrics)


def _total_value(metric) -> float:
    return sum(child._value.get() for child in metric._metrics.values())


async def _fire_proxy_failure(logger: PrometheusLogger, model: str) -> None:
    await logger.async_post_call_failure_hook(
        request_data={"model": model, "metadata": {}, "proxy_server_request": {}},
        original_exception=_ClientSideError(f"model {model} does not exist"),
        user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key-1"),
    )


@pytest.mark.asyncio
async def test_unknown_models_collapse_to_one_series_on_proxy_request_metrics(router):
    logger = PrometheusLogger()

    with patch("litellm.proxy.proxy_server.llm_router", router, create=True):  # test-quality-ok: production reads proxy_server.llm_router lazily, no injection seam
        for index in range(25):
            await _fire_proxy_failure(logger, f"agent-typo-{index}")

    for metric in (
        logger.litellm_proxy_failed_requests_metric,
        logger.litellm_proxy_total_requests_metric,
    ):
        assert _requested_model_values(metric) == {UNRECOGNIZED_REQUESTED_MODEL_LABEL}
        assert _series_count(metric) == 1
        assert _total_value(metric) == 25


@pytest.mark.asyncio
async def test_known_alias_and_wildcard_models_keep_their_own_labels(router):
    logger = PrometheusLogger()

    with patch("litellm.proxy.proxy_server.llm_router", router, create=True):  # test-quality-ok: production reads proxy_server.llm_router lazily, no injection seam
        await _fire_proxy_failure(logger, "gpt-4o-mini")
        await _fire_proxy_failure(logger, "gpt4o-alias")
        await _fire_proxy_failure(logger, "openai/gpt-4o-audio-preview")
        await _fire_proxy_failure(logger, "agent-typo-hallucinated")

    for metric in (
        logger.litellm_proxy_failed_requests_metric,
        logger.litellm_proxy_total_requests_metric,
    ):
        assert _requested_model_values(metric) == {
            "gpt-4o-mini",
            "gpt4o-alias",
            "openai/gpt-4o-audio-preview",
            UNRECOGNIZED_REQUESTED_MODEL_LABEL,
        }


@pytest.mark.asyncio
async def test_team_alias_and_team_wildcard_models_keep_their_own_labels(team_router):
    logger = PrometheusLogger()

    with patch("litellm.proxy.proxy_server.llm_router", team_router, create=True):  # test-quality-ok: production reads proxy_server.llm_router lazily, no injection seam
        await _fire_proxy_failure(logger, "team-alias-gpt")
        await _fire_proxy_failure(logger, "team-models/gpt-4o-audio-preview")
        await _fire_proxy_failure(logger, "agent-typo-hallucinated")

    for metric in (
        logger.litellm_proxy_failed_requests_metric,
        logger.litellm_proxy_total_requests_metric,
    ):
        assert _requested_model_values(metric) == {
            "team-alias-gpt",
            "team-models/gpt-4o-audio-preview",
            UNRECOGNIZED_REQUESTED_MODEL_LABEL,
        }


@pytest.mark.asyncio
async def test_unknown_models_collapse_to_other_when_router_is_unavailable():
    logger = PrometheusLogger()

    with patch("litellm.proxy.proxy_server.llm_router", None, create=True):  # test-quality-ok: production reads proxy_server.llm_router lazily, no injection seam
        await _fire_proxy_failure(logger, "agent-typo-no-router")
        await _fire_proxy_failure(logger, "gpt-4o-mini")

    assert _requested_model_values(logger.litellm_proxy_failed_requests_metric) == {
        UNRECOGNIZED_REQUESTED_MODEL_LABEL
    }


@pytest.mark.asyncio
async def test_sdk_router_originated_metrics_keep_labels_without_proxy_router():
    logger = PrometheusLogger()

    with patch("litellm.proxy.proxy_server.llm_router", None, create=True):  # test-quality-ok: production reads proxy_server.llm_router lazily, no injection seam
        logger.set_llm_deployment_failure_metrics(
            request_kwargs={
                "model": "sdk-deployment-group",
                "litellm_params": {"metadata": {}},
                "standard_logging_object": {},
                "exception": _ClientSideError("model does not exist"),
            }
        )
        await logger.log_failure_fallback_event(
            original_model_group="sdk-fallback-group",
            kwargs={"model": "sdk-fallback-group", "metadata": {}},
            original_exception=_ClientSideError("upstream unavailable"),
        )

    assert _requested_model_values(logger.litellm_deployment_failure_responses) == {"sdk-deployment-group"}
    assert _requested_model_values(logger.litellm_deployment_failed_fallbacks) == {"sdk-fallback-group"}


@pytest.mark.asyncio
async def test_sdk_fallback_labels_survive_non_import_errors_from_proxy_module(monkeypatch):
    logger = PrometheusLogger()
    broken_proxy_module = types.ModuleType("litellm.proxy.proxy_server")

    def _raise_value_error(_name: str):
        raise ValueError("bad proxy env var")

    broken_proxy_module.__getattr__ = _raise_value_error  # test-quality-ok: reproduces a proxy_server import raising non-ImportError, no injection seam
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", broken_proxy_module)  # test-quality-ok: reproduces a proxy_server import raising non-ImportError, no injection seam

    await logger.log_failure_fallback_event(
        original_model_group="sdk-fallback-group",
        kwargs={"model": "sdk-fallback-group", "metadata": {}},
        original_exception=_ClientSideError("upstream unavailable"),
    )

    assert _requested_model_values(logger.litellm_deployment_failed_fallbacks) == {"sdk-fallback-group"}


def test_unknown_models_collapse_to_one_series_on_deployment_metrics(router):
    logger = PrometheusLogger()

    with patch("litellm.proxy.proxy_server.llm_router", router, create=True):  # test-quality-ok: production reads proxy_server.llm_router lazily, no injection seam
        for index in range(25):
            logger.set_llm_deployment_failure_metrics(
                request_kwargs={
                    "model": f"agent-typo-{index}",
                    "litellm_params": {"metadata": {}},
                    "standard_logging_object": {},
                    "exception": _ClientSideError("model does not exist"),
                }
            )
        logger.set_llm_deployment_failure_metrics(
            request_kwargs={
                "model": "gpt-4o-mini",
                "litellm_params": {"metadata": {}},
                "standard_logging_object": {},
                "exception": _ClientSideError("all deployments cooling down"),
            }
        )

    for metric in (
        logger.litellm_deployment_failure_responses,
        logger.litellm_deployment_total_requests,
    ):
        assert _requested_model_values(metric) == {
            UNRECOGNIZED_REQUESTED_MODEL_LABEL,
            "gpt-4o-mini",
        }
        assert _series_count(metric) == 2
        assert _total_value(metric) == 26


@pytest.mark.asyncio
async def test_fallback_event_requested_model_is_bounded(router):
    logger = PrometheusLogger()
    kwargs = {"model": "gpt-4o-mini", "metadata": {}}

    with patch("litellm.proxy.proxy_server.llm_router", router, create=True):  # test-quality-ok: production reads proxy_server.llm_router lazily, no injection seam
        await logger.log_failure_fallback_event(
            original_model_group="agent-typo-hallucinated",
            kwargs=kwargs,
            original_exception=_ClientSideError("model does not exist"),
        )
        await logger.log_success_fallback_event(
            original_model_group="agent-typo-hallucinated",
            kwargs=kwargs,
            original_exception=_ClientSideError("model does not exist"),
        )
        await logger.log_failure_fallback_event(
            original_model_group="gpt-4o-mini",
            kwargs=kwargs,
            original_exception=_ClientSideError("upstream unavailable"),
        )

    assert _requested_model_values(logger.litellm_deployment_failed_fallbacks) == {
        UNRECOGNIZED_REQUESTED_MODEL_LABEL,
        "gpt-4o-mini",
    }
    assert _requested_model_values(logger.litellm_deployment_successful_fallbacks) == {
        UNRECOGNIZED_REQUESTED_MODEL_LABEL
    }
