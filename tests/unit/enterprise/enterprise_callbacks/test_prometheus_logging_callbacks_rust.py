import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import (
    StandardLoggingHiddenParams,
    StandardLoggingMetadata,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
)
from tests.unit.integrations.prometheus_native_support import (
    native_failure,
    enable_native_prometheus,
    native_success,
    observations,
    series,
    value_of,
)


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


def _clear_registry() -> None:
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)


@pytest.fixture
def prometheus_logger() -> PrometheusLogger:
    _clear_registry()
    logger = PrometheusLogger()
    assert logger._native_store is not None
    return logger


@pytest.fixture
def known_model_router():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5-mini",
                "litellm_params": {"model": "openai/gpt-5-mini", "api_key": "fake-key"},
            },
            {
                "model_name": "us/azure/openai/gpt-5-mini",
                "litellm_params": {"model": "openai/gpt-5-mini", "api_key": "fake-key"},
            },
        ]
    )
    with patch(
        "litellm.proxy.proxy_server.llm_router", router, create=True
    ):  # test-quality-ok: production reads proxy_server.llm_router lazily, no injection seam
        yield router


def create_standard_logging_payload() -> StandardLoggingPayload:
    return StandardLoggingPayload(
        id="test_id",
        call_type="completion",
        stream=False,
        response_cost=0.1,
        response_cost_failure_debug_info=None,
        status="success",
        total_tokens=30,
        prompt_tokens=20,
        completion_tokens=10,
        startTime=1234567890.0,
        endTime=1234567891.0,
        completionStartTime=1234567890.5,
        model_map_information=StandardLoggingModelInformation(model_map_key="gpt-5-mini", model_map_value=None),
        model="gpt-5-mini",
        model_id="model-123",
        model_group="openai-gpt",
        custom_llm_provider="openai",
        api_base="https://api.openai.com",
        metadata=StandardLoggingMetadata(
            user_api_key_hash="test_hash",
            user_api_key_alias="test_alias",
            user_api_key_team_id="test_team",
            user_api_key_user_id="test_user",
            user_api_key_user_email="test@example.com",
            user_api_key_team_alias="test_team_alias",
            user_api_key_org_id=None,
            spend_logs_metadata=None,
            requester_ip_address="127.0.0.1",
            requester_metadata=None,
            user_api_key_end_user_id="test_end_user",
        ),
        cache_hit=False,
        cache_key=None,
        saved_cache_cost=0.0,
        request_tags=[],
        end_user=None,
        requester_ip_address="127.0.0.1",
        messages=[{"role": "user", "content": "Hello, world!"}],
        response={"choices": [{"message": {"content": "Hi there!"}}]},
        error_str=None,
        model_parameters={"stream": True},
        hidden_params=StandardLoggingHiddenParams(
            model_id="model-123",
            cache_key=None,
            api_base="https://api.openai.com",
            response_cost="0.1",
            additional_headers=None,
        ),
    )


def _anonymous_payload() -> StandardLoggingPayload:
    standard_logging_payload = create_standard_logging_payload()
    standard_logging_payload["metadata"]["user_api_key_user_id"] = None
    standard_logging_payload["metadata"]["user_api_key_user_email"] = None
    standard_logging_payload["metadata"]["requester_ip_address"] = None
    standard_logging_payload["requester_ip_address"] = None
    standard_logging_payload["model_group"] = None
    return standard_logging_payload


def _success_kwargs(standard_logging_payload: StandardLoggingPayload, **extra: object) -> dict[str, object]:
    now = datetime.now()
    kwargs: dict[str, object] = {
        "model": "gpt-5-mini",
        "stream": True,
        "litellm_params": {"metadata": {}},
        "start_time": now,
        "completion_start_time": now,
        "api_call_start_time": now,
        "end_time": now + timedelta(seconds=1),
        "standard_logging_object": standard_logging_payload,
    }
    kwargs.update(extra)
    return kwargs


@pytest.mark.asyncio
async def test_async_log_success_event(prometheus_logger):
    standard_logging_object = create_standard_logging_payload()
    kwargs = {
        "model": "gpt-5-mini",
        "stream": True,
        "litellm_params": {
            "metadata": {
                "user_api_key": "test_key",
                "user_api_key_user_id": "test_user",
                "user_api_key_team_id": "test_team",
                "user_api_key_end_user_id": "test_end_user",
            }
        },
        "start_time": datetime.now(),
        "completion_start_time": datetime.now(),
        "api_call_start_time": datetime.now(),
        "end_time": datetime.now() + timedelta(seconds=1),
        "standard_logging_object": standard_logging_object,
    }
    response_obj = MagicMock()

    await native_success(prometheus_logger, kwargs, response_obj, kwargs["start_time"], kwargs["end_time"])

    for metric in (
        prometheus_logger.litellm_requests_metric,
        prometheus_logger.litellm_spend_metric,
        prometheus_logger.litellm_tokens_metric,
        prometheus_logger.litellm_input_tokens_metric,
        prometheus_logger.litellm_output_tokens_metric,
        prometheus_logger.litellm_remaining_team_budget_metric,
        prometheus_logger.litellm_remaining_api_key_budget_metric,
        prometheus_logger.litellm_remaining_api_key_requests_for_model,
        prometheus_logger.litellm_remaining_api_key_tokens_for_model,
        prometheus_logger.litellm_llm_api_time_to_first_token_metric,
        prometheus_logger.litellm_llm_api_latency_metric,
        prometheus_logger.litellm_request_total_latency_metric,
    ):
        assert series(metric.name), metric.name


_TOKEN_LABELS = {
    "end_user": None,
    "user": None,
    "user_email": None,
    "hashed_api_key": "test_hash",
    "api_key_alias": "test_alias",
    "team": "test_team",
    "team_alias": "test_team_alias",
    "org_id": None,
    "org_alias": None,
    "requested_model": None,
    "model": "gpt-5-mini",
    "model_id": "model-123",
    "api_provider": "openai",
}


@pytest.mark.asyncio
async def test_increment_token_metrics(prometheus_logger):
    standard_logging_payload = _anonymous_payload()
    standard_logging_payload["total_tokens"] = 100
    standard_logging_payload["prompt_tokens"] = 50
    standard_logging_payload["completion_tokens"] = 50

    await native_success(prometheus_logger, _success_kwargs(standard_logging_payload), MagicMock())

    assert series(prometheus_logger.litellm_tokens_metric.name) == [
        ({key: str(value) for key, value in _TOKEN_LABELS.items()}, 100)
    ]
    assert series(prometheus_logger.litellm_input_tokens_metric.name) == [
        ({key: str(value) for key, value in _TOKEN_LABELS.items()}, 50)
    ]
    assert series(prometheus_logger.litellm_output_tokens_metric.name) == [
        ({key: str(value) for key, value in _TOKEN_LABELS.items()}, 50)
    ]


@pytest.mark.asyncio
async def test_increment_remaining_budget_metrics(prometheus_logger):
    future_reset_time_team = datetime.now() + timedelta(hours=10)
    future_reset_time_key = datetime.now() + timedelta(hours=12)
    with (
        patch("litellm.proxy.auth.auth_checks.get_team_object") as mock_get_team,
        patch("litellm.proxy.auth.auth_checks.get_key_object") as mock_get_key,
    ):
        mock_get_team.return_value = MagicMock(budget_reset_at=future_reset_time_team)
        mock_get_key.return_value = MagicMock(budget_reset_at=future_reset_time_key)

        litellm_params = {
            "metadata": {
                "user_api_key_team_spend": 50,
                "user_api_key_team_max_budget": 100,
                "user_api_key_spend": 25,
                "user_api_key_max_budget": 75,
            }
        }
        standard_logging_payload = create_standard_logging_payload()
        standard_logging_payload["response_cost"] = 10
        standard_logging_payload["metadata"]["user_api_key_team_id"] = "team1"
        standard_logging_payload["metadata"]["user_api_key_team_alias"] = "team_alias1"
        standard_logging_payload["metadata"]["user_api_key_hash"] = "key1"
        standard_logging_payload["metadata"]["user_api_key_alias"] = "alias1"
        standard_logging_payload["metadata"]["user_api_key_user_id"] = None

        await native_success(
            prometheus_logger,
            _success_kwargs(standard_logging_payload, litellm_params=litellm_params),
            MagicMock(),
        )

        team = {"team": "team1", "team_alias": "team_alias1"}
        key = {"hashed_api_key": "key1", "api_key_alias": "alias1"}
        assert series(prometheus_logger.litellm_remaining_team_budget_metric.name) == [
            (team, 40)  # 100 - (50 + 10)
        ]
        assert series(prometheus_logger.litellm_remaining_api_key_budget_metric.name) == [
            (key, 40)  # 75 - (25 + 10)
        ]
        assert series(prometheus_logger.litellm_team_max_budget_metric.name) == [(team, 100)]
        assert series(prometheus_logger.litellm_api_key_max_budget_metric.name) == [(key, 75)]

        team_hours = series(prometheus_logger.litellm_team_budget_remaining_hours_metric.name)
        assert [labels for labels, _ in team_hours] == [team]
        assert 9.9 <= team_hours[0][1] <= 10.0

        key_hours = series(prometheus_logger.litellm_api_key_budget_remaining_hours_metric.name)
        assert [labels for labels, _ in key_hours] == [key]
        assert 11.9 <= key_hours[0][1] <= 12.0


_LATENCY_LABELS = {
    "end_user": None,
    "user": "test_user",
    "hashed_api_key": "test_hash",
    "api_key_alias": "test_alias",
    "team": "test_team",
    "team_alias": "test_team_alias",
    "org_id": None,
    "org_alias": None,
    "model_group": "openai-gpt",
    "requested_model": "openai-gpt",
    "model": "gpt-5-mini",
    "model_id": "model-123",
    "api_provider": "openai",
    "service_tier": None,
}


@pytest.mark.asyncio
async def test_set_latency_metrics(prometheus_logger):
    standard_logging_payload = create_standard_logging_payload()
    now = datetime.now()
    kwargs = _success_kwargs(
        standard_logging_payload,
        end_time=now,
        start_time=now - timedelta(seconds=2),
        api_call_start_time=now - timedelta(seconds=1.5),
        completion_start_time=now - timedelta(seconds=1),
        stream=True,
    )

    await native_success(prometheus_logger, kwargs, MagicMock())

    assert observations(prometheus_logger.litellm_llm_api_time_to_first_token_metric.name, _LATENCY_LABELS) == [
        (1, 0.5)
    ]
    assert observations(prometheus_logger.litellm_llm_api_latency_metric.name, _LATENCY_LABELS) == [(1, 1.5)]
    assert observations(prometheus_logger.litellm_request_total_latency_metric.name, _LATENCY_LABELS) == [(1, 2.0)]


@pytest.mark.asyncio
async def test_set_latency_metrics_missing_timestamps(prometheus_logger):
    kwargs = _success_kwargs(
        create_standard_logging_payload(),
        end_time=datetime.now(),
        start_time=datetime.now() - timedelta(seconds=2),
        api_call_start_time=datetime.now() - timedelta(seconds=1.5),
        completion_start_time=None,
        stream=True,
    )

    await native_success(prometheus_logger, kwargs, MagicMock())

    assert series(prometheus_logger.litellm_llm_api_time_to_first_token_metric.name) == []
    assert len(series(prometheus_logger.litellm_llm_api_latency_metric.name)) == 1
    assert len(series(prometheus_logger.litellm_request_total_latency_metric.name)) == 1


@pytest.mark.asyncio
async def test_set_latency_metrics_missing_api_call_start(prometheus_logger):
    kwargs = _success_kwargs(
        create_standard_logging_payload(),
        end_time=datetime.now(),
        start_time=datetime.now() - timedelta(seconds=2),
        api_call_start_time=None,
        completion_start_time=datetime.now() - timedelta(seconds=1),
        stream=True,
    )

    await native_success(prometheus_logger, kwargs, MagicMock())

    assert series(prometheus_logger.litellm_llm_api_time_to_first_token_metric.name) == []
    assert series(prometheus_logger.litellm_llm_api_latency_metric.name) == []
    assert len(series(prometheus_logger.litellm_request_total_latency_metric.name)) == 1


@pytest.mark.asyncio
async def test_increment_top_level_request_and_spend_metrics(prometheus_logger):
    await native_success(prometheus_logger, _success_kwargs(_anonymous_payload()), MagicMock())

    labels = {
        "end_user": None,
        "user": None,
        "user_email": None,
        "hashed_api_key": "test_hash",
        "api_key_alias": "test_alias",
        "team": "test_team",
        "team_alias": "test_team_alias",
        "org_id": None,
        "org_alias": None,
        "model": "gpt-5-mini",
        "model_id": "model-123",
        "api_provider": "openai",
        "client_ip": None,
        "user_agent": None,
        "requested_model": None,
    }
    assert len(series(prometheus_logger.litellm_requests_metric.name)) == 1
    assert value_of(prometheus_logger.litellm_requests_metric.name, labels) == 1
    assert len(series(prometheus_logger.litellm_spend_metric.name)) == 1
    assert value_of(prometheus_logger.litellm_spend_metric.name, {**labels, "service_tier": None}) == 0.1


def _deployment_state(logger: PrometheusLogger) -> list[tuple[dict[str, str], float]]:
    return series(logger.litellm_deployment_state.name)


@pytest.mark.asyncio
async def test_async_log_failure_event(prometheus_logger):
    standard_logging_object = create_standard_logging_payload()
    kwargs = {
        "model": "gpt-5-mini",
        "litellm_params": {
            "custom_llm_provider": "openai",
        },
        "start_time": datetime.now(),
        "completion_start_time": datetime.now(),
        "api_call_start_time": datetime.now(),
        "end_time": datetime.now() + timedelta(seconds=1),
        "standard_logging_object": standard_logging_object,
        "exception": Exception("Test error"),
    }
    response_obj = MagicMock()

    await native_failure(prometheus_logger, kwargs, response_obj, kwargs["start_time"], kwargs["end_time"])

    failed = series(prometheus_logger.litellm_llm_api_failed_requests_metric.name)
    assert failed == [
        (
            {
                "end_user": "None",
                "hashed_api_key": "test_hash",
                "api_key_alias": "test_alias",
                "model": "gpt-5-mini",
                "team": "test_team",
                "team_alias": "test_team_alias",
                "user": "test_user",
                "model_id": "model-123",
            },
            1,
        )
    ]

    assert _deployment_state(prometheus_logger) == [
        (
            {
                "litellm_model_name": "gpt-5-mini",
                "model_id": "model-123",
                "api_base": "https://api.openai.com",
                "api_provider": "openai",
            },
            1,
        )
    ]

    failure_responses = series(prometheus_logger.litellm_deployment_failure_responses.name)
    assert len(failure_responses) == 1
    actual_failure_labels, failure_count = failure_responses[0]
    expected_failure_labels = {
        "litellm_model_name": "gpt-5-mini",
        "model_id": "model-123",
        "api_base": "https://api.openai.com",
        "api_provider": "openai",
        "exception_class": "Exception",
        "requested_model": "openai-gpt",
        "hashed_api_key": "test_hash",
        "api_key_alias": "test_alias",
        "team": "test_team",
        "team_alias": "test_team_alias",
    }
    for key, expected_val in expected_failure_labels.items():
        assert key in actual_failure_labels, f"Missing label {key}"
        assert actual_failure_labels[key] == expected_val
    assert actual_failure_labels.get("exception_status") in ("None", None)
    assert actual_failure_labels.get("client_ip") == "127.0.0.1"
    assert failure_count == 1

    total_requests = series(prometheus_logger.litellm_deployment_total_requests.name)
    assert len(total_requests) == 1
    actual_total_labels, total_count = total_requests[0]
    expected_total_labels = {
        "litellm_model_name": "gpt-5-mini",
        "model_id": "model-123",
        "api_base": "https://api.openai.com",
        "api_provider": "openai",
        "requested_model": "openai-gpt",
        "hashed_api_key": "test_hash",
        "api_key_alias": "test_alias",
        "team": "test_team",
        "team_alias": "test_team_alias",
    }
    for key, expected_val in expected_total_labels.items():
        assert key in actual_total_labels, f"Missing label {key}"
        assert actual_total_labels[key] == expected_val
    assert actual_total_labels.get("client_ip") == "127.0.0.1"
    assert total_count == 1


@pytest.mark.asyncio
async def test_async_log_failure_event_litellm_side_rate_limit(prometheus_logger, known_model_router):
    standard_logging_object = create_standard_logging_payload()
    standard_logging_object["model_id"] = ""
    standard_logging_object["model_group"] = ""
    standard_logging_object["api_base"] = ""

    rate_limit_exc = Exception("LiteLLM rate limit exceeded")
    rate_limit_exc.status_code = 429
    kwargs = {
        "model": "us/azure/openai/gpt-5-mini",
        "litellm_params": {},
        "start_time": datetime.now(),
        "completion_start_time": datetime.now(),
        "api_call_start_time": datetime.now(),
        "end_time": datetime.now() + timedelta(seconds=1),
        "standard_logging_object": standard_logging_object,
        "exception": rate_limit_exc,
    }

    await native_failure(prometheus_logger, kwargs, MagicMock(), kwargs["start_time"], kwargs["end_time"])

    assert _deployment_state(prometheus_logger) == []

    failure_responses = series(prometheus_logger.litellm_deployment_failure_responses.name)
    assert len(failure_responses) == 1
    actual_failure_labels = failure_responses[0][0]
    assert actual_failure_labels["requested_model"] == "us/azure/openai/gpt-5-mini"
    assert actual_failure_labels["litellm_model_name"] == ""
    assert actual_failure_labels["model_id"] == ""
    assert actual_failure_labels["api_base"] == ""
    assert actual_failure_labels["api_provider"] == ""
    assert actual_failure_labels["exception_status"] == "429"


@pytest.mark.asyncio
async def test_async_post_call_failure_hook(known_model_router, monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_emit_rate_limit_labels", True)
    _clear_registry()
    prometheus_logger = PrometheusLogger()

    request_data = {"model": "gpt-5-mini"}

    original_exception = litellm.RateLimitError(message="Test error", llm_provider="openai", model="gpt-5-mini")

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_key",
        key_alias="test_alias",
        team_id="test_team",
        team_alias="test_team_alias",
        user_id="test_user",
        end_user_id="test_end_user",
        request_route="/chat/completions",
    )

    await prometheus_logger.async_post_call_failure_hook(
        request_data=request_data,
        original_exception=original_exception,
        user_api_key_dict=user_api_key_dict,
    )

    failed_labels = {
        "end_user": None,
        "user": "test_user",
        "user_email": None,
        "hashed_api_key": "test_key",
        "api_key_alias": "test_alias",
        "team": "test_team",
        "team_alias": "test_team_alias",
        "org_id": None,
        "org_alias": None,
        "requested_model": "gpt-5-mini",
        "exception_status": "429",
        "exception_class": "Openai.RateLimitError",
        "rate_limit_category": "vendor_rate_limit",
        "rate_limit_type": None,
        "route": user_api_key_dict.request_route,
        "model_id": None,
        "client_ip": None,
        "user_agent": None,
        "api_provider": "openai",
    }
    assert len(series(prometheus_logger.litellm_proxy_failed_requests_metric.name)) == 1
    assert value_of(prometheus_logger.litellm_proxy_failed_requests_metric.name, failed_labels) == 1

    total_labels = {
        "end_user": None,
        "hashed_api_key": "test_key",
        "api_key_alias": "test_alias",
        "requested_model": "gpt-5-mini",
        "team": "test_team",
        "team_alias": "test_team_alias",
        "org_id": None,
        "org_alias": None,
        "user": "test_user",
        "status_code": "429",
        "user_email": None,
        "route": user_api_key_dict.request_route,
        "model_id": None,
        "client_ip": None,
        "user_agent": None,
        "api_provider": "openai",
    }
    assert len(series(prometheus_logger.litellm_proxy_total_requests_metric.name)) == 1
    assert value_of(prometheus_logger.litellm_proxy_total_requests_metric.name, total_labels) == 1


@pytest.mark.asyncio
async def test_async_post_call_success_hook(prometheus_logger):
    data = {"model": "gpt-5-mini"}

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_key",
        key_alias="test_alias",
        team_id="test_team",
        team_alias="test_team_alias",
        user_id="test_user",
        end_user_id="test_end_user",
        request_route="/chat/completions",
    )

    response = {"choices": [{"message": {"content": "test response"}}]}

    await prometheus_logger.async_post_call_success_hook(
        data=data, user_api_key_dict=user_api_key_dict, response=response
    )

    assert series(prometheus_logger.litellm_proxy_total_requests_metric.name) == []


@pytest.mark.asyncio
async def test_set_llm_deployment_success_metrics(prometheus_logger):
    standard_logging_payload = create_standard_logging_payload()

    standard_logging_payload["hidden_params"]["additional_headers"] = {
        "x_ratelimit_remaining_requests": 123,
        "x_ratelimit_remaining_tokens": 4321,
    }
    standard_logging_payload["model_group"] = "my_custom_model_group"
    standard_logging_payload["hidden_params"]["litellm_overhead_time_ms"] = 100
    standard_logging_payload["metadata"]["requester_ip_address"] = None
    standard_logging_payload["requester_ip_address"] = None

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=1)

    kwargs = _success_kwargs(
        standard_logging_payload,
        litellm_params={
            "custom_llm_provider": "openai",
            "metadata": {"model_info": {"id": "model-123"}},
        },
        start_time=start_time,
        api_call_start_time=start_time,
        completion_start_time=start_time,
        end_time=end_time,
    )

    await native_success(prometheus_logger, kwargs, MagicMock())

    remaining_labels = {
        "model_group": "my_custom_model_group",
        "api_provider": "openai",
        "api_base": "https://api.openai.com",
        "litellm_model_name": "gpt-5-mini",
        "hashed_api_key": "test_hash",
        "api_key_alias": "test_alias",
        "model_id": "model-123",
    }
    assert series(prometheus_logger.litellm_remaining_requests_metric.name) == [(remaining_labels, 123)]
    assert series(prometheus_logger.litellm_remaining_tokens_metric.name) == [(remaining_labels, 4321)]

    assert _deployment_state(prometheus_logger) == [
        (
            {
                "litellm_model_name": "gpt-5-mini",
                "model_id": "model-123",
                "api_base": "https://api.openai.com",
                "api_provider": "openai",
            },
            0,
        )
    ]

    deployment_labels = {
        "litellm_model_name": "gpt-5-mini",
        "model_id": "model-123",
        "api_base": "https://api.openai.com",
        "api_provider": "openai",
        "requested_model": "my_custom_model_group",
        "model_group": "my_custom_model_group",
        "hashed_api_key": "test_hash",
        "api_key_alias": "test_alias",
        "team": "test_team",
        "team_alias": "test_team_alias",
        "client_ip": None,
        "user_agent": None,
    }
    assert len(series(prometheus_logger.litellm_deployment_success_responses.name)) == 1
    assert value_of(prometheus_logger.litellm_deployment_success_responses.name, deployment_labels) == 1
    assert len(series(prometheus_logger.litellm_deployment_total_requests.name)) == 1
    assert value_of(prometheus_logger.litellm_deployment_total_requests.name, deployment_labels) == 1

    per_token_labels = {
        "model_group": "my_custom_model_group",
        "litellm_model_name": "gpt-5-mini",
        "model_id": "model-123",
        "api_base": "https://api.openai.com",
        "api_provider": "openai",
        "hashed_api_key": "test_hash",
        "api_key_alias": "test_alias",
        "team": "test_team",
        "team_alias": "test_team_alias",
        "org_id": None,
        "org_alias": None,
    }
    assert len(series(prometheus_logger.litellm_overhead_latency_metric.name)) == 1
    assert len(observations(prometheus_logger.litellm_overhead_latency_metric.name, remaining_labels)) == 1

    expected_latency_per_token = 0.1
    assert observations(prometheus_logger.litellm_deployment_latency_per_output_token.name, per_token_labels) == [
        (1, expected_latency_per_token)
    ]


_FALLBACK_KWARGS = {
    "model": "gpt-5.5",
    "metadata": {
        "user_api_key_hash": "test_hash",
        "user_api_key_alias": "test_alias",
        "user_api_key_team_id": "test_team",
        "user_api_key_team_alias": "test_team_alias",
    },
}

_FALLBACK_LABELS = {
    "requested_model": "gpt-5-mini",
    "fallback_model": "gpt-5.5",
    "hashed_api_key": "test_hash",
    "api_key_alias": "test_alias",
    "team": "test_team",
    "team_alias": "test_team_alias",
    "exception_status": "429",
    "exception_class": "Openai.RateLimitError",
    "model_id": None,
}


@pytest.mark.asyncio
async def test_log_success_fallback_event(prometheus_logger, known_model_router):
    original_exception = litellm.RateLimitError(message="Test error", llm_provider="openai", model="gpt-5-mini")

    await prometheus_logger.log_success_fallback_event(
        original_model_group="gpt-5-mini",
        kwargs=dict(_FALLBACK_KWARGS),
        original_exception=original_exception,
    )

    assert len(series(prometheus_logger.litellm_deployment_successful_fallbacks.name)) == 1
    assert value_of(prometheus_logger.litellm_deployment_successful_fallbacks.name, _FALLBACK_LABELS) == 1


@pytest.mark.asyncio
async def test_log_failure_fallback_event(prometheus_logger, known_model_router):
    original_exception = litellm.RateLimitError(message="Test error", llm_provider="openai", model="gpt-5-mini")

    await prometheus_logger.log_failure_fallback_event(
        original_model_group="gpt-5-mini",
        kwargs=dict(_FALLBACK_KWARGS),
        original_exception=original_exception,
    )

    assert len(series(prometheus_logger.litellm_deployment_failed_fallbacks.name)) == 1
    assert value_of(prometheus_logger.litellm_deployment_failed_fallbacks.name, _FALLBACK_LABELS) == 1


def test_deployment_state_management(prometheus_logger):
    test_params = {
        "litellm_model_name": "gpt-5-mini",
        "model_id": "model-123",
        "api_base": "https://api.openai.com",
        "api_provider": "openai",
    }

    prometheus_logger.set_deployment_healthy(**test_params)
    assert _deployment_state(prometheus_logger) == [(test_params, 0)]

    prometheus_logger.set_deployment_partial_outage(**test_params)
    assert _deployment_state(prometheus_logger) == [(test_params, 1)]

    prometheus_logger.set_deployment_complete_outage(**test_params)
    assert _deployment_state(prometheus_logger) == [(test_params, 2)]


def test_increment_deployment_cooled_down(prometheus_logger):
    import inspect

    method_sig = inspect.signature(prometheus_logger.increment_deployment_cooled_down)
    expected_label_count = len([p for p in method_sig.parameters.keys() if p != "self"])

    prometheus_logger.increment_deployment_cooled_down(
        litellm_model_name="gpt-5-mini",
        model_id="model-123",
        api_base="https://api.openai.com",
        api_provider="openai",
        exception_status="429",
    )

    cooled = series(prometheus_logger.litellm_deployment_cooled_down.name)
    assert len(cooled) == 1
    labels, count = cooled[0]
    assert len(labels) == expected_label_count
    assert list(labels.values()) == ["gpt-5-mini", "model-123", "https://api.openai.com", "openai", "429"]
    assert count == 1


@pytest.mark.asyncio
async def test_async_log_success_event_with_top_level_metadata(monkeypatch):
    monkeypatch.setattr("litellm.custom_prometheus_metadata_labels", ["requester_ip_address"])
    _clear_registry()
    prometheus_logger = PrometheusLogger()

    standard_logging_object = create_standard_logging_payload()
    standard_logging_object["metadata"]["requester_ip_address"] = "10.48.203.20"
    standard_logging_object["metadata"]["requester_metadata"] = {}
    standard_logging_object["metadata"]["user_api_key_auth_metadata"] = {}

    kwargs = {
        "model": "gpt-5-mini",
        "stream": True,
        "litellm_params": {
            "metadata": {
                "user_api_key": "test_key",
                "user_api_key_user_id": "test_user",
                "user_api_key_team_id": "test_team",
                "user_api_key_end_user_id": "test_end_user",
            }
        },
        "start_time": datetime.now(),
        "completion_start_time": datetime.now(),
        "api_call_start_time": datetime.now(),
        "end_time": datetime.now() + timedelta(seconds=1),
        "standard_logging_object": standard_logging_object,
    }

    await native_success(prometheus_logger, kwargs, MagicMock(), kwargs["start_time"], kwargs["end_time"])

    assert series(prometheus_logger.litellm_requests_metric.name)
    assert series(prometheus_logger.litellm_spend_metric.name)


def _hours(name: str) -> list[float]:
    return [value for _, value in series(name)]


@pytest.mark.asyncio(scope="session")
async def test_initialize_remaining_budget_metrics(prometheus_logger, monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_initialize_budget_metrics", True)
    with (
        patch("litellm.proxy.proxy_server.prisma_client"),
        patch("litellm.proxy.management_endpoints.team_endpoints.get_paginated_teams") as mock_get_teams,
    ):
        future_reset = datetime.now() + timedelta(hours=24)
        mock_teams = [
            MagicMock(team_id="team1", team_alias="alias1", max_budget=100, spend=30, budget_reset_at=future_reset),
            MagicMock(team_id="team2", team_alias="alias2", max_budget=200, spend=50, budget_reset_at=future_reset),
            MagicMock(team_id="team3", team_alias=None, max_budget=300, spend=100, budget_reset_at=future_reset),
        ]
        mock_get_teams.return_value = (mock_teams, len(mock_teams))

        await prometheus_logger._initialize_remaining_budget_metrics()

        remaining = {
            (labels["team"], labels["team_alias"]): value
            for labels, value in series(prometheus_logger.litellm_remaining_team_budget_metric.name)
        }
        assert remaining == {("team1", "alias1"): 70, ("team2", "alias2"): 150, ("team3", ""): 200}

        hours = series(prometheus_logger.litellm_team_budget_remaining_hours_metric.name)
        assert len(hours) == 3
        for _, value in hours:
            assert 23.9 <= value <= 24.0, f"Hours value {value} not within expected range"
        assert sorted((labels["team"], labels["team_alias"]) for labels, _ in hours) == [
            ("team1", "alias1"),
            ("team2", "alias2"),
            ("team3", ""),
        ]


@pytest.mark.asyncio
async def test_initialize_remaining_budget_metrics_exception_handling(prometheus_logger, monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_initialize_budget_metrics", True)
    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.management_endpoints.team_endpoints.get_paginated_teams") as mock_get_teams,
        patch("litellm.proxy.management_endpoints.key_management_endpoints.list_key_helper") as mock_list_keys,
    ):
        mock_get_teams.side_effect = Exception("Database error")
        mock_list_keys.side_effect = Exception("Key listing error")

        mock_usertable = MagicMock()
        mock_usertable.find_many = MagicMock(side_effect=Exception("User database error"))
        mock_usertable.count = MagicMock(side_effect=Exception("User count error"))

        mock_teamtable = MagicMock()
        mock_teamtable.count = MagicMock(side_effect=Exception("Team count error"))

        mock_orgtable = MagicMock()
        mock_orgtable.find_many = MagicMock(side_effect=Exception("Org database error"))
        mock_orgtable.count = MagicMock(side_effect=Exception("Org count error"))

        mock_db = MagicMock()
        mock_db.litellm_usertable = mock_usertable
        mock_db.litellm_teamtable = mock_teamtable
        mock_db.litellm_organizationtable = mock_orgtable
        mock_prisma.db = mock_db

        with patch("litellm._logging.verbose_logger.exception") as mock_logger:
            await prometheus_logger._initialize_remaining_budget_metrics()

            assert mock_logger.call_count == 5
            logged = [call.args[0] % call.args[1:] for call in mock_logger.call_args_list]
            assert logged == [
                "Error initializing teams budget metrics: Database error",
                "Error initializing keys budget metrics: Key listing error",
                "Error initializing users budget metrics: User database error",
                "Error initializing orgs budget metrics: Org database error",
                "Error initializing user/team count metrics: User count error",
            ]

        for metric in (
            prometheus_logger.litellm_remaining_team_budget_metric,
            prometheus_logger.litellm_remaining_api_key_budget_metric,
            prometheus_logger.litellm_remaining_user_budget_metric,
            prometheus_logger.litellm_remaining_org_budget_metric,
            prometheus_logger.litellm_total_users_metric,
            prometheus_logger.litellm_teams_count_metric,
        ):
            assert series(metric.name) in ([], [({}, 0.0)]), metric.name


@pytest.mark.asyncio(scope="session")
async def test_initialize_api_key_budget_metrics(prometheus_logger, monkeypatch):
    monkeypatch.setattr(litellm, "prometheus_initialize_budget_metrics", True)
    with (
        patch("litellm.proxy.proxy_server.prisma_client"),
        patch("litellm.proxy.management_endpoints.key_management_endpoints.list_key_helper") as mock_list_keys,
    ):
        future_reset = datetime.now() + timedelta(hours=24)
        key1 = UserAPIKeyAuth(
            api_key="key1_hash",
            key_alias="alias1",
            team_id="team1",
            max_budget=100,
            spend=30,
            budget_reset_at=future_reset,
        )
        key1.token = "key1_hash"
        key2 = UserAPIKeyAuth(
            api_key="key2_hash",
            key_alias="alias2",
            team_id="team2",
            max_budget=200,
            spend=50,
            budget_reset_at=future_reset,
        )
        key2.token = "key2_hash"
        key3 = UserAPIKeyAuth(
            api_key="key3_hash",
            key_alias=None,
            team_id="team3",
            max_budget=300,
            spend=100,
            budget_reset_at=future_reset,
        )
        key3.token = "key3_hash"
        mock_keys = [key1, key2, key3]
        mock_list_keys.return_value = {"keys": mock_keys, "total_count": len(mock_keys)}

        await prometheus_logger._initialize_api_key_budget_metrics()

        def by_key(name: str) -> dict[tuple[str, str], float]:
            return {(labels["hashed_api_key"], labels["api_key_alias"]): value for labels, value in series(name)}

        assert by_key(prometheus_logger.litellm_remaining_api_key_budget_metric.name) == {
            ("key1_hash", "alias1"): 70,
            ("key2_hash", "alias2"): 150,
            ("key3_hash", ""): 200,
        }

        hours = _hours(prometheus_logger.litellm_api_key_budget_remaining_hours_metric.name)
        assert len(hours) == 3
        for value in hours:
            assert 23.9 <= value <= 24.0, f"Hours value {value} not within expected range"

        assert by_key(prometheus_logger.litellm_api_key_max_budget_metric.name) == {
            ("key1_hash", "alias1"): 100,
            ("key2_hash", "alias2"): 200,
            ("key3_hash", ""): 300,
        }


def test_set_team_budget_metrics_multiple_teams(prometheus_logger):
    teams = [
        MagicMock(
            team_id="team1",
            team_alias="alias1",
            spend=50.0,
            max_budget=100.0,
            budget_reset_at=datetime(2024, 12, 31, tzinfo=timezone.utc),
        ),
        MagicMock(
            team_id="team2",
            team_alias="alias2",
            spend=75.0,
            max_budget=150.0,
            budget_reset_at=datetime(2024, 6, 30, tzinfo=timezone.utc),
        ),
        MagicMock(
            team_id="team3",
            team_alias="alias3",
            spend=25.0,
            max_budget=200.0,
            budget_reset_at=datetime(2024, 3, 31, tzinfo=timezone.utc),
        ),
    ]

    for team in teams:
        prometheus_logger._set_team_budget_metrics(team)

    def by_team(name: str) -> dict[tuple[str, str], float]:
        return {(labels["team"], labels["team_alias"]): value for labels, value in series(name)}

    assert by_team(prometheus_logger.litellm_remaining_team_budget_metric.name) == {
        ("team1", "alias1"): 50.0,
        ("team2", "alias2"): 75.0,
        ("team3", "alias3"): 175.0,
    }
    assert by_team(prometheus_logger.litellm_team_max_budget_metric.name) == {
        ("team1", "alias1"): 100.0,
        ("team2", "alias2"): 150.0,
        ("team3", "alias3"): 200.0,
    }
    assert len(series(prometheus_logger.litellm_team_budget_remaining_hours_metric.name)) == 3


def test_set_team_budget_metrics_null_values(prometheus_logger):
    team = MagicMock(
        team_id="team_null",
        team_alias=None,
        spend=None,
        max_budget=None,
        budget_reset_at=None,
    )

    prometheus_logger._set_team_budget_metrics(team)

    assert series(prometheus_logger.litellm_remaining_team_budget_metric.name) == [
        ({"team": "team_null", "team_alias": ""}, float("inf"))
    ]
    assert series(prometheus_logger.litellm_team_max_budget_metric.name) == []
    assert series(prometheus_logger.litellm_team_budget_remaining_hours_metric.name) == []


def test_set_team_budget_metrics_with_custom_labels(monkeypatch):
    custom_labels = ["metadata.organization", "metadata.environment"]
    monkeypatch.setattr("litellm.custom_prometheus_metadata_labels", custom_labels)
    _clear_registry()
    prometheus_logger = PrometheusLogger()

    team = MagicMock(
        team_id="team1",
        team_alias="alias1",
        spend=50.0,
        max_budget=100.0,
        budget_reset_at=datetime(2024, 12, 31, tzinfo=timezone.utc),
    )

    prometheus_logger._set_team_budget_metrics(team)

    labels = {
        "team": "team1",
        "team_alias": "alias1",
        "metadata_organization": None,
        "metadata_environment": None,
    }
    assert len(series(prometheus_logger.litellm_remaining_team_budget_metric.name)) == 1
    assert value_of(prometheus_logger.litellm_remaining_team_budget_metric.name, labels) == 50.0
    assert len(series(prometheus_logger.litellm_team_max_budget_metric.name)) == 1
    assert value_of(prometheus_logger.litellm_team_max_budget_metric.name, labels) == 100.0


@pytest.mark.asyncio
async def test_set_llm_deployment_success_metrics_with_label_filtering():
    from litellm.types.integrations.prometheus import PrometheusMetricsConfig

    config = [
        PrometheusMetricsConfig(
            group="test_group",
            metrics=[
                "litellm_overhead_latency_metric",
                "litellm_remaining_requests_metric",
                "litellm_remaining_tokens_metric",
                "litellm_deployment_success_responses",
                "litellm_deployment_total_requests",
            ],
            include_labels=[
                "litellm_model_name",
                "api_provider",
                "hashed_api_key",
            ],
        )
    ]

    with patch("litellm.prometheus_metrics_config", config):
        _clear_registry()
        prometheus_logger = PrometheusLogger()

        standard_logging_payload = create_standard_logging_payload()
        standard_logging_payload["hidden_params"]["additional_headers"] = {
            "x_ratelimit_remaining_requests": 123,
            "x_ratelimit_remaining_tokens": 4321,
        }
        standard_logging_payload["hidden_params"]["litellm_overhead_time_ms"] = 100

        kwargs = _success_kwargs(
            standard_logging_payload,
            litellm_params={
                "custom_llm_provider": "openai",
                "metadata": {"model_info": {"id": "model-123"}},
            },
        )

        await native_success(prometheus_logger, kwargs, MagicMock())

        expected_filtered_labels = {
            "litellm_model_name",
            "api_provider",
            "hashed_api_key",
        }
        overhead = series(prometheus_logger.litellm_overhead_latency_metric.name)
        assert len(overhead) == 1
        assert set(overhead[0][0]) <= expected_filtered_labels

        requests = series(prometheus_logger.litellm_remaining_requests_metric.name)
        assert len(requests) == 1
        assert set(requests[0][0]) <= expected_filtered_labels

        tokens = series(prometheus_logger.litellm_remaining_tokens_metric.name)
        assert len(tokens) == 1
        assert set(tokens[0][0]) <= expected_filtered_labels

        success = series(prometheus_logger.litellm_deployment_success_responses.name)
        assert len(success) == 1
        assert set(success[0][0]) <= expected_filtered_labels

        total = series(prometheus_logger.litellm_deployment_total_requests.name)
        assert len(total) == 1
        assert set(total[0][0]).issubset(expected_filtered_labels)

        assert requests[0][1] == 123
        assert tokens[0][1] == 4321
        assert success[0][1] == 1
        assert total[0][1] == 1


@pytest.mark.asyncio
async def test_prometheus_token_metrics_with_prometheus_config():
    from litellm.types.integrations.prometheus import PrometheusMetricsConfig

    _clear_registry()

    config = [
        PrometheusMetricsConfig(
            group="token_metrics_test",
            metrics=[
                "litellm_total_tokens_metric",
                "litellm_input_tokens_metric",
                "litellm_output_tokens_metric",
                "litellm_requests_metric",
            ],
            include_labels=[
                "model",
                "hashed_api_key",
                "api_key_alias",
                "team",
                "team_alias",
            ],
        )
    ]

    with patch("litellm.prometheus_metrics_config", config):
        prometheus_logger = PrometheusLogger()

        standard_logging_payload = create_standard_logging_payload()
        standard_logging_payload["total_tokens"] = 1500
        standard_logging_payload["prompt_tokens"] = 900
        standard_logging_payload["completion_tokens"] = 600
        standard_logging_payload["response_cost"] = 0.075

        kwargs = {
            "model": "gpt-5-mini",
            "stream": False,
            "litellm_params": {
                "metadata": {
                    "user_api_key": "test_key_hash",
                    "user_api_key_user_id": "test_user",
                    "user_api_key_team_id": "test_team",
                    "user_api_key_alias": "test_alias",
                    "user_api_key_team_alias": "test_team_alias",
                }
            },
            "start_time": datetime.now() - timedelta(seconds=2),
            "completion_start_time": datetime.now() - timedelta(seconds=1),
            "api_call_start_time": datetime.now() - timedelta(seconds=1.5),
            "end_time": datetime.now(),
            "standard_logging_object": standard_logging_payload,
        }

        await native_success(prometheus_logger, kwargs, MagicMock(), kwargs["start_time"], kwargs["end_time"])

        await asyncio.sleep(2)

        expected_values = {
            "litellm_total_tokens_metric": 1500.0,
            "litellm_input_tokens_metric": 900.0,
            "litellm_output_tokens_metric": 600.0,
            "litellm_requests_metric": 1.0,
        }
        expected_label_values = {
            "api_key_alias": "test_alias",
            "hashed_api_key": "test_hash",
            "model": "gpt-5-mini",
            "team": "test_team",
            "team_alias": "test_team_alias",
        }

        for metric_name, expected_value in expected_values.items():
            found = series(metric_name)
            assert found, f"Metric {metric_name} not found in registry"
            actual_labels, actual_value = found[0]
            assert actual_value == expected_value, f"Expected {expected_value}, got {actual_value} for {metric_name}"
            for label_key, expected_label_value in expected_label_values.items():
                assert actual_labels.get(label_key) == expected_label_value
