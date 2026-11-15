import datetime
from typing import Final, Literal

import pytest
from pydantic import JsonValue

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth
from tests.unit.integrations.prometheus_native_support import (
    enable_native_prometheus,
    native_failure,
    native_success,
    series,
)


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


def _clear_prometheus_registry() -> None:
    from prometheus_client import REGISTRY

    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def _collected_samples(metric_name: str):
    from prometheus_client import REGISTRY

    return [sample for metric in REGISTRY.collect() for sample in metric.samples if sample.name == metric_name]


def _native_logger() -> PrometheusLogger:
    logger = PrometheusLogger()
    assert logger._native_store is not None
    return logger


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "t",
        "call_type": "completion",
        "response_cost": 0.001,
        "status": "success",
        "stream": True,
        "total_tokens": 30,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "startTime": 1.0,
        "endTime": 2.0,
        "completionStartTime": 1.5,
        "model": "gpt-4o-mini",
        "model_id": "model-123",
        "model_group": "gpt-4o-mini",
        "api_base": "https://api.openai.com",
        "custom_llm_provider": "openai",
        "request_tags": [],
        "end_user": None,
        "cache_hit": False,
        "metadata": {
            "user_api_key_hash": "h",
            "user_api_key_alias": "a",
            "user_api_key_team_id": "t",
            "user_api_key_team_alias": "ta",
            "user_api_key_user_id": "u",
            "user_api_key_user_email": "e@x.com",
            "user_api_key_org_id": None,
            "user_api_key_org_alias": None,
            "requester_metadata": None,
            "user_api_key_end_user_id": None,
        },
        "hidden_params": {"litellm_overhead_time_ms": None, "additional_headers": None},
    }
    payload.update(overrides)
    return payload


def _streamed_kwargs(payload: dict[str, object], litellm_params: dict[str, object] | None = None) -> dict[str, object]:
    start_time = datetime.datetime(2024, 1, 1, 0, 0, 0)
    return {
        "model": "gpt-4o-mini",
        "stream": True,
        "litellm_params": litellm_params if litellm_params is not None else {"metadata": {}},
        "standard_logging_object": payload,
        "start_time": start_time,
        "api_call_start_time": datetime.datetime(2024, 1, 1, 0, 0, 1),
        "completion_start_time": datetime.datetime(2024, 1, 1, 0, 0, 2),
        "end_time": datetime.datetime(2024, 1, 1, 0, 0, 3),
    }


def _label_values(metric_name: str, label: str) -> set[str]:
    return {labels[label] for labels, _ in series(metric_name)}


@pytest.mark.asyncio
async def test_api_provider_value_flows_through_label_factory():
    _clear_prometheus_registry()
    try:
        logger = _native_logger()
        await native_success(
            logger,
            _streamed_kwargs(
                _payload(custom_llm_provider="anthropic", model="claude-sonnet-4", model_group="claude", cache_hit=True)
            ),
            None,
        )
        for metric_name in [
            "litellm_input_tokens_metric",
            "litellm_total_tokens_metric",
            "litellm_output_tokens_metric",
            "litellm_llm_api_latency_metric",
            "litellm_request_total_latency_metric",
            "litellm_proxy_total_requests_metric",
            "litellm_cache_hits_metric",
        ]:
            values = _label_values(metric_name, "api_provider")
            assert values == {"anthropic"}, f"{metric_name} should emit api_provider=anthropic, got {values!r}"
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_prometheus_metrics_use_normalized_routes():
    _clear_prometheus_registry()
    try:
        logger = _native_logger()
        payload = _payload(model_group="gpt-4")
        payload["metadata"]["user_api_key_request_route"] = "/v1/responses/{response_id}"
        await native_success(logger, _streamed_kwargs(payload), None)

        routes = _label_values("litellm_proxy_total_requests_metric", "route")
        assert routes == {"/v1/responses/{response_id}"}, f"Expected normalized route in labels, got: {routes}"
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_prometheus_label_value_sanitization():
    _clear_prometheus_registry()
    try:
        logger = _native_logger()
        payload = _payload(model_group="claude-haiku-4-5-20251001 ")
        payload["metadata"]["user_api_key_alias"] = 'My Key "test"\nwith newline'
        payload["metadata"]["user_api_key_request_route"] = "/v1/chat/completions"
        await native_success(logger, _streamed_kwargs(payload), None)

        found = series("litellm_proxy_total_requests_metric")
        assert len(found) == 1
        labels = found[0][0]
        assert " " not in labels["requested_model"]
        assert labels["requested_model"] == "claude-haiku-4-5-20251001"
        assert "\n" not in labels["api_key_alias"]
        assert labels["api_key_alias"] == 'My Key \\"test\\" with newline'
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_bucket", ("metadata", "litellm_metadata"))
@pytest.mark.parametrize("populate_other_bucket", (False, True))
@pytest.mark.parametrize("original_model_group", (None, "", 123, "requested-source"))
@pytest.mark.parametrize("stream", (False, True))
async def test_success_hook_preserves_requested_model_and_deployment_labels(
    metadata_bucket: Literal["metadata", "litellm_metadata"],
    populate_other_bucket: bool,
    original_model_group: str | int | None,
    stream: bool,
) -> None:
    payload: Final = {
        "id": "t",
        "call_type": "completion",
        "response_cost": 0.001,
        "status": "success",
        "stream": stream,
        "total_tokens": 30,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "startTime": 1.0,
        "endTime": 2.0,
        "completionStartTime": 1.5,
        "model": "gpt-4o-mini",
        "model_id": "model-123",
        "model_group": "gpt-4o-mini",
        "api_base": "https://api.openai.com",
        "custom_llm_provider": "openai",
        "request_tags": [],
        "end_user": None,
        "cache_hit": False,
        "metadata": {
            "user_api_key_hash": "h",
            "user_api_key_alias": "a",
            "user_api_key_team_id": "t",
            "user_api_key_team_alias": "ta",
            "user_api_key_user_id": "u",
            "user_api_key_user_email": "e@x.com",
            "user_api_key_org_id": None,
            "user_api_key_org_alias": None,
            "requester_metadata": None,
            "user_api_key_end_user_id": None,
        },
        "hidden_params": {"litellm_overhead_time_ms": None, "additional_headers": None},
    }

    other_bucket: Final = "litellm_metadata" if metadata_bucket == "metadata" else "metadata"
    litellm_params: Final = {
        metadata_bucket: {
            "request_tag": "router",
            **({"original_model_group": original_model_group} if original_model_group is not None else {}),
        },
        **({other_bucket: {"request_tag": "client"}} if populate_other_bucket else {}),
    }
    _clear_prometheus_registry()
    try:
        logger: Final = _native_logger()
        now: Final = datetime.datetime.fromtimestamp(1, tz=datetime.timezone.utc)
        await native_success(
            logger,
            {
                "model": "gpt-4o-mini",
                "litellm_params": litellm_params,
                "standard_logging_object": payload,
                "start_time": now,
                "end_time": now,
            },
            None,
            now,
            now,
        )
        request_samples: Final = _collected_samples("litellm_proxy_total_requests_metric_total")
        assert len(request_samples) == 1
        assert request_samples[0].value == 1
        expected_requested_model: Final = (
            original_model_group
            if isinstance(original_model_group, str) and original_model_group
            else payload["model_group"]
        )
        assert request_samples[0].labels["requested_model"] == expected_requested_model
        assert request_samples[0].labels["model_id"] == payload["model_id"]
        assert request_samples[0].labels["api_provider"] == payload["custom_llm_provider"]
        assert request_samples[0].labels["status_code"] == "200"
        samples: Final = _collected_samples("litellm_total_tokens_metric_total")
        assert samples, "expected litellm_total_tokens_metric to be emitted"
        assert all(s.labels.get("api_provider") == "openai" for s in samples), (
            f"collected token metric must carry api_provider=openai, got {[s.labels.get('api_provider') for s in samples]}"
        )
        assert all(sample.labels["requested_model"] == payload["model_group"] for sample in samples)
        spend_samples: Final = _collected_samples("litellm_spend_metric_total")
        assert len(spend_samples) == 1
        assert spend_samples[0].value == payload["response_cost"]
        assert spend_samples[0].labels["requested_model"] == payload["model_group"]
        assert spend_samples[0].labels["model"] == payload["model"]
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_failure_hook_emits_api_provider_value_on_failed_requests_metric():
    _clear_prometheus_registry()
    try:
        logger = _native_logger()
        await logger.async_post_call_failure_hook(
            request_data={"model": "gpt-4o-mini", "metadata": {}},
            original_exception=Exception("boom"),
            user_api_key_dict=UserAPIKeyAuth(token="tok"),
        )
        samples = _collected_samples("litellm_proxy_failed_requests_metric_total")
        assert samples, "expected litellm_proxy_failed_requests_metric to be emitted"
        assert any(s.labels.get("api_provider") == "openai" for s in samples), (
            "collected failed-requests metric must carry api_provider=openai, got "
            f"{[s.labels.get('api_provider') for s in samples]}"
        )
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata_id", "request_id", "logged_id", "expected"),
    (
        (None, None, "logged-deployment", "logged-deployment"),
        ("", None, "logged-deployment", "logged-deployment"),
        ("metadata-deployment", "request-deployment", "logged-deployment", "metadata-deployment"),
        (None, "request-deployment", "logged-deployment", "request-deployment"),
        (None, None, None, "None"),
    ),
)
async def test_failure_counters_keep_logged_deployment_attribution(
    metadata_id: str | None, request_id: str | None, logged_id: str | None, expected: str
) -> None:
    from fastapi import HTTPException

    request_data: Final[dict[str, JsonValue]] = {
        "model": "metric-test-alias",
        "metadata": {"model_info": {"id": metadata_id}},
        "model_info": {"id": request_id},
        "standard_logging_object": {"model_id": logged_id, "custom_llm_provider": "openai"},
    }
    _clear_prometheus_registry()
    try:
        await _native_logger().async_post_call_failure_hook(
            request_data=request_data,
            original_exception=HTTPException(status_code=400, detail="guardrail rejected"),
            user_api_key_dict=UserAPIKeyAuth(token="metric-test-token"),
        )
        for metric_name in ("litellm_proxy_failed_requests_metric_total", "litellm_proxy_total_requests_metric_total"):
            samples: Final = _collected_samples(metric_name)
            assert [(sample.labels["model_id"], sample.value) for sample in samples] == [(expected, 1.0)]
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_failure_hook_labels_a_cli_session_with_the_per_user_alias_not_the_login_token():
    _clear_prometheus_registry()
    try:
        await _native_logger().async_post_call_failure_hook(
            request_data={"model": "gpt-4o-mini", "metadata": {}},
            original_exception=Exception("boom"),
            user_api_key_dict=UserAPIKeyAuth(
                api_key="cli-session-Qm7xJ2kP9sLw4vT1nR8yAa",
                user_id="alice",
                key_alias="cli-session-alice",
                is_session_token=True,
            ),
        )
        hashed_keys = {
            s.labels.get("hashed_api_key") for s in _collected_samples("litellm_proxy_failed_requests_metric_total")
        }
        assert hashed_keys == {"cli-session-alice"}, hashed_keys
    finally:
        _clear_prometheus_registry()


async def _failed_requests_api_provider_labels(
    request_data: dict[str, object],
    original_exception: Exception,
) -> list[str]:
    _clear_prometheus_registry()
    try:
        await _native_logger().async_post_call_failure_hook(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=UserAPIKeyAuth(token="tok"),
        )
        return [s.labels.get("api_provider") for s in _collected_samples("litellm_proxy_failed_requests_metric_total")]
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_failure_hook_emits_api_provider_from_pre_call_rate_limit_error_for_router_alias():
    from litellm.exceptions import RateLimitType
    from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError

    err = ProxyRateLimitError(
        detail={"error": "rpm exceeded"},
        rate_limit_type=RateLimitType.REQUESTS,
        model="openai/gpt-5.4-mini",
        llm_provider="openai",
    )

    assert await _failed_requests_api_provider_labels({"model": "team-chat-model", "metadata": {}}, err) == ["openai"]


@pytest.mark.asyncio
async def test_failure_hook_leaves_api_provider_unset_when_rate_limiter_could_not_resolve_provider():
    from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError

    err = ProxyRateLimitError(detail={"error": "rpm exceeded"}, model="unknown-alias")

    assert await _failed_requests_api_provider_labels({"model": "unknown-alias", "metadata": {}}, err) == ["None"]


@pytest.mark.asyncio
async def test_failure_hook_prefers_request_data_provider_over_exception_provider():
    from litellm.exceptions import RateLimitError

    err = RateLimitError(message="upstream 429", llm_provider="openai", model="gpt-4o")

    assert await _failed_requests_api_provider_labels(
        {
            "model": "gpt-4o",
            "metadata": {},
            "litellm_params": {"custom_llm_provider": "azure"},
        },
        err,
    ) == ["azure"]


_PASS_THROUGH_CALL_TYPE: Final = "pass_through_endpoint"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type, custom_llm_provider, expected_requested_model",
    [
        (_PASS_THROUGH_CALL_TYPE, "vertex_ai", "gcp/google/gemini-2.5-flash"),
        ("acompletion", "vertex_ai", "other"),
        (_PASS_THROUGH_CALL_TYPE, "bedrock", "other"),
    ],
)
async def test_failure_hook_labels_a_pass_through_failure_with_the_router_model_the_client_requested(
    call_type: str, custom_llm_provider: str, expected_requested_model: str
):
    from unittest.mock import patch

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gcp/google/gemini-2.5-flash",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-flash",
                    "vertex_project": "p",
                    "vertex_location": "us-central1",
                },
            }
        ]
    )
    _clear_prometheus_registry()
    try:
        with patch(
            "litellm.proxy.proxy_server.llm_router", router, create=True
        ):  # test-quality-ok: production reads proxy_server.llm_router lazily, no injection seam
            await _native_logger().async_post_call_failure_hook(
                request_data={
                    "model": "gemini-2.5-flash",
                    "call_type": call_type,
                    "custom_llm_provider": custom_llm_provider,
                    "metadata": {},
                    "litellm_params": {"metadata": {"model_group": "gcp/google/gemini-2.5-flash"}},
                },
                original_exception=Exception("upstream 400"),
                user_api_key_dict=UserAPIKeyAuth(token="tok"),
            )
        assert {
            s.labels.get("requested_model") for s in _collected_samples("litellm_proxy_failed_requests_metric_total")
        } == {expected_requested_model}
    finally:
        _clear_prometheus_registry()


def _failure_kwargs(litellm_params: dict[str, object], standard_logging_object: dict[str, object]) -> dict[str, object]:
    now = datetime.datetime.now()
    return {
        "model": "gpt-4o-mini",
        "litellm_params": litellm_params,
        "standard_logging_object": {
            **_payload(status="failure", custom_llm_provider=None, model_id=None, api_base=None),
            **standard_logging_object,
        },
        "exception": Exception("boom"),
        "start_time": now,
        "end_time": now,
    }


@pytest.mark.asyncio
async def test_model_group_value_flows_through_deployment_metrics_label_factory():
    _clear_prometheus_registry()
    try:
        logger = _native_logger()
        payload = _payload(model_group="example-model-group")
        await native_success(
            logger,
            _streamed_kwargs(
                payload,
                {"metadata": {"model_info": {"id": "model-123", "tpm": 1000, "rpm": 10}}},
            ),
            None,
        )
        await native_failure(
            logger,
            _failure_kwargs({"metadata": {}}, {"model_group": "example-model-group", "model_id": "model-123"}),
            None,
        )
        for metric_name in [
            "litellm_deployment_total_requests",
            "litellm_deployment_success_responses",
            "litellm_deployment_failure_responses",
            "litellm_deployment_tpm_limit",
            "litellm_deployment_rpm_limit",
        ]:
            values = _label_values(metric_name, "model_group")
            assert values == {"example-model-group"}, (
                f"{metric_name} should emit model_group=example-model-group, got {values!r}"
            )
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_deployment_failure_metrics_emit_model_group_from_standard_logging_payload():
    _clear_prometheus_registry()
    try:
        logger = _native_logger()
        await native_failure(
            logger,
            _failure_kwargs(
                {"metadata": {}},
                {
                    "model_group": "example-model-group",
                    "model_id": "model-123",
                    "api_base": "https://api.openai.com",
                    "request_tags": [],
                },
            ),
            None,
        )

        for metric in (
            logger.litellm_deployment_failure_responses,
            logger.litellm_deployment_total_requests,
        ):
            values = _label_values(metric.name, "model_group")
            assert values == {"example-model-group"}, (
                f"expected model_group=example-model-group on {metric.name}, got {values}"
            )
    finally:
        _clear_prometheus_registry()


def _routed(deployment: object) -> dict[str, object]:
    return {"deployment": deployment, "model_info": {"id": "model-123"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("litellm_params", "logged_provider", "expected"),
    [
        pytest.param(
            {"custom_llm_provider": "bedrock", "metadata": _routed("azure/gpt-4o")},
            "vertex_ai",
            "bedrock",
            id="resolved-provider-wins",
        ),
        pytest.param(
            {"custom_llm_provider": None, "metadata": _routed("anthropic/claude-haiku-4-5")},
            "openai",
            "openai",
            id="provider-on-the-payload-wins",
        ),
        pytest.param({"metadata": _routed("azure/limited-azure")}, None, "azure", id="deployment-prefix"),
        pytest.param(
            {"litellm_metadata": _routed("anthropic/claude-haiku-4-5"), "metadata": {}},
            None,
            "anthropic",
            id="litellm-metadata-deployment",
        ),
        pytest.param(
            {"litellm_metadata": _routed("anthropic/claude-haiku-4-5"), "metadata": _routed("bedrock/converse/model")},
            None,
            "anthropic",
            id="litellm-metadata-deployment-wins",
        ),
        pytest.param(
            {"litellm_metadata": {"deployment": "bedrock/forged"}, "metadata": _routed("anthropic/claude-haiku-4-5")},
            None,
            "anthropic",
            id="routed-bucket-wins-over-a-bare-deployment",
        ),
        pytest.param({"metadata": _routed("abliteration/model")}, None, "abliteration", id="json-configured-provider"),
        pytest.param({"metadata": _routed("langfuse/my-prompt")}, None, "", id="prompt-management-deployment"),
        pytest.param({"metadata": _routed("gpt-4o")}, None, "", id="unprefixed-deployment"),
        pytest.param({"metadata": _routed("meta-llama/Llama-3")}, None, "", id="prefix-is-not-a-provider"),
        pytest.param({"metadata": _routed(5)}, None, "", id="non-string-deployment"),
        pytest.param({"metadata": {}}, None, "", id="nothing-known"),
    ],
)
async def test_deployment_failure_metrics_label_the_selected_deployment_provider(
    litellm_params, logged_provider, expected
):
    _clear_prometheus_registry()
    try:
        logger = _native_logger()
        await native_failure(
            logger,
            _failure_kwargs(
                litellm_params,
                {
                    "model_group": "gpt-4o-mini",
                    "model_id": "model-123",
                    "api_base": "https://example.openai.azure.com",
                    "custom_llm_provider": logged_provider,
                    "request_tags": [],
                },
            ),
            None,
        )

        for metric in (
            logger.litellm_deployment_failure_responses,
            logger.litellm_deployment_total_requests,
        ):
            values = _label_values(metric.name, "api_provider")
            assert values == {expected}, f"expected api_provider={expected!r} on {metric.name}, got {values}"
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_deployment_failure_metrics_label_a_custom_provider_deployment(monkeypatch):
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "in-house", "custom_handler": object()}])
    _clear_prometheus_registry()
    try:
        logger = _native_logger()
        await native_failure(
            logger,
            _failure_kwargs(
                {"metadata": _routed("in-house/model")},
                {
                    "model_group": "gpt-4o-mini",
                    "model_id": "model-123",
                    "api_base": "https://in-house.example",
                    "request_tags": [],
                },
            ),
            None,
        )

        values = _label_values(logger.litellm_deployment_failure_responses.name, "api_provider")
        assert values == {"in-house"}
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_deployment_tpm_rpm_limit_metrics_emit_model_group_from_enum_values():
    _clear_prometheus_registry()
    try:
        logger = _native_logger()
        await native_success(
            logger,
            _streamed_kwargs(
                _payload(model_group="example-model-group"),
                {"metadata": {"model_info": {"id": "model-123", "tpm": 1000, "rpm": 10}}},
            ),
            None,
        )

        for metric in (logger.litellm_deployment_tpm_limit, logger.litellm_deployment_rpm_limit):
            values = _label_values(metric.name, "model_group")
            assert values == {"example-model-group"}, (
                f"expected model_group=example-model-group on {metric.name}, got {values}"
            )
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_model_group_value_flows_through_latency_metrics_label_factory():
    _clear_prometheus_registry()
    try:
        logger = _native_logger()
        await native_success(logger, _streamed_kwargs(_payload(model_group="example-model-group")), None)
        for metric_name in [
            "litellm_llm_api_latency_metric",
            "litellm_llm_api_time_to_first_token_metric",
            "litellm_request_total_latency_metric",
            "litellm_deployment_latency_per_output_token",
        ]:
            values = _label_values(metric_name, "model_group")
            assert values == {"example-model-group"}, (
                f"{metric_name} should emit model_group=example-model-group, got {values!r}"
            )
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_latency_metrics_emit_model_group_from_set_latency_metrics():
    _clear_prometheus_registry()
    try:
        logger = _native_logger()
        await native_success(logger, _streamed_kwargs(_payload(model_group="example-model-group")), None)

        for metric in (
            logger.litellm_llm_api_latency_metric,
            logger.litellm_llm_api_time_to_first_token_metric,
            logger.litellm_request_total_latency_metric,
        ):
            values = _label_values(metric.name, "model_group")
            assert values == {"example-model-group"}, (
                f"expected model_group=example-model-group on {metric.name}, got {values}"
            )
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_deployment_latency_per_output_token_emits_model_group_from_enum_values():
    _clear_prometheus_registry()
    try:
        logger = _native_logger()
        await native_success(logger, _streamed_kwargs(_payload(model_group="example-model-group")), None)

        values = _label_values(logger.litellm_deployment_latency_per_output_token.name, "model_group")
        assert values == {"example-model-group"}, (
            f"expected model_group=example-model-group on {logger.litellm_deployment_latency_per_output_token.name}, got {values}"
        )
    finally:
        _clear_prometheus_registry()
