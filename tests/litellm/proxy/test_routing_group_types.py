"""
Unit tests for Routing Group Pydantic types.
"""
from litellm.types.router import (
    DeploymentTrafficStats,
    FailureInjectionConfig,
    RoutingGroupConfig,
    RoutingGroupDeployment,
    RoutingGroupSimulationResult,
    RoutingTrace,
)


def test_routing_group_deployment_basic():
    d = RoutingGroupDeployment(
        model_id="test-id",
        model_name="nebius/meta-llama/Llama-3.3-70B",
        provider="nebius",
    )
    assert d.priority is None
    assert d.weight is None


def test_routing_group_deployment_with_priority():
    d = RoutingGroupDeployment(
        model_id="test-id",
        model_name="nebius/meta-llama/Llama-3.3-70B",
        provider="nebius",
        priority=1,
    )
    assert d.priority == 1


def test_routing_group_config_basic():
    config = RoutingGroupConfig(
        routing_group_name="test-group",
        routing_strategy="priority-failover",
        deployments=[
            RoutingGroupDeployment(
                model_id="id-1",
                model_name="nebius/meta-llama/Llama-3.3-70B",
                provider="nebius",
                priority=1,
            ),
            RoutingGroupDeployment(
                model_id="id-2",
                model_name="fireworks_ai/llama-v3p3-70b",
                provider="fireworks_ai",
                priority=2,
            ),
        ],
    )
    assert config.routing_group_name == "test-group"
    assert len(config.deployments) == 2
    assert config.is_active is True
    assert config.routing_group_id is None


def test_routing_group_config_weighted():
    config = RoutingGroupConfig(
        routing_group_name="weighted-group",
        routing_strategy="weighted",
        deployments=[
            RoutingGroupDeployment(
                model_id="id-1",
                model_name="nebius/meta-llama/Llama-3.3-70B",
                provider="nebius",
                weight=80,
            ),
            RoutingGroupDeployment(
                model_id="id-2",
                model_name="azure/gpt-4o",
                provider="azure",
                weight=20,
            ),
        ],
    )
    assert config.routing_strategy == "weighted"


def test_routing_trace():
    trace = RoutingTrace(
        deployment_id="id-1",
        deployment_name="nebius/meta-llama/Llama-3.3-70B",
        provider="nebius",
        latency_ms=149.5,
        was_fallback=False,
        status="success",
    )
    assert trace.fallback_depth == 0
    assert trace.error_message is None


def test_routing_group_simulation_result():
    result = RoutingGroupSimulationResult(
        total_requests=100,
        successful_requests=98,
        failed_requests=2,
        avg_latency_ms=169.0,
        fallback_count=5,
        traffic_distribution=[
            DeploymentTrafficStats(
                deployment_id="id-1",
                deployment_name="Nebius",
                provider="nebius",
                request_count=85,
                success_count=85,
                failure_count=0,
                avg_latency_ms=149.0,
                percent_of_total=85.0,
            )
        ],
    )
    assert result.total_requests == 100
    assert result.flow_data is None


def test_failure_injection_config():
    config = FailureInjectionConfig(
        deployment_failure_rates={"id-1": 0.5, "id-2": 0.1}
    )
    assert config.deployment_failure_rates["id-1"] == 0.5
