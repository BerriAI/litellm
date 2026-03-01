"""
Test and simulation engine for routing groups.

Executes test requests through a routing group and collects routing traces
that drive the Live Tester visualization in the UI.
"""

import asyncio
import random
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from litellm._logging import verbose_proxy_logger
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import (
    DeploymentTrafficStats,
    FailureInjectionConfig,
    FlowStep,
    RoutingGroupConfig,
    RoutingGroupSimulationResult,
    RoutingGroupTestResult,
    RoutingTrace,
)
from litellm.types.utils import Choices, ModelResponse

if TYPE_CHECKING:
    from litellm.router import Router


_DEFAULT_MESSAGES: List[Any] = [{"role": "user", "content": "Hello, respond with one word."}]


def _empty_dep_stats() -> Dict[str, Any]:
    return {
        "deployment_name": "unknown",
        "provider": "unknown",
        "request_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "latencies": [],
        "priority": None,
        "weight": None,
    }


class RoutingGroupTestEngine:
    """
    Executes test/simulation requests through a routing group and returns
    structured results for the Live Tester visualization.
    """

    async def test_single_request(
        self,
        routing_group_name: str,
        router: "Router",
        messages: Optional[List[AllMessageValues]] = None,
        mock: bool = False,
    ) -> RoutingGroupTestResult:
        """
        Send one request through the routing group and return the routing trace.

        Uses a RoutingTraceCallback temporarily attached to litellm.callbacks
        to capture which deployments were tried and their outcomes.
        """
        import litellm
        from litellm.proxy.routing_group_utils.routing_trace_callback import (
            RoutingTraceCallback,
        )

        if messages is None:
            messages = cast(List[AllMessageValues], _DEFAULT_MESSAGES)

        trace_callback = RoutingTraceCallback()

        # Attach callback temporarily
        litellm.callbacks.append(trace_callback)

        start = time.monotonic()
        success = False
        response_text: Optional[str] = None

        try:
            if mock:
                response = await router.acompletion(
                    model=routing_group_name,
                    messages=messages,
                    mock_response="Mock routing group test response",
                )
            else:
                response = await router.acompletion(
                    model=routing_group_name,
                    messages=messages,
                )
            success = True
            try:
                _choices = cast(List[Choices], cast(ModelResponse, response).choices)
                response_text = _choices[0].message.content
            except Exception:
                response_text = str(response)
        except Exception as e:
            verbose_proxy_logger.debug(
                f"RoutingGroupTestEngine: request failed for group '{routing_group_name}': {e}"
            )
        finally:
            # Always detach callback
            try:
                litellm.callbacks.remove(trace_callback)
            except ValueError:
                pass

        total_latency_ms = (time.monotonic() - start) * 1000

        # Determine final deployment from last success trace
        final_deployment = "unknown"
        final_provider = "unknown"
        success_traces = [t for t in trace_callback.traces if t.status == "success"]
        if success_traces:
            final = success_traces[-1]
            final_deployment = final.deployment_name
            final_provider = final.provider

        return RoutingGroupTestResult(
            success=success,
            response_text=response_text,
            traces=trace_callback.traces,
            total_latency_ms=total_latency_ms,
            final_deployment=final_deployment,
            final_provider=final_provider,
        )

    async def simulate_traffic(
        self,
        routing_group_name: str,
        router: "Router",
        routing_group_config: RoutingGroupConfig,
        num_requests: int = 100,
        concurrency: int = 10,
        mock: bool = True,
        failure_injection: Optional[FailureInjectionConfig] = None,
    ) -> RoutingGroupSimulationResult:
        """
        Simulate N concurrent requests through the routing group.

        Returns aggregated traffic distribution statistics and flow data
        for the Live Tester visualization.
        """
        import litellm
        from litellm.proxy.routing_group_utils.routing_trace_callback import (
            RoutingTraceCallback,
        )

        trace_callback = RoutingTraceCallback()
        litellm.callbacks.append(trace_callback)

        semaphore = asyncio.Semaphore(concurrency)

        async def _one_request() -> bool:
            async with semaphore:
                try:
                    if mock:
                        # Optionally inject failures via mock_response that raises
                        if failure_injection:
                            # Pick a deployment deterministically based on routing group config
                            # and check if we should inject a failure
                            # We can't know which deployment will be picked before the call,
                            # so we use a probabilistic approach:
                            # average failure rate across all deployments as a proxy
                            rates = list(
                                failure_injection.deployment_failure_rates.values()
                            )
                            avg_rate = sum(rates) / len(rates) if rates else 0.0
                            if random.random() < avg_rate:
                                raise Exception(
                                    "Simulated failure (failure injection)"
                                )
                        await router.acompletion(
                            model=routing_group_name,
                            messages=cast(List[AllMessageValues], _DEFAULT_MESSAGES),
                            mock_response="Simulated response",
                        )
                    else:
                        await router.acompletion(
                            model=routing_group_name,
                            messages=cast(List[AllMessageValues], _DEFAULT_MESSAGES),
                        )
                    return True
                except Exception:
                    return False

        total_latency_start = time.monotonic()
        results = await asyncio.gather(
            *[_one_request() for _ in range(num_requests)],
            return_exceptions=True,
        )
        _ = time.monotonic() - total_latency_start  # total wall time (unused but kept for future)

        litellm.callbacks.remove(trace_callback)

        successful = sum(1 for r in results if r is True)
        failed = num_requests - successful

        # Aggregate per-deployment stats from traces
        per_dep: Dict[str, Dict[str, Any]] = {}

        for trace in trace_callback.traces:
            dep_id = trace.deployment_id or trace.deployment_name
            if dep_id not in per_dep:
                per_dep[dep_id] = _empty_dep_stats()
            per_dep[dep_id]["deployment_name"] = trace.deployment_name
            per_dep[dep_id]["provider"] = trace.provider
            per_dep[dep_id]["request_count"] = cast(int, per_dep[dep_id]["request_count"]) + 1
            cast(List[float], per_dep[dep_id]["latencies"]).append(trace.latency_ms)
            if trace.status == "success":
                per_dep[dep_id]["success_count"] = cast(int, per_dep[dep_id]["success_count"]) + 1
            else:
                per_dep[dep_id]["failure_count"] = cast(int, per_dep[dep_id]["failure_count"]) + 1

        # Enrich with priority/weight from config
        dep_lookup = {d.model_id: d for d in routing_group_config.deployments}
        for dep_id, stats in per_dep.items():
            if dep_id in dep_lookup:
                dep = dep_lookup[dep_id]
                stats["priority"] = dep.priority
                stats["weight"] = dep.weight

        total_traced: int = sum(cast(int, s["request_count"]) for s in per_dep.values())

        traffic_distribution = [
            DeploymentTrafficStats(
                deployment_id=dep_id,
                deployment_name=cast(str, stats["deployment_name"]),
                provider=cast(str, stats["provider"]),
                request_count=cast(int, stats["request_count"]),
                success_count=cast(int, stats["success_count"]),
                failure_count=cast(int, stats["failure_count"]),
                avg_latency_ms=(
                    sum(cast(List[float], stats["latencies"])) / len(cast(List[float], stats["latencies"]))
                    if stats["latencies"]
                    else 0.0
                ),
                percent_of_total=(
                    cast(int, stats["request_count"]) / total_traced * 100
                    if total_traced > 0
                    else 0.0
                ),
                priority=cast(Optional[int], stats["priority"]),
                weight=cast(Optional[int], stats["weight"]),
            )
            for dep_id, stats in per_dep.items()
        ]

        # Sort: primary (priority 1 or highest traffic) first
        traffic_distribution.sort(
            key=lambda x: (x.priority or 999, -x.request_count)
        )

        # Build flow data for priority-failover strategy
        flow_data: Optional[List[FlowStep]] = None
        if routing_group_config.routing_strategy == "priority-failover":
            flow_data = _build_flow_data(
                trace_callback.traces, routing_group_config, traffic_distribution
            )

        fallback_count = sum(
            1 for t in trace_callback.traces if t.was_fallback
        )

        all_latencies = [t.latency_ms for t in trace_callback.traces]
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

        return RoutingGroupSimulationResult(
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            avg_latency_ms=avg_latency,
            fallback_count=fallback_count,
            traffic_distribution=traffic_distribution,
            flow_data=flow_data,
        )


def _build_flow_data(
    _traces: List[RoutingTrace],
    _config: RoutingGroupConfig,
    traffic_distribution: List[DeploymentTrafficStats],
) -> List[FlowStep]:
    """
    Build flow steps for the priority-failover visualization.

    For each deployment in priority order, create a FlowStep showing
    how many requests went to it and why (primary vs fallback).
    """
    flow_steps: List[FlowStep] = []

    sorted_deps = sorted(
        traffic_distribution,
        key=lambda x: (x.priority or 999, -x.request_count),
    )

    for i, dep in enumerate(sorted_deps):
        if i == 0:
            flow_steps.append(
                FlowStep(
                    from_deployment=None,
                    to_deployment=dep.deployment_name,
                    request_count=dep.request_count,
                    reason="primary",
                )
            )
        else:
            prev = sorted_deps[i - 1]
            flow_steps.append(
                FlowStep(
                    from_deployment=prev.deployment_name,
                    to_deployment=dep.deployment_name,
                    request_count=dep.request_count,
                    reason="fallback_error",
                )
            )

    return flow_steps
