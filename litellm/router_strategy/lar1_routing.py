"""
LAR-1 Semantic Routing Strategy

Routes requests based on agent confidence level (LAR-1 protocol).
Thresholds are configurable via routing_strategy_args in router config.

LAR-1 metadata is read from routing_strategy_args (server-side), NOT from
client-supplied request metadata. This prevents privilege escalation where
an end-client could force a specific deployment tier by fabricating
confidence / evidence signals.

To preserve backward compatibility for proxies that derive LAR-1 signals
server-side (e.g., in a middleware hook), set `accept_client_metadata: true`
in routing_strategy_args. This is NOT recommended for production deployments
where deployment tiers carry cost or trust differences.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional, Union

from pydantic import ValidationError

from litellm._logging import verbose_router_logger
from litellm.router import CustomRoutingStrategyBase
from litellm.types.lar1 import LAR1Metadata, LAR1Time

if TYPE_CHECKING:
    from litellm.router import Router

DEFAULT_THRESHOLDS: dict[str, float] = {"low": 0.3, "medium": 0.5, "high": 0.7}
DEFAULT_DEPLOYMENT_TYPE: str = "cloud-smart"


def _coerce_threshold(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _coerce_str(value: object, default: str) -> str:
    if isinstance(value, str):
        return value
    return default


def lar1_thresholds_from_args(
    routing_strategy_args: Optional[Mapping[str, object]] = None,
) -> dict[str, float]:
    args = routing_strategy_args or {}
    return {
        "low": _coerce_threshold(args.get("confidence_threshold_low"), DEFAULT_THRESHOLDS["low"]),
        "medium": _coerce_threshold(args.get("confidence_threshold_medium"), DEFAULT_THRESHOLDS["medium"]),
        "high": _coerce_threshold(args.get("confidence_threshold_high"), DEFAULT_THRESHOLDS["high"]),
    }


def apply_lar1_routing_strategy(
    router: Router,
    routing_strategy_args: Optional[Mapping[str, object]] = None,
) -> None:
    # Clean up any previous custom strategy before applying the new one.
    # This ensures that switching away from LAR-1 (or re-applying it)
    # does not leave stale monkey-patched methods on the router instance.
    router._reset_custom_routing_strategy()

    strategy = LAR1RoutingStrategy(
        router_instance=router,
        thresholds=lar1_thresholds_from_args(routing_strategy_args),
        args=routing_strategy_args,
    )
    router.routing_strategy = "lar1"
    router.set_custom_routing_strategy(strategy)


def _normalize_thresholds(thresholds: Optional[dict[str, float]]) -> dict[str, float]:
    merged = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    low = merged["low"]
    medium = merged["medium"]
    high = merged["high"]
    if not (0 < low < medium < high < 1):
        raise ValueError(
            f"LAR-1 thresholds must satisfy 0 < low < medium < high < 1, got low={low}, medium={medium}, high={high}"
        )
    return merged


def _parse_lar1_metadata(request_kwargs: dict) -> LAR1Metadata:
    lar1_raw = request_kwargs.get("metadata", {}).get("lar1", {})
    if not isinstance(lar1_raw, dict):
        verbose_router_logger.warning(f"[LAR-1] Invalid lar1 metadata type: {type(lar1_raw).__name__}. Using defaults")
        return LAR1Metadata()
    try:
        return LAR1Metadata.model_validate(lar1_raw)
    except ValidationError as exc:
        verbose_router_logger.warning(f"[LAR-1] Invalid lar1 metadata: {exc}. Using defaults")
        return LAR1Metadata()


class LAR1RoutingStrategy(CustomRoutingStrategyBase):
    def __init__(
        self,
        router_instance: Optional[Router] = None,
        thresholds: Optional[dict[str, float]] = None,
        args: Optional[Mapping[str, object]] = None,
    ):
        self._router = router_instance
        self.thresholds = _normalize_thresholds(thresholds)

        server_args = dict(args) if args else {}
        self._default_deployment_type = _coerce_str(
            server_args.get("default_deployment_type"), DEFAULT_DEPLOYMENT_TYPE
        )
        # SECURITY: accept_client_metadata defaults to False.
        # When False, the strategy ignores client-supplied metadata.lar1
        # and uses the default deployment type instead.
        self._accept_client_metadata = bool(server_args.get("accept_client_metadata", False))

    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[list[dict[str, str]]] = None,
        input: Optional[Union[str, list]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[dict] = None,
    ):
        if request_kwargs is None:
            request_kwargs = {}
        if self._router is None:
            return None

        if self._accept_client_metadata:
            lar1 = _parse_lar1_metadata(request_kwargs)
            confidence = lar1.confidence
            evidence = tuple(e.value for e in lar1.evidence)
            time_dim = lar1.time.value
        else:
            # Server-side mode: ignore client metadata, use default.
            confidence = 0.0
            evidence = ()
            time_dim = LAR1Time.NOW.value
            verbose_router_logger.debug(
                "[LAR-1] client metadata ignored; using default deployment type '%s'",
                self._default_deployment_type,
            )

        healthy = await self._router.async_get_healthy_deployments(
            model=model,
            request_kwargs=request_kwargs,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
        )
        if isinstance(healthy, dict):
            return healthy

        if not healthy:
            return None

        target = self._classify_request(confidence, evidence, time_dim)
        selected, exact_match = self._select_deployment(target, healthy)

        if selected is None:
            return None
        if exact_match:
            verbose_router_logger.info(f"[LAR-1] confidence={confidence} -> {target}")
        else:
            actual_type = selected.get("model_info", {}).get("type", "unknown")
            verbose_router_logger.warning(
                f"[LAR-1] No deployment for type '{target}', fallback to deployment type '{actual_type}'"
            )
        return selected

    def _classify_request(
        self,
        confidence: float,
        evidence: tuple[str, ...],
        time_dim: str,
    ) -> str:
        # When accept_client_metadata is False, this always returns the default.
        if "UNVERIFIED" in evidence:
            return "cloud-smart"

        if time_dim == LAR1Time.MEM.value:
            return "cloud-fast"

        t = self.thresholds
        if confidence < t["low"]:
            return "cloud-smart"
        if confidence < t["medium"]:
            return "cloud-fast"
        if confidence < t["high"]:
            return "local"
        return "deep"

    def _select_deployment(
        self,
        target_type: str,
        deployments: list[dict],
    ) -> tuple[Optional[dict], bool]:
        if not deployments:
            return None, False

        for deployment in deployments:
            if not isinstance(deployment, dict):
                continue
            model_type = deployment.get("model_info", {}).get("type", "")
            if model_type == target_type:
                return deployment, True

        for deployment in deployments:
            if isinstance(deployment, dict):
                return deployment, False

        return None, False

    def get_available_deployment(self, *args, **kwargs):
        raise NotImplementedError(
            "LAR-1 routing only supports async routing. Enable async_only_mode on the router or use acompletion."
        )