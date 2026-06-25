"""
LAR-1 Semantic Routing Strategy

Routes requests based on agent confidence level (LAR-1 protocol).
Thresholds are configurable via lar1_settings in router config.

LAR-1 metadata passed via request_kwargs["metadata"]["lar1"]
"""

from typing import Dict, List, Optional, Union

from litellm._logging import verbose_router_logger
from litellm.router import CustomRoutingStrategyBase


class LAR1RoutingStrategy(CustomRoutingStrategyBase):
    def __init__(self, router_instance=None, thresholds: Optional[Dict] = None):
        self._router = router_instance
        self.thresholds = thresholds or {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
        }

    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict] = None,
    ):
        if request_kwargs is None:
            request_kwargs = {}

        metadata = request_kwargs.get("metadata", {})
        lar1 = metadata.get("lar1", {})
        confidence = lar1.get("confidence", 0.5)

        if (
            not isinstance(confidence, (int, float))
            or confidence < 0.0
            or confidence > 1.0
        ):
            verbose_router_logger.warning(
                f"[LAR-1] Invalid confidence: {confidence}. Fallback to 0.5"
            )
            confidence = 0.5

        valid_evidence = {"SYNTH", "RETRIEVED", "UNVERIFIED", "CONFIRMED"}
        evidence = [e for e in lar1.get("evidence", []) if e in valid_evidence]

        valid_times = {"NOW", "MEM", "CTX", "PRE"}
        time_dim = lar1.get("time", "NOW")
        if time_dim not in valid_times:
            verbose_router_logger.warning(
                f"[LAR-1] Invalid time: {time_dim}. Fallback to NOW"
            )
            time_dim = "NOW"

        model_list = self._router.model_list if self._router else []

        target = self._classify_request(confidence, evidence, time_dim, model_list)
        selected = self._select_deployment(target, model_list)

        if selected:
            verbose_router_logger.info(f"[LAR-1] confidence={confidence} → {target}")
            return selected

        return model_list[0] if model_list else None

    def _classify_request(self, confidence, evidence, time_dim, model_list):
        """
        confidence: 0.0-1.0
        evidence: ["SYNTH", "RETRIEVED", "UNVERIFIED"]
        time_dim: "NOW", "MEM", "CTX", "PRE"
        """
        if "UNVERIFIED" in evidence:
            return "cloud-smart"

        if time_dim == "MEM":
            return "cloud-fast"

        t = self.thresholds
        if confidence < t["low"]:
            return "cloud-smart"
        elif confidence < t["medium"]:
            return "cloud-fast"
        elif confidence < t["high"]:
            return "local"
        else:
            return "deep"

    def _select_deployment(self, target_type, model_list):
        if not model_list:
            return None

        for m in model_list:
            if isinstance(m, dict):
                model_type = m.get("model_info", {}).get("type", "")
                if model_type == target_type:
                    return m

        return model_list[0]

    def get_available_deployment(self, *args, **kwargs):
        pass
