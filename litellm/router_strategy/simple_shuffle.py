"""
Returns a random deployment from the list of healthy deployments.

If weights are provided, it will return a deployment based on the weights.

"""

import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_router_logger

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


# Order of precedence when more than one weighting field is declared across
# the deployment list. ``weight`` is a generic relative-weight field and
# should win over the rate-limit-derived weights so an operator who sets
# ``weight: 100`` alongside an ``rpm:`` (used by other code paths for cap
# enforcement) gets the explicit weighting they asked for.
_WEIGHT_FIELDS_PRECEDENCE = ("weight", "rpm", "tpm")


def _pick_weight_field(
    healthy_deployments: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Return the first field in ``_WEIGHT_FIELDS_PRECEDENCE`` that *any*
    deployment in the list declares, or ``None`` if none do.

    The previous implementation only consulted ``healthy_deployments[0]``,
    which silently fell through to uniform random whenever the first
    deployment happened to be unweighted but later ones declared
    ``rpm`` / ``tpm`` / ``weight``. That made deployment-level weighting
    appear to "silently ignore" the field, depending on routing order.
    """
    for field in _WEIGHT_FIELDS_PRECEDENCE:
        for deployment in healthy_deployments:
            litellm_params = deployment.get("litellm_params") or {}
            if litellm_params.get(field) is not None:
                return field
    return None


def simple_shuffle(
    llm_router_instance: LitellmRouter,
    healthy_deployments: Union[List[Any], Dict[Any, Any]],
    model: str,
) -> Dict:
    """
    Returns a random deployment from the list of healthy deployments.

    If any deployment declares one of ``weight`` / ``rpm`` / ``tpm`` under
    ``litellm_params``, a weighted random pick is performed using that
    field as the relative weight (deployments that don't declare it get
    weight ``0``). Field precedence: ``weight`` > ``rpm`` > ``tpm``.

    Note: ``rpm`` / ``tpm`` here are **static relative weights**, not
    cap-enforced limits. Strategies like ``usage-based-routing-v2`` and
    ``latency-based-routing`` enforce caps from cached usage; for cap
    enforcement under simple-shuffle, enable
    ``router_settings.enable_pre_call_checks`` (which filters
    RPM-exhausted deployments out of ``healthy_deployments`` before the
    shuffle) or use the proxy's key-level v3 TPM/RPM limiter.

    Args:
        llm_router_instance: LitellmRouter instance
        healthy_deployments: List of healthy deployments
        model: Model name

    Returns:
        Dict: A single healthy deployment
    """
    weight_field = _pick_weight_field(healthy_deployments)
    if weight_field is not None:
        weights = [
            (m.get("litellm_params") or {}).get(weight_field, 0)
            for m in healthy_deployments
        ]
        verbose_router_logger.debug(f"\nweight {weights}")
        total_weight = sum(weights)
        # Defensive: if every deployment declared the field as 0 (or
        # negative — pathological config), total_weight == 0 and the
        # divide-by-zero below would crash. Fall through to uniform random
        # instead. Same goes for any non-finite total.
        if total_weight > 0:
            normalized = [w / total_weight for w in weights]
            verbose_router_logger.debug(f"\n weights {normalized} by {weight_field}")
            selected_index = random.choices(range(len(normalized)), weights=normalized)[
                0
            ]
            verbose_router_logger.debug(f"\n selected index, {selected_index}")
            deployment = healthy_deployments[selected_index]
            verbose_router_logger.info(
                f"get_available_deployment for model: {model}, Selected deployment: "
                f"{llm_router_instance.print_deployment(deployment) or deployment[0]} "
                f"for model: {model}"
            )
            return deployment or deployment[0]

    ############## No usable weights, we do a random pick #################
    item = random.choice(healthy_deployments)
    return item or item[0]
