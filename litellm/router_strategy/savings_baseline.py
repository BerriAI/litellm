"""The counterfactual model a complexity router's savings are measured against.

Without the router a deployment has to pick one model, and it has to be one that can
carry the hardest request it will see. That model is the baseline: what the traffic
would have cost had nobody routed it. The router's own tier ladder already names it,
so the candidates are the models in the hardest configured tier; a cheap tier is a
choice the router made, not a ceiling it was bounded by.

Baselines are always provider-qualified, whether derived or configured, because they
travel to the spend writer as a bare string with no provider beside them; an operator
who writes ``deepseek-r1`` meaning Azure would otherwise be priced against whoever
else owns that name.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from litellm._logging import verbose_router_logger

if TYPE_CHECKING:
    from litellm.router import Router


def canonical_model(model: str, custom_llm_provider: str | None = None) -> str | None:
    """``provider/model``, or ``None`` when the pair names no known provider.

    A deployment may name its vendor in the model prefix or in a separate
    ``custom_llm_provider``, and the bare name alone is not enough to price: it can
    resolve to a different vendor's rates, or to nothing at all.
    """
    import litellm

    try:
        resolved, provider, _, _ = litellm.get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
    except Exception as e:  # noqa: BLE001  # an unroutable candidate cannot be the baseline
        verbose_router_logger.debug("savings baseline: cannot resolve candidate %s (%s)", model, e)
        return None
    return f"{provider}/{resolved}"


def _models_in(router: "Router", group_name: str) -> tuple[str, ...]:
    """The models a tier entry actually calls, priced as the deployments declare.

    `litellm_params.model` is not always a model: on Azure it is the deployment name,
    absent from the cost map, and `model_info.base_model` names the real one. Wildcard
    and aliased deployments behave the same, and router.py resolves pricing through
    that same base_model chain. A name matching no deployment is a tier pointing
    straight at a provider model rather than at a configured group.
    """
    indices = router.model_name_to_deployment_indices.get(group_name)
    if not indices:
        return (qualified,) if (qualified := canonical_model(group_name)) else ()

    def priced_as(index: int) -> str | None:
        deployment = router.model_list[index]
        params = deployment.get("litellm_params")
        if not isinstance(params, dict):
            return None
        info = deployment.get("model_info")
        base = info.get("base_model") if isinstance(info, dict) else None
        model = base or params.get("base_model") or params.get("model")
        return canonical_model(model, params.get("custom_llm_provider")) if model else None

    return tuple(model for index in indices if (model := priced_as(index)))


def _most_expensive(models: Iterable[str]) -> str | None:
    """The priciest candidate by output rate, input rate breaking the tie.

    A model that costs nothing per token cannot stand in for what the traffic would
    otherwise have cost; as a baseline it would report the whole real spend as a loss,
    so it is dropped rather than allowed to win a tie at zero.
    """
    import litellm

    def rates(model: str) -> tuple[float, float, str] | None:
        try:
            info = litellm.get_model_info(model=model)
        except Exception as e:  # noqa: BLE001  # unmapped candidates simply cannot be the baseline
            verbose_router_logger.debug("savings baseline: no pricing for candidate %s (%s)", model, e)
            return None
        output_rate, input_rate = info.get("output_cost_per_token") or 0.0, info.get("input_cost_per_token") or 0.0
        if output_rate <= 0.0 and input_rate <= 0.0:
            verbose_router_logger.debug("savings baseline: candidate %s has no per-token price", model)
            return None
        return (output_rate, input_rate, model)

    priced = tuple(r for model in models if (r := rates(model)) is not None)
    if not priced:
        verbose_router_logger.debug("savings baseline: no priceable candidates; savings driver disabled")
        return None
    return max(priced)[2]


def resolve_baseline(configured: str | None, router: "Router", group_names: Iterable[str]) -> str | None:
    """The baseline for a router whose hardest tier offers ``group_names``.

    A configured override wins and is only qualified, never re-derived.

    Derived per call rather than cached: the parent router adds and removes deployments
    while it runs, so a baseline pinned on first use would keep naming a model the
    router no longer has, and a pricier one added later could never become the baseline.
    Resolving costs tens of microseconds against a network call.

    Never raises. This is read on the routing path to decorate a request that is about
    to be served, and a dashboard's counterfactual is not worth failing a live request
    over; an unresolvable baseline zeroes the savings driver instead.
    """
    try:
        if configured:
            return canonical_model(configured)
        return _most_expensive(model for name in group_names for model in _models_in(router, name))
    except Exception as e:  # noqa: BLE001  # see docstring: routing must not fail for a metric
        verbose_router_logger.warning("savings baseline: could not resolve, savings will read zero (%s)", e)
        return None
