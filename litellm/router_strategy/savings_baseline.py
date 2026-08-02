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
from typing import TYPE_CHECKING, NamedTuple

from litellm._logging import verbose_router_logger

if TYPE_CHECKING:
    from litellm.router import Router


class Baseline(NamedTuple):
    """The counterfactual model, and the key its rates are looked up under.

    Two identifiers because they answer different questions. ``model`` is what the
    operator would recognise and what decides whether the router actually switched
    away from it. ``pricing_key`` is what it costs: a deployment may override the
    per-token prices, and the router registers that override in the cost map under
    the deployment's own id, deliberately keeping it off the shared model-name key so
    deployments do not pollute each other. Pricing the name would silently read the
    public rate for a model the operator does not pay the public rate for.
    """

    model: str
    pricing_key: str | None = None


def cost_key(model: str, custom_llm_provider: str | None, deployment_id: str | None) -> str | None:
    """The key litellm prices this deployment under, or ``None`` when it is just the model.

    Delegates to `_select_model_name_for_cost_calc`, the resolver the real request is
    billed through, rather than re-deciding when an override counts. Mirroring that rule
    here would be a second model of the runtime, and it is subtler than it looks: a
    per-second or tiered override counts, a partially specified one still counts, and a
    deployment configured at zero is priced at zero rather than treated as unpriced.
    """
    from litellm.cost_calculator import _select_model_name_for_cost_calc

    if deployment_id is None:
        return None
    resolved = _select_model_name_for_cost_calc(
        model=model,
        completion_response=None,
        custom_pricing=True,
        router_model_id=str(deployment_id),
        custom_llm_provider=custom_llm_provider,
    )
    return resolved if resolved and resolved != model else None


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


def _models_in(router: "Router", group_name: str) -> tuple[Baseline, ...]:
    """The candidates a tier entry actually calls, each with its own pricing key.

    `litellm_params.model` is not always a model: on Azure it is the deployment name,
    absent from the cost map, and `model_info.base_model` names the real one. Wildcard
    and aliased deployments behave the same, and router.py resolves pricing through
    that same base_model chain. A name matching no deployment is a tier pointing
    straight at a provider model rather than at a configured group, and prices under
    its own name because there is no deployment to override it.
    """
    indices = router.model_name_to_deployment_indices.get(group_name)
    if not indices:
        return (Baseline(qualified),) if (qualified := canonical_model(group_name)) else ()

    def candidate(index: int) -> Baseline | None:
        deployment = router.model_list[index]
        params = deployment.get("litellm_params")
        if not isinstance(params, dict):
            return None
        info = deployment.get("model_info")
        base = info.get("base_model") if isinstance(info, dict) else None
        model = base or params.get("base_model") or params.get("model")
        qualified = canonical_model(model, params.get("custom_llm_provider")) if model else None
        if qualified is None:
            return None
        deployment_id = info.get("id") if isinstance(info, dict) else None
        return Baseline(qualified, cost_key(qualified, params.get("custom_llm_provider"), deployment_id))

    return tuple(c for index in indices if (c := candidate(index)) is not None)


def _priced(candidate: Baseline) -> tuple[float, float, Baseline] | None:
    """``(output_rate, input_rate, candidate)``, or ``None`` when it cannot be priced.

    A candidate that costs nothing per token cannot stand in for what the traffic would
    otherwise have cost; as a baseline it would report the whole real spend as a loss,
    so it is dropped rather than allowed to win a tie at zero.
    """
    import litellm

    try:
        info = litellm.get_model_info(model=candidate.pricing_key or candidate.model)
    except Exception as e:  # noqa: BLE001  # unmapped candidates simply cannot be the baseline
        verbose_router_logger.debug("savings baseline: no pricing for candidate %s (%s)", candidate.model, e)
        return None
    output_rate, input_rate = info.get("output_cost_per_token") or 0.0, info.get("input_cost_per_token") or 0.0
    if output_rate <= 0.0 and input_rate <= 0.0:
        verbose_router_logger.debug("savings baseline: candidate %s has no per-token price", candidate.model)
        return None
    return (output_rate, input_rate, candidate)


def _most_expensive(candidates: Iterable[Baseline]) -> Baseline | None:
    """The priciest candidate by output rate, input rate breaking the tie.

    Ranked on what each candidate really costs, so a deployment whose configured price
    is the expensive one is chosen as the counterfactual; ranking on the public rate
    picks the wrong baseline and then prices it at a rate nobody pays.
    """
    priced = tuple(r for candidate in candidates if (r := _priced(candidate)) is not None)
    if not priced:
        verbose_router_logger.debug("savings baseline: no priceable candidates; savings driver disabled")
        return None
    return max(priced)[2]


def resolve_baseline(configured: str | None, router: "Router", group_names: Iterable[str]) -> Baseline | None:
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
            # No pricing key: a configured override names a model, not a deployment,
            # so it prices under its own name like any other unmatched name.
            return Baseline(qualified) if (qualified := canonical_model(configured)) else None
        return _most_expensive(c for name in group_names for c in _models_in(router, name))
    except Exception as e:  # noqa: BLE001  # see docstring: routing must not fail for a metric
        verbose_router_logger.warning("savings baseline: could not resolve, savings will read zero (%s)", e)
        return None
