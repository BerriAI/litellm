"""The default counterfactual a complexity router's savings are measured against.

`litellm_settings.autorouter_savings_baseline_model` names the model the traffic would
have run on without a router. When the operator sets it, that answer wins and nothing
here runs. When they do not, the router's own tier ladder already names it: without a
router a deployment has to pick one model that can carry the hardest request it will
see, so the default baseline is the priciest model in the hardest configured tier. A
cheap tier is a choice the router made, not a ceiling it was bounded by.

Candidates are ranked once against a fixed reference request, not against each request
that runs. Ranking per request means reading the request, and every input shape it can
take; a default must not carry that surface. An operator whose pool ordering genuinely
depends on request shape names the baseline in config, which skips this file entirely.

Baselines are always provider-qualified, because they travel to the spend writer as a
bare string with no provider beside them; an operator who writes ``deepseek-r1`` meaning
Azure would otherwise be priced against whoever else owns that name.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Final, NamedTuple

from litellm._logging import verbose_router_logger
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

if TYPE_CHECKING:
    from litellm.router import Router


_REFERENCE_REQUEST: Final = Usage(
    prompt_tokens=20_000,
    completion_tokens=1_000,
    total_tokens=21_000,
    prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=19_000, cache_creation_tokens=1_000, text_tokens=0),
)


class Baseline(NamedTuple):
    """The counterfactual deployment: what it is called, and which deployment it was.

    ``model`` is what the operator would recognise, and the string the spend writer
    prices the counterfactual under. ``deployment_id`` only ranks: a deployment can be
    charged something other than its model's public rate, and
    `Router.get_deployment_model_info` is what merges the two.
    """

    model: str
    deployment_id: str | None = None


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
    indices: Final = router.model_name_to_deployment_indices.get(group_name)
    if not indices:
        return (Baseline(qualified),) if (qualified := canonical_model(group_name)) else ()

    def candidate(index: int) -> Baseline | None:
        deployment: Final = router.model_list[index]
        params: Final = deployment.get("litellm_params")
        if not isinstance(params, dict):
            return None
        info: Final = deployment.get("model_info")
        base: Final = info.get("base_model") if isinstance(info, dict) else None
        model: Final = base or params.get("base_model") or params.get("model")
        qualified: Final = canonical_model(model, params.get("custom_llm_provider")) if model else None
        if qualified is None:
            return None
        deployment_id: Final = info.get("id") if isinstance(info, dict) else None
        return Baseline(qualified, str(deployment_id) if deployment_id else None)

    return tuple(c for index in indices if (c := candidate(index)) is not None)


def _priced(router: "Router", candidate: Baseline) -> tuple[float, Baseline] | None:
    """``(cost_of_the_reference_request, candidate)``, or ``None`` when unpriceable.

    "Most expensive" is a property of a request, not of a rate: a deployment dearer per
    output token can be cheaper per cached token, so comparing a chosen pair of rates
    orders cache-heavy traffic backwards. Costing one reference request through the same
    engine the savings use leaves cache rates, tiered tables and every other billing
    dimension to that engine. A candidate that prices to nothing there cannot stand in
    for what the traffic would have cost.
    """
    from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token

    provider, _, model_name = candidate.model.partition("/")
    try:
        info: Final = router.get_deployment_model_info(candidate.deployment_id or "", candidate.model)
        if info is None:
            return None
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model_name or candidate.model,
            usage=_REFERENCE_REQUEST,
            custom_llm_provider=provider,
            model_info=info,
        )
    except Exception as e:  # noqa: BLE001  # an unpriceable candidate simply cannot be the baseline
        verbose_router_logger.debug("savings baseline: no pricing for candidate %s (%s)", candidate.model, e)
        return None
    cost: Final = prompt_cost + completion_cost
    if cost <= 0.0:
        verbose_router_logger.debug("savings baseline: candidate %s prices to nothing", candidate.model)
        return None
    return (cost, candidate)


def _most_expensive(router: "Router", candidates: Iterable[Baseline]) -> Baseline | None:
    """The candidate that would have cost the most on the reference request."""
    priced: Final = tuple(r for candidate in candidates if (r := _priced(router, candidate)) is not None)
    if not priced:
        verbose_router_logger.debug("savings baseline: no priceable candidates; savings driver disabled")
        return None
    return max(priced)[1]


def resolve_baseline(router: "Router", group_names: Iterable[str]) -> Baseline | None:
    """The derived baseline for a router whose hardest tier offers ``group_names``.

    Holds no cache of its own; each pricing pass walks the pool, so the caller is
    expected to bound how often it runs. The complexity router caches the result with a
    TTL, which keeps a deployment added or removed at runtime able to change the
    baseline while keeping this walk off the per-request hot path.

    Never raises. This is read on the routing path to decorate a request that is about
    to be served, and a dashboard's counterfactual is not worth failing a live request
    over; an unresolvable baseline zeroes the savings driver instead.
    """
    try:
        return _most_expensive(router, (c for name in group_names for c in _models_in(router, name)))
    except Exception as e:  # noqa: BLE001  # see docstring: routing must not fail for a metric
        verbose_router_logger.warning("savings baseline: could not resolve, savings will read zero (%s)", e)
        return None
