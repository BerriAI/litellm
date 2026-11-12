# Strategies, routing groups and sub-routers

Once the filters have left a list of deployments that may serve a request, a
strategy picks one. Routing groups let different model groups use different
strategies, and sub-routers decide which model group a request should go to
before any of that happens. This page covers all three, plus custom strategies
and routing plugins.

![A request is routed by a sub-router to a model group, resolved and filtered,
then the chosen strategy picks one deployment](../img/strategies-1.svg)

## Strategies

| Strategy | Picks | Reads |
| --- | --- | --- |
| `simple-shuffle` (default) | a weighted random deployment | per-request weights, then `weight`, then `rpm`, then `tpm`, else uniform |
| `least-busy` | the deployment with the fewest requests in flight | in-flight counters |
| `usage-based-routing-v2` | the deployment with the lowest tpm this minute that stays under its tpm and rpm | per-minute usage counters |
| `usage-based-routing` | the v1 version of the above, kept for old configs | per-minute usage |
| `latency-based-routing` | the deployment with the lowest recent latency (time to first token for streams), within a buffer | latency samples |
| `cost-based-routing` | the cheapest deployment still under its limits | model cost map, usage |
| `lar1` | a deployment by the caller's confidence level, with thresholds from `routing_strategy_args` | `metadata.lar1` on the request |

`simple-shuffle` needs no state. The other strategies keep their own counters,
fed by litellm's logging callbacks. Every strategy runs after the filters, so it
only ever sees deployments the filters already approved.

### Weights in `simple-shuffle`

![How simple-shuffle picks its weights: request weights, else weight, else rpm,
else tpm, else uniform random](../img/strategies-2.svg)

The first source with a positive value on any candidate is used for all
candidates, and a candidate that does not set it gets weight 0. Per-request
weights come from key or team level `router_settings`, and the proxy puts them
on the request as `{model_group: {deployment_id: weight}}`. One team can then
send most of a group's traffic to one deployment while everyone else gets the
global split, without a separate router per team.

## Which strategy applies

![Which strategy applies: the request's routing_strategy, else the routing
group's, else the router's, else simple-shuffle](../img/strategies-3.svg)

A per-request `routing_strategy`, which the proxy forwards from key or team
`router_settings`, wins. Next comes the strategy of the routing group the
resolved name belongs to, then the router's own `routing_strategy`. Stateful
strategies keep separate state per scope, so the same strategy used by two
routing groups does not mix their counters.

## Routing groups

A routing group is a named set of model groups with its own strategy.

```yaml
routing_groups:
  - group_name: fast
    models: [gpt-4o-mini, claude-haiku]
    routing_strategy: latency-based-routing
  - group_name: tiers
    models: [gpt-4o-ptu, gpt-4o-paygo]
    routing_strategy: priority
    model_priorities: {gpt-4o-ptu: 1, gpt-4o-paygo: 2}
```

The routing group's name is callable. A request for `fast` resolves to the
deployments of both members, and latency-based routing picks among them. A
direct request for `gpt-4o-mini` also uses latency-based routing, since that
model group belongs to `fast`. Every model group that is not a member of a
configured routing group belongs to the implicit default routing group, which
uses the router's `routing_strategy`.

A priority group is the exception. It does not pick by a strategy of its own.
Instead it serves each member's priority as its deployments' `order`, so the
order filter sends all traffic to priority 1 while any of it is healthy, and
order fallbacks walk down the priorities when it fails. See [How a deployment is
picked](selection.md) for the order filter and [Retries, fallbacks and
cooldowns](reliability.md) for order fallbacks.

## Sub-routers

The auto router, complexity router, quality router and adaptive router are
sub-routers, not strategies. They choose which model group should serve a
request, before the name is resolved. Each is configured as a deployment whose
`litellm_params.model` starts with `auto_router/`.
`auto_router/complexity_router...`, `auto_router/quality_router...` and
`auto_router/adaptive_router...` select those three routers, and any other name
under `auto_router/` is the auto router. A request for that deployment's
`model_name` runs the sub-router.

| Sub-router | Chooses by |
| --- | --- |
| auto router | how closely the prompt matches the example prompts configured for each route, by embedding similarity |
| complexity router | how hard the prompt looks, sending simple prompts to a cheaper group |
| quality router | the prompt's complexity mapped to a quality tier, then the model whose routing preferences match that tier |
| adaptive router | the kind of request (code, writing, reasoning and so on) and a bandit that learns from past responses which model serves each kind best, weighed against cost |

![A request for smart goes to the auto router, which classifies the prompt and
returns gpt-4o-mini, and the router picks within that
group](../img/strategies-4.svg)

The router asks the sub-router once per pick, before resolution, and resolves
whatever model group it returns. The attempt still reports the requested name,
`smart`, as its model group, so headers and spend logs show what the caller
asked for. Marker deployments are dropped from every candidate list, so they are
never called directly. Whether a caller may use a sub-router that belongs to a
team depends on the proxy's team checks, which the router asks the proxy for.

Session binding happens at the same point: a request that belongs to a coding
session can be bound to the model group that served the session so far.

## Routing plugins

`Router(plugins=[...])` takes routing plugins: objects that look at a request
and narrow which model groups or deployments may serve it, or that attach
signals for later stages to read. The plugins run during pre-routing, and the
router then keeps only the deployments of the models they left.

## Custom strategies

`router.register_plugin(strategy)` with a `CustomRoutingStrategyBase` does not
slot into the strategy step. It replaces the whole pick. The router serves the
deployment the custom strategy returns as is, without resolution, filters or a
strategy of its own. The custom strategy may even return a deployment the
registry does not hold. `lar1` works this way. See [Add a custom routing
strategy](../how-to/add-a-routing-strategy.md).
