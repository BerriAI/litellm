# How a deployment is picked

A pick turns a requested name into one deployment. It runs inside every attempt,
so a retry picks again and can land somewhere else. `router.select()` runs the
same pick without calling the provider.

## The filter chain

Resolution produces the candidates, and a chain of filters removes the ones that
must not serve this request right now. A filter can only remove candidates. The
strategy at the end of the chain sees what is left.

![The filter chain from pre-routing and name resolution through every filter to
the strategy, or to the no deployments available error](../img/selection-1.svg)

## The filters

**Route capability filter.** A call that only some providers serve skips the
deployments whose provider cannot serve it, as long as another candidate
remains. For example, a `/v1/decisions` call to a group with an OpenAI
deployment and an Anthropic deployment skips the Anthropic one, since litellm
has no Decisions config for that provider.

**Health check filter.** With `enable_health_check_routing`, the results of the
background health checks count too. A deployment that the last check marked
unhealthy is dropped, as long as that result is younger than
`health_check_staleness_threshold`. `background_health_check_model_groups`
limits this to some model groups. If the filter would leave nothing, it keeps
every candidate instead, so a broken health checker cannot take a whole group
down.

**Cooldown filter.** Drops every deployment with a live
`deployment:{id}:cooldown` key in the routing cache. Cooldowns are how failures
on one request protect the next ones, see [Retries, fallbacks and
cooldowns](reliability.md). When health checks drive routing with an
`allowed_fails_policy` and the cooldown filter would leave nothing, the router
ignores cooldowns for that request.

**Blocked filter.** Drops deployments with `model_info.blocked`, which an admin
sets to pause a deployment without deleting it. If every deployment of a group
is blocked, the error says so.

**Callback filters.** Callbacks can narrow the candidates too, and they run in
the chain like any other filter. The budget limiter drops deployments whose
provider is over its `provider_budget_config` budget. Deployment and session
affinity keep a conversation on the deployment that served it before by removing
all others from consideration as long as the previous deployment is healthy.

**Pre-call checks.** With `enable_pre_call_checks`, the router drops deployments
whose context window the prompt exceeds, deployments outside the request's
`allowed_model_region`, and deployments that cannot take a requested parameter.
The prompt's token count and each deployment's limits come from the model cost
map. When nothing is left, the error says which check removed the last
deployment. A context window error, for example, is one the fallback walk can
route to `context_window_fallbacks`.

**Tag filter.** With `enable_tag_filtering`, a request's tags keep only the
deployments whose `litellm_params.tags` include them. Deployments tagged
`default` serve untagged requests. `!tag` excludes deployments that carry that
tag, and `&tag` requires it.

**Routing plugin candidates.** When routing plugins ran during pre-routing, only
the deployments of the models they left remain.

**Text-only filter.** A request that carries an image skips deployments that
declare no vision support, as long as another candidate remains.

**Order filter.** Keeps only the deployments with the lowest
`litellm_params.order` among those left, or with the order an order fallback
targets. See below.

**Excluded and retry-skipped ids.** A retry skips the deployments that already
failed for this request, but only while something else remains. Weighted
failover excludes ids outright, and an excluded id is never picked.

## Priority with `order`

`order` is how in-group priority works. Deployments with `order: 1` take all
traffic while any of them passes the filters. Deployments with `order: 2` see
traffic only when the fallback walk targets the group at that order, after every
`order: 1` deployment failed for the request (see [Retries, fallbacks and
cooldowns](reliability.md)). Mixing deployments with and without an order is
easy to get wrong. While any candidate sets an order, this filter drops a
deployment without one, and the order fallbacks never target it either. So give
every deployment of a group an order, or give none of them one. A priority
routing group serves each member's priority as its deployments' `order`, so the
same filter walks the priorities (see [Strategies, routing groups and
sub-routers](strategies.md)).

![Priority with order: normal picks go to the two order 1 deployments, and the
order 2 deployment only gets a request after order 1
failed](../img/selection-2.svg)

## Shortcuts

A name that resolves to one deployment by its id is served even while that
deployment cools down. The callback filters and the strategy are skipped, since
there is nothing to choose. A custom strategy skips the whole chain, see
[Strategies, routing groups and sub-routers](strategies.md).

## When nothing is left

The pick fails with "No deployments available for selected model". The error
lists the deployments in cooldown and the seconds until the soonest cooldown
ends, and it says whether pre-call checks were on, since they are a common
cause. The fallback walk treats this error like a rate limit, so a group with
every deployment cooling down falls back to its fallback groups. [Find out why a
request went where it did](../how-to/debug-a-routing-decision.md) shows how to
find which filter removed a deployment.
