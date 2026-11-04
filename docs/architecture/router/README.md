# Router

`litellm.Router` decides which deployment serves a request, makes the call, and
handles what happens when the call fails. It retries in the same model group,
falls back to other model groups, and puts failing deployments into cooldown so
that later requests avoid them. The proxy builds one Router from its config and
sends every model request through it. SDK users can build their own.

![A caller calls litellm.Router, which picks a deployment, calls the provider,
and keeps cooldowns, usage and fail counts in the routing
cache](img/router-overview.svg)

## Concepts

Start with [Router concepts](explanation/concepts.md), which names the ideas and
shows how they fit together. Each idea then has its own page.

- [Deployments and model groups](explanation/deployments-and-model-groups.md),
  what is configured and how a deployment gets its identity
- [How a name finds its deployments](explanation/names.md), aliases, wildcards,
  routing groups, team names, deployment ids and defaults
- [Teams and access](explanation/teams-and-access.md), team deployments, access
  groups, access windows and PTU shares
- [How a deployment is picked](explanation/selection.md), the filter chain and
  priority with `order`
- [Strategies, routing groups and sub-routers](explanation/strategies.md), how
  one deployment is chosen and how a request is sent to the right group
- [Retries, fallbacks and cooldowns](explanation/reliability.md), including
  allowed fails, order fallbacks, weighted failover and streams
- [Shared state across proxy workers](explanation/shared-state.md)

Two pages cover how the router is built.

- [The core and the host](explanation/core-and-hosts.md), what the Rust core
  owns and what it asks its host for
- [The life of a request](explanation/request-lifecycle.md), from `acompletion`
  through the core to the provider

## Tutorial

New to the router? Work through [Route your first
requests](tutorial/route-your-first-requests.md). It builds a Router with mock
deployments, so it needs no API keys. It shows load balancing, retries,
cooldowns and fallbacks in about fifteen minutes.

## How-to guides

- [Read router state from outside the router](how-to/read-router-state.md)
- [Change deployments or settings at
  runtime](how-to/change-the-router-at-runtime.md)
- [Move a caller off a deprecated Router
  member](how-to/move-off-deprecated-members.md)
- [Add a custom routing strategy](how-to/add-a-routing-strategy.md)
- [Find out why a request went where it did](how-to/debug-a-routing-decision.md)

## Reference

- [Entry points](reference/entry-points.md), every method on the public surface
  and what it replaced
- [Core, config and host interfaces](reference/core-and-host.md), the step
  protocol, the Rust `Router` and the `Host` trait
- [Configuration and defaults](reference/configuration.md), constructor
  arguments, strategies and constants
- [Routing state and cache keys](reference/state-and-cache-keys.md), what the
  router stores and for how long
- [Code map](reference/code-map.md), which file holds what

## Status

These pages describe the branch `nate_router_step6_rust_core_v2` (#49513).
