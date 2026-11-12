# Deployments and model groups

Everything the router routes to is a deployment, and every request asks for a
name, which is usually a model group. This page explains those, lists the
deployment fields the router itself reads, and shows how a deployment gets its
identity.

## Deployment

A deployment is one place that can serve a model, such as an Azure OpenAI
resource in `eastus`, a Bedrock model in one AWS account, or a vLLM cluster
behind a URL. In config, it is one entry of `model_list`.

```yaml
model_list:
  - model_name: gpt-4o            # the model group callers ask for
    litellm_params:               # the provider call's arguments
      model: azure/gpt-4o-eastus
      api_base: https://eastus.example.openai.azure.com
      api_key: os.environ/AZURE_EASTUS_KEY
      rpm: 600
      order: 1
    model_info:                   # metadata about the deployment
      id: gpt-4o-eastus
```

![A deployment and its two parts: litellm_params, the provider call's arguments,
and model_info, its metadata](../img/deployments-and-model-groups-1.svg)

`litellm_params` holds mostly the arguments for the provider call, which the
router passes on unchanged when the deployment is picked. The router reads a few
of them itself.

| Field | What the router does with it |
| --- | --- |
| `weight`, `rpm`, `tpm` | `simple-shuffle` weights, in that order of preference. `rpm` and `tpm` are also limits for usage-aware strategies and pre-call checks |
| `order` | in-group priority, lowest first. See [How a deployment is picked](selection.md) |
| `tags` | which tagged requests it serves, when tag filtering is on |
| `use_in_pass_through` | whether [pass-through routes](../../../../litellm/proxy/pass_through_endpoints/architecture.md) may use it |
| `timeout`, `stream_timeout` | the attempt's timeout, which wins over the router's |
| `max_parallel_requests` | how many attempts may run on it at once |

`model_info` is metadata. The router reads who may use the deployment
(`team_id`, `access_groups`, `access_windows`, `ptu_shares`, see [Teams and
access](teams-and-access.md)), whether an admin paused it (`blocked`), and how
it cools down (`cooldown_time`, `allowed_fails`, `allowed_fails_policy`, see
[Retries, fallbacks and cooldowns](reliability.md)). Everything else is kept as
written and handed back to callers that read the registry.

## Identity

`model_info.id` is the deployment's identity. Cooldowns, failure counters, usage
and health results are all keyed by it, and a change set replaces or deletes a
deployment by it. When the config leaves it out, the router derives one with
`deployment_id()`: a SHA-256 over the model group name and every
`litellm_params` key and value, in order. The same config gets the same id on
every worker, which is what lets workers share routing state. Two entries with
the same id both stay in the group and share that state, and a lookup by id
returns the later one.

## How a deployment is built

The router builds deployments from config in order. For each entry it resolves
`os.environ/` secret references, validates the entry, resolves the provider that
serves it, and records it. An entry with an `organization` list becomes one
deployment per organization. A deployment joins the registry as soon as it is
accepted, so a later entry sees the earlier ones, and an error points at the
entry that caused it. An entry that fails to build is skipped, and the rest
still build.

![For each model_list entry the router resolves secrets, validates the entry,
finds the provider, records it, then gives it an id and adds it to the
registry](../img/deployments-and-model-groups-2.svg)

## Model group

A model group is every deployment that shares a `model_name`. Callers ask for a
group, never for a deployment, and the router picks a deployment inside it. Load
balancing, retries and cooldowns all happen inside one group, and fallbacks move
a request from one group to another.

![A request for gpt-4o reaches the gpt-4o model group of three deployments at
two orders, which falls back to the claude-sonnet model
group](../img/deployments-and-model-groups-3.svg)

Group size changes behavior in a few places. In a group of one: a 429 does not
cool the only deployment down, an auth error is not retried, and the default
error-rate rule leaves the lone deployment alone unless it saw 1,000 requests
that all failed. Those rules exist because cooling down the only deployment
would fail every request for the group. [Retries, fallbacks and
cooldowns](reliability.md) has the details.

## The registry

All deployments live in the registry, an immutable, indexed snapshot held by the
router. It is indexed by id, model group, alias, wildcard pattern and team, and
it carries a version that goes up on every change. A change replaces the
snapshot, so a request that already read the registry keeps a consistent view
while the change lands. Code outside the router reads it through
`router.registry()` and changes it through `router.apply()`, see [Read router
state](../how-to/read-router-state.md) and [Change deployments or settings at
runtime](../how-to/change-the-router-at-runtime.md).
