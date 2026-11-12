# Router concepts

The router is responsible for answering one question for every request: which
deployment should serve it. It then has to deal with what happens when that
deployment fails, retrying the request, and falling back to other deployments.

There are a few main ideas that are referenced through the rest of these docs.
This page gives a brief overview of them.

![The router concepts in order: a name resolves to deployments, access rules,
filters and a strategy pick one for an attempt, and a failure goes to retries,
fallbacks and cooldowns, which write the shared state the filters
read](../img/concepts-1.svg)

## What is configured

A **deployment** is one place that can serve a model, such as an Azure OpenAI
resource in one region or a vLLM cluster behind a URL. It is one entry of
`model_list`, with the provider call's arguments in `litellm_params` and
metadata in `model_info`. Its `model_info.id` is the key for all of its routing
state. See [Deployments and model groups](deployments-and-model-groups.md).

A **model group** is every deployment that shares a `model_name`. Callers ask
for a model group, never for a deployment. Load balancing, retries and cooldowns
happen inside one group, and fallbacks move a request between groups.

A **name** is what a request asks for. It is usually a model group, but it can
also be an alias, a wildcard such as `openai/*`, a routing group, a team's
public name for a model, or a deployment id. **Resolution** turns a name into
the list of deployments that may serve the request. See [How a name finds its
deployments](names.md).

## Who may use what

A deployment can belong to one **team**, be reserved for some teams during an
**access window**, be split across teams by **PTU shares**, or sit behind an
**access group** that only some keys reach. These rules run right after
resolution and remove the deployments the caller may not use. See [Teams and
access](teams-and-access.md).

## How one deployment is picked

A **filter** removes deployments that must not serve this request right now:
ones that are failing health checks, cooling down, blocked by an admin, too
small for the prompt, missing a tag, or at a higher **order** (a lower priority)
than another healthy one. See [How a deployment is picked](selection.md).

A **strategy** picks one deployment from the remaining eligible ones.
`simple-shuffle`, the default, picks a weighted random one. Others pick by load,
latency, usage or cost. A **routing group** gives a set of model groups its own
strategy, and **sub-routers** such as the auto router choose which model group a
request goes to before any of this runs. See [Strategies, routing groups and
sub-routers](strategies.md).

## What happens on failure

A **retry** runs the pick again in the same model group, so it can land on a
different deployment. A **fallback** moves the request to another model group
once retries are spent. A **cooldown** takes one failing deployment out of every
pick for a few seconds, and **allowed fails** decide how many failures that
takes. See [Retries, fallbacks and cooldowns](reliability.md).

## State

Cooldowns, failure counters and usage live in a **routing cache** with an
in-memory tier and an optional Redis tier, so a fleet of proxy workers shares
one view of which deployments are healthy. See [Shared state across proxy
workers](shared-state.md).

## How it is built

You do not need this section to use the router. For how the router is built
inside and how it runs a request, see [The core and the host](core-and-hosts.md)
and [The life of a request](request-lifecycle.md).
