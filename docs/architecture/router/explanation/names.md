# How a name finds its deployments

A request names a model, and the name does not have to be a model group. It can
be an alias, a wildcard match, a routing group, a team's own name for a model,
or a deployment id. Resolution turns the name into the list of deployments that
may serve this caller, which is used for later steps. Resolution runs at the
start of every pick, so a retry or a fallback resolves the name again.

## The kinds of name

| Name | Example | Request names | Resolves to |
| --- | --- | --- | --- |
| Model group | `gpt-4o` | `gpt-4o` | the deployments with that `model_name` |
| Alias | `model_group_alias={"gpt-4": "gpt-4o"}` | `gpt-4` | the target group. `{"model": "gpt-4o", "hidden": true}` also hides the alias from model listings |
| Routing group | `routing_groups=[{"group_name": "fast", "models": [...]}]` | `fast` | the deployments of every member group, see [Strategies, routing groups and sub-routers](strategies.md) |
| Wildcard | a deployment with `model_name: "openai/*"` | `openai/gpt-4o-mini` | that deployment, with the requested model filled into its params |
| Team public name | `model_info.team_public_model_name: "my-gpt"` | `my-gpt` | the team's deployment, for members of that team |
| Deployment id | `gpt-4o-eastus` | `gpt-4o-eastus` | that one deployment, even while it cools down |
| Specific deployment | a deployment's `litellm_params.model` | `azure/gpt-4o`, with `specific_deployment=True` | the deployments whose litellm model is exactly that name |
| Anything else | | `unknown-model` | the `default_deployment` if one is set, else the first `default_fallbacks` group, else an error |

## The order

The checks run in a fixed order, and the first match wins.

![How a name resolves, in order: specific_deployment, deployment id, alias,
routing group, configured group, then the litellm model name, the first
default_fallbacks group, or an error](../img/names-1.svg)

Two rules apply on every path. Deployments that are sub-router markers, such as
`auto_router/...`, are never returned, since they are not meant to be called.
Deployments that an access window reserves for other teams right now are
withheld, see [Teams and access](teams-and-access.md).

## Aliases

An alias maps one name to a model group. Aliases come from the router's
`model_group_alias` and from the process's model alias map, which the router
reads through its dynamic config. Resolution swaps the alias for its target
before it looks anything else up, so an alias can also point at a routing group.
The attempt still reports the requested name to the caller.

## Wildcards

A deployment whose `model_name` contains `*`, such as `openai/*` or
`anthropic/claude-*`, serves any name it matches that no model group claims. The
registry keeps the patterns ordered from most to least specific, so
`anthropic/claude-3-*` wins over `anthropic/*`. The matched part is substituted
into the deployment's `litellm_params.model`: a deployment with
`model_name: openai/*` and `model: openai/*` serves a request for
`openai/gpt-4o-mini` by calling `openai/gpt-4o-mini`. Wildcards are checked only
for a name that is not a configured model group, so a configured group always
wins over a pattern that also matches it.

## Deployment ids

A name that is not a model group but is a deployment id resolves to that one
deployment. The router serves it even while it cools down, and skips the
strategy, since there is nothing to choose. This is how a caller pins a request
to one deployment.

## Default deployment and default fallbacks

`default_deployment` is a deployment that serves any name nothing else matched,
with the requested model filled in. Without one, a name that resolves to nothing
tries `default_fallbacks`, the catch-all fallback list, and resolves to its
first model group. If that list is empty too, the pick fails with an error that
says no deployments are configured for the model.

## The result

Resolution returns three things: the deployments, which kind of name matched,
and the model group the attempt reports. For an alias or a routing group, that
model group is the requested name, so headers and logs show what the caller
asked for while the deployments come from the target. [How a deployment is
picked](selection.md) takes it from there.
