# The core and the host

The router has three layers. `litellm.Router` is a thin Python facade that
callers use. Behind it, the Rust crate `litellm-router-core` (the core) makes
every routing decision and owns the state those decisions need. The core reaches
outside itself only through a host, which runs the provider calls and anything
else that needs live objects from the process the core runs in. This page
explains that split and why the line is drawn where it is.

![litellm.Router and the RustCore driver over the Rust crate, which hands steps
to PythonHost, which calls the provider](../img/core-and-hosts-1.svg)

## What the core owns

The core owns the routing decisions and their state.

- the registry of deployments, its indexes and the deployment build
- name resolution, the filter chain and `simple-shuffle`
- teams, access groups, access windows and PTU shares
- routing groups, pre-call check decisions, health-check routing and the order
  filter
- the call loop: retries, retry policies, fallbacks, order fallbacks, weighted
  failover and stream recovery
- cooldowns, allowed fails counters and usage, written through the routing cache
- model info and model group info, merged from what the host reports

Everything the core decides is a function of its config, the request, the
routing cache, and the answers the host gave. It never reads process globals,
never calls a provider and never awaits Python. That is what makes the core
testable with a scripted host and portable to a host that is not Python.

## What the host answers

The core asks for what it needs as a step, and the host answers it. Every
attempt is a step: the core picks the deployment, the host runs the call, and
the core records the outcome. There is no second attempt path inside the core.
The `Host` trait (`litellm-rust/crates/router-core/src/host.rs`) groups the
hostcalls.

| Hostcall | What the core needs |
| --- | --- |
| `invoke` | run one attempt on one deployment |
| `select` | a strategy implemented in Python, or the routing plugin, tag or text-only stage |
| `pre_route` | the sub-routers and session binding before a pick resolves its name, or a whole pick by a custom strategy |
| `pre_call_facts`, `callback_filters`, `routing_plugin_models` | the facts and filters that need live objects, such as token counts and `CustomLogger` filters |
| `allow_fallback` | the access and budget checks before a cross-group fallback |
| `secrets`, `validate`, `provider`, `added` | the deployment build, one deployment at a time |
| `model_info` | cost map and capability lookups during a model info query |
| `sleep`, `observe`, `cache` | backoff, routing events, and the shared store all routing state lives in |

The rule for where a piece of logic goes is simple. If a decision needs only
config, the request, routing state and plain facts, it belongs in the core. If
it needs a live object, such as a stateful strategy, a logging callback or the
provider SDK, the host runs that part and hands the core a plain answer. The
core still makes the decision. Pre-call checks show the pattern: the host counts
the prompt's tokens and looks up context windows, and the core decides which
deployments to drop.

## The host today

Today the core runs inside the Python process, under `PythonHost`
(`litellm/router_python_host.py`) and the modules under `litellm/router_rust/`.

- `invoke/` runs one attempt. `litellm/router_attempt.py` builds the keyword
  arguments for the deployment the core picked, and `PythonHost.invoke(op)`
  calls `litellm.<op>` with them. The endpoint wrappers, stream wrappers,
  `max_parallel_requests` slots and retry and fallback metadata live here.
  Nothing here picks, retries, falls back or records cooldowns
- `steps/` answers the hostcalls that need live Python objects: the stateful
  strategies, the sub-routers, pre-call facts, `Router(plugins=[...])` objects,
  `CustomLogger` filters and caller access facts
- `surface/` keeps the Python `Router` API over the core's state. `registry()`,
  `model_list`, the indexes and `names()` read the core, and every write to them
  becomes a change set

A host that answers these steps natively, inside a Rust gateway, would need none
of `surface/`. Through `Host::supports` and `Host::can_select`, it would refuse
a config that needs Python objects when the router is built, rather than failing
partway through a request.

## Driving the core

`RustCore` holds one `RouterCoreHandle`. A call starts with
`RouterCoreHandle.start(op, request)` and then loops: each
`RouterCall.next(answer)` runs the core until it needs something only the host
can give, and returns that need as a step. `select()` and the model info queries
are driven the same way.

![The step loop for one call: the driver reads cooldowns, tries deployment a and
gets a 429, records a cooldown, tries deployment b, and returns its response
with routing headers](../img/core-and-hosts-2.svg)

Every step also carries the routing events since the last one (`Selected`,
`AttemptFailed`, `CooldownSet`, `Retry`, `Fallback` and others). The driver
hands them to the host's observers, which feed alerting and metrics, and uses
them to stamp the request's metadata and to raise the exception a caller
expects.

## Outcomes are recorded by the core

The core records each attempt's outcome itself, through cache steps, right after
the attempt, so nothing depends on a logging callback firing. A streamed attempt
is deferred: the core counts the request when the stream opens, and when the
stream ends the host's stream wrapper reports either the success, with the final
usage, or the failure.

## The public surface

Code outside the router uses the endpoint methods and eight entry points, and
nothing else: `registry()`, `settings()`, `health()`, `select()`, `apply()`,
`observe()`, `register_plugin()` and `retire()`. Routing plugins get a
`RoutingContext` with the same views instead of the Router.
`tests/code_coverage_tests/router_surface_check.py` keeps an allowlist of the
members that outside code touches, and the list may only shrink. See [Entry
points](../reference/entry-points.md).

## Settings are static

`Router(**arguments)` freezes its arguments into a `RouterConfig`, and the core
is built from it. After that, `apply(ChangeSet(...))` is the only way to change
deployments or settings. The process settings the router reads while serving,
such as `litellm.num_retries`, the fallback lists, the model cost map and the
callback lists, arrive as a versioned `RouterDynamicConfig` snapshot, so the
core never reads a global.

![How the router's settings reach the core: constructor arguments frozen into
RouterConfig, process settings snapshotted into RouterDynamicConfig, and
apply(ChangeSet) for later changes](../img/core-and-hosts-3.svg)

## The application host

Separately, the router needs a few answers that only the proxy has, such as
vertex pass-through credentials and whether a caller may use a team's auto
router. It asks a `RouterHost` (`litellm/router_host.py`). That is the proxy's
`ProxyRouterHost` when the proxy can be imported, and otherwise a default that
refuses team auto routers.
