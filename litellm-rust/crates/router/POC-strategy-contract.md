# Routing strategy contract (draft, 2026-10-07)

A contract the model-driven routing strategies (auto, complexity, adaptive, quality) implement, so whoever drives the retry and fallback loop, Python or Rust, can call them the same way. Draft only: adopting it means refactoring the Python strategies first, and the Rust side waits on that

## Why

Today a strategy is anything with `async_pre_routing_hook(model, request_kwargs, messages, input)` returning `PreRoutingHookResponse`, but that signature hides most of what they depend on:
- they read loosely typed keys out of `request_kwargs` (session id, key hash, user agent, tools, instructions, min quality tier, proxy headers)
- they write their results back into `request_kwargs` metadata for other code to find later (the adaptive chosen model for the response header and the learning hook, `quality_router_decision`, the raw model name flag)
- they hold the whole router and call into it: `cache` for session pins, `get_model_list` and `get_router_model_info` for context windows and vision support, `async_get_healthy_deployments` to ask whether a group can serve the request, `acompletion` / `aresponses` / `aembedding` for classifier calls
- context compaction and routing compression are armed through contextvars that the inference call reads later
- the adaptive router learns through a `CustomLogger` in `litellm.callbacks` that finds its decision again by reading a metadata key back off the logged kwargs

None of that crosses a process or language boundary cleanly, and most of it can't be unit tested without a full router

## Shape

A strategy is built once from its definition and a set of ports, then asked for a decision on every attempt and told the outcome after it. It never sees the router, never mutates its input, and never reaches a contextvar

```python
class RoutingStrategy(Protocol):
    async def route(self, request: RoutingRequest) -> RoutingDecision: ...
    async def observe(self, feedback: Feedback) -> None: ...


class StrategyFactory(Protocol):
    def build(self, definition: StrategyDefinition, ports: StrategyPorts) -> RoutingStrategy: ...
```

`route` is called at the start of every attempt, retries included, the same place the pre-routing hook runs today (spec 2.2). `observe` is called once per attempt after it settles. Neither may raise for an expected condition: a strategy that can't decide returns `Reject`, and a failed classifier call is the strategy's to turn into a default pick, the way the auto router already does

## Input

```python
@dataclass(frozen=True, slots=True)
class RoutingRequest:
    requested_model: str
    surface: Literal["chat", "responses", "anthropic_messages"]
    conversation: Conversation
    caller: Caller
    session: Session | None
    hints: RoutingHints
    attempt: AttemptInfo


@dataclass(frozen=True, slots=True)
class Conversation:
    messages: tuple[ChatMessage, ...]
    system_prompt: str | None
    tools: tuple[ToolDefinition, ...]
    native: NativePayload


@dataclass(frozen=True, slots=True)
class Caller:
    key_hash: str | None
    team_id: str | None
    user_id: str | None
    tags: tuple[str, ...]
    log_message_content: bool


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    agent_id: str | None
    user_agent: str | None


@dataclass(frozen=True, slots=True)
class RoutingHints:
    min_quality_tier: int | None


@dataclass(frozen=True, slots=True)
class AttemptInfo:
    number: int
    fallback_depth: int
```

`messages` is the chat-completions shape every surface already normalizes to (`resolve_structured_messages`), so strategies stop resolving it themselves. `native` keeps the original surface payload (Responses `instructions` and `input`, encrypted reasoning) for the few cases that need it, such as the complexity router's encrypted classifier task. `Session` is only set for a client-supplied id; a generated one is not a session to pin. The host builds the whole request once, from the request kwargs, with Pydantic at the boundary

## Output

```python
RoutingDecision = Route | Decline | Reject


@dataclass(frozen=True, slots=True)
class Route:
    target_group: str
    tier: str | None
    params_overlay: Mapping[str, object]
    deployment_affinity_ttl: int | None
    compaction: CompactionPlan | None
    response_headers: Mapping[str, str]
    provenance: StandardLoggingRoutingDecision
    feedback_key: str | None


@dataclass(frozen=True, slots=True)
class Decline:
    pass


@dataclass(frozen=True, slots=True)
class Reject:
    reason: str
```

`Route` covers everything the strategies hand back today through the response and the metadata side channel. `params_overlay` is the tier's `litellm_params`. `compaction` replaces arming contextvars: the host arms compaction right before the call it makes. `response_headers` replaces stashing the adaptive chosen model in metadata for a post-call hook to turn into a header. `feedback_key` is opaque to the host and handed back in `observe`. `Decline` means the strategy does not apply to this request and normal routing proceeds, which is what returning `None` means today. Strategies never rewrite the conversation: none does today apart from routing-only compression, which the host owns

## Feedback

```python
@dataclass(frozen=True, slots=True)
class Feedback:
    key: str
    target_group: str
    request: RoutingRequest
    outcome: Served | Failed


@dataclass(frozen=True, slots=True)
class Served:
    assistant_text: str | None
    tool_calls: tuple[ToolCall, ...]
    latency_ms: int


@dataclass(frozen=True, slots=True)
class Failed:
    status: int
    error_type: str
```

This replaces `AdaptiveRouterPostCallHook` finding its decision again in the logged kwargs. The host calls `observe` for every attempt that `route` decided, so a failed attempt that falls back still teaches the strategy. `observe` runs after the response is returned and must never delay or fail the request

## Ports

```python
@dataclass(frozen=True, slots=True)
class StrategyPorts:
    sessions: SessionStore
    catalog: Catalog
    eligibility: Eligibility
    models: ModelClient
    state: StrategyStateStore
    clock: Callable[[], float]
    rng: Random


class SessionStore(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def put(self, key: str, value: str, ttl_seconds: int) -> None: ...
    async def claim(self, key: str, value: str, ttl_seconds: int, eligible: frozenset[str]) -> str: ...
    async def delete(self, key: str) -> None: ...


class Catalog(Protocol):
    def groups(self) -> frozenset[str]: ...
    def facts(self, group: str) -> tuple[DeploymentFacts, ...]: ...


@dataclass(frozen=True, slots=True)
class DeploymentFacts:
    deployment_id: str
    provider: str
    context_window: int | None
    max_output_tokens: int | None
    supports_vision: bool | None
    supports_prompt_caching: bool
    input_cost_per_token: float | None
    supported_params: frozenset[str] | None


class Eligibility(Protocol):
    async def can_serve(self, group: str, request: RoutingRequest) -> bool: ...


class ModelClient(Protocol):
    async def classify(self, model: str, call: ClassifierCall, on_behalf_of: RoutingRequest) -> ClassifierResult: ...
    async def embed(self, model: str, texts: tuple[str, ...], on_behalf_of: RoutingRequest) -> tuple[Vector, ...]: ...


class StrategyStateStore(Protocol):
    async def load(self, strategy: str) -> StrategyState: ...
    async def save(self, strategy: str, delta: StrategyState) -> None: ...
```

Each port replaces one way the strategies reach into the router today:
- `SessionStore` replaces `router.cache` for the complexity router's tier and deployment pins. Keys are built by the strategy and already namespaced by caller and router name; `claim` is the existing `claim_affinity_pin` compare-and-set
- `Catalog` replaces `get_model_list` and `get_router_model_info`. It is a read-only view of one snapshot, so a model list change mid-request can't hand a strategy two different views. It also answers the question `_tier_params_the_target_accepts` asks today
- `Eligibility` replaces calling `async_get_healthy_deployments` on a copy of the request kwargs, which the health gate, the modality gate and cache-aware routing all do. It also covers the caller's access to the group (`can_key_call_resolved_model`). The host answers it with the same filters and cooldowns its own pick will use, which is the point: a Rust host answers from Rust's cooldowns, the Python host from Python's
- `ModelClient` replaces calling `router.acompletion` / `aresponses` / `aembedding` with hand-built metadata. The client attaches spend attribution, message logging and session linking from `on_behalf_of` itself, so the strategy never forwards caller metadata
- `StrategyStateStore` replaces the adaptive router's direct Prisma reads and its update queue the proxy flusher drains. The host decides where state lives and when it flushes
- `clock` and `rng` make the bandit sampling, random tier picks and TTL math deterministic in tests

## What stays with the host

These run once per attempt around `route`, today inside `PythonRouter.async_pre_routing_hook`, and are host code, not strategy code, so they're written once per host:
- resolving aliases and picking which registered strategy owns the requested name, by tags (`_select_pre_routing_strategy`)
- the Claude Code session router binding (`_resolve_claude_code_session_router`)
- routing plugins (`Router(plugins=...)`) and member authorization (`authorize_member_auto_router_inference`)
- routing-only compression of the conversation before `route` sees it
- applying a `Route`: dropping overlay params no deployment accepts, forwarding the marker's params, stamping `routing_decision`, consumed tags and the affinity TTL for the filters that read them, arming compaction, adding the response headers

## Where each strategy needs work to fit

- adaptive: closest already. `pick_model` is pure apart from sampling; it needs `rng`, `feedback_key` and `observe` in place of the CustomLogger, and `state` in place of Prisma and the update queue
- quality: small. Drop the metadata write for `quality_router_decision` in favor of `provenance`, and build the scorer from the complexity router's pure classifier instead of a whole `ComplexityRouter` holding the router
- auto: small. The encoder takes `ModelClient.embed` instead of the router
- complexity: the bulk of the work. Session and deployment pins move to `SessionStore`, the context, modality and health gates move to `Catalog` and `Eligibility`, the LLM and JEV classifiers move to `ModelClient`, plan mode, escalation and the adaptive mode stop writing metadata, and compaction becomes a `CompactionPlan` on the decision. Splitting `complexity_router.py` along those lines would fall out of the same change

## Open questions

- whether `Eligibility.can_serve` should answer per group or return the eligible deployment ids, since cache-aware routing wants to know whether one specific deployment is live
- whether custom classifier plugins (`ClassifierPlugin`) and routing plugins keep their current `RoutingContext` contract or move onto `RoutingRequest`
- how far `NativePayload` should be typed: enough for the encrypted classifier task, or the full surface payload
