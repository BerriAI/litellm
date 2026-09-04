# 0003 — OpenCode Go / Zen: one provider prefix, three wire formats

Status: accepted

Date: 2026-08-08

## Context

OpenCode offers two billing surfaces (Zen subscription, Go per-token) over a
single gateway.  Each surface speaks three different wire formats depending on
the model: OpenAI Chat Completions for GPT-series, Anthropic Messages for
Claude and select Qwen models, and OpenAI Responses for newer responses-mode
models.  Users address models with `opencode_go/…` or `opencode_zen/…`.

## Decision

We register **two** LiteLLM provider entries (`opencode_go`, `opencode_zen`)
that share one codebase and dispatch through a single `_complete_opencode()`
entry point in `litellm/main.py`.  That function resolves the surface
(`go` / `zen`), picks the base URL, then routes to the correct wire format
handler at request time based on model classification.

The three wire formats live in separate files:

| Wire format | Provider code | Entry in `_complete_opencode` |
|---|---|---|
| OpenAI Chat Completions | `litellm/llms/opencode/chat/transformation.py` | default path |
| Anthropic Messages | `litellm/llms/opencode/chat/messages_transformation.py` | `is_messages_model()` |
| OpenAI Responses | `litellm/llms/opencode/{zen,go}/responses/transformation.py` | cost map `mode: responses` |

Model classification is distributed: messages models use frozensets in
`messages_transformation.py` keyed by `@ai-sdk/anthropic` npm field, while
responses models are marked `mode: responses` in the cost map and routed
through the built-in `responses_api_bridge_check` mechanism.

### Data-placement asymmetry

Responses routing is driven by the **cost map** (`model_prices_and_context_window_backup.json`) because LiteLLM has a built-in bridge (`responses_api_bridge_check`) that reads `mode: responses` entries and dispatches to the provider
config returned by `get_opencode_config()`.  Messages routing lives inside
`_complete_opencode()` because no equivalent cost-map bridge exists for the
Anthropic Messages pass-through — the handler is called directly from the
opencode branch of the completion dispatch chain.

## Consequences

**Positive:**

- One provider prefix maps to three wire formats.  Users only need to add one
  set of credentials in the dashboard or config.
- Model classification lives next to the wire format that cares about it.
  Changing a model's arm only touches one file.
- The cost-map bridge for responses means providers can add new responses-mode
  models by editing the cost map without touching Python code.

**Negative:**

- The dispatch chain is hard to follow at a glance.  A reader seeing
  `opencode_zen/gpt-5.6-sol` must check three files (chat transformation,
  messages transformation, responses transformation) plus the cost map to
  determine which endpoint is hit.
- Messages classification requires regenerating frozensets from `models.dev`
  when model classifications change.
- The asymmetry between cost-map-driven responses routing and hard-coded
  messages routing makes the code harder to unify.  A hypothetical messages
  bridge would be a larger refactor.

## Rejected alternatives

### Per-endpoint prefixes

Use distinct provider slugs such as `opencode_chat`, `opencode_messages`,
`opencode_responses`.  This would make the wire format explicit at call time
but fragments the provider across three prefixes, doubles the dashboard entries
on each surface, and forces users to maintain three credential sets.

### Client-side model allowlist

Require the user to declare which wire format each model uses in config
(`model_alias`, `litellm_params.mode_override`, etc.).  This shifts
classification burden to the caller, defeats the value of LiteLLM's model
normalisation layer, and increases onboarding friction.

### Name heuristic ("model contains claude")

Detect the Anthropic Messages arm by inspecting the model name string.  This
is brittle: most Go models are Qwen, not Claude, and the Qwen3 family straddles
both chat and messages arms.  The `@ai-sdk/anthropic` npm classification from
`models.dev` is the authoritative source and is already embedded in the
frozensets.

### Extend the `openai_like` dynamic config mechanism

The `openai_like` provider (backed by `JSONProviderRegistry`) lets users add
new providers through JSON configuration without code changes.  It was used as
a prototype when first exploring OpenCode support on the `litellm_opencode_zen`
branch, but it only supports OpenAI Chat Completions wire format and would
require a full handler rewrite to support Messages and Responses pass-throughs.
The dedicated provider package approach is cleaner and avoids coupling to a
mechanism not designed for multi-format providers.

## Related

- PRD `.scratch/opencode-go-zen-providers/PRD.md` — feature requirements
