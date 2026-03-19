# Malachi Provider Design

**Goal:** Add `malachi` as a first-class LiteLLM provider so it appears in LiteLLM's provider catalog, provider field metadata, and provider resolution path, while reusing the existing OpenAI-compatible transport stack.

## Summary

`Malachi` should behave like a real provider slug in LiteLLM, not a local alias layered on top of `custom_openai`.

That means:
- `malachi` appears in the canonical provider enum and provider list
- `malachi` appears in provider metadata exposed to UIs and downstream apps
- `malachi/<model>` resolves through `get_llm_provider()` as provider `malachi`
- LiteLLM uses provider-specific environment variables for `Malachi`
- request execution still reuses OpenAI-compatible logic under the hood

## Architecture

The implementation should separate provider identity from transport implementation.

Provider identity lives in:
- `litellm.types.utils.LlmProviders`
- provider manifests such as `provider_endpoints_support.json`
- dashboard/provider field manifests such as `provider_create_fields.json`
- README/provider documentation coverage

Transport behavior should stay minimal:
- `Malachi` is OpenAI-compatible for the endpoints we care about
- LiteLLM should route it through the existing OpenAI-compatible provider stack
- no bespoke HTTP client or protocol fork should be introduced in v1

## Provider Behavior

### Provider slug

Use:
- `malachi`

### Model naming

Expected runtime shape:
- `malachi/gpt-5.4`
- `malachi/gpt-5.4-mini`
- `malachi/<other-malachi-model>`

This keeps model-manager and other control-plane tools aligned with normal LiteLLM provider-prefixed model references.

### Environment variables

Use provider-specific env names:
- `MALACHI_API_BASE`
- `MALACHI_API_KEY`

These should be preferred over generic OpenAI-compatible env vars for the `malachi` provider.

### Default base URL

The provider config should allow any user-supplied base URL, but the provider metadata and placeholder examples should assume a Malachi OpenAI-style endpoint.

For Foundry usage, the practical base URL is the Malachi gateway running on Foundry, but LiteLLM itself should not hardcode Foundry-specific infrastructure into library logic beyond provider defaults/examples.

## Endpoint support

The provider support manifest should only advertise endpoints Malachi actually supports today.

For v1, the conservative set is:
- `chat_completions = true`
- `responses = true`

Other endpoints should only be marked true if Malachi actually exposes them in practice.

This is important because model-manager will use LiteLLM's provider manifests dynamically, and inflated capability metadata would create broken UI affordances.

## UI / Provider Catalog Impact

The following LiteLLM surfaces should include `malachi`:
- provider list from `LlmProviders`
- provider endpoints support manifest
- provider create fields manifest
- README supported providers table if enforced by tests

The provider create fields entry should expose:
- `api_base`
- `api_key`

with a simple default model placeholder such as:
- `gpt-5.4`

This is enough for model-manager to render a clean provider dropdown and dynamic form without inventing its own local `Malachi` overlay.

## Routing Strategy

`malachi` should resolve as a first-class provider but reuse OpenAI-compatible request handling.

Recommended approach:
- add `MALACHI` to `LlmProviders`
- wire provider config resolution to an OpenAI-like config implementation
- add provider resolution rules in `get_llm_provider_logic.py`
- avoid adding a custom transport handler unless Malachi later diverges semantically from OpenAI-compatible behavior

This keeps the code small while still making the provider official.

## Testing Strategy

Add focused tests for:
- provider enum / provider list inclusion
- provider manifest inclusion
- provider field manifest inclusion
- `get_llm_provider("malachi/gpt-5.4")` resolving to provider `malachi`
- provider config reading `MALACHI_API_BASE` and `MALACHI_API_KEY`

Avoid broad transport/integration work in this patch unless a targeted test shows it is necessary.

## Constraints and Notes

- MiniMax is already present in this LiteLLM checkout and should not be re-added.
- This worktree currently cannot run pytest cleanly on Windows because `pytest-postgresql` imports `psycopg` and this environment lacks `libpq`.
- That environment issue is separate from the Malachi provider work and should not change the implementation scope.
