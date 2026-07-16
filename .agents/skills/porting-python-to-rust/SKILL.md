---
name: porting-python-to-rust
description: How to port LiteLLM Python behavior into the litellm-rust crates (litellm-core, litellm-ai-gateway, litellm-python-bridge) so the Rust path matches Python exactly. Use this whenever adding or changing any Rust route, provider transform, router, or config resolution, especially anything that resolves a provider from a model.
---

# Porting Python behavior into litellm-rust

The Rust workspace ports LiteLLM's Python behavior; it does not reinvent it. When Python already has an abstraction for something, mirror it; do not invent a local shortcut that diverges from Python. Read the Python source first, then port its contract and field names.

## Mirror Python abstractions; never hand-roll what Python centralizes

The most common mistake is reimplementing logic Python already owns in one place. Provider resolution is the canonical example:

- WRONG: hand-rolling `model.split('/')` inline in a route (a bespoke `split_provider`). This ignores the deployment's explicit `custom_llm_provider` and drifts from Python.
- RIGHT: deployments carry `custom_llm_provider` in `litellm_params` (mirroring Python's `litellm_params`), and provider resolution goes through `litellm_core::get_llm_provider::get_llm_provider`, the pure port of `litellm.get_llm_provider` (`litellm/litellm_core_utils/get_llm_provider_logic.py`).

`get_llm_provider(model, custom_llm_provider)` precedence, matching Python:
- an explicit `custom_llm_provider` wins; a matching `provider/` prefix is stripped from the model, otherwise the model is returned unchanged
- with no explicit provider, the `provider/model` prefix is split off
- empty or whitespace-only values are treated as absent (host-layer credential/config rule)

Before writing any new "figure out the provider / api_base / api_key from a model" logic, use `get_llm_provider`. If it is missing a Python behavior you need (api_base based inference, azure ai studio / cohere_chat aliasing), extend that one helper to match Python rather than branching in the caller.

## Deployment fields mirror Python litellm_params

`litellm_core::router::LiteLLMParams` mirrors Python's `litellm_params` dict. When Python reads a field from `litellm_params` (e.g. `custom_llm_provider`, `api_key`, `api_base`), add it here with `#[serde(default)]` so it deserializes straight from the proxy `model_list` (the gateway loads deployments as JSON via `litellm.proxy.read_model_list`). Do not resolve provider identity from the model string when the deployment already carries it explicitly.

## Crate boundaries (see litellm-rust/AGENTS.md + CLAUDE.md)

- `litellm-core`: pure translation only (types, route contracts, provider transforms, router, and resolution helpers like `get_llm_provider`). No network, env reads, filesystem, or server behavior. New pure port logic belongs here so it is unit-testable and reusable.
- `litellm-ai-gateway`: the only crate that does I/O. Axum routes, auth, config resolution, network. Routes stay thin (handler validates + delegates to a no-axum `service`); Axum types never leak into the service layer.
- `litellm-python-bridge`: thin PyO3 adapter; no business logic.

Do not add a crate for a new route or provider; add a module.

## Prove parity with tests

Port behavior comes with tests that would fail if the port drifts from Python. For provider/model resolution that means covering: explicit `custom_llm_provider` with and without a matching prefix, the `provider/model` split fallback, blank/whitespace provider, and the no-provider error. Add a gateway-level test that a deployment configured with an explicit `custom_llm_provider` and a bare model (no prefix) still routes correctly.

## Required checks before pushing

From `litellm-rust`:
```
cargo fmt --check
cargo clippy -p litellm-ai-gateway --all-targets --features server -- -D warnings
cargo clippy -p litellm-core -p litellm-python-bridge --all-targets -- -D warnings
cargo test --workspace
```
