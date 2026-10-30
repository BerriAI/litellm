# Model catalog

## Python surface today

- `model_prices_and_context_window.json` at the repo root, loaded into `litellm.model_cost` and refetched from a remote URL
- Capability and price questions (`supports_*`, effort tiers, token prices) answered by Python helpers such as `AnthropicModelInfo` and `litellm.get_model_info`
- `litellm.register_model` and direct writes to `litellm.model_cost` change entries at runtime
- `litellm/rust_bridge/host/model_capabilities.py` answers capability questions for the extension by running those Python helpers

## Shared crates

- `crates/model-catalog` reads the document, indexes models and aliases, lowers entries into typed prices and capabilities, and holds the in-memory store. Its rules are in its `AGENTS.md`
- `crates/model-catalog-refresh` owns refetching (source, policy, steps)
- Reading, refetching, lookups and capability rules all live in Rust

## Per host

- python-sdk and python-proxy
  - The extension loads and refetches the catalog natively, the same way rust-proxy does
  - Rust exposes the catalog to Python through `python-bridge`. Python's model info and capability helpers become shims over it
  - No host call. `host/model_capabilities.py` is deleted once the native catalog answers capability questions
  - `litellm.model_cost` stays the same writable dict and global for now. How it and `litellm.register_model` stay in sync with the Rust store: TBD
- rust-proxy
  - Same crates, loaded and refetched natively

## Wire contracts

- The JSON document is the shared input. `Catalog::problems` defines a mergeable cost map and `check-registry` guards it on pull requests
- What Python serves today for a model (`Catalog::served`) is what the Rust API returns

## Order and gates

- TBD
