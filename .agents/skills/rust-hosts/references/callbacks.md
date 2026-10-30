# Callbacks

Callbacks will be redesigned later. Until then, don't extend this page or design new callback seams from it

## Python surface today

- Legacy Python callbacks and `CustomLogger` hooks, which can read and change the whole request
- TBD

## Shared crates

- `crates/callbacks-legacy-python` runs the legacy callbacks from Rust
- TBD

## Per host

- python-sdk and python-proxy
  - Legacy callbacks run through `litellm/rust_bridge/host/callbacks_legacy_python.py`
- rust-proxy
  - A config with callbacks Rust doesn't have makes the request forward to python-proxy
- TBD

## Wire contracts

- TBD

## Order and gates

- TBD
