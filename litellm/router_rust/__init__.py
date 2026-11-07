"""
The Rust router core's Python side: ``RustCore`` (``core``) and the host-side modules router-core
(litellm-rust/crates/router-core) reaches through its steps.

- ``steps``: answers to host steps that need live Python objects (strategies, pre-routing,
  pre-call facts, plugins, callbacks, caller access)
- ``invoke``: the host side of ``Host::invoke``, one attempt against one deployment (request
  shaping, endpoints, streams, slots, retry and fallback metadata, prompt management, websockets)
- ``surface``: the Python ``Router`` API kept over router-core's state (registry views, model
  info, settings, health, retries, fallbacks, alerting); a Rust host needs none of it

Nothing is imported here, so importing one module does not import the core.
"""
