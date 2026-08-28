//! LiteLLM AI Gateway library: the axum server that fronts the Rust SDK
//! (`litellm-core`).
//!
//! The crate owns transport, config, and auth only. Route handlers translate
//! HTTP/WS to `litellm-core` entrypoints; no provider logic lives here. The
//! `python-config` feature pulls in [`python`] for the load-time config reader
//! (the only Python interop, taken once at boot).

pub mod auth;
pub mod routes;
pub mod server;
pub mod state;

/// GIL-activity tracking. Pure (atomics only); recorded by the `python-config`
/// reader and exposed by the `/health/gil` route.
pub mod gil;

mod constants;
pub mod integrations;

#[cfg(feature = "python-config")]
pub mod python;
