//! LiteLLM AI Gateway library.
//!
//! Two layers, split by feature so the Python `cdylib` can depend on the I/O
//! without pulling in the HTTP server:
//!
//! - [`io`]: all network I/O (OCR HTTP call, realtime WebSocket splice, the
//!   pre-warmed realtime pool). Always available — no feature required. The
//!   Python bridge links this for `run_ocr`.
//! - The server modules ([`auth`], [`routes`], [`state`]) and anything pulling
//!   `axum` are gated behind the `server` feature, which the `litellm-ai-gateway`
//!   binary turns on. The `python-config` feature additionally pulls in [`python`]
//!   for the load-time config reader.

pub mod io;

/// GIL-activity tracking. Pure (atomics only); shared by the `server` routes and
/// the `python-config` reader, so it is available without either feature.
pub mod gil;

#[cfg(feature = "server")]
pub mod auth;
#[cfg(feature = "server")]
pub mod routes;
#[cfg(feature = "server")]
pub mod state;

// Realtime request logging. Only the server serves realtime, so these are
// `server`-gated; `io::realtime` exposes the generic `observe` hook while the
// collector and callback fan-out live here.
mod constants;
#[cfg(feature = "server")]
pub mod integrations;
#[cfg(feature = "server")]
mod realtime;

#[cfg(feature = "python-config")]
pub mod python;
