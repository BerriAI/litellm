//! LiteLLM AI Gateway library.
//!
//! Two layers, split by feature so the Python `cdylib` can depend on the I/O
//! without pulling in the HTTP server:
//!
//! - Call-type modules such as [`ocr`]: host-specific lifecycle adapters and
//!   compatibility entrypoints. OCR provider execution lives in `litellm-runtime`.
//! - [`io`]: compatibility exports and realtime WebSocket splice helpers.
//! - The server modules ([`auth`], [`routes`], [`state`]) and anything pulling
//!   `axum` are gated behind the `server` feature, which the `litellm-ai-gateway`
//!   binary turns on. The `python-config` feature additionally pulls in [`python`]
//!   for the load-time config reader.

pub mod audio_transcription;
mod client;
pub mod io;
pub mod ocr;

/// GIL-activity tracking. Pure (atomics only); shared by the `server` routes and
/// the `python-config` reader, so it is available without either feature.
pub mod gil;

#[cfg(feature = "server")]
pub mod auth;
#[cfg(feature = "server")]
pub mod routes;
#[cfg(feature = "server")]
pub mod state;

mod constants;
pub mod integrations;
#[cfg(feature = "server")]
mod realtime;

#[cfg(feature = "python-config")]
pub mod python;
