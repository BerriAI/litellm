//! LiteLLM AI Gateway library.
//!
//! Two layers, split by feature so the Python `cdylib` can depend on the I/O
//! without pulling in the HTTP server:
//!
//! - Call-type modules such as [`ocr`]: provider transforms, lifecycle hooks,
//!   and provider I/O. Always available — no feature required. These predate the
//!   rule that a route's entrypoint and handler live in `litellm-core` (see
//!   `litellm_core::messages`) and move there as they are touched.
//! - [`io`]: compatibility exports and realtime WebSocket splice helpers.
//! - The server modules ([`auth`], [`routes`], [`state`]) and anything pulling
//!   `axum` are gated behind the `server` feature, which the `litellm-ai-gateway`
//!   binary turns on.

pub mod audio_transcription;
mod client;
pub mod io;
pub mod ocr;

#[cfg(feature = "server")]
pub mod auth;
#[cfg(feature = "server")]
pub mod routes;
#[cfg(feature = "server")]
pub mod state;
#[cfg(feature = "trace-parity")]
pub mod trace_parity;

mod constants;
pub mod integrations;
#[cfg(feature = "server")]
mod realtime;
