//! LiteLLM AI Gateway library.
//!
//! Two layers, split by feature so the Python `cdylib` can depend on the I/O
//! without pulling in the HTTP server:
//!
//! - [`io`]: compatibility exports and realtime WebSocket splice helpers.
//! - The server modules ([`routes`], [`state`]) and anything pulling
//!   `axum` are gated behind the `server` feature, which the `litellm-ai-gateway`
//!   binary turns on.

pub mod audio_transcription;
pub mod io;

#[cfg(feature = "server")]
pub mod routes;
#[cfg(feature = "server")]
pub mod state;
#[cfg(feature = "trace-parity")]
pub mod trace_parity;

mod constants;
