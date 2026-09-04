//! LiteLLM AI Gateway library.
//!
//! - Call-type modules such as [`ocr`]: provider transforms, lifecycle hooks,
//!   and provider I/O. Always available — no feature required. These predate the
//!   rule that a route's entrypoint and handler live in `litellm-core` (see
//!   `litellm_core::messages`) and move there as they are touched.
//! - [`io`]: compatibility exports and realtime WebSocket splice helpers.
//! - [`runtime`]: framework-independent orchestration used by HTTP and Python
//!   hosts.
//!
//! The Axum host lives in the separate `litellm-gateway-server` crate.

pub mod audio_transcription;
mod client;
pub mod io;
pub mod ocr;

pub mod realtime;
pub mod runtime;
#[cfg(feature = "trace-parity")]
pub mod trace_parity;

mod constants;
pub mod integrations;
