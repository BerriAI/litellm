//! LiteLLM AI Gateway library.
//!
//! Pure Rust AI gateway with no Python dependencies. All logic (auth, routing,
//! cost calculation, spend tracking) lives in `litellm-core`. This crate provides
//! the HTTP server (axum) and provider I/O.
//!
//! Modules:
//! - Call-type modules ([`ocr`], [`audio_transcription`]): provider transforms and I/O.
//! - [`io`]: realtime WebSocket splice helpers.
//! - Server modules ([`auth`], [`routes`], [`state`], [`config`]): gated behind `server`.

pub mod audio_transcription;
mod client;
pub mod io;
pub mod ocr;

pub mod gil;

#[cfg(feature = "server")]
pub mod auth;
#[cfg(feature = "server")]
pub mod config;
#[cfg(feature = "server")]
pub mod config_watcher;
#[cfg(feature = "server")]
pub mod hardening;
#[cfg(feature = "server")]
pub mod metrics;
#[cfg(feature = "server")]
pub mod routes;
#[cfg(feature = "server")]
pub mod state;

mod constants;
pub mod integrations;
#[cfg(feature = "server")]
mod realtime;
