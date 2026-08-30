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
pub mod alerting;
#[cfg(feature = "server")]
pub mod auth;
#[cfg(feature = "server")]
pub mod caching;
#[cfg(feature = "server")]
pub mod config;
#[cfg(feature = "server")]
#[cfg(test)]
mod config_tests;
#[cfg(feature = "server")]
pub mod config_watcher;
#[cfg(feature = "server")]
pub mod hardening;
#[cfg(feature = "server")]
pub mod load_testing;
#[cfg(feature = "server")]
pub mod middleware;
#[cfg(feature = "server")]
pub mod metrics;
#[cfg(feature = "server")]
pub mod routes;
#[cfg(feature = "server")]
pub mod streaming;
#[cfg(feature = "server")]
pub mod state;
#[cfg(feature = "server")]
pub mod validation;

mod constants;
pub mod integrations;
#[cfg(feature = "server")]
#[cfg(test)]
mod integration_tests;
#[cfg(feature = "server")]
#[cfg(test)]
mod routing_strategy_tests;
#[cfg(feature = "server")]
#[cfg(test)]
mod zero_alloc_tests;
#[cfg(feature = "server")]
#[cfg(test)]
mod comprehensive_integration_tests;
#[cfg(feature = "server")]
#[cfg(test)]
mod middleware_integration_tests;
#[cfg(feature = "server")]
#[cfg(test)]
mod e2e_tests;
#[cfg(feature = "server")]
mod realtime;
