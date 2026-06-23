//! Minimal Rust port of LiteLLM's `router.py`.
//!
//! A [`Router`] is built from a `model_list` of [`Deployment`]s
//! (`{ model_name, litellm_params: { model, api_key, api_base } }`) and selects
//! one per request via a [`RoutingStrategy`]. For now the only strategy is
//! `simple-shuffle` — a uniform random pick within a `model_name` group.
//!
//! Layered by responsibility:
//! - [`deployment`] — the `model_list` data types.
//! - [`strategy`] — how a deployment is chosen.
//! - [`router`] — construction, lookup, and route methods (e.g. realtime).

mod deployment;
mod router;
mod strategy;

pub use deployment::{Deployment, LiteLLMParams};
pub use router::Router;
pub use strategy::RoutingStrategy;
