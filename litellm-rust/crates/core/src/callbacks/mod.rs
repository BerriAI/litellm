//! Callback and guardrail contracts. Names map 1:1 to Python
//! `litellm/integrations/`:
//!   - [`custom_guardrail::CustomGuardrail`] — the guardrail callback trait
//!   - [`custom_logger::CustomLogger`] — the callback trait
//!   - [`types`] — the typed `StandardLoggingPayload` wire contract
//!
//! Core owns the traits, runners, and payload types; hosts own the I/O
//! implementations (e.g. the gateway's HTTP shipper to the Python proxy).

pub mod custom_guardrail;
pub mod custom_logger;
pub mod types;
