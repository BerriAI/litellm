//! Pure-Rust logging integrations. Names map 1:1 to Python
//! `litellm/integrations/`:
//!   - [`custom_guardrail::CustomGuardrail`] — the guardrail callback trait
//!   - [`custom_logger::CustomLogger`]  — the callback trait
//!   - [`litellm_python_proxy_api::LiteLLMPythonProxyAPILogger`] — ships events
//!     to the Python proxy's `/v1/rust_control_plane/logs` endpoint
//!   - [`types`] — the typed `StandardLoggingPayload` wire contract

pub mod custom_guardrail;
pub mod custom_logger;
pub mod litellm_python_proxy_api;
pub mod types;
