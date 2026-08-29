use std::sync::Arc;

use custom_guardrail::CustomGuardrail;
use custom_logger::CustomLogger;
use types::RequestMetadata;

pub mod custom_guardrail;
pub mod custom_logger;
pub mod types;

#[derive(Default)]
pub struct CallbackOptions {
    pub callbacks: Vec<Arc<dyn CustomLogger>>,
    pub guardrails: Vec<Arc<dyn CustomGuardrail>>,
    pub request_metadata: RequestMetadata,
}
