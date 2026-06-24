//! In-process PII guardrail. Pure CPU work; wraps the core engine so it slots
//! into the same dispatch as the network providers without any HTTP.

use std::time::Instant;

use litellm_core::guardrails::{
    GuardrailInput, GuardrailOutcome, InputType, LocalPiiConfig, LocalPiiEngine, ProviderError,
    RequestContext,
};

use crate::io::guardrails::provider::Guardrail;

pub struct LocalPii {
    engine: LocalPiiEngine,
}

impl LocalPii {
    pub fn new(config: LocalPiiConfig) -> Self {
        Self {
            engine: LocalPiiEngine::new(config),
        }
    }
}

impl Guardrail for LocalPii {
    fn apply(
        &self,
        input: &GuardrailInput,
        _input_type: InputType,
        _ctx: &RequestContext,
    ) -> Result<GuardrailOutcome, ProviderError> {
        let start = Instant::now();
        let verdict = self.engine.scan_texts(input.texts.clone());
        Ok(GuardrailOutcome {
            verdict,
            provider_response: serde_json::json!({"engine": "local_pii"}),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
}
