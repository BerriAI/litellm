use litellm_core::guardrails::{
    GuardrailInput, GuardrailOutcome, InputType, ProviderError, RequestContext,
};

/// A guardrail provider. Implementations run synchronously and are called with
/// the GIL released from the bridge; network providers block on a shared
/// blocking HTTP client.
pub trait Guardrail: Send + Sync {
    fn apply(
        &self,
        input: &GuardrailInput,
        input_type: InputType,
        ctx: &RequestContext,
    ) -> Result<GuardrailOutcome, ProviderError>;
}
