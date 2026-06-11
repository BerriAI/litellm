use crate::{GuardrailInput, GuardrailOutcome, InputType, ProviderError, RequestContext};

#[async_trait::async_trait]
pub trait Guardrail: Send + Sync {
    fn provider_name(&self) -> &'static str;

    async fn apply(
        &self,
        input: &GuardrailInput,
        input_type: InputType,
        ctx: &RequestContext,
    ) -> Result<GuardrailOutcome, ProviderError>;
}
