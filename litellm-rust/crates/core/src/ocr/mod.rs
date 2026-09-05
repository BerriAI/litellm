mod handler;
mod prepare;

pub use handler::ocr;
pub use prepare::{OcrPlan, decode_document, parameter_names, preflight};
pub use types::{OcrDocument, OcrRequest};

pub mod transformation;
pub mod types;

#[cfg(test)]
mod tests;

pub fn ocr_provider_supported(model: &str, provider: &str) -> bool {
    crate::routing_utils::provider::get_custom_llm_provider(
        model,
        (!provider.is_empty()).then_some(provider),
    )
    .is_some_and(|resolved| {
        prepare::provider_config(resolved.model, resolved.custom_llm_provider).is_some()
    })
}
