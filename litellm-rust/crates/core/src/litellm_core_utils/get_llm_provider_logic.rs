pub use litellm_providers::provider_resolution::{CustomLlmProvider, get_custom_llm_provider};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gets_custom_llm_provider_from_model_prefix() {
        assert_eq!(
            get_custom_llm_provider("mistral/mistral-ocr-latest", None),
            Some(CustomLlmProvider {
                model: "mistral-ocr-latest",
                custom_llm_provider: "mistral",
            })
        );
        assert_eq!(
            get_custom_llm_provider("azure_ai/doc-intelligence/prebuilt-layout", None),
            Some(CustomLlmProvider {
                model: "doc-intelligence/prebuilt-layout",
                custom_llm_provider: "azure_ai",
            })
        );
        assert_eq!(get_custom_llm_provider("mistral-ocr-latest", None), None);
        assert_eq!(get_custom_llm_provider("/model", None), None);
        assert_eq!(get_custom_llm_provider("provider/", None), None);
    }

    #[test]
    fn explicit_custom_llm_provider_strips_matching_model_prefix() {
        assert_eq!(
            get_custom_llm_provider("mistral/mistral-ocr-latest", Some("mistral")),
            Some(CustomLlmProvider {
                model: "mistral-ocr-latest",
                custom_llm_provider: "mistral",
            })
        );
        assert_eq!(
            get_custom_llm_provider("mistral/mistral-ocr-latest", Some("vertex_ai")),
            Some(CustomLlmProvider {
                model: "mistral/mistral-ocr-latest",
                custom_llm_provider: "vertex_ai",
            })
        );
    }
}
