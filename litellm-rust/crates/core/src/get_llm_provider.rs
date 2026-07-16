use crate::error::CoreError;
use crate::CoreResult;

pub fn get_llm_provider(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> CoreResult<(String, String)> {
    let model = model.trim();
    if model.is_empty() {
        return Err(CoreError::InvalidProvider(
            "deployment model is empty".to_string(),
        ));
    }

    if let Some(provider) = custom_llm_provider
        .map(str::trim)
        .filter(|provider| !provider.is_empty())
    {
        let resolved_model = model
            .strip_prefix(&format!("{provider}/"))
            .unwrap_or(model)
            .to_string();
        return Ok((resolved_model, provider.to_string()));
    }

    model
        .split_once('/')
        .filter(|(provider, rest)| !provider.is_empty() && !rest.is_empty())
        .map(|(provider, rest)| (rest.to_string(), provider.to_string()))
        .ok_or_else(|| {
            CoreError::InvalidProvider(format!(
                "deployment model '{model}' must be prefixed with a provider (e.g. \
                 'mistral/mistral-ocr-latest') or set 'custom_llm_provider'"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_provider_prefix_when_no_explicit_provider() {
        assert_eq!(
            get_llm_provider("mistral/mistral-ocr-latest", None).expect("splits"),
            ("mistral-ocr-latest".to_string(), "mistral".to_string())
        );
        assert_eq!(
            get_llm_provider("azure_ai/doc-intelligence/prebuilt-layout", None).expect("splits"),
            (
                "doc-intelligence/prebuilt-layout".to_string(),
                "azure_ai".to_string()
            )
        );
    }

    #[test]
    fn explicit_provider_wins_and_strips_matching_prefix() {
        assert_eq!(
            get_llm_provider("mistral/mistral-ocr-latest", Some("mistral")).expect("resolves"),
            ("mistral-ocr-latest".to_string(), "mistral".to_string())
        );
        assert_eq!(
            get_llm_provider("mistral-ocr-latest", Some("mistral")).expect("resolves"),
            ("mistral-ocr-latest".to_string(), "mistral".to_string())
        );
    }

    #[test]
    fn explicit_provider_keeps_unrelated_prefix() {
        assert_eq!(
            get_llm_provider("openai/some-model", Some("mistral")).expect("resolves"),
            ("openai/some-model".to_string(), "mistral".to_string())
        );
    }

    #[test]
    fn blank_explicit_provider_falls_back_to_prefix() {
        assert_eq!(
            get_llm_provider("mistral/mistral-ocr-latest", Some("   ")).expect("splits"),
            ("mistral-ocr-latest".to_string(), "mistral".to_string())
        );
    }

    #[test]
    fn rejects_model_without_provider() {
        assert!(matches!(
            get_llm_provider("mistral-ocr-latest", None),
            Err(CoreError::InvalidProvider(_))
        ));
        assert!(matches!(
            get_llm_provider("mistral/", None),
            Err(CoreError::InvalidProvider(_))
        ));
        assert!(matches!(
            get_llm_provider("   ", Some("mistral")),
            Err(CoreError::InvalidProvider(_))
        ));
    }
}
