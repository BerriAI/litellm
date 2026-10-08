use litellm_llms::base_llm::ocr::error::Error;

use super::provider_config::resolve_provider_config;

const COMMON_OPTION_FIELDS: &[&str] = &["req_format", "extra_body", "max_response_bytes"];

pub fn is_supported_request(model: &str, custom_llm_provider: Option<&str>) -> bool {
    resolve_provider_config(model, custom_llm_provider).is_ok()
}

pub fn consumed_optional_param_names(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<Vec<&'static str>, Error> {
    let (model, config) = resolve_provider_config(model, custom_llm_provider)?;
    Ok(COMMON_OPTION_FIELDS
        .iter()
        .chain(config.get_supported_ocr_params(&model))
        .copied()
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consumed_params_are_provider_options_and_never_credentials() {
        let mistral = consumed_optional_param_names("mistral/model", None).unwrap();
        assert!(mistral.contains(&"pages"));
        assert!(mistral.contains(&"req_format"));
        assert!(!mistral.contains(&"vertex_project"));

        let vertex = consumed_optional_param_names("vertex_ai/deepseek-ocr", None).unwrap();
        assert!(!vertex.contains(&"temperature"));
        assert!(!vertex.contains(&"vertex_credentials"));
        assert!(!vertex.contains(&"pages"));

        let azure = consumed_optional_param_names("model", Some("azure_ai")).unwrap();
        assert!(!azure.contains(&"client_secret"));
        assert!(!azure.contains(&"tenant_id"));
    }
}
