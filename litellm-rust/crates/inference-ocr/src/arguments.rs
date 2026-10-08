use litellm_core_utils::{
    call_arguments::ArgumentSpec,
    params::{CredentialFamily, family_names, is_secret_param},
};
use litellm_llms::base_llm::ocr::error::Error;

use super::provider_config::{OcrConfigKind, resolve_provider_config};

const COMMON_OPTION_FIELDS: &[&str] = &["req_format", "extra_body", "max_response_bytes"];

pub fn is_supported_request(model: &str, custom_llm_provider: Option<&str>) -> bool {
    resolve_provider_config(model, custom_llm_provider).is_ok()
}

pub fn consumed_optional_param_names(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<Vec<&'static str>, Error> {
    let (model, config) = resolve_provider_config(model, custom_llm_provider)?;
    let provider_fields = config.get_supported_ocr_params(&model);
    let family = match config {
        OcrConfigKind::AwsTextract | OcrConfigKind::AwsTextractAnalyze => {
            Some(CredentialFamily::Aws)
        }
        OcrConfigKind::AzureAi
        | OcrConfigKind::AzureDocumentIntelligence
        | OcrConfigKind::AzureCohere => Some(CredentialFamily::Azure),
        OcrConfigKind::VertexAi | OcrConfigKind::VertexDeepSeek => Some(CredentialFamily::Vertex),
        _ => None,
    };
    Ok(COMMON_OPTION_FIELDS
        .iter()
        .chain(provider_fields)
        .copied()
        .chain(family.into_iter().flat_map(family_names))
        .collect())
}

pub fn consumed_optional_params(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<Vec<ArgumentSpec>, Error> {
    consumed_optional_param_names(model, custom_llm_provider).map(|names| {
        names
            .into_iter()
            .map(|name| ArgumentSpec {
                name,
                secret: is_secret_param(name),
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consumed_params_include_provider_options_and_mark_credentials() {
        let mistral = consumed_optional_param_names("mistral/model", None).unwrap();
        assert!(mistral.contains(&"pages"));
        assert!(mistral.contains(&"req_format"));
        assert!(!mistral.contains(&"vertex_project"));

        let vertex = consumed_optional_param_names("vertex_ai/deepseek-ocr", None).unwrap();
        assert!(!vertex.contains(&"temperature"));
        assert!(vertex.contains(&"vertex_credentials"));
        assert!(!vertex.contains(&"pages"));

        let azure = consumed_optional_params("model", Some("azure_ai")).unwrap();
        assert!(
            azure
                .iter()
                .any(|spec| spec.name == "client_secret" && spec.secret)
        );
        assert!(
            azure
                .iter()
                .any(|spec| spec.name == "tenant_id" && !spec.secret)
        );
    }
}
