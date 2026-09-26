use litellm_core_utils::call_arguments::ArgumentSpec;
use litellm_llms::base_llm::ocr::error::Error;

use super::provider_config::{OcrConfigKind, resolve_provider_config};

const COMMON_OPTION_FIELDS: &[&str] = &["req_format", "extra_body", "max_response_bytes"];
const AZURE_AUTH_OPTION_FIELDS: &[&str] = &[
    "azure_ad_token",
    "tenant_id",
    "client_id",
    "client_secret",
    "azure_scope",
    "azure_authority_host",
    "azure_credential",
    "azure_federated_token_file",
    "enable_azure_ad_token_refresh",
];
const AWS_AUTH_OPTION_FIELDS: &[&str] = &[
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_session_name",
    "aws_profile_name",
    "aws_role_name",
    "aws_web_identity_token",
    "aws_sts_endpoint",
    "aws_external_id",
];
const VERTEX_AUTH_OPTION_FIELDS: &[&str] = &[
    "vertex_credentials",
    "vertex_ai_credentials",
    "vertex_project",
    "vertex_ai_project",
    "vertex_location",
    "vertex_ai_location",
];

pub fn is_supported_request(model: &str, custom_llm_provider: Option<&str>) -> bool {
    resolve_provider_config(model, custom_llm_provider).is_ok()
}

pub fn consumed_optional_param_names(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<Vec<&'static str>, Error> {
    let (model, config) = resolve_provider_config(model, custom_llm_provider)?;
    let provider_fields = config.get_supported_ocr_params(&model);
    let auth_fields: &[&str] = match config {
        OcrConfigKind::AwsTextract | OcrConfigKind::AwsTextractAnalyze => AWS_AUTH_OPTION_FIELDS,
        OcrConfigKind::AzureAi
        | OcrConfigKind::AzureDocumentIntelligence
        | OcrConfigKind::AzureCohere => AZURE_AUTH_OPTION_FIELDS,
        OcrConfigKind::VertexAi | OcrConfigKind::VertexDeepSeek => VERTEX_AUTH_OPTION_FIELDS,
        _ => &[],
    };
    Ok(COMMON_OPTION_FIELDS
        .iter()
        .chain(provider_fields)
        .chain(auth_fields)
        .copied()
        .collect())
}

pub(crate) fn is_secret_param(name: &str) -> bool {
    matches!(
        name,
        "azure_ad_token"
            | "client_secret"
            | "azure_federated_token_file"
            | "vertex_credentials"
            | "vertex_ai_credentials"
            | "aws_secret_access_key"
            | "aws_session_token"
            | "aws_web_identity_token"
    )
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
