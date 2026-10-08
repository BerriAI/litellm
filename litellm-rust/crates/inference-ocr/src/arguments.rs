use litellm_auth_types::{Setting, kwarg_names};
use litellm_core_utils::call_arguments::ArgumentSpec;
use litellm_llms::base_llm::ocr::error::Error;
use litellm_owned_params::is_secret;

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
    let auth_settings: &'static [Setting] = match config {
        OcrConfigKind::AwsTextract | OcrConfigKind::AwsTextractAnalyze => {
            litellm_auth_aws::settings::SETTINGS
        }
        OcrConfigKind::AzureAi
        | OcrConfigKind::AzureDocumentIntelligence
        | OcrConfigKind::AzureCohere => litellm_auth_azure::settings::SETTINGS,
        OcrConfigKind::VertexAi | OcrConfigKind::VertexDeepSeek => {
            litellm_auth_gcp::settings::SETTINGS
        }
        _ => &[],
    };
    Ok(COMMON_OPTION_FIELDS
        .iter()
        .chain(provider_fields)
        .copied()
        .chain(kwarg_names(auth_settings))
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
                secret: is_secret(name),
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::mistral_keeps_provider_and_common_options("mistral/model", None, "pages", true)]
    #[case::mistral_keeps_req_format("mistral/model", None, "req_format", true)]
    #[case::mistral_has_no_vertex_credentials("mistral/model", None, "vertex_project", false)]
    #[case::vertex_takes_its_credentials(
        "vertex_ai/deepseek-ocr",
        None,
        "vertex_credentials",
        true
    )]
    #[case::vertex_takes_the_alias("vertex_ai/deepseek-ocr", None, "vertex_ai_project", true)]
    #[case::vertex_has_no_mistral_option("vertex_ai/deepseek-ocr", None, "pages", false)]
    #[case::vertex_has_no_aws_credentials("vertex_ai/deepseek-ocr", None, "aws_region_name", false)]
    #[case::textract_takes_aws_credentials(
        "detect-document-text",
        Some("aws_textract"),
        "aws_session_token",
        true
    )]
    #[case::azure_takes_entra_fields("model", Some("azure_ai"), "azure_federated_token_file", true)]
    fn consumed_names_follow_the_provider_family(
        #[case] model: &str,
        #[case] provider: Option<&str>,
        #[case] name: &str,
        #[case] consumed: bool,
    ) {
        let names = consumed_optional_param_names(model, provider).unwrap();
        assert_eq!(names.contains(&name), consumed);
    }

    #[rstest]
    #[case::client_secret("client_secret", true)]
    #[case::azure_ad_token("azure_ad_token", true)]
    #[case::tenant_id("tenant_id", false)]
    #[case::req_format("req_format", false)]
    fn consumed_specs_carry_the_secret_flag(#[case] name: &str, #[case] secret: bool) {
        let specs = consumed_optional_params("model", Some("azure_ai")).unwrap();
        let spec = specs.iter().find(|spec| spec.name == name).unwrap();
        assert_eq!(spec.secret, secret);
    }
}
