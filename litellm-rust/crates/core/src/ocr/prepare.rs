use litellm_auth::{InputSource, SecretValue, Sourced};
use litellm_llms::base_llm::ocr::{
    handler::OcrClient,
    transformation::{OcrConnection, OcrCredentialInputs, PreparedOcrRequest},
};

use super::provider_config::OcrProvider;
use crate::ocr::types::{LiteLLMOcrRequest, ResolvedOcrRequest};

pub(crate) fn prepare_request(
    request: ResolvedOcrRequest,
    caller_document: bool,
    client: &OcrClient,
) -> PreparedOcrRequest {
    let credentials = request.credentials.clone();
    let api_base_env = match request.config.provider() {
        OcrProvider::Mistral => Some("MISTRAL_API_BASE"),
        OcrProvider::AzureAi => Some("AZURE_AI_API_BASE"),
        OcrProvider::Cohere | OcrProvider::Reducto | OcrProvider::VertexAi => None,
    };
    let dynamic_api_key = credentials.dynamic_api_key.or_else(|| {
        credentials.api_key.clone().or_else(|| {
            request
                .config
                .get_api_key_env_var()
                .and_then(|name| client.secrets().get(name))
                .map(|value| Sourced::new(SecretValue::new(value), InputSource::Environment))
        })
    });
    let dynamic_api_base = credentials.dynamic_api_base.or_else(|| {
        credentials.api_base.clone().or_else(|| {
            api_base_env
                .and_then(|name| client.secrets().get(name))
                .map(|value| Sourced::new(value, InputSource::Environment))
        })
    });
    let resolved = request
        .config
        .resolve_connection_params(OcrCredentialInputs {
            dynamic_api_key,
            dynamic_api_base,
            ..credentials
        });
    let LiteLLMOcrRequest {
        model,
        document,
        transport,
        optional_params,
        input_sources,
        azure_ad_token_provider,
        ..
    } = request;
    PreparedOcrRequest {
        model,
        document,
        connection: OcrConnection::new(
            resolved,
            transport,
            client.settings().clone(),
            client.secrets().clone(),
        ),
        caller_document,
        optional_params,
        input_sources,
        azure_ad_token_provider,
    }
}

#[cfg(test)]
pub(crate) fn prepare_request_for_test(request: ResolvedOcrRequest) -> PreparedOcrRequest {
    prepare_request(
        request,
        true,
        &OcrClient::for_test(reqwest::Client::new(), reqwest::Client::new()),
    )
}

#[cfg(test)]
mod tests {
    use litellm_core_utils::call_arguments::{CallArguments, compose_body, parse_options};
    use serde_json::json;

    #[derive(serde::Deserialize)]
    struct KnownParams {
        pages: Option<Vec<i64>>,
    }

    #[test]
    fn parsed_provider_params_separates_known_and_extra_params() {
        let arguments: CallArguments = serde_json::from_value(json!({
            "pages": [0, 2],
            "future_ocr_option": true,
            "extra_body": {"provider_option": "value"}
        }))
        .unwrap();
        let known: KnownParams = parse_options(&arguments).unwrap();
        assert_eq!(known.pages, Some(vec![0, 2]));
        assert_eq!(arguments["future_ocr_option"], true);
        assert_eq!(arguments["extra_body"], json!({"provider_option": "value"}));
        assert_eq!(
            arguments
                .iter()
                .filter(|(name, _)| name.as_str() != "pages")
                .count(),
            2
        );
        assert_eq!(
            compose_body(&arguments, &json!({"pages": known.pages}), &["pages"]).unwrap(),
            json!({
                "pages": [0, 2], "future_ocr_option": true, "provider_option": "value"
            })
        );
    }
}
