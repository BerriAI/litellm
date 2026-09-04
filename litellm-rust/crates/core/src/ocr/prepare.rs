use serde_json::{Map, Value};

use super::common_utils::ocr_provider_config;
use super::types::{OcrRequest, PreparedOcrRequest};
use crate::error::Error;
use crate::ocr::transformation::OcrProviderConfig;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(super) async fn prepare_ocr_call(request: OcrRequest<'_>) -> Result<PreparedOcrRequest, Error> {
    let provider_info = get_custom_llm_provider(request.model, request.custom_llm_provider)
        .or_else(|| {
            request
                .custom_llm_provider
                .map(|provider| CustomLlmProvider {
                    model: request.model,
                    custom_llm_provider: provider,
                })
        })
        .unwrap_or(CustomLlmProvider {
            model: request.model,
            custom_llm_provider: "mistral",
        });
    let model = provider_info.model.to_string();
    let provider = provider_info.custom_llm_provider.to_string();
    let config = ocr_provider_config(&provider, &model)
        .ok_or_else(|| Error::InvalidProvider(provider.clone()))?;
    validate_request_format(config, &request.optional_params, &provider)?;
    let optional_params = map_optional_params(config, &request.optional_params);

    Ok(PreparedOcrRequest {
        model,
        config,
        document: request.document,
        api_key: request.api_key.map(str::to_string),
        api_base: request.api_base.map(str::to_string),
        extra_headers: request.extra_headers,
        url_params: request.optional_params,
        optional_params,
        requires_reducto_upload: provider == "reducto",
        timeout: request.timeout,
    })
}

fn map_optional_params(
    config: &'static dyn OcrProviderConfig,
    optional_params: &Map<String, Value>,
) -> Map<String, Value> {
    let supported = config.supported_ocr_params();
    config.map_ocr_params(
        &optional_params
            .iter()
            .filter(|(name, _)| supported.contains(&name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
    )
}

fn validate_request_format(
    config: &'static dyn OcrProviderConfig,
    optional_params: &Map<String, Value>,
    provider: &str,
) -> Result<(), Error> {
    let Some(format) = optional_params.get("req_format") else {
        return Ok(());
    };
    match format.as_str() {
        Some("litellm") => Ok(()),
        Some("native") if config.supported_ocr_params().contains(&"req_format") => Ok(()),
        Some("native") => Err(Error::InvalidRequest(format!(
            "`req_format=native` is not supported for provider {provider}"
        ))),
        _ => Err(Error::InvalidRequest(format!(
            "Invalid `req_format`: {format}. Expected `litellm` or `native`"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};

    use super::{OcrRequest, prepare_ocr_call};
    use crate::error::Error;

    fn request_with_format(format: &str) -> OcrRequest<'_> {
        OcrRequest {
            model: "mistral/mistral-ocr-latest",
            document: json!({
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf"
            }),
            api_key: Some("sk-test"),
            api_base: None,
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: Map::from_iter([("req_format".to_string(), json!(format))]),
            timeout: None,
        }
    }

    #[tokio::test]
    async fn native_format_rejected_for_provider_without_support_as_bad_request() {
        let error = match prepare_ocr_call(request_with_format("native")).await {
            Ok(_) => panic!("native format should be rejected"),
            Err(error) => error,
        };
        assert!(
            matches!(error, Error::InvalidRequest(message) if message.contains("not supported for provider"))
        );
    }

    #[tokio::test]
    async fn unknown_format_rejected_for_provider_without_support_as_bad_request() {
        let error = match prepare_ocr_call(request_with_format("raw")).await {
            Ok(_) => panic!("unknown format should be rejected"),
            Err(error) => error,
        };
        assert!(
            matches!(error, Error::InvalidRequest(message) if message.contains("Invalid `req_format`"))
        );
    }
}
