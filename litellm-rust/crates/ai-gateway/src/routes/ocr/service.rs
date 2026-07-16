use litellm_core::error::CoreError;
use litellm_core::router::Router;
use litellm_core::CoreResult;
use serde_json::Value;

use crate::io::ocr::{ocr, OcrRequest};

use super::transport::OcrCall;

fn present(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn split_provider(model: &str) -> CoreResult<(&str, &str)> {
    model
        .split_once('/')
        .filter(|(provider, rest)| !provider.is_empty() && !rest.is_empty())
        .ok_or_else(|| {
            CoreError::InvalidProvider(format!(
                "deployment model '{model}' must be prefixed with an OCR provider, e.g. \
                 'mistral/mistral-ocr-latest'"
            ))
        })
}

pub async fn run_ocr(router: &Router, call: OcrCall) -> CoreResult<Value> {
    let deployment = router
        .get_available_deployment(&call.model)
        .ok_or_else(|| {
            CoreError::Routing(format!(
                "no deployment available for model '{}'",
                call.model
            ))
        })?;
    let params = &deployment.litellm_params;
    let (provider, provider_model) = split_provider(&params.model)?;

    ocr(OcrRequest {
        model: provider_model,
        document: call.document,
        api_key: present(params.api_key.as_deref()),
        api_base: present(params.api_base.as_deref()),
        custom_llm_provider: provider,
        extra_headers: None,
        optional_params: call.optional_params,
        timeout: call.timeout,
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_provider_prefix() {
        assert_eq!(
            split_provider("mistral/mistral-ocr-latest").expect("splits"),
            ("mistral", "mistral-ocr-latest")
        );
        assert_eq!(
            split_provider("azure_ai/doc-intelligence/prebuilt-layout").expect("splits"),
            ("azure_ai", "doc-intelligence/prebuilt-layout")
        );
    }

    #[test]
    fn present_treats_empty_and_whitespace_as_absent() {
        assert_eq!(present(Some("sk-key")), Some("sk-key"));
        assert_eq!(present(Some("  sk-key  ")), Some("sk-key"));
        assert_eq!(present(Some("")), None);
        assert_eq!(present(Some("   ")), None);
        assert_eq!(present(None), None);
    }

    #[test]
    fn rejects_model_without_provider_prefix() {
        assert!(matches!(
            split_provider("mistral-ocr-latest"),
            Err(CoreError::InvalidProvider(_))
        ));
        assert!(matches!(
            split_provider("mistral/"),
            Err(CoreError::InvalidProvider(_))
        ));
    }
}
