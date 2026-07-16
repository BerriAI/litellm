use litellm_core::error::CoreError;
use litellm_core::router::{resolve_deployment_provider, Router};
use litellm_core::CoreResult;
use serde_json::Value;

use crate::io::ocr::{ocr, OcrRequest};

use super::transport::OcrCall;

fn present(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
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
    let (provider_model, provider) =
        resolve_deployment_provider(&params.model, params.custom_llm_provider.as_deref())?;

    let OcrCall {
        model: requested_model,
        document,
        optional_params,
        timeout,
    } = call;

    let response = ocr(OcrRequest {
        model: &provider_model,
        document,
        api_key: present(params.api_key.as_deref()),
        api_base: present(params.api_base.as_deref()),
        custom_llm_provider: &provider,
        extra_headers: None,
        optional_params,
        timeout,
    })
    .await?;

    Ok(normalize_response_model(response, requested_model))
}

fn normalize_response_model(response: Value, requested_model: String) -> Value {
    let Value::Object(map) = response else {
        return response;
    };
    Value::Object(
        map.into_iter()
            .filter(|(key, _)| key != "model")
            .chain(std::iter::once((
                "model".to_string(),
                Value::String(requested_model),
            )))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn present_treats_empty_and_whitespace_as_absent() {
        assert_eq!(present(Some("sk-key")), Some("sk-key"));
        assert_eq!(present(Some("  sk-key  ")), Some("sk-key"));
        assert_eq!(present(Some("")), None);
        assert_eq!(present(Some("   ")), None);
        assert_eq!(present(None), None);
    }

    #[test]
    fn normalize_response_model_replaces_model_and_keeps_other_fields() {
        let response = serde_json::json!({
            "object": "ocr",
            "model": "mistral-ocr-latest",
            "pages": [{"markdown": "hi"}]
        });
        let normalized = normalize_response_model(response, "rust-ocr-mistral".to_string());
        assert_eq!(normalized["model"], "rust-ocr-mistral");
        assert_eq!(normalized["object"], "ocr");
        assert_eq!(normalized["pages"][0]["markdown"], "hi");
    }

    #[test]
    fn normalize_response_model_adds_model_when_absent() {
        let response = serde_json::json!({"object": "ocr"});
        let normalized = normalize_response_model(response, "rust-ocr-mistral".to_string());
        assert_eq!(normalized["model"], "rust-ocr-mistral");
    }

    #[test]
    fn normalize_response_model_leaves_non_object_untouched() {
        let response = serde_json::json!("not-an-object");
        let normalized = normalize_response_model(response, "rust-ocr-mistral".to_string());
        assert_eq!(normalized, serde_json::json!("not-an-object"));
    }
}
