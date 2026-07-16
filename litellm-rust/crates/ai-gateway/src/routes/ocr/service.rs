use litellm_core::error::CoreError;
use litellm_core::get_llm_provider::get_llm_provider;
use litellm_core::router::Router;
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
        get_llm_provider(&params.model, params.custom_llm_provider.as_deref())?;

    let OcrCall {
        model: requested_model,
        document,
        optional_params,
        timeout,
    } = call;

    let mut response = ocr(OcrRequest {
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

    if let Value::Object(map) = &mut response {
        map.insert("model".to_string(), Value::String(requested_model));
    }
    Ok(response)
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
}
