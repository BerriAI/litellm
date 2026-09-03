//! OCR service logic.

use litellm_core::router::Router;
use litellm_core::{Error, ocr};
use serde_json::Value;

#[tracing::instrument(name = "ocr_service", skip(router, body))]
pub async fn run(router: &Router, body: Value) -> Result<Value, Error> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRequest("missing model field".to_string()))?;

    let document = body
        .get("document")
        .ok_or_else(|| Error::InvalidRequest("missing document field".to_string()))?
        .clone();

    let deployment = router
        .get_deployment(model)
        .ok_or_else(|| Error::Routing(format!("no deployment found for model: {model}")))?;

    let response = ocr::ocr(
        deployment.litellm_params.model.clone(),
        document,
        deployment.litellm_params.api_key.clone(),
        deployment.litellm_params.api_base.clone(),
    )
    .await?;

    Ok(response)
}
