use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{Json, extract::State, http::HeaderMap, response::IntoResponse};
use litellm_auth::SecretValue;
use litellm_core::ocr::types::{LiteLLMOcrRequest, OcrConnectionInputs, OcrDocumentInput};
use litellm_llms::base_llm::ocr::transformation::decode_request_value;
use litellm_llms_types::formats::ocr::OcrDocument;
use serde_json::Value;

use crate::{
    Error, Gateway,
    request::{self, InferenceBody},
};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    headers: HeaderMap,
    body: InferenceBody,
) -> Result<impl IntoResponse, Error> {
    handle(&gateway, &identity, &headers, body).await.map(Json)
}

async fn handle(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    headers: &HeaderMap,
    InferenceBody {
        fields: body,
        upload,
    }: InferenceBody,
) -> Result<Value, Error> {
    let header_format = headers
        .get("x-req-format")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let deployment = request::resolve_deployment(gateway, &body)?;
    request::authorize_model(identity, deployment, &body).await?;
    let document = match upload {
        Some(upload) => OcrDocumentInput::Bytes {
            bytes: upload.bytes,
            file_name: upload.file_name,
            mime_type: upload.mime_type,
        },
        None => decode_request_value::<OcrDocument>(
            body.get("document").cloned().unwrap_or_default(),
            "document",
        )?
        .into(),
    };
    let format = body
        .get("req_format")
        .filter(|value| !value.is_null())
        .cloned()
        .or_else(|| header_format.map(Value::String));
    let format = format.map(|value| match value {
        Value::String(value) => Value::String(value.trim().to_ascii_lowercase()),
        value => value,
    });
    let options = body
        .into_iter()
        .filter(|(name, _)| !matches!(name.as_str(), "model" | "document" | "req_format"))
        .chain(format.map(|value| ("req_format".into(), value)))
        .collect();
    let call = LiteLLMOcrRequest::from_inputs(
        deployment.model.clone(),
        document,
        deployment.custom_llm_provider.as_deref(),
        options,
        OcrConnectionInputs {
            api_key: deployment.api_key.clone().map(SecretValue::new),
            api_base: deployment.api_base.clone(),
            timeout: deployment.timeout,
            ..Default::default()
        },
    )?;
    let response = gateway.ocr.execute(call, &(), None).await?;
    match response.provider_native_response {
        Some(native) => Ok(Value::Object(native)),
        None => Ok(response.into_json()),
    }
}
