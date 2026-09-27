use std::sync::Arc;

use axum::{Json, extract::State, http::HeaderMap};
use litellm_auth::SecretValue;
use litellm_core::ocr::{
    client::perform,
    types::{LiteLLMOcrRequest, OcrConnectionInputs, OcrDocumentInput},
};
use litellm_llms::base_llm::ocr::transformation::OcrDocument;
use serde_json::Value;

use crate::{
    Error, Gateway,
    request::{self, InferenceBody},
};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    headers: HeaderMap,
    body: InferenceBody,
) -> Result<Json<Value>, Error> {
    handle(&gateway, &headers, body).await.map(Json)
}

async fn handle(
    gateway: &Gateway,
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
    let deployment = request::deployment(gateway, &body)?;
    let document = match upload {
        Some(upload) => OcrDocumentInput::Bytes {
            bytes: upload.bytes,
            file_name: upload.file_name,
            mime_type: upload.mime_type,
        },
        None => OcrDocument::try_from(
            body.get("document")
                .cloned()
                .ok_or_else(|| Error::InvalidBody("document is required".into()))?,
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
    let response = perform(&gateway.ocr, call).await?;
    match response.provider_native_response {
        Some(native) => Ok(Value::Object(native)),
        None => Ok(response.into_json()),
    }
}
