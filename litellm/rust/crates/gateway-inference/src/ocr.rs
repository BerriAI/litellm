use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{Json, body::Bytes, extract::State, http::request::Parts, response::IntoResponse};
use litellm_inference_ocr::types::{LiteLLMOcrRequest, OcrConnectionInputs, OcrDocumentInput};
use litellm_llms::base_llm::ocr::transformation::decode_request_value;
use litellm_llms_types::formats::ocr::OcrDocument;
use litellm_secret_value::SecretValue;
use serde_json::Value;

use crate::{
    Error, Gateway,
    admit::{Planned, Refusal},
    request::{self, InferenceBody},
};

pub(crate) struct Call(LiteLLMOcrRequest);

pub(crate) async fn plan(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    parts: &Parts,
    body: Bytes,
) -> Result<Call, Refusal> {
    let InferenceBody {
        fields: body,
        upload,
    } = request::inference_body(parts, body).await?;
    let header_format = parts
        .headers
        .get("x-req-format")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let deployment = request::resolve_deployment(gateway, "/ocr", &body)?;
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
        )
        .map_err(Error::from)?
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
    )
    .map_err(Error::from)?;
    Ok(Call(call))
}

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    Planned(Call(call)): Planned<Call>,
) -> Result<impl IntoResponse, Error> {
    let response = gateway.ocr.execute(call, &(), None).await?;
    Ok(Json(match response.provider_native_response {
        Some(native) => Value::Object(native),
        None => response.into_json(),
    }))
}
