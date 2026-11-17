use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use litellm_inference_ocr::{
    OcrRoute,
    types::{LiteLLMOcrRequest, OcrConnectionInputs, OcrDocumentInput},
};
use litellm_llms::base_llm::ocr::transformation::decode_request_value;
use litellm_llms_types::formats::ocr::OcrDocument;
use litellm_secret_value::SecretValue;
use serde_json::Value;

use crate::{
    Error, Gateway,
    request::{InferenceBody, Upload},
    routing::{self, Target},
};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    headers: HeaderMap,
    InferenceBody {
        fields: body,
        upload,
    }: InferenceBody,
) -> Result<Response, Error> {
    let header_format = headers
        .get("x-req-format")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let route = gateway.ocr.clone();
    routing::call(gateway, &identity, "aocr", "/ocr", body, move |target| {
        attempt(route.clone(), upload.clone(), header_format.clone(), target)
    })
    .await
}

async fn attempt(
    route: OcrRoute,
    upload: Option<Upload>,
    header_format: Option<String>,
    target: Target,
) -> Result<Response, Error> {
    let api_key = target.api_key();
    let Target {
        deployment,
        body,
        timeout,
        ..
    } = target;
    let params = &deployment.params;
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
        params.model.clone(),
        document,
        params.custom_llm_provider.as_deref(),
        options,
        OcrConnectionInputs {
            api_key: api_key.map(SecretValue::new),
            api_base: params.api_base.clone(),
            timeout,
            ..Default::default()
        },
    )?;
    let response = route.execute(call, &(), None).await?;
    let body = match response.provider_native_response {
        Some(native) => Value::Object(native),
        None => response.into_json(),
    };
    Ok(Json(body).into_response())
}
