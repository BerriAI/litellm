use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{Extension, Json, extract::State, http::HeaderMap, response::IntoResponse};
use litellm_inference_ocr::{
    OcrRoute,
    types::{LiteLLMOcrRequest, OcrConnectionInputs, OcrDocumentInput},
};
use litellm_llms::base_llm::ocr::transformation::decode_request_value;
use litellm_llms_types::formats::ocr::OcrDocument;
use litellm_router_core::{Attempt, Op, Served};
use serde_json::{Map, Value};

use crate::{
    Ended, Error, Gateway,
    headers::Routing,
    request::{self, InferenceBody, Upload},
    routing::{self, Host, RouteAttempt},
};

pub(crate) async fn create<H: Host>(
    State(gateway): State<Arc<Gateway<H>>>,
    identity: AuthenticatedRequest,
    headers: HeaderMap,
    body: InferenceBody,
) -> Result<impl IntoResponse, Ended> {
    handle(&gateway, &identity, &headers, body)
        .await
        .map(|(routing, body)| (routing, Json(body)))
}

async fn handle<H: Host>(
    gateway: &Gateway<H>,
    identity: &AuthenticatedRequest,
    headers: &HeaderMap,
    InferenceBody {
        fields: body,
        upload,
    }: InferenceBody,
) -> Result<(Extension<Routing>, Value), Ended> {
    let header_format = headers
        .get("x-req-format")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let model = request::model(&body)?.to_owned();
    let document = body.get("document").cloned().unwrap_or_default();
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
    let attempt = OcrAttempt {
        route: &gateway.ocr,
        upload,
        document,
        options,
    };
    let routed = routing::call(gateway, identity, Op::OCR, "/ocr", &model, &attempt).await?;
    Ok((Extension(Routing::served(&routed)), routed.output.response))
}

struct OcrAttempt<'a> {
    route: &'a OcrRoute,
    upload: Option<Upload>,
    document: Value,
    options: Map<String, Value>,
}

impl RouteAttempt for OcrAttempt<'_> {
    type Outcome = Value;

    async fn run(&self, attempt: &Attempt) -> Result<Served<Value>, Error> {
        let deployment = &attempt.deployment;
        let document = match &self.upload {
            Some(upload) => OcrDocumentInput::Bytes {
                bytes: upload.bytes.clone(),
                file_name: upload.file_name.clone(),
                mime_type: upload.mime_type.clone(),
            },
            None => decode_request_value::<OcrDocument>(self.document.clone(), "document")?.into(),
        };
        let call = LiteLLMOcrRequest::from_inputs(
            deployment.params.model.clone(),
            document,
            deployment.params.custom_llm_provider.as_deref(),
            routing::attempt_body(&self.options, attempt).into(),
            OcrConnectionInputs {
                api_key: deployment.params.api_key.clone(),
                api_base: deployment.params.api_base.clone(),
                timeout: routing::timeout(attempt, false),
                ..Default::default()
            },
        )?;
        let response = self.route.execute(call, &(), None).await?;
        let body = match response.provider_native_response {
            Some(native) => Value::Object(native),
            None => response.into_json(),
        };
        let total_tokens = routing::total_tokens(body.get("usage"));
        Ok(Served::new(body, total_tokens))
    }
}
