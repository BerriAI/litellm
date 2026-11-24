use axum::{
    Json,
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use litellm_inference_ocr::{
    OcrRoute,
    types::{LiteLLMOcrRequest, OcrConnectionInputs, OcrDocumentInput},
};
use litellm_llms::base_llm::ocr::transformation::decode_request_value;
use litellm_llms_types::formats::ocr::OcrDocument;
use litellm_router_core::{Attempt, Op};
use serde_json::{Map, Value};

use crate::{
    Error,
    error::ErrorFormat,
    hooks::CallType,
    model_restamp::ReportedModel,
    request::{InferenceBody, Upload},
    routing::{self, Host, Prompt, RouteAttempt},
    serve::{self, Call, Endpoint},
    stream_event::Counts,
};

pub(crate) async fn create<H: Host>(call: Call<H>, body: Result<InferenceBody, Error>) -> Response {
    let (body, upload) = match body {
        Ok(InferenceBody { fields, upload }) => (Ok(fields), upload),
        Err(error) => (Err(error), None),
    };
    serve::run(&call, body, |body| {
        Ok(OcrAttempt::new(
            &call.gateway.ocr,
            &call.headers,
            upload,
            body,
        ))
    })
    .await
}

struct OcrAttempt<'a> {
    route: &'a OcrRoute,
    upload: Option<Upload>,
    document: Value,
    options: Map<String, Value>,
}

impl<'a> OcrAttempt<'a> {
    fn new(
        route: &'a OcrRoute,
        headers: &HeaderMap,
        upload: Option<Upload>,
        body: Map<String, Value>,
    ) -> Self {
        let header_format = headers
            .get("x-req-format")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
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
        Self {
            route,
            upload,
            document,
            options,
        }
    }
}

impl Endpoint for OcrAttempt<'_> {
    const CALL_TYPE: CallType = CallType::Aocr;
    const OP: Op = Op::OCR;
    const PATH: &'static str = "/ocr";
    const ERRORS: ErrorFormat = ErrorFormat::OpenAi;

    fn render(outcome: Value, _reported: ReportedModel<'_>) -> Result<Response, Error> {
        Ok(Json(outcome).into_response())
    }
}

impl RouteAttempt for OcrAttempt<'_> {
    type Outcome = Value;

    fn prompt(&self) -> Option<Prompt<'_>> {
        None
    }

    async fn run(&self, attempt: &Attempt) -> Result<Value, Error> {
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
        Ok(match response.provider_native_response {
            Some(native) => Value::Object(native),
            None => response.into_json(),
        })
    }

    /// Nothing: Python's router reads no `usage` off an OCR response.
    fn counts(_outcome: &Value) -> Option<Counts> {
        None
    }
}
