use std::future::Future;
use std::pin::Pin;

use serde::Serialize;
use serde_json::Value;

use super::types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrDocument, ResolvedOcrRequest};
use crate::ocr::Error;
use litellm_callbacks::context::{CallContext, CallTiming};

pub type OcrHookFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;
pub type OcrLogFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

#[derive(Clone, Debug, Serialize)]
pub struct OcrPreCallRequest {
    pub model: String,
    pub custom_llm_provider: String,
    pub document: OcrDocument,
    pub optional_params: Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct OcrDuringCallRequest {
    pub model: String,
    pub custom_llm_provider: String,
    pub api_key: Option<String>,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
    pub optional_params: Value,
    #[serde(skip)]
    pub retained_fields: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OcrPostCallRequest {
    pub original_response: Value,
}

pub trait OcrHooks: Send + Sync {
    fn intercepts_requests(&self) -> bool {
        false
    }
    fn pre_call(&self, request: OcrPreCallRequest) -> OcrHookFuture<'_, OcrPreCallRequest> {
        Box::pin(async move { Ok(request) })
    }
    fn during_call(
        &self,
        request: OcrDuringCallRequest,
    ) -> OcrHookFuture<'_, OcrDuringCallRequest> {
        Box::pin(async move { Ok(request) })
    }
    fn post_call(&self, request: OcrPostCallRequest) -> OcrHookFuture<'_, OcrPostCallRequest> {
        Box::pin(async move { Ok(request) })
    }
    fn success<'a>(
        &'a self,
        _context: &'a CallContext,
        _response: &'a LiteLLMOcrResponse,
        _timing: &'a CallTiming,
    ) -> OcrLogFuture<'a> {
        Box::pin(async {})
    }
    fn failure<'a>(
        &'a self,
        _context: &'a CallContext,
        _error: &'a Error,
        _timing: &'a CallTiming,
    ) -> OcrLogFuture<'a> {
        Box::pin(async {})
    }
}

pub struct NoopOcrHooks;
impl OcrHooks for NoopOcrHooks {}

pub(crate) async fn pre_call(
    hooks: &dyn OcrHooks,
    provider_name: &str,
    request: ResolvedOcrRequest,
) -> Result<ResolvedOcrRequest, Error> {
    if !hooks.intercepts_requests() {
        return Ok(request);
    }
    let changed = hooks
        .pre_call(OcrPreCallRequest {
            model: request.model.clone(),
            custom_llm_provider: provider_name.to_owned(),
            document: request.document,
            optional_params: Value::Object(request.optional_params.into()),
        })
        .await?;
    let Value::Object(optional_params) = changed.optional_params else {
        return Err(Error::RequestField {
            path: "guardrail.optional_params".into(),
        });
    };
    Ok(LiteLLMOcrRequest {
        document: changed.document,
        optional_params: optional_params.into(),
        ..request
    })
}
