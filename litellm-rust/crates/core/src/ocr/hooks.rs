use std::future::Future;
use std::pin::Pin;

use super::types::{LiteLLMOcrResponse, OcrDocument};
use crate::Error;
use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleTiming};
use serde::Serialize;
use serde_json::Value;

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
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
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
        _context: &'a CallLifecycleContext,
        _response: &'a LiteLLMOcrResponse,
        _timing: &'a CallLifecycleTiming,
    ) -> OcrLogFuture<'a> {
        Box::pin(async {})
    }
    fn failure<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        _error: &'a Error,
        _timing: &'a CallLifecycleTiming,
    ) -> OcrLogFuture<'a> {
        Box::pin(async {})
    }
}

pub struct NoopOcrHooks;
impl OcrHooks for NoopOcrHooks {}
