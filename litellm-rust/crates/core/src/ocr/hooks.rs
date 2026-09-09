use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::prepare::PreparedOcrRequest;
use super::transformation::OcrProviderConfig;
use super::types::{OcrDocument, OcrResponseData};
use crate::Error;
use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleHooks, CallLifecycleTiming};
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
    pub body: Value,
}

pub trait OcrHooks: Send + Sync {
    fn has_guardrails(&self) -> bool {
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
    fn success<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        _response: &'a OcrResponseData,
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

pub(crate) struct OcrLifecycleHooks<C: OcrProviderConfig> {
    pub hooks: Arc<dyn OcrHooks>,
    pub provider_name: String,
    pub marker: std::marker::PhantomData<C>,
}

impl<C: OcrProviderConfig>
    CallLifecycleHooks<Result<PreparedOcrRequest<C>, Error>, PreparedOcrRequest<C>, OcrResponseData>
    for OcrLifecycleHooks<C>
{
    type PreCallFuture<'a> = OcrHookFuture<'a, Result<PreparedOcrRequest<C>, Error>>;
    type DuringCallFuture<'a> = OcrHookFuture<'a, PreparedOcrRequest<C>>;
    type SuccessFuture<'a> = OcrLogFuture<'a>;
    type FailureFuture<'a> = OcrLogFuture<'a>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: Result<PreparedOcrRequest<C>, Error>,
    ) -> Self::PreCallFuture<'a> {
        Box::pin(async move {
            let request = request?;
            if !self.hooks.has_guardrails() {
                return Ok(Ok(request));
            }
            let changed = self
                .hooks
                .pre_call(OcrPreCallRequest {
                    model: request.model.clone(),
                    custom_llm_provider: self.provider_name.clone(),
                    document: request.document,
                    optional_params: serde_json::to_value(request.params).map_err(|_| {
                        super::error::OcrRequestError::RequestField {
                            path: "guardrail.optional_params".into(),
                        }
                    })?,
                })
                .await?;
            let params = request
                .config
                .map_ocr_params(super::wire::decode_request_value(
                    changed.optional_params,
                    "guardrail.optional_params",
                )?)?;
            Ok(Ok(PreparedOcrRequest {
                document: changed.document,
                params,
                ..request
            }))
        })
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: Result<PreparedOcrRequest<C>, Error>,
    ) -> Self::DuringCallFuture<'a> {
        Box::pin(async move { request })
    }

    #[tracing::instrument(
        name = "success_callback",
        target = "litellm::function_trace",
        level = "trace",
        skip_all
    )]
    fn async_log_success_event<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        response: &'a OcrResponseData,
        timing: &'a CallLifecycleTiming,
    ) -> Self::SuccessFuture<'a> {
        self.hooks.success(context, response, timing)
    }

    #[tracing::instrument(
        name = "failure_callback",
        target = "litellm::function_trace",
        level = "trace",
        skip_all
    )]
    fn async_log_failure_event<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        error: &'a Error,
        timing: &'a CallLifecycleTiming,
    ) -> Self::FailureFuture<'a> {
        self.hooks.failure(context, error, timing)
    }
}
