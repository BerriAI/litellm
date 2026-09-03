use litellm_core::call_lifecycle::{CallLifecycleContext, CallLifecycleHooks, CallLifecycleTiming};
use litellm_core::error::Error;
use serde_json::{Map, Value, json};
use std::future::Future;
use std::pin::Pin;

use super::common_utils::{convert_document_url_to_data_uri, string_headers};
use super::types::{PreparedOcrRequest, ProviderOcrRequest};
use crate::integrations::custom_guardrail::{
    CustomGuardrailRunner, GuardrailContext, GuardrailError, GuardrailRequest,
};
use crate::integrations::custom_logger::{
    CallType, CallbackTiming, CallbackValue, CustomLoggerRunner, LoggingError, ModelCallDetails,
};
use crate::integrations::types::{
    RequestMetadata, StandardLoggingMetadata, StandardLoggingPayload,
};

pub(crate) struct OcrLifecycleHooks {
    logger_runner: CustomLoggerRunner,
    guardrail_runner: CustomGuardrailRunner,
    request_metadata: RequestMetadata,
}

type OcrFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;
type OcrLogFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

impl OcrLifecycleHooks {
    pub(crate) fn new(
        logger_runner: CustomLoggerRunner,
        guardrail_runner: CustomGuardrailRunner,
        request_metadata: RequestMetadata,
    ) -> Self {
        Self {
            logger_runner,
            guardrail_runner,
            request_metadata,
        }
    }

    async fn run_pre_call_guardrails(
        &self,
        request: PreparedOcrRequest,
    ) -> Result<PreparedOcrRequest, Error> {
        if self.guardrail_runner.is_empty() {
            return Ok(request);
        }

        let context = guardrail_context(&self.request_metadata);
        let guardrail_request = GuardrailRequest::new(json!({
            "model": request.model,
            "custom_llm_provider": request.custom_llm_provider,
            "document": request.document,
            "optional_params": request.optional_params,
        }));
        let (guardrail_request, _) = self
            .guardrail_runner
            .run_pre_call(&context, guardrail_request)
            .await
            .map_err(guardrail_error_to_core_error)?;
        let (document, optional_params) = parse_ocr_pre_call_guardrail_request(guardrail_request)?;
        let optional_params = match &request.config {
            Ok(config) => config.map_ocr_params(&optional_params),
            Err(_) => optional_params,
        };
        Ok(PreparedOcrRequest {
            document,
            optional_params,
            ..request
        })
    }

    pub(crate) async fn prepare_provider_request(
        &self,
        request: PreparedOcrRequest,
    ) -> Result<ProviderOcrRequest, Error> {
        let config = request.config?;
        let env_lookup = |key: &str| std::env::var(key).ok();
        let upstream_headers = config.validate_environment(
            string_headers(request.extra_headers)?,
            request.api_key.as_deref(),
            &env_lookup,
        )?;
        let url = config.complete_url(
            request.api_base.as_deref(),
            &request.model,
            &request.optional_params,
            &env_lookup,
        )?;
        let model = request.model.clone();
        let custom_llm_provider = request.custom_llm_provider.clone();
        let document = if config.requires_data_uri_document() {
            convert_document_url_to_data_uri(request.document).await?
        } else {
            request.document
        };
        let body = config
            .transform_ocr_request(&request.model, document, request.optional_params)?
            .data;
        let body = self
            .run_during_call_guardrails(&model, &custom_llm_provider, &url, body)
            .await?;
        Ok(ProviderOcrRequest {
            model,
            config,
            url,
            body,
            upstream_headers,
            timeout: request.timeout,
        })
    }

    async fn run_during_call_guardrails(
        &self,
        model: &str,
        custom_llm_provider: &str,
        url: &str,
        body: Value,
    ) -> Result<Value, Error> {
        if self.guardrail_runner.is_empty() {
            return Ok(body);
        }

        let context = guardrail_context(&self.request_metadata);
        let guardrail_request = GuardrailRequest::new(json!({
            "model": model,
            "custom_llm_provider": custom_llm_provider,
            "url": url,
            "body": body,
        }));
        let (guardrail_request, _) = self
            .guardrail_runner
            .run_during_call(&context, guardrail_request)
            .await
            .map_err(guardrail_error_to_core_error)?;
        parse_ocr_during_call_guardrail_request(guardrail_request)
    }

    fn standard_logging_payload(
        &self,
        context: &CallLifecycleContext,
        timing: &CallLifecycleTiming,
    ) -> StandardLoggingPayload {
        StandardLoggingPayload {
            id: context.litellm_call_id.clone(),
            litellm_call_id: context.litellm_call_id.clone(),
            call_type: context.call_type.clone(),
            model: context.model.clone(),
            custom_llm_provider: context.custom_llm_provider.clone(),
            response_cost: 0.0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            start_time: timing.start_time,
            end_time: timing.end_time,
            stream: false,
            metadata: StandardLoggingMetadata {
                user_api_key_hash: self.request_metadata.user_api_key_hash.clone(),
                user_api_key_user_id: self.request_metadata.user_api_key_user_id.clone(),
                user_api_key_team_id: self.request_metadata.user_api_key_team_id.clone(),
                ..Default::default()
            },
            messages: None,
        }
    }
}

impl CallLifecycleHooks<PreparedOcrRequest, PreparedOcrRequest, Value> for OcrLifecycleHooks {
    type PreCallFuture<'a> = OcrFuture<'a, PreparedOcrRequest>;
    type DuringCallFuture<'a> = OcrFuture<'a, PreparedOcrRequest>;
    type SuccessFuture<'a> = OcrLogFuture<'a>;
    type FailureFuture<'a> = OcrLogFuture<'a>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: PreparedOcrRequest,
    ) -> Self::PreCallFuture<'a> {
        Box::pin(async move { self.run_pre_call_guardrails(request).await })
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: PreparedOcrRequest,
    ) -> Self::DuringCallFuture<'a> {
        Box::pin(async move { Ok(request) })
    }

    fn async_log_success_event<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        response: &'a Value,
        timing: &'a CallLifecycleTiming,
    ) -> Self::SuccessFuture<'a> {
        Box::pin(async move {
            if self.logger_runner.is_empty() {
                return;
            }
            let response_obj = CallbackValue::new("ocr", response.clone());
            self.logger_runner
                .async_log_success_event(
                    &ModelCallDetails::from_standard_logging_payload(
                        self.standard_logging_payload(context, timing),
                    ),
                    &response_obj,
                    CallbackTiming::new(timing.start_time, timing.end_time),
                )
                .await;
        })
    }

    fn async_log_failure_event<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        error: &'a Error,
        timing: &'a CallLifecycleTiming,
    ) -> Self::FailureFuture<'a> {
        Box::pin(async move {
            if self.logger_runner.is_empty() {
                return;
            }
            let logging_error = LoggingError {
                message: error.to_string(),
                kind: core_error_kind(error).to_string(),
            };
            let response_obj = CallbackValue::new(
                "error",
                json!({
                    "message": logging_error.message,
                    "kind": logging_error.kind,
                }),
            );
            self.logger_runner
                .async_log_failure_event(
                    &ModelCallDetails::from_standard_logging_payload(
                        self.standard_logging_payload(context, timing),
                    )
                    .with_failure_error(logging_error),
                    Some(&response_obj),
                    CallbackTiming::new(timing.start_time, timing.end_time),
                )
                .await;
        })
    }
}

fn guardrail_context(metadata: &RequestMetadata) -> GuardrailContext {
    GuardrailContext {
        call_type: CallType::Ocr,
        selected_guardrails: Vec::new(),
        metadata: std::collections::HashMap::new(),
        user_api_key_hash: metadata.user_api_key_hash.clone(),
        user_api_key_user_id: metadata.user_api_key_user_id.clone(),
        user_api_key_team_id: metadata.user_api_key_team_id.clone(),
        trace_parent: None,
    }
}

fn parse_ocr_pre_call_guardrail_request(
    request: GuardrailRequest,
) -> Result<(Value, Map<String, Value>), Error> {
    let Value::Object(mut data) = request.data else {
        return Err(Error::InvalidRequest(
            "OCR pre_call guardrail must return an object".to_string(),
        ));
    };
    let document = data.remove("document").ok_or_else(|| {
        Error::InvalidRequest("OCR pre_call guardrail removed document".to_string())
    })?;
    let optional_params = match data.remove("optional_params") {
        Some(Value::Object(params)) => params,
        Some(_) => {
            return Err(Error::InvalidRequest(
                "OCR pre_call guardrail optional_params must be an object".to_string(),
            ));
        }
        None => Map::new(),
    };
    Ok((document, optional_params))
}

fn parse_ocr_during_call_guardrail_request(request: GuardrailRequest) -> Result<Value, Error> {
    let Value::Object(mut data) = request.data else {
        return Err(Error::InvalidRequest(
            "OCR during_call guardrail must return an object".to_string(),
        ));
    };
    data.remove("body")
        .ok_or_else(|| Error::InvalidRequest("OCR during_call guardrail removed body".to_string()))
}

fn guardrail_error_to_core_error(error: GuardrailError) -> Error {
    Error::InvalidRequest(format!("{}: {}", error.kind, error.message))
}

fn core_error_kind(error: &Error) -> &'static str {
    match error {
        Error::Auth(_) => "AuthError",
        Error::InvalidProvider(_) => "InvalidProvider",
        Error::InvalidRequest(_) => "InvalidRequest",
        Error::InvalidType { .. } => "InvalidType",
        Error::MissingField(_) => "MissingField",
        Error::Http { .. } => "HttpError",
        Error::InvalidResponse(_) => "InvalidResponse",
        Error::Network(_) => "NetworkError",
        Error::Connect(_) => "ConnectError",
        Error::Routing(_) => "RoutingError",
        Error::Unsupported(_) => "UnsupportedRequest",
    }
}
