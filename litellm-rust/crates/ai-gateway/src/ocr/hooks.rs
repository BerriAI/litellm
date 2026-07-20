use std::future::Future;
use std::pin::Pin;

use litellm_core::call_lifecycle::{CallLifecycleContext, CallLifecycleHooks, CallLifecycleTiming};
use litellm_core::error::CoreError;
use litellm_core::ocr::transformation::{OcrAuth, OcrAuthStrategy, OcrDocumentPreparation};
use litellm_core::CoreResult;
use serde_json::{json, Map, Value};

use super::common_utils::{
    convert_document_url_to_data_uri, has_header, ocr_provider_config, string_headers,
    upload_reducto_document,
};
use super::types::{PreparedOcrRequest, ProviderOcrRequest};
use crate::config::resolve_env_reference;
use crate::integrations::custom_guardrail::{
    CustomGuardrailRunner, GuardrailContext, GuardrailError, GuardrailRequest,
};
use crate::integrations::custom_logger::{
    CallType, CallbackTiming, CallbackValue, CustomLoggerRunner, LoggingError, ModelCallDetails,
};
use crate::integrations::types::{
    RequestMetadata, StandardLoggingMetadata, StandardLoggingPayload,
};
use crate::io::vertex_ai::VertexAiBase;

pub(crate) struct OcrLifecycleHooks {
    logger_runner: CustomLoggerRunner,
    guardrail_runner: CustomGuardrailRunner,
    request_metadata: RequestMetadata,
}

type OcrFuture<'a, T> = Pin<Box<dyn Future<Output = CoreResult<T>> + Send + 'a>>;
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
    ) -> CoreResult<PreparedOcrRequest> {
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
        Ok(PreparedOcrRequest {
            document,
            optional_params,
            ..request
        })
    }

    async fn prepare_provider_request(
        &self,
        request: PreparedOcrRequest,
    ) -> CoreResult<ProviderOcrRequest> {
        let config = ocr_provider_config(&request.custom_llm_provider, &request.model)
            .ok_or_else(|| CoreError::InvalidProvider(request.custom_llm_provider.clone()))?;
        let env_lookup = |key: &str| std::env::var(key).ok();
        let headers = string_headers(request.extra_headers)?;
        let auth_strategy = config.auth_strategy();
        let api_key = resolve_env_reference(request.api_key.as_deref(), &env_lookup);
        let api_base = resolve_env_reference(request.api_base.as_deref(), &env_lookup);
        let api_key = if has_header(&headers, auth_strategy.header_name()) {
            None
        } else {
            match config.ocr_auth() {
                OcrAuth::ProviderKey => {
                    Some(config.resolve_api_key(api_key.as_deref(), &env_lookup)?)
                }
                OcrAuth::VertexOauth => {
                    let credentials = VertexAiBase::resolve_credential_source(
                        &request.optional_params,
                        &env_lookup,
                    );
                    Some(
                        VertexAiBase::shared()
                            .get_access_token(credentials.as_deref())
                            .await?,
                    )
                }
            }
        };
        let url = config.complete_url(
            api_base.as_deref(),
            &request.model,
            &request.optional_params,
            &env_lookup,
        )?;
        let filtered_params = config.map_ocr_params(&request.optional_params);
        let model = request.model.clone();
        let custom_llm_provider = request.custom_llm_provider.clone();
        let upstream_headers = upstream_headers(&headers, auth_strategy, api_key.as_deref());
        let document = match config.document_preparation() {
            OcrDocumentPreparation::None => request.document,
            OcrDocumentPreparation::DataUri => {
                convert_document_url_to_data_uri(request.document, request.timeout).await?
            }
            OcrDocumentPreparation::ReductoUpload => {
                upload_reducto_document(request.document, &url, &upstream_headers, request.timeout)
                    .await?
            }
        };
        let body = config
            .transform_ocr_request(&request.model, document, filtered_params)?
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
    ) -> CoreResult<Value> {
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

impl CallLifecycleHooks<PreparedOcrRequest, ProviderOcrRequest, Value> for OcrLifecycleHooks {
    type PreCallFuture<'a> = OcrFuture<'a, PreparedOcrRequest>;
    type DuringCallFuture<'a> = OcrFuture<'a, ProviderOcrRequest>;
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
        Box::pin(async move { self.prepare_provider_request(request).await })
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
        error: &'a CoreError,
        timing: &'a CallLifecycleTiming,
    ) -> Self::FailureFuture<'a> {
        Box::pin(async move {
            if self.logger_runner.is_empty() {
                return;
            }
            let logging_error = LoggingError {
                message: error.public_message(),
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

fn upstream_headers(
    headers: &[(String, String)],
    auth_strategy: OcrAuthStrategy,
    api_key: Option<&str>,
) -> Vec<(String, String)> {
    api_key
        .map(|api_key| match auth_strategy {
            OcrAuthStrategy::Bearer => ("Authorization".to_string(), format!("Bearer {api_key}")),
            OcrAuthStrategy::Header(header_name) => (header_name.to_string(), api_key.to_string()),
        })
        .into_iter()
        .chain(headers.iter().cloned())
        .collect()
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
) -> CoreResult<(Value, Map<String, Value>)> {
    let Value::Object(mut data) = request.data else {
        return Err(CoreError::InvalidRequest(
            "OCR pre_call guardrail must return an object".to_string(),
        ));
    };
    let document = data.remove("document").ok_or_else(|| {
        CoreError::InvalidRequest("OCR pre_call guardrail removed document".to_string())
    })?;
    let optional_params = match data.remove("optional_params") {
        Some(Value::Object(params)) => params,
        Some(_) => {
            return Err(CoreError::InvalidRequest(
                "OCR pre_call guardrail optional_params must be an object".to_string(),
            ))
        }
        None => Map::new(),
    };
    Ok((document, optional_params))
}

fn parse_ocr_during_call_guardrail_request(request: GuardrailRequest) -> CoreResult<Value> {
    let Value::Object(mut data) = request.data else {
        return Err(CoreError::InvalidRequest(
            "OCR during_call guardrail must return an object".to_string(),
        ));
    };
    data.remove("body").ok_or_else(|| {
        CoreError::InvalidRequest("OCR during_call guardrail removed body".to_string())
    })
}

fn guardrail_error_to_core_error(error: GuardrailError) -> CoreError {
    CoreError::InvalidRequest(format!("{}: {}", error.kind, error.message))
}

fn core_error_kind(error: &CoreError) -> &'static str {
    match error {
        CoreError::Auth(_) => "AuthError",
        CoreError::InvalidProvider(_) => "InvalidProvider",
        CoreError::InvalidRequest(_) => "InvalidRequest",
        CoreError::InvalidType { .. } => "InvalidType",
        CoreError::MissingField(_) => "MissingField",
        CoreError::Http { .. } => "HttpError",
        CoreError::InvalidResponse(_) => "InvalidResponse",
        CoreError::Timeout => "Timeout",
        CoreError::Network(_) => "NetworkError",
        CoreError::Routing(_) => "RoutingError",
    }
}
