use serde_json::{Map, Value, json};

use crate::CoreResult;
use crate::call_lifecycle::{CallContext, CallInterceptor, CallOutcome, CallSpec, CallTiming};
use crate::callbacks::CallbackOptions;
use crate::callbacks::custom_guardrail::{
    CustomGuardrailRunner, GuardrailContext, GuardrailRequest,
};
use crate::callbacks::custom_logger::{
    CallType, CallbackTiming, CallbackValue, CustomLoggerRunner, LoggingError, ModelCallDetails,
};
use crate::callbacks::types::{RequestMetadata, StandardLoggingMetadata, StandardLoggingPayload};
use crate::error::CoreError;

use super::types::{OcrCall, OcrDocument, PreparedOcrRequest, ProviderOcrRequest};

pub struct OcrCallbackInterceptor {
    logger_runner: CustomLoggerRunner,
    guardrail_runner: CustomGuardrailRunner,
    request_metadata: RequestMetadata,
}

impl OcrCallbackInterceptor {
    pub fn new(options: CallbackOptions) -> Self {
        Self {
            logger_runner: CustomLoggerRunner::new(options.callbacks),
            guardrail_runner: CustomGuardrailRunner::new(options.guardrails),
            request_metadata: options.request_metadata,
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
            .await?;
        let (document, optional_params) = parse_ocr_pre_call_guardrail_request(guardrail_request)?;
        Ok(PreparedOcrRequest {
            document,
            optional_params,
            ..request
        })
    }

    async fn run_before_send_guardrails(
        &self,
        request: ProviderOcrRequest,
    ) -> CoreResult<ProviderOcrRequest> {
        if self.guardrail_runner.is_empty() {
            return Ok(request);
        }

        let context = guardrail_context(&self.request_metadata);
        let guardrail_request = GuardrailRequest::new(json!({
            "model": request.model,
            "custom_llm_provider": request.custom_llm_provider,
            "url": request.url,
            "body": request.body,
        }));
        let (guardrail_request, _) = self
            .guardrail_runner
            .run_during_call(&context, guardrail_request)
            .await?;
        let body = parse_ocr_before_send_guardrail_request(guardrail_request)?;
        Ok(ProviderOcrRequest { body, ..request })
    }

    fn standard_logging_payload(
        &self,
        context: &CallContext,
        timing: &CallTiming,
    ) -> StandardLoggingPayload {
        StandardLoggingPayload {
            id: context.litellm_call_id.clone(),
            litellm_call_id: context.litellm_call_id.clone(),
            call_type: OcrCall::NAME.to_string(),
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

impl CallInterceptor<OcrCall> for OcrCallbackInterceptor {
    async fn before_call<'a>(
        &'a self,
        _context: &'a CallContext,
        input: PreparedOcrRequest,
    ) -> CoreResult<PreparedOcrRequest> {
        self.run_pre_call_guardrails(input).await
    }

    async fn before_send<'a>(
        &'a self,
        _context: &'a CallContext,
        input: ProviderOcrRequest,
    ) -> CoreResult<ProviderOcrRequest> {
        self.run_before_send_guardrails(input).await
    }

    async fn complete<'a>(
        &'a self,
        context: &'a CallContext,
        outcome: CallOutcome<'a, Value>,
        timing: &'a CallTiming,
    ) {
        if self.logger_runner.is_empty() {
            return;
        }

        match outcome {
            CallOutcome::Success(response) => {
                self.logger_runner
                    .async_log_success_event(
                        &ModelCallDetails::from_standard_logging_payload(
                            self.standard_logging_payload(context, timing),
                        ),
                        &CallbackValue::new("ocr", response.clone()),
                        CallbackTiming::new(timing.start_time, timing.end_time),
                    )
                    .await;
            }
            CallOutcome::Failure(error) => {
                let logging_error = LoggingError {
                    message: error.to_string(),
                    kind: error.kind().to_string(),
                };
                self.logger_runner
                    .async_log_failure_event(
                        &ModelCallDetails::from_standard_logging_payload(
                            self.standard_logging_payload(context, timing),
                        )
                        .with_failure_error(logging_error.clone()),
                        Some(&CallbackValue::new(
                            "error",
                            json!({
                                "message": logging_error.message,
                                "kind": logging_error.kind,
                            }),
                        )),
                        CallbackTiming::new(timing.start_time, timing.end_time),
                    )
                    .await;
            }
        }
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
) -> CoreResult<(OcrDocument, Map<String, Value>)> {
    let Value::Object(mut data) = request.data else {
        return Err(CoreError::invalid_request(
            "OCR pre_call guardrail must return an object".to_string(),
        ));
    };
    let document = data.remove("document").ok_or_else(|| {
        CoreError::invalid_request("OCR pre_call guardrail removed document".to_string())
    })?;
    let optional_params = match data.remove("optional_params") {
        Some(Value::Object(params)) => params,
        Some(_) => {
            return Err(CoreError::invalid_request(
                "OCR pre_call guardrail optional_params must be an object".to_string(),
            ));
        }
        None => Map::new(),
    };
    let document = serde_json::from_value::<OcrDocument>(document)
        .map_err(|error| CoreError::invalid_request(format!("invalid OCR document: {error}")))?;
    Ok((document, optional_params))
}

fn parse_ocr_before_send_guardrail_request(request: GuardrailRequest) -> CoreResult<Value> {
    let Value::Object(mut data) = request.data else {
        return Err(CoreError::invalid_request(
            "OCR before_send guardrail must return an object".to_string(),
        ));
    };
    data.remove("body").ok_or_else(|| {
        CoreError::invalid_request("OCR before_send guardrail removed body".to_string())
    })
}
