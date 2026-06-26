//! End-to-end OCR orchestration.
//!
//! Owns supported OCR provider calls so the Python side stays a thin bridge:
//! resolve the API key, build the URL + body via the pure transforms, POST it,
//! and normalize the response. The HTTP client is built once and reused.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_core::call_lifecycle::{
    CallLifecycle, CallLifecycleContext, CallLifecycleHooks, CallLifecycleTiming,
};
use litellm_core::error::CoreError;
use litellm_core::ocr::transformation::{OcrAuthStrategy, OcrProviderConfig, OcrResponseHandling};
use litellm_core::routing_utils::provider::{get_custom_llm_provider, CustomLlmProvider};
use litellm_core::CoreResult;
use serde_json::{json, Map, Value};

use crate::integrations::custom_guardrail::{
    CustomGuardrail, CustomGuardrailRunner, GuardrailContext, GuardrailError, GuardrailRequest,
};
use crate::integrations::custom_logger::{
    CallType, CallbackTiming, CallbackValue, CustomLogger, CustomLoggerRunner, LoggingError,
    ModelCallDetails,
};
use crate::integrations::types::{
    RequestMetadata, StandardLoggingMetadata, StandardLoggingPayload,
};

mod common_utils;

use common_utils::{
    convert_document_url_to_data_uri, has_header, ocr_provider_config, poll_document_intelligence,
    string_headers, truncate_error_body,
};

/// OCR over large documents can take a while; bound it generously rather than
/// hanging forever on an unresponsive upstream. The client-level limit is the
/// outer ceiling; callers can tighten it per request via ``run_ocr``'s ``timeout``.
const OCR_TIMEOUT_SECS: u64 = 600;

/// Process-wide async HTTP client (connection pool + TLS reused across calls).
///
/// The Python fallback path uses LiteLLM's standard `BaseLLMHTTPHandler`. This
/// Rust path is opt-in and owns end-to-end OCR I/O, so it cannot call the
/// Python handler directly; keep this route-scoped until litellm-rust has a
/// shared HTTP abstraction.
fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(OCR_TIMEOUT_SECS))
            .build()
            .expect("failed to build reqwest client")
    })
}

fn upstream_headers(
    headers: &[(String, String)],
    auth_strategy: OcrAuthStrategy,
    api_key: Option<&str>,
) -> Vec<(String, String)> {
    let auth_header = api_key.map(|api_key| match auth_strategy {
        OcrAuthStrategy::Bearer => ("Authorization".to_string(), format!("Bearer {api_key}")),
        OcrAuthStrategy::Header(header_name) => (header_name.to_string(), api_key.to_string()),
    });
    auth_header
        .into_iter()
        .chain(headers.iter().cloned())
        .collect()
}

pub struct OcrRequest<'a> {
    pub model: &'a str,
    pub document: Value,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
    pub callbacks: Vec<Arc<dyn CustomLogger>>,
    pub guardrails: Vec<Arc<dyn CustomGuardrail>>,
    pub request_metadata: RequestMetadata,
    pub litellm_call_id: Option<&'a str>,
}

pub async fn ocr(request: OcrRequest<'_>) -> CoreResult<Value> {
    let call_id = request
        .litellm_call_id
        .map(str::to_string)
        .unwrap_or_else(new_ocr_call_id);
    let provider_info = get_custom_llm_provider(request.model, request.custom_llm_provider)
        .unwrap_or(CustomLlmProvider {
            model: request.model,
            custom_llm_provider: "mistral",
        });
    let model = provider_info.model.to_string();
    let custom_llm_provider = provider_info.custom_llm_provider.to_string();
    let hooks = OcrLifecycleHooks {
        logger_runner: CustomLoggerRunner::new(request.callbacks),
        guardrail_runner: CustomGuardrailRunner::new(request.guardrails),
        request_metadata: request.request_metadata,
    };
    let lifecycle_request = OcrLifecycleRequest {
        model: model.clone(),
        custom_llm_provider: custom_llm_provider.clone(),
        document: request.document,
        api_key: request.api_key.map(str::to_string),
        api_base: request.api_base.map(str::to_string),
        extra_headers: request.extra_headers,
        optional_params: request.optional_params,
        timeout: request.timeout,
    };

    CallLifecycle::default()
        .run(
            CallLifecycleContext::new("ocr", model, custom_llm_provider, call_id),
            lifecycle_request,
            &hooks,
            execute_ocr_provider_call,
        )
        .await
}

fn new_ocr_call_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("ocr-{timestamp}-{sequence}")
}

struct OcrLifecycleRequest {
    model: String,
    custom_llm_provider: String,
    document: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    extra_headers: Option<Map<String, Value>>,
    optional_params: Map<String, Value>,
    timeout: Option<Duration>,
}

struct OcrProviderRequest {
    model: String,
    config: &'static dyn OcrProviderConfig,
    url: String,
    body: Value,
    upstream_headers: Vec<(String, String)>,
    timeout: Option<Duration>,
}

struct OcrLifecycleHooks {
    logger_runner: CustomLoggerRunner,
    guardrail_runner: CustomGuardrailRunner,
    request_metadata: RequestMetadata,
}

type OcrFuture<'a, T> = Pin<Box<dyn Future<Output = CoreResult<T>> + Send + 'a>>;
type OcrLogFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

impl CallLifecycleHooks<OcrLifecycleRequest, OcrProviderRequest, Value> for OcrLifecycleHooks {
    type PreCallFuture<'a> = OcrFuture<'a, OcrLifecycleRequest>;
    type DuringCallFuture<'a> = OcrFuture<'a, OcrProviderRequest>;
    type SuccessFuture<'a> = OcrLogFuture<'a>;
    type FailureFuture<'a> = OcrLogFuture<'a>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: OcrLifecycleRequest,
    ) -> Self::PreCallFuture<'a> {
        Box::pin(async move { self.run_pre_call_guardrails(request).await })
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: OcrLifecycleRequest,
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
            let payload = self.standard_logging_payload(context, timing);
            let response_obj = CallbackValue::new("ocr", response.clone());
            self.logger_runner
                .async_log_success_event(
                    &ModelCallDetails::from_standard_logging_payload(payload),
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

impl OcrLifecycleHooks {
    async fn run_pre_call_guardrails(
        &self,
        request: OcrLifecycleRequest,
    ) -> CoreResult<OcrLifecycleRequest> {
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
        Ok(OcrLifecycleRequest {
            document,
            optional_params,
            ..request
        })
    }

    async fn prepare_provider_request(
        &self,
        request: OcrLifecycleRequest,
    ) -> CoreResult<OcrProviderRequest> {
        let config = ocr_provider_config(&request.custom_llm_provider, &request.model)
            .ok_or_else(|| CoreError::InvalidProvider(request.custom_llm_provider.clone()))?;
        let env_lookup = |key: &str| std::env::var(key).ok();
        let headers = string_headers(request.extra_headers)?;
        let auth_strategy = config.auth_strategy();
        let api_key = (!has_header(&headers, auth_strategy.header_name()))
            .then(|| config.resolve_api_key(request.api_key.as_deref(), &env_lookup))
            .transpose()?;
        let url = config.complete_url(
            request.api_base.as_deref(),
            &request.model,
            &request.optional_params,
            &env_lookup,
        )?;
        let filtered_params = config.map_ocr_params(&request.optional_params);
        let model = request.model.clone();
        let custom_llm_provider = request.custom_llm_provider.clone();
        let document = if config.requires_data_uri_document() {
            convert_document_url_to_data_uri(request.document).await?
        } else {
            request.document
        };
        let body = config
            .transform_ocr_request(&request.model, document, filtered_params)?
            .data;
        let upstream_headers = upstream_headers(&headers, auth_strategy, api_key.as_deref());
        let body = self
            .run_during_call_guardrails(&model, &custom_llm_provider, &url, body)
            .await?;
        Ok(OcrProviderRequest {
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
        CoreError::Network(_) => "NetworkError",
        CoreError::Routing(_) => "RoutingError",
    }
}

async fn execute_ocr_provider_call(request: OcrProviderRequest) -> CoreResult<Value> {
    let mut request_builder = http_client().post(&request.url).json(&request.body);
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder
        .send()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;

    let status = response.status();
    if request.config.response_handling() == OcrResponseHandling::AzureDocumentIntelligencePoll
        && status.as_u16() == 202
    {
        let operation_url = response
            .headers()
            .get("operation-location")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
            .ok_or_else(|| {
                CoreError::InvalidResponse(
                    "Azure Document Intelligence returned 202 but no Operation-Location header found"
                        .to_string(),
                )
            })?;
        let response_json = poll_document_intelligence(
            &operation_url,
            &request.url,
            &request.upstream_headers,
            request.timeout,
        )
        .await?;
        return Ok(request
            .config
            .transform_ocr_response(&request.model, response_json)?
            .into_json());
    }

    let text = response
        .text()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;

    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }

    let response_json: Value = serde_json::from_str(&text)
        .map_err(|err| CoreError::InvalidResponse(format!("invalid OCR response JSON: {err}")))?;

    Ok(request
        .config
        .transform_ocr_response(&request.model, response_json)?
        .into_json())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrations::custom_guardrail::{
        GuardrailDecision, GuardrailEventHook, GuardrailFuture,
    };
    use crate::integrations::custom_logger::LogFuture;
    use serde_json::json;
    use std::sync::Mutex;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    async fn read_http_headers(socket: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        loop {
            let n = socket.read(&mut buffer).await.expect("reads request");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..n]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        String::from_utf8(request).expect("request is utf8")
    }

    async fn read_http_request(socket: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        let header_end = loop {
            let n = socket.read(&mut buffer).await.expect("reads request");
            if n == 0 {
                break request.len();
            }
            request.extend_from_slice(&buffer[..n]);
            if let Some(position) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break position + 4;
            }
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        while request.len().saturating_sub(header_end) < content_length {
            let n = socket.read(&mut buffer).await.expect("reads body");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..n]);
        }
        String::from_utf8(request).expect("request is utf8")
    }

    #[derive(Clone, Debug, PartialEq)]
    struct RecordedLogEvent {
        hook: &'static str,
        model: String,
        call_type: String,
        user_id: Option<String>,
        response_object: Option<String>,
        error_kind: Option<String>,
    }

    #[derive(Default)]
    struct RecordingOcrLogger {
        events: Mutex<Vec<RecordedLogEvent>>,
    }

    impl RecordingOcrLogger {
        fn events(&self) -> Vec<RecordedLogEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl CustomLogger for RecordingOcrLogger {
        fn async_log_success_event<'a>(
            &'a self,
            model_call_details: &'a ModelCallDetails,
            response_obj: &'a CallbackValue,
            _timing: CallbackTiming,
        ) -> LogFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push(RecordedLogEvent {
                    hook: "async_log_success_event",
                    model: model_call_details.model.clone(),
                    call_type: model_call_details.call_type.to_string(),
                    user_id: model_call_details.metadata.user_api_key_user_id.clone(),
                    response_object: Some(response_obj.object.clone()),
                    error_kind: None,
                });
                Ok(())
            })
        }

        fn async_log_failure_event<'a>(
            &'a self,
            model_call_details: &'a ModelCallDetails,
            response_obj: Option<&'a CallbackValue>,
            _timing: CallbackTiming,
        ) -> LogFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push(RecordedLogEvent {
                    hook: "async_log_failure_event",
                    model: model_call_details.model.clone(),
                    call_type: model_call_details.call_type.to_string(),
                    user_id: model_call_details.metadata.user_api_key_user_id.clone(),
                    response_object: response_obj.map(|value| value.object.clone()),
                    error_kind: model_call_details
                        .failure_error
                        .as_ref()
                        .map(|error| error.kind.clone()),
                });
                Ok(())
            })
        }
    }

    struct RecordingOcrGuardrail {
        hooks: Vec<GuardrailEventHook>,
        events: Mutex<Vec<&'static str>>,
        block_pre_call: bool,
    }

    impl RecordingOcrGuardrail {
        fn new(hooks: Vec<GuardrailEventHook>) -> Self {
            Self {
                hooks,
                events: Mutex::new(Vec::new()),
                block_pre_call: false,
            }
        }

        fn blocking_pre_call() -> Self {
            Self {
                hooks: vec![GuardrailEventHook::PreCall],
                events: Mutex::new(Vec::new()),
                block_pre_call: true,
            }
        }

        fn events(&self) -> Vec<&'static str> {
            self.events.lock().unwrap().clone()
        }
    }

    impl CustomGuardrail for RecordingOcrGuardrail {
        fn guardrail_name(&self) -> &str {
            "recording-ocr-guardrail"
        }

        fn supported_event_hooks(&self) -> &[GuardrailEventHook] {
            &self.hooks
        }

        fn async_pre_call_hook<'a>(
            &'a self,
            _context: &'a GuardrailContext,
            mut request: GuardrailRequest,
        ) -> GuardrailFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("async_pre_call_hook");
                if self.block_pre_call {
                    return Ok(GuardrailDecision::Block(GuardrailError::blocked(
                        "blocked before provider",
                    )));
                }
                request.data["document"]["guarded_pre"] = json!(true);
                Ok(GuardrailDecision::Mask(request))
            })
        }

        fn async_moderation_hook<'a>(
            &'a self,
            _context: &'a GuardrailContext,
            mut request: GuardrailRequest,
        ) -> GuardrailFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("async_moderation_hook");
                request.data["body"]["guarded_during"] = json!(true);
                Ok(GuardrailDecision::Mask(request))
            })
        }
    }

    #[test]
    fn truncate_error_body_passes_short_strings_through() {
        let body = "Unauthorized";
        assert_eq!(truncate_error_body(body), "Unauthorized");
    }

    #[test]
    fn truncate_error_body_caps_long_payloads() {
        let body = "x".repeat(306);
        let truncated = truncate_error_body(&body);

        assert!(truncated.ends_with("... (truncated)"));
        let prefix_chars = truncated
            .strip_suffix("... (truncated)")
            .expect("truncated marker present")
            .chars()
            .count();
        assert_eq!(prefix_chars, 256);
    }

    #[test]
    fn truncate_error_body_does_not_split_multibyte_chars() {
        let body = "é".repeat(266);
        let truncated = truncate_error_body(&body);
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    #[test]
    fn ocr_dispatch_supports_migrated_providers() {
        assert!(ocr_provider_config("mistral", "mistral-ocr-latest").is_some());
        assert!(ocr_provider_config("azure_ai", "pixtral-12b-2409")
            .expect("azure ai config resolves")
            .requires_data_uri_document());
        assert_eq!(
            ocr_provider_config("azure_ai", "doc-intelligence/prebuilt-read")
                .expect("document intelligence config resolves")
                .response_handling(),
            OcrResponseHandling::AzureDocumentIntelligencePoll
        );
        assert!(ocr_provider_config("vertex_ai", "deepseek-ocr-maas")
            .expect("vertex deepseek config resolves")
            .supported_ocr_params()
            .contains(&"temperature"));
        assert!(ocr_provider_config("openai", "gpt-4o").is_none());
    }

    #[test]
    fn string_headers_accepts_string_values() {
        let headers = json!({
            "x-trace-id": "trace-1"
        })
        .as_object()
        .unwrap()
        .clone();

        assert_eq!(
            string_headers(Some(headers)).expect("string headers accepted"),
            vec![("x-trace-id".to_string(), "trace-1".to_string())]
        );
    }

    #[test]
    fn auth_header_detection_is_case_insensitive() {
        let headers = vec![
            ("x-trace-id".to_string(), "trace-1".to_string()),
            ("authorization".to_string(), "Bearer sk-test".to_string()),
        ];

        assert!(has_header(&headers, "authorization"));

        let headers = vec![("Authorization".to_string(), "Bearer sk-test".to_string())];
        assert!(has_header(&headers, "authorization"));

        let headers = vec![("x-trace-id".to_string(), "trace-1".to_string())];
        assert!(!has_header(&headers, "authorization"));
    }

    #[tokio::test]
    async fn ocr_lifecycle_runs_pre_during_and_success_hooks() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let addr = listener.local_addr().expect("listener has local addr");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts one request");
            let request = read_http_request(&mut socket).await;
            let response_body = r#"{"pages":[{"index":0,"markdown":"ok"}],"model":"mistral-ocr-latest","usage_info":{"pages_processed":1}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response");
            request
        });

        let logger = Arc::new(RecordingOcrLogger::default());
        let guardrail = Arc::new(RecordingOcrGuardrail::new(vec![
            GuardrailEventHook::PreCall,
            GuardrailEventHook::DuringCall,
        ]));
        let response = ocr(OcrRequest {
            model: "mistral-ocr-latest",
            document: json!({
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf"
            }),
            api_key: Some("sk-test"),
            api_base: Some(&format!("http://{addr}")),
            custom_llm_provider: Some("mistral"),
            extra_headers: None,
            optional_params: Map::new(),
            timeout: Some(Duration::from_secs(5)),
            callbacks: vec![logger.clone()],
            guardrails: vec![guardrail.clone()],
            request_metadata: RequestMetadata {
                user_api_key_user_id: Some("user-1".to_string()),
                ..Default::default()
            },
            litellm_call_id: Some("ocr-call-1"),
        })
        .await
        .expect("ocr request succeeds");

        assert_eq!(response["pages"][0]["markdown"], "ok");
        assert_eq!(
            guardrail.events(),
            vec!["async_pre_call_hook", "async_moderation_hook"]
        );
        assert_eq!(
            logger.events(),
            vec![RecordedLogEvent {
                hook: "async_log_success_event",
                model: "mistral-ocr-latest".to_string(),
                call_type: "ocr".to_string(),
                user_id: Some("user-1".to_string()),
                response_object: Some("ocr".to_string()),
                error_kind: None,
            }]
        );

        let request = server.await.expect("server task completes");
        assert!(request.contains(r#""guarded_pre":true"#), "{request}");
        assert!(request.contains(r#""guarded_during":true"#), "{request}");
    }

    #[tokio::test]
    async fn ocr_lifecycle_runs_failure_hook_on_provider_error() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let addr = listener.local_addr().expect("listener has local addr");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts one request");
            let _request = read_http_request(&mut socket).await;
            let response_body = "provider failed";
            let response = format!(
                "HTTP/1.1 500 Internal Server Error\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response");
        });

        let logger = Arc::new(RecordingOcrLogger::default());
        let err = ocr(OcrRequest {
            model: "mistral-ocr-latest",
            document: json!({
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf"
            }),
            api_key: Some("sk-test"),
            api_base: Some(&format!("http://{addr}")),
            custom_llm_provider: Some("mistral"),
            extra_headers: None,
            optional_params: Map::new(),
            timeout: Some(Duration::from_secs(5)),
            callbacks: vec![logger.clone()],
            guardrails: Vec::new(),
            request_metadata: RequestMetadata::default(),
            litellm_call_id: Some("ocr-call-2"),
        })
        .await
        .expect_err("provider error propagates");

        assert!(matches!(err, CoreError::Http { status: 500, .. }));
        server.await.expect("server task completes");
        assert_eq!(
            logger.events(),
            vec![RecordedLogEvent {
                hook: "async_log_failure_event",
                model: "mistral-ocr-latest".to_string(),
                call_type: "ocr".to_string(),
                user_id: None,
                response_object: Some("error".to_string()),
                error_kind: Some("HttpError".to_string()),
            }]
        );
    }

    #[tokio::test]
    async fn ocr_lifecycle_pre_call_block_skips_provider_socket() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let addr = listener.local_addr().expect("listener has local addr");
        let logger = Arc::new(RecordingOcrLogger::default());
        let guardrail = Arc::new(RecordingOcrGuardrail::blocking_pre_call());

        let err = ocr(OcrRequest {
            model: "mistral-ocr-latest",
            document: json!({
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf"
            }),
            api_key: Some("sk-test"),
            api_base: Some(&format!("http://{addr}")),
            custom_llm_provider: Some("mistral"),
            extra_headers: None,
            optional_params: Map::new(),
            timeout: Some(Duration::from_millis(100)),
            callbacks: vec![logger.clone()],
            guardrails: vec![guardrail.clone()],
            request_metadata: RequestMetadata::default(),
            litellm_call_id: Some("ocr-call-3"),
        })
        .await
        .expect_err("guardrail blocks request");

        assert!(matches!(err, CoreError::InvalidRequest(_)));
        assert_eq!(guardrail.events(), vec!["async_pre_call_hook"]);
        assert_eq!(
            logger.events(),
            vec![RecordedLogEvent {
                hook: "async_log_failure_event",
                model: "mistral-ocr-latest".to_string(),
                call_type: "ocr".to_string(),
                user_id: None,
                response_object: Some("error".to_string()),
                error_kind: Some("InvalidRequest".to_string()),
            }]
        );
        let accepted = tokio::time::timeout(Duration::from_millis(100), listener.accept()).await;
        assert!(accepted.is_err(), "provider socket should not be touched");
    }

    #[tokio::test]
    async fn ocr_does_not_duplicate_authorization_header_when_header_is_supplied() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let addr = listener.local_addr().expect("listener has local addr");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts one request");
            let request = read_http_headers(&mut socket).await;

            let response_body = r#"{"pages":[{"index":0,"markdown":"ok"}],"model":"mistral-ocr-latest","usage_info":{"pages_processed":1}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response");
            request
        });

        let mut headers = Map::new();
        headers.insert(
            "Authorization".to_string(),
            Value::String("Bearer sk-from-python".to_string()),
        );
        headers.insert(
            "x-trace-id".to_string(),
            Value::String("trace-1".to_string()),
        );

        let response = ocr(OcrRequest {
            model: "mistral-ocr-latest",
            document: json!({
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf"
            }),
            api_key: Some("sk-for-rust-fallback"),
            api_base: Some(&format!("http://{addr}")),
            custom_llm_provider: Some("mistral"),
            extra_headers: Some(headers),
            optional_params: Map::new(),
            timeout: Some(Duration::from_secs(5)),
            callbacks: Vec::new(),
            guardrails: Vec::new(),
            request_metadata: RequestMetadata::default(),
            litellm_call_id: None,
        })
        .await
        .expect("ocr request succeeds");

        assert_eq!(response["pages"][0]["markdown"], "ok");

        let request = server.await.expect("server task completes");
        let authorization_count = request
            .lines()
            .filter(|line| line.to_ascii_lowercase().starts_with("authorization:"))
            .count();
        assert_eq!(authorization_count, 1, "{request}");
        assert!(
            request.contains("authorization: Bearer sk-from-python")
                || request.contains("Authorization: Bearer sk-from-python"),
            "{request}"
        );
    }

    #[tokio::test]
    async fn document_intelligence_poll_uses_resolved_subscription_key() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let addr = listener.local_addr().expect("listener has local addr");
        let operation_url = format!("http://{addr}/operations/1");

        let server = tokio::spawn(async move {
            let (mut post_socket, _) = listener.accept().await.expect("accepts post request");
            let post_request = read_http_headers(&mut post_socket).await;
            let post_response = format!(
                "HTTP/1.1 202 Accepted\r\noperation-location: {operation_url}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            );
            post_socket
                .write_all(post_response.as_bytes())
                .await
                .expect("writes post response");

            let (mut poll_socket, _) = listener.accept().await.expect("accepts poll request");
            let poll_request = read_http_headers(&mut poll_socket).await;
            let response_body = r#"{"status":"succeeded","analyzeResult":{"pages":[{"pageNumber":1,"lines":[{"content":"ok"}]}]}}"#;
            let poll_response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            poll_socket
                .write_all(poll_response.as_bytes())
                .await
                .expect("writes poll response");
            (post_request, poll_request)
        });

        let response = ocr(OcrRequest {
            model: "doc-intelligence/prebuilt-read",
            document: json!({
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf"
            }),
            api_key: Some("di-key"),
            api_base: Some(&format!("http://{addr}")),
            custom_llm_provider: Some("azure_ai"),
            extra_headers: None,
            optional_params: Map::new(),
            timeout: Some(Duration::from_secs(5)),
            callbacks: Vec::new(),
            guardrails: Vec::new(),
            request_metadata: RequestMetadata::default(),
            litellm_call_id: None,
        })
        .await
        .expect("document intelligence request succeeds");

        assert_eq!(response["pages"][0]["markdown"], "ok");

        let (post_request, poll_request) = server.await.expect("server task completes");
        assert!(
            post_request
                .to_ascii_lowercase()
                .contains("ocp-apim-subscription-key: di-key"),
            "{post_request}"
        );
        assert!(
            poll_request
                .to_ascii_lowercase()
                .contains("ocp-apim-subscription-key: di-key"),
            "{poll_request}"
        );
    }

    #[test]
    fn string_headers_rejects_non_string_values() {
        let headers = json!({
            "x-retry-count": 3
        })
        .as_object()
        .unwrap()
        .clone();

        let err = string_headers(Some(headers)).expect_err("non-string header rejected");
        assert_eq!(
            err,
            CoreError::InvalidRequest(
                "OCR extra_headers.x-retry-count must be a string, got number".to_string()
            )
        );
    }
}
